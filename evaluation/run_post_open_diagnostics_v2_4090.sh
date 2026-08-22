#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?absolute checkpoint .pt path required}"
: "${OUTPUT_DIR:?new absolute output directory required}"

INSTRUMENTATION_DIR="${INSTRUMENTATION_DIR:-/sda/mashijian/doorman_migrations/step5500_postopen_fix_v1_20260822/evals/post_open_diagnostics_v2_instrumentation}"
CODE_DIR="${CODE_DIR:-/sda/mashijian/doorman_arm_reward_v2_20260822/code}"
CONDA_SH="${CONDA_SH:-/sdb/mashijian/coordex/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/sdb/mashijian/coordex/envs/doorman}"
RUNTIME_ENV="${RUNTIME_ENV:-/sdb/mashijian/coordex/doorman/GR00T-VisualSim2Real/scripts/doorman_runtime_env.sh}"
LAFAN_DIR="${LAFAN_DIR:-/sdb/mashijian/coordex/doorman/LAFAN-G1}"
GPU="${GPU:-6}"
NUM_ENVS="${NUM_ENVS:-32}"
NUM_EPISODES="${NUM_EPISODES:-32}"
EVAL_NAME="${EVAL_NAME:-post_open_diagnostics_v2}"

# Empty preserves the checkpoint config. Accepted explicit overrides are true/false.
STAGE5_ARM_REST_ATTRACTOR="${STAGE5_ARM_REST_ATTRACTOR:-}"
extra_overrides=()
if [[ -n "$STAGE5_ARM_REST_ATTRACTOR" ]]; then
    case "$STAGE5_ARM_REST_ATTRACTOR" in
        true|false)
            extra_overrides+=(
                "++env.config.stage5_arm_rest_attractor=$STAGE5_ARM_REST_ATTRACTOR"
            )
            ;;
        *)
            printf 'STAGE5_ARM_REST_ATTRACTOR must be true, false, or empty\n' >&2
            exit 2
            ;;
    esac
fi

test ! -e "$OUTPUT_DIR"
test -f "$CHECKPOINT"
test -f "$(dirname "$CHECKPOINT")/config.yaml"
test -f "$INSTRUMENTATION_DIR/post_open_eval_launcher.py"
test -f "$INSTRUMENTATION_DIR/analyze_post_open_v2.py"
test -f "$CONDA_SH"
test -f "$RUNTIME_ENV"
test -d "$LAFAN_DIR"
test -d "$CODE_DIR"
mkdir -p "$OUTPUT_DIR/hydra" "$OUTPUT_DIR/diag"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
source "$RUNTIME_ENV"

export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/sdb/mashijian/coordex/.cache}"
export TMPDIR="${TMPDIR:-/sdb/mashijian/coordex/tmp}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
unset ACCELERATE_TORCH_DEVICE || true
ulimit -n 65536
cd "$CODE_DIR"

set +e
DOORMAN_POST_OPEN_DIAG_DIR="$OUTPUT_DIR/diag" \
CUDA_VISIBLE_DEVICES="$GPU" python "$INSTRUMENTATION_DIR/post_open_eval_launcher.py" \
    +checkpoint="$CHECKPOINT" \
    ++headless=true \
    ++num_envs="$NUM_ENVS" \
    ++seed=42 \
    ++use_wandb=false \
    ++auto_load_latest=false \
    ++simulator.config.render_results=false \
    ++env.config.enable_staged_reset=false \
    ++env.config.reset_from_dataset.motion_file_dir="$LAFAN_DIR" \
    ++algo.config.eval.num_eval_episodes="$NUM_EPISODES" \
    ++algo.config.eval.eval_num_envs_episodes=true \
    ++algo.config.eval.save_videos=false \
    ++algo.config.eval.save_goal_reached_only=false \
    ++eval_output_dir="$OUTPUT_DIR" \
    ++eval_name="$EVAL_NAME" \
    "${extra_overrides[@]}" \
    hydra.run.dir="$OUTPUT_DIR/hydra" \
    >"$OUTPUT_DIR/launcher.log" 2>&1
eval_status=$?
set -e

python "$INSTRUMENTATION_DIR/analyze_post_open_v2.py" "$OUTPUT_DIR/diag"
python - "$OUTPUT_DIR" "$eval_status" "$NUM_EPISODES" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
eval_status = int(sys.argv[2])
expected = int(sys.argv[3])
summary = json.loads((output / "diag/post_open_summary.json").read_text())
analysis = json.loads((output / "diag/post_open_analysis_v2.json").read_text())
assert eval_status == 0, eval_status
assert summary["schema_version"] == 2, summary
assert summary["episode_count"] == expected, summary
assert summary["active_unfinished_env_count"] == 0, summary
assert analysis["episode_count"] == expected, analysis
assert (output / "diag/post_open_kinematics_v2.csv").stat().st_size > 1024
print(
    f"diagnostic complete; evaluator_exit={eval_status}; "
    f"success={analysis['successful_episode_count']}/{analysis['episode_count']}; "
    f"output={output}"
)
PY
