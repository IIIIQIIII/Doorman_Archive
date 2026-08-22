#!/usr/bin/env bash
set -euo pipefail

# Required inputs. CHECKPOINT must have its matching config.yaml beside it.
: "${CHECKPOINT:?absolute path to checkpoint .pt}"
: "${OUTPUT_DIR:?absolute output directory}"

# Defaults match the 4090 reproduction environment used on 2026-08-21.
CODE_DIR="${CODE_DIR:-/sda/mashijian/doorman_ablation_20260819/grasp_torque/code}"
CONDA_SH="${CONDA_SH:-/sdb/mashijian/coordex/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/sdb/mashijian/coordex/envs/doorman}"
RUNTIME_ENV="${RUNTIME_ENV:-/sdb/mashijian/coordex/doorman/GR00T-VisualSim2Real/scripts/doorman_runtime_env.sh}"
LAFAN_DIR="${LAFAN_DIR:-/sdb/mashijian/coordex/doorman/LAFAN-G1}"
METRICS_GPU="${METRICS_GPU:-2}"
VIEWER_GPU="${VIEWER_GPU:-3}"
SEED="${SEED:-42}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-checkpoint}"

test -f "$CHECKPOINT"
test -f "$(dirname "$CHECKPOINT")/config.yaml"
test -d "$CODE_DIR"
test -f "$CONDA_SH"
test -f "$RUNTIME_ENV"
test -d "$LAFAN_DIR"

mkdir -p "$OUTPUT_DIR/metrics128/hydra"
mkdir -p "$OUTPUT_DIR/viewer8/hydra" "$OUTPUT_DIR/viewer8/viewer"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
source "$RUNTIME_ENV"

export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/sdb/mashijian/coordex/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/sdb/mashijian/coordex/.cache/pip}"
export TMPDIR="${TMPDIR:-/sdb/mashijian/coordex/tmp}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export HYDRA_FULL_ERROR=1
unset ACCELERATE_TORCH_DEVICE || true
ulimit -n 65536

cd "$CODE_DIR"

CUDA_VISIBLE_DEVICES="$METRICS_GPU" python gr00t/rl/eval_agent_trl.py \
    +checkpoint="$CHECKPOINT" \
    ++headless=true \
    ++num_envs=128 \
    ++seed="$SEED" \
    ++use_wandb=false \
    ++auto_load_latest=false \
    ++simulator.config.render_results=false \
    ++env.config.enable_staged_reset=false \
    ++env.config.reset_from_dataset.motion_file_dir="$LAFAN_DIR" \
    ++algo.config.eval.num_eval_episodes=128 \
    ++algo.config.eval.eval_num_envs_episodes=true \
    ++algo.config.eval.save_videos=false \
    ++algo.config.eval.save_goal_reached_only=false \
    ++eval_output_dir="$OUTPUT_DIR/metrics128" \
    ++eval_name="official_${CHECKPOINT_LABEL}_metrics128" \
    hydra.run.dir="$OUTPUT_DIR/metrics128/hydra" \
    >"$OUTPUT_DIR/metrics128/launcher.log" 2>&1 &
metrics_pid=$!

# render_results/save_rendering_dir creates the external scene-camera MP4s.
# save_videos=false intentionally disables the evaluator's ego-view recorder.
CUDA_VISIBLE_DEVICES="$VIEWER_GPU" python gr00t/rl/eval_agent_trl.py \
    +checkpoint="$CHECKPOINT" \
    ++headless=true \
    ++num_envs=8 \
    ++seed="$SEED" \
    ++use_wandb=false \
    ++auto_load_latest=false \
    ++simulator.config.render_results=true \
    ++env.config.save_rendering_dir="$OUTPUT_DIR/viewer8/viewer" \
    ++env.config.enable_staged_reset=false \
    ++env.config.reset_from_dataset.motion_file_dir="$LAFAN_DIR" \
    ++algo.config.eval.num_eval_episodes=8 \
    ++algo.config.eval.eval_num_envs_episodes=true \
    ++algo.config.eval.save_videos=false \
    ++algo.config.eval.save_goal_reached_only=false \
    ++eval_output_dir="$OUTPUT_DIR/viewer8" \
    ++eval_name="official_${CHECKPOINT_LABEL}_viewer8_thirdperson" \
    hydra.run.dir="$OUTPUT_DIR/viewer8/hydra" \
    >"$OUTPUT_DIR/viewer8/launcher.log" 2>&1 &
viewer_pid=$!

set +e
wait "$metrics_pid"
metrics_status=$?
wait "$viewer_pid"
viewer_status=$?
set -e

if (( metrics_status != 0 || viewer_status != 0 )); then
    printf 'Evaluation failed: metrics=%d viewer=%d\n' "$metrics_status" "$viewer_status" >&2
    exit 1
fi

test -s "$OUTPUT_DIR/metrics128/metrics_eval.json"
test -s "$OUTPUT_DIR/viewer8/metrics_eval.json"
test "$(find "$OUTPUT_DIR/viewer8/viewer" -maxdepth 1 -name '*.mp4' -size +48c | wc -l)" -eq 8

printf 'Evaluation complete: %s\n' "$OUTPUT_DIR"
