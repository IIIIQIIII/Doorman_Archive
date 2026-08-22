#!/usr/bin/env bash
set -euo pipefail

# Run on the 4090 host after the candidate checkpoint has finished writing.
# This freezes the step-6200 anchor and candidate into immutable bundles, then
# evaluates both with identical code, seed, reset semantics, and diagnostics.

: "${OUTPUT_ROOT:?new absolute output directory required}"

V4_ROOT="${V4_ROOT:-/sda/mashijian/doorman_arm_reward_v4a_timing_20260822}"
RUN_DIR="${RUN_DIR:-$V4_ROOT/runs/door_open_homie_arm_reward_v4a}"
CODE_DIR="${CODE_DIR:-$V4_ROOT/code}"
ANCHOR="${ANCHOR:-$RUN_DIR/checkpoint_anchor/model_step_006200.pt}"
CANDIDATE_STEP="${CANDIDATE_STEP:-6300}"
printf -v CANDIDATE_FILE 'model_step_%06d.pt' "$CANDIDATE_STEP"
CANDIDATE_LABEL="step${CANDIDATE_STEP}"
CANDIDATE="${CANDIDATE:-$RUN_DIR/$CANDIDATE_FILE}"
INSTRUMENTATION_DIR="${INSTRUMENTATION_DIR:-/sda/mashijian/doorman_migrations/step5500_postopen_fix_v1_20260822/evals/post_open_diagnostics_v2_instrumentation}"
COMPARATOR="${COMPARATOR:-/sda/mashijian/doorman_arm_reward_v2_20260822/compare_arm_reward_v2_gate.py}"
CONDA_SH="${CONDA_SH:-/sdb/mashijian/coordex/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/sdb/mashijian/coordex/envs/doorman}"
ANCHOR_GPU="${ANCHOR_GPU:-6}"
CANDIDATE_GPU="${CANDIDATE_GPU:-7}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EPISODES="${NUM_EPISODES:-128}"

test ! -e "$OUTPUT_ROOT"
test -f "$ANCHOR"
test -f "$CANDIDATE"
test -f "$RUN_DIR/config.yaml"
test -x "$INSTRUMENTATION_DIR/run_diagnostics_v2.sh"
test -x "$COMPARATOR"
test -f "$CONDA_SH"
test -d "$CODE_DIR"

for gpu in "$ANCHOR_GPU" "$CANDIDATE_GPU"; do
    used_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( used_mib > 512 )); then
        printf 'Refusing occupied GPU %s: %s MiB used\n' "$gpu" "$used_mib" >&2
        exit 1
    fi
done

# Reject a candidate that is still being written.
candidate_size_1="$(stat -c %s "$CANDIDATE")"
sleep 10
candidate_size_2="$(stat -c %s "$CANDIDATE")"
test "$candidate_size_1" = "$candidate_size_2"
test "$candidate_size_2" -gt 1048576

mkdir -p "$OUTPUT_ROOT/checkpoints/step6200" "$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL"
cp --reflink=auto "$ANCHOR" "$OUTPUT_ROOT/checkpoints/step6200/model_step_006200.pt"
cp "$RUN_DIR/config.yaml" "$OUTPUT_ROOT/checkpoints/step6200/config.yaml"
cp --reflink=auto "$CANDIDATE" "$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL/$CANDIDATE_FILE"
cp "$RUN_DIR/config.yaml" "$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL/config.yaml"

sha256sum \
    "$OUTPUT_ROOT/checkpoints/step6200/model_step_006200.pt" \
    "$OUTPUT_ROOT/checkpoints/step6200/config.yaml" \
    "$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL/$CANDIDATE_FILE" \
    "$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL/config.yaml" \
    >"$OUTPUT_ROOT/checkpoints/SHA256SUMS"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
python - \
    "$OUTPUT_ROOT/checkpoints/step6200/model_step_006200.pt" 6200 \
    "$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL/$CANDIDATE_FILE" "$CANDIDATE_STEP" <<'PY'
import sys
import torch

for path, expected in zip(sys.argv[1::2], sys.argv[2::2]):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual = int(checkpoint["state"].global_step)
    assert actual == int(expected), (path, actual, expected)
    print(f"verified checkpoint global_step={actual}: {path}")
PY

mkdir -p "$OUTPUT_ROOT/evals"
(
    CODE_DIR="$CODE_DIR" \
    INSTRUMENTATION_DIR="$INSTRUMENTATION_DIR" \
    CHECKPOINT="$OUTPUT_ROOT/checkpoints/step6200/model_step_006200.pt" \
    OUTPUT_DIR="$OUTPUT_ROOT/evals/step6200" \
    GPU="$ANCHOR_GPU" \
    NUM_ENVS="$NUM_ENVS" \
    NUM_EPISODES="$NUM_EPISODES" \
    EVAL_NAME="arm_v4a_gate_step6200" \
    bash "$INSTRUMENTATION_DIR/run_diagnostics_v2.sh"
) >"$OUTPUT_ROOT/evals/step6200.supervisor.log" 2>&1 &
anchor_pid=$!

(
    CODE_DIR="$CODE_DIR" \
    INSTRUMENTATION_DIR="$INSTRUMENTATION_DIR" \
    CHECKPOINT="$OUTPUT_ROOT/checkpoints/$CANDIDATE_LABEL/$CANDIDATE_FILE" \
    OUTPUT_DIR="$OUTPUT_ROOT/evals/$CANDIDATE_LABEL" \
    GPU="$CANDIDATE_GPU" \
    NUM_ENVS="$NUM_ENVS" \
    NUM_EPISODES="$NUM_EPISODES" \
    EVAL_NAME="arm_v4a_gate_$CANDIDATE_LABEL" \
    bash "$INSTRUMENTATION_DIR/run_diagnostics_v2.sh"
) >"$OUTPUT_ROOT/evals/$CANDIDATE_LABEL.supervisor.log" 2>&1 &
candidate_pid=$!

set +e
wait "$anchor_pid"
anchor_status=$?
wait "$candidate_pid"
candidate_status=$?
set -e

if (( anchor_status != 0 || candidate_status != 0 )); then
    printf 'Paired diagnostic failed: step6200=%d %s=%d\n' \
        "$anchor_status" "$CANDIDATE_LABEL" "$candidate_status" >&2
    exit 1
fi

for step in step6200 "$CANDIDATE_LABEL"; do
    test -s "$OUTPUT_ROOT/evals/$step/metrics_eval.json"
    test -s "$OUTPUT_ROOT/evals/$step/diag/post_open_analysis_v2.json"
    test -s "$OUTPUT_ROOT/evals/$step/diag/post_open_episode_metrics.jsonl"
    grep -q 'diagnostic complete; evaluator_exit=0' "$OUTPUT_ROOT/evals/$step.supervisor.log"
done

python "$COMPARATOR" \
    "$OUTPUT_ROOT/evals/step6200/diag/post_open_analysis_v2.json" \
    "$OUTPUT_ROOT/evals/$CANDIDATE_LABEL/diag/post_open_analysis_v2.json" \
    --json-out "$OUTPUT_ROOT/paired_comparison.json" \
    --markdown-out "$OUTPUT_ROOT/PAIRED_COMPARISON.md"
test -s "$OUTPUT_ROOT/paired_comparison.json"
test -s "$OUTPUT_ROOT/PAIRED_COMPARISON.md"

printf 'Paired V4-A timing diagnostic complete: %s\n' "$OUTPUT_ROOT"
