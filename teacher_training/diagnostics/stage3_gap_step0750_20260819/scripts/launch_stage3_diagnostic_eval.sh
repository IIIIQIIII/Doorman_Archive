#!/usr/bin/env bash
set -euo pipefail

REPRO_ROOT=/data1/mashijian/coordex/doorman/reproductions/official_teacher_016c70c_20260819
CODE_DIR=${REPRO_ROOT}/code
DIAG_ROOT=${REPRO_ROOT}/diagnostics/stage3_gap_step0750_20260819
CHECKPOINT=${DIAG_ROOT}/checkpoint_step_000750.pt
PYTHON_BIN=/data1/mashijian/coordex/envs/doorman/bin/python
LAFAN_DIR=/data1/mashijian/coordex/doorman/LAFAN-G1

mkdir -p "${DIAG_ROOT}/eval_output" "${DIAG_ROOT}/hydra"
cd "${CODE_DIR}"

export PYTHONPATH="${DIAG_ROOT}:${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/data1/mashijian/coordex/envs/doorman/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=2
export ACCELERATE_TORCH_DEVICE=cuda:0
export DOORMAN_DIAG_DIR="${DIAG_ROOT}"

exec "${PYTHON_BIN}" "${DIAG_ROOT}/instrumented_eval_launcher.py" \
  checkpoint="${CHECKPOINT}" \
  +num_envs=128 \
  +headless=true \
  +use_wandb=false \
  +env.config.enable_staged_reset=true \
  +env.config.randomize_door_init_state=false \
  +env.config.reset_from_dataset.motion_file_dir="${LAFAN_DIR}" \
  +simulator.config.cameras.enable_cameras=false \
  +simulator.config.render_results=false \
  algo.config.eval.num_eval_episodes=128 \
  +algo.config.eval.eval_num_envs_episodes=true \
  algo.config.eval.save_videos=false \
  algo.config.eval.save_trajectories=false \
  +eval_output_dir="${DIAG_ROOT}/eval_output" \
  hydra.run.dir="${DIAG_ROOT}/hydra"
