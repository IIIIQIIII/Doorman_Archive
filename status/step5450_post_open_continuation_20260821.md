# Step 5450 post-open continuation status (2026-08-21)

## Decision and source

- Local Archive was fast-forwarded to `9ac08c8` (`Add step 5450 continuation training plan`).
- The implementation follows `evaluation/post_open_quality_step5450_20260821/step5450_continuation_training_plan.md`.
- The official baseline was stopped gracefully after it had reached step 5700. Its files were not modified.
- The continuation deliberately uses the immutable step 5450 anchor, SHA-256 `0fa4c1269e4aea4fec7a196f143ce7ff8a224c4041460b2141126876292d7969`.

## Implemented continuation

- Changes are isolated in the A100 worktree branch `continuation/step5450-postopen-fix-v1`.
- Code commits: `6b31f3153563353e23a92a6ae0ddc7258d558047` and `80797722875f24f242fb05c9442f11d4e4b8f345`.
- Stage 0-4 reward behavior is preserved. New shaping is gated to Stage 5:
  - crab-angle penalty: `-4`
  - HOMIE lateral-command penalty: `-6`
  - raw arm-delta RMS penalty: `-3`
  - worst normalized arm-rest joint penalty: `-10`
- Stage 5 uses target root X `1.70`, distance-based slowdown, and the `0.04` arm-rest attractor. Stage 4 keeps its prior root target.
- The optimizer/scheduler is reset while model weights and global step are retained. Actor/critic LR is `2e-5 / 5e-5`, entropy `0.002`, desired KL `0.002`, policy noise `0.35`; HOMIE remains frozen.
- Runtime scale is 4096 environments on GPU map `0,1,2,7`, 64 rollout steps, 3 PPO epochs, 4 minibatches, save interval 100 global steps.

## Launch correction and verification

The trainer interprets `algo.trl.num_total_batches` as an absolute global stopping step. Setting it to `1000` after resuming at global step 5450 caused the callback to stop after iteration 5451 and enter end-of-training save. This looked like a second-rollout stall but was not a reward or physics failure.

The formal continuation now uses `algo.trl.num_total_batches=6450`, representing at most 1000 additional steps. A single-GPU smoke test from the same anchor completed iterations 5451, 5452, and 5453 in 9.86 s, 9.51 s, and 10.00 s without traceback or NaN.

## Live A100 run

- Start: `2026-08-21 23:07:39 CST`
- Main PID at launch: `1135529`
- Run directory: `/data1/mashijian/coordex/doorman/continuations/step5450_postopen_fix_v1/runs/door_open_homie_step5450_postopen_fix_v1`
- Launcher log: `/data1/mashijian/coordex/doorman/continuations/step5450_postopen_fix_v1/launcher.log`
- Confirmed all four ranks loaded step 5450, reset policy noise to 0.35, reset optimizer/scheduler, and configured actor/critic LR `2e-5 / 5e-5`.
- Formal iterations 5451-5454 completed in 41.98 s, 38.51 s, 37.98 s, and 41.44 s. Training remained active afterward.

The first meaningful decision point is the step 5500 checkpoint/eval window. The first few post-resume episode aggregates are not representative because the completed-episode metric buffers restart empty.

## Repository boundary

No files under `github/` or `arxiv/` were modified by this update.
