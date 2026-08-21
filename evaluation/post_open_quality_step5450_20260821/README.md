# Step-5450 post-open behavior-quality diagnostic

This record diagnoses the sideways/crab traversal and abnormal arm posture
seen after door opening in the official teacher step-5450 checkpoint. It adds
new evaluation documentation only; archived `github/` and `arxiv/` content was
not modified.

## Decision summary

The full task still succeeds in 117/128 episodes (91.41%), but success masks
two systematic quality failures:

1. Stage-5 traversal is strongly sideways. Across the 117 successes, the mean
   episode p95 crab angle is 49.57 degrees and 78.10% of moving Stage-5 samples
   exceed 30 degrees. The mean HOMIE lateral-command ratio is 0.706 versus an
   actual lateral-velocity ratio of 0.588. Thus the task-level command geometry
   is already strongly oblique; a mean 0.801 m/s tracking error shows a second,
   locomotion-execution contribution.
2. Arms do not obtain a strict terminal rest pose. A loose average normalized
   arm-error threshold reports recovery, but the official worst-joint terminal
   check records zero stable steps in all 117 successful episodes. The mean
   best worst-joint error is 1.154 rad. Persistent error is concentrated in
   left wrist yaw, right wrist pitch, left wrist roll, and left wrist pitch.
   Raw arm delta-action RMS rises from 0.405 in Stage 4 to 0.489 in Stage 5 and
   0.654 in the completion tail, so the policy keeps driving the arms instead
   of relaxing them.

This is evaluation evidence, not authorization to change the official reward.
The next controlled ablations should separately test command-geometry/heading
shaping and post-grasp arm-release/rest shaping, while keeping the official
baseline unchanged.

## Frozen protocol

- Checkpoint: step 5450 `last.pt`
- SHA-256: `0fa4c1269e4aea4fec7a196f143ce7ff8a224c4041460b2141126876292d7969`
- Seed: 42
- Full resets: `env.config.enable_staged_reset=false`
- Quantitative lane: 128 environments, GPU 2, no rendering
- Qualitative lane: 8 environments, GPU 3, external scene rendering
- Ego recorder: disabled (`algo.config.eval.save_videos=false`)
- Viewer output: 8 MP4s, 256x256, 50 FPS, 11.64 s
- Viewer successes: env 0, 1, 3, 4, 5, 6, 7; failure: env 2
- Runtime module: `/sdb/mashijian/coordex/doorman/GR00T-VisualSim2Real/gr00t/rl/envs/door/door_open_homie.py`
- Runtime commit observed after evaluation: `cfa27d7a262e6bae7b89e9e93b7759f5adcd0ceb`

The import hook records metrics before the stage transition/reset callback. It
does not mutate reward, observation, action, transition, termination, reset,
or checkpoint state. The unchanged 117/128 result matches the prior
uninstrumented step-5450 evaluation.

## Metric definitions

- Crab angle: `atan2(abs(body_vy), body_vx)` in the pelvis/body frame, ignored
  below 0.1 m/s.
- Lateral velocity ratio: `abs(body_vy) / norm(body_vxy)`.
- HOMIE lateral-command ratio: `abs(cmd_vy) / norm(cmd_vxy)` using
  `get_physical_homie_commands()`.
- Tracking error: Euclidean error between body-frame planar velocity and the
  physical HOMIE planar command.
- Arm rest error: absolute deviation of the 14 non-finger upper-body joints
  from `resting_dof_pos`, reported both in radians and normalized by range.
- Arm activity: RMS of raw and cumulative arm delta actions.
- Windows: Stage 4 swing, Stage 5 through, completion tail, and their post-open
  union.

## Reproduction

The exact evaluated scripts are stored locally at:

```text
/Users/Admin/Projects/msj-wujie/msj-doorman/eval_results/
  official_teacher_step5450_post_open_quality_20260821/instrumentation/
```

On the 4090 server they were deployed under:

```text
/sda/mashijian/doorman_official_eval_20260821/step5450/
  post_open_quality_v1/instrumentation/
```

The two lanes can be reproduced with:

```bash
REMOTE_ROOT=/sda/mashijian/doorman_official_eval_20260821/step5450/post_open_quality_v1
CHECKPOINT=/sda/mashijian/doorman_official_eval_20260821/step5450/checkpoint_bundle/last.pt \
OUTPUT_DIR="$REMOTE_ROOT/results_reproduction" \
INSTRUMENTATION_DIR="$REMOTE_ROOT/instrumentation" \
METRICS_GPU=2 VIEWER_GPU=3 SEED=42 \
"$REMOTE_ROOT/instrumentation/run_step5450_post_open_eval.sh"
```

The shell script launches the 128-environment non-rendered lane and the
8-environment third-person lane concurrently. Do not point it at GPU 0 or 1;
those devices were occupied by other jobs during this evaluation.

## Results and artifacts

The complete local package is:

```text
/Users/Admin/Projects/msj-wujie/msj-doorman/eval_results/
  official_teacher_step5450_post_open_quality_20260821/
```

Key files:

```text
README.md
analysis_summary.json
results/metrics128/metrics_eval.json
results/metrics128/post_open_diag/post_open_summary.json
results/metrics128/post_open_diag/post_open_episode_metrics.jsonl
results/metrics128/post_open_diag/post_open_timeseries.csv
results/viewer8/metrics_eval.json
results/viewer8/post_open_diag/post_open_summary.json
results/viewer8/post_open_diag/post_open_episode_metrics.jsonl
results/viewer8/viewer/*.mp4
previews/contact_sheet_all_t9.png
instrumentation/
```

Use the 128 lane for quantitative conclusions. The viewer lane is a separate
qualitative subset whose environment IDs map directly to `viewer_<id>.mp4`.
