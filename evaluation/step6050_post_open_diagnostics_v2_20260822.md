# Step 6050 post-open diagnostics v2

Date: 2026-08-22 (Asia/Shanghai)

This evaluation extends the earlier post-open metrics with bilateral arm groups,
per-joint time progression, all three waist joints, pelvis/torso world yaw, and
torso-relative-to-pelvis yaw. It is runtime-only instrumentation: observations,
actions, rewards, transitions, resets, terminations, and checkpoint state were
not changed.

## Checkpoint and result

- Checkpoint: `model_step_006050.pt`
- SHA-256: `2efa1b726fc96c4305eacee43749eb286d375cc28adbb1cf824ad4354b597837`
- Episodes: 32
- Successful: 26
- Success rate: 81.25%
- Successful post-open samples: 7,874
- Eval GPU: physical GPU 6 on the 4090 server
- Training continued independently on physical GPUs 2-5.

The synchronized local result is:

`/Users/Admin/Projects/msj-wujie/msj-doorman/step6050_post_open_diag_v2_20260822_0838_retry1`

The remote result is:

`/sda/mashijian/doorman_migrations/step5500_postopen_fix_v1_20260822/evals/step6050_post_open_diag_v2_20260822_0838_retry1`

## Reproduction

The runtime instrumentation is installed remotely at:

`/sda/mashijian/doorman_migrations/step5500_postopen_fix_v1_20260822/evals/post_open_diagnostics_v2_instrumentation`

Run it with:

```bash
ROOT=/sda/mashijian/doorman_migrations/step5500_postopen_fix_v1_20260822
INSTR=$ROOT/evals/post_open_diagnostics_v2_instrumentation
CHECKPOINT=$ROOT/runs/door_open_homie_step5450_postopen_fix_v1/model_step_006050.pt \
OUTPUT_DIR=$ROOT/evals/step6050_post_open_diag_v2_20260822_0838_retry1 \
INSTRUMENTATION_DIR=$INSTR \
GPU=6 NUM_ENVS=32 NUM_EPISODES=32 \
EVAL_NAME=step6050_post_open_diag_v2_retry1 \
bash "$INSTR/run_diagnostics_v2.sh"
```

## Bilateral arm progression

The successful post-open portion of each episode was normalized and split into
four equal progress quartiles.

| Metric | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Left arm normalized rest error | 0.1438 | 0.0713 | 0.1209 | 0.1997 |
| Right arm normalized rest error | 0.0392 | 0.0304 | 0.0931 | 0.1456 |
| Left proximal error | 0.1120 | 0.0507 | 0.0573 | 0.0522 |
| Right proximal error | 0.0371 | 0.0219 | 0.0826 | 0.1249 |
| Left wrist error | 0.1863 | 0.0987 | 0.2058 | 0.3965 |
| Right wrist error | 0.0419 | 0.0418 | 0.1070 | 0.1731 |
| Left arm action-delta RMS | 0.2249 | 0.1969 | 0.4439 | 0.6056 |
| Right arm action-delta RMS | 0.2310 | 0.2190 | 0.5922 | 0.6179 |

From Q2 to Q4, left-arm error increases 2.80 times and right-arm error 4.79
times. The left-side late failure is dominated by all three wrist joints. The
right shoulder, elbow, and wrist all deteriorate, so the right-side failure is
a full kinematic-chain failure rather than a wrist-only failure.

## Waist and torso diagnosis

Waist yaw rest error does not exhibit a monotonic drift: its quartile means are
0.0443, 0.0273, 0.0273, and 0.0386 rad. Torso-relative-to-pelvis absolute yaw is
similarly 0.0435, 0.0294, 0.0274, and 0.0391 rad. Their per-episode slopes are
approximately zero.

However, the waist is continually rotating back and forth:

- waist-yaw path length: mean 1.6390 rad (93.9 degrees), p95 1.8308 rad;
- waist-yaw range: mean 0.1802 rad (10.3 degrees), p95 0.2164 rad;
- torso-relative-pelvis yaw path: mean 1.6536 rad, p95 1.8548 rad;
- waist-yaw absolute velocity: approximately 0.20-0.27 rad/s across quartiles,
  with p95 0.57-0.86 rad/s.

Therefore the observed waist behavior is real, but it is oscillatory rotation,
not accumulating yaw drift. Pelvis and torso world-yaw paths are both about 2.0
rad per successful episode, while their net final rotations are small. This is
consistent with repeated locomotion corrections.

## Coupling conclusion

Waist-yaw position has only weak correlation with left/right arm posture error
(raw Pearson r approximately 0.21/0.06; first-difference r approximately
0.15/0.11). Absolute waist-yaw velocity has essentially no correlation with
left/right arm action-delta RMS (r approximately 0.00/-0.08).

The waist oscillation and bilateral arm degradation coexist during post-open
locomotion, but the metrics do not support treating them as one directly coupled
failure. The arm degradation becomes strongest in Q3-Q4 while torso world-yaw
rate is decreasing. The more likely interpretation is two concurrent failures:

1. the task policy continues issuing large bilateral arm corrections during
   locomotion, causing left-wrist drift and right full-chain drift;
2. the frozen locomotion layer repeatedly rotates the waist/torso while
   tracking body commands, with no explicit waist-yaw quality metric or reward.

Primary machine-readable outputs are `diag/post_open_analysis_v2.json`,
`diag/post_open_kinematics_v2.csv`, and `diag/arm_joint_timeseries.csv` in the
synchronized result directory.
