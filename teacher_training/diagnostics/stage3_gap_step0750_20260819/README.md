# DoorMan Stage 3 to Stage 4 diagnostic report

Date: 2026-08-19 (Asia/Shanghai)

## Scope

- Official clean source commit: `016c70c1e4e76f521963c36691ee69a6ab3ac9cd`
- Diagnostic checkpoint: `model global step 750`
- Evaluation: 128 environments, one first episode per environment
- Training reward, observations, actions, stage transitions, termination, and checkpoint weights: unchanged
- Instrumentation method: runtime-only import hook outside the clean worktree
- Cameras and trajectory saving: disabled

## Stage transition result

- Stage 3 entries: 118
- Stage 3 to Stage 4 transitions: 0
- Stage 4 reach rate conditioned on Stage 3 entry: 0%
- Stage 3 samples: 41,111 policy timesteps

## Door and handle state

- Hinge angle mean: 0.002106 rad (0.121 degrees)
- Hinge angle p95: 0.006449 rad (0.370 degrees)
- Hinge angle max: 0.016313 rad (0.935 degrees)
- Stage 4 threshold: 0.174533 rad (10 degrees)
- Hinge velocity mean: 0.000452 rad/s
- Positive hinge velocity p95: 0.010828 rad/s
- Handle angle mean: 0.739854 rad (42.39 degrees)
- Handle angle p95: 0.785398 rad (45 degrees)
- Fraction of Stage 3 time with handle angle above 35 degrees: 92.10%

## Contact force and effective torque

- Raw hand-force XYZ mean: `[-2.107, -22.374, -32.229] N`
- Raw hand-force XYZ absolute p95: `[18.099, 34.618, 56.801] N`
- Effective opening torque mean: 2.122 Nm
- Effective opening torque absolute p95: 14.856 Nm
- Effective opening torque p95: 13.245 Nm
- Effective opening torque max: 47.245 Nm
- Mean tangential alignment: 0.0589
- Positive tangential-alignment fraction: 63.92%
- Correlation of opening torque with hinge velocity: 0.108
- Correlation of opening torque with hinge angle: 0.122

The hinge point and positive hinge axis are reconstructed from the official door
generator using door width, handedness, root pose, and the generated joint
transform. Torque is computed at the official grasp target using the same force
sign convention consumed by `push_door_force`.

## Grasp persistence

- Mean handle-contact count: 2.92
- Handle-contact count p95: 4
- Fraction with at least four valid contacts: 10.72%
- Fraction with at least four contacts and opening torque above 1 Nm: 5.32%
- Longest continuous valid-contact interval: 0.68 seconds
- Longest continuous torque-above-1-Nm interval: 1.24 seconds
- Longest continuous valid-contact and torque-above-1-Nm interval: 0.38 seconds

## Diagnosis

The policy reliably depresses the handle and generates substantial contact
force, but it does not maintain the valid grasp and tangential opening torque
long enough to rotate the panel. Force and torque are oscillatory, hinge velocity
responds only weakly, and hinge angle remains below one degree in every Stage 3
rollout. The current bottleneck is therefore persistent force transmission from
the grasped handle into the door hinge, not handle release itself or a lack of
force magnitude.

No reward was changed. If a later controlled ablation introduces a bridge term,
the evidence supports gating it on handle-down plus valid multi-contact grasp and
rewarding sustained hinge-axis torque/tangential alignment. An ungated force
bonus would increase the existing contact-farming risk.

## Artifacts

Archive contents:

- `stage_events.csv`: Stage entry and transition events
- `stage3_diagnostic_summary.json`: aggregate statistics
- `stage3_timeline_longest.png`: Fx, opening torque, hinge velocity, and hinge angle on one time axis
- `diagnostic_metadata.json`: force and torque definitions
- `scripts/`: runtime instrumentation, launcher, and analysis scripts

The larger raw time-series, evaluation log, and checkpoint are retained on the
training server and are not duplicated in this compact archive.

Server directory:

`/data1/mashijian/coordex/doorman/reproductions/official_teacher_016c70c_20260819/diagnostics/stage3_gap_step0750_20260819`

- `stage3_timeseries.csv`: raw Stage 3 time-series samples
- `stage_events.csv`: Stage entry and transition events
- `stage3_diagnostic_summary.json`: aggregate statistics
- `stage3_timeline_longest.png`: Fx, opening torque, hinge velocity, and hinge angle on one time axis
- `diagnostic_metadata.json`: force and torque definitions
- `checkpoint_step_000750.pt`: immutable diagnostic checkpoint snapshot
- `diagnostic_eval.log`: evaluation log
