# 4090 arm-reward V4-B latched barrier — 2026-08-22

## Causal target

V4-A preserved full-task success and improved seven of eight late arm/action
metrics, but did not prevent bilateral error and action RMS from rising again
after recovery. Code inspection exposed a specific loophole:
`penalty_stage5_arm_settled_delta_v2` applies only while `settled=True`; after a
relapse clears that state, the constraint turns off.

An offline reconstruction from the step6300 full-reset kinematic trajectories
found 115 episodes that entered Stage 5. All 115 first satisfied the existing
persistent bilateral recovery rule and all 115 later exceeded the relapse
boundary. This supports testing an irreversible post-recovery constraint rather
than increasing the always-on pose penalty.

## Single controlled change

V4-B adds one reward term and otherwise freezes V4-A:

```text
active = Stage5 AND ever_settled
left_excess  = relu(left_error  - 0.12)
right_excess = relu(right_error - 0.12)
barrier_raw = active * 0.5 * (left_excess^2 + right_excess^2)
reward = -12 * barrier_raw * V4A_latched_task_progress_gate
```

`ever_settled` is latched for the rest of the episode, so deliberate relapse
cannot disable the barrier. Separate left/right hinges prevent cancellation.
The squared hinge is zero inside the existing hysteresis boundary and smooth at
the boundary. The unchanged V4-A progress gate keeps it out of the early
post-open locomotion interval where task conflict was previously observed.

Replay calibration estimates an ungated contribution of approximately
`-0.026/episode` at weight `-12`, comparable to the existing harmful-action
auxiliary term and not dominant over task reward.

## Implementation and verification

- server root: `/sda/mashijian/doorman_arm_reward_v4b_barrier_20260822`;
- branch: `continuation/arm-reward-v4b-barrier-20260822`;
- implementation commit: `1330a8b`;
- run: `/sda/mashijian/doorman_arm_reward_v4b_barrier_20260822/runs/door_open_homie_arm_reward_v4b`;
- anchor: V4-A step6300, SHA256
  `46af3771e4bcafd9022c5f6ec0733b7fffe3deaffaab771ffbec06193ba74271`.

Only three tracked source/config files differ from V4-A. Python compilation,
YAML parsing, reward-method registration, gate/latch formula tests, and diff
checks passed. A 32-environment runtime rollout completed 32 episodes without a
V4-B state/metric exception. Its wrapper was intentionally not used as a
performance result: requesting 16 episodes with 32 parallel environments
produced 32 episodes, so the wrapper's equality assertion rejected the count.

## Formal fixed window

The supervisor started at 18:04 CST:

- fixed window: step6300 -> step6400;
- checkpoints: every 25 iterations;
- production geometry: four ranks, 4096 environments;
- physical GPUs: 2, 3, 4, 5; GPUs 0 and 1 remain untouched;
- optimizer and policy-noise state: preserved, not reset;
- PPO, reset curriculum, task rewards, V4-A gate, observations, actions, and
  terminations: unchanged;
- read-only reporting cadence after startup: 15 minutes.

Initial process verification found all four owned ranks alive with the expected
checkpoint, target step, V4-B threshold, and `reset_optimizer=False` arguments.
No startup Traceback, OOM, NCCL, runtime, or attribute error was present.

## Metrics and predeclared acceptance

V4-B adds objective, reward-independent telemetry for barrier-active exposure,
barrier-violation fraction, raw barrier magnitude, left/right excess, and the
fraction that has relapsed after recovery. Existing full-reset success and all
bilateral total/proximal/wrist error, action RMS, settled, relapse, harmful
action, and gate-exposure metrics remain unchanged.

The final paired 128-episode full-reset evaluation must preserve step6300 task
success within five percentage points. Directionally, relapse/barrier
violations and both late action RMS values must fall without suppressing
recovery exposure. The existing absolute gates remain: both total-error Q4/Q2
ratios no greater than 1.2, all error slopes no greater than 0.005/s,
left-wrist Q4 below 0.12, right-proximal Q4 below 0.06, and both action-delta Q4
values below 0.25.

No reward or optimizer change is allowed inside the window. Intervention is
limited to completion, invalid checkpoint, rank loss, NaN/OOM/NCCL failure, or
a confirmed task-success safety violation.

This file is outside `github/` and `arxiv/`; neither directory is modified.
