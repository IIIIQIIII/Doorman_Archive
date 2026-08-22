# 4090 arm-reward V4-A timing ablation — 2026-08-22

## Why V3 was stopped

The final full-reset diagnostic compared 32 episodes from the same evaluation
configuration and seed:

- step6050: 26/32 complete tasks (81.25%);
- step6300: 20/32 complete tasks (62.50%);
- change: -18.75 percentage points, outside the predeclared -5 point safety
  margin.

At the same time, seven of eight Q4 bilateral arm/action metrics and five of six
physical-error slopes improved from step6050 to step6300. All eight Q4 metrics
also improved from step6200 to step6300 while task completion fell from 26/32
to 20/32. This is a Pareto trade-off, not evidence that the arm signal itself is
wrong: V3 increasingly optimizes arm recovery at the expense of completing the
door traversal.

The training-side `average_goal_reached` value is not treated as full-task
success because it mixes Stage-0 episodes with curriculum staged resets.

## Controlled V4-A change

V4-A changes only when the four existing Stage-5 arm terms become active. Their
functions and weights are unchanged:

- bilateral pose error V2;
- signed recovery progress V2;
- continuous harmful outward action V3;
- settled-state delta suppression V2.

Task progress follows the environment's existing Stage-5 semantics:

`progress = clamp((root_x - env_origin_x) / 1.5, 0, 1)`

The maximum progress reached in the episode is latched. A smoothstep gate rises
from zero at progress 0.70 to one at progress 0.90, and all four arm terms are
multiplied by this same gate. Because the gate is monotonic and latched, the
policy cannot turn arm shaping off by stopping or walking backwards.

No arm formula, arm weight, non-arm reward, staged-reset ratio, observation,
action scale, PPO hyperparameter, or task completion condition changed.

## Reward-independent observability

V4-A adds the following objective metrics without feeding them into reward:

- latched task-progress mean and arm reward-gate mean;
- pre-gate, transition-gate, and full-gate exposure fractions;
- episode reset-source tracking;
- cumulative Stage-0 full-reset episode count and Stage-0-to-final-stage success
  rate.

The existing bilateral total/proximal/wrist errors, action RMS, recovery
efficiency, relapse, and harmful-action magnitude remain available. The new
full-reset split is the safety metric; mixed `average_goal_reached` remains
diagnostic only.

## Formal run

- Server root: `/sda/mashijian/doorman_arm_reward_v4a_timing_20260822`
- Code branch: `continuation/arm-reward-v4a-timing-20260822`
- Implementation commit: `e11035b`
- Run: `/sda/mashijian/doorman_arm_reward_v4a_timing_20260822/runs/door_open_homie_arm_reward_v4a`
- Anchor: V3 step6200, embedded `global_step=6200`, SHA256
  `85b9a632700816319906c20b877d9adb69b331ba375feb291012779da89e9a1f`
- Fixed causal window: step6200→step6300
- Checkpoint interval: 25 iterations
- Physical GPUs: 2, 3, 4, 5; GPUs 0 and 1 remain untouched
- Optimizer state: reset once at the step6200 fork
- Supervisor: `/sda/mashijian/doorman_arm_reward_v4a_timing_20260822/supervise_4090_arm_v4a.sh`

Static compilation, diff checks, gate monotonicity checks, checkpoint integrity,
reward registration, reset-source execution, and a reduced two-rank rollout all
passed. The reduced rollout intentionally did not count as a PPO-update test:
its smaller-than-production batch geometry produced the trainer's known
zero-microbatch `IndexError` after rollout collection. The formal run retains
the already validated V3 production geometry.

The supervisor launched the four-rank formal run at 16:24 CST. Read-only
monitoring is fixed at 15-minute boundaries. No reward/config change is allowed
inside this 100-iteration window; only a crash, NaN/OOM/NCCL failure, invalid
checkpoint, or a confirmed task-success safety violation permits intervention.

Final acceptance uses the same paired full-reset protocol. V4-A must preserve
task completion within five percentage points of the reference while improving
post-open bilateral arm behavior. If it preserves success but arm degradation
returns, the next experiment is a separate V4-B no-worsening barrier; it is not
part of V4-A.

This file is outside `github/` and `arxiv/`; neither directory is modified.
