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

The supervisor launched the four-rank formal run at 16:24 CST and completed the
frozen window at 17:22 CST. It validated step6300, waited 120 seconds for the
trainer to exit after checkpoint creation, then terminated only the owned
process group. There was no training Traceback, OOM, NCCL failure, or NaN.

- step6300 checkpoint size: 32,324,131 bytes;
- embedded `global_step`: 6300;
- SHA256: `46af3771e4bcafd9022c5f6ec0733b7fffe3deaffaab771ffbec06193ba74271`;
- final process state: no V4-A trainer or evaluator remained;
- GPUs 2--7 returned to 1 MiB/0% utilization; GPUs 0 and 1 were never used by
  this run.

The final online row at iteration 6300 reported mixed
`average_goal_reached=0.8802`, cumulative reset-source Stage-0 success 0.6975,
task-progress mean 0.6037, and gate mean 0.4034. The arm values were left/right
total error 0.1697/0.1162, proximal error 0.0649/0.0642, wrist error
0.2033/0.1198, raw action-delta RMS 0.6592, harmful outward magnitude 0.0946,
recovery progress -0.0289/s, settled fraction 0.1786, and relapse count 0.4854.
The Stage-0 number is cumulative and startup-biased, so it is only a trend
diagnostic; the paired full-reset evaluation below is authoritative.

## Final paired full-reset evaluation

The same V4 code/config, seed 42, staged resets disabled, and 128 environments /
128 completed episodes were used for both frozen checkpoints. The anchor ran on
GPU 6 and the candidate on GPU 7. Both summaries are schema v2, contain exactly
128 recorded episodes, and report zero unfinished environments.

- step6200: 115/128 task successes (89.84375%);
- step6300: 112/128 task successes (87.50%);
- change: -2.34375 percentage points;
- task-preservation gate (drop no greater than five points): **PASS**.

Compared with step6200, step6300 improved Q4 on seven of eight bilateral
arm/action metrics and improved all six physical-error slopes. The Q4 changes
were: left/right total error -0.0412/-0.0033, left/right proximal error
-0.0194/-0.0050, left/right wrist error -0.0703/-0.0011, left action-delta RMS
-0.1149, and right action-delta RMS +0.0388. The last item is the only Q4
regression.

V4-A nevertheless fails the predeclared absolute arm-quality gates: both total
error Q4/Q2 ratios remain above 1.2, not every error slope is at most 0.005/s,
left-wrist Q4 remains above 0.12, right-proximal Q4 remains above 0.06, and both
action-delta Q4 values remain above 0.25. Therefore V4-A validates the timing
hypothesis and preserves the task, but it does **not** fully solve the bilateral
post-open arm attractor. A separately scoped V4-B no-worsening barrier is the
next controlled experiment; restarting or silently extending V4-A would not be
a valid causal test.

The official evaluator attempted to write `metrics_eval.json` after completing
all 128 episodes and raised `TypeError: Object of type Tensor is not JSON
serializable`; those two files are consequently truncated and must not be used.
This is a post-rollout serialization defect. The independent diagnostics hook
had already written and validated the schema-v2 summary, per-episode JSONL, and
kinematic analysis, which are the inputs to the paired comparison. The defect
and complete launcher logs are retained in the archive instead of being hidden.

Reproduction script, immutable hashes, paired report, compact per-episode
evidence, and launcher/supervisor logs are under
`evaluation/arm_reward_v4a_final_step6300_20260822/`. Large raw time-series CSVs
and checkpoint binaries remain on the server and are referenced by hash rather
than duplicated in Git.

This file is outside `github/` and `arxiv/`; neither directory is modified.
