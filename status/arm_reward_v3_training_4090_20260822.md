# 4090 arm-reward-v3 training status — 2026-08-22

## Decision basis

Arm reward-v2 completed its precommitted step6050→6400 window but failed the
final attractor-independence gate: step6400 succeeded on 115/128 full tasks with
the training attractor and only 1/128 with that attractor disabled. V3 is the
single-component fallback frozen before this result was observed.

## Controlled change

V3 replaces only the saturated aggregate harmful-action reward:

`sum(raw_arm_delta * target_offset) > 0`

with a continuous per-joint signal:

`mean(relu(raw_delta_j * sign(target_offset_j)) * normalized_abs_target_offset_j)`

Corrective commands contribute zero. More wrong-direction joints, larger raw
deltas, and larger existing target displacement increase the penalty. The old
binary harmful fraction remains logged for direct comparison. New objective
metrics are:

- `arm_v3_harmful_joint_fraction`;
- `arm_v3_harmful_outward_magnitude`;
- outward magnitude std, p95, and max.

No pose, signed-progress, settled-delta, attractor, staged-reset, PPO,
observation, action-scale, completion, or non-arm reward setting changed.

## Calibration and verification

The isolated step6050 probe initially used weight `-0.15` and measured stable
late windows with:

- old binary harmful fraction approximately 0.96–0.97;
- harmful joint fraction approximately 0.70–0.72;
- outward magnitude mean approximately 0.10–0.11;
- outward magnitude p95 approximately 0.20–0.21;
- episode reward contribution approximately `-0.0029`.

Weight `-1.4` preserves the original harmful-term budget. A separate
calibrated smoke reached iteration 6062 with contribution `-0.0253`, harmful
joint fraction 0.7132, outward mean 0.1075, p95 0.2005, and max 0.2524. Four
ranks remained healthy with no NaN, CUDA OOM, or NCCL failure.

The first attempted 256-environment probe used production batch geometry and
was discarded after a zero-microbatch `IndexError`; it produced no checkpoint.
The valid isolated probes used per-device batch 64 and accumulation 1. Formal
training retains production geometry.

## Formal run

- Server root: `/sda/mashijian/doorman_arm_reward_v3_20260822`
- Code branch: `continuation/arm-reward-v3-20260822`
- Implementation commits: `1f22389`, `a8510d5`
- Run: `/sda/mashijian/doorman_arm_reward_v3_20260822/runs/door_open_homie_arm_reward_v3`
- Anchor: step6050, SHA256
  `2efa1b726fc96c4305eacee43749eb286d375cc28adbb1cf824ad4354b597837`
- Fixed causal window: step6050→step6400
- Physical GPUs: 2, 3, 4, 5; GPUs 0 and 1 remain untouched
- Optimizer/LR state: reset once at the step6050 fork
- Supervisor: `/sda/mashijian/doorman_arm_reward_v3_20260822/supervise_4090_arm_v3.sh`

The supervisor was launched at 13:44 CST. It validates stable checkpoints,
restarts only this owned run if necessary, preserves optimizer state after the
initial fork, and stops at step6400. Read-only health and objective-metric
checks remain on fixed 15-minute boundaries; scheme changes are prohibited
inside the window except for invalid state, crash, NaN, or clear reward hacking.

The 13:57 CST startup window reached iteration 6054 with the supervisor and all
four ranks healthy, no NaN/OOM/NCCL error, and normal GPU2–5 occupancy. Stage-5
exposure was still zero in this initial pre-reset segment, so zero V3 arm metrics
were treated as expected missing exposure rather than as good behavior.

Final acceptance repeats the same 128-episode step6050/step6400 paired eval and
same-checkpoint attractor-on/off intervention used for V2. V3 is accepted only
if objective arm errors/action RMS improve, harmful joint magnitude falls,
task success stays within the safety gate, and attractor-off behavior approaches
the trained configuration.

## Step6100 direction gate

The first immutable V3 checkpoint was evaluated against the same step6050
anchor for 32 full-reset episodes per side. Both checkpoints completed 26/32
tasks, so V3 incurred no early task-success loss. Relative to step6050, only
three of eight Q4 arm/action metrics and four of six physical-error slopes
improved; all final absolute arm gates still failed.

A stricter same-step comparison against V2 step6100 found that V3 improved only
two of eight Q4 metrics and one of six slopes. V3 left/right raw action RMS was
0.6150/0.6368, compared with 0.5916/0.6174 for V2. This is retained as negative
evidence: the continuous signal is measurable and correctly budgeted, but has
not yet produced better early behavior than the binary V2 term.

The local artifact bundle is:

`../../eval_results/arm_reward_v3_gate_step6100_20260822_1428/`

Training remains frozen through the planned observation window. This early
gate does not authorize reward/config changes; the next fixed behavioral gate
is step6200.

This file is outside `github/` and `arxiv/`; neither directory is modified.
