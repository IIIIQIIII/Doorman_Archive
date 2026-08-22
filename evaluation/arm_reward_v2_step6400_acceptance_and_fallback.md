# Arm reward-v2 step6400 acceptance and controlled fallback

This document freezes the decision rule before the step6400 result is known. It
does not alter the running step6050→6400 experiment.

## Final evidence package

At step6400, freeze and hash the checkpoint and matching config before any eval.
Use the V2 code, seed 42, full resets, and identical diagnostics for:

1. step6050 anchor, attractor on, 128 completed episodes;
2. step6400 candidate, attractor on, 128 completed episodes;
3. step6400 candidate, attractor off, 128 completed episodes.

The first two runs are the controlled training comparison. The third determines
whether the policy learned recovery or merely relies on the environment's
Stage-5 target attractor. All evaluator exit markers, episode counts, checkpoint
identity, Hydra overrides, and schema-v2 outputs must pass before interpretation.

## Acceptance rule

V2 is accepted only if all of the following hold:

- attractor-on complete-task success is no more than 5 percentage points below
  the paired step6050 result;
- attractor-off success is no more than 5 percentage points below the same
  step6400 policy with the trained attractor configuration;
- left and right total-arm Q4/Q2 error ratios are at most 1.2;
- all bilateral total/proximal/wrist error slopes are at most `0.005/s`;
- Q4 left-wrist error is below 0.12 and Q4 right-proximal error below 0.06;
- Q4 raw action RMS is below 0.25 on both sides;
- training telemetry shows sustained bilateral recovery of at least 80%, relapse
  below 10%, and a material reduction from the approximately 96% harmful-action
  starting plateau;
- reward contributions and objective metrics agree; a better shaped return with
  worse physical behavior is a rejection.

Step6200 already fails these conditions and loses 28.12 percentage points of
task success when the attractor is disabled. Step6400 is still required because
the experiment committed to a fixed causal window rather than checkpoint
chasing.

## If step6400 rejects V2: first controlled V3

### Hypothesis

The current harmful-action signal is saturated and too coarse. It computes one
binary value per environment step:

`sum(raw_arm_delta * target_offset) > 0`

At approximately 96% activation it is nearly constant. Opposing joint effects
can cancel in the sum, and the term cannot distinguish a tiny outward command
from a large multi-joint outward command. Increasing its existing binary weight
would mostly increase a constant penalty and is not the first experiment.

### Single component change

Replace only `penalty_stage5_arm_harmful_action_v2` with a continuous,
per-joint, direction-sensitive outward-action score. Conceptually, for each arm
joint:

```text
outward_j = relu(raw_delta_j * sign(target_offset_j))
severity_j = clamp(abs(target_offset_j) / normalized_error_reference, 0, 1)
harmful_magnitude = mean(outward_j * severity_j)
```

Corrective commands are unpenalized. Larger wrong-direction commands and more
simultaneously wrong joints receive a larger penalty. Before choosing the new
weight, log raw mean/std/p95/max and select a weight whose initial episodic
contribution is comparable to the existing harmful term (about -0.028 at
step6200). This changes signal resolution, not the initial reward budget.

Keep all other factors fixed for the first V3 causal test:

- restart from the same step6050 anchor with the same one-time optimizer reset;
- retain arm pose and signed progress terms unchanged;
- retain the Stage-5 attractor and its alpha unchanged during training;
- retain staged-reset ratios, PPO settings, observations, action scale,
  completion semantics, and all non-arm rewards;
- add reward-independent `harmful_joint_fraction` and
  `harmful_outward_magnitude` metrics;
- use the same fixed window and attractor-on/off acceptance protocol.

### Confirmation

The hypothesis is supported only if harmful joint fraction/magnitude fall,
physical Q4 errors and slopes improve, attractor-off behavior approaches the
trained configuration, and task success remains within the paired safety gate.

### Rejection

Reject this hypothesis if the new signal contribution changes but the objective
arm metrics do not, or if the policy simply emits near-zero commands while the
accumulated target and physical arms remain displaced.

## Deferred second ablation

Do not simultaneously add a terminal arm-quality reward. If continuous harmful
action fails, the next separate hypothesis is that distributed pose/progress
penalties provide weak credit assignment for persistence at task completion. A
completion-state arm-quality term can then be tested alone. Keeping it separate
prevents an uninterpretable multi-change experiment and makes episode-length or
completion-delay hacking easier to detect.

## Observed step6400 result — 2026-08-22

The frozen V2 run reached step6400 and stopped under its supervisor. The final
checkpoint is 32,320,451 bytes with SHA256
`c63469fe7bcba1928b6929f99892a0bfe7208e48a3fd8999d30a2ef0f08a9f7c`.
The supervisor terminated only the owned process group after the configured
120-second Isaac shutdown grace period; all four ranks exited and GPUs 2–5
were released.

All three predeclared 128-episode evaluations completed with seed 42, full
resets, schema-v2 diagnostics, and verified checkpoint/config identity:

| Evaluation | Complete success |
|---|---:|
| step6050, attractor on | 116/128 (90.62%) |
| step6400, attractor on | 115/128 (89.84%) |
| step6400, attractor off | 1/128 (0.78%) |

The attractor-on candidate preserves task success and improves seven of eight
late arm/action metrics versus step6050. It nevertheless fails the absolute
Q4/Q2, slope, wrist/proximal, and action-RMS gates. More decisively, disabling
the attractor at the same step6400 checkpoint loses 89.06 percentage points of
complete-task success. V2 is therefore rejected: it improved behavior under
the training scaffold but did not learn attractor-independent post-open arm
recovery.

The immutable local evidence bundle is:

`../../eval_results/arm_reward_v2_final_step6400_20260822_1219/`

Its `PAIRED_COMPARISON.md` contains step6050→step6400 results and
`ATTRACTOR_ON_TO_OFF.md` contains the same-checkpoint attractor intervention.
The evaluator writes all requested episode and diagnostic artifacts before a
known Tensor-to-JSON exception in optional result serialization. Both
supervisors report `evaluator_exit=0`, exact 128-episode counts, and complete
diagnostic files; the exception did not invalidate the compared evidence but
remains an eval-pipeline cleanup issue.

## V3 fallback activation

The predeclared first fallback is now active on branch
`continuation/arm-reward-v3-20260822`. It changes only the harmful-action
component. The old aggregate binary remains a reward-independent comparison
metric, while the reward uses the mean per-joint outward raw-delta magnitude
weighted by the current normalized target displacement. New training metrics
log harmful joint fraction and outward mean/std/p95/max.

An isolated step6050 calibration probe measured the old binary signal at about
0.96–0.97, the new harmful-joint fraction at about 0.70–0.72, outward mean near
0.10–0.11, and outward p95 near 0.20–0.21. At weight `-0.15`, the new term
contributed about `-0.0029` per episode. The final weight `-1.4` was therefore
chosen to preserve the V2 initial reward budget; an independent calibrated
smoke measured `-0.0253` per episode without NaN, OOM, or rank failure.

All pose/progress/settled terms, Stage-5 attractor and alpha, reset ratios, PPO
settings, observations, action scale, completion semantics, and non-arm rewards
remain unchanged. The formal V3 causal window restarts from the identical
step6050 anchor with a one-time optimizer reset and targets step6400.
