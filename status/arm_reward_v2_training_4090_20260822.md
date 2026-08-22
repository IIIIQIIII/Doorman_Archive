# 4090 arm-reward-v2 training status — 2026-08-22

## Current state

- The previous `step5500_postopen_fix_v1` continuation was stopped cleanly.
- Its last baseline checkpoint is `model_step_006100.pt` (SHA256 `95e0d1e66cdd80c9f7afbe5b3ddbad353e30ddbae139ab3bc73c68a76f483351`).
- The new controlled branch starts from the fully diagnosed `model_step_006050.pt` anchor (SHA256 `2efa1b726fc96c4305eacee43749eb286d375cc28adbb1cf824ad4354b597837`), rather than inheriting the un-evaluated step6100 optimizer trajectory.
- 4090 code root: `/sda/mashijian/doorman_arm_reward_v2_20260822/code`
- Git branch/commit: `continuation/arm-reward-v2-20260822` / `e4cbca9`
- Production run: `/sda/mashijian/doorman_arm_reward_v2_20260822/runs/door_open_homie_arm_reward_v2`
- GPUs: physical GPU 2, 3, 4, 5; GPU 0 and 1 are not touched.
- Fixed training window: step6050 → step6400.

## Why V1 was a local optimum

The action is an accumulated target update. V1 simultaneously penalized raw arm corrections and asked the physical arm to return to rest. The policy could therefore reduce its penalty by issuing small corrections while leaving the accumulated target and physical arm displaced. This matches the step6050 evaluation: reward terms plateaued while both arms, especially the wrists and the right proximal chain, worsened after opening.

## V2 behavior objective and rewards

Recovery activates in Stage 5, after manipulation. Per side, normalized joint error is:

`0.5 * mean(all seven arm joints) + 0.5 * mean(worst three joints)`

The reward is split into four terms:

1. bilateral physical pose error (`-4.0`);
2. signed physical recovery progress (`+20.0`), where regressions are negative, preventing oscillation farming;
3. harmful raw action (`-0.15`) when it pushes the accumulated target farther from rest;
4. raw-delta settling penalty (`-1.0`) only after bilateral recovery is sustained.

Recovery uses hysteresis: both sides below `0.08` for `0.5 s` becomes settled; either side above `0.12` for `0.1 s` is a relapse. The existing Stage-5 rest attractor remains at alpha `0.04` for this first controlled branch; attractor-off evaluation remains mandatory before acceptance.

Staged reset ratios are fixed at `[0.30, 0.05, 0.05, 0.10, 0.20, 0.30]` to increase Stage-4/5 exposure without removing full-task coverage. The stage-state bank is populated online from real transitions and includes the accumulated delta-action state.

## Reward-independent training metrics

Training now logs `Env/arm_v2_*` metrics independently from reward:

- left/right total, proximal, and wrist normalized error;
- per-joint mean, p95, and max;
- raw delta RMS and near-limit fraction;
- accumulated effective-target error;
- harmful-action fraction, action sign reversal, and recovery efficiency;
- settled fraction, ever-settled fraction, and relapse count;
- fixed post-open time-bin exposure/error: 0–0.5 s, 0.5–1.5 s, 1.5–3 s, and >3 s;
- left/right door exposure proportion.

This makes it possible to reject a run where shaped reward improves but objective arm behavior does not.

## Verification completed

- Python compilation and Git whitespace checks passed.
- Four-rank 4090 DDP smoke completed learning iterations 6051–6053.
- All four new reward terms registered.
- The checkpoint loaded and optimizer/LR scheduler reset exactly once at the step6050 fork.
- `Env/arm_v2_*` metrics appeared in trainer logs.
- No NaN, reward-state, simulator-state, or gradient error occurred in the valid smoke.
- The two earlier smoke attempts with invalid small-run microbatch geometry were discarded; the valid smoke used 256 total environments, per-device batch 64, accumulation 1. Production retains the validated original geometry: 4096 total environments, per-device batch 256, accumulation 4.

The first real Stage-5 exposure at iteration 6056 exposed a unit bug before any new checkpoint was saved: the framework multiplies reward scales by `dt`, while the progress signal was already a per-step finite difference. This made progress approximately 50 times weaker than designed. The short 6051–6056 segment was discarded, the reward now returns `progress / dt`, and the controlled window restarts from the unchanged step6050 anchor. No reward weight or other scheme component changed.

## First post-fix online evidence

The corrected run reached real Stage-5 exposure again at iteration 6056. The progress contribution was now non-zero and comparable with the pose contribution (`+0.0126` versus `-0.0016` in the episode-normalized logger), validating the time-unit correction.

By iterations 6061–6063, full-task goal rate had recovered to approximately `0.89–0.91`, while the objective arm metrics still exposed the inherited local optimum: harmful-action fraction was approximately `0.96–0.97`, physical recovery progress remained negative, left wrist error was approximately `0.21–0.27`, right wrist error approximately `0.10–0.13`, and settled fraction approximately `0.12–0.15`. This is treated as the measured starting condition, not as an early stop signal. The reward/config remains frozen until the predefined step6100 gate.

## Stability and decision policy

- The supervisor owns only this production run and restarts it from the newest validated checkpoint.
- Optimizer state is reset only for the initial step6050 fork; a restart from a V2 checkpoint preserves optimizer state.
- After validating step6400, the supervisor allows 120 seconds for normal Isaac shutdown and then terminates only this run's process group if shutdown hangs; this prevents completed ranks from becoming GPU-holding orphans.
- A read-only monitor writes GPU, rank, checkpoint, and objective arm metrics every 15 minutes to `/sda/mashijian/doorman_arm_reward_v2_20260822/monitor_15m.log`.
- No reward/config changes are allowed inside the fixed window because of short-term metric noise. Exceptions are crash, NaN, invalid data/state restoration, or clear reward hacking (reward improves while objective arm metrics worsen).
- First decision gate: step6100. Formal paired evaluation remains required at later fixed checkpoints before declaring the arm problem solved.

## Step6100 gate protocol

The first gate uses `evaluation/run_arm_reward_v2_gate_eval_4090.sh`. It freezes
the step6050 anchor and step6100 candidate into immutable bundles, verifies the
embedded `global_step`, records SHA256 hashes, and then runs the same schema-v2
post-open diagnostic on both checkpoints concurrently on otherwise-idle physical
GPUs 6 and 7. Both sides use the V2 code, seed 42, 32 full-reset environments,
32 completed episodes, and identical evaluator overrides. This removes code,
seed, reset-distribution, and instrumentation differences from the comparison.

The gate is behavioral rather than reward-based. It compares:

- complete-task success, with a maximum tolerated paired drop of 5 percentage
  points before escalation;
- left/right total, proximal, and wrist normalized physical error across fixed
  post-open progress quartiles;
- per-episode error slopes, where non-positive is preferred and values above
  `0.005/s` remain unacceptable;
- late raw arm-action RMS, settled/relapse evidence from training telemetry, and
  whether progress reward improves while physical metrics worsen.

Step6100 is an early direction gate, not proof that the arm problem is solved.
No reward/config change is made merely because a noisy 32-episode point misses
an acceptance target. Clear reward hacking, invalid state restoration, NaN, or
a broad physical regression can trigger an exception; otherwise the controlled
window remains frozen and later fixed-checkpoint evaluation supplies the final
evidence.

## Step6100 gate result

The immutable paired diagnostic completed successfully for both checkpoints on
2026-08-22. Checkpoint identity, evaluator exit status, 32 completed episodes,
schema-v2 output completeness, and non-empty arm time series all passed. The
local artifact bundle is:

`../../eval_results/arm_reward_v2_gate_step6100_20260822_1027/`

Both checkpoints completed 26/32 tasks (81.25%), so the candidate incurred no
paired task-success loss. Relative to step6050, step6100 changed the late Q4
metrics as follows:

| Objective metric | Step6050 | Step6100 | Change |
|---|---:|---:|---:|
| Left total arm error | 0.1997 | 0.1975 | -0.0022 |
| Right total arm error | 0.1456 | 0.1427 | -0.0029 |
| Left proximal error | 0.0522 | 0.0616 | +0.0094 |
| Right proximal error | 0.1249 | 0.1244 | -0.0005 |
| Left wrist error | 0.3965 | 0.3788 | -0.0177 |
| Right wrist error | 0.1731 | 0.1671 | -0.0060 |
| Left raw action RMS | 0.6056 | 0.5916 | -0.0140 |
| Right raw action RMS | 0.6179 | 0.6174 | -0.0006 |

Seven of eight late arm/action metrics improved, and all six physical-error
slopes moved in the favorable direction. The largest useful early effect is the
left-wrist reduction; the left proximal absolute Q4 error is the one regression.
The changes are still small: total-arm Q4/Q2 ratios remain 2.55 (left) and 4.70
(right), all-error slopes do not yet meet the `0.005/s` target, and late action
RMS remains about 0.59–0.62 rather than below 0.25.

At training iteration 6100, reward-independent telemetry still showed harmful
actions at 0.9614, physical recovery progress at -0.0415/s, settled fraction at
0.1244, ever-settled fraction at 0.5630, and relapse count at 0.4386. The signed
recovery reward contribution was also negative (-0.1605), agreeing with the
objective regression signal rather than hiding it. Therefore the gate shows a
small real improvement without evidence of reward hacking, but it does not show
that the local optimum has been escaped.

Decision: keep the reward, reset distribution, PPO configuration, and fixed
step6050→6400 window unchanged. Continue 15-minute read-only monitoring and use
later fixed checkpoints for stronger paired evidence; do not declare the arm
problem solved from step6100.

The validated launcher is parameterized by `CANDIDATE_STEP`. The next
predeclared observation gate is step6200 with the same 32-episode paired
protocol; step6400 remains the fixed-window acceptance checkpoint. No extra
checkpoint-driven scheme changes are introduced between those gates.

## Step6200 observation gate

The step6200 paired diagnostic completed with the same integrity conditions and
again produced 26/32 complete tasks, equal to both step6050 and step6100. The
local artifact bundle is:

`../../eval_results/arm_reward_v2_gate_step6200_20260822_1100/`

Against the step6050 anchor, only three of eight late arm/action metrics and
four of six error slopes improved. More importantly, the direct step6100 to
step6200 comparison shows that the early improvement partly reversed:

| Objective Q4 metric | Step6100 | Step6200 | Change |
|---|---:|---:|---:|
| Left total arm error | 0.1975 | 0.2008 | +0.0033 |
| Right total arm error | 0.1427 | 0.1466 | +0.0039 |
| Left proximal error | 0.0616 | 0.0579 | -0.0037 |
| Right proximal error | 0.1244 | 0.1192 | -0.0053 |
| Left wrist error | 0.3788 | 0.3913 | +0.0125 |
| Right wrist error | 0.1671 | 0.1833 | +0.0162 |
| Left raw action RMS | 0.5916 | 0.6087 | +0.0171 |
| Right raw action RMS | 0.6174 | 0.5879 | -0.0295 |

Only three of eight late metrics and two of six slopes improved from step6100.
At training iteration 6200, harmful-action fraction remained 0.9630, physical
recovery progress remained negative at -0.0404/s, settled fraction fell to
0.0843, and ever-settled fraction fell to 0.4015. The arm pose, signed recovery,
and harmful-action reward contributions also became slightly more negative
(-0.1309, -0.1621, and -0.0284), so the reward is reflecting the deterioration
rather than being hacked. Total task success remains intact, but the policy is
still near the arm local optimum and the step6100 directional gain is not yet
sustained.

Decision: this is not a crash, task collapse, invalid-state event, or clear arm
reward-hacking exception. Preserve the precommitted reward/config through the
step6400 boundary, continue 15-minute read-only monitoring, and make the final
accept/reject/redesign decision from the step6400 paired result rather than
chasing the step6200 fluctuation with an unplanned mid-window change.

### Step6200 attractor-off diagnostic

The same frozen step6200 checkpoint was evaluated for 32 full-reset episodes
with only `env.config.stage5_arm_rest_attractor=false`. Evaluator completion and
the Hydra override were both verified. Complete-task success dropped from 26/32
(81.25%) with the trained configuration to 17/32 (53.12%) without the attractor,
a 28.12 percentage-point loss. Late left-arm error rose from 0.2008 to 0.3061,
left proximal error from 0.0579 to 0.1088, and left wrist error from 0.3913 to
0.5690. The direct report is:

`../../eval_results/arm_reward_v2_gate_step6200_20260822_1100/STEP6200_ATTRACTOR_ON_TO_OFF.md`

This proves that the current checkpoint has not learned an attractor-independent
arm recovery strategy; the environment attractor is materially supporting both
left-arm behavior and task completion. It is not accepted as a solution. The
precommitted run still continues to step6400 so the fixed-window causal test is
completed, but step6400 must pass an attractor-off evaluation before V2 can be
accepted. The reusable runner is
`evaluation/run_post_open_diagnostics_v2_4090.sh`; it records the explicit
attractor override in Hydra output.

The step6400 acceptance rule and the single-change fallback experiment were
frozen before the final result in
`evaluation/arm_reward_v2_step6400_acceptance_and_fallback.md`. The fallback is
conditional and has not modified the running V2 experiment.

## Final step6400 decision

The fixed V2 run completed and stopped at step6400. The final 128-episode
paired eval preserved task success (116/128 at step6050 versus 115/128 at
step6400) and improved seven of eight late arm/action metrics, but it failed the
absolute physical gates. The same step6400 policy achieved only 1/128 complete
tasks with the Stage-5 arm attractor disabled, compared with 115/128 when it was
enabled. This 89.06 percentage-point intervention loss proves severe
attractor dependence, so V2 is rejected rather than continued.

The final local artifact bundle is
`../../eval_results/arm_reward_v2_final_step6400_20260822_1219/`. The
predeclared single-component V3 fallback has been activated; its live status is
recorded in `status/arm_reward_v3_training_4090_20260822.md`.

This file was added outside `github/` and `arxiv/`; neither directory was modified.
