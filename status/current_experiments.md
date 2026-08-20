# Current Experiment Status

Status as of **2026-08-19 Asia/Shanghai**. Values below are operational reports;
they are not additions to the immutable official GitHub/arXiv archives.

## Experiment matrix

| Track | Compute | Latest known state | Interpretation |
|---|---|---|---|
| Official-reward baseline | A100 server, 4-GPU DDP | Step-2250 checkpoint confirmed; iteration 2249 healthy with `average_stage=3.6759`. | Continue unchanged under monitoring. No eval or ablation is currently launched. |
| `grasp_gate` | 4090 server | Reached step 900 and was evaluated; stopped/eliminated by the coordinator at **21:50**. | **Rejected for proxy hacking:** valid contact increased, but hinge progress and transitions did not. |
| `grasp_torque` (corrected) | 4090 server | Reached step 800; bridge-ready fraction 0 and bridge reward 0.0001. | Paused for evaluation: bridge opportunity/reward is too sparse to infer benefit. |
| `torque_bridge_only_c2` | 4090 server | Step 800: 118 Stage-3 entries, 0 Stage-4 transitions; stopped/eliminated. | Rejected: lowering required contacts to 2 did not produce hinge progress. |
| `p5` | 4090 server | Step 800, SHA256 `56f7e744...fd2a2`, evaluated and stopped/eliminated. | Bridge signal became nonzero but produced no useful hinge progress or Stage-4 transition. Do not continue. |
| `stage4_release` | 4090 server | Step 1850 immutable checkpoint SHA256 `25afbc...d742`; evaluated and stopped/eliminated. | Rejected: Stage 3 -> 4 fell to 13.22% and Stage 4 -> 5 remained 0. Do not continue. |
| `grasp_persistence` | 4090 server | Loaded immutable step 750 at **21:42**; hard cutoff set for **21:54**. | Short bounded probe only; outcome not yet recorded. |
| `force_gate` | 4090 server | Step 800, SHA256 prefix `b40e...`, evaluated by isolated rerun and stopped/eliminated. | **Rejected for proxy farming:** contact/torque increased without hinge progress or transitions. |
| `handle_gate` | 4090 server | Step 800, SHA256 prefix `9dda...`, evaluated by isolated rerun and stopped/eliminated. | **Rejected as contact proxy:** contact increased without hinge progress or transitions. |

The two 4090 tracks are intended as controlled ablations. `grasp_gate` gates
Stage-3 shaping by grasp quality. `grasp_torque` adds the same gating plus a
persistence-sensitive hinge-axis bridge. Their implementation and restored
runtime state must be audited before their learning curves are interpreted.

## `grasp_torque` correction chain

- The original `grasp_torque` process was stopped by itself during
  **20:21-20:22 on 2026-08-19**, before checkpoint load or a training iteration.
  It produced no checkpoint eligible for comparison.
- Its retained log was renamed
  `grasp_torque_tainted_pre_resetfix_20260819_2022.log`. It is preserved for
  audit only and must not be merged into the corrected run's curve.
- A launcher containing the reset-callback fix was synchronized to the 4090
  server and passed `py_compile`.
- Launcher SHA256:
  `0dab6a01f368972bb2fa3606b29ab2a5a8f7092992a603d417a6f4abf8bbd845`.
- Corrected `grasp_torque` was launched from the immutable step-750 checkpoint;
  reported parent PID was **4107778**.
- At **20:23 on 2026-08-19**, review found that this first corrected version
  still used `-2` leaky accumulation and therefore did not implement strict
  continuous persistence. It was stopped by itself before checkpoint load or a
  training iteration. Its audit-only log is
  `grasp_torque_pre_strict_persistence_20260819_2024.log`.
- The second correction implements the strict state machine: invalid samples or
  samples outside Stage 3 set persistence immediately to 0; valid samples add
  1 up to a cap of 25; reset `env_ids` are explicitly set to 0.
- The bridge was idempotently added to the official
  `reward_penalty_reward_names`, and five telemetry items were added for runtime
  validation.
- Seven tests, `py_compile`, and JSON validation passed. Launcher SHA256:
  `dbff03144e6fb145c5d015aab443017d728cd31b53b8e66e50b555310fb72296`.
- The second correction was synchronized to the 4090 server; remote SHA256 and
  `py_compile` independently matched/passed. Strict corrected `grasp_torque`
  was restarted from immutable step 750 with parent PID **4118507**.
- `grasp_gate` and the A100 baseline remained untouched throughout both
  corrections.

## 4096-environment OOM and resource correction

- At **20:23 on 2026-08-19**, the original 4096-environment `grasp_gate`
  configuration reached its first backward pass and failed with CUDA OOM. The
  allocation request was 564 MiB with only 561.88 MiB free.
- It completed no iteration and produced no step-751 checkpoint. Residual DDP
  processes were stopped. The audit-only log is
  `grasp_gate_4096env_oom_pre_step751_20260819_2027.log`.
- Because strict `grasp_torque` used the same resource shape, it was
  preemptively stopped before checkpoint load/iteration. Its audit-only log is
  `grasp_torque_strict_4096env_preempted_for_oom_20260819_2027.log`.
- The P0 resource correction retains 4096 environments, 64 rollout steps, and
  four logical minibatches. It changes the physical per-device microbatch to
  512 and uses gradient accumulation of 2.
- The trainer patch aggregates KL across accumulated microbatches and performs
  learning-rate adjustment, gradient clipping, optimizer step, and zeroing only
  at synchronization boundaries.
- Fifteen tests and static validation passed. SHA256 identities are:
  - trainer patch: `646715be95c50ee23b5e823fe21a5f7328db414a92d3614f15464cbcc2ada6e4`;
  - patch application script: `1463774416b31036a52f2d0bd718a8993c201fc70b49af7532c2ce1bbf897e99`;
  - ablation launcher: `41c5d75cd6df05393d127cf4c2d84cad0874ce8910aafd26a23bc7339ceda423`;
  - reward launcher: `dbff03144e6fb145c5d015aab443017d728cd31b53b8e66e50b555310fb72296`;
  - patched trainer on the remote host:
    `3146c134c511d2ce3dc6ffe833285bb4ae4bd5af4c9565a898455f1b16d53ac8`.
- At **20:35**, only `grasp_gate` was restarted, with parent PID **4139530**.
  `grasp_torque` will launch only after `grasp_gate` proves a successful
  750 -> 751 transition.
- A100 baseline was untouched. Reward configuration was not changed by this
  resource correction.

### Resource go-gate result

- All four `grasp_gate` ranks loaded the immutable step-750 checkpoint and
  completed iterations 751 through 754.
- Resolved runtime configuration is 4096 environments, 64 rollout steps, four
  logical minibatches, per-device microbatch 512, and gradient accumulation 2.
- Iteration wall time improved from 20.79 s at iteration 751 to 19.57 s at
  iteration 754.
- GPUs 0-3 used approximately 15.2-16.3 GB each, leaving about 8 GB of headroom
  relative to the previous OOM condition.
- No OOM, NCCL error, NaN, or `RuntimeError` was reported in the gate window.
- The 750 -> 751 go-gate therefore passed. As planned, strict `grasp_torque`
  was then launched at approximately 20:52 on GPUs 4-7 with parent PID
  **4178814**. Its remote patched-trainer SHA256 is
  `3146c134c511d2ce3dc6ffe833285bb4ae4bd5af4c9565a898455f1b16d53ac8`.
- Reward configuration is frozen through the formal step-900 decision point.

## Unified evaluation readiness

- A unified evaluation scaffold has been completed locally and passes 21
  tests.
- GPU smoke validation and construction/verification of the common reset/state
  bank remain pending.
- Until both are complete, the scaffold is not yet qualified to produce the
  formal cross-track step-900 comparison.

## Last established diagnostic anchor

The archived step-750 diagnostic remains the comparison anchor:

- 118 of 128 evaluated first episodes entered Stage 3;
- 0 Stage-3 episodes transitioned to Stage 4;
- maximum hinge angle was 0.935 degrees versus the 10-degree threshold;
- longest valid four-contact grasp plus opening torque above 1 Nm was 0.38 s.

Source: `teacher_training/diagnostics/stage3_gap_step0750_20260819/`.

## Official baseline step-1000 milestone

The official-reward A100 baseline step-1000 checkpoint was evaluated on idle
GPU 3 with the same 128-environment Stage-3 diagnostic used at step 750.

- Checkpoint SHA256:
  `57833ed2316bd82a392d81ed3234f2d987b6ba161b56f7ecaee9b5656f718422`.
- 120 episodes entered Stage 3; 0 transitioned to Stage 4.
- Maximum hinge angle was 0.0276987 rad (1.587 degrees), versus 0.935 degrees
  at step 750 and the 10-degree Stage-4 threshold.
- Valid handle-contact fraction was 0.1796, versus 0.1072 at step 750.
- Mean opening torque was 2.545 Nm, versus 2.122 Nm at step 750.
- Interpretation: contact, torque, and maximum hinge displacement improved, but
  the official baseline still produced no Stage 3 -> 4 transition. The baseline
  continues training unchanged.
- Runtime result path:
  `.../diagnostics/stage3_gap_step1000_20260819` on the A100 host.

## Step-900 ablation milestone

- `grasp_gate` step 900: 109 Stage-3 entries, 0 Stage-4 transitions, hinge
  maximum 0.903 degrees, valid-contact fraction 24.67%, opening-torque p95
  17.194 Nm, and maximum 81.555 Nm.
- Despite higher valid contact and large torque tails, hinge maximum was below
  the official baseline's step-1000 1.587 degrees and no transition occurred.
  The coordinator classified this as proxy hacking and eliminated the run at
  21:50.
- Strict `grasp_torque` was stopped after a pre-step-751 performance stall.
- `grasp_persistence` loaded immutable step 750 at 21:42 with a hard 21:54
  cutoff.
- Minimal `force_gate` (force-only gating; official handle reward retained)
  launched at 21:51 with PID 113931.

## Isolated-rerun milestone: force gate 800 / baseline 1250

Only the explicitly per-process-environment rerun directories are valid for
this milestone. The first evaluation outputs inherited a stale
`DOORMAN_DIAG_DIR`, were path-contaminated, and have been isolated under
`contaminated_*`; they are excluded from all formal decisions.

- `force_gate` step 800, checkpoint SHA256 prefix `b40e...`: 107 Stage-3
  entries, 0 Stage-4 transitions, hinge maximum 0.0129879 rad (0.744 degrees),
  valid-contact fraction 25.198%, torque mean/p95/max
  3.186/15.472/57.964 Nm. This is proxy farming and the run is eliminated.
- Official baseline checkpoint internal `global_step=1250`, SHA256 prefix
  `8f74...`: 121 Stage-3 entries, 0 Stage-4 transitions, hinge maximum
  0.143132 rad (8.201 degrees), valid-contact fraction 3.800%, and torque
  mean/p95/max 3.553/22.612/111.989 Nm.
- Interpretation: the baseline still has no transition, but its physical hinge
  displacement is now genuinely close to the 10-degree Stage-4 threshold;
  unlike the rejected gated runs, progress is not inferred from contact or
  torque proxies alone.
- At that milestone, `handle_gate` was the active follow-up; its later step-800
  result is recorded below.

## Contact-gate falsification / baseline near-threshold milestone

- `handle_gate` step 800, checkpoint SHA256 prefix `9dda...`: 118 Stage-3
  entries, 0 Stage-4 transitions, hinge maximum 0.0131003 rad (0.751 degrees),
  valid-contact fraction 21.985%, and torque p95 14.977 Nm. It was rejected as
  contact-proxy optimization.
- Together, the force-only and handle-contact gates are falsified: both raise
  contact/force proxies without producing useful hinge displacement or a
  Stage-4 transition.
- Official baseline isolated diagnostics remained close to but below the
  10-degree threshold: step 1300 had 0/120 transitions and 9.23-degree maximum;
  step 1350 had 0/121 and 8.92 degrees; step 1400 had 0/120 and 9.06 degrees.
  Training hinge reward at step 1350 was 0.1263.
- Corrected `grasp_torque` launched at 23:15 on healthy GPUs 0-3 with PID
  301093. The previous GPU 4-7 run directory was archived.
- Path-contaminated eval outputs remain isolated and are not used for any of
  these decisions.

## First official Stage 3 -> 4 breakthrough

- Official baseline step 1450, checkpoint SHA256 prefix `3ac0897e...`: 120
  Stage-3 entries and the first observed Stage 3 -> 4 transition, 1/120
  (0.833%).
- Hinge mean/p95/max were 0.069717/0.132897/0.173749 rad, or
  3.995/7.615/9.955 degrees. Valid-contact fraction was 2.626%; opening-torque
  p95 was 23.430 Nm.
- This falsifies the hypothesis that the official baseline is permanently
  trapped in the Stage-3 local optimum. The event rate is still too low to call
  the gap solved or the breakthrough stable, so the baseline continues
  unchanged to step 1500.
- Corrected `grasp_torque` on healthy GPUs 0-3 loaded immutable step 750 and
  completed iteration 751 in 22.89 s, using approximately 15-16 GB per GPU.
  Its first-iteration bridge reward was 0, consistent with the strict gating
  design. It continues to step 800 before evaluation.

## Baseline 1550 / torque-800 milestone

- Official baseline step 1500: 121 Stage-3 entries, 0 Stage-4 transitions;
  hinge mean/p95/max 4.296/7.714/9.919 degrees.
- Official baseline step 1550, checkpoint SHA256 prefix `9bf8...`: 3/121
  Stage 3 -> 4 transitions (2.479%); hinge mean/p95/max
  4.891/8.603/9.994 degrees.
- Compared with step 1450's 0.833% and step 1500's 0%, the transition rate is
  improving but remains checkpoint-variable and unstable. Baseline continues
  unchanged to step 1600, with later Stage 4-5 progression now also requiring
  observation.
- Corrected `grasp_torque` reached step 800 with bridge-ready fraction 0 and
  bridge reward 0.0001. It was paused for evaluation because the bridge signal
  is too sparse.
- `torque_contact2` changes only the bridge contact requirement from count 4 to
  count 2. PID 382489 is initializing on a healthy lane.
- The first torque eval failed because the reward hook was absent. A combined
  launcher with explicit mode and an independent result directory was used for
  the rerun. Failed-output artifacts are excluded from decisions.

## Baseline 1650 / bridge-contact2 decision window

- Enhanced official baseline step-1650 eval: 121 Stage-3 entries. The old
  counting convention recorded 3 Stage 3 -> 4 transitions (2.48%); the
  all-event convention recorded 4. No Stage 4 -> 5 event occurred.
- Hinge mean/p95/max were 5.262/8.659/9.999 degrees; valid-contact fraction was
  2.326%. The policy remains extremely close to the Stage-4 threshold, but
  downstream Stage-5 progression is absent.
- `torque_bridge_only_c2` step 800: 118 Stage-3 entries, 0 Stage-4 transitions;
  hinge mean/p95/max 0.226/0.491/0.908 degrees and valid-contact fraction
  10.418%. It is eliminated.
- `p5` is the only new experiment authorized in this decision window. It changes
  persistence 25 -> 5 and was launched on 4090 GPUs 0-3 under PID 505744.
- No additional experimental branch is introduced in this window.

## Official baseline step-1700 breakthrough milestone

- Enhanced step-1700 checkpoint SHA256: `9b37ca1f...e282`.
- Old counting convention: 121 Stage-3 entries and 10 Stage 3 -> 4
  transitions (8.264%). All-event convention: 12 Stage 3 -> 4 events. There
  were still 0 Stage 4 -> 5 events.
- Hinge mean/p95/max were 5.407/8.851/9.999 degrees; valid-contact fraction was
  2.089%; opening-torque p95/max were 25.52/141.99 Nm.
- The official baseline now shows a significant Stage-3 breakthrough and
  continues without reward/configuration changes. The next fixed evaluation is
  step 1800; no 50-step decision switching is authorized.
- At this milestone the independent `p5` window remained open; its completed
  result and closure are recorded below.

## p5 result / Stage-3 reward-ablation closure

- `p5` step-800 checkpoint SHA256: `56f7e744...fd2a2`.
- Training bridge-ready fraction was 0.59% and bridge reward was 0.0137.
- Enhanced A100 eval: 118 Stage-3 entries, 0 Stage-4 transitions; hinge
  mean/p95/max 0.167/0.428/0.884 degrees; valid-contact fraction 18.232%;
  opening-torque p95/max 14.41/197.37 Nm.
- Lowering persistence 25 -> 5 made the bridge less sparse, but did not produce
  Stage-3 physical progress. `p5` is eliminated and will not continue.
- Across the completed controlled runs, no contact-gating or torque-bridge
  ablation outperformed the official baseline. No further Stage-3 reward branch
  is authorized. The official baseline continues unchanged, and the active
  investigation focus moves to Stage 4 -> 5 progression.

## Step-1800 Stage-3 gap closure / persistent Stage-4 gap

- Official baseline step-1800 checkpoint SHA256: `d7568c08...3b81`.
- Enhanced eval: 121 Stage-3 entries, 60 Stage 3 -> 4 transitions (49.587%) by
  the primary convention, and 65 Stage 3 -> 4 all events. Stage 4 -> 5 remained
  0.
- Hinge mean/p95/max were 5.596/9.581/9.999 degrees. Training
  `average_stage` was approximately 3.04.
- The Stage 3 -> 4 gap is now materially solved by the unchanged official
  baseline. The persistent blocker is Stage 4 -> 5.
- The official baseline continues unchanged. The only authorized Stage-4
  single-variable direction disables the handle-retention reward group only in
  Stage 4; no additional Stage-3 reward work is reopened.

### `stage4_release` launch identity

- Active on 4090 GPUs 0-3 with effective PID **600949**.
- Source checkpoint: official step 1800, SHA256 `d7568c...3b81`.
- Reward launcher SHA256: `365ad4...9747`; launch script SHA256:
  `745434...8527`.
- The only reward change sets the four Stage-4 handle-retention-group scales to
  0. Stages 0-3 and 5, plus every other reward term, remain official.
- An earlier launch under PID 600674 exited immediately with code 2 because the
  checkpoint filename was missing. It never entered training; the filename was
  corrected before the effective launch.

## `stage4_release` step-1850 decision

- Immutable checkpoint SHA256: `25afbc...d742`.
- Eval recorded 16/121 Stage 3 -> 4 transitions (13.22%) under the old
  convention and 18 all events; Stage 4 -> 5 remained 0.
- This is a marked regression from the official step-1800 baseline's 49.59%
  Stage 3 -> 4 rate, without downstream benefit. `stage4_release` is eliminated
  and will not continue.
- Official baseline is currently around step 1966. Training `average_stage` is
  approximately 3.20, while `dont_push_handle` and `target_root` rewards are
  increasing. It continues unchanged to the next fixed enhanced eval at step
  2000.

## Official step-2000 checkpoint / health milestone

- Checkpoint payload confirms `state.global_step=2000` and 32,768,000 episodes.
- `last.pt` SHA256:
  `c44bb99a9d0916e88b03eca6b1df317174e39b40877b4c047f407b84bac991e5`;
  modification time was 02:52 on 2026-08-20.
- Iteration 2001: `average_stage=3.2521`; goal reached, last-goal reached, and
  complete were all 0. Reward components were handle 1.1835, force 0.5790,
  hinge 0.5764, `dont_push_handle` 0.1592, and `target_root` 0.2159.
- Iteration time was 71.52 s.
- Parent process and all four ranks were healthy. No OOM, NCCL error, NaN,
  traceback, or other reported training failure was present.
- This is a monitoring/checkpoint confirmation only. No enhanced eval or new
  ablation has been launched.

## Official step-2050 checkpoint / health milestone

- Checkpoint payload confirms `state.global_step=2050` and 33,587,200 episodes.
- Checkpoint SHA256:
  `ef21cc5699e85d2925f4c5cfb092935f260d0cb734aa4565739b44f5a6979e79`;
  modification time was 03:59:53 on 2026-08-20.
- Iteration 2049: `average_stage=3.3157`; goal reached and complete were 0;
  hinge reward was 0.6050, `dont_push_handle` was 0.1850, and `target_root` was
  0.2566.
- Iteration time was 80.25 s. Processes were healthy, with no reported runtime
  error.
- Monitor-only mode remains in force; no eval or ablation was launched.

## Official step-2100 checkpoint / health milestone

- Checkpoint payload confirms `global_step=2100` and
  `tot_timesteps=2202009600`.
- Checkpoint SHA256:
  `ba4fa7b91391fa2bc5b694f7419198f23f23a43176d6fcbc5428afe666aba296`;
  modification time was 05:07:36 on 2026-08-20.
- Iteration 2102: `average_stage=3.4177`; goal reached, last-goal reached, and
  complete were all 0. Reward components were handle 0.8591, force 0.4153,
  hinge 0.6438, `dont_push_handle` 0.2441, and `target_root` 0.3167.
- Iteration time was 78.87 s. Processes were healthy, with no reported error.
- Monitor-only mode remains in force; no eval or ablation was launched.

## Official step-2150 checkpoint / health milestone

- Checkpoint payload confirms `global_step=2150`, 35,225,600 episodes, and
  `tot_timesteps=2254438400`.
- Checkpoint SHA256:
  `683fa2b93d59d54dd6da3add9ce25879a7cb03b9879a514494a4b7bd5164f04f`;
  modification time was 06:15 on 2026-08-20.
- Iteration 2149: `average_stage=3.5348`; goal reached and complete were 0.
  Reward components were handle 0.7344, force 0.3550, hinge 0.7573,
  `dont_push_handle` 0.3687, and `target_root` 0.4291.
- Iteration time was 74.85 s. Processes were healthy, with no reported error.
- Monitor-only mode remains in force; no eval or ablation was launched.

## Official step-2200 checkpoint / health milestone

- Checkpoint payload confirms `global_step=2200`, 36,044,800 episodes, and
  `tot_timesteps=2306867200`.
- Checkpoint SHA256:
  `71f8e2967cf99282acd1e8312e5d6e6504b57c1a09373d7af25d6be538e4d000`;
  modification time was 07:22 on 2026-08-20.
- Iteration 2198: `average_stage=3.6170`; goal reached and complete were 0.
  Reward components were handle 0.6363, force 0.3052, hinge 0.8609,
  `dont_push_handle` 0.4814, and `target_root` 0.4823.
- Iteration time was 80.56 s. Processes were healthy, with no reported error.
- Monitor-only mode remains in force; no eval or ablation was launched.

## Official step-2250 checkpoint / health milestone

- Checkpoint payload confirms `global_step=2250`, 36,864,000 episodes, and
  `tot_timesteps=2359296000`.
- Checkpoint SHA256:
  `d76b6ee1883955e4f389bd8549593ddfb8549b5fb7168e167a23b1e6eb35e1fa`;
  modification time was 08:30 on 2026-08-20.
- Iteration 2249: `average_stage=3.6759`; goal reached and complete were 0.
  Reward components were handle 0.5259, force 0.2484, hinge 1.0368,
  `dont_push_handle` 0.6133, and `target_root` 0.5643.
- Iteration time was 80.11 s. Processes were healthy, with no reported error.
- Monitor-only mode remains in force; no eval or ablation was launched.

## Next judgment point

1. After strict corrected startup is confirmed, freeze its configuration. Do
   not switch implementation mid-run except for a crash, demonstrated data
   contamination, or a safety/P0 fault.
2. Complete the remaining validation for the state/reset risks in
   `risk_and_decision_log.md` before
   attributing any ablation curve movement to reward design.
3. Summarize routine observations by checkpoint window rather than reacting to
   individual iterations. The formal decision points are **step 900** and
   **step 1000**.
4. At comparable step-900 and step-1000 checkpoints, run the exact same
   Stage-3 diagnostic for baseline, `grasp_gate`, and `grasp_torque`.
5. Compare at minimum: Stage 3 -> 4 transition rate, hinge maximum/distribution,
   valid-grasp occupancy, grasp+torque overlap duration, positive torque
   persistence, and complete-task indicators.
6. Inspect reward components for hacking: handle/force/bridge growth without
   corresponding hinge displacement or transitions is a stop/review signal.
7. Keep the baseline running unless a safety/runtime fault is found; promotion
   of an ablation requires repeatable physical progress, not reward increase
   alone.
8. Before long-horizon promotion of `grasp_torque`, verify through runtime
   telemetry that persistence resets strictly and that the bridge coefficient
   follows the official curriculum as implemented.
9. Record ordinary improvement ideas in the backlog for the next controlled
   run; they do not trigger an immediate restart.
10. The 750 -> 751 `grasp_gate` resource gate passed; it does not replace the
    formal reward decisions at step 900 and step 1000.

## Update provenance

- A100 iteration/checkpoint state and 4090 startup state: coordinator-provided
  runtime report on 2026-08-19.
- `grasp_torque` stop, retained-log identity, launcher hash, corrected PID, and
  untouched-track statement: action/coordinator runtime report on 2026-08-19.
- Strict-persistence finding, second-correction semantics, validation results,
  SHA256, remote verification, and PID: action/coordinator report on
  2026-08-19.
- OOM measurements, quarantined logs, microbatch/accumulation design, validation
  count, sequential gate launch, and untouched A100 statement:
  action/coordinator report on 2026-08-19.
- Gate iterations/timing, GPU headroom, error scan, strict-torque launch, reward
  freeze, and eval-scaffold readiness: monitor/coordinator report on
  2026-08-19.
- Official baseline step-1000 checkpoint identity and matched diagnostic:
  evaluation/coordinator report on 2026-08-19.
- Step-900 gate diagnostic, elimination decision, stopped strict-torque state,
  persistence cutoff, and force-gate launch: coordinator report on 2026-08-19.
- Isolated force-gate-800 and baseline-1250 reruns, contaminated-output
  quarantine, and handle-gate health: evaluation/coordinator report on
  2026-08-19.
- Handle-gate-800 result, baseline 1300-1400 diagnostic sequence, corrected
  torque launch, and prior-run archival: evaluation/coordinator report on
  2026-08-19.
- Baseline step-1450 first transition and corrected-torque iteration-751 health:
  evaluation/monitor/coordinator report on 2026-08-19.
- Baseline steps 1500/1550, torque-800 sparsity, torque-contact2 launch, and
  reward-hook eval rerun: evaluation/coordinator report on 2026-08-20.
- Baseline-1650 enhanced events, contact2 elimination, and p5 launch:
  evaluation/coordinator report on 2026-08-20.
- Baseline-1700 enhanced breakthrough and step-1800 evaluation schedule:
  evaluation/coordinator report on 2026-08-20.
- p5 step-800 enhanced result and Stage-3 ablation closure decision:
  evaluation/coordinator report on 2026-08-20.
- Baseline step-1800 Stage-3 closure and Stage-4 single-variable direction:
  evaluation/coordinator report on 2026-08-20.
- `stage4_release` hashes, effective PID, single-variable scope, and failed
  pre-training launch: action/coordinator report on 2026-08-20.
- `stage4_release` step-1850 result and official step-2000 evaluation schedule:
  evaluation/monitor/coordinator report on 2026-08-20.
- Official step-2000 checkpoint identity and iteration-2001 health:
  monitor/coordinator report on 2026-08-20.
- Official step-2050 checkpoint identity and iteration-2049 health:
  monitor/coordinator report on 2026-08-20.
- Official step-2100 checkpoint identity and iteration-2102 health:
  monitor/coordinator report on 2026-08-20.
- Official step-2150 checkpoint identity and iteration-2149 health:
  monitor/coordinator report on 2026-08-20.
- Official step-2200 checkpoint identity and iteration-2198 health:
  monitor/coordinator report on 2026-08-20.
- Official step-2250 checkpoint identity and iteration-2249 health:
  monitor/coordinator report on 2026-08-20.
- Step-750 measurements: archived diagnostic cited above.
