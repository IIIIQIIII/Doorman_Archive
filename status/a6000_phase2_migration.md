# DoorMan Phase 2 A6000 Migration Status

Last updated: 2026-08-21 (Asia/Shanghai)

## Objective

Move the reproducible DoorMan runtime from the A100 server to the RTX A6000
server for Phase 2 RGB vision-student training. The A100 server remains the
source of truth for the official privileged-state teacher reproduction. The
A6000 copy will be validated before any Phase 2 training is launched.

Phase 2 is online DAgger distillation: a frozen privileged teacher labels the
states visited by an RGB + proprioception student. It is not a continuation of
the Phase 1 PPO optimization.

## Protected archive scope

The following immutable source archives must not be changed by this migration:

- `github/`
- `arxiv/`

Migration status and operational evidence are recorded only under `status/`.

## Server paths

### A100 source

- Project: `/data1/mashijian/coordex/doorman`
- Python environment: `/data1/mashijian/coordex/envs/doorman`
- Active official reproduction:
  `/data1/mashijian/coordex/doorman/reproductions/official_teacher_016c70c_20260819`

### A6000 destination

- Migration root: `/hdd0/mashijian/doorman`
- Project copy: `/hdd0/mashijian/doorman/project`
- Python environment copy: `/hdd0/mashijian/doorman/envs/doorman`
- Cache root: `/hdd0/mashijian/doorman/cache`
- Output root: `/hdd0/mashijian/doorman/outputs`
- Migration evidence: `/hdd0/mashijian/doorman/migration`
- Frozen teacher package:
  `/hdd0/mashijian/doorman/teacher/official_step4050_eval_20260821`

### 4090 evaluation source

- Host: `4090-06` (8 x RTX 4090 24GB)
- Frozen evaluation package:
  `/sda/mashijian/doorman_official_eval_20260821/step4050`
- Evaluation code worktree:
  `/sda/mashijian/doorman_ablation_20260819/grasp_torque/code`
- Base commit: `016c70c1e4e76f521963c36691ee69a6ab3ac9cd`
- Worktree note: the evaluation checkout had a tracked modification in
  `gr00t/rl/trl/trainer/ppo_trainer_homie_api.py`; the frozen package includes
  the resolved Hydra configuration and logs used for provenance.

The destination deliberately does not use the A6000 home directory, root
filesystem, or `/ssd0`. The A6000 container may bind-mount these host paths to
the original A100 absolute prefixes when compatibility requires it.

## Runtime baseline

### A100

- Ubuntu 22.04
- Python 3.11.15
- PyTorch 2.7.0+cu128
- torchvision 0.22.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab 2.3-series source/runtime
- 8 x NVIDIA A100-SXM4-80GB

### A6000

- Ubuntu 24.04 host
- 8 x NVIDIA RTX A6000 48GB, compute capability 8.6
- NVIDIA driver 550.90.12
- Docker access available to `mashijian`
- Existing images include Isaac Sim 5.1.0 and Isaac Lab 2.3.0/2.3.2

## Migration state

1. Destination directories were created under `/hdd0/mashijian/doorman`.
2. Direct A6000-to-A100 SSH connectivity was verified.
3. The A100 DoorMan project was copied to the A6000 project path: 12,072
   files and 57,693,792,290 source bytes were observed by `rsync`.
4. The A100 Python environment was copied to the A6000 environment path:
   212,357 files and 23,864,593,934 source bytes were observed by `rsync`.
5. Follow-up dry runs converged with zero project differences (excluding the
   live A100 `launcher.log`) and zero environment differences.
6. Stable teacher/HOMIE artifact hashes match on A100 and A6000.
7. A6000's native `nvcr.io/nvidia/isaac-lab:2.3.2` image is the selected
   simulator base. A small symlink-only Python compatibility overlay at
   `/hdd0/mashijian/doorman/envs/native232-overlay` supplies packages present
   in the migrated A100 environment but absent from that image.
8. Isaac 5.1 visual assets were warmed into the data-disk cache under
   `/hdd0/mashijian/doorman/cache`.
9. A one-environment, one-timestep, one-batch RGB DAgger smoke run completed
   and wrote a valid step-1 checkpoint.
10. The evaluated step-4050 teacher package was copied from the 4090 server to
    the A6000 data disk. Source, relay and destination SHA256 values match.
11. The step-4050 teacher deserialized in the selected A6000 container and
    completed a fresh RGB DAgger one-step collection/learning/save smoke run.
12. No A100 training process was stopped or modified.
13. The A100 DoorMan checkout's isolated-rank DDP launcher and trainer patch
    were reviewed and ported to the A6000 compatibility copy. Each process
    sees one physical GPU as `cuda:0`, while global `RANK` and `WORLD_SIZE`
    retain normal DDP semantics.
14. A two-rank, one-container Phase 2 gate completed successfully after
    disabling NCCL peer-to-peer and InfiniBand transports. Both ranks entered
    learning and wrote a valid step-1 checkpoint.
15. The formal single-job, seven-rank Phase 2 run was launched on A6000 GPUs
    0--6 and verified through synchronized learning iteration 3.

## Source identities captured before validation

- Clean official source commit:
  `016c70c1e4e76f521963c36691ee69a6ab3ac9cd`
- Official worktree state: detached HEAD, no tracked or untracked changes.
- `model_step_004000.pt` SHA256:
  `0480eff32c06684df17866732cce9538461480235554ccdaf1ee1e7d8a54342b`
- `last.pt` SHA256 captured during the live run:
  `7ed09d2d9d0af90682dba3b4465f9ae83055a33d92641d6a771090cba6e401fc`
  (`last.pt` is mutable while training continues and must be re-identified
  before use.)
- `model_walk.pt` SHA256:
  `2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e`
- `model_stand.pt` SHA256:
  `ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054`

### Selected frozen Phase 2 teacher

- Training identity: global step 4050, epoch 16200, episode 66,355,200.
- `checkpoint_bundle/last.pt` and
  `checkpoint_bundle/model_step_004050.pt` are byte-identical.
- Teacher checkpoint SHA256:
  `9f2457298e9e3f259bc5c8b8489dff2e2eb7c74ccdfce71910cf505d5d25d472`
- Training config SHA256:
  `3837d2f473d47a5285da3cc0eab954ed1fa27b8e28b24a432f0e484aec9c3729`
- 128-episode metrics SHA256:
  `631f8b3787383d61fb9b6599a148fa6d264ab5c5c212dd1840ff94c14f56c272`
- Frozen package size on A6000: 65 MiB, including checkpoint copies,
  training config, resolved Hydra configs, launcher/eval logs, metrics and
  eight viewer videos.

## Validation gates

Current results:

1. PASS — source/destination transfer convergence and artifact identity.
2. PASS — Python 3.11.15, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128,
   CUDA tensor execution and Isaac imports on RTX A6000.
3. PASS — selected teacher step 4050 and both HOMIE checkpoints deserialize.
4. NOT RUN — non-rendered teacher evaluation control on the A6000 copy.
5. PASS — one-environment RGB camera initialized at 216 x 384, attached to
   `d435_link`, and produced the vision observation used by a rollout.
6. PASS — the vision student, recurrent teacher and HOMIE models initialized;
   a new step-4050-teacher DAgger collection/learning iteration completed with
   `dagger_bc_loss=1.370680` and `obj_pred_loss=0.069117`.
7. PASS — single-GPU scaling reached 192 RGB environments with a measured
   peak near 40.5 GiB during the capacity probe.
8. PASS — two-rank synchronized RGB DDP gate, including first gradient
   synchronization and step-1 checkpoint creation.
9. PASS — formal seven-rank job entered synchronized learning iterations 1,
   2 and 3 on all ranks without NCCL timeout, CUDA OOM or process failure.

The migration is qualified for the current production Phase 2 launch. The
formal run is active and must continue to be monitored through its first
scheduled checkpoint and subsequent long-duration operation.

## Multi-GPU implementation and validation

The official Phase 2 code contains distributed support through Accelerate and
PyTorch DDP, although the archived usage examples document only a one-process
launch. The working A100 DoorMan checkout adds the operational detail required
by Isaac Sim: one process per physical GPU, isolated CUDA visibility per
process, `LOCAL_RANK=0`, global `RANK/WORLD_SIZE`, renderer multi-GPU disabled,
and an explicit renderer GPU. The A100 server also contains a successful
historical eight-rank student run, demonstrating that this is synchronized
data-parallel training rather than multiple independent experiments.

The first long A6000 DDP gate reached the initial NCCL scalar all-reduce but
timed out after 1,800 seconds using the default peer-to-peer transport. A
minimal two-GPU NCCL test and then the complete two-rank RGB Phase 2 gate both
passed with:

- `NCCL_P2P_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

This transport setting is used by the formal run. Earlier independent
single-GPU exploratory processes were stopped and are not part of the formal
training strategy. The supported direction is one gradient-synchronized DDP
job only.

## Active formal Phase 2 run

- Container: `doorman-phase2-step4050-ddp7`
- Output directory:
  `/hdd0/mashijian/doorman/outputs/phase2_step4050_ddp7_e192_s8_20260821_072029`
- GPUs: physical A6000 0--6; GPU 7 deliberately excluded.
- World size: 7 synchronized ranks.
- Environments: 192 per rank, 1,344 global RGB environments.
- Rollout length: 8 steps per environment.
- Global samples per learning iteration: 10,752.
- Teacher: frozen evaluated step-4050 checkpoint (91.40625% over 128
  evaluation episodes).
- Initial verification: all seven ranks completed RGB scene/sensor setup and
  reported learning iterations 1, 2 and 3 with global total timesteps 10,752,
  21,504 and 32,256 respectively.
- Observed training memory after DDP entry: approximately 20--22 GiB per GPU
  in the first iteration; utilization became active across GPUs 0--6.
- Error scan through iteration 3: no Traceback, CUDA OOM, NCCL watchdog
  timeout, segmentation fault or rank exit.
- Checkpoint cadence: every 500 iterations. No formal checkpoint is expected
  during the initial three-iteration health gate.

## A6000 smoke evidence

- Native simulator image: `nvcr.io/nvidia/isaac-lab:2.3.2`
- Physical validation GPU: GPU 3 (exposed as container `cuda:0`)
- Resolved Phase 2 config:
  `/hdd0/mashijian/doorman/outputs/phase2_config_resolved.yaml`
- Successful smoke directory:
  `/hdd0/mashijian/doorman/outputs/phase2_rgb_smoke_online_20260821_035157`
- Step-1 checkpoint:
  `run/model_step_000001.pt` (152,128,019 bytes)
- Step-1 checkpoint SHA256:
  `306fc72ba38de3ec52747f37a4aacc468870b768191fc3857f44781e954204e1`
- Last checkpoint: `run/last.pt` (152,103,838 bytes)
- Last checkpoint SHA256:
  `e95339503021bb32de4f6806fbac0a03cbcffe1f37bf0002bc21ba9b2302f29a`
- Training result: one episode, one timestep, one iteration; collection
  2.600 seconds, learning 2.191 seconds.
- The validation deliberately used an untrained ResNet (`pretrained=false`)
  to prove wiring without downloading model weights. This checkpoint is a
  smoke artifact and must not be used as a Phase 2 initialization.
- Reusable validation wrapper:
  `/hdd0/mashijian/doorman/migration/run_phase2_rgb_smoke.sh`
- The wrapper now defaults to the frozen step-4050 teacher, mounts the teacher
  directory read-only, accepts the Isaac Sim EULA non-interactively and uses
  an explicit Bash entrypoint so the image cannot swallow the train command.
- Step-4050-teacher smoke directory:
  `/hdd0/mashijian/doorman/outputs/phase2_rgb_smoke_20260821_050359`
- Step-4050-teacher smoke checkpoint SHA256:
  `8ed8fea8f66c8fcdeb6b2453a5d273758b634e3e58bd606c531089b0ddea71ea`
- Post-validation footprint: project 54 GiB, environments 23 GiB, cache
  approximately 2.1 GiB, outputs 292 MiB. `/hdd0` retained approximately
  887 GiB free.

## Compatibility findings

- The copied A100 environment itself runs CUDA on A6000, but its general
  `SimulationApp` links depend on the A100 user's external Omniverse cache.
  This is a source-runtime characteristic, not an rsync loss.
- The native Isaac Lab 2.3.2 image starts a complete `SimulationApp` and is a
  cleaner A6000 base than repairing those external links.
- Native image gaps were supplied without altering DoorMan source: TRL,
  Accelerate, Loguru, Datasets, Plotly and ONNX Runtime are exposed through
  the compatibility overlay from the migrated environment.
- The original configuration's dynamic DomeLight scripting hook uses an API
  that is not registered under Isaac Sim 5.1. Setting
  `simulator.config.randomize_dome_light=false` avoids that compatibility
  error while preserving RGB rendering. A proper 5.1 implementation is still
  needed before re-enabling dynamic DomeLight randomization.
- First use requires NVIDIA Isaac 5.1 ground/material assets. They are now
  cached on `/hdd0`; no cache was placed in `/home` or the root filesystem.

## Step-4050 teacher evaluation evidence

The previously open teacher-identity item is resolved. The principal
evaluation used `checkpoint_bundle/last.pt`, logged `Loaded checkpoint from
step 4050`, and completed 128 episodes with seed 42 and 128 environments.
Rendering, W&B and staged reset were disabled. The result was 117/128 goals,
or `goal_reached_rate=0.9140625` (91.40625%). Average maximum stage was
4.59375, average episode length was 496.265625, and average maximum door-hinge
angle was 2.224535 radians.

The package also contains a rendered eight-episode viewer sample with a 75%
goal rate and eight MP4 files. That small visual sample is supporting evidence;
the 128-episode result is the headline teacher metric.

## Safety constraints

- Do not start a second copy of the active A100 teacher training job.
- Do not write Isaac/Omniverse caches to the A6000 home or root filesystem.
- Do not scale directly to the archived 4096-environment RGB configuration on
  48GB GPUs; use staged smoke tests and measured scaling.
- Do not alter the teacher, HOMIE, observation or action definitions merely to
  make the first A6000 smoke test pass. Compatibility changes must be isolated
  and recorded.
