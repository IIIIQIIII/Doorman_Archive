# Official Teacher Reproduction Evaluation — 2026-08-21

This note records the official-teacher evaluation protocol and results used for the local 2026-08-19 reproduction. It is operational documentation, not a modification of the archived `github/` or `arxiv/` sources.

## 1. Scope and success signal

The evaluated policy is the privileged-state recurrent PPO teacher built from official commit:

```text
016c70c1e4e76f521963c36691ee69a6ab3ac9cd
```

The quantitative headline is `episode_goal_reached`, aggregated by `metrics_eval.json`. DoorMan stages are:

```text
0 Walk to door → 1 Pre-grasp → 2 Grasp → 3 Open → 4 Swing → 5 Through
```

Stage reach is diagnostic. Complete-task success is the Stage-5 completion predicate, not reward magnitude, hinge angle alone, or merely reaching Stage 4.

## 2. Frozen comparison protocol

Every checkpoint in the comparison table used:

- official environment, actor, reward, and termination definitions;
- full episode reset (`env.config.enable_staged_reset=false`);
- seed 42;
- 128 environments and exactly one completed episode per environment;
- no W&B logging and no automatic latest-checkpoint substitution;
- no reward ablation or runtime reward hook;
- the same LAFAN-G1 reset source;
- a separate 8-environment external-viewer run for qualitative video.

The 8-video run is **not a paired subset of the 128 quantitative episodes**. Changing `num_envs` changes the rollout batch. Use it to inspect behavior, never to replace the 128-episode success rate.

## 3. Checkpoint results

| Step | Success | Rate | Interpretation |
|---:|---:|---:|---|
| 3100 | 117/128 | 91.41% | Earlier high-performing reference |
| 3850 | 113/128 | 88.28% | Temporary lower checkpoint result |
| 4050 | 117/128 | 91.41% | Returned to the step-3100 rate |
| 4500 | 118/128 | 92.19% | High-performance plateau |
| 5350 | 119/128 | 92.97% | Best observed point in this fixed comparison |
| 5450 | 117/128 | 91.41% | Latest archived evaluation at update time |

The differences are small relative to binomial uncertainty. They support a stable roughly 91–93% region, not a statistically decisive ranking of adjacent checkpoints.

## 4. Latest result: step 5450

Checkpoint:

```text
global_step: 5450
total_timesteps: 5,714,739,200
checkpoint_sha256: 0fa4c1269e4aea4fec7a196f143ce7ff8a224c4041460b2141126876292d7969
config_sha256: 3837d2f473d47a5285da3cc0eab954ed1fa27b8e28b24a432f0e484aec9c3729
```

Quantitative result:

- complete success: **117/128 = 91.41%**;
- Wilson 95% interval: **85.27%–95.13%**;
- Stage 1/2/3/4/5 reach: **117/128 at every stage**;
- final maximum-stage counts: Stage 0 = 11, Stage 5 = 117;
- mean/max episode hinge maximum: 2.3184 / 2.6180 rad;
- mean/max episode root-x maximum: 1.7895 / 2.0617 m;
- mean episode length: 483.51 steps;
- termination flags: low height 6, overspeed 6, with possible overlap;
- no stage-overtime, global-timeout, or far-from-door termination.

All policies that crossed into Stage 1 subsequently completed Stage 5. Thus the two-success decrease from step 5350 is attributable to additional Stage-0 failures, not a recurrence of the historical Stage-3→4 or Stage-4→5 gap.

The separate viewer run completed 7/8 tasks. Its eight MP4 files are external third-person scene views, H.264, 256×256, 50 FPS, 582 frames, and 11.64 seconds each.

## 5. Artifact locations

Local project artifacts sit outside this archive so the archive remains compact:

```text
../eval_results/official_teacher_step3100_20260820/
../eval_results/official_teacher_step3850_20260821/
../eval_results/official_teacher_step4050_20260821/
../eval_results/official_teacher_step4500_20260821/
../eval_results/official_teacher_step5350_20260821/
../eval_results/official_teacher_step5450_20260821/
```

From this file's `evaluation/` directory, those paths are `../../eval_results/...`.

The step-5450 server-side source is:

```text
/sda/mashijian/doorman_official_eval_20260821/step5450
```

Its local directory contains:

```text
README.md
metrics/metrics128.json
metrics/viewer8.json
third_person_videos/*.mp4
preview_viewer4_t8s.png
```

## 6. Freeze the latest checkpoint before evaluation

Never evaluate a training run's mutable `last.pt` in place. First inspect and hash it on the training server, then copy it with its matching `config.yaml` into an immutable evaluation bundle.

Example inspection in the trusted training environment:

```bash
RUN=/absolute/path/to/training/run

python - <<'PY'
import torch

p = "/absolute/path/to/training/run/last.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
s = d["state"]
print(s.global_step, s.episode, s.tot_timesteps)
PY

sha256sum "$RUN/last.pt" "$RUN/config.yaml"
```

Copy both files into a checkpoint-specific directory, then compute SHA256 again on the evaluation server. The hashes must match before launch.

## 7. Run on the 4090 server

The reusable script is:

```text
evaluation/run_checkpoint_eval_4090.sh
```

Example for step 5450:

```bash
CHECKPOINT=/sda/mashijian/doorman_official_eval_20260821/step5450/checkpoint_bundle/last.pt \
OUTPUT_DIR=/sda/mashijian/doorman_official_eval_20260821/step5450 \
CHECKPOINT_LABEL=step5450 \
METRICS_GPU=2 \
VIEWER_GPU=3 \
bash evaluation/run_checkpoint_eval_4090.sh
```

The script runs two independent evaluations concurrently:

1. `metrics128`: 128 full-reset episodes without rendering;
2. `viewer8`: 8 full-reset episodes with external scene rendering.

For third-person output, the important combination is:

```text
simulator.config.render_results=true
env.config.save_rendering_dir=<viewer_directory>
algo.config.eval.save_videos=false
```

`render_results` writes the external scene-camera videos. `save_videos=false` deliberately disables the evaluator's ego-view video recorder.

Before launch, verify the selected GPUs are free:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Do not take GPU 0/1 merely because utilization is momentarily 0%; in the recorded runs those devices retained about 18 GB each for other users' tasks. GPU ownership requires checking both processes and memory.

## 8. Completion and integrity gates

A run is complete only when its log contains both:

```text
Evaluation completed - N episodes finished
Finished evaluation
```

Also require:

- `metrics128/metrics_eval.json` exists and reports `collected_episodes == 128`;
- `viewer8/metrics_eval.json` exists and reports `collected_episodes == 8`;
- eight nontrivial MP4 files exist under `viewer8/viewer/`;
- logs have no OOM, runtime, NCCL, or traceback failure before completion;
- the loaded-checkpoint log line reports the intended step.

Do not infer success from mean reward. Read `goal_reached_rate`, `episode_goal_reached`, `stage_reach_rates`, `episode_max_stage`, and termination flags.

## 9. Sync to the local project

Recommended local structure:

```text
eval_results/official_teacher_stepNNNN_YYYYMMDD/
├── README.md
├── metrics/
│   ├── metrics128.json
│   └── viewer8.json
├── third_person_videos/
│   └── *.mp4
└── preview_viewerX_tYs.png
```

Example transfer:

```bash
LOCAL=/Users/Admin/Projects/msj-wujie/msj-doorman/eval_results/official_teacher_step5450_20260821
REMOTE=/sda/mashijian/doorman_official_eval_20260821/step5450

mkdir -p "$LOCAL/metrics" "$LOCAL/third_person_videos"
scp -P 7600 'mashijian@111.2.199.31:'"$REMOTE"'/viewer8/viewer/*.mp4' \
    "$LOCAL/third_person_videos/"
scp -P 7600 "mashijian@111.2.199.31:$REMOTE/metrics128/metrics_eval.json" \
    "$LOCAL/metrics/metrics128.json"
scp -P 7600 "mashijian@111.2.199.31:$REMOTE/viewer8/metrics_eval.json" \
    "$LOCAL/metrics/viewer8.json"
```

Never put a password in this archive or in a shell script.

## 10. Local video and hash verification

Record remote SHA256 values before transfer, then compare them locally:

```bash
sha256sum "$REMOTE/metrics128/metrics_eval.json" \
    "$REMOTE/viewer8/metrics_eval.json" \
    "$REMOTE/viewer8/viewer"/*.mp4

shasum -a 256 "$LOCAL/metrics"/*.json "$LOCAL/third_person_videos"/*.mp4
```

Check every MP4 with `ffprobe`:

```bash
for f in "$LOCAL"/third_person_videos/*.mp4; do
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
        -show_entries format=duration \
        -of default=noprint_wrappers=1 "$f"
done
```

Finally extract and inspect at least one late-episode frame:

```bash
ffmpeg -y -ss 8 -i "$LOCAL/third_person_videos/<viewer>.mp4" \
    -frames:v 1 -update 1 "$LOCAL/preview.png"
```

The accepted frame must visibly show the robot, door/frame, and surrounding scene from an external camera. A valid MP4 container alone does not prove the required third-person viewpoint.

## 11. Interpretation rules

- Use the 128-run for checkpoint success rates; treat viewer8 as qualitative.
- Compare checkpoints only when seed, resets, episode count, code/config, and success predicate match.
- Report stage reach and termination causes beside aggregate success.
- If Stage 3 reach is high but Stage 4 is zero, investigate grasp/force/hinge persistence and reward exploitation.
- If Stage 4 is reached but Stage 5 is zero, inspect the real Stage-4→5 predicates rather than proxy reward totals.
- If every Stage-1 entrant reaches Stage 5, later-stage reward changes are not justified by that rollout set; inspect Stage-0 initialization/failure causes instead.
- Adjacent 128-episode differences of one or two successes are not decisive by themselves. Use confidence intervals and repeated seeds before selecting a production checkpoint.
