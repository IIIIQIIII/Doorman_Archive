# DoorMan Evaluation Notes

This folder summarizes the official DoorMan evaluation support preserved in this archive. It is documentation only; the archived `github/` and `arxiv/` contents are not modified.

## Bottom line

DoorMan **does provide an official evaluation path** for trained teacher and student checkpoints.

The main code entry point is:

```text
github/GR00T-VisualSim2Real/gr00t/rl/eval_agent_trl.py
```

The base evaluation config is:

```text
github/GR00T-VisualSim2Real/gr00t/rl/config/base_eval.yaml
```

The official README explicitly documents both Teacher Evaluation and Student Evaluation.

## Quick start

From inside `github/GR00T-VisualSim2Real`:

### Teacher

```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=<path_to_teacher_checkpoint.pt>
```

### Student

```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=<path_to_student_checkpoint.pt>
```

Both use the same evaluator. The script loads the checkpoint, looks for the associated training `config.yaml`, rebuilds the environment and policy, loads model weights, and runs evaluation rollouts.

## Default standalone eval config

`gr00t/rl/config/base_eval.yaml` currently specifies:

```yaml
checkpoint: ???

algo:
  config:
    eval:
      num_eval_episodes: 150
      save_videos: false
      video_save_prob: 1.0
      save_goal_reached_only: true
      save_trajectories: false
      num_save_episodes: 200

eval_base_dir: logs_eval
```

There is also a generic algorithm-level eval config under:

```text
gr00t/rl/config/algo/eval/base.yaml
```

where `eval_interval: 0` means periodic training-time evaluation is disabled by default.

## Output

The trainer obtains an environment-side evaluation summary via:

```python
eval_dict = self.env.get_eval_metrics_summary()
eval_dict["completed_episodes"] = completed_episodes
```

and writes:

```text
metrics_eval.json
```

Evaluation can also save videos/trajectories when the corresponding config flags are enabled.

## ONNX export

When evaluation is run with a single environment (`num_envs=1`), the evaluator can export the policy to ONNX. The official README documents evaluation as the supported export path.

## Important distinction

The open-source evaluator is an **Isaac Sim checkpoint evaluator**. It should not be confused with the full paper benchmark protocol.

The paper's benchmark additionally specifies:

- Push Lever, Pull Lever, and Push Bar door categories;
- robot initialized 1 m in front of the door;
- yaw perturbation uniformly sampled within ±0.3 rad;
- holdout visual textures in simulation;
- unseen real-world visuals;
- success when the robot traverses the doorway and reaches approximately 1 m beyond the frame;
- success rate and completion time as headline metrics.

See `paper_benchmark.md` for details.

## Files in this folder

- `README.md` — overview and quick-start eval usage.
- `code_eval_pipeline.md` — what the official evaluator actually does and what it outputs.
- `paper_benchmark.md` — evaluation protocol described in the DoorMan paper and how it differs from the code-level evaluator.
- `official_teacher_reproduction_eval_20260821.md` — the 2026-08-21 official-teacher checkpoint results, interpretation, exact 4090 protocol, video/sync procedure, and artifact map.
- `run_checkpoint_eval_4090.sh` — parameterized reproduction script for a 128-episode metrics run plus a separate 8-environment third-person viewer run.
- `post_open_quality_step5450_20260821/README.md` — metrics-only diagnosis of the step-5450 post-open crab gait and arm-rest failure, including exact reproduction and artifact locations.

## Current reproduction result

The latest checkpoint evaluated in the 2026-08-19 teacher reproduction is step 5450. Under the frozen seed-42, 128-full-reset protocol it completed 117/128 episodes (91.41%). Every trajectory that reached Stage 1 also reached Stage 5; the 11 failures remained in Stage 0. The immediately preceding step-5350 checkpoint completed 119/128 (92.97%).

See `official_teacher_reproduction_eval_20260821.md` before running or interpreting another checkpoint. In particular, the 8-environment viewer run is a separate qualitative run and must not replace the 128-episode quantitative result.

For behavior quality beyond binary task completion, see
`post_open_quality_step5450_20260821/README.md`. The diagnostic reproduces the
91.41% completion rate while showing systematic lateral traversal and a
wrist-dominated terminal arm-rest failure.

## Archived source map

Key official files referenced by these notes:

```text
github/GR00T-VisualSim2Real/README.md
github/GR00T-VisualSim2Real/gr00t/rl/eval_agent_trl.py
github/GR00T-VisualSim2Real/gr00t/rl/config/base_eval.yaml
github/GR00T-VisualSim2Real/gr00t/rl/config/algo/eval/base.yaml
github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer_homie_api.py
github/GR00T-VisualSim2Real/gr00t/rl/envs/door/door_open_homie.py
arxiv/2512.01061/tex/sec/4_experiment.tex
```
