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
