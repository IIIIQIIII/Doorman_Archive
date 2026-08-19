# Official Code Evaluation Pipeline

This note describes the evaluation path implemented in the archived DoorMan code.

## 1. Entry point

The official evaluator is:

```text
gr00t/rl/eval_agent_trl.py
```

Its module docstring describes it as an evaluation script for trained RL agents, explicitly including **teacher or student** policies.

Typical use:

```bash
python gr00t/rl/eval_agent_trl.py +checkpoint=<path_to_checkpoint.pt>
```

## 2. It restores the training configuration

Given a checkpoint path, `eval_agent_trl.py` searches for `config.yaml` in the checkpoint directory (or its parent), loads that training config, applies any stored `eval_overrides`, then merges command-line overrides.

This matters because teacher and student have different observation/model structures. The evaluator is not a hard-coded single-network playback script; it reconstructs the policy from the saved experiment configuration.

## 3. Environment recreation

The script launches Isaac Sim/Isaac Lab, creates the configured DoorMan environment, and enables cameras when needed by the student vision policy or rendering options.

The same evaluation entry point therefore supports:

- privileged-state teacher checkpoints;
- RGB student checkpoints;
- headless evaluation;
- rendered evaluation;
- optional policy export.

## 4. Model reconstruction

For the newer actor/critic configuration path, the evaluator instantiates the actor from the experiment config. If the saved experiment uses DAgger, it can also instantiate the associated teacher actor reference model. A critic is instantiated when required by the config.

For DoorMan's HOMIE-based teacher stack, the configured trainer handles the lower-body HOMIE integration while restoring the DoorMan policy checkpoint.

## 5. Checkpoint loading

The trainer is constructed with:

```text
checkpoint=config.checkpoint
```

and restores the saved state before evaluation.

The evaluator therefore expects a normal DoorMan training checkpoint plus its experiment configuration context.

## 6. Evaluation loop

After setup, the script calls:

```python
trainer.eval()
```

The PPO/HOMIE trainer switches the environment and policy into evaluation mode, resets all environments, initializes rollout/evaluation tracking, and runs episodes until the configured number of completed episodes is reached.

The environment has a dedicated evaluation flag (`set_is_evaluating`) and metric tracking hooks.

## 7. Task completion semantics

DoorMan is a six-stage task:

```text
0  Walk to door
1  Pre-grasp
2  Grasp
3  Open
4  Swing
5  Through
```

In the archived task implementation, final completion is checked by `_stage_5_to_complete_condition`. The task marks completion after the robot has traversed sufficiently far through the doorway. In the current code this is implemented using the robot root x-position relative to the environment origin:

```python
(self.simulator.robot_root_states[:, 0] - self.env_origins[:, 0]) > 1.5
```

This internal simulator condition is the concrete success signal used by the staged task implementation. The paper states the corresponding benchmark semantics as reaching about 1 m beyond the door frame; see `paper_benchmark.md`.

## 8. Evaluation metrics

The trainer asks the environment for an aggregate summary:

```python
eval_dict = self.env.get_eval_metrics_summary()
eval_dict["completed_episodes"] = completed_episodes
```

and saves it as:

```text
metrics_eval.json
```

The base environment initializes goal-reaching evaluation state including `episode_goal_reached` and a `goal_reached_buffer`. Task-specific or inherited evaluation tracking is summarized through `get_eval_metrics_summary()`.

The repository also contains `ReadEvalLocomanipCallback`, which can read `metrics_eval.json` from evaluation directories and log those metrics to Weights & Biases.

## 9. Default episode counts and saving options

Standalone `config/base_eval.yaml` currently uses:

```yaml
num_eval_episodes: 150
save_videos: false
video_save_prob: 1.0
save_goal_reached_only: true
save_trajectories: false
num_save_episodes: 200
```

The generic `config/algo/eval/base.yaml` uses a 200-episode default and also exposes video/trajectory options. Its `eval_interval: 0` means periodic evaluation during training is disabled by default.

Therefore, when reproducing evaluation, always record the resolved config instead of assuming that the number of episodes is universally 150 or 200.

## 10. Rendering and trajectory saving

The evaluation config supports:

- `save_videos`;
- `video_save_prob`;
- `save_goal_reached_only`;
- `save_trajectories`;
- `num_save_episodes`.

Rendered results can be consumed by the repository's eval callback and logged to W&B.

## 11. ONNX export

The evaluator also doubles as the official policy-export path.

When:

```text
num_envs == 1
```

it exports the inference model into the experiment's `exported/` directory. The export function is selected according to policy type, including state policies and CNN/vision policies.

Typical command:

```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    num_envs=1
```

## 12. What this evaluator does not automatically reproduce

Running `eval_agent_trl.py` is sufficient for official **simulation checkpoint evaluation**, but it is not a one-command reproduction of every table/figure in the DoorMan paper.

For paper-level comparisons, one must additionally reproduce the specified door subsets, held-out appearances, trial counts, initialization perturbations, and—where applicable—real Unitree G1 experiments and human-teleoperation baselines.

That distinction is documented in `paper_benchmark.md`.
