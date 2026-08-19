# DoorMan Teacher Training Recipe

This note reconstructs the teacher-training recipe from the archived DoorMan code and paper sources.

## 1. Training role of the teacher

DoorMan uses a three-stage teacher/student/bootstrap pipeline:

1. train a privileged-state teacher with PPO;
2. distill that teacher into an RGB student with DAgger;
3. refine the student with RL/GRPO under partial observability.

The teacher is therefore the **phase-1 simulation oracle**. It is not the final deployed policy.

## 2. Locomotion foundation: pretrained HOMIE

The teacher experiment references two existing locomotion checkpoints:

```yaml
homie_walk_model_path: ./models/model_walk.pt
homie_stand_model_path: ./models/model_stand.pt
opt_homie: False
```

The PPO trainer loads both models and, when `opt_homie=False`, disables gradients and puts them in eval mode.

This is the crucial reason the DoorMan teacher should not be described as learning full-body locomotion from scratch. PPO learns the door-specific task policy while HOMIE remains the lower-body locomotion foundation.

### Runtime behavior

The task policy emits a locomotion command. The trainer evaluates both HOMIE walk and stand models and switches according to command magnitude. The selected HOMIE output supplies lower-body actions.

Conceptually:

```text
privileged task state
        |
        v
DoorMan recurrent actor
        |
        +--> 3-D locomotion command --> frozen HOMIE walk/stand --> lower-body actions
        |
        +--> upper-body actions
        |
        +--> finger primitives
```

## 3. Task-level teacher action space

The archived teacher experiment contains:

```yaml
homie_command_dim: 3
non_homie_command_actions_dim: 16
```

with the comment:

```text
14 upper body dofs + 2 finger primitives
```

Therefore the trainable task actor produces 19 dimensions before HOMIE supplies lower-body control.

The environment also uses delta-action processing for these task-level dimensions.

## 4. Privileged teacher observation space

The archived `actor_obs` contains:

```text
dof_pos
relative_to_door
dof_vel
actions
projected_gravity
door_dof_pos
base_lin_vel
base_ang_vel
hand_force
stage
privileged_door_info
delta_actions
hand_handle_transform
unwarped_actions
b_homie_commands
```

The critic additionally receives task-progress and timing signals such as:

```text
transition
complete
time_in_stage
actual_time_in_stage
total_time
```

The paper describes privileged information including ground-truth robot/hand-to-door transforms, contact wrenches, and root linear velocity.

This means the teacher directly knows simulator state that the deployed RGB student must infer or compensate for.

## 5. Teacher network architecture

The archived teacher experiment uses a recurrent actor and recurrent critic.

### Actor

```yaml
_target_: RecurrentActor
rnn_type: lstm
rnn_hidden_dim: 256
rnn_num_layers: 2
MLP hidden_dims: [512, 256, 128]
activation: SiLU
running_mean_std: True
```

### Critic

```yaml
_target_: RecurrentCritic
rnn_type: lstm
rnn_hidden_dim: 256
rnn_num_layers: 2
MLP hidden_dims: [512, 256, 128]
activation: SiLU
running_mean_std: True
```

The paper states the teacher is trained with standard PPO; the archive implements this through the TRL-based PPO trainer adapted to humanoid environments.

## 6. Representative PPO hyperparameters

From the archived teacher experiment:

```yaml
num_envs: 4096
num_steps_per_env: 64
num_learning_epochs: 5
num_mini_batches: 4
actor_learning_rate: 1.0e-4
critic_learning_rate: 1.0e-4
entropy_coef: 0.01
desired_kl: 0.005
init_noise_std: 0.8
max_noise_std: 0.8
gamma: 0.9975
lam: 0.985
```

The repository README gives an example invocation with 1024 environments and command-line overrides, so environment count and some rollout parameters are intended to be tunable rather than immutable constants.

## 7. Door task structure

The teacher is trained on a six-stage task:

```text
0 Walk to door
1 Pre-grasp
2 Grasp
3 Open
4 Swing
5 Pass through door
```

Each stage has dedicated dense shaping rewards plus always-on safety/regularization penalties.

This decomposition injects task structure that makes the long-horizon manipulation problem much easier for on-policy RL to explore.

## 8. Staged-reset exploration

The environment caches simulator states from progressed stages and can reset an episode from those states instead of always starting at stage 0.

Archived configuration:

```yaml
enable_staged_reset: True
staged_reset_ratios: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
staged_reset_max_samples_per_stage: 200
```

The paper explains the mechanism as a way to reweight the state occupancy toward later task stages. Its ablation reports much faster exploration with a sufficiently large reset buffer and failure to progress reliably without the buffer.

## 9. Simulation/task initialization

The door environment is trained in Isaac Sim/Isaac Lab. The environment configuration also references LAFAN-G1 through `reset_from_dataset`.

This should be interpreted carefully:

- LAFAN-G1 is used for robot reset initialization;
- it is not evidence that the DoorMan teacher is behavior-cloned from demonstrations;
- the task teacher itself is trained with PPO.

## 10. Door variation

The paper describes procedural door generation and broad physical randomization, including panel dimensions, handle placement/type, opening direction/handedness, mass, hinge dynamics, and handle dynamics.

The teacher's key learning signal is still state-based control. The large photorealistic RGB randomization becomes especially important during student training because the student consumes camera observations.

## 11. How the teacher becomes a teacher

After PPO training, a teacher checkpoint is loaded into the DAgger configuration as `teacher_actor`.

The archived student configuration contains a concrete historical path:

```text
logs_rl/g1_open_door_homie/
wbmanip/door_open_homie_pregrasp-20251014_204826/
model_step_020450.pt
```

That checkpoint is then used as the state-based reference policy supervising the RGB student.

## 12. Minimal reconstruction command

The repository README presents the teacher training entry point as:

```bash
python gr00t/rl/train_agent_trl.py \
  +exp=wbmanip/door_open_homie_lstm \
  ++num_envs=1024 \
  ++algo.config.entropy_coef=0.001 \
  ++algo.config.num_steps_per_env=32 \
  ++env.config.delta_action_scale=0.3
```

This is best treated as the public reconstruction recipe, not necessarily a bit-for-bit record of every override used by the historical teacher checkpoint referenced by the DAgger config.

## Source paths

- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_lstm.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/obs/wbmanip/door_open_homie.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/env/door_open_homie.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/algo/ppo.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py`
- `github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer_homie_api.py`
- `arxiv/2512.01061/tex/sec/3_method.tex`
