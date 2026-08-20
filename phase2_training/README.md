# DoorMan Phase 2 Training Notes

> Scope: this directory documents **Phase 2: Distill onto vision student with DAgger** for DoorMan.
>
> The existing `github/` and `arxiv/` directories are intentionally left untouched. All notes here are derived from the archived DoorMan code/configuration and the archived arXiv source already present in this repository.

## 1. Executive summary

DoorMan Phase 2 is **interactive teacher-to-student policy distillation with DAgger**, not another PPO training stage.

The Phase-1 teacher is loaded from a checkpoint and frozen. During simulation, the vision student observes deployable inputs (RGB + non-privileged proprioception) and produces the high-level DoorMan action. At the same simulator state, the frozen teacher receives privileged observations and produces the target action. The student is updated with a supervised L2/MSE behavior-cloning loss against that teacher action.

The key difference from ordinary offline behavioral cloning is that supervision is collected on the **student's visited state distribution**. This is the DAgger mechanism that reduces covariate shift.

Conceptually:

```text
simulator state s_t
      |
      +--> privileged teacher_obs --> frozen teacher --> a_teacher (label)
      |
      +--> RGB + actor_obs ---------> student --------> a_student (executed)
                                                    |
                                                    +--> environment step

training loss: MSE(a_student, a_teacher)
```

In the archived DoorMan experiment configuration, the student rollout is effectively student-controlled: `rollout_with_teacher_num_steps: 0`, and no teacher-rollout enforcement is enabled in the experiment YAML.

---

## 2. Main source files

These are the most important files to read for Phase 2:

### Experiment configuration

- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml`
  - DoorMan Phase-2 experiment composition
  - teacher checkpoint path
  - number of environments / rollout settings
  - ResNet18 + LSTM student architecture
  - HOMIE configuration
  - RGB camera settings

### Generic DAgger configuration

- `github/GR00T-VisualSim2Real/gr00t/rl/config/algo/dagger_vision_distributed.yaml`
  - `use_dagger: True`
  - `dagger_bc_loss_type: l2`
  - `dagger_bc_loss_coef: 1.0`

### Observation definitions

- `github/GR00T-VisualSim2Real/gr00t/rl/config/obs/wbmanip/door_open_homie_dagger.yaml`
  - exact `actor_obs`
  - exact `vision_obs`
  - exact privileged `teacher_obs`
  - HOMIE observations

### Distillation trainer

- `github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer.py`
  - teacher checkpoint loading
  - rollout storage for `gt_actions`
  - recurrent trajectory handling
  - DAgger BC loss

### DoorMan + HOMIE distillation trainer

- `github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py`
  - frozen HOMIE walk/stand integration
  - teacher action labeling
  - construction of full robot action
  - student-controlled rollout
  - DoorMan-specific DAgger loss slicing

### Vision recurrent student

- `github/GR00T-VisualSim2Real/gr00t/rl/trl/modules/vision_actor_critic_modules_obj_pred_recurrent.py`
- `github/GR00T-VisualSim2Real/gr00t/rl/trl/modules/vision_actor_critic_modules_recurrent.py`
  - ResNet visual feature extraction
  - concatenation with proprioception
  - recurrent memory
  - action prediction

### Neural-network building blocks

- `github/GR00T-VisualSim2Real/gr00t/rl/agents/modules/modules.py`
  - torchvision ResNet construction
  - pretrained backbone
  - trainable/frozen behavior
  - output projection

### Image randomization

- `github/GR00T-VisualSim2Real/gr00t/rl/config/domain_rand/domain_rand_visual_ImageRand.yaml`

### Paper method section

- `arxiv/2512.01061/tex/sec/3_method.tex`
  - teacher-student setup
  - privileged teacher definition
  - student RGB/proprioception definition
  - DAgger motivation
  - Phase-3 partial-observability motivation

### Paper appendix

- `arxiv/2512.01061/tex/sec/supp.tex`
  - real-world D435i RGB-only deployment
  - physical/visual randomization details

---

## 3. Phase 2 prerequisites

Phase 2 assumes three previously available components:

1. **A trained Phase-1 DoorMan teacher**
2. **A pretrained HOMIE walking policy** (`models/model_walk.pt`)
3. **A pretrained HOMIE standing policy** (`models/model_stand.pt`)

The archived experiment points to a teacher checkpoint through:

```yaml
teacher_actor_path: logs_rl/g1_open_door_homie/wbmanip/door_open_homie_pregrasp-20251014_204826/model_step_020450.pt

algo:
  config:
    network_load_dict:
      teacher_actor:
        path: ${teacher_actor_path}
```

The distillation trainer loads the teacher checkpoint and places the reference model in evaluation mode. The teacher is therefore an online labeling oracle, not an optimized network during Phase 2.

HOMIE is also frozen when:

```yaml
opt_homie: False
```

This is the default DoorMan Phase-2 configuration.

---

## 4. Teacher observation versus student observation

The central idea is **asymmetric observations**.

### 4.1 Student inputs

The student receives an RGB image plus non-privileged proprioception.

`vision_obs`:

```yaml
vision_obs:
  - rgb_image
```

`actor_obs`:

```yaml
actor_obs:
  - base_ang_vel
  - projected_gravity
  - dof_pos_non_finger
  - dof_vel_non_finger
  - actions
  - delta_actions
  - b_homie_commands
  - complete
```

Therefore, the student should be described as **RGB + deployable proprioception**, rather than strictly pixel-only.

### 4.2 Teacher privileged inputs

The teacher receives much richer simulator state:

```yaml
teacher_obs:
  - dof_pos
  - relative_to_door
  - dof_vel
  - actions
  - projected_gravity
  - door_dof_pos
  - base_lin_vel
  - base_ang_vel
  - hand_force
  - stage
  - privileged_door_info
  - delta_actions
  - hand_handle_transform
  - unwarped_actions
  - b_homie_commands
```

This includes information such as door state, hand-handle transforms, stage information, root velocity, and contact-related signals that are unavailable or undesirable at deployment.

The teacher therefore solves an easier state-estimation problem than the student. Phase 2 transfers the teacher's control behavior to the student's deployable observation space.

---

## 5. Student architecture

The archived DoorMan YAML configures the student as:

```text
RGB image (216 x 384 x 3)
        |
        v
ImageNet-pretrained ResNet18 (trainable)
        |
        v
128-D visual feature
        |
        +---- concatenate ---- actor_obs / proprioception
                               |
                               v
                     2-layer LSTM (hidden=256)
                               |
                               v
                     MLP [512, 256, 128]
                               |
                               v
                        19-D high-level action
```

Important configuration:

```yaml
module_dim:
  vision_feature_dim: 128

actor:
  _target_: gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred_recurrent.VisionRecurrentActorObjPred
  running_mean_std: True
  rnn_type: "lstm"
  rnn_hidden_dim: 256
  rnn_num_layers: 2
```

Vision backbone:

```yaml
vision_module:
  module_config_dict:
    input_dim: [vision_obs]
    output_dim: [vision_feature_dim]
    layer_config:
      type: ResNet
      resnet_type: resnet18
      pretrained: true
      trainable: true
```

Action head:

```yaml
mlp_module:
  module_config_dict:
    input_dim: [actor_obs]
    output_dim:
      - ${eval:'${algo.config.homie_command_dim} + ${algo.config.non_homie_command_actions_dim}'}
    layer_config:
      type: MLP
      hidden_dims: [512, 256, 128]
      activation: SiLU
```

The ResNet is jointly optimized with the recurrent policy because `trainable: true`.

---

## 6. Camera configuration

The Phase-2 archive uses RGB only:

```yaml
camera_types:
  - rgb: true
  - depth: false

camera_resolutions: [216, 384]
camera_attached_link: "d435_link"
```

This is consistent with the paper's deployment description: the robot uses a RealSense D435i but does not use depth output.

---

## 7. Action decomposition and HOMIE

DoorMan does not ask the Phase-2 network to relearn low-level locomotion from scratch.

The student predicts:

```text
19-D high-level DoorMan action
 = 3 locomotion command dimensions
 + 14 upper-body action dimensions
 + 2 finger primitive dimensions
```

From the experiment YAML:

```yaml
homie_command_dim: 3
non_homie_command_actions_dim: 16  # 14 upper body dofs + 2 finger primitives
```

HOMIE then produces the lower-body component. The trainer concatenates the student output with the frozen HOMIE output to create the robot action used by the environment.

Conceptually:

```text
student 19-D output
   |
   +--> first 3 dims --> HOMIE walk/stand --> lower-body action
   |
   +--> remaining task/manipulation dims

combined execution action --> G1 simulator
```

The robot config uses a 31-D execution action vector in this DoorMan setup.

---

## 8. Exact DAgger interaction loop

For each simulator step:

### Step A: obtain the current simulator observations

The environment provides both:

- student-facing `actor_obs` + `vision_obs`
- privileged `teacher_obs`

They refer to the **same physical simulator state**.

### Step B: query the frozen teacher

The DoorMan distillation trainer performs deterministic teacher inference on `teacher_obs`:

```python
teacher_actions = self.ref_model.act_inference(
    obs_dict=deepcopy(obs_dict),
    input_key="teacher_obs"
)
```

The result is the supervision target for the student.

### Step C: run the student

The student processes current RGB/proprioception through the ResNet18 + LSTM + MLP policy.

The trainer deliberately uses the student's action mean for rollout:

```python
actions = torch.cat([
    student_state_dict["action_mean"],
    homie_actions
], dim=-1)
```

This means the environment is visited according to the current student policy (plus frozen HOMIE lower-body control).

### Step D: step the simulator

The student/HOMIE action advances the environment to the next state.

### Step E: store teacher labels on student-visited states

The rollout storage contains, among other values:

- `actor_obs`
- `vision_obs`
- `gt_actions`
- `dones`

For recurrent policies, trajectories are split and padded around episode boundaries before mini-batch training.

### Step F: supervised DAgger update

The student is trained to match the teacher action on those student-visited states.

Then the improved student starts the next rollout iteration, changing the visited-state distribution. The teacher continues to annotate that new distribution.

This is the interactive DAgger loop:

```text
student rollout
   -> query teacher on visited states
   -> supervised update
   -> improved student rollout
   -> query teacher again
   -> supervised update
   -> ...
```

---

## 9. DAgger loss

The generic distillation trainer defines:

```yaml
dagger_bc_loss_type: l2
dagger_bc_loss_coef: 1.0
```

`l2` maps to PyTorch `MSELoss`.

For DoorMan + HOMIE, the trainer computes the loss only over the student policy's own action dimensions:

```python
dagger_bc_loss = self.bc_loss_fn(
    policy_results["action_mean"][..., : self.policy_model.num_actions],
    mb_gt_actions[..., : self.policy_model.num_actions],
)
```

So the main Phase-2 objective is:

```text
L_DAgger = MSE(student_action_mean, teacher_action)
```

or mathematically:

```text
L_DAgger = (1/N) * sum_t || pi_S(o_t) - pi_T(s_t) ||^2
```

where:

- `pi_S` is the vision student
- `o_t` is RGB + non-privileged proprioception
- `pi_T` is the frozen privileged teacher
- `s_t`/`teacher_obs` contains privileged simulator state

This Phase-2 objective is supervised imitation. It is not the Phase-1 PPO objective and not the Phase-3 GRPO objective.

---

## 10. Who controls the rollout?

The archived DoorMan experiment YAML contains:

```yaml
rollout_with_teacher_num_steps: 0
teacher_rollout_ratio: 0.0
```

The trainer only replaces student rollout actions with teacher actions when its explicit teacher-rollout logic is enabled (for example through `rollout_with_teacher_num_steps` or `enforce_teacher_rollout`).

With the archived experiment configuration, the clean interpretation is:

```text
student controls environment visitation
teacher labels those visited states
```

This is the main mechanism that distinguishes DAgger from ordinary behavioral cloning on teacher-only demonstrations.

---

## 11. Training batch structure in the archived experiment

The archived DoorMan Phase-2 YAML uses:

```yaml
num_envs: 4096
num_steps_per_env: 8
num_learning_epochs: 1
num_mini_batches: 4
actor_learning_rate: 1e-4
critic_learning_rate: 1e-4
```

This corresponds to roughly:

```text
4096 environments x 8 steps = 32768 simulator transitions per rollout batch
```

before recurrent trajectory processing / mini-batching.

The student action noise is also effectively fixed to a very small value:

```yaml
init_noise_std: 0.001
max_noise_std: 0.001
clamp_noise_std: True
freeze_noise_std: True
```

The DoorMan-specific trainer further uses `action_mean` for distillation rollout, so the emphasis is on student-distribution correction rather than stochastic action exploration.

---

## 12. Visual randomization for sim-to-real

The Phase-2 experiment enables visual augmentation and randomized dome lighting:

```yaml
defaults:
  - /domain_rand: domain_rand_visual_ImageRand

simulator:
  config:
    randomize_dome_light: True
```

Configured RGB augmentations include:

| augmentation | probability | range |
|---|---:|---|
| brightness | 0.25 | 0.7 - 2.0 |
| contrast | 0.25 | 0.5 - 1.5 |
| hue | 0.50 | -0.1 - 0.1 |
| saturation | 0.25 | 0.5 - 2.0 |
| Gaussian noise | 0.25 | std 0.0 - 0.15 |
| Gaussian blur | 0.25 | kernel 3 - 5 |

The paper additionally describes large visual diversity from randomized PBR materials, dome-light textures, camera variation, rendering effects, and procedural door generation. This diversity is essential because the student must learn visual cues that survive the simulation-to-real appearance gap.

---

## 13. Object-position auxiliary head

The student class is `VisionRecurrentActorObjPred` and includes a 3-D object-position prediction head:

```yaml
obj_pred_mlp:
  module_config_dict:
    input_dim: [vision_feature_dim]
    output_dim: [3]
    layer_config:
      type: MLP
      hidden_dims: [256, 128, 64]
      activation: SiLU
```

The trainer has infrastructure for an auxiliary object-position loss. However, the archived DoorMan experiment YAML sets:

```yaml
obj_pred_loss_coef: 0.0
obj_pred_loss_type: "l2"
```

Therefore, **with the archived YAML as written, this auxiliary loss contributes zero weight** and the effective Phase-2 optimization is the DAgger action imitation loss.

The project README shows a runnable example overriding `obj_pred_loss_coef=1.0`; that example should be treated as an alternative training recipe rather than evidence that the archived experiment YAML used a nonzero auxiliary loss.

---

## 14. Paper/config/code differences that matter for reproduction

### 14.1 LSTM hidden size

The paper method text says the student uses a two-layer LSTM with **512 units each**.

The archived DoorMan experiment YAML configures:

```yaml
rnn_hidden_dim: 256
rnn_num_layers: 2
```

For reproduction of this archive, prefer the actual YAML/code configuration (256) unless deliberately reconstructing the paper's reported model.

### 14.2 Object prediction loss

Archived experiment YAML:

```yaml
obj_pred_loss_coef: 0.0
```

README example:

```text
++algo.config.obj_pred_loss_coef=1.0
```

These represent different recipes.

### 14.3 Teacher rollout ratio naming

The README example uses:

```text
++algo.config.teacher_rollout_ratio=0.3
```

The DoorMan trainer's explicit teacher-mixing branch reads configuration names such as:

```text
enforce_teacher_rollout
ratio_teacher_rollout
```

while the experiment YAML also contains `teacher_rollout_ratio`.

Because these names are not identical, do not assume a README override is active in this archived trainer path without tracing the resolved Hydra configuration and runtime code. The archived experiment's clearest behavior is student-driven rollout with online teacher labeling.

---

## 15. Why Phase 2 still needs Phase 3

DAgger transfers the teacher's control policy to the student's RGB/proprioception observation space, but it cannot fully remove the information asymmetry.

The privileged teacher may know the exact handle/door state even when the student's camera is occluded. Therefore, a teacher-optimal action is not always the best action for a partially observed student.

This motivates the paper's Phase 3 RL bootstrap: the student is allowed to optimize task success under its own observations and can discover compensatory behaviors, for example repositioning to keep the manipulated region visible.

The three phases can be summarized as:

```text
Phase 1: privileged PPO teacher
  learns how to solve the door task

Phase 2: DAgger vision distillation
  learns to imitate that solution from RGB + proprioception

Phase 3: student RL bootstrap (GRPO)
  improves behavior specifically for partial observability
```

---

## 16. Minimal pseudo-code of DoorMan Phase 2

```python
teacher = load_phase1_teacher()
teacher.eval()
freeze(teacher)

homie_walk = load_homie_walk()
homie_stand = load_homie_stand()
freeze(homie_walk)
freeze(homie_stand)

student = VisionStudent(
    vision_encoder="ResNet18(ImageNet pretrained, trainable)",
    vision_feature_dim=128,
    recurrent="2-layer LSTM(hidden=256)",
    action_head="MLP(512,256,128)",
    action_dim=19,
)

for iteration in range(num_iterations):
    rollout = []

    for t in range(num_steps_per_env):
        obs = env.observe()

        with torch.no_grad():
            teacher_action = teacher(obs["teacher_obs"])
            homie_action = frozen_homie(obs["homie_obs"])

        student_action = student(
            rgb=obs["vision_obs"],
            proprio=obs["actor_obs"],
        )

        # Archived DoorMan behavior: student action mean drives visitation.
        env_action = concat(student_action.mean, homie_action)
        next_obs, reward, done, info = env.step(env_action)

        rollout.append({
            "actor_obs": obs["actor_obs"],
            "vision_obs": obs["vision_obs"],
            "teacher_action": teacher_action,
            "done": done,
        })

    # recurrent split/pad around episode boundaries
    batches = build_recurrent_minibatches(rollout)

    for batch in batches:
        pred = student(batch["vision_obs"], batch["actor_obs"])
        loss = mse(pred.action_mean, batch["teacher_action"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This pseudo-code intentionally omits framework-specific TRL/PPO inheritance details and shows the actual learning mechanism of Phase 2.

---

## 17. Practical reproduction checklist

Before starting Phase 2, verify:

- [ ] Phase-1 teacher checkpoint exists and `teacher_actor_path` points to it.
- [ ] `models/model_walk.pt` exists.
- [ ] `models/model_stand.pt` exists.
- [ ] RGB cameras are enabled.
- [ ] `depth: false` if matching archived DoorMan RGB training.
- [ ] ResNet18 is initialized as pretrained and remains trainable.
- [ ] student recurrent configuration matches the intended recipe (archive: 2 x LSTM, hidden 256).
- [ ] DAgger BC loss is L2/MSE.
- [ ] teacher and HOMIE parameters are frozen.
- [ ] student rollout versus teacher rollout settings are explicitly verified from the resolved Hydra config.
- [ ] image/domain randomization is enabled for sim-to-real training.
- [ ] any auxiliary object prediction loss is intentionally selected (`0.0` in archived YAML; README demonstrates another option).

---

## 18. One-sentence definition

**DoorMan Phase 2 freezes the privileged Phase-1 teacher and HOMIE locomotion controllers, lets a ResNet18 + recurrent RGB/proprioception student interact with the simulated door task, queries the teacher for actions at the states actually visited by that student, and minimizes an MSE imitation loss so the deployable vision policy learns the teacher's high-level door-opening behavior while being trained on its own state distribution.**
