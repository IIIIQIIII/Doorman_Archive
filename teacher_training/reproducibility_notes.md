# DoorMan Teacher Reproducibility Notes

This file separates **what is directly evidenced in the archive** from **what should be treated as a reconstruction assumption**.

## 1. What is directly supported by the archive

### Teacher algorithm

The paper explicitly states that the teacher is trained with standard PPO.

Primary source:

- `arxiv/2512.01061/tex/sec/3_method.tex`

### Teacher is privileged-state, not RGB

The paper and observation config show that the teacher receives simulator-only state such as door/handle transforms, contact information, door state, and root velocity.

Primary sources:

- `arxiv/2512.01061/tex/sec/3_method.tex`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/obs/wbmanip/door_open_homie.yaml`

### DoorMan builds on pretrained HOMIE locomotion

The paper says the method is built on a pretrained whole-body controller to avoid learning locomotion from scratch.

The teacher YAML points to:

```text
./models/model_walk.pt
./models/model_stand.pt
```

and sets:

```yaml
opt_homie: False
```

The PPO trainer freezes those models under that setting.

Primary sources:

- `arxiv/2512.01061/tex/sec/3_method.tex`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_lstm.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer_homie_api.py`

### The public teacher recipe exists

The archived project README exposes a teacher training command using:

```text
+exp=wbmanip/door_open_homie_lstm
```

Primary source:

- `github/GR00T-VisualSim2Real/README.md`

### Student config points to a concrete historical teacher checkpoint

The DAgger config contains:

```text
teacher_actor_path: logs_rl/g1_open_door_homie/wbmanip/door_open_homie_pregrasp-20251014_204826/model_step_020450.pt
```

Primary source:

- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml`

## 2. What is not fully recoverable from the working tree alone

The historical checkpoint path above references an experiment directory named:

```text
door_open_homie_pregrasp-20251014_204826
```

but the archived `gr00t/rl/config/exp/wbmanip/` directory contains only:

```text
door_open_homie_lstm.yaml
door_open_homie_dagger-lstm.yaml
```

Therefore, the current public teacher YAML is clearly the intended reconstruction entry point, but the working tree alone does **not** prove that every runtime override used to create the historical `model_step_020450.pt` checkpoint is identical to the values currently stored in `door_open_homie_lstm.yaml`.

This distinction matters for exact reproduction.

## 3. The historical teacher checkpoint itself is not present as a normal archived log file

The archive contains source/configuration and model dependencies, but the referenced training-log checkpoint under `logs_rl/.../model_step_020450.pt` is not present in the normal working tree.

As a result, this archive is sufficient to reconstruct the **training mechanism and public recipe**, but not necessarily to recover the exact original teacher weights from that historical path.

## 4. Paper vs current-code differences

### Staged-reset buffer size

Paper method text discusses a rolling buffer of 100 recent snapshots, and the experiment compares 0/10/100.

Current archived env YAML uses:

```yaml
staged_reset_max_samples_per_stage: 200
```

Interpretation: the reset-buffer capacity is a tunable implementation detail, not the defining feature of staged reset.

### Teacher network wording

The paper describes the teacher conceptually as a privileged policy and gives the PPO/reward formulation. The archived code provides the more exact current recurrent architecture and HOMIE integration. For reproduction, the code/config should be considered the concrete implementation snapshot while the paper explains the intended method.

## 5. Student settings should not be accidentally attributed to teacher training

The DAgger student configuration enables visual components such as:

- camera rendering
- ResNet18
- visual randomization / dome-light randomization
- teacher checkpoint loading

Those are student-distillation concerns.

The privileged teacher itself is state-based and does not need RGB perception to learn the task.

## 6. LAFAN-G1 should not be mistaken for teacher demonstrations

The environment config references LAFAN-G1 under `reset_from_dataset`.

This is evidence of reset-state initialization, not of behavioral cloning for the teacher. The task teacher remains PPO-trained.

## 7. Recommended reproduction order

For someone attempting to reproduce the teacher from this archive:

1. restore/install the archived DoorMan code environment;
2. obtain the required HOMIE `model_walk.pt` and `model_stand.pt` dependencies;
3. obtain/configure the LAFAN-G1 reset dataset expected by the environment;
4. run the teacher experiment `wbmanip/door_open_homie_lstm`;
5. verify staged reset and six-stage reward logic are active;
6. evaluate the resulting teacher checkpoint before starting DAgger;
7. point `teacher_actor_path` in the student config to the newly trained checkpoint.

## 8. Source map for audit

### Paper

- `arxiv/2512.01061/tex/sec/3_method.tex`
  - teacher privileged observations
  - PPO teacher training
  - HOMIE dependency
  - DAgger student distinction
  - staged-reset method
  - GRPO student refinement distinction

- `arxiv/2512.01061/tex/sec/4_experiment.tex`
  - staged-reset ablation
  - teacher success reference
  - student GRPO comparison

- `arxiv/2512.01061/tex/sec/supp.tex`
  - teacher reward table
  - door randomization ranges

### Code/config

- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_lstm.yaml`
  - teacher architecture
  - PPO hyperparameters
  - HOMIE paths
  - `opt_homie=False`

- `github/GR00T-VisualSim2Real/gr00t/rl/config/obs/wbmanip/door_open_homie.yaml`
  - privileged actor/critic observations

- `github/GR00T-VisualSim2Real/gr00t/rl/config/rewards/wbmanip/reward_door_open_homie.yaml`
  - reward weights

- `github/GR00T-VisualSim2Real/gr00t/rl/config/env/door_open_homie.yaml`
  - six-stage timing
  - staged reset
  - action processing
  - LAFAN-G1 reset source

- `github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer_homie_api.py`
  - HOMIE model loading
  - freezing logic
  - walk/stand switching
  - PPO runtime integration

- `github/GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py`
  - teacher vs student construction path
  - actor/critic initialization
  - trainer startup

- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml`
  - historical teacher checkpoint reference
  - teacher loading for student distillation

## Bottom line

The archive strongly supports the following description:

> The DoorMan teacher is a recurrent privileged-state task policy trained with PPO in Isaac simulation, on top of frozen pretrained HOMIE locomotion controllers, using dense six-stage reward shaping and staged-reset exploration. The trained state policy is later loaded as the teacher for RGB DAgger distillation.

What cannot be claimed from the working tree alone is that the current public YAML reproduces every hidden/runtime override of the historical checkpoint bit-for-bit.
