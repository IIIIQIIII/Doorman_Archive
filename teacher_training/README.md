# DoorMan Teacher Training Notes

This folder documents how the **DoorMan teacher policy** is trained, based only on the archived DoorMan GitHub working tree and arXiv v1 sources already stored in this repository.

> Scope rule: the existing `github/` and `arxiv/` archives are treated as immutable source material. This folder is an analysis layer added alongside them.

## Core conclusion

The DoorMan teacher is **not** a vision policy and is **not** trained by DAgger. It is a privileged-state recurrent policy trained in simulation with PPO.

More precisely:

```text
pretrained HOMIE walk/stand controllers (frozen)
                    |
                    v
      DoorMan task-level recurrent policy
     (privileged simulation observations)
                    |
                    | PPO
                    | six-stage shaped rewards
                    | staged-reset exploration
                    v
          trained privileged teacher
                    |
                    | DAgger supervision
                    v
             RGB student policy
                    |
                    | later RL/GRPO refinement
                    v
             final deployed student
```

The task-level teacher therefore learns the door-opening behavior on top of a pretrained locomotion foundation rather than learning all humanoid locomotion from scratch.

## Important architectural point

The archived teacher configuration uses:

- `homie_walk_model_path: ./models/model_walk.pt`
- `homie_stand_model_path: ./models/model_stand.pt`
- `opt_homie: False`

The PPO trainer loads those models and freezes them when `opt_homie=False`. The DoorMan actor learns high-level locomotion commands plus upper-body and finger control while HOMIE produces lower-body locomotion actions.

The configured learned task action is effectively:

- 3 locomotion-command dimensions
- 14 upper-body DoF command dimensions
- 2 finger primitive dimensions

for a 19-dimensional task-level output before HOMIE contributes lower-body control.

## Teacher observations

The teacher uses privileged simulator information unavailable to the real-world RGB student. The archived actor observation configuration includes items such as:

- joint positions and velocities
- robot-to-door relative state
- door joint state
- root linear/angular velocity
- hand force/contact information
- task stage
- privileged door information
- hand-to-handle transforms
- previous/delta/unwarped actions
- HOMIE command state

The paper additionally describes ground-truth robot/hand-to-door transforms and contact wrenches as privileged teacher information.

## Training algorithm

The teacher is trained with PPO using a recurrent actor and critic. The archived experiment configuration uses two-layer LSTMs and MLP backbones.

Representative archived values include:

- actor learning rate: `1e-4`
- critic learning rate: `1e-4`
- `gamma: 0.9975`
- `lam: 0.985`
- desired KL: `0.005`
- initial action noise std: `0.8`
- 5 learning epochs
- 4 mini-batches

See `teacher_training_recipe.md` for the detailed configuration map.

## Why staged reset matters

Door opening is decomposed into six stages:

0. Walk to door
1. Pre-grasp
2. Grasp
3. Open
4. Swing
5. Pass through door

A normal on-policy rollout starting only from stage 0 rarely reaches the later contact-rich stages early in training. DoorMan therefore caches simulator snapshots when progress reaches later stages and resets some environments from those cached states. This greatly increases the training occupancy of downstream stages.

The archived environment configuration enables staged reset with ratios:

```text
[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
```

See `reward_and_staged_reset.md` for details.

## Files in this folder

- `README.md` — high-level explanation of how the teacher is obtained.
- `teacher_training_recipe.md` — architecture, observations, actions, PPO setup, HOMIE integration, and training flow.
- `reward_and_staged_reset.md` — six-stage reward shaping and staged-reset exploration.
- `reproducibility_notes.md` — exact archived source paths, checkpoint evidence, and caveats when reproducing the original run.
- `diagnostics/stage3_gap_step0750_20260819/` — runtime-only instrumentation and results for the current clean-source reproduction at model step 750.

## Current reproduction diagnostic

The step-750 checkpoint from the 2026-08-19 clean-source reproduction was
evaluated without changing its policy, reward, observations, actions, or stage
logic. Of 128 first episodes, 118 entered Stage 3 and none reached Stage 4.
The handle was already depressed for most Stage 3 samples, while the door hinge
never exceeded 0.935 degrees. Contact and opening torque existed but were not
sustained together: the longest interval with both a valid four-contact grasp
and opening torque above 1 Nm was 0.38 seconds.

This is a diagnosis of the in-progress local reproduction, not a claim about
the official released teacher. See
`diagnostics/stage3_gap_step0750_20260819/README.md` for the measurements,
method, scripts, and artifact locations.

## Primary archived sources

Paper sources:

- `arxiv/2512.01061/tex/sec/3_method.tex`
- `arxiv/2512.01061/tex/sec/4_experiment.tex`
- `arxiv/2512.01061/tex/sec/supp.tex`

Code/config sources:

- `github/GR00T-VisualSim2Real/README.md`
- `github/GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_lstm.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/obs/wbmanip/door_open_homie.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/rewards/wbmanip/reward_door_open_homie.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/env/door_open_homie.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer_homie_api.py`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml`

## One-sentence summary

**DoorMan first freezes pretrained HOMIE locomotion controllers, then trains a privileged-state recurrent task policy with PPO using six-stage reward shaping and staged-reset exploration; that trained state policy becomes the teacher used to supervise the later RGB student.**
