# DoorMan Teacher Reward Shaping and Staged Reset

This note focuses on the two mechanisms that make the privileged teacher trainable with on-policy PPO: **stage-specific reward shaping** and **staged-reset exploration**.

## 1. Six-stage decomposition

The paper decomposes door opening into:

| Stage | Meaning |
|---|---|
| 0 | Walk to door |
| 1 | Pre-grasp |
| 2 | Grasp |
| 3 | Open |
| 4 | Swing |
| 5 | Pass through door |

The reward configuration in the archived GitHub tree closely mirrors the appendix table in the paper.

## 2. Always-on penalties and regularization

Representative terms include:

| Reward / penalty | Archived weight |
|---|---:|
| termination | -1000.0 |
| delta action rate | -0.01 |
| DoF velocity | -1e-3 |
| DoF acceleration | -1e-5 |
| DoF position limits | -5.0 |
| finger primitive limits | -1.0 |
| humanly DoF limit | -1.0 |
| DoF overspeed | -0.1 |
| undesired contact | -0.2 |
| door frame contact | -0.1 |
| door panel contact | -0.1 |
| upright penalty | -1.0 |
| HOMIE action limit | -1.0 |

These terms keep the humanoid upright, smooth, within joint limits, and away from undesirable collisions while the task-specific terms drive progress.

## 3. Stage 0 — Walk to door

Main signals:

- `walk_to_door`: **+5.0**
- upper-body deviation penalty: **-1.0**
- face-door penalty: **-1.0**

The policy is encouraged to approach the door with an appropriate velocity while keeping a reasonable upper-body posture and orientation.

## 4. Stage 1 — Pre-grasp

Main signals:

- hand-handle orientation: **+3.0**
- pre-grasp finger pose: **+1.5**
- unused arm deviation penalty: **-1.0**
- pre-grasp target distance: **+6.0**
- not-standing-still penalty: **-15.0**

This stage uses privileged geometric information to bring the correct hand into a useful pose relative to the handle.

## 5. Stage 2 — Grasp

Main signals:

- grasp finger pose: **+3.0**
- grasp target distance: **+3.0**
- grasp/contact force shaping: **+0.2**

The teacher can use ground-truth hand/handle geometry and simulated contact information, making this stage much easier to shape than it would be from RGB alone.

## 6. Stage 3 — Open

Main signals:

- push/rotate door handle: **+6.0**
- move door hinge: **+6.0**
- push door force: **+0.3**

This directly rewards manipulation of the handle and hinge state.

## 7. Stages 4–5 — Swing and pass through

Main signals:

- don't keep pushing the handle: **+3.0**
- target root distance / progress through doorway: **+12.0**
- standing-still penalty: **-1.0**

The task therefore does not terminate at “door moved.” The policy must continue coordinating whole-body motion until the robot traverses the doorway.

## 8. Always-on positive progress signals

The appendix/config includes:

- stage progress: **+1.0**
- task completion: **+4.0**
- success / remaining-time bonus: **+0.5**

These provide additional incentive to reach later stages and finish efficiently.

## 9. Why normal PPO exploration is insufficient

For a long-horizon contact-rich task, reaching later stages has very low probability under an initially weak policy.

A typical failure pattern is:

```text
approach door
   -> random contact near handle
   -> collision / torque / stability penalties
   -> policy learns that interacting with the handle is risky
   -> later stages become even less likely to be visited
```

This can make an on-policy learner “unlearn” useful early attempts instead of advancing.

## 10. Staged-reset mechanism

DoorMan exploits simulator recoverability.

When an environment reaches a later stage, a snapshot of the robot and door state is cached. During future resets, some rollouts are initialized from these cached later-stage states instead of always restarting from the beginning.

Conceptually:

```text
Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5
             |          |          |          |
             +--cache---+--cache---+--cache---+

future reset:
  50% -> Stage 0
  10% -> Stage 1
  10% -> Stage 2
  10% -> Stage 3
  10% -> Stage 4
  10% -> Stage 5
```

The archived environment config contains:

```yaml
enable_staged_reset: True
staged_reset_ratios: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
staged_reset_max_samples_per_stage: 200
```

## 11. Paper interpretation

The paper explains staged reset as changing the effective initial-state distribution and therefore reweighting the discounted occupancy measure toward later stages. The practical effect is that PPO receives many more useful gradients from states around grasping, opening, swinging, and traversal.

The paper's ablation compares reset-buffer sizes 0, 10, and 100:

- a large buffer reaches most stages quickly and eventually explores all stages;
- a smaller buffer takes substantially longer;
- without staged reset, exploration can fail around the grasp stage.

## 12. Paper/code parameter difference

The paper method text mentions keeping the recent **100** snapshots, and the ablation explicitly studies buffer sizes including 100. The archived current environment YAML uses:

```yaml
staged_reset_max_samples_per_stage: 200
```

Therefore “100” should not be treated as a required invariant of the method. The important mechanism is stage-conditioned snapshot reset; buffer capacity is a tunable implementation parameter.

## 13. Reward curriculum detail

The archived reward config also enables a reward-penalty curriculum:

```yaml
reward_penalty_curriculum: True
reward_initial_penalty_scale: 1.0
reward_min_penalty_scale: 0.2
reward_max_penalty_scale: 1.0
```

The config lists selected task rewards whose scaling is managed based on average goal-reached rate. This adds another curriculum-like component on top of the staged state-reset mechanism.

## Source paths

- `arxiv/2512.01061/tex/sec/supp.tex`
- `arxiv/2512.01061/tex/sec/3_method.tex`
- `arxiv/2512.01061/tex/sec/4_experiment.tex`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/rewards/wbmanip/reward_door_open_homie.yaml`
- `github/GR00T-VisualSim2Real/gr00t/rl/config/env/door_open_homie.yaml`
