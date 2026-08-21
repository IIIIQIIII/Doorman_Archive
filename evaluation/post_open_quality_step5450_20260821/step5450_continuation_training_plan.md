# Step-5450 continuation training plan

This document defines the **single continuation plan** to repair the post-open behavior-quality failures diagnosed at teacher step 5450.

It is a planning/execution document only. The archived `github/` and `arxiv/` trees must remain unchanged.

## Decision

Stop the current official-reward teacher run and freeze step 5450 as the continuation anchor.

Step 5450 is retained because the official end-to-end task success is already strong:

- task success: **117/128 = 91.41%**;
- Stage-5 crab-angle p95: **49.57 deg**;
- Stage-5 moving samples with crab angle >30 deg: **78.10%**;
- HOMIE lateral-command ratio: **0.706**;
- actual lateral-velocity ratio: **0.588**;
- HOMIE planar velocity tracking error: **0.801 m/s**;
- Stage-5 raw arm-delta RMS p95: **0.489**;
- completion-tail raw arm-delta RMS: **0.654**;
- normalized arm-rest error: **0.109 in Stage 5 -> 0.154 in completion tail**;
- strict terminal stability: **0/117 successful episodes**.

The continuation must preserve the mature Stage 0-4 skill while correcting only:

1. Stage-5 locomotion command geometry;
2. Stage-5 upper-body recovery;
3. completion-tail settling behavior.

## Non-negotiable constraints

- Start from the original step-5450 checkpoint.
- Do not change actor/critic input or output dimensions.
- Do not add observations.
- Do not retrain or unfreeze HOMIE; keep `opt_homie: false`.
- Do not change the 19-D high-level action definition.
- Do not change Stage 0-4 task logic or their core rewards.
- Keep the official step-5450 checkpoint immutable.
- Run the continuation as a new experiment, for example:

```text
door_open_homie_step5450_postopen_fix_v1
```

## 1. Stage-5 locomotion shaping

Add two rewards that are active only in `STAGE_THROUGH`.

### 1.1 Crab-angle penalty

```python
@StagedTaskBase.effective_in_stage(STAGE_THROUGH)
def _reward_penalty_stage5_crab_angle(self):
    v = self.base_lin_vel[:, :2]
    speed = torch.linalg.norm(v, dim=-1)

    crab = torch.atan2(
        torch.abs(v[:, 1]),
        torch.clamp(v[:, 0], min=1e-4),
    )

    return torch.where(
        speed > 0.1,
        crab,
        torch.zeros_like(crab),
    )
```

Reward scale:

```yaml
penalty_stage5_crab_angle: -4.0
```

### 1.2 HOMIE lateral-command penalty

```python
@StagedTaskBase.effective_in_stage(STAGE_THROUGH)
def _reward_penalty_stage5_homie_lateral(self):
    cmd = self.get_physical_homie_commands()
    return torch.abs(cmd[:, 1])
```

Reward scale:

```yaml
penalty_stage5_homie_lateral: -6.0
```

Do **not** hard-clamp `vy=0`. The policy must learn to stop requesting the lateral strategy; the execution layer must not hide the bad teacher action.

## 2. Stage-5 target geometry and completion-tail deceleration

Change the Stage-5 root target from:

```yaml
target_root_pos: [2.0, 0.0, 0.72]
```

to:

```yaml
target_root_pos: [1.70, 0.0, 0.72]
```

The task currently marks completion around `root_x > 1.5`, so a 1.70 m target gives a short post-threshold settle distance instead of continuing to pull toward 2.0 m.

Replace the fixed Stage-5 target speed with distance-dependent deceleration:

```python
root_pos_diff = torch.norm(
    self.simulator.robot_root_states[:, :3]
    - self.env_origins
    - self.target_root_pos,
    dim=-1,
)

max_speed = self.config.get("target_root_vel", 0.3)

desired_speed = max_speed * torch.clamp(
    root_pos_diff / 0.4,
    min=0.0,
    max=1.0,
)

root_vel_reward = torch.exp(
    -torch.square(root_vel_along_target_direction - desired_speed)
    / (0.2 ** 2)
)
```

Keep the existing root-position reward.

Intended terminal motion:

```text
pass doorway
-> cross success threshold at x > 1.5
-> continue roughly 0.2 m
-> smoothly decelerate
-> settle near x = 1.70
```

## 3. Stage-5 upper-body resting-pose attractor

The 19-D high-level action uses delta accumulation. The Stage-5 arm repair therefore needs an explicit state-level attractor toward `resting_dof_pos`.

Add this after delta accumulation and before the accumulated actions are written back to `actor_state["actions"]`:

```python
if self.config.get("stage5_arm_rest_attractor", False):
    stage5_ids = torch.where(self.stage_buf == 5)[0]

    if len(stage5_ids) > 0:
        # High-level delta-action layout:
        # 0:3   HOMIE locomotion
        # 3:17  14 upper-body joints
        # 17:19 finger primitives
        delta_idx = torch.arange(3, 17, device=self.device)

        # G1 non-finger upper-body arm joints are indices 15:29.
        joint_idx = torch.arange(15, 29, device=self.device)

        action_scale = self.config.robot.control.action_scale

        rest_action = (
            self.resting_dof_pos[:, joint_idx]
            - self.default_dof_pos[:, joint_idx]
        ) / action_scale

        alpha = self.config.stage5_arm_rest_attractor_alpha

        current = self._delta_actions[
            stage5_ids[:, None], delta_idx[None, :]
        ]
        target = rest_action.expand(len(stage5_ids), -1)

        self._delta_actions[
            stage5_ids[:, None], delta_idx[None, :]
        ] = (1.0 - alpha) * current + alpha * target
```

Configuration:

```yaml
stage5_arm_rest_attractor: true
stage5_arm_rest_attractor_alpha: 0.04
```

Use `alpha=0.04` for the first continuation run. Do not search this parameter before the first full continuation evaluation.

## 4. Train the policy itself to stop producing arm deltas

The attractor alone is not sufficient, because a teacher whose bad raw arm deltas are merely canceled by the environment would still provide bad DAgger labels.

Add the following Stage-5-only rewards.

### 4.1 Raw arm-delta activity

```python
@StagedTaskBase.effective_in_stage(STAGE_THROUGH)
def _reward_penalty_stage5_arm_delta_rms(self):
    arm_delta = self._last_delta_actions[:, 3:17]
    return torch.sqrt(
        torch.mean(torch.square(arm_delta), dim=-1) + 1e-8
    )
```

Scale:

```yaml
penalty_stage5_arm_delta_rms: -3.0
```

### 4.2 Worst-joint resting-pose error

Do not aggregate only with mean 14-joint error. The step-5450 diagnostic showed that the wrist failures are hidden by the mean.

```python
@StagedTaskBase.effective_in_stage(STAGE_THROUGH)
def _reward_penalty_stage5_arm_rest_worst(self):
    idx = self._upper_non_finger_dof_idx

    q = self.simulator.dof_pos[:, idx]
    q_rest = self.resting_dof_pos[:, idx]

    limits = self.simulator.dof_pos_limits[idx]
    joint_range = (limits[:, 1] - limits[:, 0]).clamp_min(1e-3)

    normalized_error = torch.abs(q - q_rest) / joint_range[None, :]
    return normalized_error.max(dim=-1).values
```

Scale:

```yaml
penalty_stage5_arm_rest_worst: -10.0
```

Keep the existing:

```yaml
penalty_upper_body_non_finger_deviation_l1: -1.0
```

The new worst-joint term is an additional Stage-5 terminal constraint, not a replacement.

## 5. Preserve the 50-step completion tail

Keep:

```yaml
reset_on_complete: true
reset_on_complete_delay: 50
```

The tail should become a deliberate settle period:

```text
completion
-> reduce forward velocity
-> remove lateral command
-> raw arm delta -> 0
-> accumulated arm target -> resting target
-> maintain stable terminal posture
```

Do not remove the tail to hide the instability.

## 6. Resume model weights but reset optimizer state

Load the step-5450 actor/critic weights and trainer global step, but do **not** continue with the old optimizer momentum or LR-scheduler state after changing the objective.

Add a continuation option:

```yaml
reset_optimizer_on_resume: true
```

Checkpoint-loading behavior should be equivalent to:

```python
if not self.config.get("reset_optimizer_on_resume", False):
    if "optimizer_state_dict" in checkpoint:
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "lr_scheduler_state_dict" in checkpoint:
        self.lr_scheduler.load_state_dict(
            checkpoint["lr_scheduler_state_dict"]
        )
```

Model/value weights and `state.global_step` still load from step 5450.

## 7. Continuation PPO configuration

Use exactly this first-run fine-tuning configuration:

```yaml
algo:
  config:
    actor_learning_rate: 2.0e-5
    critic_learning_rate: 5.0e-5

    entropy_coef: 0.002
    desired_kl: 0.002

    init_noise_std: 0.35
    max_noise_std: 0.35
    clamp_noise_std: true

    num_steps_per_env: 64
    num_learning_epochs: 3
    num_mini_batches: 4

    opt_homie: false
    save_interval: 100
```

Keep unchanged:

```yaml
gamma: 0.9975
lam: 0.985
num_envs: 4096
```

Use the existing staged-reset training mechanism unchanged.

## 8. Training horizon

Resume from step 5450 and train for at most 1000 additional optimizer steps:

```text
5450 -> 6450 maximum
```

Save/evaluate every 100 steps:

```text
5550
5650
5750
5850
5950
6050
6150
6250
6350
6450
```

Do not continue past 6450 without a new decision based on the post-open metrics.

## 9. Frozen evaluation protocol

For every checkpoint use the same diagnostic protocol as the step-5450 anchor:

```text
seed = 42
128 environments
enable_staged_reset = false during eval
same full-reset initialization
same post-open instrumentation
same metric definitions
```

Do not choose checkpoints by video appearance.

## 10. Acceptance gates

A checkpoint is acceptable only if all of the following are satisfied:

```text
task success                     >= 115/128

Stage-5 crab-angle p95           <= 20 deg
Stage-5 crab >30 deg fraction    <= 10%

HOMIE lateral-command ratio      <= 0.25
actual lateral-velocity ratio    <= 0.25
HOMIE planar tracking error      <= 0.55 m/s

Stage-5 arm-delta RMS p95        <= 0.20
completion-tail arm-delta RMS    <= 0.15

strict terminal stability        >= 90% of successful episodes

left wrist yaw terminal p95      <= 0.35 rad
right wrist pitch terminal p95   <= 0.35 rad
left wrist roll terminal p95     <= 0.35 rad
left wrist pitch terminal p95    <= 0.35 rad
```

The final teacher must satisfy the entire gate for **two consecutive 100-step checkpoints**. Select the second checkpoint as the new teacher.

## 11. Failure stop condition

Protect the mature Stage 0-4 capability.

If task success is below:

```text
108/128
```

for two consecutive evaluations, stop the continuation immediately and do not train further from that branch.

## 12. Interpretation of the repair

This plan directly addresses the three failures isolated by the step-5450 instrumentation:

```text
HOMIE lateral-command ratio = 0.706
-> Stage-5 crab-angle + lateral-command shaping
-> high-level policy must stop requesting sideways traversal

arm delta RMS 0.489 -> 0.654
-> Stage-5 raw-arm-delta penalty
-> policy must stop actively driving the wrists after manipulation

persistent manipulation delta state
-> Stage-5 resting-pose attractor
-> accumulated upper-body target is explicitly released toward resting pose

success at x > 1.5 while target remains x = 2.0
-> target x = 1.70 + distance-dependent deceleration
-> completion tail becomes a settling phase instead of continued traversal
```

## Final execution instruction

Do not continue the existing official-reward teacher run.

Freeze step 5450, create the isolated continuation experiment above, apply only the Stage-5/terminal modifications in this document, keep HOMIE frozen, reset optimizer state, and evaluate every 100 steps using the frozen 128-environment post-open diagnostic.
