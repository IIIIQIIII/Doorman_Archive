# DoorMan Paper Evaluation / Benchmark Protocol

This note extracts the evaluation protocol described in the archived DoorMan paper source and separates it from the code-level `eval_agent_trl.py` workflow.

## 1. Door categories

The paper evaluates three representative door-opening categories:

1. **Push Lever** — rotational handle, door opens in the direction of travel.
2. **Pull Lever** — rotational handle, door opens against the direction of travel.
3. **Push Bar** — push-bar handle, door opens in the direction of travel.

These categories intentionally cover different manipulation and whole-body-control difficulty levels.

## 2. Initial robot placement

For evaluation, the paper states that the robot is randomly placed approximately:

```text
1 meter in front of the door
```

while facing the center of the door.

Yaw is perturbed uniformly within:

```text
±0.3 radians
```

This is an important part of the benchmark: evaluation is not only from one perfectly aligned deterministic initial pose.

## 3. Visual generalization condition

In simulation evaluation, visuals are randomized using textures from a **holdout set**.

Real-world visual appearances are unseen during training.

Therefore a faithful visual-policy benchmark should avoid evaluating only on the exact same appearance distribution/assets seen during training.

## 4. Success definition

The paper evaluates success when the robot:

1. opens the door;
2. traverses through the doorway;
3. reaches a point approximately **1 m beyond the door frame on the opposite side**.

This is stricter than simply turning the handle or opening the hinge by some angle.

The archived simulator implementation expresses the final stage completion with an internal position threshold (`_stage_5_to_complete_condition`). That code-level threshold and the paper's geometric description should be understood as two representations of the same intended end-to-end task: **open and pass through the door**.

## 5. Headline metrics

The paper reports two main task-level metrics:

### Success rate

Higher is better. A trial succeeds only when the end-to-end traversal criterion is met.

### Completion time / task fluency

Lower is better. This measures how efficiently the policy completes the entire door-opening sequence.

The paper uses completion time when comparing DoorMan with human teleoperation baselines, not just success/failure.

## 6. Human teleoperation comparison

The real-world study compares DoorMan with human teleoperators using the same general whole-body control stack.

Teleoperators are divided into:

- **non-experts**: less than one day of robot teleoperation experience;
- **experts**: more than three months of full-time experience.

The paper compares both success rate and task fluency against these baselines.

This part of the benchmark is not reproduced automatically by the open-source simulator evaluator.

## 7. Visual-randomization ablation trial count

For the photorealistic visual-randomization ablation, the paper states that **each configuration is evaluated on 120 unseen-door trials**.

The three reported sub-task columns are:

```text
Push Lever
Pull Lever
Push Bar
```

For the strongest visual-randomization setting (100% texture randomization + dome-light randomization), the paper reports:

```text
Push Lever: 85.8%
Pull Lever: 80.8%
Push Bar:   85.0%
```

These values are paper benchmark results and should not be treated as automatic expected outputs from an arbitrary checkpoint run.

## 8. Teacher / student comparison context

The paper also plots student success during GRPO bootstrapping against teacher-policy success. It describes teacher policies as reaching roughly 80–90% success on the door sub-tasks, while the initial vision student remains lower before GRPO improvement.

This is useful when validating a reproduced training pipeline: teacher evaluation is an upper-bound/reference signal for student distillation and later bootstrapping.

## 9. Code evaluator vs paper benchmark

The distinction can be summarized as:

```text
Official code eval
    checkpoint
      ↓
    eval_agent_trl.py
      ↓
    Isaac Sim rollouts
      ↓
    environment metrics
      ↓
    metrics_eval.json

Paper benchmark
    selected door categories
    + controlled/randomized initial pose
    + held-out visual conditions
    + specified trial counts
    + success-rate measurement
    + completion-time measurement
    + real-world G1 trials / teleop baselines where applicable
```

So the repository **does contain an official evaluator**, but reproducing the exact paper tables requires matching the paper's experimental protocol on top of that evaluator.

## 10. Recommended reproducibility record

When evaluating a DoorMan checkpoint, record at least:

- checkpoint path / step;
- whether it is Teacher, DAgger Student, or GRPO-refined Student;
- resolved Hydra config;
- number of environments;
- number of completed evaluation episodes;
- door category / asset subset;
- visual randomization / holdout appearance setting;
- initial position and yaw distribution;
- success criterion;
- success rate;
- completion time, when available;
- random seed;
- simulator / Isaac Lab version;
- whether videos were saved.

This makes it possible to tell whether a result is merely a smoke test, a standard simulation evaluation, or a paper-comparable benchmark.

## Archived paper source

The relevant experiment description is preserved at:

```text
arxiv/2512.01061/tex/sec/4_experiment.tex
```
