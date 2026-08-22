# V4-A step6200 vs step6300 paired evaluation

## Result

This is the final fixed-window acceptance evaluation for the V4-A timing
ablation. It uses full resets only and does not use the curriculum-mixed online
success signal.

| Checkpoint | Complete tasks | Success rate |
|---|---:|---:|
| step6200 anchor | 115/128 | 89.84375% |
| step6300 V4-A | 112/128 | 87.50% |

The -2.34375 percentage-point change passes the predeclared five-point task
safety margin. Seven of eight Q4 arm/action metrics and six of six physical
error slopes improve, but the absolute bilateral arm-quality gates do not all
pass. V4-A is therefore safe and directionally useful, not a complete solution.

See `PAIRED_COMPARISON.md` and `paired_comparison.json` for exact values and gate
results.

## Protocol and reproduction

The archived `run_arm_reward_v4a_gate_eval_4090.sh` is the exact launcher used.
On the 4090 host it was invoked with a new absolute output directory:

```bash
OUTPUT_ROOT=/sda/mashijian/doorman_arm_reward_v4a_timing_20260822/evals/arm_v4a_final_step6300_20260822_1728 \
  bash /sda/mashijian/doorman_arm_reward_v4a_timing_20260822/run_arm_reward_v4a_gate_eval_4090.sh
```

The script:

1. freezes the step6200 anchor and step6300 candidate;
2. verifies stable candidate size and embedded global steps;
3. records SHA256 hashes and identical config copies;
4. evaluates both with V4 code, seed 42, staged reset disabled, 128 environments,
   and 128 completed episodes;
5. runs the anchor on physical GPU 6 and candidate on physical GPU 7;
6. validates schema-v2 diagnostic completion and performs the paired comparison.

Checkpoint hashes are in `checkpoints/SHA256SUMS`. Checkpoint binaries are not
duplicated in this Git archive.

## Artifact map

- `PAIRED_COMPARISON.md`: human-readable comparison and acceptance gates;
- `paired_comparison.json`: machine-readable exact comparison;
- `rollouts/step*/diag/post_open_summary.json`: episode completeness and scope;
- `rollouts/step*/diag/post_open_analysis_v2.json`: aggregated physical metrics;
- `rollouts/step*/diag/post_open_episode_metrics.jsonl`: all 128 episode records;
- `rollouts/step*.supervisor.log`: diagnostic completion evidence;
- `rollouts/step*/launcher.log`: full evaluator output, including known warnings;
- `checkpoints/step*/config.yaml`: frozen evaluation configurations.

Large raw time-series/kinematics CSVs remain in the server output directory.

## Known official JSON defect

After each evaluator completed all 128 episodes, official code attempted to
serialize a PyTorch tensor into `metrics_eval.json` and raised:

```text
TypeError: Object of type Tensor is not JSON serializable
```

Accordingly, the two archived `metrics_eval.json` files are intentionally kept
as evidence of the defect but are truncated and are not evaluation inputs. The
diagnostic hook had already produced 128 episode records, schema version 2, and
`active_unfinished_env_count=0`; the paired comparator consumes those validated
diagnostic artifacts instead.
