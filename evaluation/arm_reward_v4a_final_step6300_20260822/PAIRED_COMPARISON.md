# Arm reward-v2 paired diagnostic

Anchor success: 115/128 (89.84%); candidate success: 112/128 (87.50%); delta -2.34%.

| Metric | Anchor Q4 | Candidate Q4 | Delta | Candidate Q4/Q2 | Anchor slope/s | Candidate slope/s |
|---|---:|---:|---:|---:|---:|---:|
| left_arm_rest_error_normalized | 0.2068 | 0.1656 | -0.0412 | 2.1804 | 0.0138 | 0.0070 |
| right_arm_rest_error_normalized | 0.1346 | 0.1313 | -0.0033 | 3.2001 | 0.0212 | 0.0181 |
| left_proximal_rest_error_normalized | 0.0690 | 0.0496 | -0.0194 | 0.7308 | -0.0103 | -0.0133 |
| right_proximal_rest_error_normalized | 0.1087 | 0.1037 | -0.0050 | 4.2444 | 0.0162 | 0.0131 |
| left_wrist_rest_error_normalized | 0.3904 | 0.3202 | -0.0703 | 3.6938 | 0.0460 | 0.0340 |
| right_wrist_rest_error_normalized | 0.1691 | 0.1680 | -0.0011 | 2.6613 | 0.0278 | 0.0247 |
| left_arm_action_delta_rms | 0.6375 | 0.5225 | -0.1149 | 2.3695 | n/a | n/a |
| right_arm_action_delta_rms | 0.6767 | 0.7156 | +0.0388 | 3.3740 | n/a | n/a |

## Direction summary

Q4 improved on 7/8 arm/action metrics; error slope improved on 6/6 metrics.

## Predeclared final acceptance gates

- PASS — `success_drop_le_5pp`
- FAIL — `left_arm_q4_over_q2_le_1_2`
- FAIL — `right_arm_q4_over_q2_le_1_2`
- FAIL — `all_error_slopes_le_0_005_per_s`
- FAIL — `left_wrist_q4_lt_0_12`
- FAIL — `right_proximal_q4_lt_0_06`
- FAIL — `both_arm_action_q4_lt_0_25`

This early gate reports direction and safety. Failure to meet all final targets at one checkpoint does not by itself authorize changing the frozen training scheme.
