# DoorMan post-open diagnostics v2

Episodes: 128; successful: 112 (87.5%); successful post-open samples: 35740.

## Time progression

| Metric | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Left arm error | 0.1419 | 0.0759 | 0.1151 | 0.1656 |
| Right arm error | 0.0470 | 0.0410 | 0.0779 | 0.1313 |
| Left proximal error | 0.1230 | 0.0679 | 0.0746 | 0.0496 |
| Right proximal error | 0.0433 | 0.0244 | 0.0555 | 0.1037 |
| Left wrist error | 0.1670 | 0.0867 | 0.1692 | 0.3202 |
| Right wrist error | 0.0518 | 0.0631 | 0.1078 | 0.1680 |
| Waist yaw rest error (rad) | 0.0499 | 0.0269 | 0.0266 | 0.0269 |
| Torso-pelvis abs yaw (rad) | 0.0491 | 0.0287 | 0.0270 | 0.0272 |
| Crab angle (deg) | 41.2945 | 30.2274 | 40.6631 | 33.2521 |

## Per-episode slopes

- Left arm error: mean 0.0070/s, median 0.0074/s.
- Right arm error: mean 0.0181/s, median 0.0183/s.
- Left proximal error: mean -0.0133/s, median -0.0131/s.
- Right proximal error: mean 0.0131/s, median 0.0133/s.
- Left wrist error: mean 0.0340/s, median 0.0346/s.
- Right wrist error: mean 0.0247/s, median 0.0252/s.
- Waist yaw rest error (rad): mean -0.0036/s, median -0.0037/s.
- Torso-pelvis abs yaw (rad): mean -0.0035/s, median -0.0036/s.

## Coupling correlations

- torso_rel_yaw_vs_left_arm: raw r=0.122, first-difference r=0.141.
- torso_rel_yaw_vs_right_arm: raw r=-0.085, first-difference r=0.040.
- waist_yaw_vs_left_arm: raw r=0.145, first-difference r=0.129.
- waist_yaw_vs_right_arm: raw r=-0.088, first-difference r=0.039.

## Rotation trajectories per successful episode

- pelvis_world_yaw_net_abs_rad: mean 0.5126 rad, p95 0.6539 rad.
- pelvis_world_yaw_path_rad: mean 2.3914 rad, p95 2.6306 rad.
- pelvis_world_yaw_range_rad: mean 1.3404 rad, p95 1.4156 rad.
- torso_world_yaw_net_abs_rad: mean 0.4452 rad, p95 0.5770 rad.
- torso_world_yaw_path_rad: mean 2.4637 rad, p95 2.6580 rad.
- torso_world_yaw_range_rad: mean 1.3478 rad, p95 1.4138 rad.
- torso_relative_pelvis_yaw_net_abs_rad: mean 0.0674 rad, p95 0.0937 rad.
- torso_relative_pelvis_yaw_path_rad: mean 1.6128 rad, p95 1.7787 rad.
- torso_relative_pelvis_yaw_range_rad: mean 0.2061 rad, p95 0.2449 rad.
- waist_yaw_net_abs_rad: mean 0.0675 rad, p95 0.0947 rad.
- waist_yaw_path_rad: mean 1.5884 rad, p95 1.7586 rad.
- waist_yaw_range_rad: mean 0.2018 rad, p95 0.2391 rad.
