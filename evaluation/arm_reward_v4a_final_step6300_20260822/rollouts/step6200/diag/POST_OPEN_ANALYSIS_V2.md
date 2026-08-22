# DoorMan post-open diagnostics v2

Episodes: 128; successful: 115 (89.8%); successful post-open samples: 35407.

## Time progression

| Metric | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Left arm error | 0.1514 | 0.0730 | 0.1175 | 0.2068 |
| Right arm error | 0.0427 | 0.0271 | 0.0810 | 0.1346 |
| Left proximal error | 0.1261 | 0.0556 | 0.0658 | 0.0690 |
| Right proximal error | 0.0404 | 0.0180 | 0.0675 | 0.1087 |
| Left wrist error | 0.1851 | 0.0963 | 0.1864 | 0.3904 |
| Right wrist error | 0.0458 | 0.0392 | 0.0989 | 0.1691 |
| Waist yaw rest error (rad) | 0.0500 | 0.0282 | 0.0294 | 0.0326 |
| Torso-pelvis abs yaw (rad) | 0.0492 | 0.0292 | 0.0298 | 0.0333 |
| Crab angle (deg) | 42.4041 | 31.8560 | 36.7309 | 19.6020 |

## Per-episode slopes

- Left arm error: mean 0.0138/s, median 0.0143/s.
- Right arm error: mean 0.0212/s, median 0.0211/s.
- Left proximal error: mean -0.0103/s, median -0.0096/s.
- Right proximal error: mean 0.0162/s, median 0.0164/s.
- Left wrist error: mean 0.0460/s, median 0.0465/s.
- Right wrist error: mean 0.0278/s, median 0.0278/s.
- Waist yaw rest error (rad): mean -0.0024/s, median -0.0024/s.
- Torso-pelvis abs yaw (rad): mean -0.0022/s, median -0.0021/s.

## Coupling correlations

- torso_rel_yaw_vs_left_arm: raw r=0.164, first-difference r=0.130.
- torso_rel_yaw_vs_right_arm: raw r=0.014, first-difference r=0.144.
- waist_yaw_vs_left_arm: raw r=0.172, first-difference r=0.128.
- waist_yaw_vs_right_arm: raw r=0.005, first-difference r=0.132.

## Rotation trajectories per successful episode

- pelvis_world_yaw_net_abs_rad: mean 0.1436 rad, p95 0.2994 rad.
- pelvis_world_yaw_path_rad: mean 2.2459 rad, p95 2.4560 rad.
- pelvis_world_yaw_range_rad: mean 1.1041 rad, p95 1.2002 rad.
- torso_world_yaw_net_abs_rad: mean 0.0929 rad, p95 0.2168 rad.
- torso_world_yaw_path_rad: mean 2.3249 rad, p95 2.5025 rad.
- torso_world_yaw_range_rad: mean 1.0837 rad, p95 1.1775 rad.
- torso_relative_pelvis_yaw_net_abs_rad: mean 0.0654 rad, p95 0.1012 rad.
- torso_relative_pelvis_yaw_path_rad: mean 1.6989 rad, p95 1.8932 rad.
- torso_relative_pelvis_yaw_range_rad: mean 0.2094 rad, p95 0.2617 rad.
- waist_yaw_net_abs_rad: mean 0.0645 rad, p95 0.1015 rad.
- waist_yaw_path_rad: mean 1.6829 rad, p95 1.8730 rad.
- waist_yaw_range_rad: mean 0.2055 rad, p95 0.2547 rad.
