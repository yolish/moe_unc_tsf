# Calibration results changelog

What each run of `scripts/collect_calibration_results.py` picked up, newest first. The run rescans every file under `logs/` and every `result_calibration_*.txt`, so new log files are found on their own; this file records what was new that day.

## 2026-08-28 06:00 — 4976 results / 2188 configs

No change since the previous run (2026-08-27 06:00). No new logs, no new results, no value moved.

## 2026-08-27 06:00 — 4976 results / 2188 configs

Compared with the previous run (2026-08-26 06:00):

| change | count | detail |
|---|---|---|
| new source files | 2 | `logs/traf_4024.log`, `logs/traf_4025.log` |

## 2026-08-26 06:00 — 4976 results / 2188 configs

Compared with the previous run (2026-08-25 06:00):

| change | count | detail |
|---|---|---|
| new source files | 7 | `logs/ae_variants.log`, `logs/diag5.log`, `logs/diag_traffic.log`, `logs/pearson_r.log`, `logs/traf_4021.log` … |

## 2026-08-25 06:00 — 4976 results / 2188 configs

Compared with the previous run (2026-08-24 11:00):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | traffic (1) |
| new source files | 3 | `logs/ae_ratio_diag.log`, `logs/ae_ratio_elec_traffic.log`, `logs/ae_ratio_ettm.log` |
| source files appended to | 4 | `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4024.log`, `logs/elec_traffic_seeds/worker2.log`, `logs/regen_20260824_110028.log`, `result_calibration_aci_aleatoric_only_tsf.txt` |

New results by method: **ACI aleatoric only (g=0.001)** 1.

New results by seed: 4024 (1).

## 2026-08-24 11:00 — 4975 results / 2188 configs

Compared with the previous run (2026-08-24 10:24):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/regen_20260824_110028.log` |
| source files appended to | 4 | `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4023.log`, `logs/elec_traffic_seeds/worker0.log`, `logs/elec_traffic_seeds/worker1.log`, `logs/regen_20260824_102441.log` |

## 2026-08-24 10:24 — 4975 results / 2188 configs

Compared with the previous run (2026-08-24 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | traffic (1) |
| new source files | 1 | `logs/regen_20260824_102441.log` |
| source files appended to | 2 | `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4024.log`, `result_calibration_aci_cpvs_tsf.txt` |

New results by method: **ACI CP-VS (g=0.001)** 1.

New results by seed: 4024 (1).

## 2026-08-24 06:00 — 4974 results / 2188 configs

Compared with the previous run (2026-08-23 08:14):

| change | count | detail |
|---|---|---|
| new calibration results | **12** | traffic (12) |
| new training configs | **3** | configs never seen before |
| source files appended to | 17 | `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4022.log`, `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4023.log`, `logs/elec_traffic_seeds/traffic_test_ne1_MOG_seed4024.log`, `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4023.log`, `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4024.log` … |

New results by method: **ACI aleatoric only (g=0.001)** 3, **ACI CP (g=0.001)** 3, **ACI CP-VS (g=0.001)** 3, **ACI CQR quantile (g=0.001)** 2, **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4022 (1), 4023 (3), 4024 (5), 4025 (3).

## 2026-08-23 08:14 — 4962 results / 2185 configs

Compared with the previous run (2026-08-23 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | traffic (2) |
| new training configs | **1** | configs never seen before |
| new source files | 3 | `logs/elec_traffic_seeds/worker10.log`, `logs/elec_traffic_seeds/worker9_mopup.log`, `logs/regen_20260823_081410.log` |
| source files appended to | 7 | `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4022.log`, `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4023.log`, `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4023.log`, `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4024.log`, `logs/elec_traffic_seeds/traffic_test_ne3_MOG_seed4025.log` … |

New results by method: **ACI CP (g=0.001)** 1, **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4023 (1), 4025 (1).

## 2026-08-23 06:00 — 4960 results / 2184 configs

Compared with the previous run (2026-08-22 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **34** | traffic (34) |
| new training configs | **16** | configs never seen before |
| results that moved | 2 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 16 | `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4022.log`, `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4023.log`, `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4024.log`, `logs/elec_traffic_seeds/traffic_cqr_ne3_MOE_seed4025.log`, `logs/elec_traffic_seeds/traffic_test_ne1_MOE_seed4022.log` … |
| source files appended to | 16 | `logs/elec_traffic_seeds/electricity_cqr_ne1_MOE_seed4022.log`, `logs/elec_traffic_seeds/electricity_cqr_ne1_MOE_seed4023.log`, `logs/elec_traffic_seeds/traffic_cqr_ne1_MOE_seed4022.log`, `logs/elec_traffic_seeds/traffic_cqr_ne1_MOE_seed4023.log`, `logs/elec_traffic_seeds/traffic_cqr_ne1_MOE_seed4024.log` … |

New results by method: **ACI CP (g=0.001)** 8, **ACI CQR quantile (g=0.001)** 6, **ACI CQR retrain (g=0.001)** 6, **ACI aleatoric scale (g=0.001)** 6, **ACI aleatoric only (g=0.001)** 4, **ACI CP-VS (g=0.001)** 4.

New results by seed: 4022 (11), 4023 (8), 4024 (6), 4025 (9).

## 2026-08-22 06:00 — 4926 results / 2168 configs

Compared with the previous run (2026-08-19 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **16** | electricity (16) |
| new training configs | **8** | configs never seen before |
| new source files | 20 | `logs/build_results_tex_20260819_145241.log`, `logs/build_results_tex_20260819_145335.log`, `logs/elec_traffic_seeds/electricity_cqr_ne1_MOE_seed4022.log`, `logs/elec_traffic_seeds/electricity_cqr_ne1_MOE_seed4023.log`, `logs/elec_traffic_seeds/electricity_cqr_ne1_MOE_seed4024.log` … |
| source files appended to | 2 | `result_calibration_aci_cqr_quantile_tsf.txt`, `result_calibration_aci_cqr_retrain_tsf.txt` |

New results by method: **ACI CQR quantile (g=0.001)** 8, **ACI CQR retrain (g=0.001)** 8.

New results by seed: 4022 (4), 4023 (4), 4024 (4), 4025 (4).

## 2026-08-19 06:00 — 4910 results / 2160 configs

No change since the previous run (2026-08-18 06:00). No new logs, no new results, no value moved.

## 2026-08-18 06:00 — 4910 results / 2160 configs

No change since the previous run (2026-08-17 06:00). No new logs, no new results, no value moved.

## 2026-08-17 06:00 — 4910 results / 2160 configs

Compared with the previous run (2026-08-16 15:27):

| change | count | detail |
|---|---|---|
| source files appended to | 2 | `logs/chain/chain_20260814_185716.log`, `logs/chain_launch_20260814_185716.log` |

## 2026-08-16 15:27 — 4910 results / 2160 configs

Compared with the previous run (2026-08-16 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **11** | electricity (11) |
| new training configs | **4** | configs never seen before |
| new source files | 4 | `logs/chain/elec_ne1moe_s4024_20260814_185716.log`, `logs/chain/elec_ne1moe_s4025_20260814_185716.log`, `logs/chain/elec_ne1mog_s4025_20260814_185716.log`, `logs/chain/elec_ne3mog_s4025_20260814_185716.log` |
| source files appended to | 7 | `logs/chain/chain_20260814_185716.log`, `logs/chain/elec_ne1mog_s4024_20260814_185716.log`, `logs/chain_launch_20260814_185716.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` … |

New results by method: **ACI CP (g=0.001)** 4, **ACI aleatoric only (g=0.001)** 3, **ACI aleatoric scale (g=0.001)** 2, **ACI CP-VS (g=0.001)** 2.

New results by seed: 4024 (2), 4025 (9).

## 2026-08-16 06:00 — 4899 results / 2156 configs

Compared with the previous run (2026-08-15 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **24** | electricity (24) |
| new training configs | **7** | configs never seen before |
| new source files | 7 | `logs/chain/elec_ne1moe_s4022_20260814_185716.log`, `logs/chain/elec_ne1moe_s4023_20260814_185716.log`, `logs/chain/elec_ne1mog_s4022_20260814_185716.log`, `logs/chain/elec_ne1mog_s4023_20260814_185716.log`, `logs/chain/elec_ne1mog_s4024_20260814_185716.log` … |
| source files appended to | 7 | `logs/chain/chain_20260814_185716.log`, `logs/chain/elec_ne3mog_s4022_20260814_185716.log`, `logs/chain_launch_20260814_185716.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` … |

New results by method: **ACI CP (g=0.001)** 8, **ACI CP-VS (g=0.001)** 6, **ACI aleatoric only (g=0.001)** 5, **ACI aleatoric scale (g=0.001)** 5.

New results by seed: 4022 (8), 4023 (9), 4024 (7).

## 2026-08-15 06:00 — 4875 results / 2149 configs

Compared with the previous run (2026-08-14 23:46):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | electricity (1) |
| new training configs | **1** | configs never seen before |
| new source files | 1 | `logs/chain/elec_ne3mog_s4022_20260814_185716.log` |
| source files appended to | 3 | `logs/chain/chain_20260814_185716.log`, `logs/chain_launch_20260814_185716.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` |

New results by method: **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4022 (1).

## 2026-08-14 23:46 — 4874 results / 2148 configs

Compared with the previous run (2026-08-14 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **192** | ETTh1 (48), ETTh2 (48), ETTm1 (48), ETTm2 (48) |
| new training configs | **12** | configs never seen before |
| results that moved | 6 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 119 | `logs/aci_moecp_multiseed/etth1_s4021_20260814_162850.log`, `logs/aci_moecp_multiseed/etth1_s4022_20260814_162850.log`, `logs/aci_moecp_multiseed/etth1_s4023_20260814_162850.log`, `logs/aci_moecp_multiseed/etth1_s4024_20260814_162850.log`, `logs/aci_moecp_multiseed/etth1_s4025_20260814_162850.log` … |
| source files appended to | 7 | `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt`, `result_calibration_aci_cp_tsf.txt`, `result_calibration_aci_cpvs_tsf.txt`, `result_calibration_aci_cqr_quantile_tsf.txt` … |

New results by method: **ACI CP (g=0.001)** 48, **ACI aleatoric only (g=0.001)** 32, **ACI aleatoric scale (g=0.001)** 32, **ACI CP-VS (g=0.001)** 32, **ACI CQR quantile (g=0.001)** 16, **ACI CQR retrain (g=0.001)** 16, **ACI MoECP (g=0.001)** 16.

New results by seed: 4022 (48), 4023 (48), 4024 (48), 4025 (48).

## 2026-08-14 06:00 — 4682 results / 2136 configs

No change since the previous run (2026-08-13 17:42). No new logs, no new results, no value moved.

## 2026-08-13 17:42 — 4682 results / 2136 configs

No change since the previous run (2026-08-13 16:18). No new logs, no new results, no value moved.

## 2026-08-13 16:18 — 4682 results / 2136 configs

Compared with the previous run (2026-08-13 12:30):

| change | count | detail |
|---|---|---|
| new calibration results | **8** | ETTm2 (6), ETTm1 (1), traffic (1) |
| new source files | 4 | `logs/aci_cqr_sweep_backfill/ettm2_ne2_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/ettm2_ne4_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/ettm2_ne5_aci_cqr_20260813_090917.log`, `logs/gap_traffic_cqr_retrain_base/traffic_ne1_20260813_090917.log` |
| source files appended to | 5 | `logs/aci_cqr_sweep_backfill/ettm1_ne5_aci_cqr_20260813_090917.log`, `logs/gap_backfill_driver_20260813_090917.log`, `result_calibration_aci_cqr_quantile_tsf.txt`, `result_calibration_aci_cqr_retrain_tsf.txt`, `result_calibration_cqr_retrain.txt` |

New results by method: **ACI CQR retrain (g=0.001)** 4, **ACI CQR quantile (g=0.001)** 3, **CQR retrain** 1.

New results by seed: 4021 (8).

## 2026-08-13 12:30 — 4674 results / 2136 configs

Compared with the previous run (2026-08-13 09:20):

| change | count | detail |
|---|---|---|
| **new calibration method** | 1 | `moecp` — first time this method has produced time-series results |
| new calibration results | **42** | ETTh2 (11), ETTm1 (10), ETTh1 (8), ETTm2 (5) |
| new source files | 38 | `logs/aci_cqr_sweep_backfill/etth1_ne5_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/etth2_ne2_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/etth2_ne4_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/etth2_ne5_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/ettm1_ne2_aci_cqr_20260813_090917.log` … |
| source files appended to | 5 | `logs/aci_cqr_sweep_backfill/etth1_ne4_aci_cqr_20260813_090917.log`, `logs/gap_backfill_driver_20260813_090917.log`, `result_calibration_aci_cqr_quantile_tsf.txt`, `result_calibration_aci_cqr_retrain_tsf.txt`, `result_calibration_moecp_tsf.txt` |

New results by method: **MoECP** 28, **ACI CQR retrain (g=0.001)** 7, **ACI CQR quantile (g=0.001)** 7.

New results by seed: 4021 (42).

## 2026-08-13 09:20 — 4632 results / 2136 configs

Compared with the previous run (2026-08-13 09:19):

| change | count | detail |
|---|---|---|
| results that disappeared | 115 | source file truncated, rotated or deleted |
| source files gone | 1 | `result_calibration_moecp_preclip.removed_20260813.txt` |

## 2026-08-13 09:19 — 4747 results / 2136 configs

Compared with the previous run (2026-08-13 08:22):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | ETTh1 (3) |
| new source files | 5 | `logs/aci_cqr_sweep_backfill/etth1_ne2_aci_cqr_20260813_090917.log`, `logs/aci_cqr_sweep_backfill/etth1_ne4_aci_cqr_20260813_090917.log`, `logs/gap_backfill_driver_20260813_090751.log`, `logs/gap_backfill_driver_20260813_090917.log`, `result_calibration_moecp_preclip.removed_20260813.txt` |
| source files appended to | 3 | `result_calibration_aci_cqr_quantile_tsf.txt`, `result_calibration_aci_cqr_retrain_tsf.txt`, `result_calibration_moecp_tsf.txt` |

New results by method: **ACI CQR quantile (g=0.001)** 2, **ACI CQR retrain (g=0.001)** 1.

New results by seed: 4021 (3).

## 2026-08-13 08:22 — 4744 results / 2136 configs

No change since the previous run (2026-08-13 08:15). No new logs, no new results, no value moved.

## 2026-08-13 08:15 — 4744 results / 2136 configs

No change since the previous run (2026-08-13 06:00). No new logs, no new results, no value moved.

## 2026-08-13 06:00 — 4744 results / 2136 configs

Compared with the previous run (2026-08-12 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **4** | electricity (2), traffic (2) |
| new training configs | **2** | configs never seen before |
| new source files | 6 | `logs/aci_ne1_moe/electricity_20260812_144516.log`, `logs/aci_ne1_moe/traffic_20260812_144516.log`, `logs/aci_ne1_moe_elec_traffic_driver_20260812_144516.log`, `logs/gap_seed4021_ne1_moe/electricity_moe_train_20260812_102642.log`, `logs/gap_seed4021_ne1_moe/traffic_moe_train_20260812_102642.log` … |
| source files appended to | 2 | `result_calibration_aci_cp_tsf.txt`, `result_calibration_mse_cp.txt` |

New results by method: **ACI CP (g=0.001)** 2, **Standard CP** 2.

New results by seed: 4021 (4).

## 2026-08-12 06:00 — 4740 results / 2134 configs

No change since the previous run (2026-08-11 21:38). No new logs, no new results, no value moved.

## 2026-08-11 21:38 — 4740 results / 2134 configs

Compared with the previous run (2026-08-11 20:22):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | traffic (2) |
| source files appended to | 6 | `logs/aci_cqr_retrain_all/ne1_traffic_20260811_181503.log`, `logs/aci_cqr_retrain_all/ne3_traffic_20260811_181456.log`, `logs/aci_cqr_retrain_ne1traffic_driver_20260811_181503.log`, `logs/aci_cqr_retrain_ne3traffic_driver_20260811_181456.log`, `logs/collect_calibration_results_20260811_202246.log` … |

New results by method: **ACI CQR retrain (g=0.001)** 2.

New results by seed: 4021 (2).

## 2026-08-11 20:22 — 4738 results / 2134 configs

Compared with the previous run (2026-08-11 18:07):

| change | count | detail |
|---|---|---|
| new calibration results | **37** | ETTh1 (9), ETTh2 (9), ETTm1 (9), ETTm2 (9) |
| new source files | 19 | `logs/aci_cqr_retrain_all/ne1_traffic_20260811_181503.log`, `logs/aci_cqr_retrain_all/ne3_traffic_20260811_140220.log`, `logs/aci_cqr_retrain_all/ne3_traffic_20260811_181456.log`, `logs/aci_cqr_retrain_ne1traffic_driver_20260811_181503.log`, `logs/aci_cqr_retrain_ne3traffic_driver_20260811_181456.log` … |
| source files appended to | 7 | `logs/aci_cqr_retrain_all/ne1_electricity_20260811_140220.log`, `logs/aci_cqr_retrain_all_driver_20260811_140220.log`, `logs/collector_20260811_180750.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_cp_tsf.txt` … |

New results by method: **ACI aleatoric only (g=0.001)** 12, **ACI CP (g=0.001)** 12, **ACI CP-VS (g=0.001)** 12, **ACI CQR retrain (g=0.001)** 1.

New results by seed: 4021 (37).

## 2026-08-11 18:07 — 4701 results / 2134 configs

Compared with the previous run (2026-08-11 16:27):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | electricity (1), traffic (1) |
| new source files | 2 | `logs/aci_cqr_retrain_all/ne1_electricity_20260811_140220.log`, `logs/collector_20260811_180750.log` |
| source files appended to | 6 | `logs/aci_cqr_retrain_all/ne3_electricity_20260811_140220.log`, `logs/aci_cqr_retrain_all_driver_20260811_140220.log`, `logs/aci_ne1_mog/traffic_cqr_20260811_162033.log`, `logs/aci_ne1_resume2_driver_20260811_102410.log`, `result_calibration_aci_cqr_quantile_tsf.txt` … |

New results by method: **ACI CQR retrain (g=0.001)** 1, **ACI CQR quantile (g=0.001)** 1.

New results by seed: 4021 (2).

## 2026-08-11 16:27 — 4699 results / 2134 configs

Compared with the previous run (2026-08-11 10:23):

| change | count | detail |
|---|---|---|
| new calibration results | **12** | traffic (5), ETTh1 (1), ETTh2 (1), ETTm1 (1) |
| new source files | 15 | `logs/aci_cqr_retrain_all/ne1_etth1_20260811_140220.log`, `logs/aci_cqr_retrain_all/ne1_etth2_20260811_140220.log`, `logs/aci_cqr_retrain_all/ne1_ettm1_20260811_140220.log`, `logs/aci_cqr_retrain_all/ne1_ettm2_20260811_140220.log`, `logs/aci_cqr_retrain_all/ne1_exchange_20260811_140220.log` … |
| source files appended to | 6 | `logs/collector_resync2_20260811_102326.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_cp_tsf.txt`, `result_calibration_aci_cpvs_tsf.txt`, `result_calibration_aci_cqr_quantile_tsf.txt` … |

New results by method: **ACI CQR retrain (g=0.001)** 7, **ACI aleatoric only (g=0.001)** 2, **ACI CQR quantile (g=0.001)** 1, **ACI CP (g=0.001)** 1, **ACI CP-VS (g=0.001)** 1.

New results by seed: 4021 (12).

## 2026-08-11 10:23 — 4687 results / 2134 configs

Compared with the previous run (2026-08-11 06:00):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/collector_resync2_20260811_102326.log` |

## 2026-08-11 06:00 — 4687 results / 2134 configs

Compared with the previous run (2026-08-10 18:38):

| change | count | detail |
|---|---|---|
| new calibration results | **40** | electricity (6), ETTh2 (5), ETTm1 (5), ETTm2 (5) |
| new source files | 22 | `logs/aci_ne1_mog/electricity_cqr_20260810_183846.log`, `logs/aci_ne1_mog/electricity_mog_20260810_183846.log`, `logs/aci_ne1_mog/etth1_cqr_20260810_183846.log`, `logs/aci_ne1_mog/etth1_mog_20260810_183846.log`, `logs/aci_ne1_mog/etth2_cqr_20260810_183846.log` … |
| source files appended to | 6 | `logs/collector_resync_20260810_183808.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_cp_tsf.txt`, `result_calibration_aci_cpvs_tsf.txt`, `result_calibration_aci_cqr_quantile_tsf.txt` … |

New results by method: **ACI CQR quantile (g=0.001)** 9, **ACI aleatoric only (g=0.001)** 8, **ACI CP (g=0.001)** 8, **ACI CP-VS (g=0.001)** 8, **ACI MoECP (g=0.001)** 7.

New results by seed: 4021 (40).

## 2026-08-10 18:38 — 4647 results / 2134 configs

Compared with the previous run (2026-08-10 13:11):

| change | count | detail |
|---|---|---|
| new calibration results | **4** | ETTh1 (3), electricity (1) |
| new source files | 3 | `logs/aci_ne1_mog/etth1_mog_20260810_140655.log`, `logs/aci_ne1_mog_others_driver_20260810_140655.log`, `logs/collector_resync_20260810_183808.log` |
| source files appended to | 5 | `logs/aci_ne3_elec_traffic/electricity_mog_20260810_114437.log`, `logs/collector_check_20260810_131120.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_cp_tsf.txt`, `result_calibration_aci_cpvs_tsf.txt` |

New results by method: **ACI CP-VS (g=0.001)** 2, **ACI aleatoric only (g=0.001)** 1, **ACI CP (g=0.001)** 1.

New results by seed: 4021 (4).

## 2026-08-10 13:11 — 4643 results / 2134 configs

Compared with the previous run (2026-08-10 09:24):

| change | count | detail |
|---|---|---|
| new calibration results | **8** | ETTh1 (1), ETTh2 (1), ETTm1 (1), ETTm2 (1) |
| new source files | 11 | `logs/aci_ne1_moe/etth1_20260810_114441.log`, `logs/aci_ne1_moe/etth2_20260810_114441.log`, `logs/aci_ne1_moe/ettm1_20260810_114441.log`, `logs/aci_ne1_moe/ettm2_20260810_114441.log`, `logs/aci_ne1_moe/exchange_20260810_114441.log` … |
| source files appended to | 2 | `logs/collector_final_20260810_092401.log`, `result_calibration_aci_cp_tsf.txt` |

New results by method: **ACI CP (g=0.001)** 8.

New results by seed: 4021 (8).

## 2026-08-10 09:24 — 4635 results / 2134 configs

Compared with the previous run (2026-08-10 08:32):

| change | count | detail |
|---|---|---|
| new calibration results | **6** | weather (6) |
| new source files | 1 | `logs/collector_final_20260810_092401.log` |
| source files appended to | 8 | `logs/aci_cqr_ne3/weather_20260810_083219.log`, `logs/aci_mog_ne3/weather_20260810_083148.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_cp_tsf.txt`, `result_calibration_aci_cpvs_tsf.txt` … |

New results by method: **ACI CQR quantile (g=0.001)** 1, **ACI CQR retrain (g=0.001)** 1, **ACI aleatoric only (g=0.001)** 1, **ACI CP (g=0.001)** 1, **ACI CP-VS (g=0.001)** 1, **ACI MoECP (g=0.001)** 1.

New results by seed: 4021 (6).

## 2026-08-10 08:32 — 4629 results / 2134 configs

Compared with the previous run (2026-08-10 06:00):

| change | count | detail |
|---|---|---|
| new source files | 3 | `logs/aci_cqr_ne3/weather_20260810_083219.log`, `logs/aci_mog_ne3/weather_20260810_083148.log`, `logs/plot_width_20260810_082835.log` |

## 2026-08-10 06:00 — 4629 results / 2134 configs

Compared with the previous run (2026-08-10 01:40):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/collector_20260810_014053.log` |

## 2026-08-10 01:40 — 4629 results / 2134 configs

Compared with the previous run (2026-08-10 01:40):

| change | count | detail |
|---|---|---|
| results that disappeared | 2 | source file truncated, rotated or deleted |
| new source files | 1 | `logs/collector_20260810_014053.log` |
| source files appended to | 3 | `logs/collector_20260810_014018.log`, `result_calibration_aci_cqr_quantile_tsf.txt`, `result_calibration_aci_cqr_retrain_tsf.txt` |

## 2026-08-10 01:40 — 4631 results / 2134 configs

Compared with the previous run (2026-08-10 01:01):

| change | count | detail |
|---|---|---|
| new calibration results | **5** | exchange-rate (2), national-illness (2), ETTm2 (1) |
| new source files | 3 | `logs/aci_cqr_ne3/exchange_20260810_000912.log`, `logs/aci_cqr_ne3/illness_20260810_000912.log`, `logs/collector_20260810_014018.log` |
| source files appended to | 5 | `logs/aci_cqr_driver_20260810_000912.log`, `logs/aci_cqr_ne3/ettm2_20260810_000912.log`, `logs/collector_20260810_010127.log`, `result_calibration_aci_cqr_quantile_tsf.txt`, `result_calibration_aci_cqr_retrain_tsf.txt` |

New results by method: **ACI CQR retrain (g=0.001)** 3, **ACI CQR quantile (g=0.001)** 2.

New results by seed: 4021 (5).

## 2026-08-10 01:01 — 4626 results / 2134 configs

Compared with the previous run (2026-08-10 01:00):

| change | count | detail |
|---|---|---|
| results that moved | 3 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 1 | `logs/collector_20260810_010127.log` |
| source files appended to | 1 | `logs/collector_20260810_010021.log` |
| source files gone | 1 | `result_calibration_aci_gamma0_verify.removed.txt` |

## 2026-08-10 01:00 — 4626 results / 2134 configs

Compared with the previous run (2026-08-10 00:59):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/collector_20260810_010021.log` |
| source files appended to | 1 | `logs/collector_20260810_005940.log` |
| source files gone | 7 | `logs/aci_gamma0_verify/etth1_acialeatoric_only_g0_20260809_235951.log`, `logs/aci_gamma0_verify/etth1_acicp_g0_20260809_235611.log`, `logs/aci_gamma0_verify/etth1_acicpvs_g0_20260809_235951.log`, `logs/aci_gamma0_verify/etth1_acicqr_g0_20260810_000153.log`, `logs/aci_gamma0_verify/etth1_acicqrretrain_g0_20260810_000304.log` |

## 2026-08-10 00:59 — 4626 results / 2134 configs

Compared with the previous run (2026-08-10 00:58):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| results that moved | 3 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 2 | `logs/collector_20260810_005940.log`, `result_calibration_aci_gamma0_verify.removed.txt` |
| source files appended to | 7 | `logs/aci_cqr_ne3/ettm2_20260810_000912.log`, `logs/collector_20260810_005828.log`, `result_calibration_aci_aleatoric_only_tsf.txt`, `result_calibration_aci_cp_tsf.txt`, `result_calibration_aci_cpvs_tsf.txt` … |

New results by method: **ACI CQR quantile (g=0.001)** 1.

New results by seed: 4021 (1).

## 2026-08-10 00:58 — 4625 results / 2134 configs

Compared with the previous run (2026-08-09 23:54):

| change | count | detail |
|---|---|---|
| **new calibration method** | 6 | `aci_aleatoric_only`, `aci_cp`, `aci_cpvs`, `aci_cqr_quantile`, `aci_cqr_retrain`, `aci_moecp` — first time this method has produced time-series results |
| new calibration results | **32** | ETTh1 (8), ETTh2 (6), ETTm1 (6), ETTm2 (4) |
| new source files | 26 | `logs/aci_cqr_driver_20260810_000912.log`, `logs/aci_cqr_ne3/etth1_20260810_000912.log`, `logs/aci_cqr_ne3/etth2_20260810_000912.log`, `logs/aci_cqr_ne3/ettm1_20260810_000912.log`, `logs/aci_cqr_ne3/ettm2_20260810_000912.log` … |
| source files appended to | 1 | `logs/collector_aci_check_20260809_235445.log` |

New results by method: **ACI aleatoric only (g=0.001)** 6, **ACI CP (g=0.001)** 6, **ACI CP-VS (g=0.001)** 6, **ACI MoECP (g=0.001)** 6, **ACI CQR quantile (g=0.001)** 4, **ACI CQR retrain (g=0.001)** 4.

New results by seed: 4021 (32).

## 2026-08-09 23:54 — 4593 results / 2134 configs

Compared with the previous run (2026-08-09 19:16):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/collector_aci_check_20260809_235445.log` |

## 2026-08-09 19:16 — 4593 results / 2134 configs

Compared with the previous run (2026-08-09 18:35):

| change | count | detail |
|---|---|---|
| new calibration results | **6** | ETTh1 (1), ETTh2 (1), ETTm1 (1), ETTm2 (1) |
| new source files | 7 | `logs/moecp_classicmoe_ne3_seed4021/etth1_20260809_184451.log`, `logs/moecp_classicmoe_ne3_seed4021/etth2_20260809_184451.log`, `logs/moecp_classicmoe_ne3_seed4021/ettm1_20260809_184451.log`, `logs/moecp_classicmoe_ne3_seed4021/ettm2_20260809_184451.log`, `logs/moecp_classicmoe_ne3_seed4021/exchange_20260809_184451.log` … |
| source files appended to | 3 | `logs/grid_launcher_20260806_163120.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_moecp_tsf.txt` |

New results by method: **MoECP** 6.

New results by seed: 4021 (6).

## 2026-08-09 18:35 — 4587 results / 2134 configs

Compared with the previous run (2026-08-09 18:35):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-09 18:35 — 4587 results / 2134 configs

Compared with the previous run (2026-08-09 17:04):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | electricity (1), ETTm1 (1), ETTh2 (1) |
| new source files | 3 | `logs/gap_seed4021_ne1_moe/electricity_moe_20260809_161726.log`, `logs/gap_seed4021_ne1_moe/traffic_cqr_retrain_20260809_161726.log`, `logs/gap_seed4021_ne1_moe/traffic_moe_20260809_161726.log` |
| source files appended to | 6 | `logs/gap_seed4021_ne1_moe/electricity_cqr_retrain_20260809_161726.log`, `logs/gap_seed4021_ne1_moe_driver_20260809_161726.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_cqr_retrain.txt` … |

New results by method: **MoECP** 2, **CQR retrain** 1.

New results by seed: 4021 (2), 4023 (1).

## 2026-08-09 17:04 — 4584 results / 2134 configs

Compared with the previous run (2026-08-09 16:50):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm1 (1), ETTh2 (1) |
| results that disappeared | 1164 | source file truncated, rotated or deleted |
| source files appended to | 5 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt`, `result_calibration_cp_dvs.txt` |

New results by method: **Adaptive window CP** 2.

New results by seed: 4021 (1), 4023 (1).

## 2026-08-09 16:50 — 5746 results / 2176 configs

Compared with the previous run (2026-08-09 16:43):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm1 (1), ETTh2 (1) |
| new source files | 1 | `result_calibration_moecp_nontau1.removed.txt` |
| source files appended to | 5 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_aleatoric_scale_tsf.txt`, `result_calibration_moecp_tsf.txt` |

New results by method: **Aleatoric scale CP** 1, **Adaptive variance-ratio** 1.

New results by seed: 4021 (1), 4023 (1).

## 2026-08-09 16:43 — 5744 results / 2176 configs

Compared with the previous run (2026-08-09 16:15):

| change | count | detail |
|---|---|---|
| new calibration results | **17** | ETTh2 (7), ETTm1 (5), ETTh1 (2), exchange-rate (1) |
| new training configs | **4** | configs never seen before |
| results that disappeared | 78 | source file truncated, rotated or deleted |
| new source files | 7 | `logs/build_ett_expert_sweep_20260809_161709.log`, `logs/gap_seed4021_ne1_moe/electricity_cqr_retrain_20260809_161726.log`, `logs/gap_seed4021_ne1_moe/exchange_cqr_retrain_20260809_161726.log`, `logs/gap_seed4021_ne1_moe/illness_cqr_retrain_20260809_161726.log`, `logs/gap_seed4021_ne1_moe/weather_cqr_retrain_20260809_161726.log` … |
| source files appended to | 11 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_mog.txt`, `result_calibration_aleatoric_mog_v2.txt`, `result_calibration_aleatoric_only.txt` … |

New results by method: **CQR retrain** 4, **Standard CP** 3, **Aleatoric MoG** 2, **Aleatoric MoG v2** 2, **Aleatoric only** 2, **Adaptive CPVS** 2, **CQR quantile** 1, **Aleatoric scale CP** 1.

New results by seed: 4021 (8), 4022 (2), 4023 (7).

## 2026-08-09 16:15 — 5805 results / 2172 configs

Compared with the previous run (2026-08-09 12:12):

| change | count | detail |
|---|---|---|
| new calibration results | **73** | ETTh1 (61), ETTm1 (12) |
| new training configs | **14** | configs never seen before |
| results that moved | 5 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 1 | `logs/build_ett_expert_sweep_xlsx_20260809_161346.log` |
| source files appended to | 15 | `logs/collect_calibration_20260809_121247.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt` … |

New results by method: **Standard CP** 10, **MoECP** 7, **Adaptive variance-ratio** 6, **Adaptive Window Cp** 6, **Aleatoric MoG** 6, **Aleatoric MoG v2** 6, **Aleatoric only** 6, **Aleatoric scale CP** 6, **CP-DVS** 6, **CQR quantile** 5, **CQR retrain** 5, **Adaptive CPVS** 4.

New results by seed: 4021 (9), 4022 (64).

## 2026-08-09 12:12 — 5732 results / 2158 configs

Compared with the previous run (2026-08-09 12:12):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/collect_calibration_20260809_121247.log` |
| source files appended to | 2 | `logs/collect_calibration_20260809_121204.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-09 12:12 — 5732 results / 2158 configs

Compared with the previous run (2026-08-09 11:44):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/collect_calibration_20260809_121204.log` |
| source files appended to | 2 | `logs/collect_calibration_20260809_114431.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-09 11:44 — 5732 results / 2158 configs

Compared with the previous run (2026-08-09 07:44):

| change | count | detail |
|---|---|---|
| new calibration results | **43** | ETTm1 (39), ETTm2 (4) |
| new training configs | **5** | configs never seen before |
| results that moved | 3 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 1 | `logs/collect_calibration_20260809_114431.log` |
| source files appended to | 14 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt`, `result_calibration_aleatoric_mog.txt` … |

New results by method: **Adaptive variance-ratio** 5, **Adaptive Window Cp** 5, **CP-DVS** 5, **Aleatoric MoG** 4, **Aleatoric MoG v2** 4, **Aleatoric only** 4, **Aleatoric scale CP** 4, **MoECP** 4, **Standard CP** 4, **Adaptive CPVS** 2, **CQR quantile** 1, **CQR retrain** 1.

New results by seed: 4021 (23), 4022 (20).

## 2026-08-09 07:44 — 5689 results / 2153 configs

Compared with the previous run (2026-08-09 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **12** | ETTm1 (6), ETTm2 (6) |
| new training configs | **4** | configs never seen before |
| source files appended to | 11 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_mog.txt`, `result_calibration_aleatoric_mog_v2.txt`, `result_calibration_aleatoric_only.txt` … |

New results by method: **Standard CP** 3, **Adaptive CPVS** 2, **CQR quantile** 1, **CQR retrain** 1, **Aleatoric MoG** 1, **Aleatoric MoG v2** 1, **Aleatoric only** 1, **Aleatoric scale CP** 1, **MoECP** 1.

New results by seed: 4021 (6), 4022 (6).

## 2026-08-09 06:00 — 5677 results / 2149 configs

Compared with the previous run (2026-08-08 06:00):

| change | count | detail |
|---|---|---|
| new calibration results | **509** | ETTm2 (102), ETTh2 (91), exchange-rate (76), national-illness (75) |
| new training configs | **82** | configs never seen before |
| results that moved | 29 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 27 | `logs/campaign/calib_traffic_ne3_MOGU.log`, `logs/campaign/train_exchange-rate_ne2_MOE.log`, `logs/campaign/train_exchange-rate_ne2_MOG.log`, `logs/campaign/train_exchange-rate_ne2_MOGU.log`, `logs/campaign/train_exchange-rate_ne4_MOE.log` … |
| source files appended to | 22 | `logs/campaign/calib_electricity_ne3_MOGU.log`, `logs/campaign/calib_traffic_ne3_MOG.log`, `logs/campaign/worker0.log`, `logs/campaign/worker1.log`, `logs/gap_seed4021_ne1/traffic_mog_20260807_184946.log` … |

New results by method: **Standard CP** 62, **Adaptive Window Cp** 45, **CP-DVS** 45, **MoECP** 44, **Adaptive variance-ratio** 43, **Aleatoric MoG v2** 42, **Aleatoric scale CP** 42, **Aleatoric MoG** 40, **Aleatoric only** 40, **Adaptive CPVS** 32, **CQR quantile** 23, **CQR retrain** 19, **ACI aleatoric scale (g=0.01)** 16, **ACI aleatoric scale (g=0.001)** 16.

New results by seed: 4021 (370), 4022 (139).

## 2026-08-08 06:00 — 5168 results / 2067 configs

Compared with the previous run (2026-08-07 19:27):

| change | count | detail |
|---|---|---|
| new calibration results | **296** | ETTm2 (85), ETTm1 (76), ETTh1 (56), ETTh2 (21) |
| new training configs | **34** | configs never seen before |
| results that moved | 29 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 30 | `logs/campaign/calib_ETTm1_ne5_MOE.log`, `logs/campaign/calib_ETTm1_ne5_MOG.log`, `logs/campaign/calib_ETTm1_ne5_MOGU.log`, `logs/campaign/calib_ETTm2_ne1_MOE.log`, `logs/campaign/calib_ETTm2_ne2_MOE.log` … |
| source files appended to | 22 | `logs/campaign/calib_ETTm1_ne4_MOG.log`, `logs/campaign/calib_ETTm1_ne4_MOGU.log`, `logs/campaign/worker0.log`, `logs/campaign/worker1.log`, `logs/gap_seed4021_ne1/electricity_mog_20260807_184946.log` … |

New results by method: **Aleatoric MoG v2** 31, **CP-DVS** 31, **MoECP** 30, **Adaptive variance-ratio** 27, **Adaptive Window Cp** 25, **Aleatoric scale CP** 25, **Aleatoric MoG** 22, **Standard CP** 22, **Aleatoric only** 19, **ACI aleatoric scale (g=0.01)** 14, **ACI aleatoric scale (g=0.001)** 14, **CQR quantile** 12, **CQR retrain** 12, **Adaptive CPVS** 12.

New results by seed: 4021 (275), 4022 (21).

## 2026-08-07 19:27 — 4872 results / 2033 configs

Compared with the previous run (2026-08-07 18:54):

| change | count | detail |
|---|---|---|
| new calibration results | **29** | ETTm1 (21), weather (6), ETTm2 (1), synth-moe3 (1) |
| new training configs | **1** | configs never seen before |
| new source files | 4 | `logs/campaign/calib_ETTm1_ne4_MOE.log`, `logs/campaign/calib_ETTm1_ne4_MOG.log`, `logs/campaign/calib_ETTm1_ne4_MOGU.log`, `logs/gap_seed4021_ne1/electricity_mog_20260807_184946.log` |
| source files appended to | 23 | `logs/campaign/calib_ETTm1_ne2_MOGU.log`, `logs/campaign/calib_ETTm1_ne3_MOGU.log`, `logs/campaign/worker0.log`, `logs/campaign/worker1.log`, `logs/gap_seed4021_ne1/weather_mog_20260807_184946.log` … |

New results by method: **ACI aleatoric scale (g=0.001)** 4, **CQR retrain** 3, **CP-DVS** 3, **ACI aleatoric scale (g=0.01)** 3, **Adaptive variance-ratio** 2, **Adaptive Window Cp** 2, **MoECP** 2, **Aleatoric MoG** 2, **Aleatoric MoG v2** 2, **Aleatoric only** 2, **Aleatoric scale CP** 2, **Adaptive CPVS** 1, **Standard CP** 1.

New results by seed: 4021 (29).

## 2026-08-07 18:54 — 4843 results / 2032 configs

Compared with the previous run (2026-08-07 18:48):

| change | count | detail |
|---|---|---|
| new calibration results | **15** | ETTm1 (5), exchange-rate (4), ETTm2 (3), national-illness (2) |
| new training configs | **3** | configs never seen before |
| results that moved | 1 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 6 | `logs/gap_seed4021_ne1/ettm2_mog_20260807_184946.log`, `logs/gap_seed4021_ne1/exchange_mog_20260807_184946.log`, `logs/gap_seed4021_ne1/illness_mog_20260807_184946.log`, `logs/gap_seed4021_ne1/weather_mog_20260807_184946.log`, `logs/gap_seed4021_ne1_driver.log` … |
| source files appended to | 11 | `logs/campaign/calib_ETTm1_ne2_MOGU.log`, `logs/campaign/calib_ETTm1_ne3_MOGU.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `logs/synth_moe3/stage2_cqr_ne3_seed4021_20260807_174008.log` … |

New results by method: **ACI aleatoric scale (g=0.01)** 5, **ACI aleatoric scale (g=0.001)** 4, **CQR quantile** 3, **Aleatoric scale CP** 2, **Aleatoric only** 1.

New results by seed: 4021 (15).

## 2026-08-07 18:48 — 4828 results / 2029 configs

Compared with the previous run (2026-08-07 18:03):

| change | count | detail |
|---|---|---|
| new calibration results | **74** | ETTh2 (43), ETTm1 (20), synth-moe3 (8), ETTm2 (3) |
| results that moved | 4 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 15 | `logs/campaign/calib_ETTh2_ne4_MOE.log`, `logs/campaign/calib_ETTh2_ne4_MOG.log`, `logs/campaign/calib_ETTh2_ne4_MOGU.log`, `logs/campaign/calib_ETTh2_ne5_MOE.log`, `logs/campaign/calib_ETTh2_ne5_MOG.log` … |
| source files appended to | 18 | `logs/campaign/calib_ETTh2_ne2_MOGU.log`, `logs/campaign/calib_ETTh2_ne3_MOGU.log`, `logs/campaign/worker0.log`, `logs/campaign/worker1.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` … |

New results by method: **CP-DVS** 11, **MoECP** 11, **Aleatoric MoG v2** 10, **ACI aleatoric scale (g=0.01)** 8, **ACI aleatoric scale (g=0.001)** 8, **Aleatoric MoG** 7, **Adaptive Window Cp** 6, **Aleatoric scale CP** 5, **Adaptive variance-ratio** 4, **Aleatoric only** 4.

New results by seed: 4021 (74).

## 2026-08-07 18:03 — 4754 results / 2029 configs

Compared with the previous run (2026-08-07 17:43):

| change | count | detail |
|---|---|---|
| new calibration results | **63** | ETTh1 (33), ETTh2 (21), synth-moe3 (4), ETTm2 (3) |
| new training configs | **2** | configs never seen before |
| results that moved | 5 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 14 | `logs/campaign/calib_ETTh1_ne4_MOGU.log`, `logs/campaign/calib_ETTh1_ne5_MOE.log`, `logs/campaign/calib_ETTh1_ne5_MOG.log`, `logs/campaign/calib_ETTh1_ne5_MOGU.log`, `logs/campaign/calib_ETTh2_ne1_MOE.log` … |
| source files appended to | 23 | `logs/campaign/calib_ETTh1_ne3_MOGU.log`, `logs/campaign/calib_ETTh1_ne4_MOG.log`, `logs/campaign/worker0.log`, `logs/campaign/worker1.log`, `logs/gap_seed4021/traffic_cqr_20260807_113754.log` … |

New results by method: **Aleatoric MoG v2** 10, **CP-DVS** 8, **ACI aleatoric scale (g=0.01)** 7, **ACI aleatoric scale (g=0.001)** 7, **Aleatoric MoG** 7, **MoECP** 6, **Adaptive variance-ratio** 5, **Aleatoric scale CP** 4, **Adaptive Window Cp** 3, **Aleatoric only** 3, **CQR quantile** 1, **Adaptive CPVS** 1, **Standard CP** 1.

New results by seed: 4021 (63).

## 2026-08-07 17:43 — 4691 results / 2027 configs

Compared with the previous run (2026-08-07 17:25):

| change | count | detail |
|---|---|---|
| new calibration results | **28** | ETTh1 (26), ETTm1 (2) |
| results that moved | 7 | same config+method, different coverage/width (model retrained or rerun) |
| results that disappeared | 5 | source file truncated, rotated or deleted |
| new source files | 22 | `logs/campaign/calib_ETTh1_ne1_MOE.log`, `logs/campaign/calib_ETTh1_ne1_MOG.log`, `logs/campaign/calib_ETTh1_ne2_MOE.log`, `logs/campaign/calib_ETTh1_ne2_MOG.log`, `logs/campaign/calib_ETTh1_ne2_MOGU.log` … |
| source files appended to | 21 | `logs/calib_gap_weather_mog_seed4021_20260807_104157.log`, `logs/gap_seed4021/exchange_mog_20260807_113754.log`, `logs/gap_seed4021/illness_mog_20260807_113754.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/gap_weather_aci/electricity_moecp_w16.log` … |

New results by method: **Aleatoric MoG v2** 5, **ACI aleatoric scale (g=0.01)** 4, **ACI aleatoric scale (g=0.001)** 4, **Aleatoric MoG** 4, **CP-DVS** 4, **Adaptive Window Cp** 2, **Aleatoric scale CP** 2, **Adaptive variance-ratio** 1, **Aleatoric only** 1, **MoECP** 1.

New results by seed: 4021 (28).

## 2026-08-07 17:25 — 4668 results / 2027 configs

Compared with the previous run (2026-08-07 17:12):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| results that moved | 1 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 10 | `logs/moecp_tau_sweep/etth2_tau10_20260807_171853.log`, `logs/moecp_tau_sweep/etth2_tau2_20260807_172350.log`, `logs/moecp_tau_sweep/illness_tau100_20260807_171756.log`, `logs/moecp_tau_sweep/illness_tau10_20260807_171756.log`, `logs/moecp_tau_sweep/illness_tau1_20260807_172311.log` … |
| source files appended to | 3 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_cqr_retrain.txt`, `result_calibration_moecp_tsf.txt` |

New results by method: **CQR retrain** 1.

New results by seed: 4021 (1).

## 2026-08-07 17:12 — 4667 results / 2027 configs

Compared with the previous run (2026-08-07 17:07):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm1 (1) |
| source files appended to | 2 | `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt` |

New results by method: **Adaptive variance-ratio** 1.

New results by seed: 4021 (1).

## 2026-08-07 17:07 — 4666 results / 2027 configs

No change since the previous run (2026-08-07 17:02). No new logs, no new results, no value moved.

## 2026-08-07 17:02 — 4666 results / 2027 configs

Compared with the previous run (2026-08-07 16:57):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm1 (1) |
| source files appended to | 2 | `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_scale_tsf.txt` |

New results by method: **Aleatoric scale CP** 1.

New results by seed: 4021 (1).

## 2026-08-07 16:57 — 4665 results / 2027 configs

Compared with the previous run (2026-08-07 16:52):

| change | count | detail |
|---|---|---|
| results that moved | 1 | same config+method, different coverage/width (model retrained or rerun) |
| source files appended to | 3 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_only.txt` |

## 2026-08-07 16:52 — 4665 results / 2027 configs

Compared with the previous run (2026-08-07 16:46):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| new training configs | **1** | configs never seen before |
| results that moved | 1 | same config+method, different coverage/width (model retrained or rerun) |
| source files appended to | 5 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_mog_v2.txt`, `result_calibration_cqr_quantile.txt` |

New results by method: **CQR quantile** 1.

New results by seed: 4021 (1).

## 2026-08-07 16:46 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:41):

| change | count | detail |
|---|---|---|
| results that moved | 1 | same config+method, different coverage/width (model retrained or rerun) |
| source files appended to | 4 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_mog.txt` |

## 2026-08-07 16:41 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:39):

| change | count | detail |
|---|---|---|
| source files appended to | 2 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 16:39 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:36):

| change | count | detail |
|---|---|---|
| source files appended to | 4 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_mse_cp.txt` |

## 2026-08-07 16:36 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:31):

| change | count | detail |
|---|---|---|
| source files appended to | 4 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_cpvs.txt` |

## 2026-08-07 16:31 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:26):

| change | count | detail |
|---|---|---|
| source files appended to | 3 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` |

## 2026-08-07 16:26 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:21):

| change | count | detail |
|---|---|---|
| source files appended to | 5 | `logs/aci_g001_seed4021/traffic_retry_20260807_153424.log`, `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` |

## 2026-08-07 16:21 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:16):

| change | count | detail |
|---|---|---|
| source files appended to | 3 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` |

## 2026-08-07 16:16 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:11):

| change | count | detail |
|---|---|---|
| source files appended to | 3 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` |

## 2026-08-07 16:11 — 4664 results / 2026 configs

Compared with the previous run (2026-08-07 16:06):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm2 (1), electricity (1) |
| results that moved | 1 | same config+method, different coverage/width (model retrained or rerun) |
| source files appended to | 6 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/gap_weather_aci/electricity_moecp_w16.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_moecp_tsf.txt` … |

New results by method: **MoECP** 2.

New results by seed: 4021 (2).

## 2026-08-07 16:06 — 4662 results / 2026 configs

Compared with the previous run (2026-08-07 16:01):

| change | count | detail |
|---|---|---|
| source files appended to | 3 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/gap_weather_aci/electricity_moecp_w16.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` |

## 2026-08-07 16:01 — 4662 results / 2026 configs

Compared with the previous run (2026-08-07 15:56):

| change | count | detail |
|---|---|---|
| source files appended to | 3 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/gap_weather_aci/electricity_moecp_w16.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` |

## 2026-08-07 15:56 — 4662 results / 2026 configs

Compared with the previous run (2026-08-07 15:51):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | ETTm2 (2), ETTm1 (1) |
| source files appended to | 7 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/gap_weather_aci/electricity_moecp_w16.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_window_tsf.txt` … |

New results by method: **CQR retrain** 1, **Adaptive Window Cp** 1, **CP-DVS** 1.

New results by seed: 4021 (3).

## 2026-08-07 15:51 — 4659 results / 2026 configs

Compared with the previous run (2026-08-07 15:46):

| change | count | detail |
|---|---|---|
| source files appended to | 2 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/gap_weather_aci/electricity_moecp_w16.log` |

## 2026-08-07 15:46 — 4659 results / 2026 configs

Compared with the previous run (2026-08-07 15:41):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm1 (1), ETTm2 (1) |
| new training configs | **1** | configs never seen before |
| new source files | 1 | `logs/gap_weather_aci/electricity_moecp_w16.log` |
| source files appended to | 5 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_cqr_quantile.txt` |

New results by method: **CQR quantile** 1, **Adaptive variance-ratio** 1.

New results by seed: 4021 (2).

## 2026-08-07 15:41 — 4657 results / 2025 configs

Compared with the previous run (2026-08-07 15:36):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm2 (2) |
| source files appended to | 6 | `logs/aci_g001_seed4021/traffic_retry_20260807_153424.log`, `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_only.txt` … |

New results by method: **Aleatoric only** 1, **Aleatoric scale CP** 1.

New results by seed: 4021 (2).

## 2026-08-07 15:36 — 4655 results / 2025 configs

Compared with the previous run (2026-08-07 15:34):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| source files appended to | 5 | `logs/aci_g001_seed4021/traffic_retry_20260807_153424.log`, `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_mog_v2.txt` |

New results by method: **Aleatoric MoG v2** 1.

New results by seed: 4021 (1).

## 2026-08-07 15:34 — 4654 results / 2025 configs

Compared with the previous run (2026-08-07 15:31):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| new source files | 2 | `logs/aci_g001_seed4021/traffic_20260807_144744.log`, `logs/aci_g001_seed4021/traffic_retry_20260807_153424.log` |
| source files appended to | 7 | `logs/aci_g001_seed4021/electricity_20260807_144744.log`, `logs/aci_g001_seed4021_driver.log`, `logs/gap_seed4021/traffic_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` … |

New results by method: **Aleatoric MoG** 1.

New results by seed: 4021 (1).

## 2026-08-07 15:31 — 4653 results / 2025 configs

Compared with the previous run (2026-08-07 15:30):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | electricity (1), ETTm2 (1) |
| new training configs | **1** | configs never seen before |
| new source files | 1 | `logs/gap_seed4021/traffic_cqr_20260807_113754.log` |
| source files appended to | 6 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/gap_seed4021_driver.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_cqr_quantile.txt` … |

New results by method: **CQR quantile** 1, **Standard CP** 1.

New results by seed: 4021 (2).

## 2026-08-07 15:30 — 4651 results / 2024 configs

Compared with the previous run (2026-08-07 15:25):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| new training configs | **1** | configs never seen before |
| source files appended to | 3 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_cpvs.txt` |

New results by method: **Adaptive CPVS** 1.

New results by seed: 4021 (1).

## 2026-08-07 15:25 — 4650 results / 2023 configs

Compared with the previous run (2026-08-07 15:20):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm1 (1) |
| source files appended to | 3 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_moecp_tsf.txt` |

New results by method: **MoECP** 1.

New results by seed: 4021 (1).

## 2026-08-07 15:20 — 4649 results / 2023 configs

Compared with the previous run (2026-08-07 15:15):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 15:15 — 4649 results / 2023 configs

Compared with the previous run (2026-08-07 15:14):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 15:14 — 4649 results / 2023 configs

Compared with the previous run (2026-08-07 15:10):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm2 (1), traffic (1) |
| new training configs | **1** | configs never seen before |
| source files appended to | 5 | `logs/aci_g001_driver.log`, `logs/aci_g001_seed4021/traffic_20260807_133656.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt`, `result_calibration_mse_cp.txt` |

New results by method: **Standard CP** 1, **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4021 (2).

## 2026-08-07 15:10 — 4647 results / 2022 configs

Compared with the previous run (2026-08-07 15:05):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 15:05 — 4647 results / 2022 configs

Compared with the previous run (2026-08-07 15:00):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 15:00 — 4647 results / 2022 configs

Compared with the previous run (2026-08-07 14:55):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| new source files | 3 | `logs/aci_g001_seed4021/electricity_20260807_144744.log`, `logs/aci_g001_seed4021/exchange_20260807_144744.log`, `logs/aci_g001_seed4021/illness_20260807_144744.log` |
| source files appended to | 6 | `logs/aci_g001_seed4021/ettm2_20260807_144744.log`, `logs/aci_g001_seed4021_driver.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` … |

New results by method: **CQR retrain** 1.

New results by seed: 4021 (1).

## 2026-08-07 14:55 — 4646 results / 2022 configs

Compared with the previous run (2026-08-07 14:50):

| change | count | detail |
|---|---|---|
| new source files | 1 | `logs/aci_g001_seed4021/ettm2_20260807_144744.log` |
| source files appended to | 4 | `logs/aci_g001_seed4021/ettm1_20260807_144744.log`, `logs/aci_g001_seed4021_driver.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` |

## 2026-08-07 14:50 — 4646 results / 2022 configs

Compared with the previous run (2026-08-07 14:47):

| change | count | detail |
|---|---|---|
| new source files | 2 | `logs/aci_g001_seed4021/etth2_20260807_144744.log`, `logs/aci_g001_seed4021/ettm1_20260807_144744.log` |
| source files appended to | 4 | `logs/aci_g001_seed4021/etth1_20260807_144744.log`, `logs/aci_g001_seed4021_driver.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` |

## 2026-08-07 14:47 — 4646 results / 2022 configs

Compared with the previous run (2026-08-07 14:47):

| change | count | detail |
|---|---|---|
| source files appended to | 2 | `logs/aci_g001_seed4021/etth1_20260807_144744.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log` |

## 2026-08-07 14:47 — 4646 results / 2022 configs

Compared with the previous run (2026-08-07 14:45):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | traffic (1) |
| new source files | 2 | `logs/aci_g001_seed4021/etth1_20260807_144744.log`, `logs/aci_g001_seed4021_driver.log` |
| source files appended to | 4 | `logs/aci_clip_seed4021_driver.log`, `logs/aci_gc_seed4021/traffic_20260807_131938.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `result_calibration_aci_aleatoric_scale_tsf.txt` |

New results by method: **ACI aleatoric scale (g=0.01)** 1.

New results by seed: 4021 (1).

## 2026-08-07 14:45 — 4645 results / 2022 configs

Compared with the previous run (2026-08-07 14:42):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log` |

## 2026-08-07 14:42 — 4645 results / 2022 configs

Compared with the previous run (2026-08-07 14:40):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log` |

## 2026-08-07 14:40 — 4645 results / 2022 configs

Compared with the previous run (2026-08-07 14:35):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | ETTm1 (2), ETTm2 (1) |
| new training configs | **1** | configs never seen before |
| source files appended to | 6 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_window_tsf.txt`, `result_calibration_cp_dvs.txt` … |

New results by method: **CQR quantile** 1, **Adaptive Window Cp** 1, **CP-DVS** 1.

New results by seed: 4021 (3).

## 2026-08-07 14:35 — 4642 results / 2021 configs

Compared with the previous run (2026-08-07 14:30):

| change | count | detail |
|---|---|---|
| source files appended to | 2 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 14:30 — 4642 results / 2021 configs

Compared with the previous run (2026-08-07 14:25):

| change | count | detail |
|---|---|---|
| source files appended to | 3 | `logs/aci_g001_seed4021/traffic_20260807_133656.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` |

## 2026-08-07 14:25 — 4642 results / 2021 configs

Compared with the previous run (2026-08-07 14:20):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | electricity (1) |
| new source files | 1 | `logs/aci_g001_seed4021/traffic_20260807_133656.log` |
| source files appended to | 5 | `logs/aci_g001_driver.log`, `logs/aci_g001_seed4021/electricity_20260807_133656.log`, `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` |

New results by method: **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4021 (1).

## 2026-08-07 14:20 — 4641 results / 2021 configs

Compared with the previous run (2026-08-07 14:15):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm1 (1), traffic (1) |
| source files appended to | 6 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt` … |

New results by method: **Adaptive variance-ratio** 1, **MoECP** 1.

New results by seed: 4021 (2).

## 2026-08-07 14:15 — 4639 results / 2021 configs

Compared with the previous run (2026-08-07 14:09):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| source files appended to | 4 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_moecp_tsf.txt` |

New results by method: **MoECP** 1.

New results by seed: 4021 (1).

## 2026-08-07 14:09 — 4638 results / 2021 configs

Compared with the previous run (2026-08-07 14:04):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm1 (1) |
| source files appended to | 4 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_scale_tsf.txt` |

New results by method: **Aleatoric scale CP** 1.

New results by seed: 4021 (1).

## 2026-08-07 14:04 — 4637 results / 2021 configs

Compared with the previous run (2026-08-07 13:59):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm2 (1), ETTm1 (1) |
| source files appended to | 6 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aleatoric_only.txt` … |

New results by method: **CP-DVS** 1, **Aleatoric only** 1.

New results by seed: 4021 (2).

## 2026-08-07 13:59 — 4635 results / 2021 configs

Compared with the previous run (2026-08-07 13:56):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| new source files | 1 | `logs/gap_seed4021/electricity_cqr_20260807_113754.log` |
| source files appended to | 4 | `logs/aci_gc_seed4021/traffic_20260807_131938.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_adaptive_window_tsf.txt` |

New results by method: **Adaptive Window Cp** 1.

New results by seed: 4021 (1).

## 2026-08-07 13:56 — 4634 results / 2021 configs

Compared with the previous run (2026-08-07 13:55):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | national-illness (1), ETTm1 (1), electricity (1) |
| new training configs | **1** | configs never seen before |
| new source files | 1 | `logs/aci_gc_seed4021/traffic_20260807_131938.log` |
| source files appended to | 9 | `logs/aci_clip_seed4021_driver.log`, `logs/aci_gc_seed4021/electricity_20260807_131938.log`, `logs/gap_seed4021/illness_cqr_20260807_113754.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/gap_seed4021_driver.log` … |

New results by method: **CQR quantile** 1, **Aleatoric MoG v2** 1, **ACI aleatoric scale (g=0.01)** 1.

New results by seed: 4021 (3).

## 2026-08-07 13:55 — 4631 results / 2020 configs

Compared with the previous run (2026-08-07 13:54):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | exchange-rate (1) |
| new training configs | **1** | configs never seen before |
| new source files | 1 | `logs/gap_seed4021/illness_cqr_20260807_113754.log` |
| source files appended to | 3 | `logs/gap_seed4021/exchange_cqr_20260807_113754.log`, `logs/gap_seed4021_driver.log`, `result_calibration_cqr_quantile.txt` |

New results by method: **CQR quantile** 1.

New results by seed: 4021 (1).

## 2026-08-07 13:54 — 4630 results / 2019 configs

Compared with the previous run (2026-08-07 13:53):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/gap_seed4021/exchange_cqr_20260807_113754.log` |

## 2026-08-07 13:53 — 4630 results / 2019 configs

Compared with the previous run (2026-08-07 13:49):

| change | count | detail |
|---|---|---|
| new calibration results | **4** | weather (2), ETTm2 (1), ETTm1 (1) |
| new training configs | **1** | configs never seen before |
| new source files | 2 | `logs/gap_seed4021/exchange_cqr_20260807_113754.log`, `logs/gap_weather_aci/electricity_moecp_20260807_134248.log` |
| source files appended to | 11 | `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/gap_seed4021_driver.log`, `logs/gap_weather_aci/weather_aci_both_20260807_134248.log`, `logs/gap_weather_aci_driver.log` … |

New results by method: **CQR quantile** 1, **Adaptive variance-ratio** 1, **Aleatoric MoG** 1, **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4021 (4).

## 2026-08-07 13:49 — 4626 results / 2018 configs

Compared with the previous run (2026-08-07 13:44):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | ETTm2 (1), ETTm1 (1), weather (1) |
| source files appended to | 9 | `logs/aci_g001_seed4021/electricity_20260807_133656.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/gap_weather_aci/weather_aci_both_20260807_134248.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` … |

New results by method: **Aleatoric scale CP** 1, **Standard CP** 1, **ACI aleatoric scale (g=0.01)** 1.

New results by seed: 4021 (3).

## 2026-08-07 13:44 — 4623 results / 2018 configs

Compared with the previous run (2026-08-07 13:44):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | national-illness (1) |
| new source files | 1 | `logs/aci_g001_seed4021/electricity_20260807_133656.log` |
| source files appended to | 5 | `logs/aci_g001_driver.log`, `logs/aci_g001_seed4021/illness_20260807_133656.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` |

New results by method: **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4021 (1).

## 2026-08-07 13:44 — 4622 results / 2018 configs

Compared with the previous run (2026-08-07 13:43):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm2 (1), exchange-rate (1) |
| new source files | 1 | `logs/aci_g001_seed4021/illness_20260807_133656.log` |
| source files appended to | 6 | `logs/aci_g001_driver.log`, `logs/aci_g001_seed4021/exchange_20260807_133656.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_g001_tsf.txt` … |

New results by method: **Aleatoric only** 1, **ACI aleatoric scale (g=0.001)** 1.

New results by seed: 4021 (2).

## 2026-08-07 13:43 — 4620 results / 2018 configs

Compared with the previous run (2026-08-07 13:28):

| change | count | detail |
|---|---|---|
| **new calibration method** | 1 | `aci_aleatoric_scale_g001` — first time this method has produced time-series results |
| new calibration results | **8** | ETTm2 (3), ETTm1 (2), ETTh1 (1), ETTh2 (1) |
| new training configs | **1** | configs never seen before |
| new source files | 9 | `logs/aci_g001_driver.log`, `logs/aci_g001_seed4021/etth1_20260807_133656.log`, `logs/aci_g001_seed4021/etth2_20260807_133656.log`, `logs/aci_g001_seed4021/ettm1_20260807_133656.log`, `logs/aci_g001_seed4021/ettm2_20260807_133656.log` … |
| source files appended to | 10 | `logs/aci_gc_seed4021/electricity_20260807_131938.log`, `logs/gap_seed4021/traffic_cpmog_relaunch.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` … |

New results by method: **ACI aleatoric scale (g=0.001)** 4, **Aleatoric MoG** 1, **Aleatoric MoG v2** 1, **Adaptive CPVS** 1, **Aleatoric scale CP** 1.

New results by seed: 4021 (8).

## 2026-08-07 13:28 — 4612 results / 2017 configs

Compared with the previous run (2026-08-07 13:27):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | exchange-rate (1), national-illness (1) |
| new source files | 2 | `logs/aci_gc_seed4021/electricity_20260807_131938.log`, `logs/aci_gc_seed4021/illness_20260807_131938.log` |
| source files appended to | 6 | `logs/aci_clip_seed4021_driver.log`, `logs/aci_gc_seed4021/exchange_20260807_131938.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` … |

New results by method: **ACI aleatoric scale (g=0.01)** 2.

New results by seed: 4021 (2).

## 2026-08-07 13:27 — 4610 results / 2017 configs

Compared with the previous run (2026-08-07 13:27):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTm2 (1) |
| new source files | 1 | `logs/aci_gc_seed4021/exchange_20260807_131938.log` |
| source files appended to | 5 | `logs/aci_clip_seed4021_driver.log`, `logs/aci_gc_seed4021/ettm2_20260807_131938.log`, `logs/gap_seed4021/weather_cqr_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_tsf.txt` |

New results by method: **ACI aleatoric scale (g=0.01)** 1.

New results by seed: 4021 (1).

## 2026-08-07 13:27 — 4609 results / 2017 configs

Compared with the previous run (2026-08-07 13:01):

| change | count | detail |
|---|---|---|
| new calibration results | **3** | ETTh2 (2), ETTm1 (1) |
| new training configs | **2** | configs never seen before |
| results that moved | 3 | same config+method, different coverage/width (model retrained or rerun) |
| results that disappeared | 3 | source file truncated, rotated or deleted |
| new source files | 6 | `logs/aci_clip_seed4021_driver.log`, `logs/aci_gc_seed4021/etth1_20260807_131938.log`, `logs/aci_gc_seed4021/etth2_20260807_131938.log`, `logs/aci_gc_seed4021/ettm1_20260807_131938.log`, `logs/aci_gc_seed4021/ettm2_20260807_131938.log` … |
| source files appended to | 8 | `logs/gap_seed4021/traffic_mog_20260807_113754.log`, `logs/gap_seed4021_driver.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_tsf.txt` … |
| source files gone | 8 | `logs/aci_gc_seed4021/electricity_20260807_122604.log`, `logs/aci_gc_seed4021/etth1_20260807_122604.log`, `logs/aci_gc_seed4021/etth2_20260807_122604.log`, `logs/aci_gc_seed4021/ettm1_20260807_122604.log`, `logs/aci_gc_seed4021/ettm2_20260807_122604.log` |

New results by method: **CQR quantile** 1, **CQR retrain** 1, **Standard CP** 1.

New results by seed: 4021 (3).

## 2026-08-07 13:01 — 4609 results / 2015 configs

Compared with the previous run (2026-08-07 12:51):

| change | count | detail |
|---|---|---|
| new calibration results | **2** | ETTm1 (1), ETTh2 (1) |
| source files appended to | 4 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_cqr_retrain.txt`, `result_calibration_moecp_tsf.txt` |

New results by method: **CQR retrain** 1, **MoECP** 1.

New results by seed: 4021 (2).

## 2026-08-07 12:51 — 4607 results / 2015 configs

Compared with the previous run (2026-08-07 12:44):

| change | count | detail |
|---|---|---|
| new calibration results | **6** | ETTh2 (5), ETTm1 (1) |
| new training configs | **1** | configs never seen before |
| source files appended to | 9 | `logs/gap_seed4021/traffic_mog_20260807_113754.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt` … |

New results by method: **CQR quantile** 1, **Adaptive variance-ratio** 1, **Adaptive Window Cp** 1, **Aleatoric only** 1, **Aleatoric scale CP** 1, **CP-DVS** 1.

New results by seed: 4021 (6).

## 2026-08-07 12:44 — 4601 results / 2014 configs

Compared with the previous run (2026-08-07 12:09):

| change | count | detail |
|---|---|---|
| new calibration results | **10** | ETTh2 (8), ETTm1 (1), electricity (1) |
| new training configs | **3** | configs never seen before |
| results that moved | 6 | same config+method, different coverage/width (model retrained or rerun) |
| results that disappeared | 23 | source file truncated, rotated or deleted |
| new source files | 10 | `logs/aci_gc_seed4021/electricity_20260807_122604.log`, `logs/aci_gc_seed4021/etth1_20260807_122604.log`, `logs/aci_gc_seed4021/etth2_20260807_122604.log`, `logs/aci_gc_seed4021/ettm1_20260807_122604.log`, `logs/aci_gc_seed4021/ettm2_20260807_122604.log` … |
| source files appended to | 15 | `logs/gap_seed4021/electricity_mog_20260807_113754.log`, `logs/gap_seed4021/exchange_mog_20260807_113754.log`, `logs/gap_seed4021/illness_mog_20260807_113754.log`, `logs/gap_seed4021_driver.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log` … |

New results by method: **MoECP** 2, **Standard CP** 2, **CQR quantile** 1, **CQR retrain** 1, **Aleatoric MoG** 1, **Aleatoric MoG v2** 1, **Adaptive CPVS** 1, **Aleatoric scale CP** 1.

New results by seed: 4021 (10).

## 2026-08-07 12:09 — 4614 results / 2011 configs

Compared with the previous run (2026-08-07 11:49):

| change | count | detail |
|---|---|---|
| new calibration results | **13** | ETTh2 (11), ETTm1 (2) |
| new training configs | **2** | configs never seen before |
| source files appended to | 12 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt`, `result_calibration_aleatoric_mog.txt` … |

New results by method: **Standard CP** 2, **Adaptive Window Cp** 2, **CP-DVS** 2, **CQR retrain** 1, **Adaptive variance-ratio** 1, **Aleatoric MoG** 1, **Aleatoric MoG v2** 1, **Aleatoric only** 1, **Aleatoric scale CP** 1, **Adaptive CPVS** 1.

New results by seed: 4021 (13).

## 2026-08-07 11:49 — 4601 results / 2009 configs

Compared with the previous run (2026-08-07 11:38):

| change | count | detail |
|---|---|---|
| new calibration results | **9** | exchange-rate (4), national-illness (2), ETTh2 (1), ETTm1 (1) |
| new training configs | **1** | configs never seen before |
| new source files | 2 | `logs/gap_seed4021/electricity_mog_20260807_113754.log`, `logs/gap_seed4021/illness_mog_20260807_113754.log` |
| source files appended to | 8 | `logs/collect_calibration_results_20260807_113816.log`, `logs/gap_seed4021/exchange_mog_20260807_113754.log`, `logs/gap_seed4021_driver.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aci_aleatoric_scale_tsf.txt` … |

New results by method: **ACI aleatoric scale** 4, **MoECP** 2, **CQR quantile** 1, **Aleatoric only** 1, **Aleatoric scale CP** 1.

New results by seed: 4021 (7), 4025 (2).

## 2026-08-07 11:38 — 4592 results / 2008 configs

Compared with the previous run (2026-08-07 11:35):

| change | count | detail |
|---|---|---|
| new calibration results | **1** | ETTh2 (1) |
| new source files | 2 | `logs/collect_calibration_results_20260807_113816.log`, `logs/gap_seed4021/exchange_mog_20260807_113754.log` |
| source files appended to | 5 | `logs/calib_gap_traffic_mog_seed4021_20260807_104157.log`, `logs/gap_seed4021_driver.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `result_calibration_aleatoric_only.txt`, `result_calibration_moecp_tsf.txt` |

New results by method: **MoECP** 1.

New results by seed: 4021 (1).

## 2026-08-07 11:35 — 4591 results / 2008 configs

Compared with the previous run (2026-08-07 10:37):

| change | count | detail |
|---|---|---|
| new calibration results | **70** | ETTh2 (29), ETTm1 (14), ETTm2 (9), national-illness (8) |
| new training configs | **4** | configs never seen before |
| results that moved | 6 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 22 | `logs/calib_gap_electricity_mog_seed4021_20260807_104157.log`, `logs/calib_gap_electricity_mog_seed4021_par_20260807_110814.log`, `logs/calib_gap_electricity_mog_seed4021_par_20260807_110936.log`, `logs/calib_gap_traffic_mog_seed4021_20260807_104157.log`, `logs/calib_gap_traffic_mog_seed4021_par_20260807_110814.log` … |
| source files appended to | 18 | `logs/moecp_gap_ETTh2_ne3_seed4023_20260807_102845.log`, `logs/moecp_gap_ETTm1_ne3_seed4021_20260807_102845.log`, `logs/moecp_gap_ett_ne3_launcher.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log` … |

New results by method: **ACI aleatoric scale** 24, **MoECP** 16, **Aleatoric scale CP** 9, **Adaptive variance-ratio** 3, **Aleatoric MoG v2** 3, **CQR quantile** 2, **CQR retrain** 2, **Standard CP** 2, **Adaptive Window Cp** 2, **Aleatoric MoG** 2, **Aleatoric only** 2, **CP-DVS** 2, **Adaptive CPVS** 1.

New results by seed: 4021 (45), 4022 (6), 4023 (7), 4024 (7), 4025 (5).

## 2026-08-07 10:37 — 4521 results / 2004 configs

Compared with the previous run (2026-08-07 06:00):

| change | count | detail |
|---|---|---|
| **new calibration method** | 1 | `aci_aleatoric_scale` — first time this method has produced time-series results |
| new calibration results | **110** | ETTh1 (60), ETTm1 (25), ETTm2 (16), ETTh2 (9) |
| new training configs | **20** | configs never seen before |
| results that moved | 3 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 6 | `logs/moecp_gap_ETTh2_ne3_seed4021_20260807_102845.log`, `logs/moecp_gap_ETTh2_ne3_seed4022_20260807_102845.log`, `logs/moecp_gap_ETTh2_ne3_seed4023_20260807_102845.log`, `logs/moecp_gap_ETTm1_ne3_seed4021_20260807_102845.log`, `logs/moecp_gap_ett_ne3_launcher.log` … |
| source files appended to | 14 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt`, `result_calibration_aleatoric_mog.txt` … |

New results by method: **MoECP** 11, **Standard CP** 11, **CQR quantile** 9, **CQR retrain** 9, **Adaptive variance-ratio** 9, **Adaptive Window Cp** 9, **Aleatoric MoG** 9, **Aleatoric MoG v2** 9, **Aleatoric only** 9, **Aleatoric scale CP** 9, **CP-DVS** 9, **Adaptive CPVS** 6, **ACI aleatoric scale** 1.

New results by seed: 4021 (109), 4022 (1).

## 2026-08-07 06:00 — 4411 results / 1984 configs

Compared with the previous run (2026-08-06 16:58):

| change | count | detail |
|---|---|---|
| new calibration results | **458** | ETTm2 (149), ETTh2 (140), ETTh1 (86), ETTm1 (83) |
| new training configs | **49** | configs never seen before |
| results that moved | 19 | same config+method, different coverage/width (model retrained or rerun) |
| new source files | 16 | `logs/cp_gap_etth1_moe_launcher.log`, `logs/cp_gap_etth1_moe_ne1_seed4022_20260806_224716.log`, `logs/cp_gap_etth1_moe_ne1_seed4023_20260806_224716.log`, `logs/cp_gap_etth1_moe_ne1_seed4024_20260806_224716.log`, `logs/cp_gap_etth1_moe_ne1_seed4025_20260806_224716.log` … |
| source files appended to | 14 | `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_163120.log`, `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_adaptive_window_tsf.txt`, `result_calibration_aleatoric_mog.txt` … |

New results by method: **Adaptive Window Cp** 119, **Aleatoric only** 114, **MoECP** 44, **Standard CP** 30, **Adaptive variance-ratio** 24, **Aleatoric scale CP** 23, **CP-DVS** 20, **CQR quantile** 19, **CQR retrain** 19, **Aleatoric MoG v2** 18, **Aleatoric MoG** 17, **Adaptive CPVS** 11.

New results by seed: 4021 (263), 4022 (47), 4023 (47), 4024 (47), 4025 (54).

## 2026-08-06 16:58 — 3953 results / 1935 configs

Compared with the previous run (2026-08-06 14:36):

| change | count | detail |
|---|---|---|
| **new calibration method** | 1 | `adaptive_window_cp` — first time this method has produced time-series results |
| new calibration results | **1197** | national-illness (623), exchange-rate (320), ETTh2 (80), ETTh1 (59) |
| new training configs | **522** | configs never seen before |
| new source files | 490 | `logs/gap_fill_cp_cpvs_seeds4022_4025.log`, `logs/grid_launcher_20260806_163120.log`, `logs/pl336_launcher_20260806_152526.log`, `logs/run_tsf_ett_backbones_gpu1_20260806_163120.log`, `logs/run_tsf_ett_pl336_gpu0_20260806_152526.log` … |
| source files appended to | 8 | `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_aleatoric_scale_tsf.txt`, `result_calibration_cp_dvs.txt`, `result_calibration_cpvs.txt`, `result_calibration_cqr_quantile.txt` … |

New results by method: **Standard CP** 515, **Adaptive CPVS** 337, **Adaptive variance-ratio** 119, **Aleatoric scale CP** 119, **Adaptive Window Cp** 43, **Aleatoric only** 41, **CQR retrain** 8, **CQR quantile** 7, **Aleatoric MoG** 3, **Aleatoric MoG v2** 2, **CP-DVS** 2, **MoECP** 1.

New results by seed: 4021 (57), 4022 (284), 4023 (284), 4024 (284), 4025 (288).

## 2026-08-06 14:36 — 2756 results / 1413 configs

No change since the previous run (2026-08-06 14:36). No new logs, no new results, no value moved.

## 2026-08-06 14:36 — 2756 results / 1413 configs

No change since the previous run (2026-08-06 14:35). No new logs, no new results, no value moved.

## 2026-08-06 14:35 — 2756 results / 1413 configs

No change since the previous run (2026-08-06 14:35). No new logs, no new results, no value moved.

## 2026-08-06 14:35 — 2756 results / 1413 configs

No change since the previous run (2026-08-06 14:34). No new logs, no new results, no value moved.

## 2026-08-06 14:34 — 2756 results / 1413 configs

Compared with the previous run (2026-08-06 14:19):

| change | count | detail |
|---|---|---|
| new calibration results | **18** | ETTh1 (8), ETTm2 (6), ETTm1 (4) |
| new training configs | **1** | configs never seen before |
| source files appended to | 4 | `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_aleatoric_scale_tsf.txt`, `result_calibration_cqr_quantile.txt`, `result_calibration_cqr_retrain.txt` |

New results by method: **Adaptive variance-ratio** 8, **Aleatoric scale CP** 8, **CQR quantile** 1, **CQR retrain** 1.

New results by seed: 4021 (16), 4025 (2).

## 2026-08-06 14:19 — 2738 results / 1412 configs

No change since the previous run (2026-08-06 14:19). No new logs, no new results, no value moved.

## 2026-08-06 14:19 — 2738 results / 1412 configs

No change since the previous run (2026-08-06 14:18). No new logs, no new results, no value moved.

## 2026-08-06 14:18 — 2738 results / 1412 configs

Compared with the previous run (2026-08-06 14:18):

| change | count | detail |
|---|---|---|
| new calibration results | **12** | ETTh2 (5), ETTm1 (4), ETTm2 (2), ETTh1 (1) |

New results by method: **Aleatoric scale CP** 12.

New results by seed: 4021 (12).

## 2026-08-06 14:18 — 2726 results / 1412 configs

Compared with the previous run (2026-08-06 14:16):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `result_calibration_aleatoric_scale_tsf.txt` |

## 2026-08-06 14:16 — 2726 results / 1412 configs

Compared with the previous run (2026-08-06 08:50):

| change | count | detail |
|---|---|---|
| new calibration results | **43** | ETTh2 (13), ETTh1 (11), ETTm1 (10), ETTm2 (9) |
| new training configs | **16** | configs never seen before |
| new source files | 2 | `result_calibration_adaptive_variance_tsf.txt`, `result_calibration_aleatoric_scale_tsf.txt` |
| source files appended to | 5 | `logs/run_tsf_ett_configs123_20260804_160849.log`, `result_calibration_cpvs.txt`, `result_calibration_cqr_quantile.txt`, `result_calibration_cqr_retrain.txt`, `result_calibration_mse_cp.txt` |

New results by method: **CQR quantile** 15, **CQR retrain** 15, **adaptive_variance** 11, **Adaptive CPVS** 1, **Standard CP** 1.

New results by seed: 4021 (11), 4024 (11), 4025 (21).

## 2026-08-06 08:50 — 2683 results / 1396 configs

Compared with the previous run (2026-08-06 08:49):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_configs123_20260804_160849.log` |

## 2026-08-06 08:49 — 2683 results / 1396 configs

Compared with the previous run (2026-08-06 08:49):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_configs123_20260804_160849.log` |

## 2026-08-06 08:49 — 2683 results / 1396 configs

Compared with the previous run (2026-08-06 08:49):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_configs123_20260804_160849.log` |

## 2026-08-06 08:49 — 2683 results / 1396 configs

Compared with the previous run (2026-08-06 08:49):

| change | count | detail |
|---|---|---|
| source files appended to | 1 | `logs/run_tsf_ett_configs123_20260804_160849.log` |

## 2026-08-06 08:49 — 2683 results / 1396 configs

First run — baseline snapshot, nothing to compare against yet.

