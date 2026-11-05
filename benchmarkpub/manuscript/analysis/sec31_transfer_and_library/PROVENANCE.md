# Unit Provenance: sec31_transfer_and_library

## 1. Estimands Covered
- **ΔKGE_PUB**: pooled median & CI, 36 model-level rows, 180 model×fold rows
- **Two-way ANOVA variance decomposition**: basin SS, model SS, residual SS
- **Global Seen-OOB model rank preservation**: Spearman correlation across 36 models
- **Per-basin Seen-OOB rank preservation**: distribution of within-basin Spearman correlations across 531 basins
- **Equivalence sets & size distributions**: under IC, seen-dPL, and OOB-dPL (15-block water-year bootstrap, q05 rule)
- **Coverage curves & minimal libraries**: exact (MILP) and greedy curves; 90%, 95%, 100% target cover counts and model lists

## 2. Approach Declarations & Parity Verifications

### Script 01: `scripts/sec31_01_delta_kge.py`
- **Approach**: (a) Recomputed from pub_same_pipeline_model_basin.csv
- Recomputed pooled median ΔKGE_PUB: 0.043394 vs frozen 0.043394 (abs diff = 2.776e-17)
- Model-level 36-model max abs diff vs frozen: 1.180e-16
- Models with positive median penalty: 36/36
- Model x fold cells with positive median penalty: 180/180

### Script 02: `scripts/sec31_02_variance_decomposition.py`
- **Approach**: (a) Recomputed from pub_same_pipeline_model_basin.csv
- Basin fraction of SS: 0.610121 (61.01%)
- Model fraction of SS: 0.010219 (1.02%)
- Residual fraction of SS: 0.379661 (37.97%)
- Max abs diff vs frozen `pub_variance_decomposition.csv`: 3.331e-16

### Script 03: `scripts/sec31_03_rank_preservation.py`
- **Approach**: (a) Recomputed from pub_same_pipeline_model_basin.csv and seen_ic_oob_merged.parquet
- Global model rank Spearman (canonical dPL): 0.970142 vs frozen 0.970142 (abs diff = 1.110e-16)
- Global model rank Spearman (same-pipeline): 0.976062
- Basin median rank Spearman: 0.726126 vs frozen 0.726126 (abs diff = 0.000e+00)

### Script 04: `scripts/sec31_04_equivalence_and_coverage.py`
- **Approach**: (b) Loaded frozen verified artifacts from equiv_cover_t1t2_20261003
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_size_distribution.csv` (bytes: 236, mtime: 2026-10-03 13:41:34 UTC, sha256: `4f122bb5ab29ac050faefbb0992ad3650312ca64751a3fd8524cab5205a67171`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_equivalence_sets.csv` (bytes: 182669, mtime: 2026-10-03 13:41:34 UTC, sha256: `8d5e9ee1b618b3e504a53c8c72c52a9223f38e8fe759f085dddd3a49b860bed3`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_coverage_curve.csv` (bytes: 2781, mtime: 2026-10-03 13:41:34 UTC, sha256: `a42568c95ca2e9372cfd5de8044ee48930240843e5930906de6e4e0b07fa1b50`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_greedy_curve.csv` (bytes: 1788, mtime: 2026-10-03 13:41:34 UTC, sha256: `c3f0003c3ddbc2d8e2b07774a860f7443a7eeb416bb7dbc8b125c0d0275880dd`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_min_cover.csv` (bytes: 923, mtime: 2026-10-03 13:41:34 UTC, sha256: `c109349cfe993f7d15b8ac4948a1c129f9bf1e975df6c2cb0fe59e0f3ef88809`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_model_coverage.csv` (bytes: 4048, mtime: 2026-10-03 13:41:34 UTC, sha256: `255f0278015929ad03c4d999d8dc8de8d1acd90c3b5f394d2985df3e99a41d2f`)

- Cover 95% exact models count: {'IC': 4, 'seen-dPL': 6, 'OOB-dPL': 12}
- Cover 100% exact models count: {'IC': 9, 'seen-dPL': 14, 'OOB-dPL': 21}

### Script 06: `scripts/sec31_06_fold_composition.py`
- **Approach**: (a) Recomputed from OOB_FOLD_ASSIGNMENT.csv, gauge_locations_671.csv, and basin_attribute_table.csv
- Attributes included: 13 (lat, lon, arid, f_snow, runoff_ratio...)
- Fold basin counts: [107, 106, 106, 106, 106]

### Script 07: `scripts/sec31_07_jackknife_equivalence_e1.py` (E1 Alternative Uncertainty Convention)
- **Status**: SUPERSEDED (INVALID IMPLEMENTATION)
- **Supersession Reason**: 留一水文年 jackknife 的 15 个伪值按构造远比 bootstrap 重抽值分散度小，对其直接取 q05 分位得到的不确定带过窄（中位宽度 0.012-0.015 vs bootstrap 0.052-0.062），因而所需模型数被系统性放大。保留作为对比基准，改标为 SUPERSEDED，不进入主引述行。
- **Parameters**:
  * Total water years: 15 (WY 1996..2010)
  * Resampling unit: 1 leave-out water year per replicate (15 jackknife replicates total)
  * Pooled evaluation: 14 pooled water years per replicate

### Script 08: `scripts/sec31_08_multiyear_block_bootstrap_c1.py` (C1 Multi-Year Block Bootstrap Active Alternative)
- **Scheme**: Non-overlapping multi-year block bootstrap (L=3 years, 5 blocks total; Clark et al. 2021 NBB methodology)
- **Implementation Details**:
  * Why L=3: Preserves multi-year catchment storage and interannual autocorrelation; divides 15 test years into exactly 5 equal 3-year blocks (1996-1998, 1999-2001, 2002-2004, 2005-2007, 2008-2010).
  * Replicates: 1000 bootstrap draws with replacement from the 5 blocks (seed 20260925).
  * Metric: KGE from sufficient statistics (eps=0.1, sample variance).
  * Equivalence threshold rule: q05_b := 5th percentile of {best_r(b)} across 1000 draws.
  * Equivalence set definition: S_b := { m : KGE^point_{m,b} >= q05_b }.
  * Coverage targets: 90% (478 basins), 95% (505 basins), 100% (531 basins), exact HiGHS MILP.
- **Optimality Certificates**: HiGHS solver status 0; budget K*-1 proven insufficient for all targets.
- **Tables**: `sec31_multiyear_block_bootstrap_comparison.csv`, `sec31_multiyear_block_bootstrap_certificates.csv`, `sec31_multiyear_block_bootstrap_size_distribution.csv`, `sec31_multiyear_block_bootstrap_equivalence_sets.csv`.

### Script 09: `scripts/sec31_09_uncertainty_bandwidth_comparison_c2.py` (C2 Bandwidth Comparison Across 3 Schemes)
- **Bandwidth Definition**: Delta_b := KGE_best^point - q05_b (tolerance offset from best model's point KGE).
- **Schemes Compared**:
  * Primary 1-year block bootstrap (median: IC 0.0623, seen-dPL 0.0524, OOB-dPL 0.0577)
  * Jackknife (SUPERSEDED, median: IC 0.0145, seen-dPL 0.0119, OOB-dPL 0.0138; overly narrow)
  * C1 3-year block bootstrap (median: IC 0.0604, seen-dPL 0.0501, OOB-dPL 0.0535; stable multi-year uncertainty)
- **Tables**: `sec31_uncertainty_bandwidth_comparison.csv`, `sec31_uncertainty_bandwidth_per_basin.csv`.
### Basin-order index audit note (A1/A2, 2026-10-04)
- **sec31_07 (line 124)**: `b_idx` from unsorted `S.canonical_ids()` was used to label rows of the SORTED OOB cube.
  Affected output: `sec31_jackknife_equivalence_sets.csv` (OOB-dPL `basin_id` labels mismatched at 439/531 positions).
  The table is already SUPERSEDED for the invalid jackknife construction; its OOB basin labels are additionally misassigned.
- **sec31_08 (line 158)**: same pattern. Affected output: `sec31_multiyear_block_bootstrap_equivalence_sets.csv`
  (OOB-dPL `basin_id` labels mismatched). Set-cover counts are permutation-invariant and were RE-VERIFIED unchanged.
- **sec31_09 (line 128)**: same pattern. Affected output: `sec31_uncertainty_bandwidth_per_basin.csv`
  (OOB-dPL rows carry the bandwidth of a different basin at 439/531 positions; the per-row values themselves are internally consistent).
  The aggregate bandwidth medians in `sec31_uncertainty_bandwidth_comparison.csv` were RE-VERIFIED by single-source recompute (abs diff 0.0).
- **Not affected**: sec31_01..04, sec31_06 (all use `S.basin_order()` or basin-ID joins); `sec31_equivalence_sets_per_basin.csv` PASSED the A2 ID-joined check.
## 3. Output Tables Generated
- `tables/sec31_basin_rank_spearman_per_basin.csv` (14397 bytes)
- `tables/sec31_basin_rank_spearman_summary.csv` (265 bytes)
- `tables/sec31_coverage_curves_exact.csv` (2710 bytes)
- `tables/sec31_coverage_curves_greedy.csv` (1399 bytes)
- `tables/sec31_equivalence_sets_per_basin.csv` (149740 bytes)
- `tables/sec31_equivalence_size_distribution.csv` (255 bytes)
- `tables/sec31_fold_composition.csv` (6000 bytes)
- `tables/sec31_global_rank_spearman.csv` (340 bytes)
- `tables/sec31_jackknife_equivalence_sets.csv` (64806 bytes)
- `tables/sec31_jackknife_size_distribution.csv` (371 bytes)
- `tables/sec31_jackknife_vs_bootstrap_cover_comparison.csv` (613 bytes)
- `tables/sec31_minimal_cover_targets.csv` (965 bytes)
- `tables/sec31_model_equivalence_coverage.csv` (2914 bytes)
- `tables/sec31_model_fold_delta_kge.csv` (12378 bytes)
- `tables/sec31_model_level_delta_kge.csv` (2840 bytes)
- `tables/sec31_multiyear_block_bootstrap_certificates.csv` (512 bytes)
- `tables/sec31_multiyear_block_bootstrap_comparison.csv` (665 bytes)
- `tables/sec31_multiyear_block_bootstrap_equivalence_sets.csv` (124732 bytes)
- `tables/sec31_multiyear_block_bootstrap_size_distribution.csv` (465 bytes)
- `tables/sec31_pooled_delta_kge.csv` (239 bytes)
- `tables/sec31_uncertainty_bandwidth_comparison.csv` (1557 bytes)
- `tables/sec31_uncertainty_bandwidth_per_basin.csv` (82443 bytes)
- `tables/sec31_variance_decomposition.csv` (196 bytes)
