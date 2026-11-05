# Unit Provenance: sec32_library_members

## 1. Estimands Covered
- **Candidate set union size**: 36/36 models ever appear in Top-3 and Top-5
- **Minimal cover sets (Top-3, Top-5)**: exact minimum cover (13 and 9 models, proven optimal via MILP) and alternative optimal solutions
- **OOB 100% cover member list**: 21 models covering 100% of basins under OOB-dPL equivalence sets
- **Coverage ladder (Seen Top-k -> OOB Top-1)**: k=1 (0.2524), k=3 (0.5028), k=5 (0.6271), k=10 (0.8041)
- **Regret curve**: median regret k=1 (0.0428), k>=3 (0.0000)
- **Model structural attributes**: parameter count (1..15), storage states count (1..6), explicit snow routine flag and import path compiled from MARRMoT metadata

## 2. Approach Declarations & Parity Verifications

### Script 01: `scripts/sec32_01_library_and_covers.py`
- **Approach**: (b) Loaded frozen verified artifacts from pub_minimal_model_library_20260930 and equiv_cover_t1t2_20261003
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_minimal_model_library_20260930/tables/library_model_frequency.csv` (bytes: 3082, mtime: 2026-09-30 08:56:02 UTC, sha256: `66401235ed38f3433e2dca9792fdbb67f0fa79e06427c8d5568207e952f8b05e`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_minimal_model_library_20260930/tables/library_minimum_cover.csv` (bytes: 900, mtime: 2026-09-30 08:56:02 UTC, sha256: `1bc44e4cbcdb5fbea534fe946802e48a11c766435d522c408483d0546bb17f38`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_minimal_model_library_20260930/tables/minimal_model_library.json` (bytes: 14441, mtime: 2026-09-30 08:56:03 UTC, sha256: `4d2d9e633962729a9109e1c85cc1ec00cb80c22907d21a169bb2f129144b9edb`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t1_min_cover.csv` (bytes: 923, mtime: 2026-10-03 13:41:34 UTC, sha256: `c109349cfe993f7d15b8ac4948a1c129f9bf1e975df6c2cb0fe59e0f3ef88809`)
- Top-3 union size: 36/36
- Top-5 union size: 36/36
- Top-3 minimal cover exact size: 13 models
- Top-5 minimal cover exact size: 9 models
- OOB full cover exact size: 21 models

### Script 02: `scripts/sec32_02_coverage_ladder_regret.py`
- **Approach**: (a) Recomputed from pub_same_pipeline_model_basin.csv
- Coverage ladder max abs diff vs frozen: 0.000e+00
- Regret curve max abs diff vs frozen: 7.633e-17
- Coverage k=1: 0.2524, k=3: 0.5028, k=5: 0.6271, k=10: 0.8041

### Script 03: `scripts/sec32_03_model_structural_attributes.py`
- **Approach**: (a) Compiled from model_complexity_inventory.csv and model_snow_annotation.csv
- Total models compiled: 36
- Models with explicit snow routine: 8
- Parameter range: [1, 15] (median: 7.0)
- Storage state range: [1, 6]

### Script 05: `scripts/sec32_05_process_crosstabs.py`
- **Approach**: (a) Recomputed from sec32 tables
- Snow counts: Full 36 (8), OOB 21 (6), Top-3 13 (5), Top-5 9 (3)

### Script 06: `scripts/sec32_06_bootstrap_cover_ci_e2.py` (E2 Basin-Level Bootstrap Intervals)
- **Resampling**: Basin-level bootstrap with replacement (B=1000 draws, seed 20260925)
- **Solver**: Exact partial set cover via MILP (`scipy.optimize.milp`, HiGHS)
- **Evaluated**: 3 conditions (IC, seen-dPL, OOB-dPL) x 3 targets (90%, 95%, 100%)
- **Tables**: `sec32_bootstrap_cover_ci_summary.csv`, `sec32_bootstrap_cover_distribution.csv`

### Script 07: `scripts/sec32_07_optimal_solutions_enumeration_e3.py` (E3 Optimal Solution Enumeration)
- **Enumeration**: Exact MILP with iterative no-good cuts (cutoff: 1000 solutions; none truncated)
- **Evaluated**: 3 conditions (IC, seen-dPL, OOB-dPL) x 2 targets (95%, 100%)
- **Key Counts**: IC 95% (3 sols, core=3, union=6); IC 100% (4 sols, core=7, union=11); seen-dPL 95% (33 sols, core=3, union=18); seen-dPL 100% (18 sols, core=10, union=23); OOB-dPL 95% (162 sols, core=3, union=29); OOB-dPL 100% (4 sols, core=19, union=23)
- **Tables**: `sec32_optimal_covers_summary.csv`, `sec32_optimal_covers_model_frequencies.csv`, `sec32_optimal_covers_all_solutions.csv`

### Script 08: `scripts/sec32_08_singleton_and_bottleneck_e4.py` (E4 Singletons & Bottlenecks)
- **Singletons**: IC (36 basins), seen-dPL (47 basins), OOB-dPL (63 basins)
- **100% Cover Bottlenecks (at budget K*-1)**:
  * IC (K*=9): budget 8 covers 529/531; 2 basins uncovered (09066300, 09378170; models alpine1, alpine2)
  * seen-dPL (K*=14): budget 13 covers 530/531; 1 basin uncovered (11528700; model alpine2)
  * OOB-dPL (K*=21): budget 20 covers 530/531; 1 basin uncovered (14236200; model susannah1)
- **Tables**: `sec32_singleton_model_counts.csv`, `sec32_basin_singleton_status.csv`, `sec32_bottleneck_100pct_coverage.csv`

### ID-joined re-verification (A2/A5, 2026-10-04)
- **Singleton counts** (IC 36, seen-dPL 47, OOB-dPL 63) and **100% cover bottleneck basins**
  (IC `09066300|09378170` forcing alpine1|alpine2|mopex3|mopex4; seen-dPL `11528700` forcing alpine2; OOB-dPL `14236200` forcing susannah1)
  were re-derived by strict basin-ID joins with exact HiGHS MILP (`sec34_bottleneck_singleton_idjoin.py`): all match exactly (0/108 per-model singleton mismatches).
- **Coverage ladder** (k=1 0.2524, k=3 0.5028, k=5 0.6271, k=10 0.8041) and candidate-set regret medians were
  re-derived by basin-ID joins (`sec34_16_a5_remaining_idjoin.py`): max abs diff 3.6e-07.
- Status: **RE-VERIFIED** (values unchanged).

## 3. Output Tables Generated
- `tables/sec32_basin_singleton_status.csv` (40406 bytes)
- `tables/sec32_bootstrap_cover_ci_summary.csv` (922 bytes)
- `tables/sec32_bootstrap_cover_distribution.csv` (1428 bytes)
- `tables/sec32_bottleneck_100pct_coverage.csv` (417 bytes)
- `tables/sec32_coverage_ladder.csv` (705 bytes)
- `tables/sec32_library_model_frequency.csv` (2288 bytes)
- `tables/sec32_minimal_cover_solutions.csv` (882 bytes)
- `tables/sec32_model_structural_attributes.csv` (2778 bytes)
- `tables/sec32_oob_full_cover_members.csv` (792 bytes)
- `tables/sec32_optimal_covers_all_solutions.csv` (24893 bytes)
- `tables/sec32_optimal_covers_model_frequencies.csv` (11541 bytes)
- `tables/sec32_optimal_covers_summary.csv` (1560 bytes)
- `tables/sec32_process_crosstab_parameters.csv` (678 bytes)
- `tables/sec32_process_crosstab_states.csv` (488 bytes)
- `tables/sec32_process_representation_summary.csv` (475 bytes)
- `tables/sec32_regret_curve.csv` (1026 bytes)
- `tables/sec32_singleton_model_counts.csv` (2793 bytes)
