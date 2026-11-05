# Unit Provenance: _verification (V1 Abstract Numbers Recomputation)

## 1. Scope & Isolation Declaration
- **Hard Isolation**: This unit executes an entirely independent implementation path (`scripts/v1_recompute_abstract_numbers.py`).
- **Zero Imported Analysis Modules**: Does NOT import or call `_shared`, `c4_common`, `tk_common`, or any previous calculation script.
- **Zero Reads of Derived Tables**: Does NOT read any derived tables from `equiv_cover_t1t2`, `topk`, `chapter4_final_closure`, or other analysis folders.
- **Primitive Input Files Consumed**:
  * `data/531sub_id.txt (canonical IDs)`
  * `project/benchmark/analysis/ic_dpl_gap_conditional_20260923/forward/test/water_year_sufficient_stats.csv (raw stats for IC and seen-dPL)`
  * `project/benchmarkpub/data/oob36_20260930/qsim/ and obs/ (raw daily simulated and observed discharge)`
  * `project/benchmarkpub/data/oob36_20260930/manifests/OOB_FOLD_ASSIGNMENT.csv (raw fold assignments)`
  * `project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv (benchmark model-basin table)`

## 2. Recomputed Abstract-Level Numbers (V1)
| Quantity | Condition | Recomputed | Existing Frozen | Absolute Difference | Optimality Certificate / Evidence |
|---|---|---:|---:|---:|---|
| cover95_model_count_IC | IC | 4.0 | 4.0 | 0.000e+00 | Budget 3 covers max 495 / 505 needed |
| cover95_model_count_seen_dpl | seen-dPL | 6.0 | 6.0 | 0.000e+00 | Budget 5 covers max 500 / 505 needed |
| cover95_model_count_oob_dpl | OOB-dPL | 12.0 | 12.0 | 0.000e+00 | Budget 11 covers max 502 / 505 needed |
| cover100_model_count_IC | IC | 9.0 | 9.0 | 0.000e+00 | Budget 8 covers max 529 / 531 needed |
| cover100_model_count_seen_dpl | seen-dPL | 14.0 | 14.0 | 0.000e+00 | Budget 13 covers max 530 / 531 needed |
| cover100_model_count_oob_dpl | OOB-dPL | 21.0 | 21.0 | 0.000e+00 | Budget 20 covers max 530 / 531 needed |
| fixed_global_k2_vs_k1_paired_diff_median | OOB-dPL (evaluated on held-out) | 0.011325257049407 | 0.011325 | 2.570e-07 | Recomputed via daily ensemble series mean KGE on training-fold selected members |
| full_36_model_oracle_kge_median | OOB-dPL (held-out max) | 0.7462731831537561 | 0.7462711677 | 2.015e-06 | Max KGE across all 36 models per basin |
| delta_kge_pub_pooled_median | OOB-36 same-pipeline | 0.0433943064398929 | 0.0433943064 | 3.989e-11 | Median over 19,116 cells (seenfolds_median4 - kge_heldout) |

## 3. Optimality Certificates for Set Covers
| Condition | Target | Need Basins | Proven Min Models | Budget - 1 | Max Coverable at Budget - 1 | Proves Q Impossible? |
|---|---|---:|---:|---:|---:|---|
| IC | 0.95 | 505 | 4 | 3 | 495 | True |
| IC | 1.00 | 531 | 9 | 8 | 529 | True |
| seen-dPL | 0.95 | 505 | 6 | 5 | 500 | True |
| seen-dPL | 1.00 | 531 | 14 | 13 | 530 | True |
| OOB-dPL | 0.95 | 505 | 12 | 11 | 502 | True |
| OOB-dPL | 1.00 | 531 | 21 | 20 | 530 | True |

## 4. Script Paths
- Generating script: `scripts/v1_recompute_abstract_numbers.py`
- Provenance script: `scripts/v1_provenance.py`
