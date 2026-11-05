# Unit Provenance: sec33_usage_and_conditioning

## 1. Estimands Covered
- **T2 non-oracle usage rules A-E**: summary and per-basin evaluations with daily member mean series KGE
- **T3-1 direct paired comparison**: Rule C (attribute-conditioned A2d Top-3) vs Rule B (fixed global Top-3) across 10 seeds
- **T3-2 fixed ensemble size scan**: k in {1, 2, 3, 4, 5, 7, 10} with fold-wise training-only members and paired diffs vs k=1
- **G_set across three estimands**: Seen-dPL (+0.1966), A2d (+0.0461), A1b (+0.0221) with 10 seeds and 3 components (mean best_R, mean best_F, mean VBS, numerator, denominator)
- **Paired regret difference**: per-basin regret difference (recommendation vs fixed baseline) across Seen, A2d, and A1b
- **Basin-specific vs global Top-k paired difference**: k=3 coverage gain +7.9 pp
- **Cluster leave-out OOB validation**: k=3 and k=5 coverage and regret differences, plus typology organization statistics (entropy, stability)

## 2. Approach Declarations & Parity Verifications

### Script 01: `scripts/sec33_01_usage_rules_t2.py`
- **Approach**: (b) Loaded frozen verified artifacts from equiv_cover_t1t2_20261003
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t2_rule_summary.csv` (bytes: 9844, mtime: 2026-10-03 13:45:30 UTC, sha256: `af7909e8deeff7e24eae25faa0c142de9055fa6f630584d3cddab1fd37c9b706`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t2_rule_per_basin.csv` (bytes: 1484215, mtime: 2026-10-03 13:45:30 UTC, sha256: `5e1da06fb21e16d4ed77d13e62aa63a5062969afb8d9224cef92ab3ed7ec13a9`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t2_set_construction_check.csv` (bytes: 328, mtime: 2026-10-03 13:45:30 UTC, sha256: `da965dc80d827bac4ee3b20889671de445b647edc940a01c897da1586dcab3f8`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/equiv_cover_t1t2_20261003/tables/t2_leakage_check.csv` (bytes: 132, mtime: 2026-10-03 13:45:30 UTC, sha256: `a76de0dd137b8917953898c7e1e75cca6772de863f5fb60a826a292a19402b83`)

### Script 02: `scripts/sec33_02_rule_c_vs_b_t3_1.py`
- **Approach**: (a) Recomputed from t2_rule_per_basin.csv
- T3-1 seed 20260925 paired diff mean: +0.001310 (Wilcoxon p = 0.04917)
- T3-1 across 10 seeds mean paired diff: +0.000948
- T3-1 seeds with Wilcoxon p < 0.05: 1/10

### Script 03: `scripts/sec33_03_fixed_ensemble_scan_t3_2.py`
- **Approach**: (a) Recomputed from cached OOB cube and fold Qsim
- Scanned k values: [1, 2, 3, 4, 5, 7, 10]
- KGE median: k=1 (0.6635), k=3 (0.6774), k=5 (0.6665)
- Mean diff vs k=1: k=3 (+0.01087), k=5 (-0.00205)

### Script 04: `scripts/sec33_04_gset_and_regret.py`
- **Approach**: (b) Loaded frozen verified artifacts from oob36_gset_gate_20261001, oob36_paired_regret_20261003, pub_candidate_set_transfer_20260930
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/oob36_gset_gate_20261001/tables/gset_36_summary.json` (bytes: 1817, mtime: 2026-10-03 10:28:23 UTC, sha256: `c93209b819606a5d5b784b1c8c270789076d98ed6062e7f4f4d1c5e641cef15c`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/oob36_gset_gate_20261001/tables/gset_36_A2d_per_seed.csv` (bytes: 790, mtime: 2026-10-03 10:28:23 UTC, sha256: `1dbc098405138944868bf234e655afcbe17cfd77cd5f210b0fd8504a4088b988`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/oob36_gset_gate_20261001/tables/gset_36_A1b_per_seed.csv` (bytes: 800, mtime: 2026-10-03 10:28:23 UTC, sha256: `86e002bc2def83c15a7747dfe13097cfce6ebc05e041b240dff3212f82ff399a`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/oob36_gset_gate_20261001/tables/parity_p1_seen36_per_seed.csv` (bytes: 701, mtime: 2026-10-03 10:28:23 UTC, sha256: `be21e479b72d23950c5ea8ee114ace21aa92656ccbf759234b53826188e7d055`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/oob36_paired_regret_20261003/tables/paired_regret_summary.csv` (bytes: 627, mtime: 2026-10-03 10:29:34 UTC, sha256: `e966dd57eb24a94d982a5d909f5ec429280ef8ab42f3e612114df6bd2dc185e6`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/oob36_paired_regret_20261003/tables/paired_regret_per_seed.csv` (bytes: 6540, mtime: 2026-10-03 10:29:33 UTC, sha256: `0426f2e5c7b6af3b620578cb1ce35054082d98c8104340b7c56f74e887735be3`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_candidate_set_transfer_20260930/tables/basin_specific_vs_global_topk.csv` (bytes: 2934, mtime: 2026-09-30 08:37:47 UTC, sha256: `54844ea6367cb4c1989d9fd81ce6a32c38d00cc62264f787456fff58d79a7235`)
- G_set main seed: Seen (0.1966), A2d (0.0461), A1b (0.0221)
- Paired regret diff mean A2d: -0.00247
- Basin-specific vs global k=3 coverage diff: +0.0791

### Script 05: `scripts/sec33_05_cluster_typology.py`
- **Approach**: (b) Loaded frozen verified artifacts from cluster_oob_recommendation_validation_20261001 and cluster_model_recommendation_20261001
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/cluster_oob_recommendation_validation_20261001/tables/cluster_vs_global_summary.csv` (bytes: 1537, mtime: 2026-10-01 02:43:18 UTC, sha256: `ebeb5753268e30cd547e6b1f242cd1ded64ca476af88d348c956ab7f524bde25`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/cluster_oob_recommendation_validation_20261001/tables/per_cluster_oob_validation.csv` (bytes: 3404, mtime: 2026-10-01 02:43:18 UTC, sha256: `1422e82db99170d3b6d067aa9f4f591cc823f17cd636f3fb641a2b9826114b48`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/cluster_model_recommendation_20261001/tables/cluster_concentration_summary.csv` (bytes: 6001, mtime: 2026-10-01 02:26:01 UTC, sha256: `7230b1dd3a037deffd5b89825e8421e9f27363f1dc80f16baf8cf6859c1d497a`)
  * `/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/cluster_model_recommendation_20261001/tables/cluster_seen_oob_consistency.csv` (bytes: 2117, mtime: 2026-10-01 02:26:01 UTC, sha256: `7b2abad56e510fd1660130251a49e5812a2aedc71769ce1404d8080b0b67448f`)
- Cluster OOB coverage diff: k=3 (-0.0058), k=5 (-0.0078)
- Cluster OOB regret diff mean: k=5 (-0.0040, Wilcoxon p = 0.0369)

### Script 07: `scripts/sec33_07_diversity_sensitivity_v5.py`
- **Approach**: (a) Recomputed from raw Qsim and model structural attributes on training folds
- Scanned k values: [2, 3, 4, 5, 7, 10]
- Structural diversity k=3 mean diff vs ranking: -0.06589
- Behavioral diversity k=3 mean diff vs ranking: -0.06671
- Evaluation scope: SENSITIVITY

### Script 08: `scripts/sec33_08_recommender_k2_k5_e8.py` (E8 Re-fit Recommenders, Parity, k=2 & k=5)
- **Protocol**: Exact re-fitting of Random Forest recommenders (500 trees, min_samples_leaf=5, 35 raw Caravan attributes, snow pre-filter, 5 OOB folds, 10 seeds)
- **Parity Gate**: Verified against `oob36_gset_gate_20261001` outputs; Top-3 A2d and A1b G_set matched exactly across all 10 seeds (max abs diff <= 1e-6)
- **Leakage Audit**: All 5 folds verified strictly zero data leakage (target fold never observed during training)
- **Operating Points k=2 and k=5**:
  * Rule C (attribute-conditioned Top-k equal-weight ensemble) vs Rule B (fixed global Top-k equal-weight ensemble) evaluated via daily simulated discharge mean series KGE
  * In-set best G_set evaluated for both A2d (PRIMARY) and A1b (SECONDARY) with 10,000-draw basin bootstrap 95% CIs across all 10 seeds
- **Persisted Artifacts**: Full ranked model predictions (`sec33_recommender_full_rankings.npz`), `sec33_e8_parity_gate.csv`, `sec33_e8_rule_c_vs_b_k2_k5.csv`, `sec33_e8_gset_k2_k5.csv`, `sec33_e8_leakage_check.csv`

### Script 09: `scripts/sec33_09_rules_ladder_c3.py` (C3 Comprehensive Rules Ladder with Dual Benchmarks)
- **Rules Covered on 531 Basins**: Rule A (k=1), Rule B (k in {1, 2, 3, 4, 5, 7, 10}), Rule C (A2d k in {2, 3, 5}), Rule D (A2d k=3), Rule E (36-model oracle)
- **Dual Benchmarks**: Every rule evaluated simultaneously relative to Rule A (k=1 global single best) AND relative to Rule B (k=2 fixed global ensemble)
- **Statistics**: Absolute KGE median and Q25/Q75, plus paired differences (mean, median, Wilcoxon p, basin bootstrap 95% CI, water-year block bootstrap 95% CI, N=531)
- **Tables**: `sec33_rules_ladder_unified.csv`, `sec33_rule_c_per_seed_ladder.csv`

### ID-joined re-verification (A2/A5, 2026-10-04)
- **Rules-ladder per-basin basis** recomputed from raw OOB Qsim/obs by basin ID (`sec34_12_basin_audit_supplement.py`):
  `sec33_fixed_ensemble_per_basin.csv` for k=1,2,3,4,5,7,10 and `sec33_usage_rules_per_basin.csv` for A, B3, B5,
  C_A2d x10 seeds, C_A1b x10 seeds, D_A2d x10 seeds, D_A1b x10 seeds, E — all PASS (max abs diff 5.0e-07).
  Rule C k=2 and k=5 per-basin series have no stored per-basin table (computed in-script); the aggregate ladder values were cross-checked and PASS.
- **Cluster matching** (515/531, identical unmatched ID set) and **cluster leave-out validation** coverage/regret
  (k=3: 0.4311/0.4252; k=5: 0.5204/0.5126; diff regret -0.00405/-0.00402) were re-derived by basin-ID joins
  (`sec34_16_a5_remaining_idjoin.py`): max abs diff 4.7e-05.
- Status: **RE-VERIFIED** (values unchanged).
### Script 10: `scripts/sec33_10_rule_c_k2_k5_per_basin_f1.py` (F1 Rule C k=2/k=5 per-basin series)
- **Deliverable**: `sec33_rule_c_k2_k5_per_basin.csv` — 10,620 rows (10 seeds x 531 basins x k in {2,5}),
  columns identical to `sec33_usage_rules_per_basin.csv` (rule, k, seed, basin_id, fold, kge, diff_vs_A);
  KGE computed once on the daily equal-weight ensemble mean series (never by averaging member KGEs).
- **Re-verification**: C(2)/C(5) paired differences vs A and vs B(k=2) recomputed with the T2/T3 protocol
  (Wilcoxon on non-zero diffs; 1,000-draw basin bootstrap and water-year block proxy, seed 20260925, shared-RNG order
  identical to sec33_09) and compared with `sec33_rule_c_per_seed_ladder.csv` / `sec33_rules_ladder_unified.csv`:
  all 14 quantities agree within 5.0e-07, the 6-dp write precision of the existing tables.
- **Row-by-row alignment check**: 21,240 rows (4 slices) re-derived in an independent pass from raw Qsim/obs + E8 rankings
  by basin ID; 0 mismatches, max abs diff 5.0e-07.
- **Tables**: `sec33_rule_c_k2_k5_per_basin.csv`, `sec33_rule_c_k2_k5_recompute_comparison.csv`, `sec33_rule_c_k2_k5_alignment_check.csv`.

### Script 11: `scripts/sec33_11_constrained_fixed_ensemble_f2.py` (F2 parameter-constrained fixed global ensembles)
- **Construction**: thresholds are the P25 / median / P75 of the 36-model `n_parameters` distribution (5.75 / 7.0 / 9.25);
  eligible models are those with n_parameters <= threshold (9 / 20 / 27 models); members are the top-2 and top-3 eligible
  models by the training-fold median OOB KGE — the same ranking statistic as the existing unconstrained B(k) protocol.
  The selection never uses target-fold information; the unconstrained members reproduce `sec33_fixed_ensemble_fold_members.csv` exactly.
- **Results** (paired difference constrained minus unconstrained, T2/T3 protocol, N=531):
  * P25 (9 eligible): k=2 mean -0.123235, k=3 mean -0.124117;
  * median (20 eligible): k=2 mean -0.045860, k=3 mean -0.039791;
  * P75 (27 eligible): k=2 mean -0.060332, k=3 mean -0.034161.
- **Degeneracy**: 0 of 30 fold-level tier x k cells have identical constrained/unconstrained member sets; no tier is degenerate.
- **Parameter savings** (sum of member n_parameters, per-fold range): P25 k=2 18-18, k=3 20-22;
  median k=2 14-14, k=3 15-18; P75 k=2 12-12, k=3 11-14.
- **Leakage check**: per fold, target-fold basins are excluded from member construction (`target_fold_in_construction = NO`).
- **Tables**: `sec33_constrained_ensemble_summary.csv`, `sec33_constrained_ensemble_per_basin.csv`,
  `sec33_constrained_ensemble_members.csv`, `sec33_constrained_ensemble_leakage_check.csv`,
  `sec33_constrained_ensemble_unconstrained_member_check.csv`.

## 3. Output Tables Generated
- `tables/sec33_basin_specific_vs_global_topk.csv` (2085 bytes)
- `tables/sec33_cluster_concentration_entropy.csv` (7079 bytes)
- `tables/sec33_cluster_oob_per_cluster.csv` (4397 bytes)
- `tables/sec33_cluster_oob_validation_summary.csv` (1712 bytes)
- `tables/sec33_cluster_seen_oob_stability.csv` (2479 bytes)
- `tables/sec33_constrained_ensemble_leakage_check.csv` (297 bytes)
- `tables/sec33_constrained_ensemble_members.csv` (3479 bytes)
- `tables/sec33_constrained_ensemble_per_basin.csv` (137194 bytes)
- `tables/sec33_constrained_ensemble_summary.csv` (1043 bytes)
- `tables/sec33_constrained_ensemble_unconstrained_member_check.csv` (457 bytes)
- `tables/sec33_diversity_fold_members.csv` (4453 bytes)
- `tables/sec33_diversity_leakage_check.csv` (132 bytes)
- `tables/sec33_diversity_per_basin.csv` (394903 bytes)
- `tables/sec33_diversity_sensitivity_summary.csv` (2032 bytes)
- `tables/sec33_e8_gset_k2_k5.csv` (2352 bytes)
- `tables/sec33_e8_leakage_check.csv` (497 bytes)
- `tables/sec33_e8_parity_gate.csv` (797 bytes)
- `tables/sec33_e8_rule_c_vs_b_k2_k5.csv` (2647 bytes)
- `tables/sec33_fixed_ensemble_fold_members.csv` (1678 bytes)
- `tables/sec33_fixed_ensemble_k_scan.csv` (969 bytes)
- `tables/sec33_fixed_ensemble_per_basin.csv` (117543 bytes)
- `tables/sec33_gset_per_seed.csv` (1445 bytes)
- `tables/sec33_gset_summary.csv` (710 bytes)
- `tables/sec33_paired_regret_per_seed.csv` (3549 bytes)
- `tables/sec33_paired_regret_summary.csv` (386 bytes)
- `tables/sec33_rule_c_k2_k5_alignment_check.csv` (391 bytes)
- `tables/sec33_rule_c_k2_k5_comparison_summary.csv` (663 bytes)
- `tables/sec33_rule_c_k2_k5_per_basin.csv` (485259 bytes)
- `tables/sec33_rule_c_k2_k5_recompute_comparison.csv` (16443 bytes)
- `tables/sec33_rule_c_per_seed_ladder.csv` (6061 bytes)
- `tables/sec33_rule_c_vs_b_paired_per_seed.csv` (1705 bytes)
- `tables/sec33_rule_c_vs_b_summary.csv` (363 bytes)
- `tables/sec33_rules_ladder_unified.csv` (3212 bytes)
- `tables/sec33_set_construction_spot_check.csv` (227 bytes)
- `tables/sec33_usage_rules_leakage_check.csv` (132 bytes)
- `tables/sec33_usage_rules_per_basin.csv` (1186523 bytes)
- `tables/sec33_usage_rules_summary.csv` (5285 bytes)
