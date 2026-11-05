#!/usr/bin/env python3
"""sec33_06_provenance.py — Generate PROVENANCE.md for sec33_usage_and_conditioning."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    p1 = json.loads((TABLES / "sec33_01_parity.json").read_text(encoding="utf-8"))
    p2 = json.loads((TABLES / "sec33_02_parity.json").read_text(encoding="utf-8"))
    p3 = json.loads((TABLES / "sec33_03_parity.json").read_text(encoding="utf-8"))
    p4 = json.loads((TABLES / "sec33_04_parity.json").read_text(encoding="utf-8"))
    p5 = json.loads((TABLES / "sec33_05_parity.json").read_text(encoding="utf-8"))
    p7 = json.loads((TABLES / "sec33_07_parity.json").read_text(encoding="utf-8"))

    md = [
        "# Unit Provenance: sec33_usage_and_conditioning",
        "",
        "## 1. Estimands Covered",
        "- **T2 non-oracle usage rules A-E**: summary and per-basin evaluations with daily member mean series KGE",
        "- **T3-1 direct paired comparison**: Rule C (attribute-conditioned A2d Top-3) vs Rule B (fixed global Top-3) across 10 seeds",
        "- **T3-2 fixed ensemble size scan**: k in {1, 2, 3, 4, 5, 7, 10} with fold-wise training-only members and paired diffs vs k=1",
        "- **G_set across three estimands**: Seen-dPL (+0.1966), A2d (+0.0461), A1b (+0.0221) with 10 seeds and 3 components (mean best_R, mean best_F, mean VBS, numerator, denominator)",
        "- **Paired regret difference**: per-basin regret difference (recommendation vs fixed baseline) across Seen, A2d, and A1b",
        "- **Basin-specific vs global Top-k paired difference**: k=3 coverage gain +7.9 pp",
        "- **Cluster leave-out OOB validation**: k=3 and k=5 coverage and regret differences, plus typology organization statistics (entropy, stability)",
        "",
        "## 2. Approach Declarations & Parity Verifications",
        "",
        "### Script 01: `scripts/sec33_01_usage_rules_t2.py`",
        f"- **Approach**: {p1['approach']}",
        "- Upstream artifacts verified & loaded:",
    ]
    for u in p1["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        "",
        "### Script 02: `scripts/sec33_02_rule_c_vs_b_t3_1.py`",
        f"- **Approach**: {p2['approach']}",
        f"- T3-1 seed 20260925 paired diff mean: {p2['mean_diff_seed20260925']:+.6f} (Wilcoxon p = {p2['wilcoxon_p_seed20260925']:.5f})",
        f"- T3-1 across 10 seeds mean paired diff: {p2['mean_diff_across_10_seeds']:+.6f}",
        f"- T3-1 seeds with Wilcoxon p < 0.05: {p2['seeds_significant_p_lt_005']}/10",
        "",
        "### Script 03: `scripts/sec33_03_fixed_ensemble_scan_t3_2.py`",
        f"- **Approach**: {p3['approach']}",
        f"- Scanned k values: {p3['k_values']}",
        f"- KGE median: k=1 ({p3['kge_k1_median']:.4f}), k=3 ({p3['kge_k3_median']:.4f}), k=5 ({p3['kge_k5_median']:.4f})",
        f"- Mean diff vs k=1: k=3 ({p3['diff_k3_mean']:+.5f}), k=5 ({p3['diff_k5_mean']:+.5f})",
        "",
        "### Script 04: `scripts/sec33_04_gset_and_regret.py`",
        f"- **Approach**: {p4['approach']}",
        "- Upstream artifacts verified & loaded:",
    ])
    for u in p4["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- G_set main seed: Seen ({p4['Seen_main_G_set']:.4f}), A2d ({p4['A2d_main_G_set']:.4f}), A1b ({p4['A1b_main_G_set']:.4f})",
        f"- Paired regret diff mean A2d: {p4['regret_diff_mean_A2d']:+.5f}",
        f"- Basin-specific vs global k=3 coverage diff: {p4['k3_bs_minus_glob_cov_diff']:+.4f}",
        "",
        "### Script 05: `scripts/sec33_05_cluster_typology.py`",
        f"- **Approach**: {p5['approach']}",
        "- Upstream artifacts verified & loaded:",
    ])
    for u in p5["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- Cluster OOB coverage diff: k=3 ({p5['cluster_k3_coverage_diff']:+.4f}), k=5 ({p5['cluster_k5_coverage_diff']:+.4f})",
        f"- Cluster OOB regret diff mean: k=5 ({p5['cluster_k5_regret_diff_mean']:+.4f}, Wilcoxon p = {p5['cluster_k5_wilcoxon_p']:.4f})",
        "",
        "### Script 07: `scripts/sec33_07_diversity_sensitivity_v5.py`",
        f"- **Approach**: {p7['approach']}",
        f"- Scanned k values: {p7['k_values']}",
        f"- Structural diversity k=3 mean diff vs ranking: {p7['struct_diff_k3_mean']:+.5f}",
        f"- Behavioral diversity k=3 mean diff vs ranking: {p7['behav_diff_k3_mean']:+.5f}",
        "- Evaluation scope: SENSITIVITY",
        "",
        "### Script 08: `scripts/sec33_08_recommender_k2_k5_e8.py` (E8 Re-fit Recommenders, Parity, k=2 & k=5)",
        "- **Protocol**: Exact re-fitting of Random Forest recommenders (500 trees, min_samples_leaf=5, 35 raw Caravan attributes, snow pre-filter, 5 OOB folds, 10 seeds)",
        "- **Parity Gate**: Verified against `oob36_gset_gate_20261001` outputs; Top-3 A2d and A1b G_set matched exactly across all 10 seeds (max abs diff <= 1e-6)",
        "- **Leakage Audit**: All 5 folds verified strictly zero data leakage (target fold never observed during training)",
        "- **Operating Points k=2 and k=5**:",
        "  * Rule C (attribute-conditioned Top-k equal-weight ensemble) vs Rule B (fixed global Top-k equal-weight ensemble) evaluated via daily simulated discharge mean series KGE",
        "  * In-set best G_set evaluated for both A2d (PRIMARY) and A1b (SECONDARY) with 10,000-draw basin bootstrap 95% CIs across all 10 seeds",
        "- **Persisted Artifacts**: Full ranked model predictions (`sec33_recommender_full_rankings.npz`), `sec33_e8_parity_gate.csv`, `sec33_e8_rule_c_vs_b_k2_k5.csv`, `sec33_e8_gset_k2_k5.csv`, `sec33_e8_leakage_check.csv`",
        "",
        "### Script 09: `scripts/sec33_09_rules_ladder_c3.py` (C3 Comprehensive Rules Ladder with Dual Benchmarks)",
        "- **Rules Covered on 531 Basins**: Rule A (k=1), Rule B (k in {1, 2, 3, 4, 5, 7, 10}), Rule C (A2d k in {2, 3, 5}), Rule D (A2d k=3), Rule E (36-model oracle)",
        "- **Dual Benchmarks**: Every rule evaluated simultaneously relative to Rule A (k=1 global single best) AND relative to Rule B (k=2 fixed global ensemble)",
        "- **Statistics**: Absolute KGE median and Q25/Q75, plus paired differences (mean, median, Wilcoxon p, basin bootstrap 95% CI, water-year block bootstrap 95% CI, N=531)",
        "- **Tables**: `sec33_rules_ladder_unified.csv`, `sec33_rule_c_per_seed_ladder.csv`",
        "",
        "### ID-joined re-verification (A2/A5, 2026-10-04)",
        "- **Rules-ladder per-basin basis** recomputed from raw OOB Qsim/obs by basin ID (`sec34_12_basin_audit_supplement.py`):",
        "  `sec33_fixed_ensemble_per_basin.csv` for k=1,2,3,4,5,7,10 and `sec33_usage_rules_per_basin.csv` for A, B3, B5,",
        "  C_A2d x10 seeds, C_A1b x10 seeds, D_A2d x10 seeds, D_A1b x10 seeds, E — all PASS (max abs diff 5.0e-07).",
        "  Rule C k=2 and k=5 per-basin series have no stored per-basin table (computed in-script); the aggregate ladder values were cross-checked and PASS.",
        "- **Cluster matching** (515/531, identical unmatched ID set) and **cluster leave-out validation** coverage/regret",
        "  (k=3: 0.4311/0.4252; k=5: 0.5204/0.5126; diff regret -0.00405/-0.00402) were re-derived by basin-ID joins",
        "  (`sec34_16_a5_remaining_idjoin.py`): max abs diff 4.7e-05.",
        "- Status: **RE-VERIFIED** (values unchanged)."
        "",
        "### Script 10: `scripts/sec33_10_rule_c_k2_k5_per_basin_f1.py` (F1 Rule C k=2/k=5 per-basin series)",
        "- **Deliverable**: `sec33_rule_c_k2_k5_per_basin.csv` — 10,620 rows (10 seeds x 531 basins x k in {2,5}),",
        "  columns identical to `sec33_usage_rules_per_basin.csv` (rule, k, seed, basin_id, fold, kge, diff_vs_A);",
        "  KGE computed once on the daily equal-weight ensemble mean series (never by averaging member KGEs).",
        "- **Re-verification**: C(2)/C(5) paired differences vs A and vs B(k=2) recomputed with the T2/T3 protocol",
        "  (Wilcoxon on non-zero diffs; 1,000-draw basin bootstrap and water-year block proxy, seed 20260925, shared-RNG order",
        "  identical to sec33_09) and compared with `sec33_rule_c_per_seed_ladder.csv` / `sec33_rules_ladder_unified.csv`:",
        "  all 14 quantities agree within 5.0e-07, the 6-dp write precision of the existing tables.",
        "- **Row-by-row alignment check**: 21,240 rows (4 slices) re-derived in an independent pass from raw Qsim/obs + E8 rankings",
        "  by basin ID; 0 mismatches, max abs diff 5.0e-07.",
        "- **Tables**: `sec33_rule_c_k2_k5_per_basin.csv`, `sec33_rule_c_k2_k5_recompute_comparison.csv`, `sec33_rule_c_k2_k5_alignment_check.csv`.",
        "",
        "### Script 11: `scripts/sec33_11_constrained_fixed_ensemble_f2.py` (F2 parameter-constrained fixed global ensembles)",
        "- **Construction**: thresholds are the P25 / median / P75 of the 36-model `n_parameters` distribution (5.75 / 7.0 / 9.25);",
        "  eligible models are those with n_parameters <= threshold (9 / 20 / 27 models); members are the top-2 and top-3 eligible",
        "  models by the training-fold median OOB KGE — the same ranking statistic as the existing unconstrained B(k) protocol.",
        "  The selection never uses target-fold information; the unconstrained members reproduce `sec33_fixed_ensemble_fold_members.csv` exactly.",
        "- **Results** (paired difference constrained minus unconstrained, T2/T3 protocol, N=531):",
        "  * P25 (9 eligible): k=2 mean -0.123235, k=3 mean -0.124117;",
        "  * median (20 eligible): k=2 mean -0.045860, k=3 mean -0.039791;",
        "  * P75 (27 eligible): k=2 mean -0.060332, k=3 mean -0.034161.",
        "- **Degeneracy**: 0 of 30 fold-level tier x k cells have identical constrained/unconstrained member sets; no tier is degenerate.",
        "- **Parameter savings** (sum of member n_parameters, per-fold range): P25 k=2 18-18, k=3 20-22;",
        "  median k=2 14-14, k=3 15-18; P75 k=2 12-12, k=3 11-14.",
        "- **Leakage check**: per fold, target-fold basins are excluded from member construction (`target_fold_in_construction = NO`).",
        "- **Tables**: `sec33_constrained_ensemble_summary.csv`, `sec33_constrained_ensemble_per_basin.csv`,",
        "  `sec33_constrained_ensemble_members.csv`, `sec33_constrained_ensemble_leakage_check.csv`,",
        "  `sec33_constrained_ensemble_unconstrained_member_check.csv`.",
        "",
        "## 3. Output Tables Generated",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")

    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sec33_06] PASS: generated PROVENANCE.md for sec33_usage_and_conditioning")


if __name__ == "__main__":
    main()
