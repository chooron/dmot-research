#!/usr/bin/env python3
"""sec34_05_provenance.py — Generate PROVENANCE.md for sec34_determinacy_and_bounds."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    p1 = json.loads((TABLES / "sec34_01_parity.json").read_text(encoding="utf-8"))
    p2 = json.loads((TABLES / "sec34_02_parity.json").read_text(encoding="utf-8"))
    p3 = json.loads((TABLES / "sec34_03_parity.json").read_text(encoding="utf-8"))
    p4 = json.loads((TABLES / "sec34_04_parity.json").read_text(encoding="utf-8"))

    md = [
        "# Unit Provenance: sec34_determinacy_and_bounds",
        "",
        "## 1. Estimands Covered",
        "- **Parameter count vs parameter instability**: fold variability (+0.522, p=0.0015), per-parameter displacement (+0.435, p=0.0076), total Euclidean displacement (+0.668, p=0.0002)",
        "- **Parameter displacement vs OOB transfer loss**: pooled (+0.2612, p=0.0001), within-model median (+0.2848, 36/36 positive), model-level (+0.3459, p=0.0392)",
        "- **8 models x 3 seeds parameter dispersion**: D_self median (0.0560), D_cross median (0.3767), ratio (14.9%)",
        "- **Penalty concentration**: top 5% basins share of total sum(Delta_KGE) (34.3%), top 5 models share (19.4%)",
        "- **15 basin attributes screening**: Spearman rho, permutation p, BH-FDR q (0/15 survive q<0.05, min q=0.0754)",
        "",
        "## 2. Approach Declarations & Parity Verifications",
        "",
        "### Script 01: `scripts/sec34_01_complexity_vs_instability.py`",
        f"- **Approach**: {p1['approach']}",
        "- Upstream artifacts verified & loaded:",
    ]
    for u in p1["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- n_parameters vs fold instability rho: {p1['nparam_vs_fold_instability_rho']:+.4f} (perm p = {p1['nparam_vs_fold_instability_perm_p']:.4f})",
        f"- n_parameters vs per-param displacement rho: {p1['nparam_vs_disp_rms_rho']:+.4f} (perm p = {p1['nparam_vs_disp_rms_perm_p']:.4f})",
        f"- n_parameters vs Euclidean displacement rho: {p1['nparam_vs_disp_euclid_rho']:+.4f} (perm p = {p1['nparam_vs_disp_euclid_perm_p']:.4f})",
        f"- Pooled displacement vs OOB loss rho: {p1['pooled_disp_vs_loss_rho']:+.4f}",
        "",
        "### Script 02: `scripts/sec34_02_multiseed_dispersion.py`",
        f"- **Approach**: {p2['approach']}",
        "- Upstream artifacts verified & loaded:",
    ])
    for u in p2["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- D_self median across 8 models: {p2['d_self_median_across_models']:.4f}",
        f"- D_cross median across 8 models: {p2['d_cross_median_across_models']:.4f}",
        f"- D_self / D_cross ratio median: {p2['self_to_cross_ratio_median']:.1%}",
        f"- Models with D_self << D_cross: {p2['models_self_much_smaller_than_cross']}/8",
        "",
        "### Script 03: `scripts/sec34_03_penalty_concentration.py`",
        f"- **Approach**: {p3['approach']}",
        f"- Recomputed top 5% basins share of total Delta_KGE: {p3['top5pct_basin_share']:.4f} vs frozen {p3['frozen_top5pct_basin_share']:.4f} (abs diff = {p3['top5pct_basin_share_abs_diff']:.3e})",
        f"- Recomputed top 5 models share: {p3['top5_model_share']:.4f} vs frozen {p3['frozen_top5_model_share']:.4f} (abs diff = {p3['top5_model_share_abs_diff']:.3e})",
        f"- Worst single model share ({p3['worst_model']}): {p3['worst_model_share']:.4f}",
        "",
        "### Script 04: `scripts/sec34_04_attribute_screening.py`",
        f"- **Approach**: {p4['approach']}",
        "- Upstream artifacts verified & loaded:",
    ])
    for u in p4["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- Variables evaluated: {p4['n_variables']}",
        f"- Minimum BH-FDR q-value: {p4['min_q_value']:.4f}",
        f"- Variables surviving q < 0.05: {p4['n_surviving_fdr_005']}/15",
        f"- Top associated attribute: {p4['top_attribute']} (Spearman rho = {p4['top_attribute_rho']:+.4f})",
        "",
        "### Script 06: `scripts/sec34_06_parsimony_penalty_e5.py` (E5 Parsimony Cost in Equivalence Sets)",
        "- **Protocol**: Compares training-ranking leader vs most parsimonious member within each basin's equivalence set",
        "- **Evaluated**: OOB-dPL, seen-dPL, IC (N=531 basins each)",
        "- **Results**:",
        "  * OOB-dPL: mean diff = +0.05128 [95% CI +0.03721, +0.06734], median = +0.01537, Wilcoxon p = 1.11e-13, coincided = 97/531 (18.3%), ties broken = 75, param diff median = 6.0 [IQR 2.0-9.0]",
        "  * seen-dPL: mean diff = +0.07447 [95% CI +0.06219, +0.08759], median = +0.03506, Wilcoxon p = 4.61e-36, coincided = 62/531 (11.7%), ties broken = 79, param diff median = 8.0 [IQR 3.0-11.0]",
        "  * IC: mean diff = +0.01047 [95% CI +0.00734, +0.01361], median = +0.00268, Wilcoxon p = 5.48e-11, coincided = 76/531 (14.3%), ties broken = 135, param diff median = 8.0 [IQR 5.0-10.0]",
        "- **Tables**: `sec34_parsimony_penalty_summary.csv`, `sec34_parsimony_penalty_per_basin.csv`",
        "",
        "### Script 07: `scripts/sec34_07_partial_correlation_e6.py` (E6 Partial Correlation Controlling KGE)",
        "- **Protocol**: Partial Spearman rank correlation between n_parameters and 3 instability metrics controlling for median KGE (N_PERM=10000, seed=20260930)",
        "- **Evaluated Metrics**: Fold instability (per-param RMS), Seen->OOB displacement (per-param RMS), Seen->OOB displacement (total Euclidean)",
        "- **Results**: Controlling for either seen_kge_median or oob_kge_median retains strong, statistically significant positive correlations (all p < 0.02)",
        "- **Table**: `sec34_partial_correlation_nparam_vs_instability.csv`",
        "",
        "### Script 08: `scripts/sec34_08_displacement_correlation_recompute_e7.py` (E7 Correlation Matrix & Recomputation)",
        "- **3x3 Spearman Matrix**: All three instability metrics inter-correlate strongly (rho > 0.90)",
        "- **Independent Recomputation**: Directly parses 36 raw `params/*.npz` parameter vectors without importing analysis code",
        "- **Verification**: Exact match vs frozen benchmark (max abs diff on rho = 0.000e+00, max abs diff on perm p < 1e-18)",
        "- **Tables**: `sec34_displacement_metrics_correlation_matrix.csv`, `sec34_displacement_metrics_correlation_pvalues.csv`, `sec34_nparam_instability_independent_recomputation.csv`",
        "",
        "### Script 09: `scripts/sec34_09_equivalence_dispersion_c4.py` (C4 Equivalence Set Point KGE Dispersion)",
        "- **Metrics Evaluated**: Point KGE Range (max - min), Interquartile Range (IQR), Equivalence Set Size (|S_b|), and Best-minus-worst member point KGE diff",
        "- **Results**:",
        "  * OOB-dPL: Range median = 0.1163 [Q25=0.0413, Q75=0.2162, P90=0.3745]; IQR median = 0.0464 [Q25=0.0189, Q75=0.0866]; Set size median = 4.0 [Q25=2.0, Q75=6.0]",
        "  * seen-dPL: Range median = 0.1027 [Q25=0.0402, Q75=0.1903, P90=0.3476]; IQR median = 0.0374 [Q25=0.0167, Q75=0.0758]; Set size median = 4.0 [Q25=3.0, Q75=7.0]",
        "  * IC: Range median = 0.0549 [Q25=0.0318, Q75=0.0819, P90=0.1252]; IQR median = 0.0208 [Q25=0.0111, Q75=0.0342]; Set size median = 7.0 [Q25=4.0, Q75=15.0]",
        "- **Tables**: `sec34_equivalence_set_dispersion_summary.csv`, `sec34_equivalence_set_dispersion_per_basin.csv`",
        "",
        "### Script 10: `scripts/sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py` (K1-K5 Diagnostic Audit)",
        "- **Audit Scope**: Verifies mathematical inequality range_b <= Delta_b and penalty_b <= Delta_b across IC, seen-dPL, and OOB-dPL",
        "- **Findings**:",
        "  * IC: 0/531 violations (0.0%); max ratio = 1.0000 (inequality strictly satisfied)",
        "  * seen-dPL: 330/531 violations (62.1%) in existing table; penalty violations: 195/531 (36.7%)",
        "  * OOB-dPL: 331/531 violations (62.3%) in existing table; penalty violations: 159/531 (29.9%)",
        "- **Root Cause Identified (K4)**: Implementation error (basin-order indexing misalignment in sec34_09 and sec34_06). `basins_str` was generated from unsorted `canonical_ids()`, while `oob_mat` and `seen_mat` have rows in sorted basin order. For 439/531 basins, `sub_eq.loc[bid]` extracted equivalence models from basin A but evaluated them on KGEs of basin B.",
        "- **Single-Source Recomputation (K3)**: When correctly aligned, 0/531 violations exist (max excess <= 0). True range medians are: IC 0.0549 (<= 0.0623), seen-dPL 0.0420 (<= 0.0524), OOB-dPL 0.0403 (<= 0.0577).",
        "- **Tables**: `sec34_k1_constraint_audit_summary.csv`, `sec34_k1_constraint_violations_detail.csv`, `sec34_k2_provenance_comparison.csv`, `sec34_k3_single_source_recomputation.csv`, `sec34_k4_hypotheses_evaluation.csv`, `sec34_k5_parsimony_penalty_constraint_audit.csv`",
        "",
        "### Script 11: `scripts/sec34_11_audit_basin_alignment_a1_a2.py` (A1 & A2 Basin Indexing Audit)",
        "- **A1 Static Audit**: Enumerated 23 basin-level positional indexing locations across manuscript scripts and loaded upstream modules; 18 ALIGNED, 5 MISALIGNED (all traced to using unsorted `canonical_ids()` as numerical row index for sorted performance matrices).",
        "- **A2 Empirical Audit**: Checked 8 basin-level tables (34,445 total rows) against raw upstream data joined strictly by basin ID; 5 tables PASS (0 mismatches), 3 tables FAIL_MISMATCH (mismatches confined exclusively to `sec31_uncertainty_bandwidth_per_basin.csv` OOB rows, `sec34_equivalence_set_dispersion_per_basin.csv` seen/oob rows, and `sec34_parsimony_penalty_per_basin.csv` seen/oob rows).",
        "- **Tables**: `sec34_a1_static_basin_indexing_audit.csv`, `sec34_a2_table_alignment_audit.csv`",
        "",
        "### Script 12: `scripts/sec34_12_basin_audit_supplement.py` (A1 detail + A2 checklist + extended per-basin checks)",
        "- **A1 detail**: the 5 MISALIGNED locations and their affected output tables are itemized in `sec34_a1_misalignment_detail.csv`.",
        "  * Beyond sec34_09 and sec34_06, the other 3 are: sec31_07 line 124 (affects `sec31_jackknife_equivalence_sets.csv`, OOB-dPL rows),",
        "    sec31_08 line 158 (affects `sec31_multiyear_block_bootstrap_equivalence_sets.csv`, OOB-dPL rows), and",
        "    sec31_09 line 128 (affects `sec31_uncertainty_bandwidth_per_basin.csv`, OOB-dPL rows).",
        "- **A2 extended checks**: rules-ladder per-basin basis tables recomputed from raw Qsim/obs by basin ID —",
        "  `sec33_fixed_ensemble_per_basin.csv` all k (1,2,3,4,5,7,10) and `sec33_usage_rules_per_basin.csv`",
        "  (A, B3, B5, C_A2d x10 seeds, C_A1b x10 seeds, D_A2d x10 seeds, D_A1b x10 seeds, E): all PASS (max abs diff 5.0e-07).",
        "- **15-attribute screening per-basin penalty**: `basin_attribute_table.csv` `penalty_median_pub` recomputed by basin-ID join: PASS (max abs diff 3.3e-16).",
        "- **Note**: the rules-ladder Rule C k=2 and k=5 per-basin series are computed in-script and have no stored per-basin table.",
        "- **Tables**: `sec34_a1_misalignment_detail.csv`, `sec34_a2_table_checklist.csv`, `sec34_a2_extended_checks.csv`, `sec34_recompute_rule_c_per_basin_idjoin.csv`.",
        "",
        "### Script 13: `scripts/sec34_13_attribute_screening_idjoin_a4.py` (A4 15-attribute screening re-verification)",
        "- **Method**: response = per-basin median of `delta_kge_pub_median4` over models (ID-joined from `pub_same_pipeline_model_basin.csv`);",
        "  10 Caravan attributes (ID-joined on `gauge_id`) + 5 observed-series attributes (obs assembled by basin-ID join across the 5 fold files).",
        "- **Test protocol**: Spearman rho + permutation p with N_PERM=10000, seed=20260930 (same as the frozen upstream implementation); BH-FDR over the 15-test family.",
        "- **Result**: all 15 variables match the existing table (max abs diff 5.0e-07); 0/15 survive q<0.05; min q = 0.075367; max |rho| = 0.119781.",
        "- **Tables**: `sec34_a4_attribute_screening_recompute.csv`, `sec34_a4_attribute_screening_summary.csv`.",
        "",
        "### Script 14: `scripts/sec34_14_bottleneck_singleton_idjoin.py` (bottleneck and singleton re-determination by basin ID)",
        "- **Bottleneck (100% cover)**: IC uncovered `09066300|09378170` (forcing alpine1|alpine2|mopex3|mopex4);",
        "  seen-dPL uncovered `11528700` (forcing alpine2); OOB-dPL uncovered `14236200` (forcing susannah1) — all match the existing values exactly.",
        "- **Singleton totals**: IC 36, seen-dPL 47, OOB-dPL 63 — match existing (per-model mismatches 0/108).",
        "- **Tables**: `sec34_bottleneck_idjoin_check.csv`, `sec34_singleton_idjoin_check.csv`, `sec34_singleton_totals_idjoin_check.csv`.",
        "",
        "### Script 15: `scripts/sec34_15_a3_recompute_parsimony_dispersion_idjoin.py` (A3 ID-joined recomputation; SUPERSEDES sec34_06 / sec34_09 outputs)",
        "- **Parsimony cost (ranking leader minus most-parsimonious member)**:",
        "  * IC: +0.010473 (identical to the old table; IC was unaffected).",
        "  * seen-dPL: +0.011667 (old +0.074470; abs diff 0.062805) — old table SUPERSEDED.",
        "  * OOB-dPL: +0.006157 (old +0.051277; abs diff 0.045120) — old table SUPERSEDED.",
        "- **Within-equivalence-set point-KGE range (median)**:",
        "  * IC: 0.054913 (identical to the old table).",
        "  * seen-dPL: 0.042026 (old 0.102700; abs diff 0.060674) — old table SUPERSEDED.",
        "  * OOB-dPL: 0.040345 (old 0.116300; abs diff 0.075955) — old table SUPERSEDED.",
        "- **Supersession reason**: `sec34_06` (line 92) and `sec34_09` (line 75) used the unsorted `canonical_ids()` order as a numerical row index into",
        "  SORTED performance matrices (`oob_mat`, `seen_mat`), so equivalence sets of basin A were evaluated on KGEs of basin B at 439/531 positions.",
        "  The IC condition was unaffected because its cube was built from the same unsorted list.",
        "- **Old tables retained unmodified**: `sec34_parsimony_penalty_summary.csv`, `sec34_parsimony_penalty_per_basin.csv`,",
        "  `sec34_equivalence_set_dispersion_summary.csv`, `sec34_equivalence_set_dispersion_per_basin.csv` (marked SUPERSEDED in the manifest).",
        "- **Tables**: `sec34_a3_parsimony_penalty_idjoin_summary.csv`, `sec34_a3_parsimony_penalty_idjoin_per_basin.csv`,",
        "  `sec34_a3_dispersion_idjoin_summary.csv`, `sec34_a3_dispersion_idjoin_per_basin.csv`, `sec34_a3_parsimony_penalty_comparison.csv`,",
        "  `sec34_a3_dispersion_comparison.csv`, `sec34_a3_per_basin_mismatch_summary.csv`.",
        "",
        "### Script 16: `scripts/sec34_16_a5_remaining_idjoin.py` (A5 remaining ID-joined re-verifications)",
        "- **Cluster matching**: 515/531 matched; unmatched ID set identical to the existing audit.",
        "- **Cluster leave-out validation**: k=3 global coverage 0.4311 / cluster 0.4252; k=5 global 0.5204 / cluster 0.5126;",
        "  diff regret -0.00405 (k=3) and -0.00402 (k=5) vs existing -0.00400 (max abs diff 4.7e-05).",
        "- **Fold composition**: 78 rows (13 attributes x 6 scopes) compared, max abs diff 5.0e-07.",
        "- **Penalty concentration**: top 5% basin share 0.3425 and top-5-model share 0.1935, matching existing (max abs diff 4.7e-08).",
        "- **Coverage ladder**: k=1 0.2524, k=3 0.5028, k=5 0.6271, k=10 0.8041 — all match (max abs diff 3.6e-07).",
        "- **Tables**: `sec34_a5_fold_composition_idjoin.csv`, `sec34_a5_remaining_comparison.csv`, `sec34_a5_remaining_summary.csv`.",
        "",
        "### A6 Shared basin-ID accessor: `_shared/basin_accessor.py`",
        "- **Signature**: `canonical_basin_ids()`, `file_order_basin_ids()`, `basin_position_map()`, `pad_basin_id()`,",
        "  `align_series_by_id(ids, values, canonical=None, name=...)`, `align_frame_by_id(frame, id_column, value_columns, ...)`,",
        "  `pivot_by_id(frame, id_column, column_column, value_column, ...)`, `lookup_by_id(values_by_basin, basin_id, default=None)`,",
        "  `basin_order_mismatch_count()` (returns 439 for FILE_ORDER vs SORTED).",
        "- **Consumers**: `sec34_13` (A4), `sec34_15` (A3), `sec34_16` (A5) route all basin-ID resolution through these accessors;",
        "  all three reproduce byte-identical outputs after the switch (12/12 output files unchanged).",
        "- **No existing script was rewritten** and no existing script output was altered.",
        "",
        "## 3. Output Tables Generated",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")

    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sec34_05] PASS: generated PROVENANCE.md for sec34_determinacy_and_bounds")


if __name__ == "__main__":
    main()
