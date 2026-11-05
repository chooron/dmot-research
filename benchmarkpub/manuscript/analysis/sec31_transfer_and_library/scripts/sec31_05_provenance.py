#!/usr/bin/env python3
"""sec31_05_provenance.py — Generate PROVENANCE.md for sec31_transfer_and_library."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    p1 = json.loads((TABLES / "sec31_01_parity.json").read_text(encoding="utf-8"))
    p2 = json.loads((TABLES / "sec31_02_parity.json").read_text(encoding="utf-8"))
    p3 = json.loads((TABLES / "sec31_03_parity.json").read_text(encoding="utf-8"))
    p4 = json.loads((TABLES / "sec31_04_parity.json").read_text(encoding="utf-8"))
    p6 = json.loads((TABLES / "sec31_06_parity.json").read_text(encoding="utf-8"))
    md = [
        "# Unit Provenance: sec31_transfer_and_library",
        "",
        "## 1. Estimands Covered",
        "- **ΔKGE_PUB**: pooled median & CI, 36 model-level rows, 180 model×fold rows",
        "- **Two-way ANOVA variance decomposition**: basin SS, model SS, residual SS",
        "- **Global Seen-OOB model rank preservation**: Spearman correlation across 36 models",
        "- **Per-basin Seen-OOB rank preservation**: distribution of within-basin Spearman correlations across 531 basins",
        "- **Equivalence sets & size distributions**: under IC, seen-dPL, and OOB-dPL (15-block water-year bootstrap, q05 rule)",
        "- **Coverage curves & minimal libraries**: exact (MILP) and greedy curves; 90%, 95%, 100% target cover counts and model lists",
        "",
        "## 2. Approach Declarations & Parity Verifications",
        "",
        "### Script 01: `scripts/sec31_01_delta_kge.py`",
        f"- **Approach**: {p1['approach']}",
        f"- Recomputed pooled median ΔKGE_PUB: {p1['pooled_median_recomputed']:.6f} vs frozen {p1['pooled_median_frozen']:.6f} (abs diff = {p1['pooled_median_abs_diff']:.3e})",
        f"- Model-level 36-model max abs diff vs frozen: {p1['model_level_max_abs_diff']:.3e}",
        f"- Models with positive median penalty: {p1['n_models_positive']}/36",
        f"- Model x fold cells with positive median penalty: {p1['n_fold_cells_positive']}/180",
        "",
        "### Script 02: `scripts/sec31_02_variance_decomposition.py`",
        f"- **Approach**: {p2['approach']}",
        f"- Basin fraction of SS: {p2['frac_basin']:.6f} (61.01%)",
        f"- Model fraction of SS: {p2['frac_model']:.6f} (1.02%)",
        f"- Residual fraction of SS: {p2['frac_residual']:.6f} (37.97%)",
        f"- Max abs diff vs frozen `pub_variance_decomposition.csv`: {p2['max_abs_diff_vs_frozen']:.3e}",
        "",
        "### Script 03: `scripts/sec31_03_rank_preservation.py`",
        f"- **Approach**: {p3['approach']}",
        f"- Global model rank Spearman (canonical dPL): {p3['global_spearman_canonical_recomputed']:.6f} vs frozen {p3['global_spearman_canonical_frozen']:.6f} (abs diff = {p3['global_spearman_canonical_abs_diff']:.3e})",
        f"- Global model rank Spearman (same-pipeline): {p3['global_spearman_samepipeline_recomputed']:.6f}",
        f"- Basin median rank Spearman: {p3['basin_median_spearman_recomputed']:.6f} vs frozen {p3['basin_median_spearman_frozen']:.6f} (abs diff = {p3['basin_median_spearman_abs_diff']:.3e})",
        "",
        "### Script 04: `scripts/sec31_04_equivalence_and_coverage.py`",
        f"- **Approach**: {p4['approach']}",
        "- Upstream artifacts verified & loaded:",
    ]
    for u in p4["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        "",
        f"- Cover 95% exact models count: {p4['cover95_exact']}",
        f"- Cover 100% exact models count: {p4['cover100_exact']}",
        "",
        "### Script 06: `scripts/sec31_06_fold_composition.py`",
        f"- **Approach**: {p6['approach']}",
        f"- Attributes included: {len(p6['attributes_included'])} ({', '.join(p6['attributes_included'][:5])}...)",
        f"- Fold basin counts: {p6['fold_counts']}",
        "",
        "### Script 07: `scripts/sec31_07_jackknife_equivalence_e1.py` (E1 Alternative Uncertainty Convention)",
        "- **Status**: SUPERSEDED (INVALID IMPLEMENTATION)",
        "- **Supersession Reason**: 留一水文年 jackknife 的 15 个伪值按构造远比 bootstrap 重抽值分散度小，对其直接取 q05 分位得到的不确定带过窄（中位宽度 0.012-0.015 vs bootstrap 0.052-0.062），因而所需模型数被系统性放大。保留作为对比基准，改标为 SUPERSEDED，不进入主引述行。",
        "- **Parameters**:",
        "  * Total water years: 15 (WY 1996..2010)",
        "  * Resampling unit: 1 leave-out water year per replicate (15 jackknife replicates total)",
        "  * Pooled evaluation: 14 pooled water years per replicate",
        "",
        "### Script 08: `scripts/sec31_08_multiyear_block_bootstrap_c1.py` (C1 Multi-Year Block Bootstrap Active Alternative)",
        "- **Scheme**: Non-overlapping multi-year block bootstrap (L=3 years, 5 blocks total; Clark et al. 2021 NBB methodology)",
        "- **Implementation Details**:",
        "  * Why L=3: Preserves multi-year catchment storage and interannual autocorrelation; divides 15 test years into exactly 5 equal 3-year blocks (1996-1998, 1999-2001, 2002-2004, 2005-2007, 2008-2010).",
        "  * Replicates: 1000 bootstrap draws with replacement from the 5 blocks (seed 20260925).",
        "  * Metric: KGE from sufficient statistics (eps=0.1, sample variance).",
        "  * Equivalence threshold rule: q05_b := 5th percentile of {best_r(b)} across 1000 draws.",
        "  * Equivalence set definition: S_b := { m : KGE^point_{m,b} >= q05_b }.",
        "  * Coverage targets: 90% (478 basins), 95% (505 basins), 100% (531 basins), exact HiGHS MILP.",
        "- **Optimality Certificates**: HiGHS solver status 0; budget K*-1 proven insufficient for all targets.",
        "- **Tables**: `sec31_multiyear_block_bootstrap_comparison.csv`, `sec31_multiyear_block_bootstrap_certificates.csv`, `sec31_multiyear_block_bootstrap_size_distribution.csv`, `sec31_multiyear_block_bootstrap_equivalence_sets.csv`.",
        "",
        "### Script 09: `scripts/sec31_09_uncertainty_bandwidth_comparison_c2.py` (C2 Bandwidth Comparison Across 3 Schemes)",
        "- **Bandwidth Definition**: Delta_b := KGE_best^point - q05_b (tolerance offset from best model's point KGE).",
        "- **Schemes Compared**:",
        "  * Primary 1-year block bootstrap (median: IC 0.0623, seen-dPL 0.0524, OOB-dPL 0.0577)",
        "  * Jackknife (SUPERSEDED, median: IC 0.0145, seen-dPL 0.0119, OOB-dPL 0.0138; overly narrow)",
        "  * C1 3-year block bootstrap (median: IC 0.0604, seen-dPL 0.0501, OOB-dPL 0.0535; stable multi-year uncertainty)",
        "- **Tables**: `sec31_uncertainty_bandwidth_comparison.csv`, `sec31_uncertainty_bandwidth_per_basin.csv`."
        "",
        "### Basin-order index audit note (A1/A2, 2026-10-04)",
        "- **sec31_07 (line 124)**: `b_idx` from unsorted `S.canonical_ids()` was used to label rows of the SORTED OOB cube.",
        "  Affected output: `sec31_jackknife_equivalence_sets.csv` (OOB-dPL `basin_id` labels mismatched at 439/531 positions).",
        "  The table is already SUPERSEDED for the invalid jackknife construction; its OOB basin labels are additionally misassigned.",
        "- **sec31_08 (line 158)**: same pattern. Affected output: `sec31_multiyear_block_bootstrap_equivalence_sets.csv`",
        "  (OOB-dPL `basin_id` labels mismatched). Set-cover counts are permutation-invariant and were RE-VERIFIED unchanged.",
        "- **sec31_09 (line 128)**: same pattern. Affected output: `sec31_uncertainty_bandwidth_per_basin.csv`",
        "  (OOB-dPL rows carry the bandwidth of a different basin at 439/531 positions; the per-row values themselves are internally consistent).",
        "  The aggregate bandwidth medians in `sec31_uncertainty_bandwidth_comparison.csv` were RE-VERIFIED by single-source recompute (abs diff 0.0).",
        "- **Not affected**: sec31_01..04, sec31_06 (all use `S.basin_order()` or basin-ID joins); `sec31_equivalence_sets_per_basin.csv` PASSED the A2 ID-joined check."
        "",
        "## 3. Output Tables Generated",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")

    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sec31_05] PASS: generated PROVENANCE.md for sec31_transfer_and_library")


if __name__ == "__main__":
    main()
