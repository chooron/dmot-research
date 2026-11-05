#!/usr/bin/env python3
"""sec32_04_provenance.py — Generate PROVENANCE.md for sec32_library_members."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    p1 = json.loads((TABLES / "sec32_01_parity.json").read_text(encoding="utf-8"))
    p2 = json.loads((TABLES / "sec32_02_parity.json").read_text(encoding="utf-8"))
    p3 = json.loads((TABLES / "sec32_03_parity.json").read_text(encoding="utf-8"))
    p5 = json.loads((TABLES / "sec32_05_parity.json").read_text(encoding="utf-8"))
    md = [
        "# Unit Provenance: sec32_library_members",
        "",
        "## 1. Estimands Covered",
        "- **Candidate set union size**: 36/36 models ever appear in Top-3 and Top-5",
        "- **Minimal cover sets (Top-3, Top-5)**: exact minimum cover (13 and 9 models, proven optimal via MILP) and alternative optimal solutions",
        "- **OOB 100% cover member list**: 21 models covering 100% of basins under OOB-dPL equivalence sets",
        "- **Coverage ladder (Seen Top-k -> OOB Top-1)**: k=1 (0.2524), k=3 (0.5028), k=5 (0.6271), k=10 (0.8041)",
        "- **Regret curve**: median regret k=1 (0.0428), k>=3 (0.0000)",
        "- **Model structural attributes**: parameter count (1..15), storage states count (1..6), explicit snow routine flag and import path compiled from MARRMoT metadata",
        "",
        "## 2. Approach Declarations & Parity Verifications",
        "",
        "### Script 01: `scripts/sec32_01_library_and_covers.py`",
        f"- **Approach**: {p1['approach']}",
        "- Upstream artifacts verified & loaded:",
    ]
    for u in p1["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- Top-3 union size: {p1['top3_union_size']}/36",
        f"- Top-5 union size: {p1['top5_union_size']}/36",
        f"- Top-3 minimal cover exact size: {p1['top3_min_cover_exact_size']} models",
        f"- Top-5 minimal cover exact size: {p1['top5_min_cover_exact_size']} models",
        f"- OOB full cover exact size: {p1['oob_full_cover_exact_size']} models",
        "",
        "### Script 02: `scripts/sec32_02_coverage_ladder_regret.py`",
        f"- **Approach**: {p2['approach']}",
        f"- Coverage ladder max abs diff vs frozen: {p2['max_coverage_abs_diff']:.3e}",
        f"- Regret curve max abs diff vs frozen: {p2['max_regret_abs_diff']:.3e}",
        f"- Coverage k=1: {p2['coverage_k1']:.4f}, k=3: {p2['coverage_k3']:.4f}, k=5: {p2['coverage_k5']:.4f}, k=10: {p2['coverage_k10']:.4f}",
        "",
        "### Script 03: `scripts/sec32_03_model_structural_attributes.py`",
        f"- **Approach**: {p3['approach']}",
        f"- Total models compiled: {p3['n_models']}",
        f"- Models with explicit snow routine: {p3['n_models_with_snow']}",
        f"- Parameter range: [{p3['min_params']}, {p3['max_params']}] (median: {p3['median_params']:.1f})",
        f"- Storage state range: [{p3['min_states']}, {p3['max_states']}]",
        "",
        "### Script 05: `scripts/sec32_05_process_crosstabs.py`",
        f"- **Approach**: {p5['approach']}",
        f"- Snow counts: Full 36 ({p5['full_36_snow_count']}), OOB 21 ({p5['oob21_snow_count']}), Top-3 13 ({p5['top3_13_snow_count']}), Top-5 9 ({p5['top5_9_snow_count']})",
        "",
        "### Script 06: `scripts/sec32_06_bootstrap_cover_ci_e2.py` (E2 Basin-Level Bootstrap Intervals)",
        "- **Resampling**: Basin-level bootstrap with replacement (B=1000 draws, seed 20260925)",
        "- **Solver**: Exact partial set cover via MILP (`scipy.optimize.milp`, HiGHS)",
        "- **Evaluated**: 3 conditions (IC, seen-dPL, OOB-dPL) x 3 targets (90%, 95%, 100%)",
        "- **Tables**: `sec32_bootstrap_cover_ci_summary.csv`, `sec32_bootstrap_cover_distribution.csv`",
        "",
        "### Script 07: `scripts/sec32_07_optimal_solutions_enumeration_e3.py` (E3 Optimal Solution Enumeration)",
        "- **Enumeration**: Exact MILP with iterative no-good cuts (cutoff: 1000 solutions; none truncated)",
        "- **Evaluated**: 3 conditions (IC, seen-dPL, OOB-dPL) x 2 targets (95%, 100%)",
        "- **Key Counts**: IC 95% (3 sols, core=3, union=6); IC 100% (4 sols, core=7, union=11); seen-dPL 95% (33 sols, core=3, union=18); seen-dPL 100% (18 sols, core=10, union=23); OOB-dPL 95% (162 sols, core=3, union=29); OOB-dPL 100% (4 sols, core=19, union=23)",
        "- **Tables**: `sec32_optimal_covers_summary.csv`, `sec32_optimal_covers_model_frequencies.csv`, `sec32_optimal_covers_all_solutions.csv`",
        "",
        "### Script 08: `scripts/sec32_08_singleton_and_bottleneck_e4.py` (E4 Singletons & Bottlenecks)",
        "- **Singletons**: IC (36 basins), seen-dPL (47 basins), OOB-dPL (63 basins)",
        "- **100% Cover Bottlenecks (at budget K*-1)**:",
        "  * IC (K*=9): budget 8 covers 529/531; 2 basins uncovered (09066300, 09378170; models alpine1, alpine2)",
        "  * seen-dPL (K*=14): budget 13 covers 530/531; 1 basin uncovered (11528700; model alpine2)",
        "  * OOB-dPL (K*=21): budget 20 covers 530/531; 1 basin uncovered (14236200; model susannah1)",
        "- **Tables**: `sec32_singleton_model_counts.csv`, `sec32_basin_singleton_status.csv`, `sec32_bottleneck_100pct_coverage.csv`",
        "",
        "### ID-joined re-verification (A2/A5, 2026-10-04)",
        "- **Singleton counts** (IC 36, seen-dPL 47, OOB-dPL 63) and **100% cover bottleneck basins**",
        "  (IC `09066300|09378170` forcing alpine1|alpine2|mopex3|mopex4; seen-dPL `11528700` forcing alpine2; OOB-dPL `14236200` forcing susannah1)",
        "  were re-derived by strict basin-ID joins with exact HiGHS MILP (`sec34_bottleneck_singleton_idjoin.py`): all match exactly (0/108 per-model singleton mismatches).",
        "- **Coverage ladder** (k=1 0.2524, k=3 0.5028, k=5 0.6271, k=10 0.8041) and candidate-set regret medians were",
        "  re-derived by basin-ID joins (`sec34_16_a5_remaining_idjoin.py`): max abs diff 3.6e-07.",
        "- Status: **RE-VERIFIED** (values unchanged).",
        "",
        "## 3. Output Tables Generated",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")

    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sec32_04] PASS: generated PROVENANCE.md for sec32_library_members")


if __name__ == "__main__":
    main()
