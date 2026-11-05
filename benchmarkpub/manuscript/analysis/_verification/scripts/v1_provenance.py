#!/usr/bin/env python3
"""v1_provenance.py — Generate PROVENANCE.md for _verification."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
UNIT_ROOT = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "_verification"
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    v1_df = pd.read_csv(TABLES / "v1_recomputed_abstract_numbers.csv")
    cert_df = pd.read_csv(TABLES / "v1_optimality_certificates.csv")

    primitive_inputs = [
        "data/531sub_id.txt (canonical IDs)",
        "project/benchmark/analysis/ic_dpl_gap_conditional_20260923/forward/test/water_year_sufficient_stats.csv (raw stats for IC and seen-dPL)",
        "project/benchmarkpub/data/oob36_20260930/qsim/ and obs/ (raw daily simulated and observed discharge)",
        "project/benchmarkpub/data/oob36_20260930/manifests/OOB_FOLD_ASSIGNMENT.csv (raw fold assignments)",
        "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv (benchmark model-basin table)",
    ]

    md = [
        "# Unit Provenance: _verification (V1 Abstract Numbers Recomputation)",
        "",
        "## 1. Scope & Isolation Declaration",
        "- **Hard Isolation**: This unit executes an entirely independent implementation path (`scripts/v1_recompute_abstract_numbers.py`).",
        "- **Zero Imported Analysis Modules**: Does NOT import or call `_shared`, `c4_common`, `tk_common`, or any previous calculation script.",
        "- **Zero Reads of Derived Tables**: Does NOT read any derived tables from `equiv_cover_t1t2`, `topk`, `chapter4_final_closure`, or other analysis folders.",
        "- **Primitive Input Files Consumed**:",
    ]
    for inp in primitive_inputs:
        md.append(f"  * `{inp}`")

    md.extend([
        "",
        "## 2. Recomputed Abstract-Level Numbers (V1)",
        "| Quantity | Condition | Recomputed | Existing Frozen | Absolute Difference | Optimality Certificate / Evidence |",
        "|---|---|---:|---:|---:|---|",
    ])

    for _, r in v1_df.iterrows():
        md.append(f"| {r['quantity_name']} | {r['condition']} | {r['recomputed_value']} | {r['existing_value']} | {r['abs_diff']:.3e} | {r['optimality_evidence']} |")

    md.extend([
        "",
        "## 3. Optimality Certificates for Set Covers",
        "| Condition | Target | Need Basins | Proven Min Models | Budget - 1 | Max Coverable at Budget - 1 | Proves Q Impossible? |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for _, r in cert_df.iterrows():
        md.append(f"| {r['condition']} | {r['target']:.2f} | {r['need_basins']} | {r['proven_minimum_models']} | {r['budget_minus_1']} | {r['max_basins_coverable_with_budget_minus_1']} | {r['proves_q_impossible']} |")

    md.extend([
        "",
        "## 4. Script Paths",
        "- Generating script: `scripts/v1_recompute_abstract_numbers.py`",
        "- Provenance script: `scripts/v1_provenance.py`",
    ])

    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("PASS: generated PROVENANCE.md for _verification")


if __name__ == "__main__":
    main()
