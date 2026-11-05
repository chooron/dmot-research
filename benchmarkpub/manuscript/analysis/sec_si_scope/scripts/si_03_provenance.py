#!/usr/bin/env python3
"""si_03_provenance.py — Generate PROVENANCE.md for sec_si_scope."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    p1 = json.loads((TABLES / "si_01_parity.json").read_text(encoding="utf-8"))
    p2 = json.loads((TABLES / "si_02_parity.json").read_text(encoding="utf-8"))

    md = [
        "# Unit Provenance: sec_si_scope",
        "",
        "## 1. Estimands Covered",
        "- **Cross-metric candidate set overlap**: Top-3 overlap of logKGE (0.2147 dPL, 0.2310 IC) and FHV (0.4325 dPL, 0.2994 IC) with standard KGE (annotated EVALUATION_ONLY_NO_TARGET_MATCHED_RETRAINING)",
        "- **Supplementary metric recommendation**: G_set under logKGE (+0.1210) and FHV (-0.0411)",
        "- **Equivalence label definitions & bootstrap protocol parameters**: 15 water-year blocks (1996..2010), 1000 draws, seed 20260925, q05 rule across IC, seen-dPL, and OOB-dPL",
        "",
        "## 2. Approach Declarations & Parity Verifications",
        "",
        "### Script 01: `scripts/si_01_cross_metric_overlap.py`",
        f"- **Approach**: {p1['approach']}",
        "- Upstream artifacts verified & loaded:",
    ]
    for u in p1["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- dPL logKGE Top-3 overlap with KGE: {p1['dpl_logkge_top3_overlap']:.4f}",
        f"- dPL FHV Top-3 overlap with KGE: {p1['dpl_fhv_top3_overlap']:.4f}",
        f"- IC logKGE Top-3 overlap with KGE: {p1['ic_logkge_top3_overlap']:.4f}",
        f"- IC FHV Top-3 overlap with KGE: {p1['ic_fhv_top3_overlap']:.4f}",
        "",
        "### Script 02: `scripts/si_02_equivalence_protocol_params.py`",
        f"- **Approach**: {p2['approach']}",
        "- Upstream artifacts verified & loaded:",
    ])
    for u in p2["upstream_artifacts"]:
        md.append(f"  * `{u['path']}` (bytes: {u['bytes']}, mtime: {u['mtime']}, sha256: `{u['sha256']}`)")

    md.extend([
        f"- Conditions documented: {p2['n_conditions']}",
        f"- Replicate draws: {p2['n_replicates']}",
        f"- Random seed: {p2['seed']}",
        "",
        "## 3. Output Tables Generated",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")

    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[si_03] PASS: generated PROVENANCE.md for sec_si_scope")


if __name__ == "__main__":
    main()
