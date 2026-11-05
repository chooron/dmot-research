#!/usr/bin/env python3
"""sec36_03_provenance.py — Generate PROVENANCE.md for sec36_size_matched_null (H1-H2, POST-FREEZE family)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"


def main() -> None:
    meta = pd.read_csv(TABLES / "sec36_h1_run_meta.csv").iloc[0]
    h1 = pd.read_csv(TABLES / "sec36_h1_size_matched_null_summary.csv", dtype={"direction_definition": str})
    h2 = pd.read_csv(TABLES / "sec36_h2_type_cover_absolute.csv")
    h2c = pd.read_csv(TABLES / "sec36_h2_vs_g3_comparison.csv").iloc[0]

    main = h1[h1.tier == "main_95"]
    sec90 = h1[h1.tier == "secondary_90"]
    sec100 = h1[h1.tier == "secondary_100"]

    md = [
        "# Unit Provenance: sec36_size_matched_null  (POST-FREEZE analysis family, H1-H2)",
        "",
        "## 0. POST-FREEZE registration",
        "- Created **after the freeze** as a new analysis family. It reads only frozen artifacts (the frozen",
        "  equivalence-set table and the external type assignment) and adds new tables; no frozen value, decision,",
        "  protocol or existing numeric table is modified.",
        "- Manifest rows carry the status **`POST-FREEZE`**; in the exported lists they are citable but always carry",
        "  `freeze_phase = post_freeze` (the marker is never erased or merged).",
        "- The external classification is used strictly as a grouping factor; no hydrological meaning is attached to it.",
        "- No equivalence judgement is recomputed: `t1_equivalence_sets.csv` is used as frozen.",
        "",
        "## 1. H1 — size-matched null distribution",
        "- **Question form**: is a type's minimal cover size lower than the cover size obtained from a random draw of the",
        "  **same number of basins** from the same pool?",
        f"- **Sampling pool**: the **{int(meta['matched'])} type-matched basins** used by G1-G4 (same source as the types);",
        f"  the {int(meta['unmatched'])} unmatched basins are excluded (they carry no type).",
        f"- **Main tier**: 95% cover, three conditions x 10 types, **R={int(meta['main_R'])}** draws without replacement per cell.",
        f"- **Secondary tier**: 90% and 100% cover, same structure, **R={int(meta['secondary_R'])}**.",
        f"- **Solver**: `{meta['solver']}`; every draw is solved exactly (no approximation, no reduced R).",
        f"- **Random seeds**: main tier `{int(meta['seed'])}`, secondary tier `{int(meta['seed_secondary'])}`",
        "  (one generator per tier, cells visited in the order condition = IC, seen-dPL, OOB-dPL x type ascending).",
        f"- **Direction definition**: {meta['direction_definition']}",
        f"- **Non-optimal solves**: {int(meta['nonoptimal_solves'])} (none were replaced by approximate values).",
        f"- **Solve counts**: {int(meta['total_milp_solves'])} exact MILP solves "
        f"({int(meta['total_null_draws'])} null draws + {int(meta['total_observed_solves'])} observed solves); "
        "wall-clock time is reported in the run log only and is not persisted, keeping every output deterministic.",
        "",
        "### Main-tier results (95%, R=1000)",
    ]
    for c in ["IC", "seen-dPL", "OOB-dPL"]:
        sub = main[main.condition == c]
        md.append(f"- **{c}**: " + " | ".join(
            f"t{int(r['type'])} obs={int(r['observed_cover'])} null_med={r['null_median']:.1f} "
            f"[P5={r['null_p5']:.1f}, P95={r['null_p95']:.1f}] p_le={r['one_sided_empirical_percentile']:.3f} (n={int(r['n_type'])})"
            for _, r in sub.iterrows()))
    md.append("")
    md.append("### Secondary-tier results (90% and 100%, R=200)")
    for label, sub_all in [("90%", sec90), ("100%", sec100)]:
        for c in ["IC", "seen-dPL", "OOB-dPL"]:
            sub = sub_all[sub_all.condition == c]
            md.append(f"- **{c} {label}**: " + " | ".join(
                f"t{int(r['type'])} obs={int(r['observed_cover'])} null_med={r['null_median']:.1f} "
                f"p_le={r['one_sided_empirical_percentile']:.3f}"
                for _, r in sub.iterrows()))
    md.extend([
        "",
        "## 2. H2 — absolute per-type cover counts",
        f"- Re-derived from the frozen equivalence sets with the same exact MILP: {len(h2)} cells",
        "  (3 conditions x 10 types x 90/95/100%), each with the type's N and the budget-minus-one certificate.",
        f"- Optimality: `proves_k_minus_1_insufficient` true in {int(h2.proves_k_minus_1_insufficient.sum())}/{len(h2)} cells;",
        f"  solver status 0 in {int((h2.solver_status == 0).sum())}/{len(h2)}.",
        f"- Absolute agreement with the G3 table: {int(h2c['n_rows_compared'])} rows compared, "
        f"max abs diff {float(h2c['max_abs_diff_min_models']):.3e} on the cover counts and "
        f"{float(h2c['max_abs_diff_certificate']):.3e} on the certificates, {int(h2c['n_rows_with_any_diff'])} rows with any difference.",
        "",
        "### Absolute minimal cover counts (type -> models / type N)",
    ])
    for c in ["IC", "seen-dPL", "OOB-dPL"]:
        for tgt in ["90%", "95%", "100%"]:
            sub = h2[(h2.condition == c) & (h2.target_pct == tgt)]
            md.append(f"- **{c} {tgt}**: " + " | ".join(
                f"t{int(r['type'])}:{int(r['min_models_absolute'])}/{int(r['n_type_basins'])}" for _, r in sub.iterrows()))
    md.extend([
        "",
        "## 3. H3 — manifest list handling (see `_manifest/PROVENANCE.md`)",
        "- The sec35 POST-FREEZE rows were moved from the excluded list into the citable list; the H1/H2 rows added",
        "  here are likewise citable. Both lists gained a `freeze_phase` column (`pre_freeze` / `post_freeze`).",
        "- The remaining exclusion reasons (`LOADED`, `SUPERSEDED`, `SENSITIVITY`) and their wording are unchanged.",
        "",
        "## 4. Output tables",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")
    md.extend([
        "",
        "## 5. Compliance notes",
        "- No hydrological model, CMA-ES run, dPL training or recommender fit was re-run; no equivalence rule was recomputed.",
        "- No type was merged, dropped or re-clustered; small types are reported with their own N.",
        "- All conditions, all types and all three coverage levels are reported, without selecting by significance.",
    ])
    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[sec36_03] PASS: generated PROVENANCE.md for sec36_size_matched_null")


if __name__ == "__main__":
    main()
