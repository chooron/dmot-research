#!/usr/bin/env python3
"""sec35_05_provenance.py — Generate PROVENANCE.md for sec35_cluster_stratified (POST-FREEZE analysis family)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
HERE = Path(__file__).resolve().parent
UNIT_ROOT = HERE.parent
TABLES = UNIT_ROOT / "tables"
GLOBAL_95 = {"IC": 4, "seen-dPL": 6, "OOB-dPL": 12}


def main() -> None:
    g1 = pd.read_csv(TABLES / "sec35_g1_kruskal_wallis.csv", dtype={"p_value_scientific": str}).iloc[0]
    g1s = pd.read_csv(TABLES / "sec35_g1_matched_summary.csv").iloc[0]
    g1t = pd.read_csv(TABLES / "sec35_g1_penalty_by_type.csv")
    g1p = pd.read_csv(TABLES / "sec35_g1_pairwise_dunn_holm.csv")
    g1b = pd.read_csv(TABLES / "sec35_g1b_attribute_penalty_spearman_by_type.csv")
    g2 = pd.read_csv(TABLES / "sec35_g2_fisher_exact.csv").iloc[0]
    g3k = pd.read_csv(TABLES / "sec35_g3_equivalence_size_kruskal.csv", dtype={"p_value_scientific": str})
    g3c = pd.read_csv(TABLES / "sec35_g3_min_cover_by_type.csv")
    g4k = pd.read_csv(TABLES / "sec35_g4_parsimony_kruskal.csv", dtype={"p_value_scientific": str})

    md = [
        "# Unit Provenance: sec35_cluster_stratified  (POST-FREEZE analysis family)",
        "",
        "## 0. POST-FREEZE registration",
        "- This unit was created **after the freeze** as a new analysis family (G1-G4). It reads only frozen artifacts",
        "  and adds new tables; no frozen value, decision, protocol or existing numeric table is modified.",
        "- Manifest rows for this family carry the status **`POST-FREEZE`** so they are not mixed with the pre-freeze",
        "  `RECOMPUTED` / `RE-VERIFIED` rows; they are also excluded from the freeze-ready citable list.",
        "- **The external classification is used strictly as a grouping factor.** No hydrological meaning, naming or",
        "  interpretation is attached to any type label; no type is merged, re-clustered, dropped or re-weighted.",
        "",
        "## 1. Estimands covered",
        f"- **G1**: PUB-penalty (`penalty_median_pub`, median over the 36 models of `delta_kge_pub_median4`) across the",
        f"  10 external types; matched {int(g1s['n_matched'])} basins, unmatched {int(g1s['n_unmatched'])} reported and excluded;",
        f"  Kruskal-Wallis H={float(g1['statistic']):.4f}, df={int(g1['df'])}, p={g1['p_value_scientific']}, {g1['effect_size_name']}={float(g1['effect_size_value']):.4f};",
        f"  all {int(g1p['family_size'].iloc[0])} pairwise Dunn comparisons with Holm correction reported ({int(g1p['reject_005_holm'].sum())} with Holm p < 0.05).",
        f"- **G1b** (descriptive only, no test, no correction): per-type Spearman rho between `{g1b['attribute'].iloc[0]}`",
        f"  (global BH-FDR q = {float(g1b['attribute_global_bh_fdr_q'].iloc[0]):.6f}, the smallest-q attribute of the A4 recomputed screening)",
        "  and the same penalty; N per type reported.",
        f"- **G2**: type distribution of the 26 worst-penalty basins (`int(0.05 x 531) = 26`, unchanged rule):",
        f"  {int(g2['n_top_matched'])} of 26 are type-matched ({int(g2['n_top_unmatched'])} unmatched);",
        f"  2 x 10 contingency table with observed and expected counts; exact test p={float(g2['p_value_exact_monte_carlo']):.6f}",
        f"  (Monte-Carlo Fisher-type exact test, {int(g2['n_draws'])} draws, seed {int(g2['seed'])}, se={float(g2['monte_carlo_standard_error']):.2e});",
        f"  asymptotic chi-square p={float(g2['p_value_chi2_asymptotic']):.6f} reported alongside.",
        "- **G3**: equivalence-set size per condition x type with Kruskal-Wallis; exact minimal cover per condition x type",
        "  for 90 / 95 / 100% of that type's basins with optimality certificates (max basins coverable at budget K*-1);",
        "  95%-cover counts compared with the global values (IC 4 / seen-dPL 6 / OOB-dPL 12).",
        "- **G4**: parsimony cost (ranking member minus most-parsimonious member) per condition x type with Kruskal-Wallis,",
        "  using only the ID-joined A3 recomputation.",
        "",
        "## 2. Approach declarations",
        "- **G1** `scripts/sec35_01_penalty_by_type_g1.py`: response recomputed by basin-ID join from",
        "  `pub_same_pipeline_model_basin.csv`; grouping from `data/gauge_cluster.csv` (ID-joined); Kruskal-Wallis via",
        "  `scipy.stats.kruskal`; effect size **rank epsilon-squared**, defined as epsilon^2 = (H - k + 1) / (n - k)",
        "  (H = Kruskal-Wallis statistic, k = number of types, n = number of basins); pairwise **Dunn z-test** on average",
        f"  ranks with tie correction, two-sided, **Holm** correction over the full family of {int(g1p['family_size'].iloc[0])} pairs.",
        "  The epsilon-squared estimator can be negative when H < k - 1; negative values are reported as computed.",
        "- **G1b**: Spearman rho computed within each type on the same ID-joined arrays; **no p-value, no correction**.",
        "- **G2** `scripts/sec35_02_worst_basins_by_type_g2.py`: basin sums of `kge_seendpl_canonical - kge_heldout`",
        "  (negative values retained), descending, top 26; contingency table restricted to matched basins;",
        "  exact test implemented as a **Monte-Carlo enumeration of the multivariate hypergeometric null**",
        "  (Fisher probability-ordering statistic, 100,000 draws, seed 20261005) because the 2 x 10 table has no",
        "  closed-form exact test in the available stack; the asymptotic chi-square result is reported next to it.",
        "- **G3** `scripts/sec35_03_equivalence_library_by_type_g3.py`: equivalence sizes read from",
        "  `t1_equivalence_sets.csv` by basin-ID join; minimal covers solved exactly with `scipy.optimize.milp` (HiGHS),",
        "  status 0 = proven optimal, plus the budget-minus-one certificate; targets are fractions of each type's own basin count",
        "  (`ceil(target x n_type)`), so for small types the 95% target can equal the type's full basin count (reported as-is).",
        "- **G4** `scripts/sec35_04_parsimony_by_type_g4.py`: response read from",
        "  `sec34_a3_parsimony_penalty_idjoin_per_basin.csv` (ID-joined A3 recomputation; the SUPERSEDED table is never used);",
        "  the subtraction order is asserted to be `ranking_member_minus_parsimonious_member`.",
        "- All basin-level lookups and joins go through `_shared/basin_accessor.py` (`canonical_basin_ids`, `basin_position_map`,",
        "  `pad_basin_id`, `align_series_by_id`).",
        "",
        "## 3. Key results (as written to the tables)",
        f"- G1 per-type penalty median range: {g1t['median'].min():.6f} - {g1t['median'].max():.6f}; type sizes {dict(zip(g1t['type'], g1t['n']))}",
        f"- G1 unmatched basins ({int(g1s['n_unmatched'])}): {g1s['unmatched_basin_ids']}",
        "- G3 equivalence-size Kruskal-Wallis:",
    ]
    for _, r in g3k.iterrows():
        md.append(f"  * {r['condition']}: H={r['statistic']:.4f}, df={int(r['df'])}, p={r['p_value_scientific']}, epsilon^2={r['effect_size_value']:.4f}")
    md.append("- G3 95% minimal cover per type (type -> models; global value in brackets):")
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        row = ", ".join(f"{int(r['type'])}:{int(r['min_models_exact'])}" for _, r in
                        g3c[(g3c.condition == cond) & (g3c.target_pct == "95%")].iterrows())
        md.append(f"  * {cond} (global all-531 value {GLOBAL_95[cond]}): {row}")
    md.append(f"- G3 optimality certificates: {int(g3c['proves_k_impossible'].sum())}/{len(g3c)} type x condition x target cells prove budget K*-1 insufficient (solver status 0 in {int((g3c['solver_status']==0).sum())}/{len(g3c)}).")
    md.append("- G4 parsimony-cost Kruskal-Wallis:")
    for _, r in g4k.iterrows():
        md.append(f"  * {r['condition']}: H={r['statistic']:.4f}, df={int(r['df'])}, p={r['p_value_scientific']}, epsilon^2={r['effect_size_value']:.4f}")
    md.extend([
        "",
        "## 4. Output tables",
    ])
    for f in sorted(TABLES.glob("*.csv")):
        md.append(f"- `tables/{f.name}` ({f.stat().st_size} bytes)")
    md.extend([
        "",
        "## 5. Compliance notes",
        "- No hydrological model, CMA-ES run, dPL training or recommender fit was re-run.",
        "- No frozen value (fold split, snow filter, equivalence threshold rule, bootstrap protocol, seed list) was changed.",
        "- SUPERSEDED tables were not used: G4 reads only the ID-joined A3 table; G1/G1b use the A4 recomputed screening for the attribute choice.",
        "- All 10 types, all 45 pairwise comparisons and all three conditions are reported; nothing is selected by significance.",
        "- No 10 x 15 stratified significance screen was performed (G1b is descriptive only).",
    ])
    (UNIT_ROOT / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[sec35_05] PASS: generated PROVENANCE.md for sec35_cluster_stratified")


if __name__ == "__main__":
    main()
