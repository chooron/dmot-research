#!/usr/bin/env python3
"""sec35_04_parsimony_by_type_g4.py — G4: within-equivalence parsimony cost, stratified by external type.

Uses only the ID-joined A3 recomputation (`sec34_a3_parsimony_penalty_idjoin_per_basin.csv`), never the
SUPERSEDED table. Response: parsimony cost = ranking-member KGE minus most-parsimonious-member KGE.
Per condition x type: median / Q25 / Q75 / N, plus a Kruskal-Wallis test with a rank effect size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEC34 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec34_determinacy_and_bounds" / "tables"

CONDITIONS = ["IC", "seen-dPL", "OOB-dPL"]


def main() -> None:
    basins = S.canonical_basin_ids()
    pos = S.basin_position_map(basins)
    cf = pd.read_csv(ROOT / "data" / "gauge_cluster.csv")
    cf["bs"] = cf["gauge_index"].map(S.pad_basin_id)
    cmap = dict(zip(cf["bs"], cf["gauge_cluster"]))
    matched = [b for b in basins if b in cmap]
    types_sorted = sorted(set(cmap[b] for b in matched))

    a3 = pd.read_csv(SEC34 / "sec34_a3_parsimony_penalty_idjoin_per_basin.csv")
    a3["bs"] = a3["basin_id"].map(S.pad_basin_id)
    assert "subtraction_order" in a3.columns
    assert set(a3["subtraction_order"].unique()) == {"ranking_member_minus_parsimonious_member"}
    assert set(a3["join_method"].unique()) == {"basin_id_keyed"}

    rows, kw_rows = [], []
    for cond in CONDITIONS:
        sub = a3[a3.condition == cond].set_index("bs")
        groups = []
        for t in types_sorted:
            ids = [b for b in matched if cmap[b] == t]
            v = np.array([float(sub.loc[b, "diff_kge_rank_minus_pars"]) for b in ids])
            groups.append(v)
            rows.append({"condition": cond, "type": int(t), "n": int(v.size),
                         "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
                         "q75": float(np.quantile(v, 0.75)), "mean": float(v.mean()),
                         "min": float(v.min()), "max": float(v.max())})
        H, p_kw = stats.kruskal(*groups)
        n_tot = int(sum(g.size for g in groups))
        eps2 = (H - len(groups) + 1.0) / (n_tot - len(groups))
        kw_rows.append({"condition": cond, "test": "Kruskal-Wallis", "statistic": float(H),
                        "df": len(groups) - 1, "p_value": float(p_kw), "p_value_scientific": f"{float(p_kw):.6e}", "n_total": n_tot,
                        "n_types": len(groups),
                        "effect_size_name": "rank epsilon-squared (epsilon^2)",
                        "effect_size_definition": "epsilon^2 = (H - k + 1) / (n - k), H = Kruskal-Wallis statistic, k = number of types, n = number of basins",
                        "effect_size_value": float(eps2),
                        "response_definition": "ranking-member KGE minus most-parsimonious-member KGE (ID-joined A3 recomputation)"})

    S.write_tidy_table(pd.DataFrame(rows), TABLES / "sec35_g4_parsimony_by_type.csv")
    df_kw = pd.DataFrame(kw_rows)
    S.write_tidy_table(df_kw, TABLES / "sec35_g4_parsimony_kruskal.csv")

    print("[G4] parsimony cost by type (median / q25 / q75 / n):")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("[G4] Kruskal-Wallis:")
    print(df_kw[["condition", "statistic", "df", "p_value", "effect_size_value", "n_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
