#!/usr/bin/env python3
"""sec35_03_equivalence_library_by_type_g3.py — G3: equivalence-set size and exact minimal library size, stratified by type.

Per external type (10 types, matched basins only):
  * equivalence-set size |S_b| statistics for IC / seen-dPL / OOB-dPL and a Kruskal-Wallis test with a rank effect size;
  * exact minimal cover (MILP, HiGHS) for 90 / 95 / 100% of that type's basins, with an optimality certificate
    (maximum basins coverable with budget K*-1). Small types are reported as-is; nothing is merged or dropped;
  * the 95%-cover model count against the global value (IC 4 / seen-dPL 6 / OOB-dPL 12).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["IC", "seen-dPL", "OOB-dPL"]
TARGETS = [0.90, 0.95, 1.00]
GLOBAL_95 = {"IC": 4, "seen-dPL": 6, "OOB-dPL": 12}


def solve_max_cov(B: np.ndarray, budget: int):
    nb, nm = B.shape
    nv = nm + nb
    c = np.concatenate([np.zeros(nm), -np.ones(nb)])
    rows, lb, ub = [], [], []
    for i in range(nb):
        r = np.zeros(nv)
        r[nm + i] = 1.0
        r[:nm] = -B[i].astype(float)
        rows.append(r); lb.append(-np.inf); ub.append(0.0)
    b_row = np.zeros(nv); b_row[:nm] = 1.0
    rows.append(b_row); lb.append(budget); ub.append(budget)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return int(round(-res.fun)), int(res.status)


def solve_min_cover(B: np.ndarray, needed: int):
    nm, nb = B.shape[1], B.shape[0]
    for q in range(1, nm + 1):
        cov, st = solve_max_cov(B, q)
        if cov >= needed:
            return q, cov, st
    return nm, nb, 0


def main() -> None:
    basins = S.canonical_basin_ids()
    pos = S.basin_position_map(basins)
    cf = pd.read_csv(ROOT / "data" / "gauge_cluster.csv")
    cf["bs"] = cf["gauge_index"].map(S.pad_basin_id)
    cmap = dict(zip(cf["bs"], cf["gauge_cluster"]))
    matched = [b for b in basins if b in cmap]
    types_sorted = sorted(set(cmap[b] for b in matched))
    models = S.MODELS_36
    midx = {m: i for i, m in enumerate(models)}

    eq = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")
    eq["bs"] = eq["basin_id"].map(S.pad_basin_id)
    eq = eq.set_index(["bs", "condition"])

    size_rows, kw_rows, cover_rows, vs_rows = [], [], [], []
    for cond in CONDITIONS:
        sizes_by_type = {}
        for t in types_sorted:
            ids = [b for b in matched if cmap[b] == t]
            sizes = np.array([int(eq.loc[(b, cond), "n_equivalent"]) for b in ids])
            sizes_by_type[t] = sizes
            size_rows.append({"condition": cond, "type": int(t), "n": int(sizes.size),
                              "median": float(np.median(sizes)), "q25": float(np.quantile(sizes, 0.25)),
                              "q75": float(np.quantile(sizes, 0.75)), "mean": float(sizes.mean()),
                              "min": int(sizes.min()), "max": int(sizes.max())})
        groups = [sizes_by_type[t] for t in types_sorted]
        H, p_kw = stats.kruskal(*groups)
        n_tot = int(sum(g.size for g in groups))
        eps2 = (H - len(groups) + 1.0) / (n_tot - len(groups))
        kw_rows.append({"condition": cond, "test": "Kruskal-Wallis", "statistic": float(H),
                        "df": len(groups) - 1, "p_value": float(p_kw), "p_value_scientific": f"{float(p_kw):.6e}", "n_total": n_tot,
                        "n_types": len(groups),
                        "effect_size_name": "rank epsilon-squared (epsilon^2)",
                        "effect_size_definition": "epsilon^2 = (H - k + 1) / (n - k), H = Kruskal-Wallis statistic, k = number of types, n = number of basins",
                        "effect_size_value": float(eps2)})

        for t in types_sorted:
            ids = [b for b in matched if cmap[b] == t]
            nb_t = len(ids)
            B = np.zeros((nb_t, len(models)), dtype=bool)
            for i, b in enumerate(ids):
                for m in str(eq.loc[(b, cond), "equivalent_models"]).split("|"):
                    B[i, midx[m]] = True
            for target in TARGETS:
                need = int(np.ceil(target * nb_t))
                k_star, cov_star, st = solve_min_cover(B, need)
                max_minus1, _ = solve_max_cov(B, k_star - 1)
                cover_rows.append({"condition": cond, "type": int(t), "n_type_basins": nb_t,
                                   "target": target, "target_pct": f"{int(target*100)}%",
                                   "need_basins": need, "min_models_exact": k_star,
                                   "basins_covered": cov_star, "budget_minus_1": k_star - 1,
                                   "max_basins_covered_budget_minus_1": max_minus1,
                                   "proves_k_impossible": bool(max_minus1 < need),
                                   "solver_status": st})
                if abs(target - 0.95) < 1e-9:
                    g95 = GLOBAL_95[cond]
                    vs_rows.append({"condition": cond, "type": int(t), "n_type_basins": nb_t,
                                    "min_models_95_type": k_star, "global_95_all_531": g95,
                                    "diff_type_minus_global": k_star - g95,
                                    "type_need_basins": need,
                                    "type_target_equals_all_type_basins": bool(need == nb_t)})

    S.write_tidy_table(pd.DataFrame(size_rows), TABLES / "sec35_g3_equivalence_size_by_type.csv")
    S.write_tidy_table(pd.DataFrame(kw_rows), TABLES / "sec35_g3_equivalence_size_kruskal.csv")
    S.write_tidy_table(pd.DataFrame(cover_rows), TABLES / "sec35_g3_min_cover_by_type.csv")
    S.write_tidy_table(pd.DataFrame(vs_rows), TABLES / "sec35_g3_min_cover_95_vs_global.csv")

    df_kw = pd.DataFrame(kw_rows)
    print("[G3] equivalence-size Kruskal-Wallis by condition:")
    print(df_kw[["condition", "statistic", "df", "p_value", "effect_size_value", "n_total"]].to_string(index=False))
    df_cov = pd.DataFrame(cover_rows)
    print("\n[G3] 95% minimal cover per type (exact, with certificate):")
    print(df_cov[df_cov.target_pct == "95%"][["condition", "type", "n_type_basins", "need_basins", "min_models_exact",
                                               "max_basins_covered_budget_minus_1", "proves_k_impossible"]].to_string(index=False))
    print("\n[G3] 95% cover vs global:")
    print(pd.DataFrame(vs_rows).to_string(index=False))


if __name__ == "__main__":
    main()
