#!/usr/bin/env python3
"""sec36_02_type_cover_absolute_h2.py — H2: absolute per-type minimal cover counts for all three coverage targets.

Re-derives, from the frozen equivalence sets and with the same exact MILP implementation, the absolute minimal
number of models needed to cover 90 / 95 / 100% of each type's own basins, for IC / seen-dPL / OOB-dPL,
together with each type's N and the optimality certificate (maximum basins coverable with budget K*-1).
Also reports the absolute difference against the G3 table values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEC35 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec35_cluster_stratified" / "tables"

CONDITIONS = ["IC", "seen-dPL", "OOB-dPL"]
TARGETS = [0.90, 0.95, 1.00]


def solve_max_cov(B: np.ndarray, budget: int):
    nb, nm = B.shape
    nv = nm + nb
    c = np.concatenate([np.zeros(nm), -np.ones(nb)])
    A1 = np.hstack([-B.astype(np.float64), np.eye(nb)])
    A2 = np.concatenate([np.ones(nm), np.zeros(nb)])[None, :]
    A = np.vstack([A1, A2])
    lb = np.concatenate([np.full(nb, -np.inf), [float(budget)]])
    ub = np.concatenate([np.zeros(nb), [float(budget)]])
    res = milp(c=c, constraints=LinearConstraint(A, lb=lb, ub=ub),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if res.status != 0:
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

    rows = []
    for c in CONDITIONS:
        for t in types_sorted:
            ids = [b for b in matched if cmap[b] == t]
            n_t = len(ids)
            B = np.zeros((n_t, len(models)), dtype=bool)
            for i, b in enumerate(ids):
                for m in str(eq.loc[(b, c), "equivalent_models"]).split("|"):
                    B[i, midx[m]] = True
            for target in TARGETS:
                need = int(np.ceil(target * n_t))
                k, cov, st = solve_min_cover(B, need)
                max_minus1, _ = solve_max_cov(B, k - 1)
                rows.append({"condition": c, "type": int(t), "n_type_basins": n_t,
                             "target": target, "target_pct": f"{int(target*100)}%",
                             "need_basins": need, "min_models_absolute": k,
                             "basins_covered": cov, "budget_minus_1": k - 1,
                             "max_basins_covered_budget_minus_1": max_minus1,
                             "proves_k_minus_1_insufficient": bool(max_minus1 < need),
                             "solver_status": st})
    df = pd.DataFrame(rows)
    S.write_tidy_table(df, TABLES / "sec36_h2_type_cover_absolute.csv")

    # absolute difference against the G3 table (which reports the same quantities)
    g3 = pd.read_csv(SEC35 / "sec35_g3_min_cover_by_type.csv")
    m = df.merge(g3, on=["condition", "type", "target"], suffixes=("_h2", "_g3"))
    cmp_rows = [{
        "n_rows_compared": int(len(m)),
        "max_abs_diff_min_models": float(np.abs(m["min_models_absolute"] - m["min_models_exact"]).max()),
        "max_abs_diff_certificate": float(np.abs(m["max_basins_covered_budget_minus_1_h2"]
                                                 - m["max_basins_covered_budget_minus_1_g3"]).max()),
        "n_rows_with_any_diff": int(((m["min_models_absolute"] != m["min_models_exact"])
                                     | (m["max_basins_covered_budget_minus_1_h2"] != m["max_basins_covered_budget_minus_1_g3"])).sum()),
    }]
    S.write_tidy_table(pd.DataFrame(cmp_rows), TABLES / "sec36_h2_vs_g3_comparison.csv")

    print("[H2] absolute per-type minimal cover (condition x type x target):")
    for c in CONDITIONS:
        for tgt in ["90%", "95%", "100%"]:
            sub = df[(df.condition == c) & (df.target_pct == tgt)]
            print(f"  {c} {tgt}: " + " | ".join(f"t{int(r['type'])}:{int(r['min_models_absolute'])}/{int(r['n_type_basins'])}" for _, r in sub.iterrows()))
    print(f"[H2] certificates: {int(df.proves_k_minus_1_insufficient.sum())}/{len(df)} cells prove budget K*-1 insufficient")
    print(f"[H2] vs G3: {cmp_rows[0]}")


if __name__ == "__main__":
    main()
