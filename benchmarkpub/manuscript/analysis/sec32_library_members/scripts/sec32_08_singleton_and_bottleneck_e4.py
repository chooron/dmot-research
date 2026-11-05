#!/usr/bin/env python3
"""sec32_08_singleton_and_bottleneck_e4.py — E4: Unique required models (singletons) and 100% cover bottleneck basins.

For IC, seen-dPL, and OOB-dPL:
1. Counts singleton basins per model (where |S_b| = 1).
2. Marks singleton status per basin.
3. Explicitly identifies the bottleneck basins and models that force the library size to increase
   to its final member under 100% coverage (at budget = K* - 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def build_B_matrix(df_eq: pd.DataFrame, condition: str, models: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build binary coverage matrix B [nb, nm] from equivalence sets."""
    sub = df_eq[df_eq.condition == condition].sort_values("basin_id").reset_index(drop=True)
    nb, nm = len(sub), len(models)
    model_map = {m: i for i, m in enumerate(models)}
    B = np.zeros((nb, nm), dtype=bool)
    for row_idx, (_, r) in enumerate(sub.iterrows()):
        if pd.notna(r["equivalent_models"]):
            for m in str(r["equivalent_models"]).split("|"):
                if m in model_map:
                    B[row_idx, model_map[m]] = True
    return B, sub["basin_id"].tolist()


def solve_max_coverage_budget(B: np.ndarray, budget: int) -> Tuple[int, np.ndarray]:
    """Solve max-coverage MILP for a fixed model budget; returns (max_basins, covered_mask)."""
    nb, nm = B.shape
    nv = nm + nb
    c = np.concatenate([np.zeros(nm), -np.ones(nb)])
    rows, lb, ub = [], [], []
    for i in range(nb):
        r = np.zeros(nv)
        r[nm + i] = 1.0
        r[:nm] = -B[i].astype(float)
        rows.append(r)
        lb.append(-np.inf)
        ub.append(0.0)
    budget_row = np.zeros(nv)
    budget_row[:nm] = 1.0
    rows.append(budget_row)
    lb.append(budget)
    ub.append(budget)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    cov = int(round(-res.fun))
    covered = (res.x[nm:] > 0.5)
    return cov, covered


def main() -> None:
    models = S.MODELS_36
    eq_path = S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv"
    df_eq = pd.read_csv(eq_path)

    # 1. Singleton counts per model & per basin
    model_counts_rows = []
    basin_status_rows = []
    bottleneck_rows = []

    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        sub = df_eq[df_eq.condition == cond].sort_values("basin_id").reset_index(drop=True)
        nb = len(sub)

        # Basin-level singleton labeling
        singletons = sub[sub.n_equivalent == 1].copy()
        singleton_counts = singletons["equivalent_models"].value_counts().to_dict()

        for m in models:
            cnt = int(singleton_counts.get(m, 0))
            model_counts_rows.append({
                "condition": cond,
                "model": m,
                "singleton_basin_count": cnt,
                "fraction_of_total_basins": float(cnt / nb),
            })

        for _, r in sub.iterrows():
            is_single = bool(r["n_equivalent"] == 1)
            basin_status_rows.append({
                "basin_id": str(r["basin_id"]).zfill(8),
                "condition": cond,
                "equivalence_size": int(r["n_equivalent"]),
                "is_singleton": is_single,
                "singleton_model": str(r["equivalent_models"]) if is_single else "",
            })

        # Bottleneck at 100% coverage
        B, basin_ids = build_B_matrix(df_eq, cond, models)
        # Find 100% cover size K*
        k_100_row = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_min_cover.csv")
        k_star = int(k_100_row[(k_100_row.condition == cond) & (np.isclose(k_100_row.target, 1.0))].iloc[0]["n_models_exact"])

        budget_minus_1 = k_star - 1
        max_cov, cov_mask = solve_max_coverage_budget(B, budget_minus_1)
        uncovered_idx = np.flatnonzero(~cov_mask)
        uncovered_basins = sub.iloc[uncovered_idx]

        b_ids = [str(x).zfill(8) for x in uncovered_basins["basin_id"]]
        eq_mods = [str(x) for x in uncovered_basins["equivalent_models"]]
        all_req_models = sorted(list(set([m for em in eq_mods for m in em.split("|")])))

        bottleneck_rows.append({
            "condition": cond,
            "k_100pct_opt": k_star,
            "budget_minus_1": budget_minus_1,
            "max_basins_covered_budget_minus_1": max_cov,
            "uncovered_basin_count": len(uncovered_basins),
            "uncovered_basin_ids": "|".join(b_ids),
            "uncovered_basin_equivalent_models": "; ".join(f"{b}: {m}" for b, m in zip(b_ids, eq_mods)),
            "bottleneck_forcing_models": "|".join(all_req_models),
        })

        print(f"[sec32_08] [{cond}] singletons={len(singletons)}/531, 100% cover k*={k_star}, budget {budget_minus_1} covers {max_cov}/{nb}; {len(uncovered_basins)} uncovered basin(s): {b_ids} requiring models in {all_req_models}")

    df_mc = pd.DataFrame(model_counts_rows)
    S.write_tidy_table(df_mc, TABLES / "sec32_singleton_model_counts.csv")

    df_bs = pd.DataFrame(basin_status_rows)
    S.write_tidy_table(df_bs, TABLES / "sec32_basin_singleton_status.csv")

    df_bn = pd.DataFrame(bottleneck_rows)
    S.write_tidy_table(df_bn, TABLES / "sec32_bottleneck_100pct_coverage.csv")

    print("[sec32_08] PASS: compiled singleton counts and bottleneck analysis")


if __name__ == "__main__":
    main()
