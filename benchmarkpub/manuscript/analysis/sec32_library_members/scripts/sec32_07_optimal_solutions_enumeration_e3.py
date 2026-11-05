#!/usr/bin/env python3
"""sec32_07_optimal_solutions_enumeration_e3.py — E3: Complete enumeration of optimal set covers.

Enumerates all optimal solutions via MILP with iterative no-good cuts for IC, seen-dPL, and OOB-dPL
at 95% and 100% coverage targets (up to 1000 solutions).
Outputs summary (solution count, core set, union set), model frequency table, and solution listings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

MAX_SOLUTIONS = 1000


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


def enumerate_optimal_covers(B: np.ndarray, needed: int, max_solutions: int = MAX_SOLUTIONS) -> Tuple[int, List[List[int]], bool]:
    """Enumerate all binary solutions achieving min cover size K* via no-good cut iterations."""
    nb, nm = B.shape
    nv = nm + nb
    c = np.concatenate([np.ones(nm), np.zeros(nb)])

    # 1. Base constraints: B x >= y, sum(y) >= needed
    base_rows, base_lb, base_ub = [], [], []
    for i in range(nb):
        r = np.zeros(nv)
        r[nm + i] = 1.0
        r[:nm] = -B[i].astype(float)
        base_rows.append(r)
        base_lb.append(-np.inf)
        base_ub.append(0.0)

    cover = np.zeros(nv)
    cover[nm:] = 1.0
    base_rows.append(cover)
    base_lb.append(needed)
    base_ub.append(np.inf)

    # Solve initial min cover
    A_init = np.array(base_rows)
    res_init = milp(c=c, constraints=LinearConstraint(A_init, lb=np.array(base_lb), ub=np.array(base_ub)),
                    integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res_init.success:
        return 0, [], False
    k_star = int(round(res_init.fun))

    # Fix exact budget = k_star
    budget_row = np.zeros(nv)
    budget_row[:nm] = 1.0
    base_rows.append(budget_row)
    base_lb.append(k_star)
    base_ub.append(k_star)

    c_zero = np.zeros(nv)
    solutions: List[List[int]] = []
    current_rows = list(base_rows)
    current_lb = list(base_lb)
    current_ub = list(base_ub)

    is_truncated = False
    while True:
        if len(solutions) >= max_solutions:
            is_truncated = True
            break
        A = np.array(current_rows)
        res = milp(c=c_zero, constraints=LinearConstraint(A, lb=np.array(current_lb), ub=np.array(current_ub)),
                   integrality=np.ones(nv), bounds=Bounds(0, 1))
        if not res.success:
            break
        sol_x = (res.x[:nm] > 0.5).astype(int)
        sol_indices = sorted(np.flatnonzero(sol_x).tolist())
        solutions.append(sol_indices)

        # No-good cut: sum_{j in sol_indices} x_j <= k_star - 1
        ng_row = np.zeros(nv)
        ng_row[sol_indices] = 1.0
        current_rows.append(ng_row)
        current_lb.append(-np.inf)
        current_ub.append(k_star - 1)

    return k_star, solutions, is_truncated


def main() -> None:
    models = S.MODELS_36
    eq_path = S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv"
    df_eq = pd.read_csv(eq_path)

    summary_rows = []
    freq_rows = []
    sol_rows = []

    t0_all = time.time()
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        B, basin_ids = build_B_matrix(df_eq, cond, models)
        nb = len(B)

        for target in [0.95, 1.00]:
            needed = int(np.ceil(target * nb))
            t0 = time.time()
            k_opt, sols, is_trunc = enumerate_optimal_covers(B, needed, max_solutions=MAX_SOLUTIONS)
            dt = time.time() - t0

            n_sols = len(sols)
            # Core set: intersection of all optimal solutions
            if n_sols > 0:
                core_set_idx: Set[int] = set(sols[0])
                union_set_idx: Set[int] = set(sols[0])
                for s in sols[1:]:
                    core_set_idx &= set(s)
                    union_set_idx |= set(s)
                core_models = sorted([models[i] for i in core_set_idx])
                union_models = sorted([models[i] for i in union_set_idx])
            else:
                core_models, union_models = [], []

            summary_rows.append({
                "condition": cond,
                "target": target,
                "target_pct": f"{int(target * 100)}%",
                "need_basins": needed,
                "k_opt": k_opt,
                "n_optimal_solutions": n_sols,
                "is_truncated": is_trunc,
                "max_solutions_cap": MAX_SOLUTIONS,
                "core_set_size": len(core_models),
                "core_set_models": "|".join(core_models),
                "union_set_size": len(union_models),
                "union_set_models": "|".join(union_models),
            })

            # Record per-solution listings
            for s_id, s in enumerate(sols):
                sol_rows.append({
                    "condition": cond,
                    "target": target,
                    "target_pct": f"{int(target * 100)}%",
                    "solution_id": s_id + 1,
                    "k_models": len(s),
                    "solution_models": "|".join(models[i] for i in s),
                })

            # Per-model appearance frequency
            model_counts = {m: 0 for m in models}
            for s in sols:
                for idx in s:
                    model_counts[models[idx]] += 1

            for m in models:
                cnt = model_counts[m]
                freq_rows.append({
                    "condition": cond,
                    "target": target,
                    "target_pct": f"{int(target * 100)}%",
                    "model": m,
                    "appearance_count": cnt,
                    "total_solutions": n_sols,
                    "appearance_frequency": float(cnt / n_sols) if n_sols > 0 else 0.0,
                    "in_core_set": bool(m in core_models),
                    "in_union_set": bool(m in union_models),
                })

            print(f"[sec32_07] [{cond}] {int(target*100)}% cover: k*={k_opt}, n_solutions={n_sols} (trunc={is_trunc}), core={len(core_models)}, union={len(union_models)} ({dt:.2f}s)")

    df_sum = pd.DataFrame(summary_rows)
    S.write_tidy_table(df_sum, TABLES / "sec32_optimal_covers_summary.csv")

    df_freq = pd.DataFrame(freq_rows)
    S.write_tidy_table(df_freq, TABLES / "sec32_optimal_covers_model_frequencies.csv")

    df_sols = pd.DataFrame(sol_rows)
    S.write_tidy_table(df_sols, TABLES / "sec32_optimal_covers_all_solutions.csv")

    print(f"[sec32_07] PASS: complete enumeration finished in {time.time()-t0_all:.1f}s")


if __name__ == "__main__":
    main()
