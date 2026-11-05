#!/usr/bin/env python3
"""sec32_06_bootstrap_cover_ci_e2.py — E2: Basin-level bootstrap confidence intervals for minimal library size.

Resamples basins with replacement (N_BOOT=1000, seed=20260925) and solves exact partial set cover via MILP
for IC, seen-dPL, and OOB-dPL at 90%, 95%, and 100% coverage targets.
Outputs summary CI table and full empirical distribution of library sizes.
"""
from __future__ import annotations

import json
import sys
import time
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

N_BOOT = S.N_BOOT_BLOCK  # 1000 draws
SEED = S.SEED_BOOT       # 20260925


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


def solve_bootstrap_cover(B: np.ndarray, counts: np.ndarray, needed: int) -> int:
    """Exact weighted set cover on unique basins with multiplicity counts > 0."""
    active = np.flatnonzero(counts > 0)
    w = counts[active]
    B_act = B[active]
    na = len(active)
    nm = B.shape[1]

    nv = nm + na
    c = np.concatenate([np.ones(nm), np.zeros(na)])
    rows, lb, ub = [], [], []
    for i in range(na):
        r = np.zeros(nv)
        r[nm + i] = 1.0
        r[:nm] = -B_act[i].astype(float)
        rows.append(r)
        lb.append(-np.inf)
        ub.append(0.0)

    # Weight constraint: sum_i w_i z_i >= needed
    r_cov = np.zeros(nv)
    r_cov[nm:] = w.astype(float)
    rows.append(r_cov)
    lb.append(needed)
    ub.append(np.inf)

    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return int(round(res.fun))


def main() -> None:
    models = S.MODELS_36
    # Load equivalence sets
    eq_path = S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv"
    df_eq = pd.read_csv(eq_path)

    # Read point estimate values
    df_min = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_min_cover.csv")

    summary_rows = []
    dist_rows = []

    t0_all = time.time()
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        B, basin_ids = build_B_matrix(df_eq, cond, models)
        nb = len(B)

        for target in [0.90, 0.95, 1.00]:
            needed = int(np.ceil(target * nb))
            pt_row = df_min[(df_min.condition == cond) & (np.isclose(df_min.target, target))].iloc[0]
            k_obs = int(pt_row["n_models_exact"])

            t0 = time.time()
            draws = np.empty(N_BOOT, dtype=int)
            rng = np.random.default_rng(SEED)
            for r in range(N_BOOT):
                idx = rng.integers(0, nb, size=nb)
                counts = np.bincount(idx, minlength=nb)
                draws[r] = solve_bootstrap_cover(B, counts, needed)

            dt = time.time() - t0
            lo95, hi95 = np.percentile(draws, [2.5, 97.5])

            summary_rows.append({
                "condition": cond,
                "target": target,
                "target_pct": f"{int(target * 100)}%",
                "need_basins": needed,
                "k_observed": k_obs,
                "n_boot": N_BOOT,
                "seed": SEED,
                "median": float(np.median(draws)),
                "mean": float(draws.mean()),
                "ci95_lo": float(lo95),
                "ci95_hi": float(hi95),
                "min": int(draws.min()),
                "max": int(draws.max()),
                "std": float(draws.std(ddof=1)),
            })

            # Record frequency distribution of draws
            vals, counts_val = np.unique(draws, return_counts=True)
            for v, cnt in zip(vals, counts_val):
                dist_rows.append({
                    "condition": cond,
                    "target": target,
                    "target_pct": f"{int(target * 100)}%",
                    "k_models": int(v),
                    "frequency": int(cnt),
                    "fraction": float(cnt / N_BOOT),
                })

            print(f"[sec32_06] [{cond}] {int(target*100)}% cover: obs={k_obs}, boot median={np.median(draws):.1f} [95% CI {lo95:.1f} - {hi95:.1f}], range {draws.min()}-{draws.max()} ({dt:.1f}s)")

    df_summary = pd.DataFrame(summary_rows)
    S.write_tidy_table(df_summary, TABLES / "sec32_bootstrap_cover_ci_summary.csv")

    df_dist = pd.DataFrame(dist_rows)
    S.write_tidy_table(df_dist, TABLES / "sec32_bootstrap_cover_distribution.csv")

    print(f"[sec32_06] PASS: completed all 9 condition x target bootstrap sweeps in {time.time()-t0_all:.1f}s")


if __name__ == "__main__":
    main()
