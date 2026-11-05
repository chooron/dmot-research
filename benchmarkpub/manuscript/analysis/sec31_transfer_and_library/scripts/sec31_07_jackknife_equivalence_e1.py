#!/usr/bin/env python3
"""sec31_07_jackknife_equivalence_e1.py — E1: Alternative uncertainty convention (leave-one-water-year jackknife).

Computes performance equivalence sets and minimal cover models under leave-one-water-year jackknife
across IC, seen-dPL, and OOB-dPL conditions for targets 90%, 95%, and 100%.
Presents results side-by-side with the primary water-year block bootstrap convention.
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


def solve_min_cover(B: np.ndarray, needed: int) -> int:
    """Exact minimal set cover via MILP (HiGHS)."""
    nb, nm = B.shape
    nv = nm + nb
    c = np.concatenate([np.ones(nm), np.zeros(nb)])
    rows, lb, ub = [], [], []
    for i in range(nb):
        r = np.zeros(nv)
        r[nm + i] = 1.0
        r[:nm] = -B[i].astype(float)
        rows.append(r)
        lb.append(-np.inf)
        ub.append(0.0)
    cover = np.zeros(nv)
    cover[nm:] = 1.0
    rows.append(cover)
    lb.append(needed)
    ub.append(np.inf)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return int(round(res.fun))


def main() -> None:
    models = S.MODELS_36
    basins = S.canonical_ids()
    basins_str = [str(b).zfill(8) for b in basins]
    years = S.WATER_YEARS
    nb, nm, ny = len(basins), len(models), len(years)

    # 1. Load sufficient stats cubes
    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    mi = {m: i for i, m in enumerate(models)}
    bi = {b: i for i, b in enumerate(basins_str)}
    yi = {y: i for i, y in enumerate(years)}

    cube_ic = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    cube_dp = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    stat_cols_dp = ["n_dpl", "sum_sim_dpl", "sum_obs_dpl", "sum_sim2_dpl", "sum_obs2_dpl", "sum_simobs_dpl"]

    idx_m = df_raw["model"].map(mi).to_numpy()
    idx_b = df_raw["basin_id"].astype(str).str.zfill(8).map(bi).to_numpy()
    idx_y = df_raw["water_year"].map(yi).to_numpy()

    cube_ic[idx_m, idx_b, idx_y] = df_raw[stat_cols_ic].to_numpy(dtype=np.float64)
    cube_dp[idx_m, idx_b, idx_y] = df_raw[stat_cols_dp].to_numpy(dtype=np.float64)
    cube_oob = S.load_oob_cube()

    cubes = {"IC": cube_ic, "seen-dPL": cube_dp, "OOB-dPL": cube_oob}

    # 2. Existing primary baseline (water-year block bootstrap)
    # Read existing minimal cover targets
    df_boot = pd.read_csv(TABLES / "sec31_minimal_cover_targets.csv")
    df_boot_size = pd.read_csv(TABLES / "sec31_equivalence_size_distribution.csv").set_index("condition")

    comp_rows = []
    eq_rows = []
    size_rows = []

    for cname, cmat in cubes.items():
        point = S.kge_from_sufficient(cmat.sum(axis=2))  # [nm, nb]
        total_sum = cmat.sum(axis=2)  # [nm, nb, 6]

        # 15 leave-one-water-year replicates
        jk_bests = np.empty((15, nb), dtype=np.float64)
        for y in range(15):
            rem_sum = total_sum - cmat[:, :, y, :]
            kge_y = S.kge_from_sufficient(rem_sum)  # [nm, nb]
            jk_bests[y] = kge_y.max(axis=0)  # [nb]

        # Equivalence threshold: 5th percentile across 15 leave-one-year replicates
        q05_jk = np.quantile(jk_bests, 0.05, axis=0)  # [nb]
        eq_jk = (point >= q05_jk[None, :])  # [nm, nb]
        B_jk = eq_jk.T.astype(bool)  # [nb, nm]
        sizes = eq_jk.sum(axis=0)  # [nb]

        # Size distribution
        size_rows.append({
            "condition": cname,
            "uncertainty_convention": "leave_one_water_year_jackknife",
            "n_basins": nb,
            "median_size": float(np.median(sizes)),
            "q25_size": float(np.quantile(sizes, 0.25)),
            "q75_size": float(np.quantile(sizes, 0.75)),
            "mean_size": float(sizes.mean()),
            "min_size": int(sizes.min()),
            "max_size": int(sizes.max()),
            "n_singleton": int((sizes == 1).sum()),
        })

        # Basin-level equivalence sets
        for b_idx, b_id in enumerate(basins_str):
            eq_mods = [models[m_idx] for m_idx in np.flatnonzero(eq_jk[:, b_idx])]
            eq_rows.append({
                "basin_id": b_id,
                "condition": cname,
                "q05_jackknife": float(q05_jk[b_idx]),
                "n_equivalent": len(eq_mods),
                "equivalent_models": "|".join(eq_mods),
            })

        # Coverage ladder for 90%, 95%, 100%
        for target in [0.90, 0.95, 1.00]:
            nd = int(np.ceil(target * nb))
            k_jk = solve_min_cover(B_jk, nd)
            # Find matching bootstrap row
            boot_row = df_boot[(df_boot.condition == cname) & (np.isclose(df_boot.target, target))].iloc[0]
            k_boot = int(boot_row["n_models_exact"])

            comp_rows.append({
                "condition": cname,
                "target": target,
                "target_pct": f"{int(target*100)}%",
                "need_basins": nd,
                "n_models_block_bootstrap": k_boot,
                "n_models_jackknife": k_jk,
                "diff_jackknife_minus_bootstrap": k_jk - k_boot,
                "median_size_block_bootstrap": float(df_boot_size.loc[cname, "median_size"]),
                "median_size_jackknife": float(np.median(sizes)),
            })

    df_comp = pd.DataFrame(comp_rows)
    S.write_tidy_table(df_comp, TABLES / "sec31_jackknife_vs_bootstrap_cover_comparison.csv")

    df_eq = pd.DataFrame(eq_rows)
    S.write_tidy_table(df_eq, TABLES / "sec31_jackknife_equivalence_sets.csv")

    df_size = pd.DataFrame(size_rows)
    S.write_tidy_table(df_size, TABLES / "sec31_jackknife_size_distribution.csv")

    print("[sec31_07] PASS: compiled E1 jackknife vs bootstrap comparison across all 3 conditions:")
    print(df_comp.to_string(index=False))


if __name__ == "__main__":
    main()
