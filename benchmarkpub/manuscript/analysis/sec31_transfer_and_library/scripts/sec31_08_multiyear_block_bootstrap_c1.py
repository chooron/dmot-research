#!/usr/bin/env python3
"""sec31_08_multiyear_block_bootstrap_c1.py — C1: Multi-year block bootstrap (L=3 years) alternative uncertainty convention.

Implements non-overlapping multi-year block bootstrap (Clark et al. 2021 NBB methodology)
using 5 non-overlapping 3-water-year blocks across 15 test water years (1996..2010).
Recomputes minimal covers for 90%, 95%, and 100% targets under IC, seen-dPL, and OOB-dPL with HiGHS MILP optimality certificates.
Outputs side-by-side comparison vs the 1-year block bootstrap convention.
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

N_BOOT = S.N_BOOT_BLOCK  # 1000 draws
SEED = S.SEED_BOOT       # 20260925


def to_3yr_blocks(cube: np.ndarray) -> np.ndarray:
    """Pool 15 1-year sufficient statistics into 5 non-overlapping 3-year blocks."""
    nm, nb, ny, ns = cube.shape
    assert ny == 15 and ns == 6
    b3 = np.zeros((nm, nb, 5, 6), dtype=np.float64)
    for b_idx in range(5):
        b3[:, :, b_idx, :] = cube[:, :, b_idx * 3 : (b_idx + 1) * 3, :].sum(axis=2)
    return b3


def solve_max_cov(B: np.ndarray, budget: int) -> Tuple[int, int]:
    """Solve max-coverage MILP for a fixed budget; returns (max_basins, status)."""
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
    b_row = np.zeros(nv)
    b_row[:nm] = 1.0
    rows.append(b_row)
    lb.append(budget)
    ub.append(budget)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return int(round(-res.fun)), int(res.status)


def solve_min_cover(B: np.ndarray, needed: int) -> Tuple[int, int, int]:
    """Exact minimal set cover via MILP."""
    nm = B.shape[1]
    nb = B.shape[0]
    for q in range(1, nm + 1):
        cov, st = solve_max_cov(B, q)
        if cov >= needed:
            return q, cov, st
    return nm, nb, 0


def main() -> None:
    models = S.MODELS_36
    basins = sorted(S.canonical_ids())  # OOB cube uses c4.stats_cube's sorted basin order
    basins_str = [str(b).zfill(8) for b in basins]
    nb, nm = len(basins), len(models)

    # 1. Load sufficient statistics cubes
    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    mi = {m: i for i, m in enumerate(models)}
    bi = {b: i for i, b in enumerate(basins_str)}
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    stat_cols_dp = ["n_dpl", "sum_sim_dpl", "sum_obs_dpl", "sum_sim2_dpl", "sum_obs2_dpl", "sum_simobs_dpl"]

    cube_ic = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    cube_dp = np.zeros((nm, nb, 15, 6), dtype=np.float64)

    idx_m = df_raw["model"].map(mi).to_numpy()
    idx_b = df_raw["basin_id"].astype(str).str.zfill(8).map(bi).to_numpy()
    idx_y = df_raw["water_year"].map(lambda y: y - 1996).to_numpy()

    cube_ic[idx_m, idx_b, idx_y] = df_raw[stat_cols_ic].to_numpy(dtype=np.float64)
    cube_dp[idx_m, idx_b, idx_y] = df_raw[stat_cols_dp].to_numpy(dtype=np.float64)
    cube_oob = S.load_oob_cube()

    # Convert to 5 non-overlapping 3-year blocks
    cubes_3yr = {
        "IC": to_3yr_blocks(cube_ic),
        "seen-dPL": to_3yr_blocks(cube_dp),
        "OOB-dPL": to_3yr_blocks(cube_oob),
    }

    # 3-year block bootstrap counts [1000, 5]
    rng = np.random.default_rng(SEED)
    block_counts = np.zeros((N_BOOT, 5), dtype=np.float64)
    for r in range(N_BOOT):
        block_counts[r] = np.bincount(rng.integers(0, 5, size=5), minlength=5)

    # Primary baseline results
    df_boot_primary = pd.read_csv(TABLES / "sec31_minimal_cover_targets.csv")
    df_boot_size_primary = pd.read_csv(TABLES / "sec31_equivalence_size_distribution.csv").set_index("condition")

    comp_rows = []
    size_rows = []
    eq_rows = []
    cert_rows = []
    curve_rows = []
    for cname, cmat in cubes_3yr.items():
        point = S.kge_from_sufficient(cmat.sum(axis=2))  # [nm, nb]

        # Block bootstrap replicates
        from _shared.data_loaders import _c4
        best_reps = _c4.bootstrap_best_kge(cmat, block_counts)  # [1000, nb], 50 draws per batch

        q05 = np.quantile(best_reps, 0.05, axis=0)  # [nb]
        eq = (point >= q05[None, :])  # [nm, nb]
        B_mat = eq.T.astype(bool)     # [nb, nm]
        sizes = eq.sum(axis=0)
        for budget in range(1, nm + 1):
            covered, status = solve_max_cov(B_mat, budget)
            curve_rows.append({"condition": cname, "n_models": budget,
                               "basins_covered_exact": covered,
                               "frac_covered_exact": covered / nb, "solver_status": status})

        # Size distribution
        size_rows.append({
            "condition": cname,
            "uncertainty_convention": "multiyear_block_bootstrap_L3",
            "block_length_years": 3,
            "n_blocks": 5,
            "n_replicates": N_BOOT,
            "seed": SEED,
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
            eq_mods = [models[m_idx] for m_idx in np.flatnonzero(eq[:, b_idx])]
            eq_rows.append({
                "basin_id": b_id,
                "condition": cname,
                "q05_multiyear_boot": float(q05[b_idx]),
                "n_equivalent": len(eq_mods),
                "equivalent_models": "|".join(eq_mods),
            })

        # Targets 90%, 95%, 100%
        for target in [0.90, 0.95, 1.00]:
            nd = int(np.ceil(target * nb))
            k_new, cov_new, st_new = solve_min_cover(B_mat, nd)
            max_minus_1, _ = solve_max_cov(B_mat, k_new - 1)

            # Optimality certificate
            cert_rows.append({
                "condition": cname,
                "target": target,
                "target_pct": f"{int(target * 100)}%",
                "need_basins": nd,
                "proven_minimum_models": k_new,
                "budget_minus_1": k_new - 1,
                "max_basins_coverable_with_budget_minus_1": max_minus_1,
                "proves_k_impossible": bool(max_minus_1 < nd),
                "solver_status": st_new,
            })

            # Primary baseline row
            p_row = df_boot_primary[(df_boot_primary.condition == cname) & (np.isclose(df_boot_primary.target, target))].iloc[0]
            k_primary = int(p_row["n_models_exact"])

            comp_rows.append({
                "condition": cname,
                "target": target,
                "target_pct": f"{int(target * 100)}%",
                "need_basins": nd,
                "n_models_1yr_block_bootstrap": k_primary,
                "n_models_3yr_block_bootstrap": k_new,
                "diff_3yr_minus_1yr": k_new - k_primary,
                "median_size_1yr_boot": float(df_boot_size_primary.loc[cname, "median_size"]),
                "median_size_3yr_boot": float(np.median(sizes)),
                "optimality_certified": bool(max_minus_1 < nd and st_new == 0),
            })

    df_comp = pd.DataFrame(comp_rows)
    S.write_tidy_table(df_comp, TABLES / "sec31_multiyear_block_bootstrap_comparison.csv")

    df_cert = pd.DataFrame(cert_rows)
    S.write_tidy_table(df_cert, TABLES / "sec31_multiyear_block_bootstrap_certificates.csv")

    df_size = pd.DataFrame(size_rows)
    S.write_tidy_table(df_size, TABLES / "sec31_multiyear_block_bootstrap_size_distribution.csv")

    df_eq = pd.DataFrame(eq_rows)
    S.write_tidy_table(df_eq, TABLES / "sec31_multiyear_block_bootstrap_equivalence_sets.csv")
    S.write_tidy_table(pd.DataFrame(curve_rows), TABLES / "sec31_multiyear_block_bootstrap_coverage_curve.csv")

    print("[sec31_08] PASS: C1 multi-year block bootstrap completed across all 3 conditions:")
    print(df_comp.to_string(index=False))


if __name__ == "__main__":
    main()
