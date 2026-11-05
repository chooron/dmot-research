#!/usr/bin/env python3
"""sec31_09_uncertainty_bandwidth_comparison_c2.py — C2: Uncertainty bandwidth comparison across 3 resampling schemes.

Compares the uncertainty bandwidth (Delta_b = KGE_best^point - q05_b) across:
1. Primary 1-year water-year block bootstrap (L=1 year, 15 blocks, 1000 draws)
2. Leave-one-water-year jackknife (SUPERSEDED: 15 replicates of 14 pooled years)
3. C1 Multi-year block bootstrap (L=3 years, 5 blocks, 1000 draws)
Evaluated across IC, seen-dPL, and OOB-dPL conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
    b3 = np.zeros((nm, nb, 5, 6), dtype=np.float64)
    for b_idx in range(5):
        b3[:, :, b_idx, :] = cube[:, :, b_idx * 3 : (b_idx + 1) * 3, :].sum(axis=2)
    return b3


def main() -> None:
    models = S.MODELS_36
    basins = sorted(S.canonical_ids())  # OOB cube uses c4.stats_cube's sorted basin order
    basins_str = [str(b).zfill(8) for b in basins]
    nb, nm = len(basins), len(models)

    # 1. Load sufficient stats cubes
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

    cubes_1yr = {"IC": cube_ic, "seen-dPL": cube_dp, "OOB-dPL": cube_oob}

    # 2. Resampling counts
    counts_1yr = S.load_year_counts(n_reps=N_BOOT, seed=SEED)

    rng = np.random.default_rng(SEED)
    counts_3yr = np.zeros((N_BOOT, 5), dtype=np.float64)
    for r in range(N_BOOT):
        counts_3yr[r] = np.bincount(rng.integers(0, 5, size=5), minlength=5)

    summary_rows = []
    basin_rows = []

    for cname, cmat in cubes_1yr.items():
        point = S.kge_from_sufficient(cmat.sum(axis=2))
        best_pt = point.max(axis=0)  # [nb]

        # Scheme 1: Primary 1-year block bootstrap (L=1, 15 blocks)
        flat_1yr = cmat.transpose(2, 1, 0, 3).reshape(15, nb * nm * 6)
        pool_1yr = (counts_1yr @ flat_1yr).reshape(N_BOOT, nb, nm, 6)
        best_1yr = S.kge_from_sufficient(pool_1yr).max(axis=2)
        q05_1yr = np.quantile(best_1yr, 0.05, axis=0)
        width_1yr = best_pt - q05_1yr

        # Scheme 2: Leave-one-water-year jackknife (SUPERSEDED)
        tot_15yr = cmat.sum(axis=2)
        best_jk = np.empty((15, nb), dtype=np.float64)
        for y in range(15):
            best_jk[y] = S.kge_from_sufficient(tot_15yr - cmat[:, :, y, :]).max(axis=0)
        q05_jk = np.quantile(best_jk, 0.05, axis=0)
        width_jk = best_pt - q05_jk

        # Scheme 3: C1 Multi-year block bootstrap (L=3, 5 blocks)
        c3 = to_3yr_blocks(cmat)
        flat_3yr = c3.transpose(2, 1, 0, 3).reshape(5, nb * nm * 6)
        pool_3yr = (counts_3yr @ flat_3yr).reshape(N_BOOT, nb, nm, 6)
        best_3yr = S.kge_from_sufficient(pool_3yr).max(axis=2)
        q05_3yr = np.quantile(best_3yr, 0.05, axis=0)
        width_3yr = best_pt - q05_3yr

        schemes = [
            ("1yr_block_bootstrap_primary", "PRIMARY", width_1yr),
            ("leave_one_water_year_jackknife", "SUPERSEDED", width_jk),
            ("3yr_multiyear_block_bootstrap_c1", "ACTIVE_ALTERNATIVE", width_3yr),
        ]

        for s_name, s_status, w_arr in schemes:
            summary_rows.append({
                "condition": cname,
                "scheme": s_name,
                "scheme_status": s_status,
                "n_basins": nb,
                "bandwidth_definition": "best_model_point_kge_minus_q05_threshold",
                "median_width": float(np.median(w_arr)),
                "q25_width": float(np.quantile(w_arr, 0.25)),
                "q75_width": float(np.quantile(w_arr, 0.75)),
                "min_width": float(w_arr.min()),
                "max_width": float(w_arr.max()),
                "mean_width": float(w_arr.mean()),
                "std_width": float(w_arr.std(ddof=1)),
            })

        for b_idx, b_id in enumerate(basins_str):
            basin_rows.append({
                "basin_id": b_id,
                "condition": cname,
                "best_point_kge": float(best_pt[b_idx]),
                "bandwidth_1yr_block_bootstrap_primary": float(width_1yr[b_idx]),
                "bandwidth_jackknife_superseded": float(width_jk[b_idx]),
                "bandwidth_3yr_block_bootstrap_c1": float(width_3yr[b_idx]),
            })

        print(f"[sec31_09] [{cname}] Uncertainty Band Width (best_point - q05):")
        print(f"  Primary 1-yr boot:       median={np.median(width_1yr):.4f} [Q25={np.quantile(width_1yr, 0.25):.4f}, Q75={np.quantile(width_1yr, 0.75):.4f}], range [{width_1yr.min():.4f}, {width_1yr.max():.4f}]")
        print(f"  Jackknife (SUPERSEDED):  median={np.median(width_jk):.4f} [Q25={np.quantile(width_jk, 0.25):.4f}, Q75={np.quantile(width_jk, 0.75):.4f}], range [{width_jk.min():.4f}, {width_jk.max():.4f}]")
        print(f"  C1 3-yr block boot:      median={np.median(width_3yr):.4f} [Q25={np.quantile(width_3yr, 0.25):.4f}, Q75={np.quantile(width_3yr, 0.75):.4f}], range [{width_3yr.min():.4f}, {width_3yr.max():.4f}]")

    df_sum = pd.DataFrame(summary_rows)
    S.write_tidy_table(df_sum, TABLES / "sec31_uncertainty_bandwidth_comparison.csv")

    df_basin = pd.DataFrame(basin_rows)
    S.write_tidy_table(df_basin, TABLES / "sec31_uncertainty_bandwidth_per_basin.csv")

    print("[sec31_09] PASS: compiled uncertainty bandwidth comparison across 3 schemes")


if __name__ == "__main__":
    main()
