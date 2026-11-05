#!/usr/bin/env python3
"""sec34_09_equivalence_dispersion_c4.py — C4: Point KGE dispersion within performance equivalence sets.

Evaluates the internal point KGE dispersion within each basin's equivalence set across:
  * IC, seen-dPL, and OOB-dPL conditions
Reports distribution metrics for:
  1. Point KGE Range (max - min)
  2. Point KGE Interquartile Range (IQR)
  3. Equivalence Set Size (|S_b|)
  4. Best member vs worst member point KGE difference (median, Q25, Q75, P90)
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


def main() -> None:
    models = S.MODELS_36
    basins = S.canonical_ids()
    basins_str = [str(b).zfill(8) for b in basins]
    nb, nm = len(basins), len(models)

    # 1. Performance matrices
    models_oob, _, oob_mat = S.load_oob_kge_matrix()  # [531, 36]
    models_seen, _, seen_mat = S.load_canonical_seen_kge_matrix()  # [531, 36]

    # IC matrix
    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    mi = {m: i for i, m in enumerate(models)}
    bi = {b: i for i, b in enumerate(basins_str)}
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    cube_ic = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    idx_m = df_raw["model"].map(mi).to_numpy()
    idx_b = df_raw["basin_id"].astype(str).str.zfill(8).map(bi).to_numpy()
    idx_y = df_raw["water_year"].map(lambda y: y - 1996).to_numpy()
    cube_ic[idx_m, idx_b, idx_y] = df_raw[stat_cols_ic].to_numpy(dtype=np.float64)
    ic_mat = S.kge_from_sufficient(cube_ic.sum(axis=2)).T  # [531, 36]

    perf_mats = {
        "OOB-dPL": (oob_mat, models_oob),
        "seen-dPL": (seen_mat, models_seen),
        "IC": (ic_mat, models),
    }

    # 2. Equivalence sets
    eq_df = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")

    summary_rows = []
    basin_rows = []

    for cond, (mat, mod_list) in perf_mats.items():
        sub_eq = eq_df[eq_df.condition == cond].set_index("basin_id")

        ranges = np.empty(nb, dtype=np.float64)
        iqrs = np.empty(nb, dtype=np.float64)
        sizes = np.empty(nb, dtype=np.float64)
        best_worst_diffs = np.empty(nb, dtype=np.float64)

        for b_idx, bid in enumerate(basins_str):
            eq_mods = str(sub_eq.loc[int(bid), "equivalent_models"]).split("|")
            m_indices = [mod_list.index(m) for m in eq_mods]
            kges = mat[b_idx, m_indices]

            s_size = len(kges)
            k_min = float(kges.min())
            k_max = float(kges.max())
            k_range = k_max - k_min
            k_iqr = float(np.quantile(kges, 0.75) - np.quantile(kges, 0.25)) if s_size > 1 else 0.0

            sizes[b_idx] = s_size
            ranges[b_idx] = k_range
            iqrs[b_idx] = k_iqr
            best_worst_diffs[b_idx] = k_range

            basin_rows.append({
                "basin_id": bid,
                "condition": cond,
                "equivalence_set_size": int(s_size),
                "point_kge_min": k_min,
                "point_kge_max": k_max,
                "point_kge_range": k_range,
                "point_kge_iqr": k_iqr,
                "best_minus_worst_diff": k_range,
            })

        metric_dict = {
            "point_kge_range": ranges,
            "point_kge_iqr": iqrs,
            "equivalence_set_size": sizes,
            "best_minus_worst_diff": best_worst_diffs,
        }

        for m_name, arr in metric_dict.items():
            summary_rows.append({
                "condition": cond,
                "metric": m_name,
                "n_basins": nb,
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "p90": float(np.quantile(arr, 0.90)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)),
            })

        print(f"[sec34_09] [{cond}] Within-Equivalence-Set Dispersion:")
        print(f"  Range (best-worst): median={np.median(ranges):.4f} [Q25={np.quantile(ranges, 0.25):.4f}, Q75={np.quantile(ranges, 0.75):.4f}, P90={np.quantile(ranges, 0.90):.4f}]")
        print(f"  IQR:                median={np.median(iqrs):.4f} [Q25={np.quantile(iqrs, 0.25):.4f}, Q75={np.quantile(iqrs, 0.75):.4f}]")
        print(f"  Set Size:           median={np.median(sizes):.1f} [Q25={np.quantile(sizes, 0.25):.1f}, Q75={np.quantile(sizes, 0.75):.1f}]")

    df_sum = pd.DataFrame(summary_rows)
    S.write_tidy_table(df_sum, TABLES / "sec34_equivalence_set_dispersion_summary.csv")

    df_basin = pd.DataFrame(basin_rows)
    S.write_tidy_table(df_basin, TABLES / "sec34_equivalence_set_dispersion_per_basin.csv")

    print("[sec34_09] PASS: compiled equivalence set dispersion tables")


if __name__ == "__main__":
    main()
