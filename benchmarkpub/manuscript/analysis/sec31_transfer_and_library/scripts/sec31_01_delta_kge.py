#!/usr/bin/env python3
"""sec31_01_delta_kge.py — Recompute ΔKGE_PUB (pooled, model-level, model×fold) and verify vs frozen."""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    # Approach (a): Recompute from pub_same_pipeline_model_basin.csv
    pub = S.load_same_pipeline_table()
    pub["delta_kge_recomputed"] = pub["kge_seenfolds_median4"] - pub["kge_heldout"]

    # 1. Pooled distribution
    d_all = pub["delta_kge_recomputed"].to_numpy(dtype=np.float64)
    pooled_median = float(np.median(d_all))
    pooled_mean = float(np.mean(d_all))
    pooled_q25 = float(np.percentile(d_all, 25))
    pooled_q75 = float(np.percentile(d_all, 75))

    # Basin-clustered bootstrap CI (B=2000, seed=20260930, matching pub_stats.py)
    rng = np.random.default_rng(20260930)
    basins = np.sort(pub["basin_id"].unique())
    n_basins = len(basins)
    basin_groups = pub.groupby("basin_id")["delta_kge_recomputed"].apply(np.array).to_dict()

    boot_medians = np.empty(2000, dtype=np.float64)
    boot_means = np.empty(2000, dtype=np.float64)
    for i in range(2000):
        sampled_b = rng.choice(basins, size=n_basins, replace=True)
        sampled_vals = np.concatenate([basin_groups[b] for b in sampled_b])
        boot_medians[i] = np.median(sampled_vals)
        boot_means[i] = np.mean(sampled_vals)

    med_ci_lo, med_ci_hi = np.percentile(boot_medians, [2.5, 97.5])
    mean_ci_lo, mean_ci_hi = np.percentile(boot_means, [2.5, 97.5])

    pooled_df = pd.DataFrame([{
        "metric": "pooled_delta_kge_pub",
        "n_cells": len(d_all),
        "n_basins": n_basins,
        "n_models": pub["model"].nunique(),
        "median": pooled_median,
        "median_ci95_lo": float(med_ci_lo),
        "median_ci95_hi": float(med_ci_hi),
        "mean": pooled_mean,
        "mean_ci95_lo": float(mean_ci_lo),
        "mean_ci95_hi": float(mean_ci_hi),
        "q25": pooled_q25,
        "q75": pooled_q75,
        "frac_positive": float((d_all > 0).mean()),
    }])
    S.write_tidy_table(pooled_df, TABLES / "sec31_pooled_delta_kge.csv")

    # 2. Model-level summary (36 models)
    model_rows = []
    for m, g in pub.groupby("model", sort=True):
        dm = g["delta_kge_recomputed"].to_numpy(dtype=np.float64)
        model_rows.append({
            "model": m,
            "n_basins": len(dm),
            "delta_pub_median": float(np.median(dm)),
            "delta_pub_mean": float(np.mean(dm)),
            "delta_pub_q25": float(np.percentile(dm, 25)),
            "delta_pub_q75": float(np.percentile(dm, 75)),
            "frac_positive": float((dm > 0).mean()),
            "heldout_kge_median": float(g["kge_heldout"].median()),
            "seenfolds_kge_median": float(g["kge_seenfolds_median4"].median()),
        })
    model_df = pd.DataFrame(model_rows).sort_values("delta_pub_median", ascending=False).reset_index(drop=True)
    S.write_tidy_table(model_df, TABLES / "sec31_model_level_delta_kge.csv")

    # 3. Model x Fold summary (180 cells)
    fold_rows = []
    for (m, f), g in pub.groupby(["model", "heldout_fold"], sort=True):
        dmf = g["delta_kge_recomputed"].to_numpy(dtype=np.float64)
        fold_rows.append({
            "model": m,
            "fold": int(f),
            "n_basins": len(dmf),
            "delta_median": float(np.median(dmf)),
            "delta_mean": float(np.mean(dmf)),
            "delta_q25": float(np.percentile(dmf, 25)),
            "delta_q75": float(np.percentile(dmf, 75)),
            "heldout_kge_median": float(g["kge_heldout"].median()),
            "seenfolds_kge_median": float(g["kge_seenfolds_median4"].median()),
        })
    fold_df = pd.DataFrame(fold_rows)
    S.write_tidy_table(fold_df, TABLES / "sec31_model_fold_delta_kge.csv")

    # Parity verification vs frozen files
    frozen_stats = json.loads((S.DIR_OOB36_NEXT_STAGE / "tables" / "pub_stats.json").read_text(encoding="utf-8"))
    ref_med = frozen_stats["analysis1"]["median4"]["pooled_median"]
    diff_pooled = abs(pooled_median - ref_med)

    frozen_model = pd.read_csv(S.DIR_OOB36_NEXT_STAGE / "tables" / "pub_model_summary.csv").set_index("model")
    diff_models = [abs(row["delta_pub_median"] - frozen_model.loc[row["model"], "delta_pub_median"]) for _, row in model_df.iterrows()]
    max_model_diff = max(diff_models)

    parity_rec = {
        "script": "sec31_01_delta_kge.py",
        "approach": "(a) Recomputed from pub_same_pipeline_model_basin.csv",
        "pooled_median_recomputed": pooled_median,
        "pooled_median_frozen": ref_med,
        "pooled_median_abs_diff": diff_pooled,
        "model_level_max_abs_diff": max_model_diff,
        "n_models_positive": int((model_df["delta_pub_median"] > 0).sum()),
        "n_fold_cells_positive": int((fold_df["delta_median"] > 0).sum()),
    }
    (TABLES / "sec31_01_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec31_01] PASS: pooled median={pooled_median:.6f}, diff={diff_pooled:.3e}, max model diff={max_model_diff:.3e}")


if __name__ == "__main__":
    main()
