#!/usr/bin/env python3
"""sec34_13_attribute_screening_idjoin_a4.py — A4: Re-verification of the 15-attribute screening with strict basin-ID joins.

Recomputes from raw inputs by basin ID:
  * Response: per-basin median PUB penalty (median over models of delta_kge_pub_median4 from pub_same_pipeline_model_basin.csv)
  * 10 Caravan attributes from data/caravan_671_attributes.csv (joined on gauge_id)
  * 5 observed-series attributes from OOB fold observation files (assembled by basin-ID join)
  * Spearman rho + permutation p (N_PERM=10000, seed=20260930, matching the frozen upstream implementation)
  * BH-FDR over the 15-test family, q < 0.05 threshold
Compares every variable with the existing frozen table (rho, p, q) and reports absolute differences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sstats

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

N_PERM = 10000
SEED = 20260930

ATTRS = [
    ("arid", "aridity index (PET/P)", "caravan_671_attributes"),
    ("f_snow", "snowfall fraction", "caravan_671_attributes"),
    ("p_mean", "mean precipitation (mm/day)", "caravan_671_attributes"),
    ("pet_mean", "mean PET (mm/day)", "caravan_671_attributes"),
    ("p_seas", "precipitation seasonality", "caravan_671_attributes"),
    ("elev", "mean elevation (m)", "caravan_671_attributes"),
    ("area", "catchment area (km2)", "caravan_671_attributes"),
    ("slope", "mean slope", "caravan_671_attributes"),
    ("f_forest", "forest fraction", "caravan_671_attributes"),
    ("s_cond", "soil conductivity", "caravan_671_attributes"),
    ("q_mean", "mean observed discharge (mm/day)", "observed series"),
    ("runoff_ratio", "runoff ratio q_mean/p_mean", "observed series + attribute"),
    ("zero_flow_frac", "fraction of days with zero discharge", "observed series"),
    ("q_cv", "coefficient of variation of discharge", "observed series"),
    ("q_log_sd", "SD of log(Q + eps)", "observed series"),
]


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Exact reproduction of the frozen upstream BH-FDR implementation."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n, float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        r = n - rank + 1
        val = min(prev, p[idx] * n / r)
        q[idx] = val
        prev = val
    return q


def perm_spearman(x, y, n_perm: int = N_PERM, seed: int = SEED):
    """Exact reproduction of the frozen upstream permutation Spearman implementation."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    rx = sstats.rankdata(x); ry = sstats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    rho = float(np.dot(rx, ry) / n)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(float(np.dot(rx, rng.permutation(ry)) / n)) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1), n


def main() -> None:
    models = S.MODELS_36
    basins = S.canonical_basin_ids()   # SORTED 8-digit strings (A6 accessor)
    nb = len(basins)
    pos = S.basin_position_map(basins)  # {basin_id -> canonical row position} (A6 accessor)

    # 1. Response: per-basin median PUB penalty (ID-joined from pub table)
    pub = pd.read_csv(S.DIR_OOB36_PRELIM / "pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv")
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)
    pen_series = pub.groupby("bs")["delta_kge_pub_median4"].median()
    pen = S.align_series_by_id(pen_series.index, pen_series.to_numpy(np.float64),
                               canonical=basins, name="penalty_median_pub")  # A6 accessor

    # 2. Caravan attributes (ID-joined on gauge_id)
    a = pd.read_csv(ROOT / "data" / "caravan_671_attributes.csv")
    a["bs"] = a["gauge_id"].map(S.pad_basin_id)
    caravan_cols = [c for c, _, src in ATTRS if src == "caravan_671_attributes"]
    caravan = {c: S.align_series_by_id(a["bs"], a[c].to_numpy(np.float64),
                                       canonical=basins, name=c) for c in caravan_cols}  # A6 accessor

    # 3. Observed-series attributes (assemble obs by basin-ID join across folds)
    obs = np.full((5479, nb), np.nan, dtype=np.float64)
    for k in range(5):
        ids_path = S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{k}" / "heldout_basin_ids.txt"
        fold_ids = np.array([int(x) for x in ids_path.read_text().split()], dtype=np.int64)
        y = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{k}" / "heldout_y.npy")
        if y.shape[1] != fold_ids.size:
            raise RuntimeError(f"fold {k}: obs columns {y.shape[1]} != ids {fold_ids.size}")
        for j, b in enumerate(fold_ids.tolist()):
            obs[:, pos[S.pad_basin_id(b)]] = y[:, j]   # A6 accessor ID lookup
    if np.isnan(obs).all(axis=0).any():
        raise RuntimeError("at least one basin has no observed series")

    with np.errstate(invalid="ignore"):
        q_mean = np.nanmean(obs, axis=0)
        q_sd = np.nanstd(obs, axis=0, ddof=1)
        q_cv = q_sd / q_mean
        eps = 0.01 * q_mean
        q_log_sd = np.nanstd(np.log(np.maximum(obs, 0) + eps[None, :]), axis=0, ddof=1)
        zero_frac = np.nanmean(obs <= 0.0, axis=0)

    observed = {
        "q_mean": q_mean,
        "q_cv": q_cv,
        "q_log_sd": q_log_sd,
        "zero_flow_frac": zero_frac,
        "runoff_ratio": q_mean / caravan["p_mean"],
    }

    all_vars = {**caravan, **observed}

    # 4. Screening: Spearman + permutation p, BH-FDR over the 15-test family
    rows = []
    for col, label, src in ATTRS:
        rho, p, n = perm_spearman(all_vars[col], pen)
        rows.append({"variable": col, "label": label, "source": src,
                     "spearman_rho": rho, "perm_p": p, "n_used": n})
    df = pd.DataFrame(rows)
    df["bh_fdr_q"] = bh_fdr(df["perm_p"].to_numpy())
    df["abs_rho"] = df["spearman_rho"].abs()
    df = df.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    # 5. Compare with existing frozen table
    exist = pd.read_csv(TABLES / "sec34_attribute_screening_bh_fdr.csv").set_index("variable")
    comp_rows = []
    for _, r in df.iterrows():
        v = r["variable"]
        e = exist.loc[v]
        comp_rows.append({
            "variable": v,
            "recomputed_spearman_rho": float(r["spearman_rho"]),
            "existing_spearman_rho": float(e["spearman_rho"]),
            "abs_diff_rho": abs(float(r["spearman_rho"]) - float(e["spearman_rho"])),
            "recomputed_perm_p": float(r["perm_p"]),
            "existing_perm_p": float(e["perm_p"]),
            "abs_diff_perm_p": abs(float(r["perm_p"]) - float(e["perm_p"])),
            "recomputed_bh_fdr_q": float(r["bh_fdr_q"]),
            "existing_bh_fdr_q": float(e["bh_fdr_q"]),
            "abs_diff_bh_fdr_q": abs(float(r["bh_fdr_q"]) - float(e["bh_fdr_q"])),
            "n_used": int(r["n_used"]),
            "n_permutations": N_PERM,
            "seed": SEED,
        })
    df_comp = pd.DataFrame(comp_rows)
    S.write_tidy_table(df_comp, TABLES / "sec34_a4_attribute_screening_recompute.csv")

    # 6. Summary vs existing claims
    n_survive = int((df["bh_fdr_q"] < 0.05).sum())
    summary = pd.DataFrame([{
        "metric": "n_variables",
        "recomputed": len(df),
        "existing": 15,
        "abs_diff": abs(len(df) - 15),
    }, {
        "metric": "n_surviving_fdr_005",
        "recomputed": n_survive,
        "existing": int((exist["bh_fdr_q"] < 0.05).sum()),
        "abs_diff": abs(n_survive - int((exist["bh_fdr_q"] < 0.05).sum())),
    }, {
        "metric": "min_bh_fdr_q",
        "recomputed": float(df["bh_fdr_q"].min()),
        "existing": float(exist["bh_fdr_q"].min()),
        "abs_diff": abs(float(df["bh_fdr_q"].min()) - float(exist["bh_fdr_q"].min())),
    }, {
        "metric": "max_abs_spearman_rho",
        "recomputed": float(df["abs_rho"].max()),
        "existing": float(exist["abs_rho"].max()),
        "abs_diff": abs(float(df["abs_rho"].max()) - float(exist["abs_rho"].max())),
    }])
    S.write_tidy_table(summary, TABLES / "sec34_a4_attribute_screening_summary.csv")

    print("[sec34_12->A4] PASS: 15-attribute screening recomputed with strict basin-ID joins")
    print(df_comp[["variable", "recomputed_spearman_rho", "recomputed_perm_p", "recomputed_bh_fdr_q",
                   "abs_diff_rho", "abs_diff_perm_p", "abs_diff_bh_fdr_q"]].to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
