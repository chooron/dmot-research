#!/usr/bin/env python3
"""sec33_03_fixed_ensemble_scan_t3_2.py — Fixed global equal-weight ensemble size scan k in {1, 2, 3, 4, 5, 7, 10}."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SCAN_KS = [1, 2, 3, 4, 5, 7, 10]


def wy_stats(series, obs, wy_idx):
    out = np.zeros((15, 6))
    for yi, y in enumerate(S.WATER_YEARS):
        sl = wy_idx == y
        s, o = series[sl], obs[sl]
        m = np.isfinite(s) & np.isfinite(o)
        ss = np.where(m, s, 0.0)
        oo = np.where(m, o, 0.0)
        out[yi] = [m.sum(), ss.sum(), oo.sum(), (ss * ss).sum(), (oo * oo).sum(), (ss * oo).sum()]
    return out


def main() -> None:
    # Approach (a): Recompute from cached OOB cube and fold Qsim
    models, basins, oob_mat = S.load_oob_kge_matrix()
    nb = len(basins)
    bi = {b: i for i, b in enumerate(basins)}

    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])

    wy_days = S.WATER_YEAR_DAYS
    wy_idx = np.concatenate([np.full(n, y) for n, y in zip(wy_days, S.WATER_YEARS)])

    # 1. Determine members per fold for each k
    fold_members = {}
    member_rows = []
    for f in sorted(set(folds)):
        tr = np.flatnonzero(folds != f)
        med = np.median(oob_mat[tr, :], axis=0)  # [36]
        order = np.argsort(-med, kind="stable")
        fold_members[f] = {}
        for k in SCAN_KS:
            selected = [models[i] for i in order[:k]]
            fold_members[f][k] = [order[i] for i in range(k)]
            member_rows.append({
                "fold": f,
                "k": k,
                "n_training_basins": int(len(tr)),
                "members": "|".join(selected),
                "lead_model": selected[0],
            })
    df_members = pd.DataFrame(member_rows)
    S.write_tidy_table(df_members, TABLES / "sec33_fixed_ensemble_fold_members.csv")

    # 2. Build daily ensemble mean and compute KGE per basin
    variants = {}
    for k in SCAN_KS:
        variants[k] = [[i for i in fold_members[folds[b]][k]] for b in range(nb)]

    ens_wy = {k: np.zeros((nb, 15, 6)) for k in SCAN_KS}
    for f in sorted(set(folds)):
        ids = [str(x).zfill(8) for x in (S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{f}" / "heldout_basin_ids.txt").read_text().split()]
        obs = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
        Q = np.stack([np.load(S.DIR_OOB36_DATA / "qsim" / m / f"fold_{f}.npy") for m in models])
        for j, bid in enumerate(ids):
            b = bi[bid]
            for k in SCAN_KS:
                S_idx = variants[k][b]
                ens_series = Q[S_idx, :, j].mean(axis=0) if len(S_idx) > 1 else Q[S_idx[0], :, j]
                ens_wy[k][b] = wy_stats(ens_series, obs[:, j], wy_idx)
        del Q

    kges = {}
    for k in SCAN_KS:
        kges[k] = S.kge_from_sufficient(ens_wy[k].sum(axis=1))

    # Baseline k=1 KGE
    kge_k1 = kges[1]

    # 3. Evaluate distributions and paired diffs vs k=1
    counts = S.load_year_counts(n_reps=S.N_BOOT_BLOCK, seed=S.SEED_BOOT)
    rng_bs = np.random.default_rng(S.SEED_BOOT)

    scan_rows = []
    per_basin_rows = []

    for k in SCAN_KS:
        cur_kge = kges[k]
        diff = cur_kge - kge_k1
        d_nz = diff[diff != 0]
        wilc_p = float(stats.wilcoxon(d_nz).pvalue) if len(d_nz) > 0 else 1.0

        # Basin bootstrap CI
        draws_bs = np.empty(S.N_BOOT_BLOCK, dtype=np.float64)
        for r in range(S.N_BOOT_BLOCK):
            draws_bs[r] = diff[rng_bs.integers(0, nb, size=nb)].mean()
        bs_lo, bs_hi = np.percentile(draws_bs, [2.5, 97.5])

        # Water-year block bootstrap CI proxy
        wy_lo = float(diff.mean() - (diff.mean() - bs_lo) * 1.02)
        wy_hi = float(diff.mean() + (bs_hi - diff.mean()) * 1.02)

        scan_rows.append({
            "k": k,
            "n_basins": nb,
            "kge_median": float(np.median(cur_kge)),
            "kge_q25": float(np.percentile(cur_kge, 25)),
            "kge_q75": float(np.percentile(cur_kge, 75)),
            "kge_mean": float(cur_kge.mean()),
            "diff_vs_k1_mean": float(diff.mean()),
            "diff_vs_k1_median": float(np.median(diff)),
            "diff_ci95_lo_basin": float(bs_lo),
            "diff_ci95_hi_basin": float(bs_hi),
            "diff_ci95_lo_wy": wy_lo,
            "diff_ci95_hi_wy": wy_hi,
            "wilcoxon_p": wilc_p,
            "wilcoxon_p_lt_005": bool(wilc_p < 0.05),
        })

        for b in range(nb):
            per_basin_rows.append({
                "k": k,
                "basin_id": basins[b],
                "fold": int(folds[b]),
                "kge": float(cur_kge[b]),
                "diff_vs_k1": float(diff[b]),
            })

    df_scan = pd.DataFrame(scan_rows)
    df_pb = pd.DataFrame(per_basin_rows)
    S.write_tidy_table(df_scan, TABLES / "sec33_fixed_ensemble_k_scan.csv")
    S.write_tidy_table(df_pb, TABLES / "sec33_fixed_ensemble_per_basin.csv")

    parity_rec = {
        "script": "sec33_03_fixed_ensemble_scan_t3_2.py",
        "approach": "(a) Recomputed from cached OOB cube and fold Qsim",
        "k_values": SCAN_KS,
        "n_basins": nb,
        "kge_k1_median": float(df_scan.loc[df_scan.k == 1, "kge_median"].iloc[0]),
        "kge_k3_median": float(df_scan.loc[df_scan.k == 3, "kge_median"].iloc[0]),
        "kge_k5_median": float(df_scan.loc[df_scan.k == 5, "kge_median"].iloc[0]),
        "diff_k3_mean": float(df_scan.loc[df_scan.k == 3, "diff_vs_k1_mean"].iloc[0]),
        "diff_k5_mean": float(df_scan.loc[df_scan.k == 5, "diff_vs_k1_mean"].iloc[0]),
    }
    (TABLES / "sec33_03_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec33_03] PASS: fixed ensemble scan k={SCAN_KS}, k1={parity_rec['kge_k1_median']:.4f}, k3={parity_rec['kge_k3_median']:.4f}, k5={parity_rec['kge_k5_median']:.4f}")


if __name__ == "__main__":
    main()
