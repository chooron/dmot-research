#!/usr/bin/env python3
"""sec34_08_displacement_correlation_recompute_e7.py — E7: Inter-metric correlation & independent recomputation from raw parameters.

1. Calculates the 3x3 Spearman correlation matrix between the three instability metrics across 36 models.
2. Independently parses the raw per-model parameter files (pub_parsimony_complexity_20260930/params/*.npz),
   computes the three basin medians, and recomputes their Spearman rho and permutation p vs n_parameters.
   Compares recomputed vs frozen benchmark values and reports absolute differences.
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


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> Tuple[float, float]:
    """Exact reproduction of benchmark spearman_perm."""
    rho, _ = sstats.spearmanr(x, y)
    rx = sstats.rankdata(x).astype(float)
    ry = sstats.rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    rx /= np.sqrt((rx ** 2).sum())
    rng = np.random.default_rng(seed)
    idx = rng.random((n_perm, len(x))).argsort(axis=1)
    ryp = ry[idx]
    ryp /= np.sqrt((ryp ** 2).sum(axis=1, keepdims=True))
    perm_rho = ryp @ rx
    p_perm = (np.count_nonzero(np.abs(perm_rho) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)
    return float(rho), float(p_perm)


def main() -> None:
    # 1. 3x3 Inter-metric Spearman correlation matrix
    src_model_perf = S.DIR_OOB36_PARSIMONY / "tables" / "model_level_complexity_performance.csv"
    df_exist = pd.read_csv(src_model_perf)

    metric_cols = [
        "param_fold_instability_median",
        "param_displacement_median",
        "param_displacement_euclid_median",
    ]
    metric_labels = [
        "Fold instability (per-param RMS)",
        "Seen->OOB disp (per-param RMS)",
        "Seen->OOB disp (total Euclid)",
    ]

    mat_corr = np.zeros((3, 3), dtype=np.float64)
    mat_pval = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rho, p = sstats.spearmanr(df_exist[metric_cols[i]], df_exist[metric_cols[j]])
            mat_corr[i, j] = rho
            mat_pval[i, j] = p

    df_corr = pd.DataFrame(mat_corr, columns=metric_cols)
    df_corr.insert(0, "instability_metric", metric_cols)
    df_pval = pd.DataFrame(mat_pval, columns=metric_cols)
    df_pval.insert(0, "instability_metric", metric_cols)
    S.write_tidy_table(df_corr, TABLES / "sec34_displacement_metrics_correlation_matrix.csv")
    S.write_tidy_table(df_pval, TABLES / "sec34_displacement_metrics_correlation_pvalues.csv")

    # 2. Independent recomputation from raw parameter npz files
    params_dir = S.DIR_OOB36_PARSIMONY / "params"
    raw_npz_files = sorted(list(params_dir.glob("*.npz")))
    assert len(raw_npz_files) == 36, f"Expected 36 npz files, got {len(raw_npz_files)}"

    recomp_model_rows = []
    for p_path in raw_npz_files:
        m = p_path.stem
        z = np.load(p_path)
        basin_ids = z["basin_ids"]
        order = np.argsort(basin_ids)
        # stored [fold, basin, P] -> [basin, fold, P], sorted by basin_id
        T = np.transpose(z["theta_norm"], (1, 0, 2))[order]
        hf = z["heldout_fold"][order]
        P = T.shape[2]
        nb = len(order)

        b_idx = np.arange(nb)
        seen = np.empty((nb, 4, P), dtype=np.float64)
        for i in b_idx:
            seen[i] = np.delete(T[i], hf[i], axis=0)
        oob = T[b_idx, hf, :]

        # Fold instability (pairwise RMS across 4 seen folds)
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        pr_rms = np.empty((nb, 6))
        for k, (i, j) in enumerate(pairs):
            d = seen[:, i, :] - seen[:, j, :]
            pr_rms[:, k] = np.sqrt(np.mean(d ** 2, axis=1))
        fold_inst = pr_rms.mean(axis=1)

        # Seen -> OOB displacement
        seen_mean = seen.mean(axis=1)
        d_mean = seen_mean - oob
        disp_rms = np.sqrt(np.mean(d_mean ** 2, axis=1))
        disp_euclid = np.sqrt(np.sum(d_mean ** 2, axis=1))

        recomp_model_rows.append({
            "model": m,
            "n_parameters": P,
            "param_fold_instability_median": float(np.median(fold_inst)),
            "param_displacement_median": float(np.median(disp_rms)),
            "param_displacement_euclid_median": float(np.median(disp_euclid)),
        })

    df_recomp_models = pd.DataFrame(recomp_model_rows).sort_values("model").reset_index(drop=True)

    # Load existing benchmark values from complexity_vs_transfer.csv
    df_frozen_trans = pd.read_csv(S.DIR_OOB36_PARSIMONY / "tables" / "complexity_vs_transfer.csv").set_index("y_quantity")

    recomp_summary_rows = []
    n_p = df_recomp_models["n_parameters"].to_numpy(float)

    for col, desc in zip(metric_cols, metric_labels):
        y_recomp = df_recomp_models[col].to_numpy(float)
        rho_rec, p_rec = spearman_perm(n_p, y_recomp, n_perm=N_PERM, seed=SEED)

        rho_exist = float(df_frozen_trans.loc[col, "spearman_rho"])
        p_exist = float(df_frozen_trans.loc[col, "spearman_p_permutation"])

        recomp_summary_rows.append({
            "metric_name": col,
            "metric_description": desc,
            "n_models": len(df_recomp_models),
            "recomputed_spearman_rho": rho_rec,
            "existing_spearman_rho": rho_exist,
            "abs_diff_spearman_rho": abs(rho_rec - rho_exist),
            "recomputed_p_permutation": p_rec,
            "existing_p_permutation": p_exist,
            "abs_diff_p_permutation": abs(p_rec - p_exist),
            "n_permutations": N_PERM,
            "seed": SEED,
            "input_source": "params/*.npz (theta_norm arrays; 36 models)",
        })
        print(f"[sec34_08] {col}:")
        print(f"  rho:    recomputed={rho_rec:.6f}, existing={rho_exist:.6f}, abs_diff={abs(rho_rec - rho_exist):.3e}")
        print(f"  p_perm: recomputed={p_rec:.6f}, existing={p_exist:.6f}, abs_diff={abs(p_rec - p_exist):.3e}")

    df_recomp_sum = pd.DataFrame(recomp_summary_rows)
    S.write_tidy_table(df_recomp_sum, TABLES / "sec34_nparam_instability_independent_recomputation.csv")

    print("[sec34_08] PASS: 3x3 correlation matrix and independent recomputation completed")


if __name__ == "__main__":
    main()
