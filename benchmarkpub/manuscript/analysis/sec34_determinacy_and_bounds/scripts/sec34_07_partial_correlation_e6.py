#!/usr/bin/env python3
"""sec34_07_partial_correlation_e6.py — E6: Partial Spearman correlation controlling for model performance.

Computes partial Spearman rank correlation between parameter count (n_parameters) and the three
instability metrics controlling for model-level median KGE (both seen_kge_median and oob_kge_median).
Two-sided permutation test with N_PERM=10000, seed=20260930 matching the benchmark protocol.
Reports uncontrolled and controlled metrics side-by-side.
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
    """Uncontrolled Spearman correlation and permutation p-value."""
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


def partial_spearman_perm(x: np.ndarray, y: np.ndarray, z: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> Tuple[float, float]:
    """Partial Spearman correlation controlling for z and two-sided permutation p-value."""
    rx = sstats.rankdata(x).astype(float)
    ry = sstats.rankdata(y).astype(float)
    rz = sstats.rankdata(z).astype(float)

    # Residualize rx on rz (with intercept)
    A_z = np.column_stack([np.ones_like(rz), rz])
    coef_x, _, _, _ = np.linalg.lstsq(A_z, rx, rcond=None)
    res_x = rx - A_z @ coef_x

    coef_y, _, _, _ = np.linalg.lstsq(A_z, ry, rcond=None)
    res_y = ry - A_z @ coef_y

    # Pearson correlation of ranked residuals = partial Spearman
    res_x_c = res_x - res_x.mean()
    res_y_c = res_y - res_y.mean()
    denom = np.sqrt(np.sum(res_x_c ** 2) * np.sum(res_y_c ** 2))
    rho_partial = float(np.sum(res_x_c * res_y_c) / denom)

    # Permutation test on residuals
    rng = np.random.default_rng(seed)
    idx = rng.random((n_perm, len(x))).argsort(axis=1)
    res_y_perm = res_y_c[idx]
    res_y_perm_denom = np.sqrt(np.sum(res_y_perm ** 2, axis=1, keepdims=True))
    res_x_norm = res_x_c / np.sqrt(np.sum(res_x_c ** 2))
    perm_rhos = (res_y_perm / res_y_perm_denom) @ res_x_norm
    p_perm = (np.count_nonzero(np.abs(perm_rhos) >= abs(rho_partial) - 1e-12) + 1) / (n_perm + 1)

    return float(rho_partial), float(p_perm)


def main() -> None:
    src_path = S.DIR_OOB36_PARSIMONY / "tables" / "model_level_complexity_performance.csv"
    df = pd.read_csv(src_path)

    y_vars = [
        ("param_fold_instability_median", "Fold-to-fold variability (per-parameter RMS)"),
        ("param_displacement_median", "Seen->OOB displacement (per-parameter RMS)"),
        ("param_displacement_euclid_median", "Seen->OOB displacement (total Euclidean)"),
    ]

    n_params = df["n_parameters"].to_numpy(float)
    seen_kge = df["seen_kge_median"].to_numpy(float)
    oob_kge = df["oob_kge_median"].to_numpy(float)

    rows = []
    for yv, desc in y_vars:
        y = df[yv].to_numpy(float)
        rho_un, p_un = spearman_perm(n_params, y)
        rho_c_seen, p_c_seen = partial_spearman_perm(n_params, y, seen_kge)
        rho_c_oob, p_c_oob = partial_spearman_perm(n_params, y, oob_kge)

        rows.append({
            "instability_metric": yv,
            "metric_description": desc,
            "n_models": len(df),
            "spearman_rho_uncontrolled": rho_un,
            "p_permutation_uncontrolled": p_un,
            "partial_spearman_rho_control_seen_kge": rho_c_seen,
            "p_permutation_control_seen_kge": p_c_seen,
            "partial_spearman_rho_control_oob_kge": rho_c_oob,
            "p_permutation_control_oob_kge": p_c_oob,
            "n_permutations": N_PERM,
            "seed": SEED,
        })
        print(f"[sec34_07] {yv}:")
        print(f"  Uncontrolled:            rho={rho_un:+.4f}, p_perm={p_un:.4f}")
        print(f"  Controlling seen_kge:    rho={rho_c_seen:+.4f}, p_perm={p_c_seen:.4f}")
        print(f"  Controlling oob_kge:     rho={rho_c_oob:+.4f}, p_perm={p_c_oob:.4f}")

    df_out = pd.DataFrame(rows)
    S.write_tidy_table(df_out, TABLES / "sec34_partial_correlation_nparam_vs_instability.csv")
    print("[sec34_07] PASS: compiled uncontrolled and controlled partial Spearman correlations")


if __name__ == "__main__":
    main()
