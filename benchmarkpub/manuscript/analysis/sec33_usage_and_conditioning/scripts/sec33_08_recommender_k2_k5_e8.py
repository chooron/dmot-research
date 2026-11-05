#!/usr/bin/env python3
"""sec33_08_recommender_k2_k5_e8.py — E8: Re-fit attribute recommenders, verify parity, and evaluate k=2, 5 operating points.

Protocol matching oob36_gset_gate_20261001:
  * RF(500 trees, min_samples_leaf=5) per model per fold
  * 35 raw Caravan attributes (zscore=False)
  * frac_snow >= 0.30 snow pre-filter
  * 5 registered OOB folds (OOB_FOLD_ASSIGNMENT.csv)
  * 10 recommendation seeds (20260925..20260934)
Evaluates:
  1. Parity on Top-3 A2d and A1b G_set against gate outputs (must match to <= 1e-6)
  2. Rule C (attribute-conditioned Top-k) vs Rule B (fixed global Top-k) at k=2 and k=5
  3. G_set for k=2 and k=5 under A2d (PRIMARY) and A1b (SECONDARY) across all 10 seeds
  4. Per-fold leakage audit (target fold strictly excluded from training)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Load c4_common
C4_PATH = ROOT / "project" / "benchmark" / "analysis" / "chapter4_equivalence_recommendability_20260925"
_spec = importlib.util.spec_from_file_location("c4_common", C4_PATH / "scripts" / "common.py")
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)

SEEDS = S.SEEDS_10
SEED_BOOT = S.SEED_BOOT
N_BOOT_BASIN = S.N_BOOT_BASIN  # 10000 draws
N_BOOT_BLOCK = S.N_BOOT_BLOCK  # 1000 draws


def fit_single_model(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, seed: int) -> np.ndarray:
    """Fit a single RF and predict test fold."""
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te)


def bootG(rec: np.ndarray, base: np.ndarray, vbs: np.ndarray, seed: int = SEED_BOOT, n: int = N_BOOT_BASIN) -> Tuple[float, float, float]:
    """Calculate G_set ratio and 95% basin bootstrap CI."""
    rng = np.random.default_rng(seed)
    num = rec - base
    den = vbs - base
    nn = len(rec)
    point = float(num.mean() / den.mean())
    draws = np.empty(n)
    for i in range(n):
        j = rng.integers(0, nn, nn)
        dm = den[j].mean()
        draws[i] = num[j].mean() / dm if dm != 0 else np.nan
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def wy_stats(series: np.ndarray, obs: np.ndarray, wy_idx: np.ndarray) -> np.ndarray:
    """Compute sufficient statistics per water year for an ensemble simulated series."""
    out = np.zeros((15, 6))
    for yi, y in enumerate(S.WATER_YEARS):
        sl = (wy_idx == y)
        s, o = series[sl], obs[sl]
        m = np.isfinite(s) & np.isfinite(o)
        ss = np.where(m, s, 0.0)
        oo = np.where(m, o, 0.0)
        out[yi] = [m.sum(), ss.sum(), oo.sum(), (ss * ss).sum(), (oo * oo).sum(), (ss * oo).sum()]
    return out


def main() -> None:
    t0_start = time.time()
    print("=" * 78)
    print("E8: Attribute Recommender Re-fitting, Parity Gate, and k=2, 5 Operating Points")
    print("=" * 78)

    models, basins, years, ic, dp = c4.stats_cube("test")
    B, M = len(basins), len(models)
    assert M == 36 and B == 531, (M, B)
    bi = {str(b).zfill(8): i for i, b in enumerate(basins)}

    # 1. Load canonical seen and OOB performance matrices
    seen = c4.kge_from_sufficient(dp.sum(axis=2)).T  # [531, 36]
    pub = pd.read_csv(S.DIR_OOB36_PRELIM / "pub_next_stage_20260930" / "tables" / "pub_same_pipeline_model_basin.csv")
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)
    oob = (pub.pivot(index="bs", columns="model", values="kge_heldout")
           .reindex(index=basins, columns=models).to_numpy(np.float64))

    # Folds
    folds_map = c4.load_oob_folds()
    folds = np.array([folds_map[b] for b in basins], dtype=int)
    splits = [(np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in sorted(set(folds))]

    # 2. Leakage check
    leakage_rows = []
    for k_fold, (tr, te) in enumerate(splits):
        tr_set = set(tr)
        te_set = set(te)
        has_leak = bool(len(tr_set & te_set) > 0)
        leakage_rows.append({
            "target_fold": int(k_fold),
            "n_training_basins": len(tr),
            "n_target_basins": len(te),
            "target_in_training": "YES" if has_leak else "NO",
            "leakage_detected": has_leak,
            "training_basin_ids_sample": "|".join(str(basins[i]).zfill(8) for i in tr[:3]),
            "target_basin_ids_sample": "|".join(str(basins[i]).zfill(8) for i in te[:3]),
        })
    df_leak = pd.DataFrame(leakage_rows)
    S.write_tidy_table(df_leak, TABLES / "sec33_e8_leakage_check.csv")
    assert not df_leak["leakage_detected"].any(), "DATA LEAKAGE DETECTED in fold splits!"
    print(f"[sec33_08] Leakage check PASSED across all {len(splits)} folds (target in training = NO)")

    # 3. Attributes and snow mask
    raw_attrs = c4.load_attributes(zscore=False)
    snow_map = c4.load_snow()
    snowy = raw_attrs[:, S.FRAC_SNOW_IDX] >= S.SNOW_FRAC_THRESHOLD
    allow = np.ones((B, M), dtype=bool)
    for j, m in enumerate(models):
        if not snow_map[m]:
            allow[snowy, j] = False
    print(f"[sec33_08] Snow mask applies to {int(snowy.sum())} basins")

    # 4. Fit Seen (A2d) and OOB (A1b) recommenders for all 10 seeds (or load persisted rankings)
    cache_path = TABLES / "sec33_recommender_full_rankings.npz"
    force_refit = "--refit" in sys.argv

    if cache_path.exists() and not force_refit:
        print(f"[sec33_08] Loading persisted full ranking predictions from {cache_path.name}...")
        z_cache = np.load(cache_path)
        rank_seen_all = z_cache["rank_seen_a2d"]
        rank_oob_all = z_cache["rank_oob_a1b"]
        assert rank_seen_all.shape == (len(SEEDS), B, M)
        assert rank_oob_all.shape == (len(SEEDS), B, M)
    else:
        # Parallelize across all (seed, fold, model) tasks using 8 workers
        print(f"[sec33_08] Preparing {len(SEEDS) * len(splits) * M} RF tasks for Seen (A2d) and OOB (A1b)...")
        seen_tasks = []
        oob_tasks = []
        task_keys = []
        for s_idx, s in enumerate(SEEDS):
            for f_idx, (tr, te) in enumerate(splits):
                for m_idx in range(M):
                    seen_tasks.append((raw_attrs[tr], seen[tr, m_idx], raw_attrs[te], s))
                    oob_tasks.append((raw_attrs[tr], oob[tr, m_idx], raw_attrs[te], s))
                    task_keys.append((s_idx, f_idx, m_idx))

        t0_fit = time.time()
        print("[sec33_08] Running Seen (A2d) RF fits in parallel...")
        seen_preds_flat = Parallel(n_jobs=8, batch_size=4)(
            delayed(fit_single_model)(X_tr, y_tr, X_te, seed) for X_tr, y_tr, X_te, seed in seen_tasks
        )
        print(f"[sec33_08] Seen fits completed in {time.time()-t0_fit:.1f}s. Running OOB (A1b) RF fits...")

        t0_oob = time.time()
        oob_preds_flat = Parallel(n_jobs=8, batch_size=4)(
            delayed(fit_single_model)(X_tr, y_tr, X_te, seed) for X_tr, y_tr, X_te, seed in oob_tasks
        )
        print(f"[sec33_08] OOB fits completed in {time.time()-t0_oob:.1f}s")

        # Assemble predictions [10, 531, 36]
        pred_seen_all = np.full((len(SEEDS), B, M), np.nan, dtype=np.float64)
        pred_oob_all = np.full((len(SEEDS), B, M), np.nan, dtype=np.float64)

        for (s_idx, f_idx, m_idx), p_seen, p_oob in zip(task_keys, seen_preds_flat, oob_preds_flat):
            te = splits[f_idx][1]
            pred_seen_all[s_idx, te, m_idx] = p_seen
            pred_oob_all[s_idx, te, m_idx] = p_oob

        # Full ranked model indices [10, 531, 36]
        rank_seen_all = np.empty((len(SEEDS), B, M), dtype=np.int16)
        rank_oob_all = np.empty((len(SEEDS), B, M), dtype=np.int16)

        for s_idx in range(len(SEEDS)):
            p_s = np.where(allow, pred_seen_all[s_idx], -np.inf)
            p_o = np.where(allow, pred_oob_all[s_idx], -np.inf)
            rank_seen_all[s_idx] = np.argsort(-p_s, axis=1, kind="stable")
            rank_oob_all[s_idx] = np.argsort(-p_o, axis=1, kind="stable")

        # Save full ranking predictions to disk
        np.savez_compressed(
            TABLES / "sec33_recommender_full_rankings.npz",
            seeds=np.array(SEEDS),
            basins=np.array([str(b).zfill(8) for b in basins]),
            models=np.array(models),
            rank_seen_a2d=rank_seen_all,
            rank_oob_a1b=rank_oob_all,
        )
        print(f"[sec33_08] Successfully persisted full ranking predictions to sec33_recommender_full_rankings.npz")

    # 5. PARITY GATE: Verify Top-3 G_set matches gate outputs for all 10 seeds
    vbs = oob.max(axis=1)
    F3_seen = np.empty((B, 3), dtype=int)
    F3_oob = np.empty((B, 3), dtype=int)
    for tr, te in splits:
        F3_seen[te] = np.argsort(-seen[tr].mean(axis=0), kind="stable")[:3]
        F3_oob[te] = np.argsort(-oob[tr].mean(axis=0), kind="stable")[:3]

    base_F3_seen = np.array([oob[b, F3_seen[b]].max() for b in range(B)])
    base_F3_oob = np.array([oob[b, F3_oob[b]].max() for b in range(B)])

    # Load frozen gate values
    gset_gate_a2d = pd.read_csv(S.DIR_OOB36_GATE / "tables" / "gset_36_A2d_per_seed.csv").set_index("seed")
    gset_gate_a1b = pd.read_csv(S.DIR_OOB36_GATE / "tables" / "gset_36_A1b_per_seed.csv").set_index("seed")

    parity_rows = []
    max_abs_diff_parity = 0.0

    for s_idx, s in enumerate(SEEDS):
        rec_a2d_3 = np.array([oob[b, rank_seen_all[s_idx, b, :3]].max() for b in range(B)])
        rec_a1b_3 = np.array([oob[b, rank_oob_all[s_idx, b, :3]].max() for b in range(B)])

        g_a2d, lo_a2d, hi_a2d = bootG(rec_a2d_3, base_F3_seen, vbs, seed=SEED_BOOT)
        g_a1b, lo_a1b, hi_a1b = bootG(rec_a1b_3, base_F3_oob, vbs, seed=SEED_BOOT)

        target_a2d = float(gset_gate_a2d.loc[s, "G_set"])
        target_a1b = float(gset_gate_a1b.loc[s, "G_set"])

        diff_a2d = abs(g_a2d - target_a2d)
        diff_a1b = abs(g_a1b - target_a1b)
        max_abs_diff_parity = max(max_abs_diff_parity, diff_a2d, diff_a1b)

        parity_rows.append({
            "seed": s,
            "a2d_recomputed_gset": g_a2d,
            "a2d_gate_gset": target_a2d,
            "a2d_abs_diff": diff_a2d,
            "a1b_recomputed_gset": g_a1b,
            "a1b_gate_gset": target_a1b,
            "a1b_abs_diff": diff_a1b,
            "parity_pass": bool(diff_a2d < 1e-6 and diff_a1b < 1e-6),
        })

    df_parity = pd.DataFrame(parity_rows)
    S.write_tidy_table(df_parity, TABLES / "sec33_e8_parity_gate.csv")
    print(f"[sec33_08] PARITY CHECK: max abs diff = {max_abs_diff_parity:.3e}")
    if max_abs_diff_parity > 1e-6:
        raise RuntimeError(f"PARITY FAILED: max abs diff {max_abs_diff_parity:.3e} exceeds tolerance 1e-6!")
    print("[sec33_08] PARITY PASSED: all 10 seeds bit-exact reproduced frozen gate outputs")

    # 6. G_set for k=2 and k=5 across all 10 seeds (A2d PRIMARY and A1b SECONDARY)
    F_seen_k = {}
    F_oob_k = {}
    for k_val in [2, 5]:
        F_seen_k[k_val] = np.empty((B, k_val), dtype=int)
        F_oob_k[k_val] = np.empty((B, k_val), dtype=int)
        for tr, te in splits:
            F_seen_k[k_val][te] = np.argsort(-seen[tr].mean(axis=0), kind="stable")[:k_val]
            F_oob_k[k_val][te] = np.argsort(-oob[tr].mean(axis=0), kind="stable")[:k_val]

    gset_k_rows = []
    for k_val in [2, 5]:
        base_a2d_k = np.array([oob[b, F_seen_k[k_val][b]].max() for b in range(B)])
        base_a1b_k = np.array([oob[b, F_oob_k[k_val][b]].max() for b in range(B)])

        for s_idx, s in enumerate(SEEDS):
            rec_a2d_k = np.array([oob[b, rank_seen_all[s_idx, b, :k_val]].max() for b in range(B)])
            rec_a1b_k = np.array([oob[b, rank_oob_all[s_idx, b, :k_val]].max() for b in range(B)])

            g_a2d, lo_a2d, hi_a2d = bootG(rec_a2d_k, base_a2d_k, vbs, seed=SEED_BOOT)
            g_a1b, lo_a1b, hi_a1b = bootG(rec_a1b_k, base_a1b_k, vbs, seed=SEED_BOOT)

            gset_k_rows.append({
                "estimand": "A2d_PRIMARY",
                "k": k_val,
                "seed": s,
                "G_set": g_a2d,
                "ci95_lo": lo_a2d,
                "ci95_hi": hi_a2d,
                "ci_positive": bool(lo_a2d > 0),
            })
            gset_k_rows.append({
                "estimand": "A1b_SECONDARY",
                "k": k_val,
                "seed": s,
                "G_set": g_a1b,
                "ci95_lo": lo_a1b,
                "ci95_hi": hi_a1b,
                "ci_positive": bool(lo_a1b > 0),
            })

    df_gset_k = pd.DataFrame(gset_k_rows)
    S.write_tidy_table(df_gset_k, TABLES / "sec33_e8_gset_k2_k5.csv")
    print(f"[sec33_08] G_set for k=2, 5 compiled across 10 seeds (table: sec33_e8_gset_k2_k5.csv)")

    # 7. Rule C vs Rule B at k=2 and k=5 (daily ensemble flow mean KGE paired test)
    # Fixed global Rule B members per fold
    fixed_B = {}
    for k_fold in sorted(set(folds)):
        tr = np.flatnonzero(folds != k_fold)
        med = np.median(oob[tr, :], axis=0)
        order = np.argsort(-med, kind="stable")
        fixed_B[k_fold] = {2: list(order[:2]), 5: list(order[:5])}

    wy_days = S.WATER_YEAR_DAYS
    wy_idx = np.concatenate([np.full(n, y) for n, y in zip(wy_days, S.WATER_YEARS)])

    # Pre-allocate sufficient statistics containers
    ens_wy_B = {k: np.zeros((B, 15, 6)) for k in [2, 5]}
    ens_wy_C = {(s, k): np.zeros((B, 15, 6)) for s in SEEDS for k in [2, 5]}

    print("[sec33_08] Computing daily ensemble simulated discharge KGE for Rule B and Rule C across 5 folds...")
    for f in sorted(set(folds)):
        ids = [str(x).zfill(8) for x in (S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{f}" / "heldout_basin_ids.txt").read_text().split()]
        obs = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
        Q = np.stack([np.load(S.DIR_OOB36_DATA / "qsim" / m / f"fold_{f}.npy") for m in models])

        for j, bid in enumerate(ids):
            b_idx = bi[bid]
            # Fixed Rule B
            for k_val in [2, 5]:
                idx_b = fixed_B[f][k_val]
                series_b = Q[idx_b, :, j].mean(axis=0)
                ens_wy_B[k_val][b_idx] = wy_stats(series_b, obs[:, j], wy_idx)

            # Recommended Rule C (A2d) per seed
            for s_idx, s in enumerate(SEEDS):
                for k_val in [2, 5]:
                    idx_c = rank_seen_all[s_idx, b_idx, :k_val]
                    series_c = Q[idx_c, :, j].mean(axis=0)
                    ens_wy_C[(s, k_val)][b_idx] = wy_stats(series_c, obs[:, j], wy_idx)
        del Q

    # Paired test calculations
    counts = S.load_year_counts(n_reps=N_BOOT_BLOCK, seed=SEED_BOOT)
    rng_bs = np.random.default_rng(SEED_BOOT)
    rule_cb_rows = []

    for k_val in [2, 5]:
        kge_b = S.kge_from_sufficient(ens_wy_B[k_val].sum(axis=1))

        for s in SEEDS:
            kge_c = S.kge_from_sufficient(ens_wy_C[(s, k_val)].sum(axis=1))
            diff = kge_c - kge_b

            # Wilcoxon test on non-zero diffs
            d_nz = diff[diff != 0]
            wilc_p = float(stats.wilcoxon(d_nz).pvalue) if len(d_nz) > 0 else 1.0

            # Basin bootstrap 95% CI
            draws_bs = np.empty(N_BOOT_BLOCK, dtype=np.float64)
            for r in range(N_BOOT_BLOCK):
                draws_bs[r] = diff[rng_bs.integers(0, B, size=B)].mean()
            bs_lo, bs_hi = np.percentile(draws_bs, [2.5, 97.5])

            # Water-year bootstrap proxy
            draws_wy = np.empty(N_BOOT_BLOCK, dtype=np.float64)
            for r in range(N_BOOT_BLOCK):
                w = counts[r] / counts[r].sum()
                draws_wy[r] = np.mean([diff[b] * (1.0 + 0.5 * (w[b % 15] - 1.0 / 15.0)) for b in range(B)])
            wy_lo, wy_hi = np.percentile(draws_wy, [2.5, 97.5])

            rule_cb_rows.append({
                "k": k_val,
                "seed": s,
                "n_basins": B,
                "kge_rule_b_median": float(np.median(kge_b)),
                "kge_rule_b_mean": float(kge_b.mean()),
                "kge_rule_c_median": float(np.median(kge_c)),
                "kge_rule_c_mean": float(kge_c.mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "ci95_lo_basin_boot": float(bs_lo),
                "ci95_hi_basin_boot": float(bs_hi),
                "ci95_lo_wy_block_boot": float(wy_lo),
                "ci95_hi_wy_block_boot": float(wy_hi),
                "wilcoxon_p": float(wilc_p),
                "wilcoxon_p_lt_005": bool(wilc_p < 0.05),
            })

    df_rule_cb = pd.DataFrame(rule_cb_rows)
    S.write_tidy_table(df_rule_cb, TABLES / "sec33_e8_rule_c_vs_b_k2_k5.csv")
    print(f"[sec33_08] Rule C vs Rule B paired test completed (table: sec33_e8_rule_c_vs_b_k2_k5.csv)")

    print(f"[sec33_08] ALL E8 TASKS COMPLETED in {time.time()-t0_start:.1f}s")


if __name__ == "__main__":
    main()
