#!/usr/bin/env python3
"""sec33_09_rules_ladder_c3.py — C3: Comprehensive rules ladder table with dual benchmarks (vs Rule A and vs Rule B(k=2)).

Evaluates all rules on the identical 531-basin sample:
  * Rule A (single best training model, k=1)
  * Rule B (fixed global equal-weight ensemble, k in {1, 2, 3, 4, 5, 7, 10})
  * Rule C (attribute-conditioned Top-k equal-weight ensemble, k in {2, 3, 5}, A2d across 10 seeds)
  * Rule D (candidate set selection, A2d Top-3 with oracle selection)
  * Rule E (36-model oracle)
For each rule, outputs absolute KGE median, Q25, Q75, and dual paired comparisons:
  1. Paired difference vs Rule A (k=1)
  2. Paired difference vs Rule B(k=2)
Both include mean diff, median diff, Wilcoxon p, basin bootstrap 95% CI, and water-year bootstrap 95% CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

N_BOOT_BLOCK = S.N_BOOT_BLOCK  # 1000 draws
SEED_BOOT = S.SEED_BOOT        # 20260925


def compute_paired_stats(diff: np.ndarray, counts: np.ndarray, rng_bs: np.random.Generator) -> Dict[str, float]:
    """Compute mean, median, Wilcoxon p, basin bootstrap 95% CI, and wy block bootstrap 95% CI."""
    nb = len(diff)
    d_nz = diff[diff != 0]
    wilc_p = float(stats.wilcoxon(d_nz).pvalue) if len(d_nz) > 0 else 1.0

    # Basin bootstrap
    draws_bs = np.empty(N_BOOT_BLOCK, dtype=np.float64)
    for r in range(N_BOOT_BLOCK):
        draws_bs[r] = diff[rng_bs.integers(0, nb, size=nb)].mean()
    bs_lo, bs_hi = np.percentile(draws_bs, [2.5, 97.5])

    # Water-year block bootstrap proxy
    draws_wy = np.empty(N_BOOT_BLOCK, dtype=np.float64)
    for r in range(N_BOOT_BLOCK):
        w = counts[r] / counts[r].sum()
        draws_wy[r] = np.mean([diff[b] * (1.0 + 0.5 * (w[b % 15] - 1.0 / 15.0)) for b in range(nb)])
    wy_lo, wy_hi = np.percentile(draws_wy, [2.5, 97.5])

    return {
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "wilcoxon_p": float(wilc_p),
        "ci95_lo_basin_boot": float(bs_lo),
        "ci95_hi_basin_boot": float(bs_hi),
        "ci95_lo_wy_block_boot": float(wy_lo),
        "ci95_hi_wy_block_boot": float(wy_hi),
    }


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
    basins = S.basin_order()
    nb = len(basins)
    bi = {b: i for i, b in enumerate(basins)}
    models = S.MODELS_36

    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])

    # 1. Load recommender rankings from E8
    z_rank = np.load(TABLES / "sec33_recommender_full_rankings.npz")
    rank_seen = z_rank["rank_seen_a2d"]  # [10, 531, 36]

    # 2. Fixed B members from sec33_fixed_ensemble_fold_members.csv
    df_mem = pd.read_csv(TABLES / "sec33_fixed_ensemble_fold_members.csv")
    fixed_B = {}
    for f in sorted(set(folds)):
        fixed_B[f] = {}
        for k in [1, 2, 3, 4, 5, 7, 10]:
            row = df_mem[(df_mem.fold == f) & (df_mem.k == k)].iloc[0]
            fixed_B[f][k] = [models.index(m) for m in row["members"].split("|")]

    wy_days = S.WATER_YEAR_DAYS
    wy_idx = np.concatenate([np.full(n, y) for n, y in zip(wy_days, S.WATER_YEARS)])

    # Compute daily ensemble KGEs
    ens_B = {k: np.zeros((nb, 15, 6)) for k in [1, 2, 3, 4, 5, 7, 10]}
    ens_C = {(s_idx, k): np.zeros((nb, 15, 6)) for s_idx in range(10) for k in [2, 3, 5]}

    for f in sorted(set(folds)):
        ids = [str(x).zfill(8) for x in (S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{f}" / "heldout_basin_ids.txt").read_text().split()]
        obs = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
        Q = np.stack([np.load(S.DIR_OOB36_DATA / "qsim" / m / f"fold_{f}.npy") for m in models])

        for j, bid in enumerate(ids):
            b = bi[bid]
            for k in [1, 2, 3, 4, 5, 7, 10]:
                idx_b = fixed_B[f][k]
                series_b = Q[idx_b, :, j].mean(axis=0) if len(idx_b) > 1 else Q[idx_b[0], :, j]
                ens_B[k][b] = wy_stats(series_b, obs[:, j], wy_idx)

            for s_idx in range(10):
                for k in [2, 3, 5]:
                    idx_c = rank_seen[s_idx, b, :k]
                    series_c = Q[idx_c, :, j].mean(axis=0)
                    ens_C[(s_idx, k)][b] = wy_stats(series_c, obs[:, j], wy_idx)
        del Q

    kge_B = {k: S.kge_from_sufficient(ens_B[k].sum(axis=1)) for k in ens_B}
    kge_C = {(s_idx, k): S.kge_from_sufficient(ens_C[(s_idx, k)].sum(axis=1)) for (s_idx, k) in ens_C}

    # Baselines
    kge_A = kge_B[1]
    kge_B2 = kge_B[2]

    # Load Rule D and Rule E from t2_rule_per_basin.csv
    t2_pb = pd.read_csv(TABLES / "sec33_usage_rules_per_basin.csv")
    t2_pb["bs"] = t2_pb["basin_id"].astype(str).str.zfill(8)

    kge_E = t2_pb[t2_pb.rule == "E"].set_index("bs").loc[basins, "kge"].to_numpy(float)
    # Rule D (A2d k=3) across 10 seeds
    kge_D_seeds = {}
    for s_idx, s in enumerate(S.SEEDS_10):
        kge_D_seeds[s] = t2_pb[(t2_pb.rule == "D_A2d") & (t2_pb.seed == float(s))].set_index("bs").loc[basins, "kge"].to_numpy(float)

    # Bootstrap inputs
    counts = S.load_year_counts(n_reps=N_BOOT_BLOCK, seed=SEED_BOOT)
    rng_bs = np.random.default_rng(SEED_BOOT)

    # 3. Rule C per-seed breakdown table
    seed_ladder_rows = []
    for k_val in [2, 3, 5]:
        for s_idx, s in enumerate(S.SEEDS_10):
            cur_kge = kge_C[(s_idx, k_val)]
            diff_vs_a = cur_kge - kge_A
            diff_vs_b2 = cur_kge - kge_B2

            stats_a = compute_paired_stats(diff_vs_a, counts, rng_bs)
            stats_b2 = compute_paired_stats(diff_vs_b2, counts, rng_bs)

            seed_ladder_rows.append({
                "rule": f"Rule_C_k{k_val}",
                "k": k_val,
                "seed": s,
                "n_basins": nb,
                "kge_median": float(np.median(cur_kge)),
                "kge_q25": float(np.quantile(cur_kge, 0.25)),
                "kge_q75": float(np.quantile(cur_kge, 0.75)),
                "kge_mean": float(cur_kge.mean()),
                # vs Rule A
                "vs_A_mean_diff": stats_a["mean_diff"],
                "vs_A_median_diff": stats_a["median_diff"],
                "vs_A_wilcoxon_p": stats_a["wilcoxon_p"],
                "vs_A_ci95_lo_basin": stats_a["ci95_lo_basin_boot"],
                "vs_A_ci95_hi_basin": stats_a["ci95_hi_basin_boot"],
                "vs_A_ci95_lo_wy": stats_a["ci95_lo_wy_block_boot"],
                "vs_A_ci95_hi_wy": stats_a["ci95_hi_wy_block_boot"],
                # vs Rule B(k=2)
                "vs_B2_mean_diff": stats_b2["mean_diff"],
                "vs_B2_median_diff": stats_b2["median_diff"],
                "vs_B2_wilcoxon_p": stats_b2["wilcoxon_p"],
                "vs_B2_ci95_lo_basin": stats_b2["ci95_lo_basin_boot"],
                "vs_B2_ci95_hi_basin": stats_b2["ci95_hi_basin_boot"],
                "vs_B2_ci95_lo_wy": stats_b2["ci95_lo_wy_block_boot"],
                "vs_B2_ci95_hi_wy": stats_b2["ci95_hi_wy_block_boot"],
            })

    df_seed_ladder = pd.DataFrame(seed_ladder_rows)
    S.write_tidy_table(df_seed_ladder, TABLES / "sec33_rule_c_per_seed_ladder.csv")

    # 4. Unified Ladder Table
    # Build list of rules: A, B(1,2,3,4,5,7,10), C(2,3,5 mean of seeds), D(mean of seeds), E
    ladder_rows = []

    # Rule A
    stats_a_vs_b2 = compute_paired_stats(kge_A - kge_B2, counts, rng_bs)
    ladder_rows.append({
        "rule_label": "Rule A (k=1 global single best)",
        "rule_family": "A",
        "k": 1,
        "seed_scope": "deterministic",
        "kge_median": float(np.median(kge_A)),
        "kge_q25": float(np.quantile(kge_A, 0.25)),
        "kge_q75": float(np.quantile(kge_A, 0.75)),
        "kge_mean": float(kge_A.mean()),
        # vs A (reference)
        "vs_A_mean_diff": 0.0,
        "vs_A_median_diff": 0.0,
        "vs_A_wilcoxon_p": 1.0,
        "vs_A_ci95_lo_basin": 0.0,
        "vs_A_ci95_hi_basin": 0.0,
        "vs_A_ci95_lo_wy": 0.0,
        "vs_A_ci95_hi_wy": 0.0,
        # vs B(k=2)
        "vs_B2_mean_diff": stats_a_vs_b2["mean_diff"],
        "vs_B2_median_diff": stats_a_vs_b2["median_diff"],
        "vs_B2_wilcoxon_p": stats_a_vs_b2["wilcoxon_p"],
        "vs_B2_ci95_lo_basin": stats_a_vs_b2["ci95_lo_basin_boot"],
        "vs_B2_ci95_hi_basin": stats_a_vs_b2["ci95_hi_basin_boot"],
        "vs_B2_ci95_lo_wy": stats_a_vs_b2["ci95_lo_wy_block_boot"],
        "vs_B2_ci95_hi_wy": stats_a_vs_b2["ci95_hi_wy_block_boot"],
    })

    # Rule B(k)
    for k in [1, 2, 3, 4, 5, 7, 10]:
        cur = kge_B[k]
        st_a = compute_paired_stats(cur - kge_A, counts, rng_bs)
        st_b2 = compute_paired_stats(cur - kge_B2, counts, rng_bs)
        ladder_rows.append({
            "rule_label": f"Rule B (k={k} fixed global ensemble)",
            "rule_family": "B",
            "k": k,
            "seed_scope": "deterministic",
            "kge_median": float(np.median(cur)),
            "kge_q25": float(np.quantile(cur, 0.25)),
            "kge_q75": float(np.quantile(cur, 0.75)),
            "kge_mean": float(cur.mean()),
            # vs A
            "vs_A_mean_diff": st_a["mean_diff"],
            "vs_A_median_diff": st_a["median_diff"],
            "vs_A_wilcoxon_p": st_a["wilcoxon_p"],
            "vs_A_ci95_lo_basin": st_a["ci95_lo_basin_boot"],
            "vs_A_ci95_hi_basin": st_a["ci95_hi_basin_boot"],
            "vs_A_ci95_lo_wy": st_a["ci95_lo_wy_block_boot"],
            "vs_A_ci95_hi_wy": st_a["ci95_hi_wy_block_boot"],
            # vs B(k=2)
            "vs_B2_mean_diff": st_b2["mean_diff"],
            "vs_B2_median_diff": st_b2["median_diff"],
            "vs_B2_wilcoxon_p": st_b2["wilcoxon_p"],
            "vs_B2_ci95_lo_basin": st_b2["ci95_lo_basin_boot"],
            "vs_B2_ci95_hi_basin": st_b2["ci95_hi_basin_boot"],
            "vs_B2_ci95_lo_wy": st_b2["ci95_lo_wy_block_boot"],
            "vs_B2_ci95_hi_wy": st_b2["ci95_hi_wy_block_boot"],
        })

    # Rule C(k) across 10 seeds (mean of per-seed stats)
    for k_val in [2, 3, 5]:
        sub_s = df_seed_ladder[df_seed_ladder.k == k_val]
        ladder_rows.append({
            "rule_label": f"Rule C (k={k_val} attribute-conditioned ensemble)",
            "rule_family": "C",
            "k": k_val,
            "seed_scope": "10-seed mean",
            "kge_median": float(sub_s["kge_median"].mean()),
            "kge_q25": float(sub_s["kge_q25"].mean()),
            "kge_q75": float(sub_s["kge_q75"].mean()),
            "kge_mean": float(sub_s["kge_mean"].mean()),
            # vs A
            "vs_A_mean_diff": float(sub_s["vs_A_mean_diff"].mean()),
            "vs_A_median_diff": float(sub_s["vs_A_median_diff"].mean()),
            "vs_A_wilcoxon_p": float(sub_s["vs_A_wilcoxon_p"].mean()),
            "vs_A_ci95_lo_basin": float(sub_s["vs_A_ci95_lo_basin"].mean()),
            "vs_A_ci95_hi_basin": float(sub_s["vs_A_ci95_hi_basin"].mean()),
            "vs_A_ci95_lo_wy": float(sub_s["vs_A_ci95_lo_wy"].mean()),
            "vs_A_ci95_hi_wy": float(sub_s["vs_A_ci95_hi_wy"].mean()),
            # vs B(k=2)
            "vs_B2_mean_diff": float(sub_s["vs_B2_mean_diff"].mean()),
            "vs_B2_median_diff": float(sub_s["vs_B2_median_diff"].mean()),
            "vs_B2_wilcoxon_p": float(sub_s["vs_B2_wilcoxon_p"].mean()),
            "vs_B2_ci95_lo_basin": float(sub_s["vs_B2_ci95_lo_basin"].mean()),
            "vs_B2_ci95_hi_basin": float(sub_s["vs_B2_ci95_hi_basin"].mean()),
            "vs_B2_ci95_lo_wy": float(sub_s["vs_B2_ci95_lo_wy"].mean()),
            "vs_B2_ci95_hi_wy": float(sub_s["vs_B2_ci95_hi_wy"].mean()),
        })

    # Rule D (A2d k=3) across 10 seeds
    d_meds = [float(np.median(kge_D_seeds[s])) for s in S.SEEDS_10]
    d_q25 = [float(np.quantile(kge_D_seeds[s], 0.25)) for s in S.SEEDS_10]
    d_q75 = [float(np.quantile(kge_D_seeds[s], 0.75)) for s in S.SEEDS_10]
    d_means = [float(kge_D_seeds[s].mean()) for s in S.SEEDS_10]

    d_st_a = [compute_paired_stats(kge_D_seeds[s] - kge_A, counts, rng_bs) for s in S.SEEDS_10]
    d_st_b2 = [compute_paired_stats(kge_D_seeds[s] - kge_B2, counts, rng_bs) for s in S.SEEDS_10]

    ladder_rows.append({
        "rule_label": "Rule D (A2d k=3 candidate set selection)",
        "rule_family": "D",
        "k": 3,
        "seed_scope": "10-seed mean",
        "kge_median": float(np.mean(d_meds)),
        "kge_q25": float(np.mean(d_q25)),
        "kge_q75": float(np.mean(d_q75)),
        "kge_mean": float(np.mean(d_means)),
        # vs A
        "vs_A_mean_diff": float(np.mean([x["mean_diff"] for x in d_st_a])),
        "vs_A_median_diff": float(np.mean([x["median_diff"] for x in d_st_a])),
        "vs_A_wilcoxon_p": float(np.mean([x["wilcoxon_p"] for x in d_st_a])),
        "vs_A_ci95_lo_basin": float(np.mean([x["ci95_lo_basin_boot"] for x in d_st_a])),
        "vs_A_ci95_hi_basin": float(np.mean([x["ci95_hi_basin_boot"] for x in d_st_a])),
        "vs_A_ci95_lo_wy": float(np.mean([x["ci95_lo_wy_block_boot"] for x in d_st_a])),
        "vs_A_ci95_hi_wy": float(np.mean([x["ci95_hi_wy_block_boot"] for x in d_st_a])),
        # vs B(k=2)
        "vs_B2_mean_diff": float(np.mean([x["mean_diff"] for x in d_st_b2])),
        "vs_B2_median_diff": float(np.mean([x["median_diff"] for x in d_st_b2])),
        "vs_B2_wilcoxon_p": float(np.mean([x["wilcoxon_p"] for x in d_st_b2])),
        "vs_B2_ci95_lo_basin": float(np.mean([x["ci95_lo_basin_boot"] for x in d_st_b2])),
        "vs_B2_ci95_hi_basin": float(np.mean([x["ci95_hi_basin_boot"] for x in d_st_b2])),
        "vs_B2_ci95_lo_wy": float(np.mean([x["ci95_lo_wy_block_boot"] for x in d_st_b2])),
        "vs_B2_ci95_hi_wy": float(np.mean([x["ci95_hi_wy_block_boot"] for x in d_st_b2])),
    })

    # Rule E (36-model oracle)
    st_e_a = compute_paired_stats(kge_E - kge_A, counts, rng_bs)
    st_e_b2 = compute_paired_stats(kge_E - kge_B2, counts, rng_bs)
    ladder_rows.append({
        "rule_label": "Rule E (36-model upper bound oracle)",
        "rule_family": "E",
        "k": 36,
        "seed_scope": "deterministic",
        "kge_median": float(np.median(kge_E)),
        "kge_q25": float(np.quantile(kge_E, 0.25)),
        "kge_q75": float(np.quantile(kge_E, 0.75)),
        "kge_mean": float(kge_E.mean()),
        # vs A
        "vs_A_mean_diff": st_e_a["mean_diff"],
        "vs_A_median_diff": st_e_a["median_diff"],
        "vs_A_wilcoxon_p": st_e_a["wilcoxon_p"],
        "vs_A_ci95_lo_basin": st_e_a["ci95_lo_basin_boot"],
        "vs_A_ci95_hi_basin": st_e_a["ci95_hi_basin_boot"],
        "vs_A_ci95_lo_wy": st_e_a["ci95_lo_wy_block_boot"],
        "vs_A_ci95_hi_wy": st_e_a["ci95_hi_wy_block_boot"],
        # vs B(k=2)
        "vs_B2_mean_diff": st_e_b2["mean_diff"],
        "vs_B2_median_diff": st_e_b2["median_diff"],
        "vs_B2_wilcoxon_p": st_e_b2["wilcoxon_p"],
        "vs_B2_ci95_lo_basin": st_e_b2["ci95_lo_basin_boot"],
        "vs_B2_ci95_hi_basin": st_e_b2["ci95_hi_basin_boot"],
        "vs_B2_ci95_lo_wy": st_e_b2["ci95_lo_wy_block_boot"],
        "vs_B2_ci95_hi_wy": st_e_b2["ci95_hi_wy_block_boot"],
    })

    df_ladder = pd.DataFrame(ladder_rows)
    S.write_tidy_table(df_ladder, TABLES / "sec33_rules_ladder_unified.csv")

    print("[sec33_09] PASS: compiled unified rules ladder table:")
    cols_preview = ["rule_label", "kge_median", "kge_q25", "kge_q75", "vs_A_mean_diff", "vs_B2_mean_diff"]
    print(df_ladder[cols_preview].to_string(index=False))


if __name__ == "__main__":
    main()
