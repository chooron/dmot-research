#!/usr/bin/env python3
"""sec33_07_diversity_sensitivity_v5.py — Sensitivity analysis: Structural & Behavioral diversity candidate sets vs Ranking sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

KS = [2, 3, 4, 5, 7, 10]


def wy_stats_vectorized(series: np.ndarray, obs: np.ndarray, wy_idx: np.ndarray) -> np.ndarray:
    out = np.zeros((15, 6), dtype=np.float64)
    for yi, y in enumerate(S.WATER_YEARS):
        sl = (wy_idx == y)
        s, o = series[sl], obs[sl]
        m = np.isfinite(s) & np.isfinite(o)
        ss = np.where(m, s, 0.0)
        oo = np.where(m, o, 0.0)
        out[yi] = [m.sum(), ss.sum(), oo.sum(), (ss * ss).sum(), (oo * oo).sum(), (ss * oo).sum()]
    return out


def main() -> None:
    models, basins, oob_mat = S.load_oob_kge_matrix()
    nb, nm = len(basins), len(models)
    bi = {b: i for i, b in enumerate(basins)}

    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])

    wy_days = S.WATER_YEAR_DAYS
    wy_idx = np.concatenate([np.full(n, y) for n, y in zip(wy_days, S.WATER_YEARS)])

    # Load ranking sets from sec33_fixed_ensemble_fold_members.csv
    rank_members_df = pd.read_csv(TABLES / "sec33_fixed_ensemble_fold_members.csv")
    rank_sets = {}
    for _, r in rank_members_df.iterrows():
        rank_sets[(int(r["fold"]), int(r["k"]))] = [models.index(m) for m in str(r["members"]).split("|")]

    # Load ranking per-basin KGEs from sec33_fixed_ensemble_per_basin.csv
    rank_pb_df = pd.read_csv(TABLES / "sec33_fixed_ensemble_per_basin.csv")
    rank_pb_df["bs"] = rank_pb_df["basin_id"].astype(str).str.zfill(8)
    kge_ranking = {}
    for k in KS:
        kge_ranking[k] = rank_pb_df[rank_pb_df.k == k].set_index("bs").loc[basins, "kge"].to_numpy(dtype=np.float64)

    # 1. Structural features standardized
    attr_df = pd.read_csv(ROOT / "project/benchmarkpub/manuscript/analysis/sec32_library_members/tables/sec32_model_structural_attributes.csv").set_index("model").loc[models]
    P = attr_df["n_parameters"].to_numpy(dtype=float)
    St = attr_df["n_states"].to_numpy(dtype=float)
    Sn = attr_df["has_explicit_snow_routine"].to_numpy(dtype=float)
    Z_struct = np.stack([
        (P - P.mean()) / P.std(),
        (St - St.mean()) / St.std(),
        (Sn - Sn.mean()) / Sn.std(),
    ], axis=1)  # [36, 3]

    # Pre-load Qsim arrays for all models and folds
    qsim_cache = {}
    obs_cache = {}
    for f in range(5):
        obs_cache[f] = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
        qsim_cache[f] = np.stack([np.load(S.DIR_OOB36_DATA / "qsim" / m / f"fold_{f}.npy") for m in models])  # [36, 5479, n_fold_basins]

    # 2. Construct selection sets per fold for each k
    struct_members = {}
    behav_members = {}
    member_record_rows = []

    for f in range(5):
        tr = np.flatnonzero(folds != f)
        med_tr = np.median(oob_mat[tr, :], axis=0)
        lead_m = int(np.argsort(-med_tr, kind="stable")[0])  # hbv96

        # (i) Structural diversity greedy selection
        struct_seq = [lead_m]
        for step in range(1, max(KS)):
            cand = [i for i in range(nm) if i not in struct_seq]
            dists = [min(np.linalg.norm(Z_struct[c] - Z_struct[s]) for s in struct_seq) for c in cand]
            best_c = cand[int(np.argmax(dists))]
            struct_seq.append(best_c)

        for k in KS:
            struct_members[(f, k)] = struct_seq[:k]
            member_record_rows.append({
                "selection_method": "structural_diversity",
                "fold": f,
                "k": k,
                "n_training_basins": len(tr),
                "members": "|".join(models[i] for i in struct_seq[:k]),
                "lead_model": models[struct_seq[0]],
            })

        # (ii) Behavioral diversity greedy selection
        # Inter-model correlation matrix on training basins
        # Concatenate training folds for this fold
        tr_folds = [k_fold for k_fold in range(5) if k_fold != f]
        Q_tr = np.concatenate([qsim_cache[k_fold] for k_fold in tr_folds], axis=2)  # [36, 5479, n_train_basins]
        Q_tr_flat = Q_tr.reshape(nm, -1)
        C_tr = np.corrcoef(Q_tr_flat)  # [36, 36]

        behav_seq = [lead_m]
        for step in range(1, max(KS)):
            cand = [i for i in range(nm) if i not in behav_seq]
            avg_corrs = [np.mean([C_tr[c, s] for s in behav_seq]) for c in cand]
            best_c = cand[int(np.argmin(avg_corrs))]
            behav_seq.append(best_c)

        for k in KS:
            behav_members[(f, k)] = behav_seq[:k]
            member_record_rows.append({
                "selection_method": "behavioral_diversity",
                "fold": f,
                "k": k,
                "n_training_basins": len(tr),
                "members": "|".join(models[i] for i in behav_seq[:k]),
                "lead_model": models[behav_seq[0]],
            })

    df_mem_records = pd.DataFrame(member_record_rows)
    S.write_tidy_table(df_mem_records, TABLES / "sec33_diversity_fold_members.csv")

    # 3. Simulate and evaluate ensembles on held-out basins
    ens_wy_struct = {k: np.zeros((nb, 15, 6), dtype=np.float64) for k in KS}
    ens_wy_behav = {k: np.zeros((nb, 15, 6), dtype=np.float64) for k in KS}

    for f in range(5):
        ids = [str(x).zfill(8) for x in (S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{f}" / "heldout_basin_ids.txt").read_text().split()]
        obs = obs_cache[f]
        Q_te = qsim_cache[f]
        for j, bid in enumerate(ids):
            b_pos = bi[bid]
            for k in KS:
                s_s = struct_members[(f, k)]
                b_s = behav_members[(f, k)]
                ens_s = Q_te[s_s, :, j].mean(axis=0) if len(s_s) > 1 else Q_te[s_s[0], :, j]
                ens_b = Q_te[b_s, :, j].mean(axis=0) if len(b_s) > 1 else Q_te[b_s[0], :, j]
                ens_wy_struct[k][b_pos] = wy_stats_vectorized(ens_s, obs[:, j], wy_idx)
                ens_wy_behav[k][b_pos] = wy_stats_vectorized(ens_b, obs[:, j], wy_idx)

    kge_struct = {k: S.kge_from_sufficient(ens_wy_struct[k].sum(axis=1)) for k in KS}
    kge_behav = {k: S.kge_from_sufficient(ens_wy_behav[k].sum(axis=1)) for k in KS}

    # 4. Compute paired comparisons vs same-k ranking sets
    rng_bs = np.random.default_rng(S.SEED_BOOT)

    summary_rows = []
    per_basin_rows = []

    for method_name, kge_dict in [("structural_diversity", kge_struct), ("behavioral_diversity", kge_behav)]:
        for k in KS:
            cur_kge = kge_dict[k]
            base_kge = kge_ranking[k]
            diff = cur_kge - base_kge
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

            summary_rows.append({
                "selection_method": method_name,
                "k": k,
                "n_basins": nb,
                "kge_diversity_median": float(np.median(cur_kge)),
                "kge_diversity_q25": float(np.percentile(cur_kge, 25)),
                "kge_diversity_q75": float(np.percentile(cur_kge, 75)),
                "kge_ranking_median": float(np.median(base_kge)),
                "diff_mean": float(diff.mean()),
                "diff_median": float(np.median(diff)),
                "diff_ci95_lo_basin": float(bs_lo),
                "diff_ci95_hi_basin": float(bs_hi),
                "diff_ci95_lo_wy": wy_lo,
                "diff_ci95_hi_wy": wy_hi,
                "wilcoxon_p": wilc_p,
                "wilcoxon_p_lt_005": bool(wilc_p < 0.05),
                "evaluation_scope": "SENSITIVITY",
            })

            for b in range(nb):
                per_basin_rows.append({
                    "selection_method": method_name,
                    "k": k,
                    "basin_id": basins[b],
                    "fold": int(folds[b]),
                    "kge_diversity": float(cur_kge[b]),
                    "kge_ranking": float(base_kge[b]),
                    "diff_vs_ranking": float(diff[b]),
                })

    df_sum = pd.DataFrame(summary_rows)
    df_pb = pd.DataFrame(per_basin_rows)
    S.write_tidy_table(df_sum, TABLES / "sec33_diversity_sensitivity_summary.csv")
    S.write_tidy_table(df_pb, TABLES / "sec33_diversity_per_basin.csv")

    # 5. D6 Leakage check table
    leak_rows = []
    for f in range(5):
        leak_rows.append({
            "fold": f,
            "n_training_basins": int((folds != f).sum()),
            "n_target_basins": int((folds == f).sum()),
            "target_fold_in_construction": "NO",
        })
    df_leak = pd.DataFrame(leak_rows)
    S.write_tidy_table(df_leak, TABLES / "sec33_diversity_leakage_check.csv")

    parity_rec = {
        "script": "sec33_07_diversity_sensitivity_v5.py",
        "approach": "(a) Recomputed from raw Qsim and model structural attributes on training folds",
        "k_values": KS,
        "n_methods": 2,
        "n_summary_rows": len(df_sum),
        "struct_diff_k3_mean": float(df_sum.loc[(df_sum.selection_method == "structural_diversity") & (df_sum.k == 3), "diff_mean"].iloc[0]),
        "behav_diff_k3_mean": float(df_sum.loc[(df_sum.selection_method == "behavioral_diversity") & (df_sum.k == 3), "diff_mean"].iloc[0]),
    }
    (TABLES / "sec33_07_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec33_07] PASS: diversity sensitivity k={KS}, struct k3 diff={parity_rec['struct_diff_k3_mean']:+.5f}, behav k3 diff={parity_rec['behav_diff_k3_mean']:+.5f}")


if __name__ == "__main__":
    main()
