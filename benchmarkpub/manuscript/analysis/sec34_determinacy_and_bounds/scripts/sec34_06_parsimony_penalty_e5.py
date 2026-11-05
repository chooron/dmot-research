#!/usr/bin/env python3
"""sec34_06_parsimony_penalty_e5.py — E5: Within-equivalence parsimony cost analysis.

Compares OOB performance within each basin's equivalence set between:
(i) Training-fold median ranking leader (highest seen skill)
(ii) Most parsimonious model (minimum parameter count, ties broken by training rank)
Evaluated across OOB-dPL, seen-dPL, and IC conditions.
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


def main() -> None:
    models = S.MODELS_36
    basins = S.canonical_ids()
    basins_str = [str(b).zfill(8) for b in basins]
    nb = len(basins)

    # 1. Outer fold assignments
    fa_df = pd.read_csv(S.DIR_OOB36_DATA / "manifests" / "OOB_FOLD_ASSIGNMENT.csv")
    b2f = {str(b).zfill(8): int(f) for b, f in zip(fa_df.basin_id, fa_df.fold)}
    folds = np.array([b2f[b] for b in basins_str])

    # 2. Structural attributes (parameter counts)
    attr_df = pd.read_csv(ROOT / "project/benchmarkpub/manuscript/analysis/sec32_library_members/tables/sec32_model_structural_attributes.csv").set_index("model")
    n_params = {m: int(attr_df.loc[m, "n_parameters"]) for m in models}

    # 3. Equivalence sets
    eq_df = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")

    # 4. Performance matrices
    models_oob, _, oob_mat = S.load_oob_kge_matrix()  # [531, 36]
    models_seen, _, seen_mat = S.load_canonical_seen_kge_matrix()  # [531, 36]

    # IC matrix from sufficient statistics
    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    mi = {m: i for i, m in enumerate(models)}
    bi = {b: i for i, b in enumerate(basins_str)}
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    cube_ic = np.zeros((len(models), nb, 15, 6), dtype=np.float64)
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

    summary_rows = []
    basin_diff_rows = []

    for cond, (mat, mod_list) in perf_mats.items():
        sub_eq = eq_df[eq_df.condition == cond].set_index("basin_id")

        kge_rank = np.empty(nb)
        kge_pars = np.empty(nb)
        diff_kge = np.empty(nb)
        param_diff = np.empty(nb)

        ties_broken = 0
        coincided = 0

        for f in range(5):
            tr = np.flatnonzero(folds != f)
            te = np.flatnonzero(folds == f)

            med_tr = np.median(mat[tr], axis=0)
            rank_order = {mod_list[i]: rank for rank, i in enumerate(np.argsort(-med_tr))}

            for b_i in te:
                bid = basins_str[b_i]
                eq_mods = str(sub_eq.loc[int(bid), "equivalent_models"]).split("|")

                # (i) Ranking member: top training median rank
                best_rank_mod = min(eq_mods, key=lambda m: rank_order[m])

                # (ii) Parsimonious member: min n_params, break ties by training rank
                min_p = min(n_params[m] for m in eq_mods)
                cands_p = [m for m in eq_mods if n_params[m] == min_p]
                if len(cands_p) > 1:
                    ties_broken += 1
                best_pars_mod = min(cands_p, key=lambda m: rank_order[m])

                if best_rank_mod == best_pars_mod:
                    coincided += 1

                idx_r = mod_list.index(best_rank_mod)
                idx_p = mod_list.index(best_pars_mod)

                kge_r = mat[b_i, idx_r]
                kge_p = mat[b_i, idx_p]

                kge_rank[b_i] = kge_r
                kge_pars[b_i] = kge_p
                # Positive diff indicates performance loss when choosing parsimonious over ranking leader
                d_val = kge_r - kge_p
                p_diff = n_params[best_rank_mod] - n_params[best_pars_mod]

                diff_kge[b_i] = d_val
                param_diff[b_i] = p_diff

                basin_diff_rows.append({
                    "condition": cond,
                    "basin_id": bid,
                    "fold": int(f),
                    "ranking_member": best_rank_mod,
                    "ranking_member_kge": float(kge_r),
                    "ranking_member_nparams": n_params[best_rank_mod],
                    "parsimonious_member": best_pars_mod,
                    "parsimonious_member_kge": float(kge_p),
                    "parsimonious_member_nparams": n_params[best_pars_mod],
                    "diff_kge_rank_minus_pars": float(d_val),
                    "diff_nparams_rank_minus_pars": int(p_diff),
                    "is_identical": bool(best_rank_mod == best_pars_mod),
                    "subtraction_order": "ranking_member_minus_parsimonious_member",
                })

        non_zero = diff_kge[diff_kge != 0]
        w_res = sstats.wilcoxon(non_zero)
        ci_lo, ci_hi = S.paired_bootstrap_ci(diff_kge, n_boot=1000, seed=20260925)

        summary_rows.append({
            "condition": cond,
            "n_basins": nb,
            "n_coincided": coincided,
            "frac_coincided": float(coincided / nb),
            "subtraction_order": "ranking_member_minus_parsimonious_member",
            "n_ties_broken_min_params": ties_broken,
            "mean_diff_kge": float(diff_kge.mean()),
            "median_diff_kge": float(np.median(diff_kge)),
            "ci95_lo_basin_boot": float(ci_lo),
            "ci95_hi_basin_boot": float(ci_hi),
            "wilcoxon_stat": float(w_res.statistic),
            "wilcoxon_p": float(w_res.pvalue),
            "param_diff_median": float(np.median(param_diff)),
            "param_diff_q25": float(np.quantile(param_diff, 0.25)),
            "param_diff_q75": float(np.quantile(param_diff, 0.75)),
            "param_diff_mean": float(param_diff.mean()),
        })

        print(f"[sec34_06] [{cond}] mean diff={diff_kge.mean():+.5f} [95% CI {ci_lo:+.5f}, {ci_hi:+.5f}], median={np.median(diff_kge):+.5f}, wilcoxon p={w_res.pvalue:.4e}, coincided={coincided}/{nb} ({coincided/nb:.1%}), param diff median={np.median(param_diff):.1f}")

    df_sum = pd.DataFrame(summary_rows)
    S.write_tidy_table(df_sum, TABLES / "sec34_parsimony_penalty_summary.csv")

    df_basins = pd.DataFrame(basin_diff_rows)
    S.write_tidy_table(df_basins, TABLES / "sec34_parsimony_penalty_per_basin.csv")

    print("[sec34_06] PASS: parsimony penalty analysis completed across all 3 conditions")


if __name__ == "__main__":
    main()
