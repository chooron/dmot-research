#!/usr/bin/env python3
"""sec34_15_a3_recompute_parsimony_dispersion_idjoin.py — A3: ID-joined recomputation of parsimony cost and equivalence-set dispersion.

Recomputes, with strict basin-ID joins, for IC, seen-dPL, and OOB-dPL:
  * Parsimony cost (ranking leader minus most-parsimonious member): mean/median diff, Wilcoxon p,
    basin-bootstrap 95% CI, N, min-parameter tie frequency, parameter-count diff median/Q25/Q75.
  * Within-equivalence-set point-KGE dispersion: range, IQR, set size, best-minus-worst diff
    (median/Q25/Q75/P90), plus full distribution statistics.
Reports old vs new side by side with absolute differences. The old tables are marked SUPERSEDED in PROVENANCE;
their values are not modified.
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
    basins = S.canonical_basin_ids()   # SORTED (A6 accessor)
    nb = len(basins)
    bi = S.basin_position_map(basins)  # A6 accessor

    # Fold assignment keyed by basin ID
    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])

    # n_parameters per model
    attr = pd.read_csv(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"
                       / "sec32_library_members" / "tables" / "sec32_model_structural_attributes.csv")
    n_params = attr.set_index("model")["n_parameters"].to_dict()

    # Performance matrices (sorted)
    models_oob, _, oob_mat = S.load_oob_kge_matrix()
    models_seen, _, seen_mat = S.load_canonical_seen_kge_matrix()

    # IC matrix (sorted)
    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    mi = {m: i for i, m in enumerate(models)}
    bi_sorted = {b: i for i, b in enumerate(basins)}
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    cube_ic = np.zeros((len(models), nb, 15, 6), dtype=np.float64)
    idx_m = df_raw["model"].map(mi).to_numpy()
    idx_b = df_raw["basin_id"].map(S.pad_basin_id).map(bi_sorted).to_numpy()   # A6 accessor
    idx_y = df_raw["water_year"].map(lambda y: y - 1996).to_numpy()
    cube_ic[idx_m, idx_b, idx_y] = df_raw[stat_cols_ic].to_numpy(dtype=np.float64)
    ic_mat = S.kge_from_sufficient(cube_ic.sum(axis=2)).T  # [531, 36] sorted

    perf_mats = {
        "IC": (ic_mat, models),
        "seen-dPL": (seen_mat, models_seen),
        "OOB-dPL": (oob_mat, models_oob),
    }

    # Equivalence sets keyed by (basin_str, condition)
    eq_df = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")
    eq_df["bs"] = eq_df["basin_id"].map(S.pad_basin_id)   # A6 accessor
    eq_dict = eq_df.set_index(["bs", "condition"])["equivalent_models"].to_dict()

    # Existing tables
    exist_pen_sum = pd.read_csv(TABLES / "sec34_parsimony_penalty_summary.csv").set_index("condition")
    exist_pen_pb = pd.read_csv(TABLES / "sec34_parsimony_penalty_per_basin.csv")
    exist_pen_pb["bs"] = exist_pen_pb["basin_id"].map(S.pad_basin_id)   # A6 accessor
    exist_disp_sum = pd.read_csv(TABLES / "sec34_equivalence_set_dispersion_summary.csv")
    exist_disp_pb = pd.read_csv(TABLES / "sec34_equivalence_set_dispersion_per_basin.csv")
    exist_disp_pb["bs"] = exist_disp_pb["basin_id"].map(S.pad_basin_id)   # A6 accessor

    # =====================================================================
    # A3.1 Parsimony cost (ID-joined)
    # =====================================================================
    pen_sum_rows = []
    pen_pb_rows = []
    pen_compare_rows = []

    for cond, (mat, mod_list) in perf_mats.items():
        kge_rank = np.empty(nb)
        kge_pars = np.empty(nb)
        diff_kge = np.empty(nb)
        param_diff = np.empty(nb)
        ties_broken = 0
        coincided = 0

        for f in sorted(set(folds)):
            tr = np.flatnonzero(folds != f)
            te = np.flatnonzero(folds == f)
            med_tr = np.median(mat[tr], axis=0)  # [36] sorted row positions
            rank_order = {mod_list[i]: rank for rank, i in enumerate(np.argsort(-med_tr, kind="stable"))}

            for b_i in te:
                bid = basins[b_i]
                eq_mods = str(eq_dict[(bid, cond)]).split("|")
                # (i) ranking leader
                best_rank_mod = min(eq_mods, key=lambda m: rank_order[m])
                # (ii) parsimonious member (tie-break by training rank)
                min_p = min(n_params[m] for m in eq_mods)
                cands = [m for m in eq_mods if n_params[m] == min_p]
                if len(cands) > 1:
                    ties_broken += 1
                best_pars_mod = min(cands, key=lambda m: rank_order[m])
                if best_rank_mod == best_pars_mod:
                    coincided += 1

                idx_r = mod_list.index(best_rank_mod)
                idx_p = mod_list.index(best_pars_mod)
                kge_r = mat[b_i, idx_r]
                kge_p = mat[b_i, idx_p]
                kge_rank[b_i] = kge_r
                kge_pars[b_i] = kge_p
                diff_kge[b_i] = kge_r - kge_p
                param_diff[b_i] = n_params[best_rank_mod] - n_params[best_pars_mod]

                pen_pb_rows.append({
                    "condition": cond,
                    "basin_id": bid,
                    "fold": int(f),
                    "ranking_member": best_rank_mod,
                    "ranking_member_kge": float(kge_r),
                    "ranking_member_nparams": int(n_params[best_rank_mod]),
                    "parsimonious_member": best_pars_mod,
                    "parsimonious_member_kge": float(kge_p),
                    "parsimonious_member_nparams": int(n_params[best_pars_mod]),
                    "diff_kge_rank_minus_pars": float(kge_r - kge_p),
                    "diff_nparams_rank_minus_pars": int(n_params[best_rank_mod] - n_params[best_pars_mod]),
                    "is_identical": bool(best_rank_mod == best_pars_mod),
                    "subtraction_order": "ranking_member_minus_parsimonious_member",
                    "join_method": "basin_id_keyed",
                })

        nz = diff_kge[diff_kge != 0]
        w_res = sstats.wilcoxon(nz)
        ci_lo, ci_hi = S.paired_bootstrap_ci(diff_kge, n_boot=1000, seed=20260925)

        new_row = {
            "condition": cond,
            "n_basins": nb,
            "n_coincided": coincided,
            "frac_coincided": float(coincided / nb),
            "n_ties_broken_min_params": ties_broken,
            "subtraction_order": "ranking_member_minus_parsimonious_member",
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
            "join_method": "basin_id_keyed",
        }
        pen_sum_rows.append(new_row)

        e = exist_pen_sum.loc[cond]
        for metric in ["mean_diff_kge", "median_diff_kge", "wilcoxon_p",
                       "ci95_lo_basin_boot", "ci95_hi_basin_boot",
                       "param_diff_median", "param_diff_q25", "param_diff_q75",
                       "n_coincided", "n_ties_broken_min_params"]:
            pen_compare_rows.append({
                "condition": cond,
                "metric": metric,
                "old_value": float(e[metric]),
                "new_value": float(new_row[metric]),
                "abs_diff": abs(float(e[metric]) - float(new_row[metric])),
            })

        print(f"[sec34_15] [{cond}] parsimony: old mean diff={float(e['mean_diff_kge']):+.5f}, "
              f"new={new_row['mean_diff_kge']:+.5f} (abs diff {abs(float(e['mean_diff_kge'])-new_row['mean_diff_kge']):.5f})")

    df_pen_sum = pd.DataFrame(pen_sum_rows)
    S.write_tidy_table(df_pen_sum, TABLES / "sec34_a3_parsimony_penalty_idjoin_summary.csv")
    df_pen_pb = pd.DataFrame(pen_pb_rows)
    S.write_tidy_table(df_pen_pb, TABLES / "sec34_a3_parsimony_penalty_idjoin_per_basin.csv")
    df_pen_cmp = pd.DataFrame(pen_compare_rows)
    S.write_tidy_table(df_pen_cmp, TABLES / "sec34_a3_parsimony_penalty_comparison.csv")

    # =====================================================================
    # A3.2 Equivalence-set dispersion (ID-joined)
    # =====================================================================
    disp_sum_rows = []
    disp_pb_rows = []
    disp_compare_rows = []

    for cond, (mat, mod_list) in perf_mats.items():
        ranges = np.empty(nb)
        iqrs = np.empty(nb)
        sizes = np.empty(nb)
        bw = np.empty(nb)

        for b_i, bid in enumerate(basins):
            eq_mods = str(eq_dict[(bid, cond)]).split("|")
            m_idx = [mod_list.index(m) for m in eq_mods]
            kges = mat[b_i, m_idx]
            s_size = len(kges)
            k_min, k_max = float(kges.min()), float(kges.max())
            k_range = k_max - k_min
            k_iqr = float(np.quantile(kges, 0.75) - np.quantile(kges, 0.25)) if s_size > 1 else 0.0
            sizes[b_i] = s_size
            ranges[b_i] = k_range
            iqrs[b_i] = k_iqr
            bw[b_i] = k_range

            disp_pb_rows.append({
                "basin_id": bid,
                "condition": cond,
                "equivalence_set_size": int(s_size),
                "point_kge_min": k_min,
                "point_kge_max": k_max,
                "point_kge_range": k_range,
                "point_kge_iqr": k_iqr,
                "best_minus_worst_diff": k_range,
                "join_method": "basin_id_keyed",
            })

        for m_name, arr in [("point_kge_range", ranges), ("point_kge_iqr", iqrs),
                            ("equivalence_set_size", sizes), ("best_minus_worst_diff", bw)]:
            disp_sum_rows.append({
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
                "join_method": "basin_id_keyed",
            })

        # Compare with existing summary
        for metric in ["point_kge_range", "point_kge_iqr", "equivalence_set_size", "best_minus_worst_diff"]:
            e = exist_disp_sum[(exist_disp_sum.condition == cond) & (exist_disp_sum.metric == metric)].iloc[0]
            n = [r for r in disp_sum_rows if r["condition"] == cond and r["metric"] == metric][0]
            for stat in ["median", "q25", "q75", "p90"]:
                disp_compare_rows.append({
                    "condition": cond,
                    "metric": metric,
                    "statistic": stat,
                    "old_value": float(e[stat]),
                    "new_value": float(n[stat]),
                    "abs_diff": abs(float(e[stat]) - float(n[stat])),
                })

        old_med = float(exist_disp_sum[(exist_disp_sum.condition == cond) & (exist_disp_sum.metric == "point_kge_range")]["median"].iloc[0])
        print(f"[sec34_15] [{cond}] dispersion range median: old={old_med:.4f}, new={np.median(ranges):.4f}")

    df_disp_sum = pd.DataFrame(disp_sum_rows)
    S.write_tidy_table(df_disp_sum, TABLES / "sec34_a3_dispersion_idjoin_summary.csv")
    df_disp_pb = pd.DataFrame(disp_pb_rows)
    S.write_tidy_table(df_disp_pb, TABLES / "sec34_a3_dispersion_idjoin_per_basin.csv")
    df_disp_cmp = pd.DataFrame(disp_compare_rows)
    S.write_tidy_table(df_disp_cmp, TABLES / "sec34_a3_dispersion_comparison.csv")

    # Per-basin mismatch counts (old vs new) — both sides zero-padded string IDs
    pb_pen = exist_pen_pb.assign(bs=exist_pen_pb["basin_id"].map(S.pad_basin_id)).merge(
        df_pen_pb.assign(bs=df_pen_pb["basin_id"].map(S.pad_basin_id))[["condition", "bs", "diff_kge_rank_minus_pars"]],
        on=["condition", "bs"], suffixes=("_old", "_new"))
    pb_pen["abs_diff"] = (pb_pen["diff_kge_rank_minus_pars_old"] - pb_pen["diff_kge_rank_minus_pars_new"]).abs()
    pb_pen_mismatch = pb_pen.groupby("condition")["abs_diff"].agg(
        n_rows="size", n_mismatch=lambda s: int((s > 1e-4).sum()), max_abs_diff="max").reset_index()

    pb_disp = exist_disp_pb.assign(bs=exist_disp_pb["basin_id"].map(S.pad_basin_id)).merge(
        df_disp_pb.assign(bs=df_disp_pb["basin_id"].map(S.pad_basin_id))[["condition", "bs", "point_kge_range"]],
        on=["condition", "bs"], suffixes=("_old", "_new"))
    pb_disp["abs_diff"] = (pb_disp["point_kge_range_old"] - pb_disp["point_kge_range_new"]).abs()
    pb_disp_mismatch = pb_disp.groupby("condition")["abs_diff"].agg(
        n_rows="size", n_mismatch=lambda s: int((s > 1e-4).sum()), max_abs_diff="max").reset_index()

    pb_disp_mismatch["table"] = "dispersion_point_kge_range"
    pb_pen_mismatch["table"] = "parsimony_diff_kge"

    df_mismatch = pd.concat([pb_pen_mismatch, pb_disp_mismatch], ignore_index=True)
    S.write_tidy_table(df_mismatch, TABLES / "sec34_a3_per_basin_mismatch_summary.csv")

    print("\n[A3] Per-basin old-vs-new mismatch summary:")
    print(df_mismatch[["table", "condition", "n_rows", "n_mismatch", "max_abs_diff"]].to_string(index=False))
    print("\n[A3] Parsimony summary comparison (abs diffs):")
    print(df_pen_cmp.pivot_table(index="condition", columns="metric", values="abs_diff").to_string())
    print("\n[A3] Dispersion summary comparison (abs diffs):")
    print(df_disp_cmp.pivot_table(index="condition", columns="metric", values="abs_diff").to_string())


if __name__ == "__main__":
    main()
