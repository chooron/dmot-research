#!/usr/bin/env python3
"""sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py — Comprehensive diagnostic audit of equivalence range vs bandwidth constraint.

Performs:
  K1: Basin-by-basin constraint audit (range_b <= Delta_b) on existing tables; details for violating basins.
  K2: Provenance trace of bandwidth vs dispersion inputs and pipelines side-by-side.
  K3: Single-source aligned recomputation of bandwidth and dispersion from a unified data source.
  K4: Hypothesis testing (S_b aligned vs All-36 vs Top-3 vs Top-5 vs basin-order index mismatch).
  K5: Parsimony penalty constraint audit (penalty_b <= Delta_b).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    models = S.MODELS_36
    models_oob, basins_sorted, oob_mat_sorted = S.load_oob_kge_matrix()  # [531, 36], sorted
    models_seen, _, seen_mat_sorted = S.load_canonical_seen_kge_matrix()  # [531, 36], sorted
    nb, nm = len(basins_sorted), len(models)

    # 1. Load IC sorted
    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    mi = {m: i for i, m in enumerate(models)}
    bi_sorted = {b: i for i, b in enumerate(basins_sorted)}
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    stat_cols_dp = ["n_dpl", "sum_sim_dpl", "sum_obs_dpl", "sum_sim2_dpl", "sum_obs2_dpl", "sum_simobs_dpl"]

    cube_ic_sorted = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    cube_dp_sorted = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    idx_m = df_raw["model"].map(mi).to_numpy()
    idx_b_sort = df_raw["basin_id"].astype(str).str.zfill(8).map(bi_sorted).to_numpy()
    idx_y = df_raw["water_year"].map(lambda y: y - 1996).to_numpy()

    cube_ic_sorted[idx_m, idx_b_sort, idx_y] = df_raw[stat_cols_ic].to_numpy(dtype=np.float64)
    cube_dp_sorted[idx_m, idx_b_sort, idx_y] = df_raw[stat_cols_dp].to_numpy(dtype=np.float64)
    cube_oob_sorted = S.load_oob_cube()

    ic_mat_sorted = S.kge_from_sufficient(cube_ic_sorted.sum(axis=2)).T  # [531, 36]

    # Equivalence sets
    eq_df = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")

    # Existing tables
    df_bw_exist = pd.read_csv(ROOT / "project/benchmarkpub/manuscript/analysis/sec31_transfer_and_library/tables/sec31_uncertainty_bandwidth_per_basin.csv")
    df_disp_exist = pd.read_csv(TABLES / "sec34_equivalence_set_dispersion_per_basin.csv")
    df_pen_exist = pd.read_csv(TABLES / "sec34_parsimony_penalty_per_basin.csv")

    # =========================================================================
    # K1: Basin-by-basin constraint audit (range_b <= Delta_b) on existing tables
    # =========================================================================
    merged_k1 = df_disp_exist.merge(df_bw_exist, on=["basin_id", "condition"])

    k1_summary_rows = []
    k1_viol_detail_rows = []

    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        sub = merged_k1[merged_k1.condition == cond].copy()
        delta_b = sub["bandwidth_1yr_block_bootstrap_primary"].to_numpy(float)
        range_b = sub["point_kge_range"].to_numpy(float)

        viol_mask = range_b > (delta_b + 1e-12)
        n_viol = int(viol_mask.sum())
        frac_viol = float(n_viol / nb)

        ratio = np.where(delta_b > 1e-9, range_b / delta_b, np.nan)
        r_clean = ratio[np.isfinite(ratio)]

        k1_summary_rows.append({
            "condition": cond,
            "n_basins_total": nb,
            "n_violations": n_viol,
            "violation_fraction": frac_viol,
            "inequality_holds_globally": bool(n_viol == 0),
            "ratio_range_over_delta_median": float(np.median(r_clean)),
            "ratio_range_over_delta_q25": float(np.quantile(r_clean, 0.25)),
            "ratio_range_over_delta_q75": float(np.quantile(r_clean, 0.75)),
            "ratio_range_over_delta_p90": float(np.quantile(r_clean, 0.90)),
            "ratio_range_over_delta_max": float(r_clean.max()),
        })

        # Detailed records for violating basins
        sub_viol = sub[viol_mask].copy()
        sub_eq = eq_df[eq_df.condition == cond].set_index("basin_id")
        for _, r in sub_viol.iterrows():
            bid_str = str(r["basin_id"]).zfill(8)
            bid_int = int(bid_str)
            eq_row = sub_eq.loc[bid_int]
            eq_mods_str = str(eq_row["equivalent_models"])
            q05_val = float(eq_row["q05"])
            k1_viol_detail_rows.append({
                "condition": cond,
                "basin_id": bid_str,
                "equivalent_models": eq_mods_str,
                "equivalence_size": int(r["equivalence_set_size"]),
                "point_kge_min_recorded": float(r["point_kge_min"]),
                "point_kge_max_recorded": float(r["point_kge_max"]),
                "point_kge_range_recorded": float(r["point_kge_range"]),
                "best_point_kge": float(r["best_point_kge"]),
                "q05_threshold": q05_val,
                "bandwidth_delta_b": float(r["bandwidth_1yr_block_bootstrap_primary"]),
                "excess_range_minus_delta": float(r["point_kge_range"] - r["bandwidth_1yr_block_bootstrap_primary"]),
                "ratio_range_over_delta": float(r["point_kge_range"] / r["bandwidth_1yr_block_bootstrap_primary"]) if r["bandwidth_1yr_block_bootstrap_primary"] > 0 else np.nan,
            })

    df_k1_sum = pd.DataFrame(k1_summary_rows)
    S.write_tidy_table(df_k1_sum, TABLES / "sec34_k1_constraint_audit_summary.csv")

    df_k1_detail = pd.DataFrame(k1_viol_detail_rows)
    S.write_tidy_table(df_k1_detail, TABLES / "sec34_k1_constraint_violations_detail.csv")

    # =========================================================================
    # K2: Provenance trace of bandwidth vs dispersion inputs and pipelines
    # =========================================================================
    k2_rows = [
        {
            "dimension": "source_table",
            "uncertainty_bandwidth_Delta": "sec31_transfer_and_library/tables/sec31_uncertainty_bandwidth_per_basin.csv",
            "equivalence_dispersion_range": "sec34_determinacy_and_bounds/tables/sec34_equivalence_set_dispersion_per_basin.csv",
        },
        {
            "dimension": "generating_script",
            "uncertainty_bandwidth_Delta": "sec31_transfer_and_library/scripts/sec31_09_uncertainty_bandwidth_comparison_c2.py",
            "equivalence_dispersion_range": "sec34_determinacy_and_bounds/scripts/sec34_09_equivalence_dispersion_c4.py",
        },
        {
            "dimension": "q05_definition",
            "uncertainty_bandwidth_Delta": "5th percentile of {max_m KGE_{m,b}(r)} across 1000 water-year block draws",
            "equivalence_dispersion_range": "Inherited from t1_equivalence_sets.csv (same 5th percentile cutoff)",
        },
        {
            "dimension": "resampling_protocol",
            "uncertainty_bandwidth_Delta": "Water-year block bootstrap (L=1 year, 15 blocks, 1000 draws, seed 20260925)",
            "equivalence_dispersion_range": "Water-year block bootstrap (L=1 year, 15 blocks, 1000 draws, seed 20260925)",
        },
        {
            "dimension": "basin_order_and_indexing",
            "uncertainty_bandwidth_Delta": "Sorted basin order matching stats_cube and Qsim arrays (bi[basin_id])",
            "equivalence_dispersion_range": "Unsorted canonical_ids() order used to index sorted performance matrices mat[b_idx, :]",
        },
        {
            "dimension": "basin_alignment_consequence",
            "uncertainty_bandwidth_Delta": "Consistent: best_point_kge and q05 belong to the same basin (531/531)",
            "equivalence_dispersion_range": "Misaligned: S_b from basin A evaluated on KGEs of basin B for 439/531 basins in seen-dPL/OOB-dPL",
        },
        {
            "dimension": "n_basins_evaluated",
            "uncertainty_bandwidth_Delta": "531",
            "equivalence_dispersion_range": "531",
        },
        {
            "dimension": "n_basins_excluded",
            "uncertainty_bandwidth_Delta": "0",
            "equivalence_dispersion_range": "0",
        },
    ]
    df_k2 = pd.DataFrame(k2_rows)
    S.write_tidy_table(df_k2, TABLES / "sec34_k2_provenance_comparison.csv")

    # =========================================================================
    # K3: Single-source aligned recomputation of bandwidth and dispersion
    # =========================================================================
    counts = S.load_year_counts(n_reps=1000, seed=20260925)
    cubes_aligned = {
        "IC": cube_ic_sorted,
        "seen-dPL": cube_dp_sorted,
        "OOB-dPL": cube_oob_sorted,
    }

    existing_bw_med = {"IC": 0.0623, "seen-dPL": 0.0524, "OOB-dPL": 0.0577}
    existing_range_med = {"IC": 0.0549, "seen-dPL": 0.1027, "OOB-dPL": 0.1163}

    k3_rows = []
    for cond, cmat in cubes_aligned.items():
        point = S.kge_from_sufficient(cmat.sum(axis=2))  # [36, 531]
        best_pt = point.max(axis=0)                      # [531]

        flat = cmat.transpose(2, 1, 0, 3).reshape(15, nb * 36 * 6)
        pool = (counts @ flat).reshape(1000, nb, 36, 6)
        best_reps = S.kge_from_sufficient(pool).max(axis=2)  # [1000, 531]
        q05 = np.quantile(best_reps, 0.05, axis=0)           # [531]

        delta_b = best_pt - q05
        eq_mask = (point >= q05[None, :])  # [36, 531]

        ranges_aligned = np.empty(nb)
        for b in range(nb):
            m_idx = np.flatnonzero(eq_mask[:, b])
            kges = point[m_idx, b]
            ranges_aligned[b] = kges.max() - kges.min()

        viol = ranges_aligned > (delta_b + 1e-12)

        med_bw_recomp = float(np.median(delta_b))
        med_rng_recomp = float(np.median(ranges_aligned))
        diff_bw = abs(med_bw_recomp - existing_bw_med[cond])
        diff_rng = abs(med_rng_recomp - existing_range_med[cond])

        k3_rows.append({
            "condition": cond,
            "recomputed_bandwidth_median": med_bw_recomp,
            "existing_bandwidth_median": existing_bw_med[cond],
            "bandwidth_abs_diff": diff_bw,
            "recomputed_range_median": med_rng_recomp,
            "existing_range_median": existing_range_med[cond],
            "range_abs_diff": diff_rng,
            "recomputed_violations_count": int(viol.sum()),
            "recomputed_max_excess_range_minus_delta": float((ranges_aligned - delta_b).max()),
            "mathematical_constraint_satisfied": bool(viol.sum() == 0),
        })

    df_k3 = pd.DataFrame(k3_rows)
    S.write_tidy_table(df_k3, TABLES / "sec34_k3_single_source_recomputation.csv")

    # =========================================================================
    # K4: Hypothesis testing (S_b vs All-36 vs Top-3 vs Top-5 vs Index Mismatch)
    # =========================================================================
    canonical_ids = [str(b).zfill(8) for b in S.canonical_ids()]
    # Reconstruct the exact buggy matrix used in sec34_09
    bi_canon = {b: i for i, b in enumerate(canonical_ids)}
    cube_ic_canon = np.zeros((nm, nb, 15, 6), dtype=np.float64)
    idx_b_c = df_raw["basin_id"].astype(str).str.zfill(8).map(bi_canon).to_numpy()
    cube_ic_canon[idx_m, idx_b_c, idx_y] = df_raw[stat_cols_ic].to_numpy(float)
    ic_mat_canon = S.kge_from_sufficient(cube_ic_canon.sum(axis=2)).T

    k4_rows = []
    perf_mats_aligned = {
        "OOB-dPL": (oob_mat_sorted, models_oob),
        "seen-dPL": (seen_mat_sorted, models_seen),
        "IC": (ic_mat_sorted, models),
    }

    for cond, (mat, mod_list) in perf_mats_aligned.items():
        sub_eq = eq_df[eq_df.condition == cond].set_index("basin_id")

        # Hyp A: S_b correctly aligned
        r_Sb_aligned = []
        for b_idx, bid in enumerate(basins_sorted):
            eq_mods = str(sub_eq.loc[int(bid), "equivalent_models"]).split("|")
            m_idx = [mod_list.index(m) for m in eq_mods]
            kges = mat[b_idx, m_idx]
            r_Sb_aligned.append(kges.max() - kges.min())

        # Hyp B: All 36 models
        r_all36 = mat.max(axis=1) - mat.min(axis=1)

        # Hyp C: Top-3 models
        top3_kges = np.sort(mat, axis=1)[:, -3:]
        r_top3 = top3_kges[:, -1] - top3_kges[:, 0]

        # Hyp D: Top-5 models
        top5_kges = np.sort(mat, axis=1)[:, -5:]
        r_top5 = top5_kges[:, -1] - top5_kges[:, 0]

        # Hyp E: S_b with basin-order index mismatch (sec34_09 code path)
        mat_buggy = ic_mat_canon if cond == "IC" else mat
        r_buggy = []
        for b_idx, bid in enumerate(canonical_ids):
            eq_mods = str(sub_eq.loc[int(bid), "equivalent_models"]).split("|")
            m_idx = [mod_list.index(m) for m in eq_mods]
            kges = mat_buggy[b_idx, m_idx]
            r_buggy.append(kges.max() - kges.min())

        hypotheses = [
            ("Hypothesis_A_Sb_aligned", float(np.median(r_Sb_aligned))),
            ("Hypothesis_B_All36_models", float(np.median(r_all36))),
            ("Hypothesis_C_Top3_models", float(np.median(r_top3))),
            ("Hypothesis_D_Top5_models", float(np.median(r_top5))),
            ("Hypothesis_E_Sb_index_mismatch", float(np.median(r_buggy))),
        ]

        target_exist = existing_range_med[cond]
        for h_name, h_val in hypotheses:
            k4_rows.append({
                "condition": cond,
                "hypothesis": h_name,
                "range_median": h_val,
                "existing_reported_median": target_exist,
                "abs_diff_vs_existing": abs(h_val - target_exist),
                "is_exact_match": bool(abs(h_val - target_exist) < 1e-4),
            })

    df_k4 = pd.DataFrame(k4_rows)
    S.write_tidy_table(df_k4, TABLES / "sec34_k4_hypotheses_evaluation.csv")

    # =========================================================================
    # K5: Parsimony penalty constraint audit (penalty_b <= Delta_b)
    # =========================================================================
    merged_k5 = df_pen_exist.merge(df_bw_exist, on=["basin_id", "condition"])

    k5_rows = []
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        sub = merged_k5[merged_k5.condition == cond].copy()
        delta_b = sub["bandwidth_1yr_block_bootstrap_primary"].to_numpy(float)
        penalty_b = sub["diff_kge_rank_minus_pars"].to_numpy(float)

        viol_mask = penalty_b > (delta_b + 1e-12)
        n_viol = int(viol_mask.sum())
        frac_viol = float(n_viol / nb)

        ratio = np.where(delta_b > 1e-9, penalty_b / delta_b, np.nan)
        r_clean = ratio[np.isfinite(ratio)]

        k5_rows.append({
            "condition": cond,
            "n_basins_total": nb,
            "n_violations_penalty_gt_delta": n_viol,
            "violation_fraction": frac_viol,
            "ratio_penalty_over_delta_median": float(np.median(r_clean)),
            "ratio_penalty_over_delta_q25": float(np.quantile(r_clean, 0.25)),
            "ratio_penalty_over_delta_q75": float(np.quantile(r_clean, 0.75)),
            "ratio_penalty_over_delta_p90": float(np.quantile(r_clean, 0.90)),
            "ratio_penalty_over_delta_max": float(r_clean.max()),
        })

    df_k5 = pd.DataFrame(k5_rows)
    S.write_tidy_table(df_k5, TABLES / "sec34_k5_parsimony_penalty_constraint_audit.csv")

    print("[sec34_10] PASS: completed all diagnostic audits K1-K5")
    print("\n--- K1 Summary (Range <= Delta_b) ---")
    print(df_k1_sum[["condition", "n_violations", "violation_fraction", "ratio_range_over_delta_median", "ratio_range_over_delta_p90"]].to_string(index=False))
    print("\n--- K3 Single Source Recomputation ---")
    print(df_k3[["condition", "recomputed_bandwidth_median", "recomputed_range_median", "recomputed_violations_count"]].to_string(index=False))
    print("\n--- K5 Parsimony Penalty Audit (Penalty <= Delta_b) ---")
    print(df_k5[["condition", "n_violations_penalty_gt_delta", "violation_fraction", "ratio_penalty_over_delta_median", "ratio_penalty_over_delta_p90"]].to_string(index=False))


if __name__ == "__main__":
    main()
