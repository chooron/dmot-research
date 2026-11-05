#!/usr/bin/env python3
"""sec34_11_audit_basin_alignment_a1_a2.py — Tasks A1 & A2: Static code audit and empirical table alignment verification.

A1: Static audit of every basin-level positional indexing or concatenation across manuscript scripts and loaded upstream code.
A2: Empirical row-by-row verification of all tables with basin-level rows against raw upstream data joined strictly by basin ID.
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


def run_a1_static_audit() -> pd.DataFrame:
    """Enumerate every location performing positional indexing/slicing on basin collections."""
    records = [
        # _shared/data_loaders.py
        {
            "file": "_shared/data_loaders.py",
            "line_number": 47,
            "code_snippet": "return [str(b).zfill(8) for b in sorted(canonical_ids())]",
            "index_source_and_ordering": "sorted(canonical_ids()) [SORTED]",
            "indexed_object_and_ordering": "canonical_ids() [UNSORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Explicitly sorts canonical_ids() to establish the canonical sorted basin order.",
        },
        {
            "file": "_shared/data_loaders.py",
            "line_number": 86,
            "code_snippet": "seen = _c4.kge_from_sufficient(dp.sum(axis=2)).T",
            "index_source_and_ordering": "c4.stats_cube('test') basins [SORTED]",
            "indexed_object_and_ordering": "seen array [531, 36] [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "c4.stats_cube internally sorts basin IDs; seen rows align with sorted basins.",
        },
        {
            "file": "_shared/data_loaders.py",
            "line_number": 97,
            "code_snippet": "oob = (pub.pivot(...).reindex(index=basins, columns=models).to_numpy())",
            "index_source_and_ordering": "basin_order() [SORTED]",
            "indexed_object_and_ordering": "DataFrame pivot index [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Explicit reindex(index=basins) guarantees sorted basin alignment.",
        },
        {
            "file": "_shared/data_loaders.py",
            "line_number": 141,
            "code_snippet": "cube[mi, bi[bid], yi] = st[j]",
            "index_source_and_ordering": "heldout_basin_ids.txt mapped via bi = {b: i for i, b in enumerate(basin_order())} [SORTED]",
            "indexed_object_and_ordering": "cube [36, 531, 15, 6] [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "bi maps basin ID string to its index in sorted basin_order(); no positional slicing.",
        },
        # _verification/scripts/v1_recompute_abstract_numbers.py
        {
            "file": "_verification/scripts/v1_recompute_abstract_numbers.py",
            "line_number": 148,
            "code_snippet": "basins = sorted([int(x) for x in raw_ids]); basins_str = [str(b).zfill(8) for b in basins]",
            "index_source_and_ordering": "sorted(raw_ids) [SORTED]",
            "indexed_object_and_ordering": "bi = {b: i for i, b in enumerate(basins_str)} [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Independent recomputation explicitly sorted raw_ids before creating bi dictionary.",
        },
        {
            "file": "_verification/scripts/v1_recompute_abstract_numbers.py",
            "line_number": 198,
            "code_snippet": "cube_oob[mi_idx, bi[bid], yi_idx] = st[j]",
            "index_source_and_ordering": "bid from heldout_basin_ids.txt mapped by bi [SORTED]",
            "indexed_object_and_ordering": "cube_oob [36, 531, 15, 6] [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Basin dimension indexed strictly by dictionary lookup bi[bid].",
        },
        # sec31_transfer_and_library/scripts/sec31_03_rank_preservation.py
        {
            "file": "sec31_transfer_and_library/scripts/sec31_03_rank_preservation.py",
            "line_number": 29,
            "code_snippet": "seen_med_mat = pub.pivot(...).loc[basins, models].to_numpy()",
            "index_source_and_ordering": "S.basin_order() [SORTED]",
            "indexed_object_and_ordering": "pub pivot table [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Explicit .loc[basins] reindexing with sorted basin_order() ensures alignment.",
        },
        {
            "file": "sec31_transfer_and_library/scripts/sec31_03_rank_preservation.py",
            "line_number": 64,
            "code_snippet": "for b in range(len(basins)): s_row = seen_med_mat[b]; o_row = oob_mat[b]",
            "index_source_and_ordering": "range(len(basins)) [SORTED]",
            "indexed_object_and_ordering": "seen_med_mat and oob_mat [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Both matrices were reindexed by the exact same sorted basins list.",
        },
        # sec31_transfer_and_library/scripts/sec31_07_jackknife_equivalence_e1.py (SUPERSEDED)
        {
            "file": "sec31_transfer_and_library/scripts/sec31_07_jackknife_equivalence_e1.py",
            "line_number": 124,
            "code_snippet": "for b_idx, b_id in enumerate(basins_str): eq_mods = [models[m_idx] for m_idx in np.flatnonzero(eq_jk[:, b_idx])]",
            "index_source_and_ordering": "basins_str from S.canonical_ids() [UNSORTED]",
            "indexed_object_and_ordering": "cube_oob loaded via S.load_oob_cube() [SORTED]",
            "verdict": "MISALIGNED",
            "evidence_and_basis": "For OOB-dPL, cube_oob was in sorted order while basins_str was unsorted; row labels misassigned in output CSV (superseded script).",
        },
        # sec31_transfer_and_library/scripts/sec31_08_multiyear_block_bootstrap_c1.py
        {
            "file": "sec31_transfer_and_library/scripts/sec31_08_multiyear_block_bootstrap_c1.py",
            "line_number": 158,
            "code_snippet": "for b_idx, b_id in enumerate(basins_str): eq_mods = [models[m_idx] for m_idx in np.flatnonzero(eq[:, b_idx])]",
            "index_source_and_ordering": "basins_str from S.canonical_ids() [UNSORTED]",
            "indexed_object_and_ordering": "cube_oob loaded via S.load_oob_cube() [SORTED]",
            "verdict": "MISALIGNED",
            "evidence_and_basis": "For OOB-dPL, cube_oob had sorted rows while basins_str was unsorted; set cover size is permutation-invariant, but basin_id in equivalence_sets.csv was mislabeled.",
        },
        # sec31_transfer_and_library/scripts/sec31_09_uncertainty_bandwidth_comparison_c2.py
        {
            "file": "sec31_transfer_and_library/scripts/sec31_09_uncertainty_bandwidth_comparison_c2.py",
            "line_number": 128,
            "code_snippet": "for b_idx, b_id in enumerate(basins_str): best_point_kge = float(best_pt[b_idx])",
            "index_source_and_ordering": "basins_str from S.canonical_ids() [UNSORTED]",
            "indexed_object_and_ordering": "cube_oob loaded via S.load_oob_cube() [SORTED]",
            "verdict": "MISALIGNED",
            "evidence_and_basis": "For OOB-dPL, best_pt[b_idx] was from sorted cube_oob, while b_id was from unsorted basins_str; Delta_b value correct for the row but attached to mismatched basin_id in per-basin CSV.",
        },
        # sec32_library_members/scripts/sec32_02_coverage_ladder_regret.py
        {
            "file": "sec32_library_members/scripts/sec32_02_coverage_ladder_regret.py",
            "line_number": 27,
            "code_snippet": "basins = S.basin_order(); ... reindex(index=basins, columns=models)",
            "index_source_and_ordering": "S.basin_order() [SORTED]",
            "indexed_object_and_ordering": "pub pivot table [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Uses basin_order() for reindexing both seen and OOB matrices.",
        },
        # sec32_library_members/scripts/sec32_08_singleton_and_bottleneck_e4.py
        {
            "file": "sec32_library_members/scripts/sec32_08_singleton_and_bottleneck_e4.py",
            "line_number": 36,
            "code_snippet": "sub = df_eq[df_eq.condition == condition].sort_values('basin_id').reset_index(drop=True)",
            "index_source_and_ordering": "sort_values('basin_id') [SORTED]",
            "indexed_object_and_ordering": "B matrix rows [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Explicitly sorts sub by basin_id before building B matrix; uncovered basins indexed from sub.iloc.",
        },
        # sec33_usage_and_conditioning/scripts/sec33_02_rule_c_vs_b_t3_1.py
        {
            "file": "sec33_usage_and_conditioning/scripts/sec33_02_rule_c_vs_b_t3_1.py",
            "line_number": 28,
            "code_snippet": "basins = S.basin_order(); ... set_index('bs').loc[basins, 'kge']",
            "index_source_and_ordering": "S.basin_order() [SORTED]",
            "indexed_object_and_ordering": "t2_pb DataFrame indexed by bs string [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Explicit .loc[basins] by ID string ensures row alignment across Rule B and Rule C.",
        },
        # sec33_usage_and_conditioning/scripts/sec33_03_fixed_ensemble_scan_t3_2.py
        {
            "file": "sec33_usage_and_conditioning/scripts/sec33_03_fixed_ensemble_scan_t3_2.py",
            "line_number": 81,
            "code_snippet": "b = bi[bid]; ... ens_wy[k][b] = wy_stats(ens_series, obs[:, j])",
            "index_source_and_ordering": "bid from heldout_basin_ids.txt mapped via bi = {b: i for i, b in enumerate(basins)} [SORTED]",
            "indexed_object_and_ordering": "ens_wy [531, 15, 6] [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "basins is loaded from load_oob_kge_matrix() (sorted); bi[bid] correctly routes fold files to sorted index.",
        },
        # sec33_usage_and_conditioning/scripts/sec33_07_diversity_sensitivity_v5.py
        {
            "file": "sec33_usage_and_conditioning/scripts/sec33_07_diversity_sensitivity_v5.py",
            "line_number": 146,
            "code_snippet": "b_pos = bi[bid]; ... ens_wy[k][b_pos] = wy_stats(...)",
            "index_source_and_ordering": "bid from heldout_basin_ids.txt mapped via bi [SORTED]",
            "indexed_object_and_ordering": "ens_wy [531, 15, 6] [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Identical dictionary lookup bi[bid] from sorted basins.",
        },
        # sec33_usage_and_conditioning/scripts/sec33_08_recommender_k2_k5_e8.py
        {
            "file": "sec33_usage_and_conditioning/scripts/sec33_08_recommender_k2_k5_e8.py",
            "line_number": 326,
            "code_snippet": "b_idx = bi[bid]; ... ens_wy_B[k_val][b_idx] = wy_stats(...)",
            "index_source_and_ordering": "bid from heldout_basin_ids.txt mapped via bi = {str(b).zfill(8): i for i, b in enumerate(basins)} [SORTED]",
            "indexed_object_and_ordering": "ens_wy_B and ens_wy_C [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "basins from c4.stats_cube is sorted; bi[bid] routes strictly by ID.",
        },
        # sec33_usage_and_conditioning/scripts/sec33_09_rules_ladder_c3.py
        {
            "file": "sec33_usage_and_conditioning/scripts/sec33_09_rules_ladder_c3.py",
            "line_number": 117,
            "code_snippet": "b = bi[bid]; ... ens_B[k][b] = wy_stats(...)",
            "index_source_and_ordering": "bid from heldout_basin_ids.txt mapped via bi = {b: i for i, b in enumerate(basins)} [SORTED]",
            "indexed_object_and_ordering": "ens_B and ens_C [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "basins is basin_order() (sorted); bi[bid] routes strictly by ID.",
        },
        # sec34_determinacy_and_bounds/scripts/sec34_06_parsimony_penalty_e5.py
        {
            "file": "sec34_determinacy_and_bounds/scripts/sec34_06_parsimony_penalty_e5.py",
            "line_number": 92,
            "code_snippet": "for b_i in te: bid = basins_str[b_i]; ... kge_r = mat[b_i, idx_r]",
            "index_source_and_ordering": "b_i index from te (splits of folds array aligned to unsorted basins_str) [UNSORTED]",
            "indexed_object_and_ordering": "mat = oob_mat or seen_mat loaded from _shared [SORTED]",
            "verdict": "MISALIGNED",
            "evidence_and_basis": "b_i indexes basins_str in unsorted order (data/531sub_id.txt), while mat has rows in sorted order. For 439/531 basins, bid != sorted_basins[b_i].",
        },
        # sec34_determinacy_and_bounds/scripts/sec34_09_equivalence_dispersion_c4.py
        {
            "file": "sec34_determinacy_and_bounds/scripts/sec34_09_equivalence_dispersion_c4.py",
            "line_number": 75,
            "code_snippet": "for b_idx, bid in enumerate(basins_str): eq_mods = str(sub_eq.loc[int(bid), ...]); kges = mat[b_idx, m_indices]",
            "index_source_and_ordering": "b_idx from enumerate(basins_str) where basins_str is S.canonical_ids() [UNSORTED]",
            "indexed_object_and_ordering": "mat = oob_mat or seen_mat loaded from _shared [SORTED]",
            "verdict": "MISALIGNED",
            "evidence_and_basis": "sub_eq.loc[int(bid)] took equivalence models from basin A, while mat[b_idx] took KGEs from basin B (sorted row b_idx). 439/531 basins mismatched in seen-dPL and OOB-dPL.",
        },
        # sec34_determinacy_and_bounds/scripts/sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py
        {
            "file": "sec34_determinacy_and_bounds/scripts/sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py",
            "line_number": 254,
            "code_snippet": "for b_idx, bid in enumerate(basins_sorted): eq_mods = str(sub_eq.loc[int(bid), ...]); kges = mat[b_idx, m_idx]",
            "index_source_and_ordering": "basins_sorted from load_oob_kge_matrix() [SORTED]",
            "indexed_object_and_ordering": "mat = oob_mat_sorted [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Diagnostic recomputation explicitly aligned basins_sorted with sorted mat rows.",
        },
        # Upstream: project/benchmark/analysis/chapter4_equivalence_recommendability_20260925/scripts/common.py
        {
            "file": "project/benchmark/analysis/.../common.py",
            "line_number": 178,
            "code_snippet": "ids = np.asarray(sorted(canonical_ids()), dtype=np.int64); raw = CatchmentAttributeBuilder().load_raw_attributes(ids)",
            "index_source_and_ordering": "sorted(canonical_ids()) [SORTED]",
            "indexed_object_and_ordering": "raw attribute matrix [SORTED]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Explicitly notes in docstring that data/531sub_id.txt is NOT sorted (439 positions disagree) and forces sorting.",
        },
        # Upstream: project/benchmarkpub/scripts/oob36_prelim/13_basin_attribute_screening.py
        {
            "file": "project/benchmarkpub/scripts/oob36_prelim/13_basin_attribute_screening.py",
            "line_number": 170,
            "code_snippet": "BAS = BAS.merge(a[['basin_id'] + ...], on='basin_id', how='left')",
            "index_source_and_ordering": "basin_id column merge [KEYED_BY_ID]",
            "indexed_object_and_ordering": "caravan_671_attributes.csv [KEYED_BY_ID]",
            "verdict": "ALIGNED",
            "evidence_and_basis": "Joined strictly on basin_id integer column, never positional.",
        },
    ]
    df = pd.DataFrame(records)
    S.write_tidy_table(df, TABLES / "sec34_a1_static_basin_indexing_audit.csv")
    return df


def run_a2_table_alignment_audit() -> pd.DataFrame:
    """Empirically test every 'one-row-per-basin' table in the analysis tree by recomputing via ID-join."""
    # Build raw lookup tables
    pub = pd.read_csv(S.DIR_OOB36_PRELIM / "pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv")
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)

    raw_stats_path = S.DIR_GAP_CONDITIONAL / "forward/test/water_year_sufficient_stats.csv"
    df_raw = pd.read_csv(raw_stats_path)
    df_raw["bs"] = df_raw["basin_id"].astype(str).str.zfill(8)

    models = S.MODELS_36

    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    stat_cols_dp = ["n_dpl", "sum_sim_dpl", "sum_obs_dpl", "sum_sim2_dpl", "sum_obs2_dpl", "sum_simobs_dpl"]

    grp_ic = df_raw.groupby(["bs", "model"])[stat_cols_ic].sum()
    kge_ic_dict = {(b, m): S.kge_from_sufficient(grp_ic.loc[(b, m)].to_numpy()) for b, m in grp_ic.index}

    grp_dp = df_raw.groupby(["bs", "model"])[stat_cols_dp].sum()
    kge_dp_dict = {(b, m): S.kge_from_sufficient(grp_dp.loc[(b, m)].to_numpy()) for b, m in grp_dp.index}

    kge_oob_dict = pub.set_index(["bs", "model"])["kge_heldout"].to_dict()

    kge_map = {"IC": kge_ic_dict, "seen-dPL": kge_dp_dict, "OOB-dPL": kge_oob_dict}

    eq_df = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")
    eq_df["bs"] = eq_df["basin_id"].astype(str).str.zfill(8)
    eq_dict = eq_df.set_index(["bs", "condition"])["equivalent_models"].to_dict()
    eq_size_dict = eq_df.set_index(["bs", "condition"])["n_equivalent"].to_dict()
    eq_best_kge_dict = eq_df.set_index(["bs", "condition"])["best_model_kge"].to_dict()

    def audit_one(table_name: str, rel_path: str, col_name: str, source_desc: str, calc_fn) -> Dict:
        p = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / rel_path
        df = pd.read_csv(p)
        df["bs"] = df["basin_id"].astype(str).str.zfill(8)
        diffs = []
        for _, r in df.iterrows():
            expected = calc_fn(r)
            observed = r[col_name]
            diffs.append(abs(expected - observed))
        diffs = np.array(diffs)
        n_mismatches = int((diffs > 1e-4).sum())
        return {
            "table_name": table_name,
            "file_path": rel_path,
            "tested_column": col_name,
            "raw_upstream_source": source_desc,
            "join_method": "Strict basin_id string join",
            "n_rows_checked": len(df),
            "n_mismatches": n_mismatches,
            "max_abs_diff": float(diffs.max()),
            "alignment_status": "PASS" if n_mismatches == 0 else "FAIL_MISMATCH",
        }

    results = []

    # 1. sec31_basin_rank_spearman_per_basin.csv
    results.append(audit_one(
        "sec31_basin_rank_spearman_per_basin.csv",
        "sec31_transfer_and_library/tables/sec31_basin_rank_spearman_per_basin.csv",
        "spearman_rho",
        "pub_same_pipeline_model_basin.csv (kge_seenfolds_median4 vs kge_heldout)",
        lambda r: float(sstats.spearmanr(
            [pub.loc[(pub.bs == r["bs"]) & (pub.model == m), "kge_seenfolds_median4"].iloc[0] for m in models],
            [kge_oob_dict[(r["bs"], m)] for m in models]
        ).statistic)
    ))

    # 2. sec31_equivalence_sets_per_basin.csv
    results.append(audit_one(
        "sec31_equivalence_sets_per_basin.csv",
        "sec31_transfer_and_library/tables/sec31_equivalence_sets_per_basin.csv",
        "best_model_kge",
        "equiv_cover_t1t2_20261003/tables/t1_equivalence_sets.csv",
        lambda r: eq_best_kge_dict[(r["bs"], r["condition"])]
    ))

    # 3. sec31_uncertainty_bandwidth_per_basin.csv
    results.append(audit_one(
        "sec31_uncertainty_bandwidth_per_basin.csv",
        "sec31_transfer_and_library/tables/sec31_uncertainty_bandwidth_per_basin.csv",
        "best_point_kge",
        "water_year_sufficient_stats.csv + pub_same_pipeline_model_basin.csv",
        lambda r: max(kge_map[r["condition"]][(r["bs"], m)] for m in models)
    ))

    # 4. sec32_basin_singleton_status.csv
    results.append(audit_one(
        "sec32_basin_singleton_status.csv",
        "sec32_library_members/tables/sec32_basin_singleton_status.csv",
        "equivalence_size",
        "equiv_cover_t1t2_20261003/tables/t1_equivalence_sets.csv",
        lambda r: eq_size_dict[(r["bs"], r["condition"])]
    ))

    # 5. sec33_fixed_ensemble_per_basin.csv (k=1)
    mem_df = pd.read_csv(TABLES.parent.parent / "sec33_usage_and_conditioning" / "tables" / "sec33_fixed_ensemble_fold_members.csv")
    lead_models = mem_df[mem_df.k == 1].set_index("fold")["lead_model"].to_dict()
    results.append(audit_one(
        "sec33_fixed_ensemble_per_basin.csv (k=1)",
        "sec33_usage_and_conditioning/tables/sec33_fixed_ensemble_per_basin.csv",
        "kge",
        "pub_same_pipeline_model_basin.csv (lead model KGE)",
        lambda r: pub.loc[(pub.bs == r["bs"]) & (pub.model == lead_models[r["fold"]]), "kge_heldout"].iloc[0] if r["k"] == 1 else r["kge"]
    ))

    # 6. sec33_usage_rules_per_basin.csv (Rule A)
    results.append(audit_one(
        "sec33_usage_rules_per_basin.csv (Rule A)",
        "sec33_usage_and_conditioning/tables/sec33_usage_rules_per_basin.csv",
        "kge",
        "pub_same_pipeline_model_basin.csv (hbv96 heldout KGE)",
        lambda r: pub.loc[(pub.bs == r["bs"]) & (pub.model == "hbv96"), "kge_heldout"].iloc[0] if r["rule"] == "A" else r["kge"]
    ))

    # 7. sec34_equivalence_set_dispersion_per_basin.csv
    results.append(audit_one(
        "sec34_equivalence_set_dispersion_per_basin.csv",
        "sec34_determinacy_and_bounds/tables/sec34_equivalence_set_dispersion_per_basin.csv",
        "point_kge_range",
        "t1_equivalence_sets.csv + raw performance matrices joined by basin ID",
        lambda r: (
            max(kge_map[r["condition"]][(r["bs"], m)] for m in str(eq_dict[(r["bs"], r["condition"])]).split("|")) -
            min(kge_map[r["condition"]][(r["bs"], m)] for m in str(eq_dict[(r["bs"], r["condition"])]).split("|"))
        )
    ))

    # 8. sec34_parsimony_penalty_per_basin.csv
    results.append(audit_one(
        "sec34_parsimony_penalty_per_basin.csv",
        "sec34_determinacy_and_bounds/tables/sec34_parsimony_penalty_per_basin.csv",
        "ranking_member_kge",
        "t1_equivalence_sets.csv + raw performance matrices joined by basin ID",
        lambda r: kge_map[r["condition"]][(r["bs"], r["ranking_member"])]
    ))

    df_out = pd.DataFrame(results)
    S.write_tidy_table(df_out, TABLES / "sec34_a2_table_alignment_audit.csv")
    return df_out


def main() -> None:
    print("=" * 78)
    print("A1 & A2: Comprehensive Audit of Basin Positional Indexing and Table Alignment")
    print("=" * 78)

    df_a1 = run_a1_static_audit()
    print(f"\n[A1 Static Audit] Processed {len(df_a1)} positional indexing locations:")
    verdict_counts = df_a1["verdict"].value_counts().to_dict()
    for v, c in verdict_counts.items():
        print(f"  {v}: {c}")

    df_a2 = run_a2_table_alignment_audit()
    print(f"\n[A2 Empirical Audit] Checked {len(df_a2)} basin-level tables across manuscript tree:")
    print(df_a2[["table_name", "n_rows_checked", "n_mismatches", "max_abs_diff", "alignment_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
