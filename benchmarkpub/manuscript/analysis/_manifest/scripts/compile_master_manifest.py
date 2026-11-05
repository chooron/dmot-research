#!/usr/bin/env python3
"""compile_master_manifest.py — Generate the master manifest of all cited manuscript numbers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
BASE = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"
MANIFEST_DIR = BASE / "_manifest"
TABLES = MANIFEST_DIR / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Sec 3.1: Transfer and Library
    # -------------------------------------------------------------------------
    p1 = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_pooled_delta_kge.csv").iloc[0]
    records.append({
        "quantity_name": "pooled_delta_kge_pub_median",
        "value": f"{p1['median']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_pooled_delta_kge.csv",
        "column_or_key": "median",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_01_delta_kge.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": f"95% CI [{p1['median_ci95_lo']:.6f}, {p1['median_ci95_hi']:.6f}], N=19116 cells",
    })
    records.append({
        "quantity_name": "pooled_delta_kge_pub_mean",
        "value": f"{p1['mean']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_pooled_delta_kge.csv",
        "column_or_key": "mean",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_01_delta_kge.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": f"95% CI [{p1['mean_ci95_lo']:.6f}, {p1['mean_ci95_hi']:.6f}]",
    })
    records.append({
        "quantity_name": "models_positive_delta_kge_fraction",
        "value": "36 / 36 (1.000)",
        "file_path": "sec31_transfer_and_library/tables/sec31_model_level_delta_kge.csv",
        "column_or_key": "delta_pub_median > 0",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_01_delta_kge.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "Sign-test p = 2.9e-11",
    })

    vd = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_variance_decomposition.csv").set_index("component")
    records.append({
        "quantity_name": "two_way_anova_basin_ss_share",
        "value": f"{vd.loc['basin', 'frac_of_total_ss']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_variance_decomposition.csv",
        "column_or_key": "frac_of_total_ss (basin)",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_02_variance_decomposition.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "61.01% of total sum of squares on Delta_KGE_PUB",
    })
    records.append({
        "quantity_name": "two_way_anova_model_ss_share",
        "value": f"{vd.loc['model', 'frac_of_total_ss']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_variance_decomposition.csv",
        "column_or_key": "frac_of_total_ss (model)",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_02_variance_decomposition.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "1.02% of total sum of squares on Delta_KGE_PUB",
    })

    # V4: Global Seen vs OOB model rank Spearman (both registered, same-pipeline marked as manuscript value)
    gr = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_global_rank_spearman.csv").set_index("seen_definition")
    records.append({
        "quantity_name": "global_seen_vs_oob_rank_spearman_samepipeline",
        "value": f"{gr.loc['samepipeline_kge_seenfolds_median4', 'spearman_rho']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_global_rank_spearman.csv",
        "column_or_key": "spearman_rho (samepipeline_kge_seenfolds_median4)",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_03_rank_preservation.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "MANUSCRIPT_VALUE: same-pipeline in-sample seen folds median vs OOB KGE (0.976062; marked as official manuscript value per V4)",
    })
    records.append({
        "quantity_name": "global_seen_vs_oob_rank_spearman_canonical",
        "value": f"{gr.loc['crosspipeline_canonical_dpl', 'spearman_rho']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_global_rank_spearman.csv",
        "column_or_key": "spearman_rho (crosspipeline_canonical_dpl)",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_03_rank_preservation.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv + seen_ic_oob_merged.parquet",
        "derivation_type": "RECOMPUTED",
        "notes": "REFERENCE_SECONDARY_CONVENTION: cross-pipeline canonical seen-dPL vs OOB KGE (0.970142; cited as 0.97)",
    })

    br = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_basin_rank_spearman_summary.csv").iloc[0]
    records.append({
        "quantity_name": "basin_level_seen_vs_oob_rank_spearman_median",
        "value": f"{br['spearman_median']:.6f}",
        "file_path": "sec31_transfer_and_library/tables/sec31_basin_rank_spearman_summary.csv",
        "column_or_key": "spearman_median",
        "results_section": "Results 3.1",
        "generating_script": "sec31_transfer_and_library/scripts/sec31_03_rank_preservation.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": f"95% CI [{br['spearman_ci95_lo']:.6f}, {br['spearman_ci95_hi']:.6f}], mean {br['spearman_mean']:.4f}",
    })

    sz = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_equivalence_size_distribution.csv").set_index("condition")
    for cond, lab in [("IC", "IC"), ("seen-dPL", "seen_dpl"), ("OOB-dPL", "oob_dpl")]:
        records.append({
            "quantity_name": f"equivalence_set_size_median_{lab}",
            "value": f"{sz.loc[cond, 'median_size']:.1f}",
            "file_path": "sec31_transfer_and_library/tables/sec31_equivalence_size_distribution.csv",
            "column_or_key": "median_size",
            "results_section": "Results 3.1",
            "generating_script": "sec31_transfer_and_library/scripts/sec31_04_equivalence_and_coverage.py",
            "upstream_source": "equiv_cover_t1t2_20261003/tables/t1_size_distribution.csv",
            "derivation_type": "LOADED",
            "notes": f"Q25={sz.loc[cond, 'q25_size']:.1f}, Q75={sz.loc[cond, 'q75_size']:.1f}, mean={sz.loc[cond, 'mean_size']:.2f}",
        })

    mc = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_minimal_cover_targets.csv")
    for _, row in mc.iterrows():
        records.append({
            "quantity_name": f"minimal_cover_{row['condition']}_target_{int(row['target']*100)}pct",
            "value": str(int(row["n_models_exact"])),
            "file_path": "sec31_transfer_and_library/tables/sec31_minimal_cover_targets.csv",
            "column_or_key": "n_models_exact",
            "results_section": "Results 3.1",
            "generating_script": "sec31_transfer_and_library/scripts/sec31_04_equivalence_and_coverage.py",
            "upstream_source": "equiv_cover_t1t2_20261003/tables/t1_min_cover.csv",
            "derivation_type": "LOADED",
            "notes": f"Greedy={int(row['n_models_greedy'])}, models: {row['exact_models']}",
        })

    # V3: Fold composition
    fc = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_fold_composition.csv")
    for k_fold in range(5):
        sub_fc = fc[(fc.scope == f"fold_{k_fold}")].set_index("attribute")
        records.append({
            "quantity_name": f"fold_{k_fold}_basin_count",
            "value": str(int(sub_fc["n_basins"].iloc[0])),
            "file_path": "sec31_transfer_and_library/tables/sec31_fold_composition.csv",
            "column_or_key": "n_basins",
            "results_section": "Results 3.1",
            "generating_script": "sec31_transfer_and_library/scripts/sec31_06_fold_composition.py",
            "upstream_source": "OOB_FOLD_ASSIGNMENT.csv + gauge_locations_671.csv + basin_attribute_table.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"Fold {k_fold} basin count",
        })
        for attr_name in ["lat", "lon", "arid", "f_snow", "runoff_ratio"]:
            records.append({
                "quantity_name": f"fold_{k_fold}_{attr_name}_median",
                "value": f"{sub_fc.loc[attr_name, 'median']:.4f}",
                "file_path": "sec31_transfer_and_library/tables/sec31_fold_composition.csv",
                "column_or_key": "median",
                "results_section": "Results 3.1",
                "generating_script": "sec31_transfer_and_library/scripts/sec31_06_fold_composition.py",
                "upstream_source": "basin_attribute_table.csv / gauge_locations_671.csv",
                "derivation_type": "RECOMPUTED",
                "notes": f"Q25={sub_fc.loc[attr_name, 'q25']:.4f}, Q75={sub_fc.loc[attr_name, 'q75']:.4f}",
            })

    # E1: Jackknife alternative uncertainty convention (SUPERSEDED)
    jk_comp = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_jackknife_vs_bootstrap_cover_comparison.csv")
    for _, row in jk_comp.iterrows():
        records.append({
            "quantity_name": f"jackknife_cover_{row['condition']}_{row['target_pct']}",
            "value": str(int(row["n_models_jackknife"])),
            "file_path": "sec31_transfer_and_library/tables/sec31_jackknife_vs_bootstrap_cover_comparison.csv",
            "column_or_key": "n_models_jackknife",
            "results_section": "Results 3.1",
            "generating_script": "sec31_transfer_and_library/scripts/sec31_07_jackknife_equivalence_e1.py",
            "upstream_source": "water_year_sufficient_stats.csv + OOB Qsim/obs",
            "derivation_type": "SUPERSEDED",
            "notes": f"SUPERSEDED: 15-pseudovalue dispersion too narrow; bootstrap={int(row['n_models_block_bootstrap'])}, diff={int(row['diff_jackknife_minus_bootstrap'])}",
        })

    # C1: Multi-year block bootstrap (L=3 years) active alternative uncertainty convention
    c1_comp = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_multiyear_block_bootstrap_comparison.csv")
    for _, row in c1_comp.iterrows():
        records.append({
            "quantity_name": f"multiyear_3yr_block_boot_cover_{row['condition']}_{row['target_pct']}",
            "value": str(int(row["n_models_3yr_block_bootstrap"])),
            "file_path": "sec31_transfer_and_library/tables/sec31_multiyear_block_bootstrap_comparison.csv",
            "column_or_key": "n_models_3yr_block_bootstrap",
            "results_section": "Results 3.1",
            "generating_script": "sec31_transfer_and_library/scripts/sec31_08_multiyear_block_bootstrap_c1.py",
            "upstream_source": "water_year_sufficient_stats.csv + OOB Qsim/obs",
            "derivation_type": "RECOMPUTED",
            "notes": f"1yr_boot={int(row['n_models_1yr_block_bootstrap'])}, diff={int(row['diff_3yr_minus_1yr'])}, median_size={row['median_size_3yr_boot']:.1f}, opt_cert={row['optimality_certified']}",
        })

    # C2: Uncertainty bandwidth comparison across 3 schemes
    c2_bw = pd.read_csv(BASE / "sec31_transfer_and_library" / "tables" / "sec31_uncertainty_bandwidth_comparison.csv")
    for _, row in c2_bw.iterrows():
        records.append({
            "quantity_name": f"uncertainty_bandwidth_{row['condition']}_{row['scheme']}_median",
            "value": f"{row['median_width']:.4f}",
            "file_path": "sec31_transfer_and_library/tables/sec31_uncertainty_bandwidth_comparison.csv",
            "column_or_key": "median_width",
            "results_section": "Results 3.1",
            "generating_script": "sec31_transfer_and_library/scripts/sec31_09_uncertainty_bandwidth_comparison_c2.py",
            "upstream_source": "water_year_sufficient_stats.csv + OOB Qsim/obs",
            "derivation_type": "SUPERSEDED" if row["scheme_status"] == "SUPERSEDED" else "RECOMPUTED",
            "notes": f"status={row['scheme_status']}, Q25={row['q25_width']:.4f}, Q75={row['q75_width']:.4f}, range [{row['min_width']:.4f}, {row['max_width']:.4f}]",
        })

    # -------------------------------------------------------------------------
    # Sec 3.2: Library Members
    # -------------------------------------------------------------------------
    lib_f = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_library_model_frequency.csv")
    records.append({
        "quantity_name": "seen_top3_candidate_union_size",
        "value": str(int(lib_f["in_any_top3"].sum())),
        "file_path": "sec32_library_members/tables/sec32_library_model_frequency.csv",
        "column_or_key": "in_any_top3.sum()",
        "results_section": "Results 3.2",
        "generating_script": "sec32_library_members/scripts/sec32_01_library_and_covers.py",
        "upstream_source": "pub_minimal_model_library_20260930/tables/library_model_frequency.csv",
        "derivation_type": "LOADED",
        "notes": "36 of 36 models appear in at least one basin's Top-3",
    })
    records.append({
        "quantity_name": "seen_top5_candidate_union_size",
        "value": str(int(lib_f["in_any_top5"].sum())),
        "file_path": "sec32_library_members/tables/sec32_library_model_frequency.csv",
        "column_or_key": "in_any_top5.sum()",
        "results_section": "Results 3.2",
        "generating_script": "sec32_library_members/scripts/sec32_01_library_and_covers.py",
        "upstream_source": "pub_minimal_model_library_20260930/tables/library_model_frequency.csv",
        "derivation_type": "LOADED",
        "notes": "36 of 36 models appear in at least one basin's Top-5",
    })

    lcov = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_minimal_cover_solutions.csv").set_index("k")
    records.append({
        "quantity_name": "exact_minimum_cover_top3",
        "value": str(int(lcov.loc[3, "min_cover_size"])),
        "file_path": "sec32_library_members/tables/sec32_minimal_cover_solutions.csv",
        "column_or_key": "min_cover_size (k=3)",
        "results_section": "Results 3.2",
        "generating_script": "sec32_library_members/scripts/sec32_01_library_and_covers.py",
        "upstream_source": "pub_minimal_model_library_20260930/tables/library_minimum_cover.csv",
        "derivation_type": "LOADED",
        "notes": f"MILP proven optimal (status 0); models: {lcov.loc[3, 'min_cover_models']}",
    })
    records.append({
        "quantity_name": "exact_minimum_cover_top5",
        "value": str(int(lcov.loc[5, "min_cover_size"])),
        "file_path": "sec32_library_members/tables/sec32_minimal_cover_solutions.csv",
        "column_or_key": "min_cover_size (k=5)",
        "results_section": "Results 3.2",
        "generating_script": "sec32_library_members/scripts/sec32_01_library_and_covers.py",
        "upstream_source": "pub_minimal_model_library_20260930/tables/library_minimum_cover.csv",
        "derivation_type": "LOADED",
        "notes": f"MILP proven optimal (status 0); models: {lcov.loc[5, 'min_cover_models']}",
    })

    ladder = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_coverage_ladder.csv").set_index("k")
    for k_val in [1, 3, 5, 10]:
        records.append({
            "quantity_name": f"seen_topk_to_oob_top1_coverage_k{k_val}",
            "value": f"{ladder.loc[k_val, 'coverage_top1']:.4f}",
            "file_path": "sec32_library_members/tables/sec32_coverage_ladder.csv",
            "column_or_key": "coverage_top1",
            "results_section": "Results 3.2",
            "generating_script": "sec32_library_members/scripts/sec32_02_coverage_ladder_regret.py",
            "upstream_source": "pub_same_pipeline_model_basin.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"n_covered = {int(ladder.loc[k_val, 'n_covered'])}/531",
        })

    reg_curve = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_regret_curve.csv").set_index("k")
    for k_val in [1, 3, 5]:
        records.append({
            "quantity_name": f"candidate_set_median_regret_k{k_val}",
            "value": f"{reg_curve.loc[k_val, 'regret_median']:.4f}",
            "file_path": "sec32_library_members/tables/sec32_regret_curve.csv",
            "column_or_key": "regret_median",
            "results_section": "Results 3.2",
            "generating_script": "sec32_library_members/scripts/sec32_02_coverage_ladder_regret.py",
            "upstream_source": "pub_same_pipeline_model_basin.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"mean regret = {reg_curve.loc[k_val, 'regret_mean']:.4f}",
        })

    attr_df = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_model_structural_attributes.csv")
    records.append({
        "quantity_name": "models_with_explicit_snow_routine_count",
        "value": f"{int(attr_df['has_explicit_snow_routine'].sum())} / {len(attr_df)}",
        "file_path": "sec32_library_members/tables/sec32_model_structural_attributes.csv",
        "column_or_key": "has_explicit_snow_routine.sum()",
        "results_section": "Results 3.2",
        "generating_script": "sec32_library_members/scripts/sec32_03_model_structural_attributes.py",
        "upstream_source": "model_complexity_inventory.csv + model_snow_annotation.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "8 models contain explicit snow routine in AST",
    })

    # V2: Process representation cross-tabulations
    proc_sum = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_process_representation_summary.csv").set_index("subset_name")
    for sname in ["oob_100pct_cover_21", "seen_top3_min_cover_13", "seen_top5_min_cover_9"]:
        records.append({
            "quantity_name": f"{sname}_snow_routine_count",
            "value": f"{int(proc_sum.loc[sname, 'n_with_snow_routine'])} / {int(proc_sum.loc[sname, 'n_models'])}",
            "file_path": "sec32_library_members/tables/sec32_process_representation_summary.csv",
            "column_or_key": "n_with_snow_routine",
            "results_section": "Results 3.2",
            "generating_script": "sec32_library_members/scripts/sec32_05_process_crosstabs.py",
            "upstream_source": "sec32_model_structural_attributes.csv + sec32_oob_full_cover_members.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"pct with snow = {proc_sum.loc[sname, 'pct_with_snow_routine']:.1%}",
        })

    # E2: Basin-level bootstrap minimal library size CIs
    boot_ci_df = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_bootstrap_cover_ci_summary.csv")
    for _, row in boot_ci_df.iterrows():
        records.append({
            "quantity_name": f"bootstrap_cover_ci_{row['condition']}_{row['target_pct']}_median",
            "value": f"{row['median']:.1f}",
            "file_path": "sec32_library_members/tables/sec32_bootstrap_cover_ci_summary.csv",
            "column_or_key": "median",
            "results_section": "Results 3.2",
            "generating_script": "sec32_library_members/scripts/sec32_06_bootstrap_cover_ci_e2.py",
            "upstream_source": "t1_equivalence_sets.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"obs={int(row['k_observed'])}, 95% CI [{row['ci95_lo']:.1f}, {row['ci95_hi']:.1f}], range [{int(row['min'])}, {int(row['max'])}]",
        })

    # E3: Optimal set cover enumeration
    opt_sum_df = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_optimal_covers_summary.csv")
    for _, row in opt_sum_df.iterrows():
        records.append({
            "quantity_name": f"optimal_solutions_count_{row['condition']}_{row['target_pct']}",
            "value": str(int(row["n_optimal_solutions"])),
            "file_path": "sec32_library_members/tables/sec32_optimal_covers_summary.csv",
            "column_or_key": "n_optimal_solutions",
            "results_section": "Results 3.2",
            "generating_script": "sec32_library_members/scripts/sec32_07_optimal_solutions_enumeration_e3.py",
            "upstream_source": "t1_equivalence_sets.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"k*={int(row['k_opt'])}, core_size={int(row['core_set_size'])}, union_size={int(row['union_set_size'])}, truncated={row['is_truncated']}",
        })

    # E4: Singletons and bottlenecks
    btn_df = pd.read_csv(BASE / "sec32_library_members" / "tables" / "sec32_bottleneck_100pct_coverage.csv")
    for _, row in btn_df.iterrows():
        records.append({
            "quantity_name": f"bottleneck_forcing_models_{row['condition']}_100pct",
            "value": str(row["bottleneck_forcing_models"]),
            "file_path": "sec32_library_members/tables/sec32_bottleneck_100pct_coverage.csv",
            "column_or_key": "bottleneck_forcing_models",
            "results_section": "Results 3.2",
            "generating_script": "sec32_library_members/scripts/sec32_08_singleton_and_bottleneck_e4.py",
            "upstream_source": "t1_equivalence_sets.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"K*={int(row['k_100pct_opt'])}, budget {int(row['budget_minus_1'])} covers {int(row['max_basins_covered_budget_minus_1'])}/531; uncovered: {row['uncovered_basin_ids']}",
        })

    # -------------------------------------------------------------------------
    # Sec 3.3: Usage and Conditioning
    # -------------------------------------------------------------------------
    t2_sum = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_usage_rules_summary.csv")
    for _, r in t2_sum.iterrows():
        records.append({
            "quantity_name": f"rule_{r['rule']}_k{r['k']}_median_kge" if pd.isna(r["seed"]) else f"rule_{r['rule']}_seed_{int(r['seed'])}_median_kge",
            "value": f"{r['kge_median']:.5f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_usage_rules_summary.csv",
            "column_or_key": "kge_median",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_01_usage_rules_t2.py",
            "upstream_source": "equiv_cover_t1t2_20261003/tables/t2_rule_summary.csv",
            "derivation_type": "LOADED",
            "notes": f"diff vs A mean: {r['diff_mean']:+.5f}, wilcoxon p: {r['wilcoxon_p']:.4e}",
        })

    t31_sum = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_rule_c_vs_b_summary.csv").iloc[0]
    records.append({
        "quantity_name": "t3_1_rule_c_vs_b_diff_mean",
        "value": f"{t31_sum['mean_diff_mean']:+.6f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_rule_c_vs_b_summary.csv",
        "column_or_key": "mean_diff_mean",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_02_rule_c_vs_b_t3_1.py",
        "upstream_source": "t2_rule_per_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": f"10-seed range [{t31_sum['mean_diff_min']:+.6f}, {t31_sum['mean_diff_max']:+.6f}], wilcoxon p mean={t31_sum['wilcoxon_p_mean']:.4f}",
    })

    t32_scan = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_fixed_ensemble_k_scan.csv").set_index("k")
    for k_val in [1, 2, 3, 4, 5, 7, 10]:
        records.append({
            "quantity_name": f"t3_2_fixed_ensemble_k{k_val}_median_kge",
            "value": f"{t32_scan.loc[k_val, 'kge_median']:.5f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_fixed_ensemble_k_scan.csv",
            "column_or_key": "kge_median",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_03_fixed_ensemble_scan_t3_2.py",
            "upstream_source": "oob_cube.npy + Qsim fold files",
            "derivation_type": "RECOMPUTED",
            "notes": f"diff vs k=1: {t32_scan.loc[k_val, 'diff_vs_k1_mean']:+.5f} (wilcoxon p={t32_scan.loc[k_val, 'wilcoxon_p']:.4e})",
        })

    # V5: Diversity sensitivities
    div_sum = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_diversity_sensitivity_summary.csv")
    for _, r in div_sum.iterrows():
        records.append({
            "quantity_name": f"{r['selection_method']}_k{r['k']}_diff_vs_ranking_mean",
            "value": f"{r['diff_mean']:+.6f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_diversity_sensitivity_summary.csv",
            "column_or_key": "diff_mean",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_07_diversity_sensitivity_v5.py",
            "upstream_source": "raw Qsim on training folds + model structural attributes",
            "derivation_type": "SENSITIVITY",
            "notes": f"kge_median={r['kge_diversity_median']:.4f} vs ranking {r['kge_ranking_median']:.4f}, wilcoxon p={r['wilcoxon_p']:.4e}",
        })

    gset_df = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_gset_summary.csv").set_index("role")
    records.append({
        "quantity_name": "gset_A2d_primary_point",
        "value": f"{gset_df.loc['PRIMARY', 'G_set']:.6f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_gset_summary.csv",
        "column_or_key": "G_set (PRIMARY)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_04_gset_and_regret.py",
        "upstream_source": "oob36_gset_gate_20261001/tables/gset_36_summary.json",
        "derivation_type": "LOADED",
        "notes": f"95% CI [{gset_df.loc['PRIMARY', 'lo95']:.6f}, {gset_df.loc['PRIMARY', 'hi95']:.6f}], 10-seed range [{gset_df.loc['PRIMARY', 'G_set_min']:.4f}, {gset_df.loc['PRIMARY', 'G_set_max']:.4f}], CI>0 in 0/10",
    })
    records.append({
        "quantity_name": "gset_A1b_secondary_point",
        "value": f"{gset_df.loc['SECONDARY_SENSITIVITY', 'G_set']:.6f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_gset_summary.csv",
        "column_or_key": "G_set (SECONDARY_SENSITIVITY)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_04_gset_and_regret.py",
        "upstream_source": "oob36_gset_gate_20261001/tables/gset_36_summary.json",
        "derivation_type": "LOADED",
        "notes": f"95% CI [{gset_df.loc['SECONDARY_SENSITIVITY', 'lo95']:.6f}, {gset_df.loc['SECONDARY_SENSITIVITY', 'hi95']:.6f}], CI>0 in 0/10",
    })
    records.append({
        "quantity_name": "gset_Seen_reference_point",
        "value": f"{gset_df.loc['REFERENCE_CEILING', 'G_set']:.6f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_gset_summary.csv",
        "column_or_key": "G_set (REFERENCE_CEILING)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_04_gset_and_regret.py",
        "upstream_source": "oob36_gset_gate_20261001/tables/parity_p1_seen36_per_seed.csv",
        "derivation_type": "LOADED",
        "notes": f"95% CI [{gset_df.loc['REFERENCE_CEILING', 'lo95']:.6f}, {gset_df.loc['REFERENCE_CEILING', 'hi95']:.6f}], 10/10 CI>0",
    })

    reg_sum = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_paired_regret_summary.csv").set_index("case")
    for case in ["Seen", "A2d", "A1b"]:
        records.append({
            "quantity_name": f"paired_regret_diff_mean_{case}",
            "value": f"{reg_sum.loc[case, 'mean_diff_mean']:+.6f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_paired_regret_summary.csv",
            "column_or_key": f"mean_diff_mean ({case})",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_04_gset_and_regret.py",
            "upstream_source": "oob36_paired_regret_20261003/tables/paired_regret_summary.csv",
            "derivation_type": "LOADED",
            "notes": f"wilcoxon p mean={reg_sum.loc[case, 'wilcoxon_p_mean']:.4f}, p<0.05 in {int(reg_sum.loc[case, 'n_seeds_p_lt_05'])}/10 seeds",
        })

    bs_glob = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_basin_specific_vs_global_topk.csv").set_index("k")
    records.append({
        "quantity_name": "basin_specific_vs_global_top3_cov_diff",
        "value": f"{bs_glob.loc[3, 'coverage_diff_bs_minus_global']:+.6f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_basin_specific_vs_global_topk.csv",
        "column_or_key": "coverage_diff_bs_minus_global (k=3)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_04_gset_and_regret.py",
        "upstream_source": "pub_candidate_set_transfer_20260930/tables/basin_specific_vs_global_topk.csv",
        "derivation_type": "LOADED",
        "notes": f"+7.91 pp, 95% CI [{bs_glob.loc[3, 'coverage_diff_ci95_lo']:+.4f}, {bs_glob.loc[3, 'coverage_diff_ci95_hi']:+.4f}], McNemar p={bs_glob.loc[3, 'mcnemar_exact_p']:.4e}",
    })

    cl_sum = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_cluster_oob_validation_summary.csv").set_index("k")
    records.append({
        "quantity_name": "cluster_oob_top3_coverage_diff",
        "value": f"{cl_sum.loc[3, 'diff_coverage']:+.4f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_cluster_oob_validation_summary.csv",
        "column_or_key": "diff_coverage (k=3)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_05_cluster_typology.py",
        "upstream_source": "cluster_oob_recommendation_validation_20261001/tables/cluster_vs_global_summary.csv",
        "derivation_type": "LOADED",
        "notes": f"McNemar p={cl_sum.loc[3, 'mcnemar_p_value']:.4f}",
    })
    records.append({
        "quantity_name": "cluster_oob_top5_regret_diff_mean",
        "value": f"{cl_sum.loc[5, 'diff_regret_mean']:+.4f}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_cluster_oob_validation_summary.csv",
        "column_or_key": "diff_regret_mean (k=5)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_05_cluster_typology.py",
        "upstream_source": "cluster_oob_recommendation_validation_20261001/tables/cluster_vs_global_summary.csv",
        "derivation_type": "LOADED",
        "notes": f"Wilcoxon p={cl_sum.loc[5, 'wilcoxon_regret_p_value']:.4f}, 95% CI [{cl_sum.loc[5, 'diff_regret_mean_ci95_lo']:+.4f}, {cl_sum.loc[5, 'diff_regret_mean_ci95_hi']:+.4f}]",
    })

    # E8: Operating points k=2 and k=5 conditional evaluation
    e8_parity_df = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_e8_parity_gate.csv")
    records.append({
        "quantity_name": "e8_recommender_gate_parity_max_abs_diff",
        "value": f"{max(e8_parity_df['a2d_abs_diff'].max(), e8_parity_df['a1b_abs_diff'].max()):.3e}",
        "file_path": "sec33_usage_and_conditioning/tables/sec33_e8_parity_gate.csv",
        "column_or_key": "max(a2d_abs_diff, a1b_abs_diff)",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_08_recommender_k2_k5_e8.py",
        "upstream_source": "oob36_gset_gate_20261001",
        "derivation_type": "RECOMPUTED",
        "notes": "Parity passed across all 10 seeds; max abs diff <= 1e-16",
    })

    e8_cb_df = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_e8_rule_c_vs_b_k2_k5.csv")
    for k_val in [2, 5]:
        sub_cb = e8_cb_df[e8_cb_df.k == k_val]
        records.append({
            "quantity_name": f"e8_rule_c_vs_b_k{k_val}_mean_diff_mean",
            "value": f"{sub_cb['mean_diff'].mean():+.6f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_e8_rule_c_vs_b_k2_k5.csv",
            "column_or_key": "mean_diff.mean()",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_08_recommender_k2_k5_e8.py",
            "upstream_source": "raw Qsim + RF recommenders",
            "derivation_type": "RECOMPUTED",
            "notes": f"range [{sub_cb['mean_diff'].min():+.5f}, {sub_cb['mean_diff'].max():+.5f}], sig seeds: {int(sub_cb['wilcoxon_p_lt_005'].sum())}/10",
        })

    e8_gset_df = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_e8_gset_k2_k5.csv")
    for est in ["A2d_PRIMARY", "A1b_SECONDARY"]:
        for k_val in [2, 5]:
            sub_g = e8_gset_df[(e8_gset_df.estimand == est) & (e8_gset_df.k == k_val)]
            records.append({
                "quantity_name": f"e8_gset_{est}_k{k_val}_mean",
                "value": f"{sub_g['G_set'].mean():+.6f}",
                "file_path": "sec33_usage_and_conditioning/tables/sec33_e8_gset_k2_k5.csv",
                "column_or_key": "G_set.mean()",
                "results_section": "Results 3.3",
                "generating_script": "sec33_usage_and_conditioning/scripts/sec33_08_recommender_k2_k5_e8.py",
                "upstream_source": "oob_kge + RF recommenders",
                "derivation_type": "RECOMPUTED",
                "notes": f"range [{sub_g['G_set'].min():+.5f}, {sub_g['G_set'].max():+.5f}], CI>0 in {int(sub_g['ci_positive'].sum())}/10 seeds",
            })

    # C3: Rules ladder unified table
    ladder_df = pd.read_csv(BASE / "sec33_usage_and_conditioning" / "tables" / "sec33_rules_ladder_unified.csv")
    for _, row in ladder_df.iterrows():
        records.append({
            "quantity_name": f"rules_ladder_{row['rule_family']}_k{row['k']}_median",
            "value": f"{row['kge_median']:.5f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_rules_ladder_unified.csv",
            "column_or_key": "kge_median",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_09_rules_ladder_c3.py",
            "upstream_source": "raw Qsim + RF recommenders",
            "derivation_type": "RECOMPUTED",
            "notes": f"{row['rule_label']}: vs_A={row['vs_A_mean_diff']:+.5f} (p={row['vs_A_wilcoxon_p']:.4e}), vs_B2={row['vs_B2_mean_diff']:+.5f} (p={row['vs_B2_wilcoxon_p']:.4e})",
        })

    # -------------------------------------------------------------------------
    # Sec 3.4: Determinacy and Bounds
    # -------------------------------------------------------------------------
    nparam_inst = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_nparam_vs_instability.csv").set_index("y_quantity")
    records.append({
        "quantity_name": "nparam_vs_fold_instability_spearman",
        "value": f"{nparam_inst.loc['param_fold_instability_median', 'spearman_rho']:+.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_nparam_vs_instability.csv",
        "column_or_key": "spearman_rho (param_fold_instability_median)",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_01_complexity_vs_instability.py",
        "upstream_source": "pub_parsimony_complexity_20260930/tables/complexity_vs_transfer.csv",
        "derivation_type": "LOADED",
        "notes": f"perm p = {nparam_inst.loc['param_fold_instability_median', 'spearman_p_permutation']:.4f}",
    })
    records.append({
        "quantity_name": "nparam_vs_disp_rms_spearman",
        "value": f"{nparam_inst.loc['param_displacement_median', 'spearman_rho']:+.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_nparam_vs_instability.csv",
        "column_or_key": "spearman_rho (param_displacement_median)",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_01_complexity_vs_instability.py",
        "upstream_source": "pub_parsimony_complexity_20260930/tables/complexity_vs_transfer.csv",
        "derivation_type": "LOADED",
        "notes": f"perm p = {nparam_inst.loc['param_displacement_median', 'spearman_p_permutation']:.4f}",
    })
    records.append({
        "quantity_name": "nparam_vs_disp_euclidean_spearman",
        "value": f"{nparam_inst.loc['param_displacement_euclid_median', 'spearman_rho']:+.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_nparam_vs_instability.csv",
        "column_or_key": "spearman_rho (param_displacement_euclid_median)",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_01_complexity_vs_instability.py",
        "upstream_source": "pub_parsimony_complexity_20260930/tables/complexity_vs_transfer.csv",
        "derivation_type": "LOADED",
        "notes": f"perm p = {nparam_inst.loc['param_displacement_euclid_median', 'spearman_p_permutation']:.4f}",
    })

    disp_loss = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_displacement_vs_oob_loss.csv").set_index("instability_metric")
    records.append({
        "quantity_name": "disp_rms_vs_oob_loss_pooled_spearman",
        "value": f"{disp_loss.loc['param_displacement_seen_mean_rms', 'pooled_spearman_rho']:+.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_displacement_vs_oob_loss.csv",
        "column_or_key": "pooled_spearman_rho",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_01_complexity_vs_instability.py",
        "upstream_source": "pub_parsimony_complexity_20260930/tables/parameter_instability_vs_oob_loss.csv",
        "derivation_type": "LOADED",
        "notes": f"N=19116 cells, perm p={disp_loss.loc['param_displacement_seen_mean_rms', 'pooled_spearman_p_permutation']:.4e}",
    })
    records.append({
        "quantity_name": "disp_rms_vs_oob_loss_model_level_spearman",
        "value": f"{disp_loss.loc['param_displacement_seen_mean_rms', 'model_median_spearman_rho']:+.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_displacement_vs_oob_loss.csv",
        "column_or_key": "model_median_spearman_rho",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_01_complexity_vs_instability.py",
        "upstream_source": "pub_parsimony_complexity_20260930/tables/parameter_instability_vs_oob_loss.csv",
        "derivation_type": "LOADED",
        "notes": f"N=36 models, perm p={disp_loss.loc['param_displacement_seen_mean_rms', 'model_median_spearman_p_permutation']:.4f}",
    })

    mseed = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_multiseed_8model_dispersion.csv")
    records.append({
        "quantity_name": "multiseed_dpl_d_self_median",
        "value": f"{mseed['D_self_median'].median():.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_multiseed_8model_dispersion.csv",
        "column_or_key": "D_self_median.median()",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_02_multiseed_dispersion.py",
        "upstream_source": "dpl_multiseed_sensitivity_20260914/tables/table_q2_parameter_distance.csv",
        "derivation_type": "LOADED",
        "notes": "8 models x 3 seeds",
    })
    records.append({
        "quantity_name": "multiseed_dpl_d_cross_median",
        "value": f"{mseed['D_cross_median'].median():.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_multiseed_8model_dispersion.csv",
        "column_or_key": "D_cross_median.median()",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_02_multiseed_dispersion.py",
        "upstream_source": "dpl_multiseed_sensitivity_20260914/tables/table_q2_parameter_distance.csv",
        "derivation_type": "LOADED",
        "notes": "8 models x 3 seeds",
    })
    records.append({
        "quantity_name": "multiseed_dpl_self_to_cross_ratio_median",
        "value": f"{mseed['self_to_cross_ratio'].median():.3%}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_multiseed_8model_dispersion.csv",
        "column_or_key": "self_to_cross_ratio.median()",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_02_multiseed_dispersion.py",
        "upstream_source": "dpl_multiseed_sensitivity_20260914/tables/table_q2_parameter_distance.csv",
        "derivation_type": "LOADED",
        "notes": "14.9%",
    })

    p_conc = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_penalty_concentration.csv").set_index("group_label")
    records.append({
        "quantity_name": "top5pct_basins_share_of_total_delta_kge",
        "value": f"{p_conc.loc['top_5pct_basins (top 26 basins)', 'share_of_total_delta']:.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_penalty_concentration.csv",
        "column_or_key": "share_of_total_delta (top 5% basins)",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_03_penalty_concentration.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "34.25% (cited as 34.3%), top 26 basins",
    })
    records.append({
        "quantity_name": "top5_models_share_of_total_delta_kge",
        "value": f"{p_conc.loc['top_5_models', 'share_of_total_delta']:.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_penalty_concentration.csv",
        "column_or_key": "share_of_total_delta (top 5 models)",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_03_penalty_concentration.py",
        "upstream_source": "pub_same_pipeline_model_basin.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "19.35% (cited as 19.4%)",
    })

    scr = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_attribute_screening_bh_fdr.csv")
    records.append({
        "quantity_name": "basin_attribute_screening_min_q_value",
        "value": f"{scr['bh_fdr_q'].min():.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_attribute_screening_bh_fdr.csv",
        "column_or_key": "bh_fdr_q.min()",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_04_attribute_screening.py",
        "upstream_source": "pub_next_stage_20260930/tables/basin_attribute_screening.csv",
        "derivation_type": "LOADED",
        "notes": "0/15 variables survive BH-FDR at q < 0.05",
    })

    # E5: Parsimony penalty within equivalence sets
    pars_sum = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_parsimony_penalty_summary.csv")
    for _, row in pars_sum.iterrows():
        notes_audit = " [AUDIT: INDEX_MISALIGNED_UNSORTED_VS_SORTED in sec34_06]" if row["condition"] in ["seen-dPL", "OOB-dPL"] else " [AUDIT: ALIGNED]"
        records.append({
            "quantity_name": f"parsimony_penalty_mean_diff_{row['condition']}",
            "value": f"{row['mean_diff_kge']:+.5f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_parsimony_penalty_summary.csv",
            "column_or_key": "mean_diff_kge",
            "results_section": "Results 3.4",
            "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_06_parsimony_penalty_e5.py",
            "upstream_source": "t1_equivalence_sets.csv + performance matrices",
            "derivation_type": "RECOMPUTED",
            "notes": f"median={row['median_diff_kge']:+.5f}, wilcoxon p={row['wilcoxon_p']:.4e}, coincided={int(row['n_coincided'])}/531, param diff median={row['param_diff_median']:.1f}{notes_audit}",
        })

    # E6: Partial correlation controlling model performance
    pcorr_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_partial_correlation_nparam_vs_instability.csv")
    for _, row in pcorr_df.iterrows():
        records.append({
            "quantity_name": f"partial_spearman_{row['instability_metric']}_control_seen_kge",
            "value": f"{row['partial_spearman_rho_control_seen_kge']:+.4f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_partial_correlation_nparam_vs_instability.csv",
            "column_or_key": "partial_spearman_rho_control_seen_kge",
            "results_section": "Results 3.4",
            "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_07_partial_correlation_e6.py",
            "upstream_source": "model_level_complexity_performance.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"perm p={row['p_permutation_control_seen_kge']:.4f} (raw rho={row['spearman_rho_uncontrolled']:+.4f}, raw p={row['p_permutation_uncontrolled']:.4f})",
        })

    # E7: Inter-metric correlation and independent recomputation
    corr_mat_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_displacement_metrics_correlation_matrix.csv").set_index("instability_metric")
    records.append({
        "quantity_name": "inter_metric_spearman_fold_instability_vs_disp_rms",
        "value": f"{corr_mat_df.loc['param_fold_instability_median', 'param_displacement_median']:.4f}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_displacement_metrics_correlation_matrix.csv",
        "column_or_key": "spearman_rho (fold_inst vs disp_rms)",
        "results_section": "Results 3.4",
        "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_08_displacement_correlation_recompute_e7.py",
        "upstream_source": "model_level_complexity_performance.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "Strong correlation between fold variability and OOB displacement",
    })

    e7_rec_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_nparam_instability_independent_recomputation.csv")
    for _, row in e7_rec_df.iterrows():
        records.append({
            "quantity_name": f"independent_recompute_nparam_vs_{row['metric_name']}",
            "value": f"{row['recomputed_spearman_rho']:+.6f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_nparam_instability_independent_recomputation.csv",
            "column_or_key": "recomputed_spearman_rho",
            "results_section": "Results 3.4",
            "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_08_displacement_correlation_recompute_e7.py",
            "upstream_source": "params/*.npz (raw theta_norm arrays)",
            "derivation_type": "RECOMPUTED",
            "notes": f"existing={row['existing_spearman_rho']:+.6f}, abs diff={row['abs_diff_spearman_rho']:.3e}, p_perm={row['recomputed_p_permutation']:.4f}",
        })

    # C4: Equivalence set point KGE dispersion
    disp_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_equivalence_set_dispersion_summary.csv")
    for _, row in disp_df.iterrows():
        notes_audit = " [AUDIT: INDEX_MISALIGNED_UNSORTED_VS_SORTED in sec34_09; aligned true median in sec34_k3]" if row["condition"] in ["seen-dPL", "OOB-dPL"] else " [AUDIT: ALIGNED]"
        records.append({
            "quantity_name": f"equiv_dispersion_{row['condition']}_{row['metric']}_median",
            "value": f"{row['median']:.4f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_equivalence_set_dispersion_summary.csv",
            "column_or_key": "median",
            "results_section": "Results 3.4",
            "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_09_equivalence_dispersion_c4.py",
            "upstream_source": "t1_equivalence_sets.csv + performance matrices",
            "derivation_type": "RECOMPUTED",
            "notes": f"Q25={row['q25']:.4f}, Q75={row['q75']:.4f}, P90={row['p90']:.4f}, mean={row['mean']:.4f}{notes_audit}",
        })

    # K1 & K3: Inequality constraint diagnostic audit and single-source recomputation
    k1_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_k1_constraint_audit_summary.csv")
    for _, row in k1_df.iterrows():
        records.append({
            "quantity_name": f"k1_constraint_audit_{row['condition']}_violations",
            "value": f"{int(row['n_violations'])} / {int(row['n_basins_total'])}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_k1_constraint_audit_summary.csv",
            "column_or_key": "n_violations",
            "results_section": "Results 3.4",
            "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py",
            "upstream_source": "sec34_equivalence_set_dispersion + sec31_uncertainty_bandwidth",
            "derivation_type": "RECOMPUTED",
            "notes": f"violation_fraction={row['violation_fraction']:.1%}, ratio_median={row['ratio_range_over_delta_median']:.4f}, ratio_p90={row['ratio_range_over_delta_p90']:.4f}",
        })

    k3_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_k3_single_source_recomputation.csv")
    for _, row in k3_df.iterrows():
        records.append({
            "quantity_name": f"k3_aligned_recompute_range_{row['condition']}_median",
            "value": f"{row['recomputed_range_median']:.4f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_k3_single_source_recomputation.csv",
            "column_or_key": "recomputed_range_median",
            "results_section": "Results 3.4",
            "generating_script": "sec34_determinacy_and_bounds/scripts/sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py",
            "upstream_source": "raw sufficient stats + OOB cube (aligned)",
            "derivation_type": "RECOMPUTED",
            "notes": f"existing={row['existing_range_median']:.4f}, abs diff={row['range_abs_diff']:.4f}, violations={int(row['recomputed_violations_count'])}, max excess={row['recomputed_max_excess_range_minus_delta']:.3e}",
        })

    # -------------------------------------------------------------------------
    # Sec SI: Scope
    # -------------------------------------------------------------------------
    si_ov = pd.read_csv(BASE / "sec_si_scope" / "tables" / "sec_si_cross_metric_overlap.csv").set_index(["paradigm", "metric"])
    records.append({
        "quantity_name": "cross_metric_overlap_dpl_logkge_vs_kge",
        "value": f"{si_ov.loc[('dPL', 'logKGE'), 'top3_overlap_with_KGE']:.4f}",
        "file_path": "sec_si_scope/tables/sec_si_cross_metric_overlap.csv",
        "column_or_key": "top3_overlap_with_KGE (dPL, logKGE)",
        "results_section": "Supplementary Information",
        "generating_script": "sec_si_scope/scripts/si_01_cross_metric_overlap.py",
        "upstream_source": "chapter4_robustness_supplement_20260925/tables/rb2_cross_metric_overlap.csv",
        "derivation_type": "LOADED",
        "notes": "0.2147; evaluation-only, no target-matched retraining",
    })
    records.append({
        "quantity_name": "cross_metric_overlap_dpl_fhv_vs_kge",
        "value": f"{si_ov.loc[('dPL', 'FHV'), 'top3_overlap_with_KGE']:.4f}",
        "file_path": "sec_si_scope/tables/sec_si_cross_metric_overlap.csv",
        "column_or_key": "top3_overlap_with_KGE (dPL, FHV)",
        "results_section": "Supplementary Information",
        "generating_script": "sec_si_scope/scripts/si_01_cross_metric_overlap.py",
        "upstream_source": "chapter4_robustness_supplement_20260925/tables/rb2_cross_metric_overlap.csv",
        "derivation_type": "LOADED",
        "notes": "0.4325; evaluation-only, no target-matched retraining",
    })

    si_p = pd.read_csv(BASE / "sec_si_scope" / "tables" / "sec_si_equivalence_protocol_params.csv")
    records.append({
        "quantity_name": "equivalence_bootstrap_replicates",
        "value": str(int(si_p.loc[0, "n_replicates"])),
        "file_path": "sec_si_scope/tables/sec_si_equivalence_protocol_params.csv",
        "column_or_key": "n_replicates",
        "results_section": "Supplementary Information",
        "generating_script": "sec_si_scope/scripts/si_02_equivalence_protocol_params.py",
        "upstream_source": "chapter4_equivalence_recommendability_20260925/scripts/s2_equivalence.py",
        "derivation_type": "LOADED",
        "notes": "1000 water-year block draws, seed 20260925, q05 cutoff",
    })

    # -------------------------------------------------------------------------
    # Unit: _verification (V1 Abstract Numbers Independent Recomputation)
    # -------------------------------------------------------------------------
    v1_df = pd.read_csv(BASE / "_verification" / "tables" / "v1_recomputed_abstract_numbers.csv")
    for _, r in v1_df.iterrows():
        records.append({
            "quantity_name": f"v1_recomputed_{r['quantity_name']}",
            "value": f"{r['recomputed_value']}",
            "file_path": "_verification/tables/v1_recomputed_abstract_numbers.csv",
            "column_or_key": "recomputed_value",
            "results_section": "Independent Verification",
            "generating_script": "_verification/scripts/v1_recompute_abstract_numbers.py",
            "upstream_source": "primitive raw inputs (water_year_sufficient_stats, Qsim, obs)",
            "derivation_type": "RECOMPUTED",
            "notes": f"abs diff vs frozen = {r['abs_diff']:.3e} ({r['optimality_evidence']})",
        })

    # -------------------------------------------------------------------------
    # A1/A2/A3/A4/A5 audit registration: annotate affected rows and add recompute rows
    # -------------------------------------------------------------------------
    a1_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a1_static_basin_indexing_audit.csv")
    a2_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a2_table_alignment_audit.csv")
    a2_ext = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a2_extended_checks.csv")
    a3_pen = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a3_parsimony_penalty_idjoin_summary.csv")
    a3_disp = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a3_dispersion_idjoin_summary.csv")
    a4_sum = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a4_attribute_screening_summary.csv")
    a5_sum = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a5_remaining_summary.csv")
    a5_cmp = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_a5_remaining_comparison.csv")
    btn_df = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_bottleneck_idjoin_check.csv")
    sing_tot = pd.read_csv(BASE / "sec34_determinacy_and_bounds" / "tables" / "sec34_singleton_totals_idjoin_check.csv")

    # --- A. Annotate affected existing rows ---
    AUDIT_SCRIPT = "sec34_determinacy_and_bounds/scripts/sec34_11_audit_basin_alignment_a1_a2.py"
    A3_SCRIPT = "sec34_determinacy_and_bounds/scripts/sec34_15_a3_recompute_parsimony_dispersion_idjoin.py"
    A4_SCRIPT = "sec34_determinacy_and_bounds/scripts/sec34_13_attribute_screening_idjoin_a4.py"
    A5_SCRIPT = "sec34_determinacy_and_bounds/scripts/sec34_16_a5_remaining_idjoin.py"
    BTN_SCRIPT = "sec34_determinacy_and_bounds/scripts/sec34_14_bottleneck_singleton_idjoin.py"
    EXT_SCRIPT = "sec34_determinacy_and_bounds/scripts/sec34_12_basin_audit_supplement.py"

    SUPERSEDED_PATTERNS = {
        "parsimony_penalty_mean_diff_seen-dPL": "SUPERSEDED (basin-order index misalignment in sec34_06; ID-joined recompute in sec34_a3_parsimony_penalty_idjoin_summary.csv: +0.011667)",
        "parsimony_penalty_mean_diff_OOB-dPL": "SUPERSEDED (basin-order index misalignment in sec34_06; ID-joined recompute in sec34_a3_parsimony_penalty_idjoin_summary.csv: +0.006157)",
    }
    for cond in ["seen-dPL", "OOB-dPL"]:
        for metric in ["point_kge_range", "point_kge_iqr", "best_minus_worst_diff"]:
            SUPERSEDED_PATTERNS[f"equiv_dispersion_{cond}_{metric}_median"] = (
                f"SUPERSEDED (basin-order index misalignment in sec34_09; ID-joined recompute in sec34_a3_dispersion_idjoin_summary.csv)")

    REVERIFIED_PATTERNS = {
        "parsimony_penalty_mean_diff_IC": "RE-VERIFIED by ID-joined recompute (sec34_a3_parsimony_penalty_idjoin_summary.csv); IC was unaffected by the misalignment",
        "basin_attribute_screening_min_q_value": "RE-VERIFIED by ID-joined recompute (sec34_13_attribute_screening_idjoin_a4.py; abs diff 0.0 exact)",
        "cluster_oob_top3_coverage_diff": "RE-VERIFIED by ID-joined recompute (sec34_16_a5_remaining_idjoin.py)",
        "cluster_oob_top5_regret_diff_mean": "RE-VERIFIED by ID-joined recompute (sec34_16_a5_remaining_idjoin.py)",
        "top5pct_basins_share_of_total_delta_kge": "RE-VERIFIED by ID-joined recompute (sec34_16_a5_remaining_idjoin.py; abs diff 4.7e-08)",
    }
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        for metric in ["point_kge_range", "point_kge_iqr", "equivalence_set_size", "best_minus_worst_diff"]:
            if cond == "IC":
                REVERIFIED_PATTERNS[f"equiv_dispersion_IC_{metric}_median"] = (
                    "RE-VERIFIED by ID-joined recompute (sec34_a3_dispersion_idjoin_summary.csv); IC was unaffected by the misalignment")
    for cond in ["seen-dPL", "OOB-dPL"]:
        REVERIFIED_PATTERNS[f"equiv_dispersion_{cond}_equivalence_set_size_median"] = (
            "RE-VERIFIED by ID-joined recompute (sec34_a3_dispersion_idjoin_summary.csv; abs diff 0.0): "
            "set size depends only on the equivalence set, which was already ID-keyed, so it was unaffected by the misalignment")
        REVERIFIED_PATTERNS[f"bottleneck_forcing_models_{cond}_100pct"] = (
            "RE-VERIFIED by ID-joined recompute (sec34_14_bottleneck_singleton_idjoin.py; uncovered basins and forcing models match exactly)")
    for k in [1, 3, 5, 10]:
        REVERIFIED_PATTERNS[f"seen_topk_to_oob_top1_coverage_k{k}"] = "RE-VERIFIED by ID-joined recompute (sec34_16_a5_remaining_idjoin.py)"
    for k in [1, 3, 5]:
        REVERIFIED_PATTERNS[f"candidate_set_median_regret_k{k}"] = "RE-VERIFIED by ID-joined recompute (sec34_16_a5_remaining_idjoin.py)"
    for k in [1, 2, 3, 4, 5, 7, 10]:
        REVERIFIED_PATTERNS[f"t3_2_fixed_ensemble_k{k}_median_kge"] = (
            "RE-VERIFIED by per-basin daily ensemble recompute (sec34_12_basin_audit_supplement.py; all k PASS)")
    for fam_k in ["A_k1", "B_k1", "B_k2", "B_k3", "B_k4", "B_k5", "B_k7", "B_k10",
                  "C_k2", "C_k3", "C_k5", "D_k3", "E_k36"]:
        REVERIFIED_PATTERNS[f"rules_ladder_{fam_k}_median"] = (
            "RE-VERIFIED by per-basin daily ensemble recompute (sec34_12_basin_audit_supplement.py; all rules PASS)")
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        for k in [0, 1, 2, 3, 4]:
            for attr in ["basin_count", "lat_median", "lon_median", "arid_median", "f_snow_median", "runoff_ratio_median"]:
                REVERIFIED_PATTERNS[f"fold_{k}_{attr}"] = "RE-VERIFIED by ID-joined recompute (sec34_16_a5_remaining_idjoin.py; max abs diff 5.0e-07)"
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        for scheme in ["1yr_block_bootstrap_primary", "3yr_multiyear_block_bootstrap_c1"]:
            REVERIFIED_PATTERNS[f"uncertainty_bandwidth_{cond}_{scheme}_median"] = (
                "RE-VERIFIED by single-source recompute (sec34_10; abs diff 0.0); note: the per-basin OOB-dPL basin_id labels in "
                "sec31_uncertainty_bandwidth_per_basin.csv are superseded (basin-order index misalignment in sec31_09)")
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        for pct in ["90%", "95%", "100%"]:
            REVERIFIED_PATTERNS[f"multiyear_3yr_block_boot_cover_{cond}_{pct}"] = (
                "RE-VERIFIED: set-cover counts are permutation-invariant (sec34_a1 audit; K3 recompute); "
                "per-basin OOB-dPL basin_id labels in the C1 equivalence-sets table are superseded (sec31_08 misalignment)")
    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        for pct in ["90pct", "95pct", "100pct"]:
            REVERIFIED_PATTERNS[f"minimal_cover_{cond}_target_{pct}"] = (
                "RE-VERIFIED by ID-joined recompute (sec34_14_bottleneck_singleton_idjoin.py and _verification/V1; exact match)")
    for k in ["IC", "seen-dPL", "OOB-dPL"]:
        REVERIFIED_PATTERNS[f"k1_constraint_audit_{k}_violations"] = "RE-VERIFIED audit artifact (sec34_10 diagnostic; recomputed in sec34_15 A3)"
        REVERIFIED_PATTERNS[f"k3_aligned_recompute_range_{k}_median"] = "RE-VERIFIED single-source recompute (sec34_10; reproduced in sec34_15 A3)"

    for rec in records:
        qn = rec["quantity_name"]
        if qn in SUPERSEDED_PATTERNS:
            rec["derivation_type"] = "SUPERSEDED"
            rec["notes"] = f"{rec['notes']} | {SUPERSEDED_PATTERNS[qn]}"
        elif qn in REVERIFIED_PATTERNS:
            rec["derivation_type"] = "RE-VERIFIED"
            rec["notes"] = f"{rec['notes']} | {REVERIFIED_PATTERNS[qn]}"

    # --- B. Add new audit/recompute rows ---
    n_a1_total = len(a1_df)
    n_a1_mis = int((a1_df.verdict == "MISALIGNED").sum())
    records.append({
        "quantity_name": "a1_static_audit_positions_total",
        "value": str(n_a1_total),
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_a1_static_basin_indexing_audit.csv",
        "column_or_key": "n_rows",
        "results_section": "Independent Verification",
        "generating_script": AUDIT_SCRIPT,
        "upstream_source": "manuscript analysis tree scripts + loaded upstream modules",
        "derivation_type": "RECOMPUTED",
        "notes": f"{n_a1_mis} MISALIGNED, {n_a1_total - n_a1_mis} ALIGNED",
    })
    records.append({
        "quantity_name": "a1_static_audit_misaligned_positions",
        "value": str(n_a1_mis),
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_a1_static_basin_indexing_audit.csv",
        "column_or_key": "verdict == MISALIGNED",
        "results_section": "Independent Verification",
        "generating_script": AUDIT_SCRIPT,
        "upstream_source": "manuscript analysis tree scripts + loaded upstream modules",
        "derivation_type": "RECOMPUTED",
        "notes": "sec31_07:124, sec31_08:158, sec31_09:128, sec34_06:92, sec34_09:75 (unsorted canonical_ids index into sorted matrices)",
    })
    records.append({
        "quantity_name": "a2_tables_checked_primary",
        "value": f"{len(a2_df)} ({int((a2_df.alignment_status == 'PASS').sum())} PASS / {int((a2_df.alignment_status != 'PASS').sum())} FAIL)",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_a2_table_alignment_audit.csv",
        "column_or_key": "alignment_status",
        "results_section": "Independent Verification",
        "generating_script": AUDIT_SCRIPT,
        "upstream_source": "raw sufficient stats + OOB Qsim/obs + pub table (basin-ID joins)",
        "derivation_type": "RECOMPUTED",
        "notes": f"total rows checked {int(a2_df.n_rows_checked.sum())}; mismatched tables: sec31_uncertainty_bandwidth_per_basin, sec34_equivalence_set_dispersion_per_basin, sec34_parsimony_penalty_per_basin",
    })
    records.append({
        "quantity_name": "a2_extended_checks_pass",
        "value": f"{int((a2_ext.alignment_status == 'PASS').sum())} / {len(a2_ext)}",
        "file_path": "sec34_determinacy_and_bounds/tables/sec34_a2_extended_checks.csv",
        "column_or_key": "alignment_status",
        "results_section": "Independent Verification",
        "generating_script": EXT_SCRIPT,
        "upstream_source": "raw OOB Qsim/obs + E8 rankings + pub table (basin-ID joins)",
        "derivation_type": "RECOMPUTED",
        "notes": f"rules-ladder per-basin basis (fixed ensemble all k, usage rules A/B3/B5/C/D/E) + screening per-basin penalty all PASS; {int(a2_ext.n_rows_checked.sum())} rows checked",
    })
    for _, r in a3_pen.iterrows():
        records.append({
            "quantity_name": f"a3_idjoin_parsimony_mean_diff_{r['condition']}",
            "value": f"{r['mean_diff_kge']:+.6f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_a3_parsimony_penalty_idjoin_summary.csv",
            "column_or_key": "mean_diff_kge",
            "results_section": "Independent Verification",
            "generating_script": A3_SCRIPT,
            "upstream_source": "t1_equivalence_sets.csv + performance matrices (basin-ID joins)",
            "derivation_type": "RECOMPUTED",
            "notes": f"median={r['median_diff_kge']:+.6f}, wilcoxon p={r['wilcoxon_p']:.4e}, coincided={int(r['n_coincided'])}/531, ties={int(r['n_ties_broken_min_params'])}",
        })
    for _, r in a3_disp[a3_disp.metric == "point_kge_range"].iterrows():
        records.append({
            "quantity_name": f"a3_idjoin_dispersion_range_median_{r['condition']}",
            "value": f"{r['median']:.4f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_a3_dispersion_idjoin_summary.csv",
            "column_or_key": "median (point_kge_range)",
            "results_section": "Independent Verification",
            "generating_script": A3_SCRIPT,
            "upstream_source": "t1_equivalence_sets.csv + performance matrices (basin-ID joins)",
            "derivation_type": "RECOMPUTED",
            "notes": f"Q25={r['q25']:.4f}, Q75={r['q75']:.4f}, P90={r['p90']:.4f}",
        })
    for _, r in a4_sum.iterrows():
        records.append({
            "quantity_name": f"a4_idjoin_{r['metric']}",
            "value": f"{r['recomputed']:.6f}",
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_a4_attribute_screening_summary.csv",
            "column_or_key": "recomputed",
            "results_section": "Independent Verification",
            "generating_script": A4_SCRIPT,
            "upstream_source": "pub table + caravan_671_attributes + OOB obs (basin-ID joins)",
            "derivation_type": "RECOMPUTED",
            "notes": f"existing={r['existing']:.6f}, abs diff={r['abs_diff']:.3e}; N_PERM=10000, seed=20260930",
        })
    for _, r in a5_cmp.iterrows():
        if pd.isna(r["abs_diff"]) or r["sub_item"] in ("unmatched_id_set",):
            continue
        records.append({
            "quantity_name": f"a5_idjoin_{r['item']}_{r['sub_item']}",
            "value": f"{r['new_value']:.6f}" if isinstance(r["new_value"], (int, float)) and not isinstance(r["new_value"], bool) else str(r["new_value"]),
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_a5_remaining_comparison.csv",
            "column_or_key": "new_value",
            "results_section": "Independent Verification",
            "generating_script": A5_SCRIPT,
            "upstream_source": "raw upstream tables (basin-ID joins)",
            "derivation_type": "RECOMPUTED",
            "notes": f"existing={r['old_value']}, abs diff={r['abs_diff']:.3e}",
        })
    for _, r in btn_df.iterrows():
        records.append({
            "quantity_name": f"bottleneck_uncovered_ids_idjoin_{r['condition']}",
            "value": str(r["uncovered_basin_ids_idjoin"]),
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_bottleneck_idjoin_check.csv",
            "column_or_key": "uncovered_basin_ids_idjoin",
            "results_section": "Independent Verification",
            "generating_script": BTN_SCRIPT,
            "upstream_source": "t1_equivalence_sets.csv (basin-ID joins) + HiGHS MILP",
            "derivation_type": "RECOMPUTED",
            "notes": f"forcing models {r['bottleneck_forcing_models_idjoin']}; matches existing: ids={r['basin_ids_match']}, models={r['forcing_models_match']}",
        })
    for _, r in sing_tot.iterrows():
        records.append({
            "quantity_name": f"singleton_total_idjoin_{r['condition']}",
            "value": str(int(r["total_singletons_idjoin"])),
            "file_path": "sec34_determinacy_and_bounds/tables/sec34_singleton_totals_idjoin_check.csv",
            "column_or_key": "total_singletons_idjoin",
            "results_section": "Independent Verification",
            "generating_script": BTN_SCRIPT,
            "upstream_source": "t1_equivalence_sets.csv (basin-ID joins)",
            "derivation_type": "RECOMPUTED",
            "notes": f"existing={int(r['total_singletons_existing'])}, abs diff={int(r['abs_diff'])}",
        })

    # -------------------------------------------------------------------------
    # POST-FREEZE family: sec35 cluster-stratified analyses (G1-G4)
    # -------------------------------------------------------------------------
    SEC35 = BASE / "sec35_cluster_stratified" / "tables"
    S35_SCRIPT = {
        "g1": "sec35_cluster_stratified/scripts/sec35_01_penalty_by_type_g1.py",
        "g2": "sec35_cluster_stratified/scripts/sec35_02_worst_basins_by_type_g2.py",
        "g3": "sec35_cluster_stratified/scripts/sec35_03_equivalence_library_by_type_g3.py",
        "g4": "sec35_cluster_stratified/scripts/sec35_04_parsimony_by_type_g4.py",
    }
    PF_NOTE = "POST-FREEZE: analysis family added after the freeze; external basin type used strictly as a grouping factor"

    g1s = pd.read_csv(SEC35 / "sec35_g1_matched_summary.csv").iloc[0]
    records.append({
        "quantity_name": "pf_g1_matched_unmatched_counts",
        "value": f"{int(g1s['n_matched'])} matched / {int(g1s['n_unmatched'])} unmatched",
        "file_path": "sec35_cluster_stratified/tables/sec35_g1_matched_summary.csv",
        "column_or_key": "n_matched / n_unmatched",
        "results_section": "POST-FREEZE G1",
        "generating_script": S35_SCRIPT["g1"],
        "upstream_source": "data/gauge_cluster.csv + pub_same_pipeline_model_basin.csv (basin-ID joins)",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF_NOTE}; unmatched ids {g1s['unmatched_basin_ids']}",
    })
    g1k = pd.read_csv(SEC35 / "sec35_g1_kruskal_wallis.csv", dtype={"p_value_scientific": str}).iloc[0]
    records.append({
        "quantity_name": "pf_g1_kruskal_wallis",
        "value": f"H={g1k['statistic']:.4f}, p={g1k['p_value_scientific']}",
        "file_path": "sec35_cluster_stratified/tables/sec35_g1_kruskal_wallis.csv",
        "column_or_key": "statistic / p_value",
        "results_section": "POST-FREEZE G1",
        "generating_script": S35_SCRIPT["g1"],
        "upstream_source": "ID-joined penalty_median_pub",
        "derivation_type": "POST-FREEZE",
        "notes": f"df={int(g1k['df'])}, {g1k['effect_size_name']}={g1k['effect_size_value']:.4f}; {g1k['effect_size_definition']}",
    })
    g1t = pd.read_csv(SEC35 / "sec35_g1_penalty_by_type.csv")
    for _, r in g1t.iterrows():
        records.append({
            "quantity_name": f"pf_g1_penalty_median_type{int(r['type'])}",
            "value": f"{r['median']:.6f}",
            "file_path": "sec35_cluster_stratified/tables/sec35_g1_penalty_by_type.csv",
            "column_or_key": "median",
            "results_section": "POST-FREEZE G1",
            "generating_script": S35_SCRIPT["g1"],
            "upstream_source": "ID-joined penalty_median_pub",
            "derivation_type": "POST-FREEZE",
            "notes": f"{PF_NOTE}; n={int(r['n'])}, Q25={r['q25']:.6f}, Q75={r['q75']:.6f}",
        })
    g1p = pd.read_csv(SEC35 / "sec35_g1_pairwise_dunn_holm.csv")
    records.append({
        "quantity_name": "pf_g1_pairwise_dunn_holm",
        "value": f"{len(g1p)} pairs; {int(g1p['reject_005_holm'].sum())} with Holm p<0.05",
        "file_path": "sec35_cluster_stratified/tables/sec35_g1_pairwise_dunn_holm.csv",
        "column_or_key": "n_rows / reject_005_holm.sum()",
        "results_section": "POST-FREEZE G1",
        "generating_script": S35_SCRIPT["g1"],
        "upstream_source": "ID-joined penalty_median_pub",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF_NOTE}; Dunn z-test on average ranks with tie correction; Holm correction over the full family",
    })
    g1b = pd.read_csv(SEC35 / "sec35_g1b_attribute_penalty_spearman_by_type.csv")
    records.append({
        "quantity_name": "pf_g1b_attribute_choice",
        "value": f"{g1b['attribute'].iloc[0]} (global q={float(g1b['attribute_global_bh_fdr_q'].iloc[0]):.6f})",
        "file_path": "sec35_cluster_stratified/tables/sec35_g1b_attribute_penalty_spearman_by_type.csv",
        "column_or_key": "attribute / attribute_global_bh_fdr_q",
        "results_section": "POST-FREEZE G1",
        "generating_script": S35_SCRIPT["g1"],
        "upstream_source": "sec34_a4_attribute_screening_recompute.csv",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF_NOTE}; descriptive only, no significance test and no multiplicity correction",
    })
    for _, r in g1b.iterrows():
        records.append({
            "quantity_name": f"pf_g1b_spearman_type{int(r['type'])}",
            "value": f"{r['spearman_rho']:+.6f}",
            "file_path": "sec35_cluster_stratified/tables/sec35_g1b_attribute_penalty_spearman_by_type.csv",
            "column_or_key": "spearman_rho",
            "results_section": "POST-FREEZE G1",
            "generating_script": S35_SCRIPT["g1"],
            "upstream_source": "caravan attributes + ID-joined penalty_median_pub",
            "derivation_type": "POST-FREEZE",
            "notes": f"{PF_NOTE}; n={int(r['n'])}; descriptive only (no test, no correction)",
        })
    g2 = pd.read_csv(SEC35 / "sec35_g2_fisher_exact.csv").iloc[0]
    records.append({
        "quantity_name": "pf_g2_worst26_matched_count",
        "value": f"{int(g2['n_top_matched'])} / {int(g2['n_top_defined'])}",
        "file_path": "sec35_cluster_stratified/tables/sec35_g2_fisher_exact.csv",
        "column_or_key": "n_top_matched",
        "results_section": "POST-FREEZE G2",
        "generating_script": S35_SCRIPT["g2"],
        "upstream_source": "pub_same_pipeline_model_basin.csv (basin sums, unchanged top-5% rule)",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF_NOTE}; {int(g2['n_top_unmatched'])} of the 26 are type-unmatched",
    })
    records.append({
        "quantity_name": "pf_g2_fisher_exact_p",
        "value": f"{float(g2['p_value_exact_monte_carlo']):.6f}",
        "file_path": "sec35_cluster_stratified/tables/sec35_g2_fisher_exact.csv",
        "column_or_key": "p_value_exact_monte_carlo",
        "results_section": "POST-FREEZE G2",
        "generating_script": S35_SCRIPT["g2"],
        "upstream_source": "2 x 10 contingency table over matched basins",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF_NOTE}; Monte-Carlo multivariate-hypergeometric exact test, {int(g2['n_draws'])} draws, seed {int(g2['seed'])}, se={float(g2['monte_carlo_standard_error']):.2e}; chi2 p={float(g2['p_value_chi2_asymptotic']):.6f}",
    })
    g3k = pd.read_csv(SEC35 / "sec35_g3_equivalence_size_kruskal.csv", dtype={"p_value_scientific": str})
    for _, r in g3k.iterrows():
        records.append({
            "quantity_name": f"pf_g3_equivalence_size_kw_{r['condition']}",
            "value": f"H={r['statistic']:.4f}, p={r['p_value_scientific']}",
            "file_path": "sec35_cluster_stratified/tables/sec35_g3_equivalence_size_kruskal.csv",
            "column_or_key": "statistic / p_value",
            "results_section": "POST-FREEZE G3",
            "generating_script": S35_SCRIPT["g3"],
            "upstream_source": "t1_equivalence_sets.csv (basin-ID joins)",
            "derivation_type": "POST-FREEZE",
            "notes": f"{PF_NOTE}; df={int(r['df'])}, epsilon^2={r['effect_size_value']:.4f}, n={int(r['n_total'])}",
        })
    g3c = pd.read_csv(SEC35 / "sec35_g3_min_cover_by_type.csv")
    for _, r in g3c[g3c.target_pct == "95%"].iterrows():
        records.append({
            "quantity_name": f"pf_g3_min_cover_95_{r['condition']}_type{int(r['type'])}",
            "value": str(int(r["min_models_exact"])),
            "file_path": "sec35_cluster_stratified/tables/sec35_g3_min_cover_by_type.csv",
            "column_or_key": "min_models_exact",
            "results_section": "POST-FREEZE G3",
            "generating_script": S35_SCRIPT["g3"],
            "upstream_source": "t1_equivalence_sets.csv (basin-ID joins) + HiGHS MILP",
            "derivation_type": "POST-FREEZE",
            "notes": f"{PF_NOTE}; n_type={int(r['n_type_basins'])}, need={int(r['need_basins'])}, budget-1 covers {int(r['max_basins_covered_budget_minus_1'])} (proves K*-1 impossible: {r['proves_k_impossible']})",
        })
    records.append({
        "quantity_name": "pf_g3_optimality_certificates",
        "value": f"{int(g3c['proves_k_impossible'].sum())} / {len(g3c)}",
        "file_path": "sec35_cluster_stratified/tables/sec35_g3_min_cover_by_type.csv",
        "column_or_key": "proves_k_impossible.sum()",
        "results_section": "POST-FREEZE G3",
        "generating_script": S35_SCRIPT["g3"],
        "upstream_source": "t1_equivalence_sets.csv + HiGHS MILP",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF_NOTE}; solver status 0 in {int((g3c['solver_status'] == 0).sum())}/{len(g3c)} cells",
    })
    g4k = pd.read_csv(SEC35 / "sec35_g4_parsimony_kruskal.csv", dtype={"p_value_scientific": str})
    for _, r in g4k.iterrows():
        records.append({
            "quantity_name": f"pf_g4_parsimony_kw_{r['condition']}",
            "value": f"H={r['statistic']:.4f}, p={r['p_value_scientific']}",
            "file_path": "sec35_cluster_stratified/tables/sec35_g4_parsimony_kruskal.csv",
            "column_or_key": "statistic / p_value",
            "results_section": "POST-FREEZE G4",
            "generating_script": S35_SCRIPT["g4"],
            "upstream_source": "sec34_a3_parsimony_penalty_idjoin_per_basin.csv (ID-joined; SUPERSEDED table not used)",
            "derivation_type": "POST-FREEZE",
            "notes": f"{PF_NOTE}; df={int(r['df'])}, epsilon^2={r['effect_size_value']:.4f}, n={int(r['n_total'])}",
        })

    # -------------------------------------------------------------------------
    # POST-FREEZE family: sec36 size-matched null (H1) and absolute type covers (H2)
    # -------------------------------------------------------------------------
    SEC36 = BASE / "sec36_size_matched_null" / "tables"
    H1_SCRIPT = "sec36_size_matched_null/scripts/sec36_01_size_matched_null_h1.py"
    H2_SCRIPT = "sec36_size_matched_null/scripts/sec36_02_type_cover_absolute_h2.py"
    PF36 = "POST-FREEZE: analysis added after the freeze (sec36); external basin type used strictly as a grouping factor"

    h1 = pd.read_csv(SEC36 / "sec36_h1_size_matched_null_summary.csv", dtype={"direction_definition": str})
    h1_main = h1[h1.tier == "main_95"]
    for _, r in h1_main.iterrows():
        records.append({
            "quantity_name": f"pf_h1_null_95_{r['condition']}_type{int(r['type'])}",
            "value": f"obs={int(r['observed_cover'])}; null median={r['null_median']:.1f}; p_le={r['one_sided_empirical_percentile']:.3f}",
            "file_path": "sec36_size_matched_null/tables/sec36_h1_size_matched_null_summary.csv",
            "column_or_key": "observed_cover / null_median / one_sided_empirical_percentile",
            "results_section": "POST-FREEZE H1",
            "generating_script": H1_SCRIPT,
            "upstream_source": "t1_equivalence_sets.csv (frozen) + 515-basin matched pool (basin-ID joins) + HiGHS MILP",
            "derivation_type": "POST-FREEZE",
            "notes": f"{PF36}; n_type={int(r['n_type'])}, need={int(r['need_basins'])}, R={int(r['R'])}, seed={int(r['seed'])}, "
                     f"P5={r['null_p5']:.1f}, P95={r['null_p95']:.1f}, nonoptimal={int(r['n_nonoptimal_solves'])}; direction: p_null_le_observed",
        })
    h1_meta = pd.read_csv(SEC36 / "sec36_h1_run_meta.csv").iloc[0]
    records.append({
        "quantity_name": "pf_h1_run_meta",
        "value": f"main R={int(h1_meta['main_R'])}; secondary R={int(h1_meta['secondary_R'])}; nonoptimal={int(h1_meta['nonoptimal_solves'])}",
        "file_path": "sec36_size_matched_null/tables/sec36_h1_run_meta.csv",
        "column_or_key": "main_R / secondary_R / nonoptimal_solves",
        "results_section": "POST-FREEZE H1",
        "generating_script": H1_SCRIPT,
        "upstream_source": "515-basin matched pool",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF36}; seed={int(h1_meta['seed'])} (secondary {int(h1_meta['seed_secondary'])}), "
                 f"{int(h1_meta['total_milp_solves'])} exact MILP solves; {h1_meta['direction_definition']}",
    })
    h2 = pd.read_csv(SEC36 / "sec36_h2_type_cover_absolute.csv")
    h2c = pd.read_csv(SEC36 / "sec36_h2_vs_g3_comparison.csv").iloc[0]
    records.append({
        "quantity_name": "pf_h2_cells_and_certificates",
        "value": f"{len(h2)} cells; {int(h2.proves_k_minus_1_insufficient.sum())} certificates",
        "file_path": "sec36_size_matched_null/tables/sec36_h2_type_cover_absolute.csv",
        "column_or_key": "n_rows / proves_k_minus_1_insufficient.sum()",
        "results_section": "POST-FREEZE H2",
        "generating_script": H2_SCRIPT,
        "upstream_source": "t1_equivalence_sets.csv (frozen) + HiGHS MILP",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF36}; absolute minimal cover counts for 3 conditions x 10 types x 90/95/100%; "
                 f"solver status 0 in {int((h2.solver_status == 0).sum())}/{len(h2)}",
    })
    records.append({
        "quantity_name": "pf_h2_vs_g3_max_abs_diff",
        "value": f"{float(h2c['max_abs_diff_min_models']):.3e}",
        "file_path": "sec36_size_matched_null/tables/sec36_h2_vs_g3_comparison.csv",
        "column_or_key": "max_abs_diff_min_models",
        "results_section": "POST-FREEZE H2",
        "generating_script": H2_SCRIPT,
        "upstream_source": "sec35_g3_min_cover_by_type.csv",
        "derivation_type": "POST-FREEZE",
        "notes": f"{PF36}; {int(h2c['n_rows_compared'])} rows compared, {int(h2c['n_rows_with_any_diff'])} with any difference; "
                 f"certificate max abs diff {float(h2c['max_abs_diff_certificate']):.3e}",
    })

    # -------------------------------------------------------------------------
    # F1/F2/F3 data-side final round
    # -------------------------------------------------------------------------
    SEC33T = BASE / "sec33_usage_and_conditioning" / "tables"
    f1_pb = pd.read_csv(SEC33T / "sec33_rule_c_k2_k5_per_basin.csv")
    f1_chk = pd.read_csv(SEC33T / "sec33_rule_c_k2_k5_alignment_check.csv", dtype={"max_abs_diff_sci": str})
    f1_cmp = pd.read_csv(SEC33T / "sec33_rule_c_k2_k5_recompute_comparison.csv")
    f1_cmp_sum = pd.read_csv(SEC33T / "sec33_rule_c_k2_k5_comparison_summary.csv", dtype={"max_abs_diff_sci": str})
    records.append({
        "quantity_name": "f1_rule_c_k2_k5_per_basin_rows",
        "value": str(len(f1_pb)),
        "file_path": "sec33_usage_and_conditioning/tables/sec33_rule_c_k2_k5_per_basin.csv",
        "column_or_key": "n_rows",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_10_rule_c_k2_k5_per_basin_f1.py",
        "upstream_source": "E8 sec33_recommender_full_rankings.npz + OOB Qsim/obs daily (basin-ID joins)",
        "derivation_type": "RECOMPUTED",
        "notes": f"10 seeds x 531 basins x k in {{2,5}}; alignment check rows {int(f1_chk.n_rows_checked.sum())}, mismatches {int(f1_chk.n_mismatches.sum())}, max abs diff {f1_chk.max_abs_diff_sci.max()}",
    })
    records.append({
        "quantity_name": "f1_rule_c_k2_k5_ladder_comparison_max_abs_diff",
        "value": str(f1_cmp_sum.max_abs_diff_sci.max()),
        "file_path": "sec33_usage_and_conditioning/tables/sec33_rule_c_k2_k5_comparison_summary.csv",
        "column_or_key": "max_abs_diff_sci",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_10_rule_c_k2_k5_per_basin_f1.py",
        "upstream_source": "sec33_rule_c_per_seed_ladder.csv + sec33_rules_ladder_unified.csv (6-dp storage)",
        "derivation_type": "RE-VERIFIED",
        "notes": f"{len(f1_cmp)} quantity rows compared (14 quantities x 10 seeds + 6 x 10-seed means); {int(f1_cmp_sum.n_rows_agree_within_1e6.sum())}/{int(f1_cmp_sum.n_rows.sum())} rows agree within 1e-6 (the 6-dp write precision of the existing tables)",
    })

    f2_sum = pd.read_csv(SEC33T / "sec33_constrained_ensemble_summary.csv")
    for _, r in f2_sum.iterrows():
        records.append({
            "quantity_name": f"f2_constrained_{r['tier']}_k{int(r['k'])}_mean_diff",
            "value": f"{r['mean_diff']:+.6f}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_constrained_ensemble_summary.csv",
            "column_or_key": "mean_diff",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_11_constrained_fixed_ensemble_f2.py",
            "upstream_source": "sec32_model_structural_attributes.csv + OOB Qsim/obs daily (basin-ID joins)",
            "derivation_type": "RECOMPUTED",
            "notes": f"threshold n_parameters <= {r['threshold_n_parameters']:.2f}, eligible {int(r['n_eligible_models'])} models; median diff {r['median_diff']:+.6f}, wilcoxon p {float(r['wilcoxon_p_scientific']):.3e}, identical members all folds={r['identical_ensemble_all_folds']}",
        })
    f2_mem = pd.read_csv(SEC33T / "sec33_constrained_ensemble_members.csv")
    deg = f2_mem.groupby(["tier", "k"])["members_identical"].sum()
    records.append({
        "quantity_name": "f2_degenerate_tier_k_cells",
        "value": str(int(deg.sum())),
        "file_path": "sec33_usage_and_conditioning/tables/sec33_constrained_ensemble_members.csv",
        "column_or_key": "members_identical.sum()",
        "results_section": "Results 3.3",
        "generating_script": "sec33_usage_and_conditioning/scripts/sec33_11_constrained_fixed_ensemble_f2.py",
        "upstream_source": "sec32_model_structural_attributes.csv",
        "derivation_type": "RECOMPUTED",
        "notes": f"out of {len(deg)} tier x k cells x 5 folds = {len(f2_mem)} fold cells; no degenerate tier",
    })
    for (tier, k), g in f2_mem.groupby(["tier", "k"]):
        records.append({
            "quantity_name": f"f2_params_saved_{tier}_k{int(k)}",
            "value": f"{int(g.params_saved_sum.min())}-{int(g.params_saved_sum.max())}",
            "file_path": "sec33_usage_and_conditioning/tables/sec33_constrained_ensemble_members.csv",
            "column_or_key": "params_saved_sum (per fold range)",
            "results_section": "Results 3.3",
            "generating_script": "sec33_usage_and_conditioning/scripts/sec33_11_constrained_fixed_ensemble_f2.py",
            "upstream_source": "sec32_model_structural_attributes.csv",
            "derivation_type": "RECOMPUTED",
            "notes": f"sum of member n_parameters saved vs unconstrained B({int(k)}), min-max over the 5 folds",
        })

    # F3 counts are computed from the in-memory records (including these two rows), so the manifest value
    # and the exported CSV count are always identical and free of a stale-CSV lag.
    records.append({
        "quantity_name": "f3_citable_numbers_rows",
        "value": "",
        "file_path": "_manifest/tables/manuscript_citable_numbers.csv",
        "column_or_key": "n_rows",
        "results_section": "Independent Verification",
        "generating_script": "_manifest/scripts/export_citable_numbers_f3.py",
        "upstream_source": "manuscript_master_manifest.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "",
    })
    records.append({
        "quantity_name": "f3_excluded_numbers_rows",
        "value": "",
        "file_path": "_manifest/tables/manuscript_excluded_numbers.csv",
        "column_or_key": "n_rows",
        "results_section": "Independent Verification",
        "generating_script": "_manifest/scripts/export_citable_numbers_f3.py",
        "upstream_source": "manuscript_master_manifest.csv",
        "derivation_type": "RECOMPUTED",
        "notes": "",
    })
    CIT_SET = ("RECOMPUTED", "RE-VERIFIED", "POST-FREEZE")   # H3: post-freeze rows are citable, flagged by phase
    EXC_SET = ("SUPERSEDED", "LOADED", "SENSITIVITY")
    _cit_rows = [r for r in records if r["derivation_type"] in CIT_SET]
    _exc_rows = [r for r in records if r["derivation_type"] in EXC_SET]
    _cit_counts = {}
    for r in _cit_rows:
        _cit_counts[r["derivation_type"]] = _cit_counts.get(r["derivation_type"], 0) + 1
    _exc_counts = {}
    for r in _exc_rows:
        _exc_counts[r["derivation_type"]] = _exc_counts.get(r["derivation_type"], 0) + 1
    _cit_secs = len({r["results_section"] for r in _cit_rows})
    _cit_phases = {}
    for r in _cit_rows:
        ph = "post_freeze" if r["derivation_type"] == "POST-FREEZE" else "pre_freeze"
        _cit_phases[ph] = _cit_phases.get(ph, 0) + 1
    for r in records:
        if r["quantity_name"] == "f3_citable_numbers_rows":
            r["value"] = str(len(_cit_rows))
            r["notes"] = f"status counts {_cit_counts}; freeze_phase counts {_cit_phases}; sections {_cit_secs}"
        elif r["quantity_name"] == "f3_excluded_numbers_rows":
            r["value"] = str(len(_exc_rows))
            r["notes"] = f"status counts {_exc_counts}; each row carries an exclusion reason"

    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(TABLES / "manuscript_master_manifest.csv", index=False)
    print(f"PASS: compiled master manifest with {len(df_manifest)} cited numbers across {df_manifest['results_section'].nunique()} sections")
    print(f"Derivation type counts:\n{df_manifest['derivation_type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
