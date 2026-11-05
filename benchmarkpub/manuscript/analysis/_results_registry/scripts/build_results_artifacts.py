#!/usr/bin/env python3
"""Generate the frozen Results claim registry, audits, and plotting data packages."""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ANALYSIS_ROOT.parents[3]
REGISTRY_ROOT = ANALYSIS_ROOT / "_results_registry"
DECLARATIONS_ROOT = REGISTRY_ROOT / "declarations"
FIGURES_ROOT = ANALYSIS_ROOT / "_figures"
TABLES_ROOT = REGISTRY_ROOT / "tables"
DATA_ROOT = FIGURES_ROOT / "data"
CITABLE_PATH = ANALYSIS_ROOT / "_manifest" / "tables" / "manuscript_citable_numbers.csv"
EXCLUDED_PATH = ANALYSIS_ROOT / "_manifest" / "tables" / "manuscript_excluded_numbers.csv"
MASTER_PATH = ANALYSIS_ROOT / "_manifest" / "tables" / "manuscript_master_manifest.csv"

sys.path.insert(0, str(ANALYSIS_ROOT))
sys.dont_write_bytecode = True
import _shared as S  # noqa: E402
from _shared.basin_accessor import (  # noqa: E402
    align_frame_by_id,
    align_series_by_id,
    canonical_basin_ids,
)

CONDITIONS = ["IC", "seen-dPL", "OOB-dPL"]
REGISTRY_FIELDS = [
    "section", "claim_id", "claim_slug", "estimand_name", "value",
    "source_file", "source_column", "source_row_key", "source_row_keys", "is_composite", "source_row_status", "status",
    "freeze_phase", "population_N", "quantity_class", "sign_convention", "better_direction",
    "primary_or_secondary",
    "venue",
]
AUDIT_FIELDS = ["check_id", "check_name", "status", "evidence", "failed_claim_ids", "failed_values"]
OPEN_FIELDS = ["section", "claim_id", "item_type", "related_value"]
PANEL_MAP_FIELDS = ["figure_id", "panel_id", "panel_content", "source_file", "source_columns", "section"]

TABLE_PATHS = {
    "pooled": "sec31_transfer_and_library/tables/sec31_pooled_delta_kge.csv",
    "model_delta": "sec31_transfer_and_library/tables/sec31_model_level_delta_kge.csv",
    "model_fold_delta": "sec31_transfer_and_library/tables/sec31_model_fold_delta_kge.csv",
    "variance": "sec31_transfer_and_library/tables/sec31_variance_decomposition.csv",
    "rank_basin": "sec31_transfer_and_library/tables/sec31_basin_rank_spearman_summary.csv",
    "rank_global": "sec31_transfer_and_library/tables/sec31_global_rank_spearman.csv",
    "eq_basin": "sec31_transfer_and_library/tables/sec31_equivalence_sets_per_basin.csv",
    "eq_size": "sec31_transfer_and_library/tables/sec31_equivalence_size_distribution.csv",
    "cover_primary": "sec31_transfer_and_library/tables/sec31_minimal_cover_targets.csv",
    "cover_nbb": "sec31_transfer_and_library/tables/sec31_multiyear_block_bootstrap_comparison.csv",
    "cover_ci": "sec32_library_members/tables/sec32_bootstrap_cover_ci_summary.csv",
    "coverage_exact": "sec31_transfer_and_library/tables/sec31_coverage_curves_exact.csv",
    "uncertainty_band": "sec31_transfer_and_library/tables/sec31_uncertainty_bandwidth_per_basin.csv",
    "optimal_summary": "sec32_library_members/tables/sec32_optimal_covers_summary.csv",
    "optimal_solutions": "sec32_library_members/tables/sec32_optimal_covers_all_solutions.csv",
    "optimal_frequency": "sec32_library_members/tables/sec32_optimal_covers_model_frequencies.csv",
    "oob_members": "sec32_library_members/tables/sec32_oob_full_cover_members.csv",
    "bottleneck": "sec32_library_members/tables/sec32_bottleneck_100pct_coverage.csv",
    "singleton": "sec34_determinacy_and_bounds/tables/sec34_singleton_totals_idjoin_check.csv",
    "coverage_ladder": "sec32_library_members/tables/sec32_coverage_ladder.csv",
    "regret_curve": "sec32_library_members/tables/sec32_regret_curve.csv",
    "model_attributes": "sec32_library_members/tables/sec32_model_structural_attributes.csv",
    "rules_ladder": "sec33_usage_and_conditioning/tables/sec33_rules_ladder_unified.csv",
    "rule_c_b": "sec33_usage_and_conditioning/tables/sec33_e8_rule_c_vs_b_k2_k5.csv",
    "rule_c_b_k3": "sec33_usage_and_conditioning/tables/sec33_rule_c_vs_b_summary.csv",
    "gset": "sec33_usage_and_conditioning/tables/sec33_gset_summary.csv",
    "gset_e8": "sec33_usage_and_conditioning/tables/sec33_e8_gset_k2_k5.csv",
    "diversity": "sec33_usage_and_conditioning/tables/sec33_diversity_sensitivity_summary.csv",
    "constrained": "sec33_usage_and_conditioning/tables/sec33_constrained_ensemble_summary.csv",
    "cluster_oob": "sec33_usage_and_conditioning/tables/sec33_cluster_oob_validation_summary.csv",
    "k1_constraint": "sec34_determinacy_and_bounds/tables/sec34_k1_constraint_audit_summary.csv",
    "k3_constraint": "sec34_determinacy_and_bounds/tables/sec34_k3_single_source_recomputation.csv",
    "k5_constraint": "sec34_determinacy_and_bounds/tables/sec34_k5_parsimony_penalty_constraint_audit.csv",
    "parsimony": "sec34_determinacy_and_bounds/tables/sec34_parsimony_penalty_summary.csv",
    "displacement_corr": "sec34_determinacy_and_bounds/tables/sec34_displacement_metrics_correlation_matrix.csv",
    "displacement_oob": "sec34_determinacy_and_bounds/tables/sec34_displacement_vs_oob_loss.csv",
    "nparam_corr": "sec34_determinacy_and_bounds/tables/sec34_nparam_instability_independent_recomputation.csv",
    "partial_corr": "sec34_determinacy_and_bounds/tables/sec34_partial_correlation_nparam_vs_instability.csv",
    "screening": "sec34_determinacy_and_bounds/tables/sec34_attribute_screening_bh_fdr.csv",
    "concentration": "sec34_determinacy_and_bounds/tables/sec34_penalty_concentration.csv",
    "type_penalty": "sec35_cluster_stratified/tables/sec35_g1_penalty_by_type.csv",
    "type_kw": "sec35_cluster_stratified/tables/sec35_g1_kruskal_wallis.csv",
    "type_pairwise": "sec35_cluster_stratified/tables/sec35_g1_pairwise_dunn_holm.csv",
    "type_matched": "sec35_cluster_stratified/tables/sec35_g1_matched_summary.csv",
    "type_attribute": "sec35_cluster_stratified/tables/sec35_g1b_attribute_penalty_spearman_by_type.csv",
    "worst26": "sec35_cluster_stratified/tables/sec35_g2_worst26_contingency.csv",
    "worst26_test": "sec35_cluster_stratified/tables/sec35_g2_fisher_exact.csv",
    "parsimony_type": "sec35_cluster_stratified/tables/sec35_g4_parsimony_by_type.csv",
    "parsimony_kw": "sec35_cluster_stratified/tables/sec35_g4_parsimony_kruskal.csv",
    "null95": "sec36_size_matched_null/tables/sec36_h1_size_matched_null_summary.csv",
    "null_meta": "sec36_size_matched_null/tables/sec36_h1_run_meta.csv",
}

PROVENANCE_FILES = [
    ANALYSIS_ROOT / "sec31_transfer_and_library/scripts/sec31_05_provenance.py",
    ANALYSIS_ROOT / "sec32_library_members/scripts/sec32_04_provenance.py",
    ANALYSIS_ROOT / "sec33_usage_and_conditioning/scripts/sec33_06_provenance.py",
    ANALYSIS_ROOT / "sec34_determinacy_and_bounds/scripts/sec34_05_provenance.py",
    ANALYSIS_ROOT / "sec35_cluster_stratified/scripts/sec35_05_provenance.py",
    ANALYSIS_ROOT / "sec36_size_matched_null/scripts/sec36_03_provenance.py",
]

# These are only the requested estimand families which have no matching line in either frozen manifest.
# They are audit keys, not claims and contain no values.
REQUIRED_NAME_CHECKS = [
    ("R1", "top3_top5_exact_set_retention", r"top.?[35].*(exact.?set|retention)|retention.*top.?[35]"),
    ("R1", "residual_variance_share", r"residual.*(ss|variance)|variance.*residual"),
    ("R1", "equivalence_membership_overlap", r"membership.*overlap|overlap.*equivalence|shared.*member"),
    ("R1", "model_fold_same_sign_180", r"model.?fold.*positive|180.*positive|model.?fold.*same.?sign"),
    ("R2", "top3_top5_candidate_union", r"candidate.*union|union.*top.?[35]"),
    ("R2", "optimal_solution_core_union", r"optimal.*(core|union)|(core|union).*optimal"),
    ("R2", "oob_100_cover_member_list", r"oob.*100.*(member|model).*(list|name)|oob.*full.*cover.*members"),
    ("R3", "conditionalized_upper_bound", r"conditional.*upper|upper.*bound"),
    ("R3", "single_model_attribute_negative_result", r"single.?model.*(attribute|recommend)|attribute.*single.?model"),
    ("R4", "attribute_screen_survivor_count_and_max_abs_rho", r"(screen|screening).*(surviv|count|max.*rho)|max.*abs.*rho"),
    ("R4", "penalty_effect_size_and_pairwise_adjusted_values", r"effect.?size|pairwise.*(holm|adjusted).*p"),
    ("R4", "size_matched_null_90_and_100", r"pf_h1_null_(90|100)"),
    ("R4", "displacement_oob_loss_three_aggregations", r"displacement.*oob.*loss"),
]
DECLARATION_FIELDS = {
    "primary_specification.csv": ["family", "manuscript_primary", "secondary_retained", "note"],
    "sign_and_direction.csv": ["quantity_class", "subtraction_order", "better_direction", "note"],
    "population_taxonomy.csv": ["n_value", "category", "meaning"],
    "population_n_overrides.csv": ["claim_slug", "population_category", "population_meaning", "basis_note"],
    "claim_quantity_class_map.csv": ["slug_pattern", "quantity_class"],
}
POPULATION_N_OVERRIDE_ROWS = (
    ("two_way_anova_basin_ss_share", "population", "model_by_basin_cells", "按总单元自由度和来源表结构解析"),
    ("two_way_anova_model_ss_share", "population", "model_by_basin_cells", "按总单元自由度和来源表结构解析"),
    ("candidate_set_median_regret_k1", "population", "full_basin_domain", "按完整 basin ID 集合解析并核验"),
    ("candidate_set_median_regret_k3", "population", "full_basin_domain", "按完整 basin ID 集合解析并核验"),
    ("candidate_set_median_regret_k5", "population", "full_basin_domain", "按完整 basin ID 集合解析并核验"),
    ("inter_metric_spearman_fold_instability_3caed92d", "analysis_unit_count", "model_count", "按冻结模型折分表中的不同 model ID 解析"),
    ("pf_g1b_attribute_choice", "not_applicable", "not_applicable", "分类标识论断没有适用的总体 N"),
)
POPULATION_OVERRIDE_RESOLUTION_FIELDS = [
    "claim_id", "claim_slug", "population_category", "population_meaning", "population_N_value",
    "claim_source_file", "population_N_source_file", "population_N_source_column", "population_N_source_row_key",
    "population_N_source_row_status", "claim_status", "claim_freeze_phase", "resolution_rule", "validation",
]
T3_FIELDS = [
    "gap_id", "proposal_shape", "proposal_status", "record_type", "candidate_rank", "candidate_estimand_name",
    "source_file", "source_column", "source_row_key", "value", "source_row_status", "status", "freeze_phase", "basis",
]
PRIMARY_SPECIFICATION_ROWS = (
    ("global_seen_oob_rank_spearman", "same_pipeline", "canonical", ""),
    ("equivalence_and_cover", "one_year_waterblock_q05", "three_year_nonoverlapping_block", ""),
    ("uncertainty_bandwidth", "one_year_waterblock", "three_year_nonoverlapping_block", "jackknife 为 SUPERSEDED，不得引用"),
    ("transfer_penalty_delta_kge", "same_pipeline_seen_minus_heldout", "cross_pipeline", ""),
    ("winner_retention", "same_pipeline", "cross_pipeline", ""),
    ("top5_set_overlap", "same_pipeline", "cross_pipeline", ""),
    ("variance_decomposition", "thirtysix_model_delta_kge_pub", "none", "八模型 dPL−IC 版属另一稿，本稿任何论断不得引用"),
    ("gset_heldout", "A2d", "A1b", ""),
    ("parsimony_penalty", "id_joined_ranking_minus_parsimonious", "none", "旧值 SUPERSEDED"),
    ("equivalence_set_dispersion", "id_joined", "none", "旧值 SUPERSEDED"),
    ("rule_c_k", "all_reported", "none", "阶梯量，无单一 primary"),
    ("fixed_global_ensemble_k", "k2_as_operating_point", "k_sweep", ""),
    ("paired_difference_baseline", "relative_to_B_k2", "relative_to_A", "两组并列报出"),
    ("paired_statistic", "paired_difference", "none", "边际中位数之差不得用作配对效应"),
)
SIGN_AND_DIRECTION_ROWS = (
    ("kge_level", "n/a", "higher", ""),
    ("cover_count / library_size", "n/a", "lower", ""),
    ("transfer_penalty", "seen_minus_heldout", "lower", ""),
    ("parsimony_penalty", "ranking_member_minus_parsimonious_member", "lower", ""),
    ("rule_paired_difference", "rule_minus_baseline", "higher", ""),
    ("constrained_set_paired_difference", "constrained_minus_unconstrained", "higher", ""),
    ("diversity_set_paired_difference", "diversity_minus_ranking", "higher", ""),
    ("regret_level", "n/a", "lower", ""),
    ("regret_paired_difference", "recommended_minus_fixed", "lower", ""),
    ("gset", "n/a", "higher", ""),
    ("spearman / retention / coverage_fraction", "n/a", "higher", ""),
    ("equivalence_set_size", "n/a", "none", ""),
    ("parameter_displacement", "n/a", "lower", ""),
    ("p_value / effect_size", "n/a", "none", ""),
    ("bandwidth", "n/a", "none", ""),
    ("variance_or_share", "n/a", "none", ""),
    ("descriptive_count", "n/a", "none", ""),
    ("categorical_identifier", "n/a", "none", ""),
)

CLAIM_QUANTITY_CLASS_MAP_ROWS = (
    ("rules_ladder_*_median", "kge_level"),
    ("t3_2_fixed_ensemble_*_median_kge", "kge_level"),
    ("minimal_cover_*", "cover_count"),
    ("multiyear_*_cover_*", "cover_count"),
    ("bootstrap_cover_ci_*", "cover_count"),
    ("exact_minimum_cover_*", "cover_count"),
    ("pf_h1_null_95_*", "cover_count"),
    ("pooled_delta_kge_pub_median", "transfer_penalty"),
    ("pf_g1_penalty_median_type*", "transfer_penalty"),
    ("*parsimony*mean_diff*", "parsimony_penalty"),
    ("parsimony_penalty_mean_diff*", "parsimony_penalty"),
    ("*rule_c_vs_b*", "rule_paired_difference"),
    ("f2_constrained_*", "constrained_set_paired_difference"),
    ("*diversity_k3_diff_vs_ranking_mean", "diversity_set_paired_difference"),
    ("candidate_set_median_regret_*", "regret_level"),
    ("cluster_oob_top5_regret_diff_mean", "regret_paired_difference"),
    ("e8_gset_*", "gset"),
    ("*spearman*", "spearman"),
    ("independent_recompute_nparam_*", "spearman"),
    ("*coverage_k*", "coverage_fraction"),
    ("cluster_oob_top3_coverage_diff", "coverage_fraction"),
    ("equivalence_set_size_median_*", "equivalence_set_size"),
    ("uncertainty_bandwidth_*", "bandwidth"),
    ("two_way_anova_*_ss_share", "variance_or_share"),
    ("top5pct_basins_share_*", "variance_or_share"),
    ("top5_models_share_*", "variance_or_share"),
    ("*kruskal_wallis*", "p_value_or_effect_size"),
    ("pf_g4_parsimony_kw_*", "p_value_or_effect_size"),
    ("pf_g2_fisher_exact_p", "p_value_or_effect_size"),
    ("pf_g1_pairwise_dunn_holm", "p_value_or_effect_size"),
    ("basin_attribute_screening_min_q_value", "p_value_or_effect_size"),
    ("bottleneck_forcing_models_*", "categorical_identifier"),
    ("pf_g1b_attribute_choice", "categorical_identifier"),
    ("*_count", "descriptive_count"),
    ("*_counts", "descriptive_count"),
    ("*_union_size", "descriptive_count"),
    ("optimal_solutions_count_*", "descriptive_count"),
    ("singleton_total_idjoin_*", "descriptive_count"),
    ("models_positive_delta_kge_fraction", "descriptive_count"),
    ("*", "unclassified"),
)

_read_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]] = {}


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    key = str(path.resolve())
    if key not in _read_cache:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]
        _read_cache[key] = (fields, rows)
    return _read_cache[key]


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def table_path(rel: str) -> Path:
    return ANALYSIS_ROOT / rel


def table_data(rel: str) -> Tuple[List[str], List[Dict[str, str]]]:
    return read_csv_rows(table_path(rel))


def write_rows(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})

    _read_cache.pop(str(path.resolve()), None)


def quantity_class_from_slug(slug: str) -> str:
    path = DECLARATIONS_ROOT / "claim_quantity_class_map.csv"
    rows = read_csv_rows(path)[1] if path.exists() else []
    rules = [(r["slug_pattern"], r["quantity_class"]) for r in rows] if rows else CLAIM_QUANTITY_CLASS_MAP_ROWS
    for pattern, quantity_class in rules:
        if fnmatch.fnmatchcase(slug, pattern):
            return quantity_class
    return "unclassified"


def write_prompt_declarations() -> Dict[str, int]:
    DECLARATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    primary = [dict(zip(DECLARATION_FIELDS["primary_specification.csv"], values)) for values in PRIMARY_SPECIFICATION_ROWS]
    directions = [dict(zip(DECLARATION_FIELDS["sign_and_direction.csv"], values)) for values in SIGN_AND_DIRECTION_ROWS]
    quantity_map = [dict(zip(DECLARATION_FIELDS["claim_quantity_class_map.csv"], values)) for values in CLAIM_QUANTITY_CLASS_MAP_ROWS]
    population_overrides = [dict(zip(DECLARATION_FIELDS["population_n_overrides.csv"], values)) for values in POPULATION_N_OVERRIDE_ROWS]
    write_rows(DECLARATIONS_ROOT / "primary_specification.csv", DECLARATION_FIELDS["primary_specification.csv"], primary)
    write_rows(DECLARATIONS_ROOT / "sign_and_direction.csv", DECLARATION_FIELDS["sign_and_direction.csv"], directions)
    write_rows(DECLARATIONS_ROOT / "claim_quantity_class_map.csv", DECLARATION_FIELDS["claim_quantity_class_map.csv"], quantity_map)
    write_rows(DECLARATIONS_ROOT / "population_n_overrides.csv", DECLARATION_FIELDS["population_n_overrides.csv"], population_overrides)
    _, primary_check = read_csv_rows(DECLARATIONS_ROOT / "primary_specification.csv")
    _, direction_check = read_csv_rows(DECLARATIONS_ROOT / "sign_and_direction.csv")
    _, quantity_map_check = read_csv_rows(DECLARATIONS_ROOT / "claim_quantity_class_map.csv")
    _, population_override_check = read_csv_rows(DECLARATIONS_ROOT / "population_n_overrides.csv")
    if primary_check != primary:
        raise ValueError("primary_specification.csv differs from the prompt table")
    if direction_check != directions:
        raise ValueError("sign_and_direction.csv differs from the prompt table")
    if quantity_map_check != quantity_map:
        raise ValueError("claim_quantity_class_map.csv differs from the prompt table")
    if population_override_check != population_overrides:
        raise ValueError("population_n_overrides.csv differs from the prompt table")
    return {
        "primary_specification": len(primary_check),
        "sign_and_direction": len(direction_check),
        "claim_quantity_class_map": len(quantity_map_check),
        "population_n_overrides": len(population_override_check),
    }


def write_population_taxonomy(registry: List[Dict[str, str]]) -> int:
    _, e5_rows = read_csv_rows(TABLES_ROOT / "evidence" / "E5_population_N_evidence.csv")
    t2_path = TABLES_ROOT / "T2_metadata_resolution.csv"
    t2_by_claim = {r["claim_id"]: r for r in read_csv_rows(t2_path)[1]} if t2_path.exists() else {}
    e5_by_claim = defaultdict(list)
    for evidence in e5_rows:
        if evidence.get("claim_id"):
            e5_by_claim[evidence["claim_id"]].append(evidence)
    e5_values = {
        evidence.get("population_N_value", "").strip()
        for evidence in e5_rows
        if evidence.get("record_type") == "resolved_population_N_reference"
        and as_decimal(evidence.get("population_N_value", "")) is not None
    }
    by_n = defaultdict(list)
    for claim in registry:
        n_value = claim.get("population_N", "").strip()
        if n_value in e5_values and as_decimal(n_value) is not None:
            by_n[n_value].append(claim)
    total_basins = str(len(canonical_basin_ids()))
    taxonomy = []
    for n_value, claims in sorted(by_n.items(), key=lambda item: as_decimal(item[0]) or Decimal(0)):
        signals = set()
        contexts = []
        for claim in claims:
            t2 = t2_by_claim.get(claim["claim_id"], {})
            if t2.get("population_N") == n_value and t2.get("population_category"):
                signals.add(t2["population_category"])
                contexts.append(t2.get("population_meaning", ""))
                continue
            evidence_rows = e5_by_claim.get(claim["claim_id"], [])
            evidence_context = " ".join(
                value for evidence in evidence_rows
                for value in (evidence.get("source_table_literal_explanation", ""), evidence.get("source_row_key", ""), evidence.get("manifest_notes", ""))
            ).lower()
            context = " ".join((claim.get("source_file", ""), claim.get("source_column", ""), claim.get("source_row_key", ""), claim.get("estimand_name", ""), evidence_context)).lower()
            contexts.append(context)
            if claim.get("section") in {"R1", "R2", "R3"} and claim.get("quantity_class") in {"cover_count", "descriptive_count", "equivalence_set_size"}:
                signals.add("population")
                continue
            if re.search(r"\bn_type\s*=|\bn\s*=\s*[0-9][0-9,]*\s*\[type=", context) or (claim.get("section") == "R4" and type_from_name(claim.get("estimand_name", ""))):
                signals.add("group_size")
            if re.search(r"\bn_models\s*=", context) or "models_positive_delta_kge_fraction" in context or re.search(rf"\b{re.escape(n_value)}\s+seeds?\b|\bseeds?\b[^;|]{{0,24}}\b{re.escape(n_value)}\b", context):
                signals.add("analysis_unit_count")
            if re.search(r"\bn_basins(?:_total|_covered)?\s*=|\bn_matched_total\s*=|\bn_total\s*=|\bn_used\s*=|\bcells\b", context):
                signals.add("population")
            if "pairwise" in context and "n_rows" in claim.get("source_column", ""):
                signals.add("comparison_count")
        if n_value == total_basins and "population" in signals:
            category = "population"
        elif "group_size" in signals and not any(signal != "group_size" for signal in signals):
            category = "group_size"
        elif len(signals) == 1:
            category = next(iter(signals))
        else:
            category = "unclassified"
        joined = " ".join(contexts)
        if category == "population":
            if re.search(r"\bn_cells(?:_total)?\s*=|\bcells\b", joined):
                meaning = "model×basin 单元总数"
            elif "n_matched_total=" in joined or re.search(r"\bn_total\s*=", joined):
                meaning = "已匹配类型的流域数"
            elif "need_basins=" in joined and re.search(r"\bn_basins\s*=", joined):
                meaning = "全域流域总数；覆盖阈值所需流域数"
            elif "need_basins=" in joined:
                meaning = "覆盖阈值所需流域数"
            else:
                meaning = "全域流域总数"
        elif category == "analysis_unit_count":
            meaning = "model×fold 单元数" if "model×fold" in joined else "种子数" if "seed" in joined else "模型数"
        elif category == "group_size":
            meaning = "逐流域类型的 N"
        elif category == "comparison_count":
            meaning = "两两比较对数"
        elif len(signals) > 1:
            meaning = "同一 N 对应多个单位类别，无法唯一归类"
        elif "seed" in joined:
            meaning = "E5 记载为种子数，不属于给定四类"
        else:
            meaning = "E5 证据不足以归入给定四类"
        taxonomy.append({"n_value": n_value, "category": category, "meaning": meaning})
    taxonomy.append({"n_value": "EXPLANATORY_NOTE", "category": "declaration_exception", "meaning": "S3 数值例外：本分类表按定义列出 E5 的数值 N，由脚本从 E5 证据生成，非人工转录；此例外不扩展至口径或方向声明。"})
    write_rows(DECLARATIONS_ROOT / "population_taxonomy.csv", DECLARATION_FIELDS["population_taxonomy.csv"], taxonomy)
    _read_cache.pop(str((DECLARATIONS_ROOT / "population_taxonomy.csv").resolve()), None)
    if len({row["n_value"] for row in taxonomy}) != len(taxonomy):
        raise ValueError("population_taxonomy.csv contains duplicate N values")
    return len(taxonomy)

def as_float(value: object) -> Optional[float]:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def as_decimal(value: object) -> Optional[Decimal]:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if len(slug) > 48:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        slug = slug[:39].rstrip("_") + "_" + digest
    return slug or "claim"


def condition_from_name(name: str) -> str:
    lowered = name.lower()
    if "oob-dpl" in lowered or "oob_dpl" in lowered:
        return "OOB-dPL"
    if "seen-dpl" in lowered or "seen_dpl" in lowered:
        return "seen-dPL"
    if re.search(r"(^|[_-])ic([_-]|$)", lowered):
        return "IC"
    return ""


def target_from_name(name: str) -> str:
    match = re.search(r"(?:target[_-]?)?(90|95|100)(?:%|pct|percent)(?:_|$)", name, flags=re.I)
    if not match:
        match = re.search(r"_(90|95|100)%", name)
    if not match:
        match = re.search(r"(?:^|_)(90|95|100)(?:_|$)", name, flags=re.I)
    return f"{match.group(1)}%" if match else ""


def type_from_name(name: str) -> str:
    match = re.search(r"type[_-]?(\d+)", name, flags=re.I)
    return match.group(1) if match else ""


def k_from_name(name: str) -> str:
    match = re.search(r"(?:^|_)k(\d+)(?:_|$)", name, flags=re.I)
    if not match:
        match = re.search(r"top[_-]?(\d+)", name, flags=re.I)
    return match.group(1) if match else ""


def normalize_condition(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_source_row(entry: Dict[str, str]) -> Tuple[Optional[Dict[str, str]], List[str]]:
    source = entry.get("file_path", "")
    path = table_path(source)
    if not source or not path.is_file():
        return None, []
    fields, rows = read_csv_rows(path)
    if not rows:
        return None, fields
    name = entry.get("estimand", "")
    candidates = list(rows)
    condition = condition_from_name(name)
    target = target_from_name(name)
    typ = type_from_name(name)
    k = k_from_name(name)
    family_match = re.search(r"rules_ladder_([A-E])_k(\d+)", name, flags=re.I)
    tier_match = re.search(r"constrained_(P25|median|P75)_k(\d+)", name, flags=re.I)

    if condition:
        for col in ("condition", "estimand", "strategy"):
            if col in fields:
                filtered = [r for r in candidates if normalize_condition(condition) in normalize_condition(r.get(col, ""))]
                if filtered:
                    candidates = filtered
                    break
    if target:
        for col in ("target_pct", "target"):
            if col in fields:
                want = target if col == "target_pct" else f"{int(target[:-1]) / 100:.6f}"
                filtered = [r for r in candidates if r.get(col, "").strip() == want]
                if filtered:
                    candidates = filtered
                    break
    if typ and "type" in fields:
        filtered = [r for r in candidates if r.get("type", "").strip() == typ]
        if filtered:
            candidates = filtered
    if k and "k" in fields:
        filtered = [r for r in candidates if r.get("k", "").strip() == k]
        if filtered:
            candidates = filtered
    if family_match:
        fam, fam_k = family_match.group(1).upper(), family_match.group(2)
        if "rule_family" in fields:
            filtered = [r for r in candidates if r.get("rule_family", "").strip().upper() == fam]
            if filtered:
                candidates = filtered
        if "k" in fields:
            filtered = [r for r in candidates if r.get("k", "").strip() == fam_k]
            if filtered:
                candidates = filtered
    if tier_match:
        tier, tier_k = tier_match.group(1).lower(), tier_match.group(2)
        for col in ("tier", "parameter_tier", "threshold_tier"):
            if col in fields:
                filtered = [r for r in candidates if r.get(col, "").strip().lower() == tier]
                if filtered:
                    candidates = filtered
                    break
        if "k" in fields:
            filtered = [r for r in candidates if r.get("k", "").strip() == tier_k]
            if filtered:
                candidates = filtered
    column_key = entry.get("column_or_key", "")
    for selector in re.findall(r"\(([^()]*)\)", column_key):
        assignment = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", selector.strip())
        if assignment and assignment.group(1) in fields:
            field, wanted = assignment.group(1), assignment.group(2).strip()
            filtered = [r for r in candidates if normalize_condition(r.get(field, "")) == normalize_condition(wanted)]
        else:
            wanted = normalize_condition(selector)
            if wanted == "top5basins":
                wanted = "top5pctbasins"
            filtered = [r for r in candidates if any(wanted and wanted in normalize_condition(str(value)) for value in r.values())]
        if filtered:
            candidates = filtered
    for col in ("selection_method", "subset_name", "metric_name", "instability_metric"):
        if col in fields:
            filtered = [r for r in candidates if r.get(col, "").strip() and normalize_condition(r[col]) in normalize_condition(name)]
            if filtered and len(filtered) < len(candidates):
                candidates = filtered
    scheme_match = re.search(r"(1yr_block_bootstrap_primary|3yr_multiyear_block_bootstrap_c1)", name, flags=re.I)
    if scheme_match and "scheme" in fields:
        filtered = [r for r in candidates if normalize_condition(r.get("scheme", "")) == normalize_condition(scheme_match.group(1))]
        if filtered:
            candidates = filtered
    if len(candidates) == 1:
        return candidates[0], fields
    return None, fields


def source_row_status(entry: Dict[str, str], row: Optional[Dict[str, str]]) -> str:
    if row is not None:
        return "matched"
    source = entry.get("file_path", "")
    path = table_path(source) if source else None
    if path is None or not path.is_file():
        return "unresolved"
    fields, _ = read_csv_rows(path)
    column_key = entry.get("column_or_key", "")
    aggregate_fields = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:sum|mean|min|max|median)\s*\(", column_key, flags=re.I)
    predicate_fields = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:>|<|==)\s*[-+]?[0-9.]+", column_key)
    referenced = aggregate_fields + predicate_fields
    if referenced and all(field in fields for field in referenced):
        return "aggregate_expression"
    direct_columns = [part.strip() for part in column_key.split("/") if part.strip() in fields]
    if len(direct_columns) > 1:
        column_rows = read_csv_rows(path)[1]
        value_tuples = {tuple(r.get(field, "").strip() for field in direct_columns) for r in column_rows}
        if len(value_tuples) == 1:
            return "unique_source_tuple"
    return "unresolved"


def source_row_key(entry: Dict[str, str], row: Optional[Dict[str, str]], status: str) -> str:
    if row is None:
        if status == "unique_source_tuple":
            source = entry.get("file_path", "")
            fields, rows = table_data(source)
            columns = [part.strip() for part in entry.get("column_or_key", "").split("/") if part.strip() in fields]
            values = {tuple(r.get(field, "").strip() for field in columns) for r in rows}
            if len(values) == 1:
                return "unique_source_tuple=" + "|".join(f"{field}={value}" for field, value in zip(columns, next(iter(values))))
        if status == "aggregate_expression":
            return f"aggregate_expression={entry.get('column_or_key', '')}"
        return f"unresolved_estimand={entry.get('estimand', '')}|column_or_key={entry.get('column_or_key', '')}"
    parts = []
    for key in ("condition", "target_pct", "target", "type", "k", "rule_family", "model", "component", "metric", "entity_type", "group_label", "n_units", "estimand", "selection_method", "subset_name", "scheme", "seen_definition", "metric_name", "instability_metric"):
        if row.get(key, ""):
            parts.append(f"{key}={row[key]}")
    return "|".join(parts) if parts else f"manifest_estimand={entry.get('estimand', '')}"

def explicit_sign_and_direction(row: Optional[Dict[str, str]]) -> Tuple[str, str]:
    if row is None:
        return "", ""
    sign = ""
    for key in ("sign_convention", "subtraction_order", "difference_definition", "delta_definition", "direction_definition"):
        if row.get(key, "").strip():
            sign = row[key].strip()
            break
    direction = ""
    for key in ("better_direction", "better_direction_definition", "direction_definition"):
        if row.get(key, "").strip():
            direction = row[key].strip()
            break
    return sign, direction


def primary_secondary(name: str, entry: Dict[str, str], source_row: Optional[Dict[str, str]]) -> str:
    if source_row:
        for key in ("primary_or_secondary", "manuscript_role", "role"):
            value = source_row.get(key, "").strip().lower()
            if value in {"primary", "secondary"}:
                return value
    if re.search(r"(?:^|[_-])primary(?:[_-]|$)", name, flags=re.I):
        return "primary"
    if re.search(r"(?:^|[_-])secondary(?:[_-]|$)", name, flags=re.I):
        return "secondary"
    return ""


def provenance_population_n(entry: Dict[str, str]) -> str:
    source_basename = Path(entry.get("file_path", "")).name
    if not source_basename:
        return ""
    declared = set()
    for provenance_path in PROVENANCE_FILES:
        if not provenance_path.is_file():
            continue
        lines = provenance_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if source_basename not in line:
                continue
            window = "\n".join(lines[max(0, index - 3):index + 4])
            declared.update(match.replace(",", "") for match in re.findall(r"\bN\s*=\s*([0-9][0-9,]*)", window))
    return next(iter(declared)) if len(declared) == 1 else ""


def seed_population_n(entry: Dict[str, str], fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> str:
    if ".mean()" not in entry.get("column_or_key", "") or "seed" not in fields:
        return ""
    name = entry.get("estimand", "")
    candidates = list(rows)
    k = k_from_name(name)
    if k and "k" in fields:
        filtered = [r for r in candidates if r.get("k", "").strip() == k]
        if filtered:
            candidates = filtered
    condition = condition_from_name(name)
    if condition and "condition" in fields:
        filtered = [r for r in candidates if normalize_condition(condition) in normalize_condition(r.get("condition", ""))]
        if filtered:
            candidates = filtered
    if "estimand" in fields:
        filtered = [r for r in candidates if r.get("estimand", "").strip() and normalize_condition(r["estimand"]) in normalize_condition(name)]
        if filtered:
            candidates = filtered
    seeds = {r.get("seed", "").strip() for r in candidates if r.get("seed", "").strip()}
    return str(len(seeds)) if seeds else ""


def population_n(entry: Dict[str, str], row: Optional[Dict[str, str]], fields: Sequence[str], master: Dict[str, str]) -> str:
    notes = master.get("notes", "")
    matches = re.findall(r"\b[Nn]\s*=\s*([0-9][0-9,]*)(?:\s+(cells?|models?|basins?|pairs?|rows?|folds?|types?))?", notes)
    if matches:
        return ";".join(f"{number.replace(',', '')} {unit or ''}".strip() for number, unit in matches)
    fraction = re.search(r"(?:covers?|n_covered)\s*(?:=|:)?\s*[0-9][0-9,]*\s*/\s*([0-9][0-9,]*)", notes, flags=re.I)
    if fraction:
        return fraction.group(1).replace(",", "")
    provenance_n = provenance_population_n(entry)
    if provenance_n:
        return provenance_n
    source = entry.get("file_path", "")
    column_key = entry.get("column_or_key", "")
    source_fields, source_rows = table_data(source) if source and table_path(source).is_file() else ([], [])
    if source_rows and (".sum()" in column_key or re.search(r"[A-Za-z_]\w*\s*>\s*[-+]?[0-9.]+", column_key)):
        return str(len(source_rows))
    seed_count = seed_population_n(entry, source_fields, source_rows)
    if seed_count:
        return f"{seed_count} seeds"
    estimand = entry.get("estimand", "")
    if estimand.startswith("top5") and "entity_type" in source_fields and "n_units" in source_fields:
        totals = [r for r in source_rows if r.get("entity_type") == "all_total" and as_decimal(r.get("n_units", "")) is not None]
        if len(totals) == 1:
            return totals[0]["n_units"]
    if row and "need_basins" in row and source_rows:
        condition = row.get("condition", "")
        full_rows = []
        for candidate in source_rows:
            same_condition = not condition or candidate.get("condition", "") == condition
            target = candidate.get("target_pct", "")
            target_value = as_decimal(candidate.get("target", ""))
            if same_condition and (target == "100%" or target_value == Decimal("1")):
                full_rows.append(candidate)
        if len(full_rows) == 1 and as_decimal(full_rows[0].get("need_basins", "")) is not None:
            return full_rows[0]["need_basins"]
    population_keys = ("population_N", "population_n", "n_basins_total", "n_basins", "n_basins_covered", "n_type_basins", "n_basins_in_type", "n_total", "n_total_defined", "n_matched_total", "n_top_defined", "n_type", "n_used", "N", "n_models_total", "n_models", "n_rows")
    if row:
        for key in population_keys:
            value = row.get(key, "").strip()
            if value and as_decimal(value) is not None:
                return value
        if "type" in row and as_decimal(row.get("n", "")) is not None:
            return row["n"]
    if source_fields:
        for key in population_keys:
            if key not in source_fields:
                continue
            values = {r.get(key, "").strip() for r in source_rows if as_decimal(r.get(key, "")) is not None}
            if len(values) == 1:
                return next(iter(values))
    # A type-specific table declares its population in the per-type n field.
    if "type" in source_fields and "n" in source_fields:
        values = {r.get("n", "").strip() for r in source_rows if as_decimal(r.get("n", "")) is not None}
        if len(values) == 1:
            return next(iter(values))
    for pattern in (r"\b([0-9][0-9,]*)\s+models\b", r"\b([0-9][0-9,]*)\s+model.?fold\b", r"\b([0-9][0-9,]*)\s+cells\b"):
        match = re.search(pattern, notes, flags=re.I)
        if match:
            return match.group(1).replace(",", "")
    return ""

def selected_section(entry: Dict[str, str]) -> str:
    name = entry.get("estimand", "")
    source_section = entry.get("results_section", "")
    status = entry.get("status", "")

    # R1: only the listed transfer, variance, ranking, equivalence, and cover quantities.
    if (
        name in {
            "pooled_delta_kge_pub_median", "models_positive_delta_kge_fraction",
            "two_way_anova_basin_ss_share", "two_way_anova_model_ss_share",
            "global_seen_vs_oob_rank_spearman_canonical", "basin_level_seen_vs_oob_rank_spearman_median",
        }
        or name.startswith("minimal_cover_")
        or name.startswith("multiyear_3yr_block_boot_cover_")
        or name.startswith("uncertainty_bandwidth_")
        or name.startswith("bootstrap_cover_ci_")
        or name.startswith("equivalence_set_size_median_")
    ):
        return "R1"

    # R2: the frozen library, exact-solution, bottleneck, ladder, regret, and structure quantities.
    if source_section == "Results 3.2" or name in {"exact_minimum_cover_top3", "exact_minimum_cover_top5"}:
        return "R2"
    if re.search(r"singleton", name, re.I) and source_section in {"Results 3.2", "Independent Verification"}:
        return "R2"
    if re.search(r"oob.*(100|full).*cover.*(member|model)", name, re.I):
        return "R2"

    # R3: the requested rule ladder and paired comparisons, plus only the named k=3 diversity variants.
    if source_section == "Results 3.3":
        if name.startswith("rules_ladder_") or name.startswith("t3_2_fixed_ensemble_"):
            return "R3"
        if name in {"t3_1_rule_c_vs_b_diff_mean", "e8_rule_c_vs_b_k2_mean_diff_mean", "e8_rule_c_vs_b_k5_mean_diff_mean"}:
            return "R3"
        if name.startswith("e8_gset_"):
            return "R3"
        if re.match(r"f2_constrained_(P25|median|P75)_k[23]_mean_diff$", name):
            return "R3"
        if name in {"cluster_oob_top3_coverage_diff", "cluster_oob_top5_regret_diff_mean"}:
            return "R3"
    if status == "SENSITIVITY" and re.match(r"(structural|behavioral)_diversity_k3_diff_vs_ranking_mean$", name):
        return "R3"

    # R4: the specified type, screening, concentration, parsimony, and determinacy estimates.
    if name.startswith(("pf_g1_", "pf_g1b_", "pf_g2_", "pf_g4_", "pf_h1_null_95_")):
        return "R4"
    if name == "basin_attribute_screening_min_q_value" or name.startswith("top5_models_share_of_total_delta_kge") or name.startswith("top5pct_basins_share_of_total_delta_kge"):
        return "R4"
    if name.startswith(("independent_recompute_nparam_vs_", "partial_spearman_param_", "inter_metric_spearman_")):
        return "R4"
    if name.startswith("parsimony_penalty_mean_diff_") or name.startswith("a3_idjoin_parsimony_mean_diff_"):
        return "R4"
    if name.startswith("a5_idjoin_displacement_vs_oob_loss_") or name.startswith("displacement_vs_oob_loss_"):
        return "R4"
    return ""


def claim_order(section: str, name: str) -> Tuple[int, str]:
    if section == "R1":
        if name.startswith("pooled_delta") or name.startswith("models_positive"):
            return (0, name)
        if name.startswith("two_way_anova"):
            return (1, name)
        if "rank_spearman" in name:
            return (2, name)
        if name.startswith("equivalence_set_size"):
            return (3, name)
        if name.startswith("minimal_cover"):
            return (4, name)
        if name.startswith("multiyear_3yr"):
            return (5, name)
        if name.startswith("bootstrap_cover_ci"):
            return (6, name)
        return (7, name)
    if section == "R2":
        if "candidate" in name:
            return (0, name)
        if name.startswith("optimal_solutions_count"):
            return (1, name)
        if name.startswith("exact_minimum_cover"):
            return (2, name)
        if "oob" in name and "cover" in name and ("member" in name or "model" in name):
            return (3, name)
        if "bottleneck" in name:
            return (4, name)
        if "singleton" in name:
            return (5, name)
        if "coverage" in name:
            return (6, name)
        if "regret" in name:
            return (7, name)
        return (8, name)
    if section == "R3":
        if name.startswith(("rules_ladder_", "t3_2_fixed_ensemble_")):
            fam_match = re.search(r"(?:rules_ladder_|ensemble_)([ABCDE])?_?k(\d+)", name)
            return (0, f"{fam_match.group(1) if fam_match else ''}:{int(fam_match.group(2)) if fam_match else 0:03d}:{name}")
        if name.startswith(("t3_1_", "e8_rule_c_vs_b_")):
            return (1, name)
        if name.startswith("e8_gset"):
            return (2, name)
        if "diversity" in name:
            return (3, name)
        if name.startswith("f2_constrained"):
            return (4, name)
        if "cluster_oob" in name:
            return (6, name)
        return (5, name)
    if section == "R4":
        if name.startswith(("pf_g1_", "pf_g1b_")):
            return (0 if "kruskal" in name or "pairwise" in name or "median" in name else 1, name)
        if name == "basin_attribute_screening_min_q_value":
            return (2, name)
        if name.startswith("pf_g2_"):
            return (3, name)
        if name.startswith("top5"):
            return (4, name)
        if name.startswith("pf_h1_null"):
            return (5, name)
        if "parsimony" in name or name.startswith("pf_g4"):
            return (6, name)
        return (7, name)
    return (99, name)


def build_registry() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    _, citable = read_csv_rows(CITABLE_PATH)
    _, excluded = read_csv_rows(EXCLUDED_PATH)
    _, master_rows = read_csv_rows(MASTER_PATH)
    master_by_name = {r.get("quantity_name", ""): r for r in master_rows}
    all_entries = citable + excluded
    selected = []
    for entry in all_entries:
        if entry.get("status") == "SUPERSEDED":
            continue
        section = selected_section(entry)
        if not section:
            continue
        source_row, _ = find_source_row(entry)
        source_status = source_row_status(entry, source_row)
        row_key = source_row_key(entry, source_row, source_status)
        master = master_by_name.get(entry.get("estimand", ""), {})
        value = entry.get("value", "")
        source_column = entry.get("column_or_key", "")
        # The pooled-median row carries its frozen CI endpoints in the same master-manifest record.
        if entry.get("estimand") == "pooled_delta_kge_pub_median":
            ci = re.search(r"95% CI\s*\[([^]]+)\]", master.get("notes", ""))
            if ci:
                value = f"median={value};CI95=[{ci.group(1)}]"
                source_column = f"{source_column};master_manifest.notes[CI95]"
        sign, direction = ("unresolved", "unresolved")
        claim_slug = slugify(entry.get("estimand", ""))
        selected.append({
            "section": section,
            "claim_id": f"{section}:{claim_slug}",
            "claim_slug": claim_slug,
            "estimand_name": entry.get("estimand", ""),
            "value": value,
            "source_file": entry.get("file_path", ""),
            "source_column": source_column,
            "source_row_key": row_key,
            "source_row_keys": row_key,
            "is_composite": "false",
            "source_row_status": source_status,
            "status": entry.get("status", ""),
            "freeze_phase": entry.get("freeze_phase", ""),
            "population_N": "unresolved",
            "sign_convention": sign,
            "better_direction": direction,
            "primary_or_secondary": "unresolved",
            "venue": "undecided",
        })
    override_resolutions = apply_declaration_fields(selected)
    write_rows(TABLES_ROOT / "population_n_override_resolution.csv", POPULATION_OVERRIDE_RESOLUTION_FIELDS, override_resolutions)
    selected.sort(key=lambda r: (int(r["section"][1:]), claim_order(r["section"], r["estimand_name"])))

    prompt_gaps = []
    manifest_names = [r.get("estimand", "") for r in all_entries if r.get("status") != "SUPERSEDED"]
    for section, gap_id, pattern in REQUIRED_NAME_CHECKS:
        if not any(re.search(pattern, name, flags=re.I) for name in manifest_names):
            prompt_gaps.append({"section": section, "gap_id": gap_id, "pattern": pattern})
    write_rows(TABLES_ROOT / "results_claim_registry.csv", REGISTRY_FIELDS, selected)
    return selected, all_entries, prompt_gaps


def multi_spec_groups(registry: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in registry:
        name = row["estimand_name"]
        match = re.match(r"minimal_cover_(IC|seen-dPL|OOB-dPL)_target_(90|95|100)pct$", name)
        if match:
            groups[f"cover:{match.group(1)}:{match.group(2)}"].append(row)
            continue
        match = re.match(r"multiyear_3yr_block_boot_cover_(IC|seen-dPL|OOB-dPL)_(90|95|100)%$", name)
        if match:
            groups[f"cover:{match.group(1)}:{match.group(2)}"].append(row)
            continue
        match = re.match(r"uncertainty_bandwidth_(IC|seen-dPL|OOB-dPL)_(1yr_block_bootstrap_primary|3yr_multiyear_block_bootstrap_c1)_median$", name)
        if match:
            groups[f"bandwidth:{match.group(1)}"].append(row)
            continue
        match = re.match(r"e8_gset_(A1b_SECONDARY|A2d_PRIMARY)_k(\d+)_mean$", name)
        if match:
            groups[f"gset:{match.group(2)}"].append(row)
    return {key: value for key, value in groups.items() if len(value) > 1}


def declaration_tables() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    _, primary_rows = read_csv_rows(DECLARATIONS_ROOT / "primary_specification.csv")
    _, direction_rows = read_csv_rows(DECLARATIONS_ROOT / "sign_and_direction.csv")
    _, taxonomy_rows = read_csv_rows(DECLARATIONS_ROOT / "population_taxonomy.csv")
    primary = {row["family"]: row for row in primary_rows}
    direction = {row["quantity_class"]: row for row in direction_rows}
    taxonomy = {row["n_value"]: row for row in taxonomy_rows}
    return primary, direction, taxonomy


def quantity_class_for_claim(row: Dict[str, str]) -> str:
    slug = row.get("claim_slug", "") or slugify(row.get("estimand_name", ""))
    return quantity_class_from_slug(slug)

def _role_from_spec(declaration: Dict[str, str], observed_spec: Optional[str]) -> str:
    if not observed_spec:
        return "unresolved"
    if observed_spec == declaration.get("manuscript_primary"):
        return "primary"
    secondary = declaration.get("secondary_retained", "")
    if secondary not in {"", "none"} and observed_spec == secondary:
        return "secondary"
    return "unresolved"


def s1_family_for_claim(row: Dict[str, str]) -> Optional[str]:
    name = row.get("estimand_name", "").lower()
    source = " ".join((row.get("source_file", ""), row.get("source_column", ""), row.get("source_row_key", ""))).lower()
    if "global_seen_vs_oob_rank_spearman" in name:
        return "global_seen_oob_rank_spearman"
    if re.match(r"(?:minimal_cover_|multiyear_3yr_block_boot_cover_|bootstrap_cover_ci_|equivalence_set_size_median_)", name):
        return "equivalence_and_cover"
    if name.startswith("uncertainty_bandwidth_"):
        return "uncertainty_bandwidth"
    if name.startswith("pooled_delta_kge_") or name == "models_positive_delta_kge_fraction" or "transfer_penalty" in name:
        return "transfer_penalty_delta_kge"
    if "variance" in name or "anova" in name:
        return "variance_decomposition"
    if "e8_gset_" in name:
        return "gset_heldout"
    if "parsimony" in name and row.get("quantity_class") == "parsimony_penalty":
        return "parsimony_penalty"
    if "dispersion" in name:
        return "equivalence_set_dispersion"
    if name.startswith("t3_2_fixed_ensemble_"):
        return "fixed_global_ensemble_k"
    if "relative_to_b_k2" in source or "relative_to_b_k2" in name or "relative_to_a" in source or "relative_to_a" in name:
        return "paired_difference_baseline"
    if row.get("quantity_class") in {"rule_paired_difference", "constrained_set_paired_difference", "diversity_set_paired_difference", "regret_paired_difference"}:
        return "paired_statistic"
    if re.match(r"rules_ladder_c(?:_|$)", name):
        return "rule_c_k"
    return None


def declared_role_for_claim(row: Dict[str, str], primary: Dict[str, Dict[str, str]]) -> str:
    name = row.get("estimand_name", "")
    lower = name.lower()
    source = " ".join((row.get("source_file", ""), row.get("source_column", ""), row.get("source_row_key", ""))).lower()
    family = s1_family_for_claim(row)
    if family is None:
        return "not_applicable" if row.get("quantity_class") == "categorical_identifier" else "single_specification"
    if family == "rule_c_k":
        return "single_specification"
    if family == "paired_statistic":
        return _role_from_spec(primary.get(family, {}), "paired_difference")
    if family == "equivalence_and_cover":
        if name.startswith("minimal_cover_"):
            return _role_from_spec(primary.get(family, {}), "one_year_waterblock_q05")
        if name.startswith("multiyear_3yr_block_boot_cover_"):
            return _role_from_spec(primary.get(family, {}), "three_year_nonoverlapping_block")
        if name.startswith("equivalence_set_size_median_"):
            return "primary"
        return "unresolved"
    if family == "uncertainty_bandwidth":
        match = re.search(r"_(1yr_block_bootstrap_primary|3yr_multiyear_block_bootstrap_c1)_median$", name)
        observed = "one_year_waterblock" if match and match.group(1) == "1yr_block_bootstrap_primary" else "three_year_nonoverlapping_block" if match else None
        return _role_from_spec(primary.get(family, {}), observed)
    if family == "gset_heldout":
        match = re.search(r"e8_gset_(A1b_SECONDARY|A2d_PRIMARY)_k\d+_mean$", name)
        observed = "A1b" if match and match.group(1).startswith("A1b") else "A2d" if match else None
        return _role_from_spec(primary.get(family, {}), observed)
    if family == "global_seen_oob_rank_spearman":
        observed = "canonical" if "canonical" in lower or "canonical" in source else "same_pipeline" if "same_pipeline" in lower or "samepipeline" in source else None
        return _role_from_spec(primary.get(family, {}), observed)
    if family == "transfer_penalty_delta_kge":
        observed = "cross_pipeline" if "cross_pipeline" in source or "cross_pipeline" in lower else "same_pipeline_seen_minus_heldout" if "same_pipeline" in source or "same_pipeline" in lower or "delta_pub" in source or "delta_kge_pub" in lower else None
        return _role_from_spec(primary.get(family, {}), observed)
    if family == "variance_decomposition":
        if "dpl_ic" in source or "8_model" in source or "eight_model" in source:
            return "unresolved"
        if "delta_kge_pub" in source or "delta_kge_pub" in lower or "sec31_variance_decomposition.csv" in source:
            return _role_from_spec(primary.get(family, {}), "thirtysix_model_delta_kge_pub")
        return "unresolved"
    if family == "parsimony_penalty":
        if "a3" in source or "idjoin" in source or "id_join" in source:
            return _role_from_spec(primary.get(family, {}), "id_joined_ranking_minus_parsimonious")
        return "unresolved"
    if family == "equivalence_set_dispersion":
        if "a3" in source or "idjoin" in source or "id_join" in source:
            return _role_from_spec(primary.get(family, {}), "id_joined")
        return "unresolved"
    if family == "fixed_global_ensemble_k":
        match = re.search(r"fixed_ensemble_k(\d+)", lower)
        selected = re.search(r"k(\d+)", primary.get(family, {}).get("manuscript_primary", ""))
        if match and selected and match.group(1) == selected.group(1):
            return "primary"
        if match and primary.get(family, {}).get("secondary_retained") == "k_sweep":
            return "secondary"
        return "unresolved"
    if family == "paired_difference_baseline":
        if "relative_to_b_k2" in source or "relative_to_b_k2" in lower:
            return _role_from_spec(primary.get(family, {}), "relative_to_B_k2")
        if "relative_to_a" in source or "relative_to_a" in lower:
            return _role_from_spec(primary.get(family, {}), "relative_to_A")
        return "unresolved"
    return "unresolved"

def resolve_population_n_override(claim: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    meaning = override.get("population_meaning", "").strip()
    category = override.get("population_category", "").strip()
    expected_category = {
        "model_by_basin_cells": "population",
        "full_basin_domain": "population",
        "model_count": "analysis_unit_count",
        "not_applicable": "not_applicable",
    }
    if meaning not in expected_category or category != expected_category[meaning]:
        raise ValueError(f"invalid population_N override for {claim.get('claim_slug')}: {category}/{meaning}")

    if meaning == "model_by_basin_cells":
        source_file = TABLE_PATHS["variance"]
        _, source_rows = table_data(source_file)
        matches = [source for source in source_rows if source.get("component", "").strip().lower() == "total"]
        if len(matches) != 1:
            raise ValueError(f"expected one total variance-decomposition row, got {len(matches)}")
        df = as_decimal(matches[0].get("df", ""))
        if df is None or df != df.to_integral_value():
            raise ValueError("total variance-decomposition df is not an integer")
        population_n = str(int(df) + 1)
        source_column = "df"
        source_row_key = "component=total"
        source_row_status = "unit_table_row"
        resolution_rule = "N derived as total model×basin degrees of freedom plus one"
        validation = "unique total row; integral df"
    elif meaning == "full_basin_domain":
        source_file = "sec32_library_members/tables/sec32_basin_singleton_status.csv"
        fields, source_rows = table_data(source_file)
        if "basin_id" not in fields or "condition" not in fields:
            raise ValueError("basin singleton status table lacks basin_id/condition")
        basin_ids = canonical_basin_ids()
        for condition in CONDITIONS:
            condition_rows = [source for source in source_rows if source.get("condition", "") == condition]
            ids = [source.get("basin_id", "") for source in condition_rows]
            aligned = align_series_by_id(ids, list(range(len(ids))), canonical=basin_ids, name=f"population_N_{condition}")
            if len(aligned) != len(basin_ids):
                raise ValueError(f"basin ID alignment length mismatch for {condition}")
        population_n = str(len(basin_ids))
        source_column = "basin_id"
        source_row_key = "distinct basin_id set per condition; validated against canonical basin IDs"
        source_row_status = "ID-aligned"
        resolution_rule = "N derived from canonical basin ID count after condition-wise basin_accessor alignment"
        validation = "all conditions align one-to-one to the canonical basin ID set"
    elif meaning == "model_count":
        source_file = TABLE_PATHS["model_fold_delta"]
        fields, source_rows = table_data(source_file)
        if "model" not in fields or not source_rows:
            raise ValueError("model-fold table lacks model identifiers")
        models = {source.get("model", "").strip() for source in source_rows if source.get("model", "").strip()}
        population_n = str(len(models))
        source_column = "model"
        source_row_key = "distinct model identifiers in model-fold table"
        source_row_status = "distinct_source_values"
        resolution_rule = "N derived by counting distinct model IDs in the frozen model-fold table"
        validation = f"nonempty model ID set; rows={len(source_rows)}"
    else:
        if claim.get("source_row_status") not in {"matched", "aggregate_expression", "unique_source_tuple"}:
            raise ValueError(f"not_applicable override lacks a resolved categorical source row: {claim.get('claim_id')}")
        population_n = "not_applicable"
        source_file = claim.get("source_file", "")
        source_column = claim.get("source_column", "")
        source_row_key = claim.get("source_row_key", "")
        source_row_status = claim.get("source_row_status", "")
        resolution_rule = "categorical identifier; no numeric population N"
        validation = "resolved source row supports a categorical identifier"

    return {
        "claim_id": claim.get("claim_id", ""),
        "claim_slug": claim.get("claim_slug", ""),
        "population_category": category,
        "population_meaning": meaning,
        "population_N_value": population_n,
        "claim_source_file": claim.get("source_file", ""),
        "population_N_source_file": source_file,
        "population_N_source_column": source_column,
        "population_N_source_row_key": source_row_key,
        "population_N_source_row_status": source_row_status,
        "claim_status": claim.get("status", ""),
        "claim_freeze_phase": claim.get("freeze_phase", ""),
        "resolution_rule": f"{override.get('basis_note', '').strip()}; {resolution_rule}".strip(" ;"),
        "validation": validation,
    }


def apply_declaration_fields(registry: List[Dict[str, str]]) -> List[Dict[str, str]]:
    primary, directions, _taxonomy = declaration_tables()
    _, e5_rows = read_csv_rows(TABLES_ROOT / "evidence" / "E5_population_N_evidence.csv")
    _, override_rows = read_csv_rows(DECLARATIONS_ROOT / "population_n_overrides.csv")
    overrides = {row.get("claim_slug", ""): row for row in override_rows}
    if len(overrides) != len(override_rows):
        raise ValueError("population_n_overrides.csv contains duplicate claim slugs")
    n_by_claim = defaultdict(set)
    for evidence in e5_rows:
        if evidence.get("record_type") == "resolved_population_N_reference":
            n_value = evidence.get("population_N_value", "").strip()
            if as_decimal(n_value) is not None:
                n_by_claim[evidence.get("claim_id", "")].add(n_value)
    population_classes = {"cover_count", "descriptive_count", "equivalence_set_size"}
    direction_aliases = {
        "cover_count": "cover_count / library_size",
        "spearman": "spearman / retention / coverage_fraction",
        "coverage_fraction": "spearman / retention / coverage_fraction",
        "p_value_or_effect_size": "p_value / effect_size",
    }
    resolutions = []
    matched_overrides = set()
    for row in registry:
        row["quantity_class"] = quantity_class_for_claim(row)
        row["is_composite"] = "false"
        row["source_row_keys"] = row.get("source_row_key", "")
        override = overrides.get(row.get("claim_slug", ""))
        if override:
            resolution = resolve_population_n_override(row, override)
            row["population_N"] = resolution["population_N_value"]
            resolutions.append(resolution)
            matched_overrides.add(row["claim_slug"])
        else:
            candidates = n_by_claim.get(row["claim_id"], set())
            n_value = next(iter(candidates)) if len(candidates) == 1 else ""
            if row.get("section") in {"R1", "R2", "R3"} and row["quantity_class"] in population_classes:
                n_value = str(len(canonical_basin_ids()))
            row["population_N"] = n_value if as_decimal(n_value) is not None else "unresolved"
        row["venue"] = "undecided"
        declaration = directions.get(direction_aliases.get(row["quantity_class"], row["quantity_class"]))
        if declaration:
            row["sign_convention"] = declaration.get("subtraction_order", "")
            row["better_direction"] = declaration.get("better_direction", "")
        else:
            row["sign_convention"] = "unresolved"
            row["better_direction"] = "unresolved"
        row["primary_or_secondary"] = declared_role_for_claim(row, primary)
    unmatched_overrides = sorted(set(overrides) - matched_overrides)
    if unmatched_overrides:
        raise ValueError(f"population_N overrides did not match registry claims: {unmatched_overrides}")
    return resolutions
def fmt_decimal(value: Decimal) -> str:
    return format(value, "f")


def add_audit(audits: List[Dict[str, str]], check_id: str, name: str, passed: bool, evidence: str,
              claim_ids: Sequence[str] = (), values: Sequence[str] = ()) -> None:
    audits.append({
        "check_id": check_id,
        "check_name": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "failed_claim_ids": ";".join(claim_ids),
        "failed_values": ";".join(values),
    })



def recompute_t5_bandwidth() -> Dict[str, List[Dict[str, object]]]:
    """Recompute only the declared primary 1-year bandwidth and its K1/K5 checks."""
    models = S.MODELS_36
    basins = canonical_basin_ids()
    basin_count, model_count = len(basins), len(models)
    raw_path = S.DIR_GAP_CONDITIONAL / "forward" / "test" / "water_year_sufficient_stats.csv"
    raw = pd.read_csv(raw_path, dtype={"basin_id": str})
    model_pos = {model: index for index, model in enumerate(models)}
    basin_pos = {basin: index for index, basin in enumerate(basins)}
    idx_model = raw["model"].map(model_pos).to_numpy()
    idx_basin = raw["basin_id"].astype(str).str.zfill(8).map(basin_pos).to_numpy()
    idx_year = (raw["water_year"].astype(int) - 1996).to_numpy()
    expected_rows = model_count * basin_count * 15
    if len(raw) != expected_rows or raw.duplicated(["model", "basin_id", "water_year"]).any():
        raise ValueError(f"T5 sufficient-stat rows invalid: rows={len(raw)};expected={expected_rows};duplicates={raw.duplicated(['model', 'basin_id', 'water_year']).sum()}")
    if pd.isna(idx_model).any() or pd.isna(idx_basin).any() or ((idx_year < 0) | (idx_year >= 15)).any():
        raise ValueError("T5 sufficient-stat rows contain unrecognized model, basin ID, or water year")
    stat_cols_ic = ["n_ic", "sum_sim_ic", "sum_obs_ic", "sum_sim2_ic", "sum_obs2_ic", "sum_simobs_ic"]
    stat_cols_dp = ["n_dpl", "sum_sim_dpl", "sum_obs_dpl", "sum_sim2_dpl", "sum_obs2_dpl", "sum_simobs_dpl"]
    cube_ic = np.zeros((model_count, basin_count, 15, 6), dtype=np.float64)
    cube_dp = np.zeros_like(cube_ic)
    cube_ic[idx_model, idx_basin, idx_year] = raw[stat_cols_ic].to_numpy(dtype=np.float64)
    cube_dp[idx_model, idx_basin, idx_year] = raw[stat_cols_dp].to_numpy(dtype=np.float64)
    cubes = {"IC": cube_ic, "seen-dPL": cube_dp, "OOB-dPL": S.load_oob_cube()}
    counts = S.load_year_counts(n_reps=S.N_BOOT_BLOCK, seed=S.SEED_BOOT)
    if counts.shape != (S.N_BOOT_BLOCK, 15):
        raise ValueError(f"unexpected T5 resampling-count shape: {counts.shape}")

    _, old_rows = read_csv_rows(table_path(TABLE_PATHS["uncertainty_band"]))
    old_by_key = {}
    for row in old_rows:
        key = (row.get("condition", ""), str(row.get("basin_id", "")).zfill(8))
        if key in old_by_key:
            raise ValueError(f"duplicate old bandwidth key: {key}")
        old_by_key[key] = row
    alignment_rows = read_csv_rows(table_path("sec34_determinacy_and_bounds/tables/sec34_a2_table_alignment_audit.csv"))[1]
    alignment_row = next((row for row in alignment_rows if row.get("table_name") == Path(TABLE_PATHS["uncertainty_band"]).name), {})
    legacy_alignment_status = alignment_row.get("alignment_status", "unresolved")
    precision = max((len(row.get("bandwidth_1yr_block_bootstrap_primary", "").split(".", 1)[1])
                     for row in old_rows if "." in row.get("bandwidth_1yr_block_bootstrap_primary", "")), default=0)

    penalty_path = table_path("sec34_determinacy_and_bounds/tables/sec34_a3_parsimony_penalty_idjoin_per_basin.csv")
    _, penalty_rows = read_csv_rows(penalty_path)
    bandwidth_rows = []
    bound_rows = []
    reconciliation_rows = []
    summary_rows = []
    for condition in CONDITIONS:
        cmat = cubes[condition]
        point = S.kge_from_sufficient(cmat.sum(axis=2))
        best_point = point.max(axis=0)
        q05 = np.empty(basin_count, dtype=np.float64)
        for start in range(0, basin_count, 12):
            stop = min(start + 12, basin_count)
            block = cmat[:, start:stop, :, :]
            flat = block.transpose(2, 1, 0, 3).reshape(15, (stop - start) * model_count * 6)
            pooled = (counts @ flat).reshape(S.N_BOOT_BLOCK, stop - start, model_count, 6)
            best_replicates = S.kge_from_sufficient(pooled).max(axis=2)
            q05[start:stop] = np.quantile(best_replicates, 0.05, axis=0)
        bandwidth = best_point - q05
        eq_mask = point >= q05[None, :]
        point_range = np.empty(basin_count, dtype=np.float64)
        for basin_index in range(basin_count):
            included = point[eq_mask[:, basin_index], basin_index]
            point_range[basin_index] = included.max() - included.min()

        condition_penalty = [row for row in penalty_rows if row.get("condition") == condition]
        if len(condition_penalty) != basin_count or any(row.get("join_method") != "basin_id_keyed" for row in condition_penalty):
            raise ValueError(f"T5 K5 penalty input is incomplete or not ID-keyed: {condition};rows={len(condition_penalty)}")
        penalty_frame = pd.DataFrame(condition_penalty)
        penalty = align_frame_by_id(penalty_frame, "basin_id", ["diff_kge_rank_minus_pars"], canonical=basins,
                                    name=f"T5 ID-joined parsimony penalty ({condition})")[:, 0]
        range_violations = 0
        penalty_violations = 0
        old_mismatches = 0
        max_abs_difference = 0.0
        old_values_seen = 0
        for index, basin_id in enumerate(basins):
            bid = str(basin_id).zfill(8)
            new_width = float(bandwidth[index])
            old = old_by_key.get((condition, bid))
            if not old:
                raise ValueError(f"old bandwidth table missing ID-joined key: {condition}/{bid}")
            old_value = as_float(old.get("bandwidth_1yr_block_bootstrap_primary", ""))
            if old_value is None:
                raise ValueError(f"old bandwidth value is not numeric: {condition}/{bid}")
            old_values_seen += 1
            absolute_difference = abs(new_width - old_value)
            max_abs_difference = max(max_abs_difference, absolute_difference)
            if format(new_width, f".{precision}f") != old.get("bandwidth_1yr_block_bootstrap_primary", ""):
                old_mismatches += 1
            range_violation = bool(point_range[index] > new_width + 1e-12)
            penalty_violation = bool(penalty[index] > new_width + 1e-12)
            range_violations += int(range_violation)
            penalty_violations += int(penalty_violation)
            bandwidth_rows.append({
                "condition": condition, "basin_id": bid,
                "best_point_kge": float(best_point[index]), "q05_threshold": float(q05[index]),
                "bandwidth_1yr_block_bootstrap_primary": new_width,
                "bandwidth_definition": "best_model_point_kge_minus_5th_percentile_of_best_model_across_1000_L1_water_year_block_draws",
                "n_bootstrap_draws": S.N_BOOT_BLOCK, "seed": S.SEED_BOOT,
                "old_table_value": old.get("bandwidth_1yr_block_bootstrap_primary", ""),
                "old_table_status": "SUPERSEDED_FOR_ADJUDICATION;UPSTREAM_FILE_UNMODIFIED",
                "old_table_abs_difference": absolute_difference,
                "legacy_table_alignment_status": legacy_alignment_status,
            })
            bound_rows.append({
                "condition": condition, "basin_id": bid,
                "point_kge_range_within_q05_equivalence_set": float(point_range[index]),
                "bandwidth_1yr_block_bootstrap_primary_recomputed": new_width,
                "range_exceeds_bandwidth": str(range_violation).lower(),
                "range_excess_minus_bandwidth": float(point_range[index] - new_width),
                "parsimony_penalty_idjoined": float(penalty[index]),
                "parsimony_exceeds_bandwidth": str(penalty_violation).lower(),
                "parsimony_excess_minus_bandwidth": float(penalty[index] - new_width),
            })
        reconciliation_rows.append({
            "condition": condition, "id_joined_rows": old_values_seen,
            "old_table_display_precision_decimals": precision,
            "nonidentical_rows_at_stored_precision": old_mismatches,
            "max_absolute_difference_vs_old_rounded_value": max_abs_difference,
            "legacy_alignment_status": legacy_alignment_status,
            "old_table_treatment": "SUPERSEDED_FOR_ADJUDICATION;SOURCE_UNMODIFIED",
        })
        summary_rows.append({
            "condition": condition, "n_basins": basin_count,
            "k1_range_violation_count": range_violations,
            "k5_idjoined_parsimony_violation_count": penalty_violations,
            "recomputed_bandwidth_median": float(np.median(bandwidth)),
            "legacy_alignment_status": legacy_alignment_status,
            "bandwidth_source": relpath(TABLES_ROOT / "S5_C2_06_T5_bandwidth_per_basin.csv"),
            "recomputation_input": relpath(raw_path),
        })
    if len(old_by_key) != basin_count * len(CONDITIONS):
        raise ValueError(f"old bandwidth table key count mismatch: {len(old_by_key)}")
    write_rows(TABLES_ROOT / "S5_C2_06_T5_bandwidth_per_basin.csv", [
        "condition", "basin_id", "best_point_kge", "q05_threshold", "bandwidth_1yr_block_bootstrap_primary",
        "bandwidth_definition", "n_bootstrap_draws", "seed", "old_table_value", "old_table_status",
        "old_table_abs_difference", "legacy_table_alignment_status",
    ], bandwidth_rows)
    write_rows(TABLES_ROOT / "S5_C2_06_T5_bandwidth_reconciliation.csv", [
        "condition", "id_joined_rows", "old_table_display_precision_decimals", "nonidentical_rows_at_stored_precision",
        "max_absolute_difference_vs_old_rounded_value", "legacy_alignment_status", "old_table_treatment",
    ], reconciliation_rows)
    write_rows(TABLES_ROOT / "S5_C2_06_T5_bounds_per_basin.csv", [
        "condition", "basin_id", "point_kge_range_within_q05_equivalence_set",
        "bandwidth_1yr_block_bootstrap_primary_recomputed", "range_exceeds_bandwidth", "range_excess_minus_bandwidth",
        "parsimony_penalty_idjoined", "parsimony_exceeds_bandwidth", "parsimony_excess_minus_bandwidth",
    ], bound_rows)
    write_rows(TABLES_ROOT / "S5_C2_06_T5_bounds_summary.csv", [
        "condition", "n_basins", "k1_range_violation_count", "k5_idjoined_parsimony_violation_count",
        "recomputed_bandwidth_median", "legacy_alignment_status", "bandwidth_source", "recomputation_input",
    ], summary_rows)
    return {"bandwidth_rows": bandwidth_rows, "bounds_rows": bound_rows, "reconciliation_rows": reconciliation_rows, "summary_rows": summary_rows}


def recompute_c2_07_source_pairing() -> List[Dict[str, object]]:
    """Pair the three reported G_set cells to the same seed/k source rows."""
    summary_path = table_path(TABLE_PATHS["gset"])
    _, summary_rows = read_csv_rows(summary_path)
    paired_path = Path(S.DIR_OOB36_PAIRED_REGRET) / "tables" / "paired_regret_per_seed.csv"
    identity_path = Path(S.DIR_OOB36_PAIRED_REGRET) / "tables" / "d1_identity_parity.csv"
    gate_tables = Path(S.DIR_OOB36_GATE) / "tables"
    paired_rows = read_csv_rows(paired_path)[1]
    identity_rows = read_csv_rows(identity_path)[1]
    specs = {
        "Seen-dPL (t2 baseline)": ("Seen", "parity_p1_seen36_per_seed.csv", "REFERENCE_CEILING"),
        "A2d (Seen-trained, OOB-scored)": ("A2d", "gset_36_A2d_per_seed.csv", "PRIMARY"),
        "A1b (OOB-trained, OOB-scored)": ("A1b", "gset_36_A1b_per_seed.csv", "SECONDARY_SENSITIVITY"),
    }
    output = []
    for summary in summary_rows:
        if summary.get("k") != "3" or summary.get("estimand") not in specs:
            continue
        case, gate_name, _role = specs[summary["estimand"]]
        seed = summary.get("main_seed", "")
        key = (case, seed, summary.get("k", ""))
        paired = next((row for row in paired_rows if (row.get("case"), row.get("seed"), row.get("k")) == key), None)
        identity = next((row for row in identity_rows if (row.get("case"), row.get("seed"), row.get("k")) == key), None)
        if not paired or not identity:
            raise ValueError(f"C2-07 verified source row missing: {key}")
        gate_path = gate_tables / gate_name
        gate_rows = read_csv_rows(gate_path)[1]
        gate = next((row for row in gate_rows if row.get("seed") == seed), None)
        if not gate:
            raise ValueError(f"C2-07 reference source row missing: {gate_path.name};seed={seed}")
        components = [as_float(summary.get(column, "")) for column in ("mean_best_R", "mean_best_F", "mean_VBS")]
        stored = as_float(summary.get("G_set", ""))
        paired_value = as_float(paired.get("gset_from_regret", ""))
        reference_value = as_float(gate.get("G_set", ""))
        identity_diff = as_float(identity.get("abs_diff", ""))
        identity_left = as_float(identity.get("gset_from_regret", ""))
        identity_right = as_float(identity.get("gset_reference", ""))
        n_basins = as_decimal(summary.get("n_basins", ""))
        paired_n = as_decimal(paired.get("n_basins", ""))
        if any(value is None for value in components) or None in (stored, paired_value, reference_value, identity_diff, identity_left, identity_right) or n_basins is None or paired_n is None:
            raise ValueError(f"C2-07 source row contains missing numeric evidence: {key}")
        best_r, best_f, vbs = components
        regret_r, regret_f = vbs - best_r, vbs - best_f
        displayed_formula = 1.0 - regret_r / regret_f if regret_f else float("nan")
        precision = len(summary.get("G_set", "").split(".", 1)[1]) if "." in summary.get("G_set", "") else 0
        display_tolerance = 0.5 * (10 ** (-precision))
        verified_tolerance = display_tolerance
        summary_to_pair_diff = abs(stored - paired_value)
        identity_source_match = abs(identity_left - paired_value) <= 1e-15 and abs(identity_right - reference_value) <= 1e-15
        verified_ok = (
            summary_to_pair_diff <= verified_tolerance
            and identity_source_match
            and identity_diff <= verified_tolerance
            and abs(paired_value - reference_value) <= verified_tolerance
            and n_basins == Decimal(str(len(canonical_basin_ids()))) and paired_n == n_basins
            and summary.get("k") == "3"
            and identity.get("seed") == seed and paired.get("seed") == seed
        )
        formula_abs_difference = abs(displayed_formula - stored)
        summary_key = f"estimand={summary['estimand']};k={summary['k']};main_seed={seed};n_basins={summary['n_basins']}"
        paired_key = f"case={case};seed={seed};k={summary['k']};n_basins={paired.get('n_basins', '')}"
        gate_key = f"seed={seed};k={summary['k']} (k=3 fixed by source table scope)"
        output.append({
            "identity": "G_set_formula", "estimand": summary["estimand"], "case": case,
            "k": summary["k"], "seed": seed, "n_basins": n_basins,
            "prior_stored_gset": stored, "prior_summary_formula_gset": displayed_formula,
            "prior_summary_abs_difference": formula_abs_difference,
            "prior_summary_precision_decimals": precision, "prior_summary_tolerance": display_tolerance,
            "prior_left_source_file": relpath(summary_path), "prior_left_row_key": summary_key,
            "prior_left_aggregation": "per-seed G_set derived from full-precision per-basin regrets; summary cell displayed to source precision",
            "prior_right_source_file": relpath(summary_path), "prior_right_row_key": summary_key,
            "prior_right_aggregation": "formula applied to displayed mean_best_R, mean_best_F, mean_VBS summary cells",
            "verified_left_source_file": relpath(paired_path), "verified_left_row_key": paired_key,
            "verified_left_aggregation": "1 - mean_regret_R / mean_regret_F over 531 paired basin rows; existing full-precision output",
            "verified_left_value": paired_value,
            "verified_right_source_file": relpath(gate_path), "verified_right_row_key": gate_key,
            "verified_right_aggregation": "existing per-seed G_set reference output; k=3 source-table scope",
            "verified_right_value": reference_value,
            "verified_abs_difference": identity_diff, "verified_summary_abs_difference": summary_to_pair_diff,
            "verified_direct_abs_difference": abs(paired_value - reference_value),
            "identity_table_gset_from_regret": identity_left, "identity_table_gset_reference": identity_right,
            "identity_table_abs_diff": identity_diff, "identity_table_source_matches_verified_rows": str(identity_source_match).lower(),
            "verified_tolerance": verified_tolerance,
            "source_pairing_status": "PASS_SAME_CASE_SEED_K_N" if verified_ok else "FAIL_SOURCE_PAIRING",
            "prior_formula_status": "MISMATCH_AT_DISPLAY_PRECISION" if formula_abs_difference > display_tolerance else "MATCH_AT_DISPLAY_PRECISION",
        })
    if len(output) != 3:
        raise ValueError(f"C2-07 expected exactly 3 k=3 G_set source pairs, got {len(output)}")
    fields = list(output[0])
    write_rows(TABLES_ROOT / "S5_C2_07_gset_source_pairing.csv", fields, output)
    prior_failures = [row for row in output if row["prior_formula_status"] == "MISMATCH_AT_DISPLAY_PRECISION"]
    write_rows(TABLES_ROOT / "S5_C2_07_gset_formula_failures.csv", fields, prior_failures)
    return output

def build_manifest_source_key_index(all_entries: List[Dict[str, str]]):
    accepted = {"matched", "aggregate_expression", "unique_source_tuple"}
    by_file_key = defaultdict(list)
    by_key = defaultdict(list)
    for entry in all_entries:
        if entry.get("status") == "SUPERSEDED":
            continue
        source_row, _ = find_source_row(entry)
        status = source_row_status(entry, source_row)
        if status not in accepted:
            continue
        key = source_row_key(entry, source_row, status)
        record = {"entry": entry, "source_row_status": status, "source_row_key": key}
        by_file_key[(entry.get("file_path", ""), key)].append(record)
        by_key[key].append(record)
    return by_file_key, by_key


def source_key_tokens(row: Dict[str, str]) -> List[str]:
    raw = row.get("source_row_keys", "").strip()
    if not raw:
        return []
    if str(row.get("is_composite", "false")).strip().lower() == "true":
        return [token.strip() for token in raw.split(";") if token.strip()]
    return [raw]


def manifest_matches_for_token(row: Dict[str, str], token: str, by_file_key, by_key):
    if "::" in token:
        source_file, key = token.split("::", 1)
        return by_file_key.get((source_file, key), [])
    files = [part.strip() for part in row.get("source_file", "").split(";") if part.strip()]
    if files:
        return [record for source_file in files for record in by_file_key.get((source_file, token), [])]
    return by_key.get(token, [])


def composite_component_signature(row: Dict[str, str], by_file_key, by_key):
    tokens = source_key_tokens(row)
    if not tokens:
        return None
    columns = [part.strip() for part in row.get("source_column", "").split(";") if part.strip()]
    signature = []
    for index, token in enumerate(tokens):
        column = columns[index] if len(columns) == len(tokens) else ""
        matches = manifest_matches_for_token(row, token, by_file_key, by_key)
        if column:
            matches = [record for record in matches if record["entry"].get("column_or_key", "").strip() == column]
        values = {record["entry"].get("value", "") for record in matches}
        if len(values) != 1:
            return None
        signature.append((token, column, next(iter(values))))
    return tuple(sorted(signature))


def source_manifest_checks(registry: List[Dict[str, str]], all_entries: List[Dict[str, str]], prompt_gaps: List[Dict[str, str]]) -> List[Dict[str, str]]:
    audits: List[Dict[str, str]] = []
    by_name = defaultdict(list)
    for entry in all_entries:
        by_name[entry.get("estimand", "")].append(entry)

    manifest_by_file_key, manifest_by_key = build_manifest_source_key_index(all_entries)
    orphans = [r for r in registry if not by_name.get(r["estimand_name"])]
    unresolved_source_rows = []
    unresolved_components = []
    noncomposite_key_mismatches = []
    for row in registry:
        composite = str(row.get("is_composite", "false")).strip().lower() == "true"
        tokens = source_key_tokens(row)
        if not composite:
            if row.get("source_row_keys", "") != row.get("source_row_key", ""):
                noncomposite_key_mismatches.append(row)
            if row.get("source_row_status") not in {"matched", "aggregate_expression", "unique_source_tuple"} or tokens != [row.get("source_row_key", "")]:
                unresolved_source_rows.append(row)
                continue
            if not tokens[0] or not manifest_matches_for_token(row, tokens[0], manifest_by_file_key, manifest_by_key):
                unresolved_source_rows.append(row)
        else:
            if not tokens:
                unresolved_source_rows.append(row)
                continue
            missing = [token for token in tokens if not manifest_matches_for_token(row, token, manifest_by_file_key, manifest_by_key)]
            if missing:
                unresolved_source_rows.append(row)
                unresolved_components.extend((row, token) for token in missing)

    gaps_text = ";".join(f"{x['section']}:{x['gap_id']}" for x in prompt_gaps)
    failed_c201 = {r["claim_id"]: r for r in orphans + unresolved_source_rows + noncomposite_key_mismatches}
    add_audit(audits, "C2-01", "manifest_row_resolution", not failed_c201 and not prompt_gaps,
              f"registry_rows={len(registry)};orphan_claims={len(orphans)};unresolved_source_rows={len(unresolved_source_rows)};"
              f"unresolved_composite_components={len(unresolved_components)};noncomposite_source_key_mismatches={len(noncomposite_key_mismatches)};"
              f"unmapped_requested_estimands={len(prompt_gaps)};gaps={gaps_text}",
              sorted(failed_c201), [failed_c201[key]["value"] for key in sorted(failed_c201)])

    superseded = [r for r in registry if r.get("status") == "SUPERSEDED"]
    loaded = [r for r in registry if r.get("status") == "LOADED"]
    loaded_evidence = ";".join(f"{r['claim_id']}={r['value']}" for r in loaded)
    add_audit(audits, "C2-02", "superseded_and_loaded_status", not superseded,
              f"superseded_claims={len(superseded)};loaded_claims={len(loaded)};loaded={loaded_evidence}",
              [r["claim_id"] for r in superseded], [r["value"] for r in superseded])

    duplicate_failures = []
    by_estimand = defaultdict(list)
    for row in registry:
        by_estimand[row["estimand_name"]].append(row)
    component_groups_checked = 0
    for estimand, rows in by_estimand.items():
        if len(rows) <= 1:
            continue
        if any(str(row.get("is_composite", "false")).strip().lower() == "true" for row in rows):
            signatures = []
            unresolved_signature = False
            for row in rows:
                if str(row.get("is_composite", "false")).strip().lower() == "true":
                    signature = composite_component_signature(row, manifest_by_file_key, manifest_by_key)
                else:
                    tokens = source_key_tokens(row)
                    signature = tuple(sorted((token, row.get("source_column", ""), row.get("value", "")) for token in tokens)) if tokens else None
                if signature is None:
                    unresolved_signature = True
                else:
                    signatures.append(signature)
            component_groups_checked += 1
            if unresolved_signature or len(signatures) != len(rows) or len(set(signatures)) > 1:
                duplicate_failures.extend(rows)
        elif len({row["value"] for row in rows}) > 1:
            duplicate_failures.extend(rows)
    duplicate_failures.extend(noncomposite_key_mismatches)
    duplicate_failures = list({row["claim_id"]: row for row in duplicate_failures}.values())
    add_audit(audits, "C2-03", "same_estimand_value_identity", not duplicate_failures,
              f"repeated_estimands={sum(1 for rows in by_estimand.values() if len(rows)>1)};composite_claim_rows={sum(str(r.get('is_composite','false')).lower() == 'true' for r in registry)};"
              f"component_groups_checked={component_groups_checked};noncomposite_source_keys_match={sum(str(r.get('is_composite','false')).lower() != 'true' for r in registry)-len(noncomposite_key_mismatches)}/{sum(str(r.get('is_composite','false')).lower() != 'true' for r in registry)};"
              f"conflicting_claim_rows={len(duplicate_failures)}",
              [r["claim_id"] for r in duplicate_failures], [r["value"] for r in duplicate_failures])

    multi = multi_spec_groups(registry)
    _, primary_rows = read_csv_rows(DECLARATIONS_ROOT / "primary_specification.csv")
    primary, _, _ = declaration_tables()
    family_counts = Counter(row.get("family", "") for row in primary_rows)
    declarations_valid = (
        len(primary_rows) == len(PRIMARY_SPECIFICATION_ROWS)
        and all(family_counts.get(row["family"], 0) == 1 and row.get("manuscript_primary", "").strip() for row in primary_rows)
    )
    multi_details = []
    multi_failures = []
    passed_groups = 0
    details = []
    for group, rows in sorted(multi.items()):
        if group.startswith("cover:"):
            declaration_family = "equivalence_and_cover"
        elif group.startswith("bandwidth:"):
            declaration_family = "uncertainty_bandwidth"
        elif group.startswith("gset:"):
            declaration_family = "gset_heldout"
        else:
            declaration_family = ""
        declaration_exists = declaration_family in primary
        expected = {r["claim_id"]: declared_role_for_claim(r, primary) for r in rows}
        selected = [r for r in rows if r["primary_or_secondary"] == "primary"]
        roles_match = declaration_exists and all(
            expected[r["claim_id"]] in {"primary", "secondary"}
            and r["primary_or_secondary"] == expected[r["claim_id"]]
            for r in rows
        )
        family_valid = declaration_exists and len(selected) == 1 and roles_match
        if family_valid:
            passed_groups += 1
        row_records = []
        for row in rows:
            marked = family_valid and row["primary_or_secondary"] == "primary"
            row_records.append({
                "family_id": group,
                "estimand_name": row["estimand_name"],
                "value": row["value"],
                "primary_or_secondary": row["primary_or_secondary"],
                "manuscript_selected": "true" if marked else "false",
                "source_file": row["source_file"],
                "source_row_key": row["source_row_key"],
            })
        multi_details.extend(row_records)
        if not family_valid:
            multi_failures.extend(rows)
        details.append(group + ":declaration=" + (declaration_family or "unmapped") + ";" + ",".join(
            "{}={}[{}]".format(r["estimand_name"], r["value"], r["primary_or_secondary"]) for r in rows
        ))
    write_rows(TABLES_ROOT / "multi_estimand_audit.csv",
               ["family_id", "estimand_name", "value", "primary_or_secondary", "manuscript_selected", "source_file", "source_row_key"], multi_details)
    prohibited_variance_refs = [
        row for row in registry
        if ("variance" in (row["estimand_name"] + row["source_column"]).lower() or "anova" in (row["estimand_name"] + row["source_column"]).lower())
        and re.search(r"(?:8|eight)[-_ ]?models?|dpl[-_]ic", " ".join(row.get(k, "") for k in ("estimand_name", "source_file", "source_column", "source_row_key")).lower())
    ]
    c204_pass = declarations_valid and not multi_failures and not prohibited_variance_refs
    multi_evidence = (
        f"declaration_families={len(primary_rows)};declarations_valid={declarations_valid};multi_spec_families={len(multi)};"
        f"families_matching_declared_primary={passed_groups};prohibited_8model_dpl_ic_claims={len(prohibited_variance_refs)};"
        f"details={'|'.join(details)}"
    )
    add_audit(audits, "C2-04", "declared_primary_specification_and_multi_spec_claims", c204_pass,
              multi_evidence,
              [r["claim_id"] for r in multi_failures + prohibited_variance_refs],
              [r["value"] for r in multi_failures + prohibited_variance_refs] + ([] if declarations_valid else ["invalid_primary_specification_declarations"]))

    monotone_issues = []
    cover_rows = table_data(TABLE_PATHS["cover_primary"])[1]
    by_condition = defaultdict(dict)
    for r in cover_rows:
        pct = int(round(float(r["target"]) * 100))
        by_condition[r["condition"]][pct] = int(r["n_models_exact"])
    for cond, vals in by_condition.items():
        ordered = [vals.get(p) for p in (90, 95, 100)]
        if None in ordered or not (ordered[0] <= ordered[1] <= ordered[2]):
            monotone_issues.append(f"cover:{cond}:{ordered}")

    ladder = table_data(TABLE_PATHS["coverage_ladder"])[1]
    for strategy in sorted({r["strategy"] for r in ladder}):
        rows = sorted((r for r in ladder if r["strategy"] == strategy), key=lambda r: int(r["k"]))
        vals = [as_float(r["coverage_top1"]) for r in rows]
        if any(v is None for v in vals) or any(vals[i] > vals[i+1] for i in range(len(vals)-1)):
            monotone_issues.append(f"coverage_ladder:{strategy}:{vals}")

    regrets = table_data(TABLE_PATHS["regret_curve"])[1]
    for strategy in sorted({r["strategy"] for r in regrets}):
        rows = sorted((r for r in regrets if r["strategy"] == strategy), key=lambda r: int(r["k"]))
        vals = [as_float(r["regret_median"]) for r in rows]
        if any(v is None for v in vals) or any(vals[i] < vals[i+1] for i in range(len(vals)-1)):
            monotone_issues.append(f"regret_curve:{strategy}:{vals}")

    opt_summary = table_data(TABLE_PATHS["optimal_summary"])[1]
    opt_solutions = table_data(TABLE_PATHS["optimal_solutions"])[1]
    for summary in opt_summary:
        key = (summary["condition"], summary["target_pct"])
        core = set(filter(None, summary["core_set_models"].split("|")))
        union = set(filter(None, summary["union_set_models"].split("|")))
        solutions = [r for r in opt_solutions if (r["condition"], r["target_pct"]) == key]
        for solution in solutions:
            members = set(filter(None, solution["solution_models"].split("|")))
            if not core.issubset(members) or not members.issubset(union):
                monotone_issues.append(f"optimal_sets:{key}:{solution['solution_id']}:core={sorted(core-members)};union={sorted(members-union)}")
        if not solutions:
            monotone_issues.append(f"optimal_sets:{key}:no_solutions")

    singleton_rows = table_data(TABLE_PATHS["singleton"])[1]
    for row in singleton_rows:
        n = as_decimal(row.get("total_singletons_idjoin", ""))
        total = Decimal("531")
        if n is None or n > total:
            monotone_issues.append(f"singleton:{row.get('condition')}:{row.get('total_singletons_idjoin')}>531")
    add_audit(audits, "C2-05", "coverage_ladder_regret_and_set_inclusions", not monotone_issues,
              f"coverage_conditions={len(by_condition)};coverage_strategies={len({r['strategy'] for r in ladder})};regret_strategies={len({r['strategy'] for r in regrets})};optimal_solutions={len(opt_solutions)};singleton_conditions={len(singleton_rows)};issues={';'.join(monotone_issues)}",
              (), monotone_issues)

    bounds_summary = read_csv_rows(TABLES_ROOT / "S5_C2_06_T5_bounds_summary.csv")[1]
    bound_failures = [
        row for row in bounds_summary
        if as_decimal(row.get("k1_range_violation_count", "")) is None
        or as_decimal(row.get("k5_idjoined_parsimony_violation_count", "")) is None
        or as_decimal(row["k1_range_violation_count"]) != 0
        or as_decimal(row["k5_idjoined_parsimony_violation_count"]) != 0
    ]
    reconciliation = read_csv_rows(TABLES_ROOT / "S5_C2_06_T5_bandwidth_reconciliation.csv")[1]
    relation_pass = not bound_failures and len(bounds_summary) == len(CONDITIONS)
    relation_evidence = (
        "k1_range_violations=" + ";".join(f"{row['condition']}:{row['k1_range_violation_count']}/{row['n_basins']}" for row in bounds_summary)
        + ";k5_idjoined_parsimony_violations=" + ";".join(f"{row['condition']}:{row['k5_idjoined_parsimony_violation_count']}/{row['n_basins']}" for row in bounds_summary)
        + ";old_bandwidth_reconciliation=" + ";".join(f"{row['condition']}:mismatch={row['nonidentical_rows_at_stored_precision']};max_abs_diff={row['max_absolute_difference_vs_old_rounded_value']}" for row in reconciliation)
        + ";old_bandwidth_status=SUPERSEDED_FOR_ADJUDICATION;source_file_unmodified"
    )
    add_audit(audits, "C2-06", "per_basin_range_and_parsimony_bounds", relation_pass,
              relation_evidence, (), [relation_evidence] if not relation_pass else [])

    c2_07_pairs = read_csv_rows(TABLES_ROOT / "S5_C2_07_gset_source_pairing.csv")[1]
    gset_errors = [row for row in c2_07_pairs if row.get("source_pairing_status") != "PASS_SAME_CASE_SEED_K_N"]
    prior_formula_mismatches = [row for row in c2_07_pairs if row.get("prior_formula_status") == "MISMATCH_AT_DISPLAY_PRECISION"]
    rule_rows = table_data(TABLE_PATHS["rules_ladder"])[1]
    a1 = next((r for r in rule_rows if r.get("rule_family") == "A" and r.get("k") == "1"), None)
    b1 = next((r for r in rule_rows if r.get("rule_family") == "B" and r.get("k") == "1"), None)
    if not a1 or not b1 or a1.get("kge_median") != b1.get("kge_median"):
        gset_errors.append({"estimand": "Rule_A_B_k1_identity", "source_pairing_status": "FAIL"})
    add_audit(audits, "C2-07", "gset_identity_and_rule_b1_identity", not gset_errors,
              f"verified_source_pairs={len(c2_07_pairs)};verified_pair_failures={len(gset_errors)};prior_display_formula_mismatches={len(prior_formula_mismatches)};rule_A_k1={a1.get('kge_median') if a1 else 'missing'};rule_B_k1={b1.get('kge_median') if b1 else 'missing'};issues=" + ";".join(row.get("estimand", row.get("case", "identity")) for row in gset_errors),
              (), [row.get("estimand", row.get("case", "identity")) for row in gset_errors])

    worst = table_data(TABLE_PATHS["worst26_test"])[1]
    matched = table_data(TABLE_PATHS["type_matched"])[1]
    type_penalty = table_data(TABLE_PATHS["type_penalty"])[1]
    count_errors = []
    if len(worst) != 1:
        count_errors.append(f"worst26_rows={len(worst)}")
    else:
        top_m = as_decimal(worst[0].get("n_top_matched", ""))
        top_u = as_decimal(worst[0].get("n_top_unmatched", ""))
        top_n = as_decimal(worst[0].get("n_top_defined", ""))
        if None in (top_m, top_u, top_n) or top_m + top_u != top_n or top_n != Decimal("26"):
            count_errors.append(f"worst26:{top_m}+{top_u}!={top_n}")
    if len(matched) != 1:
        count_errors.append(f"matched_summary_rows={len(matched)}")
    else:
        n_matched = as_decimal(matched[0].get("n_matched", ""))
        n_unmatched = as_decimal(matched[0].get("n_unmatched", ""))
        n_basins_total = as_decimal(matched[0].get("n_basins_total", ""))
        type_sum = sum((as_decimal(r.get("n", "")) or Decimal(0) for r in type_penalty), Decimal(0))
        if None in (n_matched, n_unmatched, n_basins_total) or type_sum != n_matched or n_matched + n_unmatched != n_basins_total or type_sum + n_unmatched != Decimal("531"):
            count_errors.append(f"type_N_sum={type_sum};matched={n_matched};unmatched={n_unmatched};total={n_basins_total};expected_total=531")
    add_audit(audits, "C2-08", "worst26_and_type_population_counts", not count_errors,
              f"worst26_rows={len(worst)};type_rows={len(type_penalty)};type_N_sum={sum((as_decimal(r.get('n','')) or Decimal(0) for r in type_penalty), Decimal(0))};unmatched={matched[0].get('n_unmatched','') if matched else 'missing'};issues={';'.join(count_errors)}",
              (), count_errors)

    _, _, taxonomy = declaration_tables()
    _, override_rows = read_csv_rows(DECLARATIONS_ROOT / "population_n_overrides.csv")
    override_by_slug = {row.get("claim_slug", ""): row for row in override_rows}
    population_bad = []
    n_groups = defaultdict(list)
    valid_not_applicable = 0
    for row in registry:
        n_value = row.get("population_N", "").strip()
        override = override_by_slug.get(row.get("claim_slug", ""), {})
        if n_value == "not_applicable":
            valid = override.get("population_category") == "not_applicable" and override.get("population_meaning") == "not_applicable"
            taxon = {"category": "not_applicable", "meaning": "声明不适用数值 population N"} if valid else None
            valid_not_applicable += int(valid)
        else:
            taxon = taxonomy.get(n_value)
        if not taxon or taxon.get("category") == "unclassified":
            population_bad.append(row)
        if taxon:
            n_groups[n_value].append(row)
    inventory = []
    for n_value, rows in sorted(n_groups.items(), key=lambda item: (as_decimal(item[0]) is None, as_decimal(item[0]) or Decimal(0), item[0])):
        taxon = taxonomy.get(n_value) or {"category": "not_applicable", "meaning": "声明不适用数值 population N"}
        meanings = sorted({f"{row['section']}:{row['estimand_name']}[{row['source_file']}|{row['source_column']}]" for row in rows})
        inventory.append({
            "population_N": n_value,
            "meaning": f"{taxon['category']}:{taxon['meaning']};" + ";".join(meanings),
            "claim_count": len(rows),
        })
    write_rows(TABLES_ROOT / "population_n_inventory.csv", ["population_N", "meaning", "claim_count"], inventory)
    population_issue_text = ";".join(
        f"{row['claim_id']}={row['population_N'] or 'blank'}[{taxonomy.get(row['population_N'], {}).get('category', 'not_applicable' if row.get('population_N') == 'not_applicable' else 'unresolved')}]"
        for row in population_bad
    )
    unclassified_n = ",".join(sorted((n for n, taxon in taxonomy.items() if taxon.get("category") == "unclassified"), key=lambda value: as_decimal(value) or Decimal(0)))
    taxonomy_numeric_values = sum(as_decimal(n_value) is not None for n_value in taxonomy)
    add_audit(audits, "C2-09", "population_N_by_declared_taxonomy", not population_bad,
              f"registry_claims={len(registry)};taxonomy_N_values={taxonomy_numeric_values};unclassified_N={unclassified_n};valid_not_applicable_claims={valid_not_applicable};claims_with_unresolved_or_unclassified_N={len(population_bad)};issues={population_issue_text}",
              [r["claim_id"] for r in population_bad], [r.get("population_N", "") for r in population_bad])

    _, direction_rows = read_csv_rows(DECLARATIONS_ROOT / "sign_and_direction.csv")
    directions = {row["quantity_class"]: row for row in direction_rows}
    declaration_missing = []
    class_counts = Counter()
    for row in registry:
        quantity_class = quantity_class_for_claim(row)
        class_counts[quantity_class or "unresolved"] += 1
        direction_class = {"cover_count": "cover_count / library_size", "spearman": "spearman / retention / coverage_fraction", "coverage_fraction": "spearman / retention / coverage_fraction", "p_value_or_effect_size": "p_value / effect_size"}.get(quantity_class, quantity_class)
        expected = directions.get(direction_class or "")
        if (
            not expected
            or row.get("sign_convention") != expected.get("subtraction_order")
            or row.get("better_direction") != expected.get("better_direction")
        ):
            declaration_missing.append((row, quantity_class or "unresolved"))
    decl_values = ";".join(
        f"{row['claim_id']}[quantity_class={quantity_class};sign={row['sign_convention']};direction={row['better_direction']}]"
        for row, quantity_class in declaration_missing
    )
    add_audit(audits, "C2-10", "sign_and_direction_by_quantity_class", not declaration_missing,
              f"registry_claims={len(registry)};declared_quantity_classes={len(directions)};resolved_quantity_class_claims={len(registry)-class_counts.get('unresolved', 0)-class_counts.get('unclassified', 0)};claims_missing_or_mismatching={len(declaration_missing)};class_counts={dict(sorted(class_counts.items()))};issues={decl_values}",
              [row["claim_id"] for row, _ in declaration_missing], [row["value"] for row, _ in declaration_missing])

    write_rows(TABLES_ROOT / "consistency_audit.csv", AUDIT_FIELDS, audits)
    return audits


def source_columns(rel: str, wanted: Optional[Sequence[str]] = None) -> str:
    path = table_path(rel)
    if not path.is_file():
        return ""
    fields, _ = read_csv_rows(path)
    return ";".join([f for f in (wanted or fields) if f in fields])


def number_or_blank(value: object) -> str:
    return "" if value is None else str(value)


def export_panels() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    panel_map: List[Dict[str, str]] = []
    missing: List[Dict[str, str]] = []

    def record(figure_id: str, panel_id: str, content: str, sources: Sequence[str], columns: Sequence[str], section: str,
               rows: List[Dict[str, object]], fields: Sequence[str]) -> None:
        source_list = [s for s in sources if s]
        present = [s for s in source_list if (ANALYSIS_ROOT / s).is_file() or (REPO_ROOT / s).is_file()]
        column_parts = []
        for src in source_list:
            if src in TABLE_PATHS.values():
                c = source_columns(src, columns)
                column_parts.append(f"{src}[{c}]")
            elif figure_id == "Fig1" and panel_id == "a":
                c = "basin_id;fold" if "OOB_FOLD_ASSIGNMENT" in src else "gage_id;lat;lon"
                column_parts.append(f"{src}[{c}]")
            else:
                column_parts.append(f"{src}[{';'.join(columns)}]")
        panel_map.append({
            "figure_id": figure_id,
            "panel_id": panel_id,
            "panel_content": content,
            "source_file": ";".join(source_list),
            "source_columns": " | ".join(column_parts),
            "section": section,
        })
        file_path = DATA_ROOT / f"{figure_id}_{panel_id}.csv"
        write_rows(file_path, fields, rows)
        if not rows or len(present) != len(source_list):
            missing.append({
                "figure_id": figure_id,
                "panel_id": panel_id,
                "source_file": ";".join(source_list),
                "reason": "empty_panel_data" if not rows else "missing_source_file",
            })

    # Fig1(a): all basin identifiers and all joins are performed with basin_accessor.
    location_path = REPO_ROOT / "project/flexmopex/archive/20261003_cleanup/legacy_training/results/formal_671_unified_nmul4/lambda_invariance/inputs/gauge_locations_671.csv"
    locations = pd.read_csv(location_path, dtype={"gage_id": str})
    basin_ids = canonical_basin_ids()
    coordinates = align_frame_by_id(locations, "gage_id", ["lat", "lon"], canonical=basin_ids, name="fig1_coordinates")
    fold_path = REPO_ROOT / "project/benchmarkpub/data/oob36_20260930/manifests/OOB_FOLD_ASSIGNMENT.csv"
    fold_frame = pd.read_csv(fold_path, dtype={"basin_id": str})
    fold_values = align_frame_by_id(fold_frame, "basin_id", ["fold"], canonical=basin_ids, name="fig1_folds")[:, 0].astype(int).tolist()
    fig1a = [{"basin_id": basin_ids[i], "latitude": coordinates[i, 0], "longitude": coordinates[i, 1], "fold": fold_values[i]} for i in range(len(basin_ids))]
    src1a = ["project/benchmarkpub/data/oob36_20260930/manifests/OOB_FOLD_ASSIGNMENT.csv", "project/flexmopex/archive/20261003_cleanup/legacy_training/results/formal_671_unified_nmul4/lambda_invariance/inputs/gauge_locations_671.csv"]
    record("Fig1", "a", "basin_id,latitude,longitude,fold", src1a, ["basin_id", "fold", "gage_id", "lat", "lon"], "R1", fig1a, ["basin_id", "latitude", "longitude", "fold"])

    eq_rows = table_data(TABLE_PATHS["eq_basin"])[1]
    fig1b = []
    for condition in CONDITIONS:
        subset = [r for r in eq_rows if r.get("condition") == condition]
        values = align_series_by_id([r["basin_id"] for r in subset], [float(r["n_equivalent"]) for r in subset], canonical=basin_ids, name=f"fig1_equivalence_{condition}")
        fig1b.extend({"basin_id": basin_ids[i], "condition": condition, "equivalence_set_size": values[i]} for i in range(len(basin_ids)))
    record("Fig1", "b", "equivalence_set_size_by_condition", [TABLE_PATHS["eq_basin"]], ["basin_id", "condition", "n_equivalent"], "R1", fig1b, ["basin_id", "condition", "equivalence_set_size"])

    exact_curve = table_data(TABLE_PATHS["coverage_exact"])[1]
    nbb = table_data(TABLE_PATHS["cover_nbb"])[1]
    fig1c = []
    for r in exact_curve:
        fig1c.append({"condition": r["condition"], "method": "1yr_block_bootstrap_q05", "series_type": "coverage_curve", "library_size": r["n_models"], "target_pct": "", "coverage_fraction": r["frac_covered_exact"]})
        fig1c.append({"condition": r["condition"], "method": "95_percent_reference", "series_type": "reference_line", "library_size": r["n_models"], "target_pct": "95%", "coverage_fraction": "0.95"})
    for r in nbb:
        fig1c.append({"condition": r["condition"], "method": "3yr_nonoverlap_block_bootstrap_q05", "series_type": "target_point", "library_size": r["n_models_3yr_block_bootstrap"], "target_pct": r["target_pct"], "coverage_fraction": r["target"]})
    record("Fig1", "c", "exact_coverage_fraction,three_year_cover_targets,95_percent_reference", [TABLE_PATHS["coverage_exact"], TABLE_PATHS["cover_nbb"]], ["condition", "n_models", "frac_covered_exact", "target_pct", "n_models_3yr_block_bootstrap", "target"], "R1", fig1c, ["condition", "method", "series_type", "library_size", "target_pct", "coverage_fraction"])

    model_delta = table_data(TABLE_PATHS["model_delta"])[1]
    fig1d = []
    for r in model_delta:
        delta = as_decimal(r.get("delta_pub_median", ""))
        fig1d.append({"model": r["model"], "delta_pub_median": r["delta_pub_median"], "frac_positive": r["frac_positive"], "sign": "positive" if delta is not None and delta > 0 else "negative" if delta is not None and delta < 0 else "zero"})
    record("Fig1", "d", "delta_pub_median,frac_positive,sign", [TABLE_PATHS["model_delta"]], ["model", "delta_pub_median", "frac_positive"], "R1", fig1d, ["model", "delta_pub_median", "frac_positive", "sign"])

    cover_curve = table_data(TABLE_PATHS["coverage_exact"])[1]
    opt_summary = table_data(TABLE_PATHS["optimal_summary"])[1]
    fig2a = []
    for r in cover_curve:
        fig2a.append({"record_type": "coverage_curve", "condition": r["condition"], "target_pct": "", "library_size": r["n_models"], "estimand": "frac_covered_exact", "value": r["frac_covered_exact"], "set_role": ""})
    for r in opt_summary:
        for role, col in (("core", "core_set_size"), ("union", "union_set_size")):
            fig2a.append({"record_type": "optimal_set_size", "condition": r["condition"], "target_pct": r["target_pct"], "library_size": r["k_opt"], "estimand": f"{role}_set_size", "value": r[col], "set_role": role})
    record("Fig2", "a", "exact_fraction_covered,core_set_size,union_set_size", [TABLE_PATHS["coverage_exact"], TABLE_PATHS["optimal_summary"]], ["condition", "n_models", "frac_covered_exact", "target_pct", "k_opt", "core_set_size", "union_set_size"], "R2", fig2a, ["record_type", "condition", "target_pct", "library_size", "estimand", "value", "set_role"])

    freq_rows = table_data(TABLE_PATHS["optimal_frequency"])[1]
    record("Fig2", "b", "optimal_solution_model_appearance_frequency", [TABLE_PATHS["optimal_frequency"]], ["condition", "target_pct", "model", "appearance_count", "total_solutions", "appearance_frequency", "in_core_set", "in_union_set"], "R2", freq_rows, table_data(TABLE_PATHS["optimal_frequency"])[0])

    ladder = table_data(TABLE_PATHS["coverage_ladder"])[1]
    regrets = table_data(TABLE_PATHS["regret_curve"])[1]
    cov_map = {(r["strategy"], r["k"]): r for r in ladder}
    reg_map = {(r["strategy"], r["k"]): r for r in regrets}
    fig2c = []
    for strategy, k in sorted(set(cov_map) | set(reg_map), key=lambda x: (x[0], int(x[1]))):
        c = cov_map.get((strategy, k), {})
        g = reg_map.get((strategy, k), {})
        fig2c.append({"strategy": strategy, "k": k, "coverage_top1": c.get("coverage_top1", ""), "regret_median": g.get("regret_median", "")})
    record("Fig2", "c", "coverage_top1,regret_median", [TABLE_PATHS["coverage_ladder"], TABLE_PATHS["regret_curve"]], ["strategy", "k", "coverage_top1", "regret_median"], "R2", fig2c, ["strategy", "k", "coverage_top1", "regret_median"])

    rules = table_data(TABLE_PATHS["rules_ladder"])[1]
    abs_fields = ["rule_label", "rule_family", "k", "seed_scope", "kge_median", "kge_q25", "kge_q75", "kge_mean"]
    record("Fig3", "a", "rule_ladder_absolute_kge", [TABLE_PATHS["rules_ladder"]], abs_fields, "R3", [{k: r.get(k, "") for k in abs_fields} for r in rules], abs_fields)

    paired_fields = ["rule_label", "rule_family", "k", "vs_B2_mean_diff", "vs_B2_median_diff", "vs_B2_ci95_lo_basin", "vs_B2_ci95_hi_basin", "vs_B2_wilcoxon_p"]
    record("Fig3", "b", "paired_difference_vs_B2,basin_bootstrap_interval", [TABLE_PATHS["rules_ladder"]], paired_fields, "R3", [{k: r.get(k, "") for k in paired_fields} for r in rules], paired_fields)

    fig3c = []
    for r in table_data(TABLE_PATHS["rule_c_b"])[1]:
        fig3c.append({"criterion": "attribute_conditioned", "condition": "A2d", "k": r.get("k", ""), "replicate": r.get("seed", ""), "estimand": "mean_diff", "value": r.get("mean_diff", ""), "ci_low": r.get("ci95_lo_basin_boot", ""), "ci_high": r.get("ci95_hi_basin_boot", ""), "source_file": TABLE_PATHS["rule_c_b"]})
    for r in table_data(TABLE_PATHS["rule_c_b_k3"])[1]:
        fig3c.append({"criterion": "attribute_conditioned", "condition": r.get("comparison", ""), "k": r.get("k", ""), "replicate": "all_seeds_summary", "estimand": "mean_diff_mean", "value": r.get("mean_diff_mean", ""), "ci_low": r.get("ci_lo_basin_mean", ""), "ci_high": r.get("ci_hi_basin_mean", ""), "source_file": TABLE_PATHS["rule_c_b_k3"]})
    for r in table_data(TABLE_PATHS["diversity"])[1]:
        if r.get("k") == "3":
            fig3c.append({"criterion": "diversity_conditioned", "condition": r.get("diversity_type", r.get("strategy", "")), "k": r.get("k", ""), "replicate": "summary", "estimand": "diff_mean", "value": r.get("diff_mean", ""), "ci_low": r.get("ci95_lo", ""), "ci_high": r.get("ci95_hi", ""), "source_file": TABLE_PATHS["diversity"]})
    for r in table_data(TABLE_PATHS["parsimony"])[1]:
        fig3c.append({"criterion": "parsimony_conditioned", "condition": r.get("condition", ""), "k": "", "replicate": "summary", "estimand": "mean_diff_kge", "value": r.get("mean_diff_kge", ""), "ci_low": r.get("ci95_lo_basin_boot", ""), "ci_high": r.get("ci95_hi_basin_boot", ""), "source_file": TABLE_PATHS["parsimony"]})
    record("Fig3", "c", "paired_difference_attribute_diversity_parsimony", [TABLE_PATHS["rule_c_b"], TABLE_PATHS["rule_c_b_k3"], TABLE_PATHS["diversity"], TABLE_PATHS["parsimony"]], ["k", "seed", "mean_diff", "ci95_lo_basin_boot", "ci95_hi_basin_boot", "comparison", "mean_diff_mean", "ci_lo_basin_mean", "ci_hi_basin_mean", "diversity_type", "strategy", "diff_mean", "condition", "mean_diff_kge", "ci95_lo_basin", "ci95_hi_basin"], "R3", fig3c, ["criterion", "condition", "k", "replicate", "estimand", "value", "ci_low", "ci_high", "source_file"])

    type_penalty = table_data(TABLE_PATHS["type_penalty"])[1]
    fig4a = []
    for r in type_penalty:
        for stat in ("min", "q25", "median", "q75", "max", "mean"):
            fig4a.append({"type": r["type"], "n": r.get("n", ""), "statistic": stat, "value": r.get(stat, "")})
    record("Fig4", "a", "penalty_delta_distribution_by_type", [TABLE_PATHS["type_penalty"]], ["type", "n", "min", "q25", "median", "q75", "max", "mean"], "R4", fig4a, ["type", "n", "statistic", "value"])

    attr_rows = table_data(TABLE_PATHS["type_attribute"])[1]
    record("Fig4", "b", "selected_attribute_spearman_by_type", [TABLE_PATHS["type_attribute"]], ["type", "attribute", "spearman_rho", "n", "attribute_global_bh_fdr_q"], "R4", attr_rows, table_data(TABLE_PATHS["type_attribute"])[0])

    worst_rows = table_data(TABLE_PATHS["worst26"])[1]
    record("Fig4", "c", "worst26_type_observed_expected_counts", [TABLE_PATHS["worst26"]], ["type", "n_matched", "n_in_top26", "expected_in_top26", "obs_minus_expected"], "R4", worst_rows, table_data(TABLE_PATHS["worst26"])[0])

    null_rows = [r for r in table_data(TABLE_PATHS["null95"])[1] if r.get("target_pct") == "95%" and r.get("tier") == "main_95"]
    record("Fig4", "d", "size_matched_null_quantiles_and_observed_cover_95", [TABLE_PATHS["null95"]], ["condition", "type", "target_pct", "observed_cover", "null_p5", "null_q25", "null_median", "null_q75", "null_p95", "one_sided_empirical_percentile"], "R4", null_rows, table_data(TABLE_PATHS["null95"])[0])

    table1 = []
    for r in table_data(TABLE_PATHS["cover_primary"])[1]:
        table1.append({"condition": r["condition"], "target_pct": f"{int(round(float(r['target'])*100))}%", "method": "1yr_block_bootstrap_q05", "estimand": "n_models_exact", "value": r["n_models_exact"], "interval_low": "", "interval_high": "", "source_file": TABLE_PATHS["cover_primary"]})
    for r in table_data(TABLE_PATHS["cover_nbb"])[1]:
        table1.append({"condition": r["condition"], "target_pct": r["target_pct"], "method": "3yr_nonoverlap_block_bootstrap_q05", "estimand": "n_models_3yr_block_bootstrap", "value": r["n_models_3yr_block_bootstrap"], "interval_low": "", "interval_high": "", "source_file": TABLE_PATHS["cover_nbb"]})
    for r in table_data(TABLE_PATHS["cover_ci"])[1]:
        table1.append({"condition": r["condition"], "target_pct": r["target_pct"], "method": "basin_bootstrap", "estimand": "median_cover_with_ci95", "value": r["median"], "interval_low": r["ci95_lo"], "interval_high": r["ci95_hi"], "source_file": TABLE_PATHS["cover_ci"]})
    record("Table1", "table", "minimal_cover_90_95_100_two_bootstrap_schemes_basin_bootstrap_intervals", [TABLE_PATHS["cover_primary"], TABLE_PATHS["cover_nbb"], TABLE_PATHS["cover_ci"]], ["condition", "target", "target_pct", "n_models_exact", "n_models_3yr_block_bootstrap", "median", "ci95_lo", "ci95_hi"], "R1", table1, ["condition", "target_pct", "method", "estimand", "value", "interval_low", "interval_high", "source_file"])

    table2 = []
    paired_numeric = ["kge_median", "vs_A_mean_diff", "vs_A_median_diff", "vs_B2_mean_diff", "vs_B2_median_diff", "vs_A_ci95_lo_basin", "vs_A_ci95_hi_basin", "vs_B2_ci95_lo_basin", "vs_B2_ci95_hi_basin"]
    for r in rules:
        for metric in paired_numeric:
            if r.get(metric, "") != "":
                table2.append({"rule_label": r.get("rule_label", ""), "rule_family": r.get("rule_family", ""), "k": r.get("k", ""), "estimand": metric, "value": r.get(metric, ""), "source_file": TABLE_PATHS["rules_ladder"]})
    record("Table2", "table", "absolute_kge_and_paired_differences_vs_A_and_B2", [TABLE_PATHS["rules_ladder"]], ["rule_label", "rule_family", "k", *paired_numeric], "R3", table2, ["rule_label", "rule_family", "k", "estimand", "value", "source_file"])

    write_rows(FIGURES_ROOT / "figure_panel_map.csv", PANEL_MAP_FIELDS, panel_map)
    write_rows(TABLES_ROOT / "missing_panels.csv", ["figure_id", "panel_id", "source_file", "reason"], missing)
    return panel_map, missing


def build_open_decisions(registry: List[Dict[str, str]], audits: List[Dict[str, str]], prompt_gaps: List[Dict[str, str]]) -> List[Dict[str, str]]:
    decisions: List[Dict[str, str]] = []
    _, _, population_taxonomy = declaration_tables()
    _, override_rows = read_csv_rows(DECLARATIONS_ROOT / "population_n_overrides.csv")
    population_overrides = {row.get("claim_slug", ""): row for row in override_rows}
    for row in registry:
        if row.get("status") == "LOADED":
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "loaded_source", "related_value": f"{row['estimand_name']}={row['value']}"})
        if row.get("source_row_status") not in {"matched", "aggregate_expression", "unique_source_tuple"}:
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "unresolved_source_row_key", "related_value": f"{row['source_file']}|{row['source_row_key']}"})
        if row.get("sign_convention", "").strip() in {"", "unresolved"}:
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "missing_sign_convention", "related_value": row["estimand_name"]})
        if row.get("better_direction", "").strip() in {"", "unresolved"}:
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "missing_better_direction", "related_value": row["estimand_name"]})
        if row.get("population_N", "").strip() in {"", "unresolved"}:
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "missing_population_N", "related_value": row["estimand_name"]})
        n_value = row.get("population_N", "").strip()
        override = population_overrides.get(row.get("claim_slug", ""), {})
        taxon = population_taxonomy.get(n_value, {})
        if n_value and n_value not in {"unresolved", "not_applicable"} and (not taxon or taxon.get("category") == "unclassified"):
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "unclassified_population_N", "related_value": f"{n_value}:{taxon.get('meaning', '')}"})
        elif n_value == "not_applicable" and not (override.get("population_category") == "not_applicable" and override.get("population_meaning") == "not_applicable"):
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "unclassified_population_N", "related_value": f"{n_value}:missing_or_invalid_override"})
        if row.get("primary_or_secondary", "") in {"", "unresolved"}:
            decisions.append({"section": row["section"], "claim_id": row["claim_id"], "item_type": "missing_primary_or_secondary", "related_value": row["estimand_name"]})
    for group, rows in sorted(multi_spec_groups(registry).items()):
        primary = next((r for r in rows if r.get("primary_or_secondary") == "primary"), None)
        selected_id = primary["claim_id"] if primary else ""
        spec_values = ";".join(f"{r['estimand_name']}={r['value']}[{r['primary_or_secondary']}]" for r in rows)
        decisions.append({"section": rows[0]["section"], "claim_id": selected_id, "item_type": "multi_spec_selection", "related_value": f"{group}:{spec_values}"})
    for gap in prompt_gaps:
        decisions.append({"section": gap["section"], "claim_id": "", "item_type": "unmapped_requested_estimand", "related_value": gap["gap_id"]})
    for audit in audits:
        if audit["status"] != "FAIL":
            continue
        ids = audit.get("failed_claim_ids", "").split(";") if audit.get("failed_claim_ids") else [""]
        vals = audit.get("failed_values", "")
        for claim_id in ids:
            section = claim_id.split(":", 1)[0] if claim_id.startswith("R") and ":" in claim_id else ""
            decisions.append({"section": section, "claim_id": claim_id, "item_type": "failed_check", "related_value": f"{audit['check_id']}:{audit['evidence']}:{vals}"})
    decisions.sort(key=lambda r: (r["section"], r["item_type"], r["claim_id"], r["related_value"]))
    write_rows(TABLES_ROOT / "open_decisions.csv", OPEN_FIELDS, decisions)
    return decisions

def run_evidence_packet_builder() -> None:
    script = REGISTRY_ROOT / "scripts" / "build_evidence_packet.py"
    spec = importlib.util.spec_from_file_location("_results_registry_evidence_builder", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evidence builder: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def write_t3_proposed_mapping(all_entries: List[Dict[str, str]]) -> int:
    e2_path = TABLES_ROOT / "evidence" / "E2_unmapped_and_unresolved_sources.csv"
    _, e2_rows = read_csv_rows(e2_path)
    gaps = [
        ("strict_winner_retention", "composite", "needs_contrast_definition", "逐 basin 的 argmax_model 可提供候选输入；未指定需比较的 condition 对或 retention 规则，故不计算 retention。"),
        ("top3_top5_exact_set_retention", "composite", "confirm_exact_set_definition", "单元表列出 seen/OOB 的 top-3 与 top-5 模型集合及 overlap；需确认 retention 指标与汇总层级。"),
        ("residual_variance_share", "single", "direct_source_candidate", "variance decomposition 表含 residual component；候选源行直接取 share 列，不改写或重算 share。"),
        ("equivalence_membership_overlap", "composite", "needs_condition_pair_definition", "逐 basin 等价模型成员表可供跨 condition 比较；未指定比较条件与 overlap 定义，故仅列来源行。"),
        ("model_fold_same_sign_180", "composite", "needs_delta_and_sign_definition", "模型×fold 表含多个 delta 汇总列；列出原始值供裁定，不擅自选择 delta 列或判断符号。"),
        ("optimal_solution_core_union", "composite", "direct_source_candidate", "optimal-covers summary 逐 condition/target 同列 core 与 union 成员；保留所有源单元以供审批。"),
        ("oob_100_cover_member_list", "composite", "direct_source_candidate", "OOB full-cover members 表直接列出成员；只提出源行，不覆盖现有登记册。"),
        ("singleton_basin_counts", "composite", "direct_source_candidate", "ID-joined singleton summary 按 condition 列出总数；候选映射保留三条源行。"),
        ("conditionalized_upper_bound", "composite", "no_exact_estimand_match", "E2 候选没有可确认的 conditionalized upper bound；仅列 minimal-cover LP bound 作为近邻候选，不视为同一 estimand。"),
        ("single_model_attribute_negative_result", "single", "scope_review_required", "E2 的 minimum BH q 汇总是最接近的单值候选；需确认其 basin-attribute screening 范围符合 single-model attribute 论断。"),
        ("penalty_effect_size_and_pairwise_adjusted_values", "composite", "direct_source_candidate", "Kruskal-Wallis 表含 effect size，Dunn-Holm 表含 pairwise rank differences 与 adjusted p；将其作为两个来源组件，不合成新统计量。"),
        ("size_matched_null_90_and_100", "composite", "source_rows_not_manifested", "按 target 90%/100% 过滤既有 H1 summary 源行；不重跑 null draws，且标明未在 citable/excluded manifest 单独登记。"),
        ("displacement_oob_loss_three_aggregations", "composite", "direct_source_candidate", "以 E2 pooled 与 model-median manifest 值在原表精度唯一定位同一行，再列 pooled、within-model median、model-median 三列；不重算相关。"),
    ]
    output: List[Dict[str, str]] = []

    def emit(gap_id, shape, proposal_status, record_type, candidate_rank, candidate_name,
             source_file, source_column, source_row_key, value, source_row_status,
             status, freeze_phase, basis):
        output.append({
            "gap_id": gap_id, "proposal_shape": shape, "proposal_status": proposal_status,
            "record_type": record_type, "candidate_rank": candidate_rank,
            "candidate_estimand_name": candidate_name, "source_file": source_file,
            "source_column": source_column, "source_row_key": source_row_key,
            "value": value, "source_row_status": source_row_status,
            "status": status, "freeze_phase": freeze_phase, "basis": basis,
        })

    def emit_e2_candidates(gap_id, shape, rationale):
        candidates = sorted(
            (row for row in e2_rows if row.get("requested_gap_id") == gap_id),
            key=lambda row: int(row.get("candidate_rank") or 0),
        )
        for candidate in candidates:
            entry = {
                "file_path": candidate.get("candidate_source_file", ""),
                "estimand": candidate.get("candidate_estimand_name", ""),
                "column_or_key": candidate.get("candidate_source_column", ""),
            }
            source_row, _ = find_source_row(entry)
            row_status = source_row_status(entry, source_row)
            row_key = candidate.get("candidate_source_row_key", "").strip() or source_row_key(entry, source_row, row_status)
            if candidate.get("candidate_source_row_key", "").strip():
                row_status = "source_matrix_cell_candidate"
            rank = candidate.get("candidate_rank", "")
            candidate_status = "E2_alternative_only"
            if gap_id == "single_model_attribute_negative_result" and rank == "1":
                candidate_status = "proposed_single_if_scope_matches"
            basis = (
                f"E2 rank={rank}; {candidate.get('mapping_failure_reason', '')}; "
                f"selection={candidate.get('candidate_selection_method', '')}; {rationale}"
            )
            emit(
                gap_id, shape, candidate_status, "E2_candidate", rank,
                candidate.get("candidate_estimand_name", ""), candidate.get("candidate_source_file", ""),
                candidate.get("candidate_source_column", ""), row_key, candidate.get("candidate_value", ""),
                row_status, candidate.get("candidate_status", "NOT_LISTED"),
                candidate.get("candidate_freeze_phase", "NOT_LISTED"), basis,
            )

    manifest_cell_index = defaultdict(list)
    for entry in all_entries:
        if entry.get("status") == "SUPERSEDED":
            continue
        found, _ = find_source_row(entry)
        if found is None:
            continue
        cell_key = (entry.get("file_path", ""), tuple(sorted(found.items())), entry.get("column_or_key", "").strip())
        manifest_cell_index[cell_key].append(entry)

    def manifest_cell_metadata(source_file, source_row, source_column):
        cell_key = (source_file, tuple(sorted(source_row.items())), source_column)
        matching = manifest_cell_index.get(cell_key, [])
        statuses = sorted({entry.get("status", "") for entry in matching if entry.get("status", "")})
        phases = sorted({entry.get("freeze_phase", "") for entry in matching if entry.get("freeze_phase", "")})
        return (";".join(statuses) if statuses else "NOT_LISTED", ";".join(phases) if phases else "NOT_LISTED")

    def row_key(source_row, fields):
        return "|".join(f"{field}={source_row[field]}" for field in fields if source_row.get(field, "") != "")

    def emit_unit_rows(gap_id, shape, proposal_status, rationale, source_file,
                       source_rows, key_fields, source_columns, predicate=None):
        selected = [source for source in source_rows if predicate is None or predicate(source)]
        selected = sorted(selected, key=lambda source: tuple(source.get(field, "") for field in key_fields))
        for source in selected:
            key = row_key(source, key_fields)
            for column in source_columns:
                if column not in source:
                    raise ValueError(f"T3 source column missing: {source_file}:{column}")
                status, phase = manifest_cell_metadata(source_file, source, column)
                emit(gap_id, shape, proposal_status, "unit_table_component", "", "",
                     source_file, column, key, source.get(column, ""), "unit_table_row",
                     status, phase, rationale)

    direct_plans = {
        "strict_winner_retention": (
            "sec31_transfer_and_library/tables/sec31_equivalence_sets_per_basin.csv",
            ["condition", "basin_id"], ["argmax_model"], None,
        ),
        "top3_top5_exact_set_retention": (
            "sec33_usage_and_conditioning/tables/sec33_cluster_seen_oob_stability.csv",
            ["cluster"], ["top3_seen_models", "top3_oob_models", "top3_overlap_count", "top3_jaccard", "top5_seen_models", "top5_oob_models", "top5_overlap_count", "top5_jaccard"],
            lambda row: row.get("cluster") != "cluster_mean",
        ),
        "residual_variance_share": (
            TABLE_PATHS["variance"], ["component"], ["frac_of_total_ss"],
            lambda row: row.get("component", "").strip().lower() == "residual",
        ),
        "equivalence_membership_overlap": (
            TABLE_PATHS["eq_basin"], ["condition", "basin_id"], ["n_equivalent", "equivalent_models"], None,
        ),
        "model_fold_same_sign_180": (
            TABLE_PATHS["model_fold_delta"], ["model", "fold"], ["delta_median", "delta_mean", "delta_q25", "delta_q75"], None,
        ),
        "optimal_solution_core_union": (
            TABLE_PATHS["optimal_summary"], ["condition", "target_pct"], ["core_set_models", "union_set_models"], None,
        ),
        "oob_100_cover_member_list": (
            TABLE_PATHS["oob_members"], ["condition", "target_coverage", "model"], ["model"], None,
        ),
        "singleton_basin_counts": (
            TABLE_PATHS["singleton"], ["condition"], ["total_singletons_idjoin"], None,
        ),
        "conditionalized_upper_bound": (
            "sec32_library_members/tables/sec32_minimal_cover_solutions.csv", ["k"], ["lp_relaxation_bound", "lp_ceiling"], None,
        ),
    }

    for gap_id, shape, proposal_status, rationale in gaps:
        emit_e2_candidates(gap_id, shape, rationale)
        if gap_id in direct_plans:
            source_file, keys, columns, predicate = direct_plans[gap_id]
            emit_unit_rows(gap_id, shape, proposal_status, rationale, source_file,
                           table_data(source_file)[1], keys, columns, predicate)
        elif gap_id == "penalty_effect_size_and_pairwise_adjusted_values":
            source_file = TABLE_PATHS["type_kw"]
            emit_unit_rows(gap_id, shape, proposal_status, rationale, source_file,
                           table_data(source_file)[1], ["test"], ["effect_size_value"])
            source_file = TABLE_PATHS["type_pairwise"]
            emit_unit_rows(gap_id, shape, proposal_status, rationale, source_file,
                           table_data(source_file)[1], ["type_a", "type_b"], ["rank_diff", "p_holm"])
        elif gap_id == "size_matched_null_90_and_100":
            source_file = TABLE_PATHS["null95"]
            null_rows = table_data(source_file)[1]
            emit_unit_rows(
                gap_id, shape, proposal_status, rationale, source_file, null_rows,
                ["tier", "target_pct", "condition", "type"],
                ["observed_cover", "null_median", "one_sided_empirical_percentile"],
                lambda row: row.get("target_pct") in {"90%", "100%"},
            )
        elif gap_id == "displacement_oob_loss_three_aggregations":
            source_file = TABLE_PATHS["displacement_oob"]
            displacement_rows = table_data(source_file)[1]
            candidates = [row for row in e2_rows if row.get("requested_gap_id") == gap_id and row.get("candidate_source_file") == source_file]
            matched_sets = []
            for candidate in candidates:
                column = candidate.get("candidate_source_column", "")
                raw_value = candidate.get("candidate_value", "").strip().lstrip("+")
                target = as_decimal(raw_value)
                if target is None or column not in (displacement_rows[0] if displacement_rows else {}):
                    continue
                decimals = len(raw_value.split(".", 1)[1]) if "." in raw_value else 0
                quantum = Decimal(1).scaleb(-decimals)
                matched_sets.append({
                    row.get("instability_metric", ""): row for row in displacement_rows
                    if as_decimal(row.get(column, "")) is not None
                    and as_decimal(row.get(column, "")).quantize(quantum) == target
                })
            common_metrics = set.intersection(*(set(matches) for matches in matched_sets)) if matched_sets else set()
            if len(common_metrics) == 1:
                metric = next(iter(common_metrics))
                source = next(row for row in displacement_rows if row.get("instability_metric") == metric)
                key = row_key(source, ["instability_metric"])
                for column in ("pooled_spearman_rho", "within_model_spearman_median", "model_median_spearman_rho"):
                    status, phase = manifest_cell_metadata(source_file, source, column)
                    if column in {candidate.get("candidate_source_column") for candidate in candidates}:
                        candidate = next(item for item in candidates if item.get("candidate_source_column") == column)
                        status, phase = candidate.get("candidate_status", status), candidate.get("candidate_freeze_phase", phase)
                    emit(gap_id, shape, proposal_status, "unit_table_component", "", "",
                         source_file, column, key, source.get(column, ""), "manifest_precision_row_match",
                         status, phase, rationale)
            else:
                emit(gap_id, shape, "no_unique_source_row", "unit_table_candidate", "", "",
                     source_file, "pooled_spearman_rho/model_median_spearman_rho", "", "", "unresolved",
                     "NOT_LISTED", "NOT_LISTED", f"Manifest-display-precision candidates do not identify one common row; {rationale}")

    inter_slug = "inter_metric_spearman_fold_instability_3caed92d"
    inter_rows = sorted(
        (row for row in e2_rows if row.get("claim_slug") == inter_slug),
        key=lambda row: int(row.get("candidate_rank") or 0),
    )
    for candidate in inter_rows:
        emit(
            inter_slug, "single", "symmetric_cell_alternatives_require_approval", "E2_unresolved_key_alternative",
            candidate.get("candidate_rank", ""), candidate.get("candidate_estimand_name", ""),
            candidate.get("candidate_source_file", ""), candidate.get("candidate_source_column", ""),
            candidate.get("candidate_source_row_key", ""), candidate.get("candidate_value", ""),
            "source_matrix_cell_candidate", candidate.get("candidate_status", "NOT_LISTED"),
            candidate.get("candidate_freeze_phase", "NOT_LISTED"),
            "E2 在对称 correlation matrix 中按 manifest 显示精度匹配到该单元格；转置单元格亦匹配同值。仅列备选来源，不更新登记册 source_row_key。",
        )

    output.sort(key=lambda row: (
        next((index for index, spec in enumerate(gaps) if spec[0] == row["gap_id"]), len(gaps)),
        1 if row["record_type"].startswith("E2") else 2,
        row.get("candidate_rank", ""), row.get("source_file", ""), row.get("source_row_key", ""), row.get("source_column", ""),
    ))
    write_rows(TABLES_ROOT / "T3_proposed_mapping_for_approval.csv", T3_FIELDS, output)
    return len(output)


def write_s5_adjudication_outputs() -> Dict[str, object]:
    penalty_rel = "sec34_determinacy_and_bounds/tables/sec34_a3_parsimony_penalty_idjoin_per_basin.csv"
    bandwidth_path = TABLES_ROOT / "S5_C2_06_T5_bandwidth_per_basin.csv"
    bounds_rows = read_csv_rows(TABLES_ROOT / "S5_C2_06_T5_bounds_per_basin.csv")[1]
    bounds_summary = read_csv_rows(TABLES_ROOT / "S5_C2_06_T5_bounds_summary.csv")[1]
    reconciliation = read_csv_rows(TABLES_ROOT / "S5_C2_06_T5_bandwidth_reconciliation.csv")[1]
    detail_rows = []
    for row in bounds_rows:
        detail_rows.append({
            "condition": row["condition"], "basin_id": row["basin_id"],
            "penalty_source_file": relpath(table_path(penalty_rel)),
            "penalty_source_column": "diff_kge_rank_minus_pars",
            "penalty_value": row["parsimony_penalty_idjoined"],
            "bandwidth_source_file": relpath(bandwidth_path),
            "bandwidth_source_column": "bandwidth_1yr_block_bootstrap_primary",
            "bandwidth_value": row["bandwidth_1yr_block_bootstrap_primary_recomputed"],
            "comparison": "penalty > bandwidth + 1e-12",
            "violation": row["parsimony_exceeds_bandwidth"],
            "penalty_join_method": "basin_id_keyed via _shared.basin_accessor.align_frame_by_id",
            "bandwidth_file_alignment_status": "CORRECTED_RECOMPUTED_BY_ID",
            "adjudicative_status": "CHECKED",
        })
    violations = [row for row in detail_rows if row["violation"] == "true"]
    k5_summary = []
    for summary in bounds_summary:
        k5_summary.append({
            "condition": summary["condition"], "basin_count": summary["n_basins"],
            "mechanical_violation_count": summary["k5_idjoined_parsimony_violation_count"],
            "penalty_source_file": relpath(table_path(penalty_rel)),
            "bandwidth_source_file": relpath(bandwidth_path),
            "bandwidth_file_alignment_status": "CORRECTED_RECOMPUTED_BY_ID",
            "adjudicative_status": "CHECKED",
            "join_method": "basin_id_keyed via _shared.basin_accessor.align_frame_by_id",
        })
    write_rows(TABLES_ROOT / "S5_C2_06_K5_idjoin_per_basin.csv", [
        "condition", "basin_id", "penalty_source_file", "penalty_source_column", "penalty_value",
        "bandwidth_source_file", "bandwidth_source_column", "bandwidth_value", "comparison", "violation",
        "penalty_join_method", "bandwidth_file_alignment_status", "adjudicative_status",
    ], detail_rows)
    write_rows(TABLES_ROOT / "S5_C2_06_K5_idjoin_violations.csv", [
        "condition", "basin_id", "penalty_source_file", "penalty_source_column", "penalty_value",
        "bandwidth_source_file", "bandwidth_source_column", "bandwidth_value", "comparison",
        "bandwidth_file_alignment_status", "adjudicative_status",
    ], violations)
    write_rows(TABLES_ROOT / "S5_C2_06_K5_idjoin_summary.csv", [
        "condition", "basin_count", "mechanical_violation_count", "penalty_source_file", "bandwidth_source_file",
        "bandwidth_file_alignment_status", "adjudicative_status", "join_method",
    ], k5_summary)

    pairing_rows = read_csv_rows(TABLES_ROOT / "S5_C2_07_gset_source_pairing.csv")[1]
    prior_formula_failures = [row for row in pairing_rows if row.get("prior_formula_status") == "MISMATCH_AT_DISPLAY_PRECISION"]
    verified_pairs_ok = bool(pairing_rows) and all(row.get("source_pairing_status") == "PASS_SAME_CASE_SEED_K_N" for row in pairing_rows)
    c2_06_has_violations = any(
        (as_decimal(row.get("k1_range_violation_count", "")) or Decimal(0)) > 0
        or (as_decimal(row.get("k5_idjoined_parsimony_violation_count", "")) or Decimal(0)) > 0
        for row in bounds_summary
    )
    c2_06_classification = "真实不一致" if c2_06_has_violations else "未见不一致"
    if verified_pairs_ok:
        c2_07_classification = "旧汇总公式显示精度差异" if prior_formula_failures else "未见不一致"
    else:
        c2_07_classification = "比对口径错误／证据缺失"
    issue_rows = [
        {
            "check_id": "C2-06", "classification": c2_06_classification,
            "evidence_file": "S5_C2_06_T5_bandwidth_reconciliation.csv; S5_C2_06_T5_bounds_summary.csv; S5_C2_06_K5_idjoin_summary.csv",
            "basis": "Primary 1-year bandwidth recomputed from water_year_sufficient_stats with the frozen 1000-draw seed; K1 and K5 aligned by basin ID. The legacy bandwidth table remains unchanged and is marked SUPERSEDED_FOR_ADJUDICATION; its reconciliation is diagnostic only.",
        },
        {
            "check_id": "C2-07", "classification": c2_07_classification,
            "evidence_file": "S5_C2_07_gset_source_pairing.csv; S5_C2_07_gset_formula_failures.csv; E4_C2_07_identity_evidence.csv",
            "basis": f"Verified paired-regret and G_set reference rows use the same case/seed/k/N and match within stored precision; prior summary-formula display mismatches={len(prior_formula_failures)} were caused by applying the formula to rounded aggregate cells, not a full-precision same-seed source comparison." if verified_pairs_ok else "Verified same-case/seed/k/N source pairing is incomplete or does not match within the source precision.",
        },
    ]
    write_rows(TABLES_ROOT / "S5_issue_classification.csv", ["check_id", "classification", "evidence_file", "basis"], issue_rows)
    return {
        "k5_summary": k5_summary, "k5_violation_rows": len(violations),
        "c2_07_failures": len(prior_formula_failures), "c2_07_classification": c2_07_classification,
        "c2_06_classification": c2_06_classification, "bandwidth_reconciliation": reconciliation,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_roots() -> List[Path]:
    return [
        REPO_ROOT / "project" / "benchmarkpub",
        REPO_ROOT / "project" / "benchmark" / "analysis",
        REPO_ROOT / "project" / "benchmark" / "manuscript",
        REPO_ROOT / "project" / "flexmopex" / "archive" / "20261003_cleanup" / "legacy_training" / "results" / "formal_671_unified_nmul4" / "lambda_invariance" / "inputs",
        REPO_ROOT / "data",
    ]


def is_generated_output(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ANALYSIS_ROOT.resolve())
    except ValueError:
        return False
    return relative.parts and relative.parts[0] in {"_results_registry", "_figures"}


def snapshot_inputs() -> Dict[str, Tuple[int, int, str]]:
    snapshot: Dict[str, Tuple[int, int, str]] = {}
    for root in snapshot_roots():
        if not root.exists():
            continue
        for base, dirs, files in os.walk(root, followlinks=False):
            base_path = Path(base)
            dirs[:] = [d for d in dirs if not (base_path / d).is_symlink()]
            for filename in files:
                path = base_path / filename
                if is_generated_output(path) or path.is_symlink():
                    continue
                st = path.stat()
                digest = sha256(path) if path.suffix.lower() in {".csv", ".tsv"} and "tables" in path.parts else ""
                snapshot[relpath(path)] = (st.st_mtime_ns, st.st_size, digest)
    return snapshot


def write_source_integrity(before: Dict[str, Tuple[int, int, str]], after: Dict[str, Tuple[int, int, str]]) -> List[Dict[str, str]]:
    rows = []
    for path in sorted(set(before) | set(after)):
        b = before.get(path)
        a = after.get(path)
        rows.append({
            "source_file": path,
            "before_mtime_ns": b[0] if b else "",
            "after_mtime_ns": a[0] if a else "",
            "before_size_bytes": b[1] if b else "",
            "after_size_bytes": a[1] if a else "",
            "before_table_sha256": b[2] if b else "",
            "after_table_sha256": a[2] if a else "",
            "status": "UNCHANGED" if b == a else "CHANGED_OR_MISSING",
        })
    write_rows(TABLES_ROOT / "upstream_file_integrity.csv",
               ["source_file", "before_mtime_ns", "after_mtime_ns", "before_size_bytes", "after_size_bytes", "before_table_sha256", "after_table_sha256", "status"], rows)
    return rows


def write_artifact_manifest() -> List[Dict[str, str]]:
    output_files = []
    evidence_script = REGISTRY_ROOT / "scripts" / "build_evidence_packet.py"
    for root in (TABLES_ROOT, FIGURES_ROOT, DECLARATIONS_ROOT):
        if not root.exists():
            continue
        for path in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in {".csv", ".json"}):
            if path in {TABLES_ROOT / "artifact_manifest.csv", TABLES_ROOT / "T3_proposed_mapping_for_approval.csv"}:
                continue
            rows = [json.loads(path.read_text())] if path.suffix == ".json" else read_csv_rows(path)[1]
            generator = evidence_script if path.parent == TABLES_ROOT / "evidence" else Path(__file__)
            if path.name.startswith("M1_") or path.name == "venue_assignment_rules.csv":
                generator = REGISTRY_ROOT / "scripts" / "close_registry_m1_m2.py"
            if path.name.startswith(("T1_", "T2_")) or path.name in {"claim_quantity_class_map.csv", "sign_and_direction.csv"}:
                generator = REGISTRY_ROOT / "scripts" / "finalize_registry_t1_t2.py"
            if path.name.startswith("T3_isolated_"):
                generator = REGISTRY_ROOT / "scripts" / "run_isolated_reproduction.py"
            output_files.append({
                "file_path": relpath(path),
                "generating_script": relpath(generator),
                "row_count": len(rows),
                "byte_count": path.stat().st_size,
                "sha256": sha256(path),
            })
    write_rows(TABLES_ROOT / "artifact_manifest.csv", ["file_path", "generating_script", "row_count", "byte_count", "sha256"], output_files)
    return output_files


def main() -> None:
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    declaration_counts = write_prompt_declarations()
    before = snapshot_inputs()
    registry, all_entries, prompt_gaps = build_registry()
    declaration_counts["population_taxonomy"] = write_population_taxonomy(registry)
    t5_results = recompute_t5_bandwidth()
    c2_07_pairs = recompute_c2_07_source_pairing()
    s5 = write_s5_adjudication_outputs()
    audits = source_manifest_checks(registry, all_entries, prompt_gaps)
    panel_map, missing_panels = export_panels()
    decisions = build_open_decisions(registry, audits, prompt_gaps)
    after = snapshot_inputs()
    integrity = write_source_integrity(before, after)
    run_evidence_packet_builder()
    declaration_counts["population_taxonomy"] = write_population_taxonomy(registry)
    audits = source_manifest_checks(registry, all_entries, prompt_gaps)
    decisions = build_open_decisions(registry, audits, prompt_gaps)
    run_evidence_packet_builder()
    t3_rows = write_t3_proposed_mapping(all_entries)
    artifact_manifest = write_artifact_manifest()
    counts = Counter(r["section"] for r in registry)
    check_failures = [r for r in audits if r["status"] != "PASS"]
    changed_inputs = [r for r in integrity if r["status"] != "UNCHANGED"]
    decision_counts = Counter(r["item_type"] for r in decisions)
    print(f"declaration_rows={declaration_counts}")
    print(f"registry_rows={len(registry)};section_rows={dict(sorted(counts.items()))};columns={len(REGISTRY_FIELDS)}")
    print(f"audit_checks={len(audits)};failed_checks={len(check_failures)};open_decisions={len(decisions)};decision_types={dict(sorted(decision_counts.items()))}")
    print(f"figure_panel_rows={len(panel_map)};panel_data_files={len(panel_map)};missing_panels={len(missing_panels)}")
    print(f"upstream_files={len(integrity)};upstream_changed={len(changed_inputs)};generated_csvs={len(artifact_manifest)}")
    print(f"t3_proposed_mapping_rows={t3_rows};excluded_from_artifact_manifest=true")
    print(f"k5_mechanical_violations={[(r['condition'], r['mechanical_violation_count'], r['adjudicative_status']) for r in s5['k5_summary']]};c2_07={s5['c2_07_classification']};prior_display_formula_mismatches={s5['c2_07_failures']}")
    if check_failures:
        print("failed_check_ids=" + ";".join(r["check_id"] for r in check_failures))
    if missing_panels:
        print("missing_panel_ids=" + ";".join(f"{r['figure_id']}_{r['panel_id']}" for r in missing_panels))
    if changed_inputs:
        print("changed_source_files=" + ";".join(r["source_file"] for r in changed_inputs[:20]))


if __name__ == "__main__":
    from close_registry_m1_m2 import run
    run(sys.modules[__name__])
