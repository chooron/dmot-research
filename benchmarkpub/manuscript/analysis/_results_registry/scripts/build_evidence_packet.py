#!/usr/bin/env python3
"""Create read-only E1–E7 evidence tables from the existing registry, manifests and source tables.

This producer does not run the registry/audit builder and does not recompute analysis estimates.
It writes only its own CSVs beneath _results_registry/tables/evidence/.
"""
from __future__ import annotations

import ast
import csv
import difflib
import re
import sys
from collections import defaultdict
from functools import lru_cache
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ANALYSIS_ROOT.parents[3]
REGISTRY_ROOT = ANALYSIS_ROOT / "_results_registry"
TABLES_ROOT = REGISTRY_ROOT / "tables"
OUT_ROOT = TABLES_ROOT / "evidence"
BUILDER_PATH = REGISTRY_ROOT / "scripts" / "build_results_artifacts.py"
MANIFEST_ROOT = ANALYSIS_ROOT / "_manifest" / "tables"
REGISTRY_PATH = TABLES_ROOT / "results_claim_registry.csv"
AUDIT_PATH = TABLES_ROOT / "consistency_audit.csv"
MULTI_PATH = TABLES_ROOT / "multi_estimand_audit.csv"
N_INVENTORY_PATH = TABLES_ROOT / "population_n_inventory.csv"
MASTER_PATH = MANIFEST_ROOT / "manuscript_master_manifest.csv"
CITABLE_PATH = MANIFEST_ROOT / "manuscript_citable_numbers.csv"
EXCLUDED_PATH = MANIFEST_ROOT / "manuscript_excluded_numbers.csv"

POPULATION_KEYS = (
    "population_N", "population_n", "n_basins_total", "n_basins", "n_basins_covered",
    "n_units", "need_basins", "n_type_basins", "n_basins_in_type", "n_total",
    "n_cells", "n_cells_total",
    "n_total_defined", "n_matched_total", "n_top_defined", "n_type", "n_used", "N",
    "n_models_total", "n_models", "n_rows", "n",
)
ROLE_KEYS = ("primary_or_secondary", "manuscript_role", "role")
SIGN_KEYS = ("sign_convention", "subtraction_order", "difference_definition", "delta_definition", "direction_definition")
DIRECTION_KEYS = ("better_direction", "better_direction_definition", "direction_definition")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=256)
def read_relative_table(relative_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if ";" in relative_path:
        fields, rows = [], []
        for component_path in dict.fromkeys(relative_path.split(";")):
            component_fields, component_rows = read_relative_table(component_path)
            fields.extend(field for field in component_fields if field not in fields)
            rows.extend(component_rows)
        return fields, rows
    path = (ANALYSIS_ROOT / relative_path).resolve()
    allowed = (ANALYSIS_ROOT.resolve(),
               (ANALYSIS_ROOT.parents[3] / "project/benchmark/analysis").resolve(),
               (ANALYSIS_ROOT.parents[3] / "project/benchmarkpub/analysis").resolve(),
               (ANALYSIS_ROOT.parents[3] / "project/benchmark/manuscript").resolve())
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError(f"Source path outside approved result directories: {path}")
    if path.suffix == ".json":
        import json
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return list(data), [{key: str(value) for key, value in data.items()}]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in fields})


def literal_assignment(source_path: Path, variable: str):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == variable for t in targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"literal assignment not found: {variable}")


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def to_decimal(value: object) -> Optional[Decimal]:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def table_manifest_rows(master: List[Dict[str, str]], citable: List[Dict[str, str]], excluded: List[Dict[str, str]]):
    result: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for manifest_name, rows, name_key in (
        (MASTER_PATH.name, master, "quantity_name"),
        (CITABLE_PATH.name, citable, "estimand"),
        (EXCLUDED_PATH.name, excluded, "estimand"),
    ):
        for row in rows:
            name = row.get(name_key, "").strip()
            if not name:
                continue
            key = (name, row.get("value", ""), row.get("file_path", ""), row.get("column_or_key", ""))
            item = result.setdefault(key, {
                "name": name, "value": row.get("value", ""), "source_file": row.get("file_path", ""),
                "source_column": row.get("column_or_key", ""), "statuses": set(), "phases": set(),
                "manifest_sources": set(), "derivation_types": set(),
            })
            item["manifest_sources"].add(manifest_name)
            if row.get("status", "").strip():
                item["statuses"].add(row["status"].strip())
            if row.get("freeze_phase", "").strip():
                item["phases"].add(row["freeze_phase"].strip())
            if row.get("derivation_type", "").strip():
                item["derivation_types"].add(row["derivation_type"].strip())
    return list(result.values())


def manifest_status_for_path(relative_path: str, master, citable, excluded) -> str:
    target_path = relative_path
    try:
        candidate_path = (REPO_ROOT / relative_path).resolve()
        target_path = candidate_path.relative_to(ANALYSIS_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        pass
    found = []
    for manifest_name, rows, key_name in (
        (MASTER_PATH.name, master, "quantity_name"),
        (CITABLE_PATH.name, citable, "estimand"),
        (EXCLUDED_PATH.name, excluded, "estimand"),
    ):
        matched = [r for r in rows if r.get("file_path", "") == target_path]
        if matched:
            statuses = sorted({r.get("status", "").strip() for r in matched if r.get("status", "").strip()})
            estimands = sorted({r.get(key_name, "").strip() for r in matched if r.get(key_name, "").strip()})
            found.append(f"{manifest_name}:status={('|'.join(statuses) if statuses else 'not_declared')};names={'|'.join(estimands)}")
    return ";".join(found) if found else "not_listed_in_three_manifests"


def build_e1(multi, registry):
    by_name: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in registry:
        by_name[row.get("estimand_name", "")].append(row)
    output = []
    for item in sorted(multi, key=lambda r: (r.get("family_id", ""), r.get("estimand_name", ""))):
        refs = by_name.get(item.get("estimand_name", ""), [])
        role = item.get("primary_or_secondary", "").strip()
        token = re.search(r"(?:^|[_-])(primary|secondary)(?:[_-]|$)", item.get("estimand_name", ""), re.I)
        if role and token:
            role_origin = "estimand_name_literal_token"
            role_origin_value = token.group(1)
        elif role:
            role_origin = "source_row_role_field_or_existing_registry_value;source_origin_not_resolved_by_multi_table"
            role_origin_value = role
        else:
            role_origin = "not_declared_in_available_source_row_or_estimand_name"
            role_origin_value = ""
        output.append({
            "family_id": item.get("family_id", ""),
            "estimand_name": item.get("estimand_name", ""),
            "value": item.get("value", ""),
            "source_file": item.get("source_file", ""),
            "source_column": ";".join(sorted({r.get("source_column", "") for r in refs if r.get("source_column", "")})),
            "status": ";".join(sorted({r.get("status", "") for r in refs if r.get("status", "")})),
            "freeze_phase": ";".join(sorted({r.get("freeze_phase", "") for r in refs if r.get("freeze_phase", "")})),
            "claim_id_references": ";".join(sorted({r.get("claim_id", "") for r in refs if r.get("claim_id", "")})),
            "primary_or_secondary_existing": role,
            "role_source_field": role_origin,
            "role_source_value": role_origin_value,
            "manuscript_selected_existing": item.get("manuscript_selected", ""),
            "source_row_key": ";".join(sorted({r.get("source_row_key", "") for r in refs if r.get("source_row_key", "")})),
            "registry_reference_count": len(refs),
        })
    return output


def candidate_rows(manifest_candidates, gap_id: str):
    scored = []
    target = norm(gap_id)
    for item in manifest_candidates:
        score = difflib.SequenceMatcher(None, target, norm(str(item["name"]))).ratio()
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], str(x[1]["name"]), str(x[1]["source_file"]), str(x[1]["source_column"])))
    return scored[:3]


def rounded_manifest_match(source_value: str, manifest_value: str) -> bool:
    source_number = to_decimal(source_value)
    target_number = to_decimal(manifest_value)
    if source_number is None or target_number is None:
        return False
    decimals = len(manifest_value.split(".", 1)[1]) if "." in manifest_value else 0
    quantum = Decimal(1).scaleb(-decimals)
    return source_number.quantize(quantum, rounding=ROUND_HALF_UP) == target_number


def build_e2(audit_by_id, registry, builder_config, all_manifests, master, citable, excluded):
    required_names = builder_config["REQUIRED_NAME_CHECKS"]
    patterns = {(section, gap): pattern for section, gap, pattern in required_names}
    citable_and_excluded_names = [r.get("estimand", "") for r in citable + excluded if r.get("estimand", "")]
    evidence = audit_by_id.get("C2-01", {}).get("evidence", "")
    gap_text = evidence.partition("gaps=")[2]
    gaps = []
    if gap_text:
        for item in gap_text.split(";"):
            if ":" in item:
                section, gap = item.split(":", 1)
                gaps.append((section, gap))
    output = []
    for section, gap in gaps:
        pattern = patterns.get((section, gap), "")
        matched = [name for name in citable_and_excluded_names if pattern and re.search(pattern, name, re.I)]
        reason = "名称不存在" if not matched else "名称多义" if len(matched) > 1 else "名称存在但未映射"
        ranked = candidate_rows(all_manifests, gap)
        if not ranked:
            ranked = [(Decimal(0), {})]
        for rank, (score, item) in enumerate(ranked, 1):
            output.append({
                "evidence_item": f"{section}:{gap}", "claim_id": "", "section": section,
                "claim_slug": "", "estimand_name": "", "requested_gap_id": gap,
                "required_name_pattern": pattern, "mapping_failure_reason": reason,
                "candidate_rank": rank, "candidate_match_score": format(score, ".8f"),
                "candidate_selection_method": "SequenceMatcher over normalized requested_gap_id and candidate estimand_name",
                "candidate_estimand_name": item.get("name", ""), "candidate_value": item.get("value", ""),
                "candidate_source_file": item.get("source_file", ""),
                "candidate_source_column": item.get("source_column", ""),
                "candidate_status": "|".join(sorted(item.get("statuses", set()))),
                "candidate_freeze_phase": "|".join(sorted(item.get("phases", set()))),
                "candidate_manifest_sources": "|".join(sorted(item.get("manifest_sources", set()))),
            })
    unresolved = [r for r in registry if r.get("source_row_status") == "unresolved"]
    for claim in unresolved:
        fields, rows = read_relative_table(claim.get("source_file", ""))
        direct_column = claim.get("source_column", "")
        reason = "列名不符" if direct_column not in fields else "来源行缺失"
        matrix_fields = [field for field in fields if field != "instability_metric"]
        cells = []
        for source_row in rows:
            row_key_parts = [f"{key}={source_row.get(key, '')}" for key in ("instability_metric",) if source_row.get(key, "")]
            for column in matrix_fields:
                raw = source_row.get(column, "")
                if rounded_manifest_match(raw, claim.get("value", "")):
                    cells.append(("|".join(row_key_parts), column, raw))
        cells.sort(key=lambda x: (x[0], x[1]))
        for rank, (row_key, column, raw) in enumerate(cells[:5], 1):
            output.append({
                "evidence_item": claim.get("claim_id", ""), "claim_id": claim.get("claim_id", ""),
                "section": claim.get("section", ""), "claim_slug": claim.get("claim_slug", ""),
                "estimand_name": claim.get("estimand_name", ""), "requested_gap_id": "",
                "required_name_pattern": "", "mapping_failure_reason": reason,
                "candidate_rank": rank, "candidate_match_score": "",
                "candidate_selection_method": "source matrix cells matching manifest value at manifest decimal precision",
                "candidate_estimand_name": claim.get("estimand_name", ""),
                "candidate_value": raw, "candidate_source_file": claim.get("source_file", ""),
                "candidate_source_column": column, "candidate_status": claim.get("status", ""),
                "candidate_freeze_phase": claim.get("freeze_phase", ""),
                "candidate_manifest_sources": "source_matrix_cell_candidate",
                "source_table_fields": "|".join(fields), "candidate_source_row_key": row_key,
                "manifest_source_column": direct_column, "manifest_value": claim.get("value", ""),
            })
    return output


def equivalence_sets_source_path(analysis_root: Path) -> str:
    paths_source = (analysis_root / "_shared" / "paths.py").read_text(encoding="utf-8", errors="replace")
    line = next((line for line in paths_source.splitlines() if re.match(r"\s*DIR_EQUIV_T1T2\s*=", line)), "")
    pieces = re.findall(r"[\"']([^\"']+)[\"']", line)
    if not pieces:
        return ""
    source = analysis_root.parents[1].joinpath(*pieces, "tables", "t1_equivalence_sets.csv")
    return source.relative_to(REPO_ROOT).as_posix()


def build_e3(analysis_root, table_paths, audit_by_id, master, citable, excluded):
    k1_rel = table_paths["k1_constraint"]
    k5_rel = table_paths["k5_constraint"]
    k1_rows = read_relative_table(k1_rel)[1]
    k5_rows = read_relative_table(k5_rel)[1]
    method_script = analysis_root / "sec34_determinacy_and_bounds" / "scripts" / "sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py"
    method_text = method_script.read_text(encoding="utf-8")
    bandwidth_match = re.search(r'df_bw_exist\s*=\s*pd\.read_csv\(ROOT\s*/\s*"([^"]+)"', method_text)
    dispersion_match = re.search(r'df_disp_exist\s*=\s*pd\.read_csv\(TABLES\s*/\s*"([^"]+)"', method_text)
    penalty_match = re.search(r'df_pen_exist\s*=\s*pd\.read_csv\(TABLES\s*/\s*"([^"]+)"', method_text)
    bandwidth_rel = bandwidth_match.group(1) if bandwidth_match else ""
    section_tables = method_script.parent.parent / "tables"
    dispersion_rel = section_tables.joinpath(dispersion_match.group(1)).relative_to(REPO_ROOT).as_posix() if dispersion_match else ""
    penalty_rel = section_tables.joinpath(penalty_match.group(1)).relative_to(REPO_ROOT).as_posix() if penalty_match else ""
    scheme_match = re.search(r'delta_b\s*=\s*sub\["([^"]+)"\]', method_text)
    scheme_column = scheme_match.group(1) if scheme_match else ""
    scheme = scheme_column.removeprefix("bandwidth_") if scheme_column else ""
    detail_rel = (analysis_root / "sec34_determinacy_and_bounds" / "tables" / "sec34_k1_constraint_violations_detail.csv").relative_to(analysis_root).as_posix()
    detail_rows = read_relative_table(detail_rel)[1]
    q05_source = equivalence_sets_source_path(analysis_root)

    provenance_path = analysis_root / "sec34_determinacy_and_bounds" / "scripts" / "sec34_05_provenance.py"
    provenance_lines = provenance_path.read_text(encoding="utf-8", errors="replace").splitlines()
    provenance_evidence = " | ".join(line.strip().strip(',') for line in provenance_lines if any(key in line for key in ("Root Cause Identified", "Single-Source Recomputation", "SUPERSEDES sec34_06 / sec34_09")))

    dispersion_idjoin = "sec34_determinacy_and_bounds/tables/sec34_a3_dispersion_idjoin_per_basin.csv"
    penalty_idjoin = "sec34_determinacy_and_bounds/tables/sec34_a3_parsimony_penalty_idjoin_per_basin.csv"
    old_disp_summary = "sec34_determinacy_and_bounds/tables/sec34_equivalence_set_dispersion_summary.csv"
    dispersion_idjoin_summary = "sec34_determinacy_and_bounds/tables/sec34_a3_dispersion_idjoin_summary.csv"
    penalty_idjoin_summary = "sec34_determinacy_and_bounds/tables/sec34_a3_parsimony_penalty_idjoin_summary.csv"
    old_pen_summary = "sec34_determinacy_and_bounds/tables/sec34_parsimony_penalty_summary.csv"
    old_disp_status = manifest_status_for_path(old_disp_summary, master, citable, excluded)
    old_pen_status = manifest_status_for_path(old_pen_summary, master, citable, excluded)
    k1_status = manifest_status_for_path(k1_rel, master, citable, excluded)
    k5_status = manifest_status_for_path(k5_rel, master, citable, excluded)
    detail_status = manifest_status_for_path(detail_rel, master, citable, excluded)
    dispersion_status = manifest_status_for_path(dispersion_rel, master, citable, excluded)
    penalty_status = manifest_status_for_path(penalty_rel, master, citable, excluded)
    dispersion_idjoin_status = manifest_status_for_path(dispersion_idjoin, master, citable, excluded)
    penalty_idjoin_status = manifest_status_for_path(penalty_idjoin, master, citable, excluded)
    dispersion_idjoin_summary_status = manifest_status_for_path(dispersion_idjoin_summary, master, citable, excluded)
    penalty_idjoin_summary_status = manifest_status_for_path(penalty_idjoin_summary, master, citable, excluded)
    dispersion_idjoin_rows = read_relative_table(dispersion_idjoin)[1]
    penalty_idjoin_rows = read_relative_table(penalty_idjoin)[1]
    dispersion_join_method = ";".join(sorted({r.get("join_method", "") for r in dispersion_idjoin_rows if r.get("join_method", "")}))
    penalty_join_method = ";".join(sorted({r.get("join_method", "") for r in penalty_idjoin_rows if r.get("join_method", "")}))

    output = []
    k1_by_condition = {r.get("condition", ""): r for r in k1_rows}
    for condition, summary in sorted(k1_by_condition.items()):
        detail_for_condition = [r for r in detail_rows if r.get("condition", "") == condition]
        for row in detail_for_condition:
            output.append({
                "record_type": "K1_per_basin_violation_detail",
                "per_basin_violation_values_available_in_C2_06_source": "yes",
                "comparison": "point_kge_range_recorded <= bandwidth_delta_b",
                "key_matching_path": "C2-06 merged dispersion and bandwidth on basin_id+condition; filtered by condition; q05 from t1_equivalence_sets filtered by condition and indexed by basin_id; detail row keyed by basin_id+condition",
                "condition": condition, "basin_id": row.get("basin_id", ""),
                "left_quantity": "point_kge_range_recorded", "left_source_file": detail_rel,
                "left_source_column": "point_kge_range_recorded", "left_value": row.get("point_kge_range_recorded", ""),
                "right_quantity": "bandwidth_delta_b", "right_source_file": detail_rel,
                "right_source_column": "bandwidth_delta_b", "right_value": row.get("bandwidth_delta_b", ""),
                "q05_threshold_source_file": detail_rel,
                "q05_threshold_source_column": "q05_threshold", "q05_threshold_value": row.get("q05_threshold", ""),
                "violation_excess_source_column": "excess_range_minus_delta", "violation_excess_value": row.get("excess_range_minus_delta", ""),
                "violation_excess_source_file": detail_rel,
                "left_underlying_source_file": dispersion_rel, "left_underlying_source_column": "point_kge_range",
                "right_underlying_source_file": bandwidth_rel, "right_underlying_source_column": scheme_column,
                "q05_underlying_source_file": q05_source, "q05_underlying_source_column": "q05",
                "summary_source_file": k1_rel, "summary_count_column": "n_violations",
                "summary_count_value": summary.get("n_violations", ""), "summary_denominator_column": "n_basins_total",
                "summary_denominator_value": summary.get("n_basins_total", ""), "scheme": scheme,
                "check_input_class": "existing_sec34_10_inputs_not_ID_joined",
                "detail_table_manifest_status": detail_status,
                "dispersion_input_manifest_status": dispersion_status,
                "check_summary_manifest_status": k1_status,
                "related_superseded_summary_status": old_disp_status,
                "idjoin_alternative_file_not_used_by_C2_06": dispersion_idjoin,
                "idjoin_alternative_manifest_status": dispersion_idjoin_status,
                "idjoin_alternative_join_method": dispersion_join_method,
                "idjoin_alternative_summary_file_not_used_by_C2_06": dispersion_idjoin_summary,
                "idjoin_alternative_summary_manifest_status": dispersion_idjoin_summary_status,
                "provenance_evidence": "",
            })
        output.append({
            "record_type": "K1_summary_cell", "comparison": "count of existing-table rows with point_kge_range > bandwidth_delta_b",
            "key_matching_path": f"condition={condition} matched directly to the existing C2-06 K1 summary row",
            "condition": condition, "left_quantity": "n_violations", "left_source_file": k1_rel,
            "left_source_column": "n_violations", "left_value": summary.get("n_violations", ""),
            "right_quantity": "n_basins_total", "right_source_file": k1_rel,
            "right_source_column": "n_basins_total", "right_value": summary.get("n_basins_total", ""),
            "summary_source_file": k1_rel, "summary_count_column": "n_violations",
            "summary_count_value": summary.get("n_violations", ""), "summary_denominator_column": "n_basins_total",
            "summary_denominator_value": summary.get("n_basins_total", ""), "scheme": scheme,
            "summary_fraction_column": "violation_fraction", "summary_fraction_value": summary.get("violation_fraction", ""),
            "check_input_class": "existing_sec34_10_inputs_not_ID_joined",
            "check_summary_manifest_status": k1_status, "related_superseded_summary_status": old_disp_status,
            "idjoin_alternative_file_not_used_by_C2_06": dispersion_idjoin,
            "idjoin_alternative_manifest_status": dispersion_idjoin_status,
            "idjoin_alternative_join_method": dispersion_join_method,
            "idjoin_alternative_summary_file_not_used_by_C2_06": dispersion_idjoin_summary,
            "idjoin_alternative_summary_manifest_status": dispersion_idjoin_summary_status,
            "provenance_evidence": provenance_evidence,
        })

    for summary in sorted(k5_rows, key=lambda r: r.get("condition", "")):
        condition = summary.get("condition", "")
        output.append({
            "record_type": "K5_summary_only_per_basin_cells_not_persisted",
            "comparison": "diff_kge_rank_minus_pars <= bandwidth_delta_b",
            "key_matching_path": f"condition={condition} matched directly to the existing C2-06 K5 summary row; no per-basin violation rows are persisted",
            "condition": condition, "basin_id": "",
            "left_quantity": "diff_kge_rank_minus_pars", "left_source_file": penalty_rel,
            "left_source_column": "diff_kge_rank_minus_pars", "left_value": "",
            "right_quantity": "bandwidth_delta_b", "right_source_file": bandwidth_rel,
            "right_source_column": scheme_column, "right_value": "",
            "summary_source_file": k5_rel, "summary_count_column": "n_violations_penalty_gt_delta",
            "summary_count_value": summary.get("n_violations_penalty_gt_delta", ""),
            "summary_denominator_column": "n_basins_total", "summary_denominator_value": summary.get("n_basins_total", ""),
            "summary_fraction_column": "violation_fraction", "summary_fraction_value": summary.get("violation_fraction", ""),
            "summary_ratio_median_column": "ratio_penalty_over_delta_median",
            "summary_ratio_median_value": summary.get("ratio_penalty_over_delta_median", ""),
            "summary_ratio_max_column": "ratio_penalty_over_delta_max",
            "summary_ratio_max_value": summary.get("ratio_penalty_over_delta_max", ""),
            "scheme": scheme, "check_input_class": "existing_sec34_10_inputs_not_ID_joined",
            "per_basin_violation_values_available_in_C2_06_source": "no",
            "check_summary_manifest_status": k5_status,
            "penalty_input_manifest_status": penalty_status,
            "related_superseded_summary_status": old_pen_status,
            "idjoin_alternative_file_not_used_by_C2_06": penalty_idjoin,
            "idjoin_alternative_manifest_status": penalty_idjoin_status,
            "idjoin_alternative_join_method": penalty_join_method,
            "idjoin_alternative_summary_file_not_used_by_C2_06": penalty_idjoin_summary,
            "idjoin_alternative_summary_manifest_status": penalty_idjoin_summary_status,
            "provenance_evidence": provenance_evidence,
        })
    t5_summary_rel = "_results_registry/tables/S5_C2_06_T5_bounds_summary.csv"
    t5_reconciliation_rel = "_results_registry/tables/S5_C2_06_T5_bandwidth_reconciliation.csv"
    t5_summary_rows = read_relative_table(t5_summary_rel)[1]
    t5_reconciliation_rows = read_relative_table(t5_reconciliation_rel)[1]
    penalty_source = "sec34_determinacy_and_bounds/tables/sec34_a3_parsimony_penalty_idjoin_per_basin.csv"
    t5_raw_source = (analysis_root.parents[2] / "benchmark" / "analysis" / "ic_dpl_gap_conditional_20260923" / "forward" / "test" / "water_year_sufficient_stats.csv").relative_to(REPO_ROOT).as_posix()
    for summary in t5_summary_rows:
        output.append({
            "record_type": "T5_recomputed_idjoined_bounds_summary",
            "comparison": "range <= recomputed primary bandwidth; ID-joined parsimony penalty <= recomputed primary bandwidth",
            "key_matching_path": "basin_id keyed through _shared.basin_accessor; primary bandwidth recomputed from water_year_sufficient_stats",
            "condition": summary.get("condition", ""),
            "left_quantity": "K1 range_violation_count", "left_source_file": t5_summary_rel,
            "left_source_column": "k1_range_violation_count", "left_value": summary.get("k1_range_violation_count", ""),
            "left_underlying_source_file": t5_raw_source, "left_underlying_source_column": "IC/seen-dPL sufficient-stat columns; 15 water years",
            "right_quantity": "K5 idjoined_parsimony_violation_count", "right_source_file": t5_summary_rel,
            "right_source_column": "k5_idjoined_parsimony_violation_count", "right_value": summary.get("k5_idjoined_parsimony_violation_count", ""),
            "right_underlying_source_file": penalty_source, "right_underlying_source_column": "diff_kge_rank_minus_pars",
            "summary_source_file": t5_summary_rel, "summary_count_column": "n_basins", "summary_count_value": summary.get("n_basins", ""),
            "summary_denominator_column": "n_basins", "summary_denominator_value": summary.get("n_basins", ""),
            "scheme": "1yr_block_bootstrap_primary;frozen seed and 1000 draws",
            "check_input_class": "T5_recomputed_primary_bandwidth_and_id_joined_bounds",
            "per_basin_violation_values_available_in_C2_06_source": "yes",
            "provenance_evidence": "Legacy bandwidth table is unchanged; reconciliation is diagnostic and does not drive the corrected join.",
        })
    for reconciliation in t5_reconciliation_rows:
        output.append({
            "record_type": "T5_legacy_bandwidth_reconciliation",
            "comparison": "ID-joined recomputed bandwidth versus legacy table at stored decimal precision",
            "key_matching_path": "condition and basin_id joined through shared basin accessor; old input table not modified",
            "condition": reconciliation.get("condition", ""),
            "left_quantity": "nonidentical_rows_at_stored_precision", "left_source_file": t5_reconciliation_rel,
            "left_source_column": "nonidentical_rows_at_stored_precision", "left_value": reconciliation.get("nonidentical_rows_at_stored_precision", ""),
            "right_quantity": "max_absolute_difference_vs_old_rounded_value", "right_source_file": t5_reconciliation_rel,
            "right_source_column": "max_absolute_difference_vs_old_rounded_value", "right_value": reconciliation.get("max_absolute_difference_vs_old_rounded_value", ""),
            "summary_source_file": t5_reconciliation_rel, "summary_count_column": "id_joined_rows",
            "summary_count_value": reconciliation.get("id_joined_rows", ""), "summary_denominator_column": "id_joined_rows",
            "summary_denominator_value": reconciliation.get("id_joined_rows", ""),
            "scheme": reconciliation.get("old_table_treatment", ""),
            "check_input_class": "legacy_input_reconciliation_only_not_adjudicative",
            "provenance_evidence": f"Legacy alignment audit status={reconciliation.get('legacy_alignment_status', '')}; source table left untouched.",
        })
    return output


def build_e4(analysis_root, audit_by_id):
    gset_rel = "sec33_usage_and_conditioning/tables/sec33_gset_summary.csv"
    rule_rel = "sec33_usage_and_conditioning/tables/sec33_rules_ladder_unified.csv"
    gset_rows = read_relative_table(gset_rel)[1]
    rule_rows = read_relative_table(rule_rel)[1]
    check = audit_by_id.get("C2-07", {})
    evidence = check.get("evidence", "")
    issue_text = evidence.partition("issues=")[2]
    issue_pattern = re.compile(r"([^;:]+):stored=([^;]+);calculated=([^;]+);tol=([^;]+)")
    issue_by_label = {m.group(1): {"stored": m.group(2), "calculated": m.group(3), "tolerance": m.group(4)} for m in issue_pattern.finditer(issue_text)}
    output = []
    pairing_rel = "_results_registry/tables/S5_C2_07_gset_source_pairing.csv"
    pairing_rows = read_relative_table(pairing_rel)[1]
    pairing_by_label = {row.get("estimand", ""): row for row in pairing_rows}
    for row in gset_rows:
        label = row.get("estimand", "")
        issue = issue_by_label.get(label, {})
        source_pair = pairing_by_label.get(label, {})
        output.append({
            "identity": "G_set_formula",
            "left_quantity": "G_set", "left_source_file": gset_rel, "left_source_column": "G_set",
            "left_key": f"estimand={label}", "left_value": row.get("G_set", ""),
            "right_quantity": "1 - ((mean_VBS - mean_best_R) / (mean_VBS - mean_best_F))",
            "right_source_file": gset_rel, "right_source_column": "mean_best_R;mean_best_F;mean_VBS",
            "right_key": f"estimand={label}", "right_value_as_recorded_by_audit": issue.get("calculated", source_pair.get("prior_summary_formula_gset", "")),
            "formula_input_mean_best_R": row.get("mean_best_R", ""),
            "formula_input_mean_best_F": row.get("mean_best_F", ""),
            "formula_input_mean_VBS": row.get("mean_VBS", ""),
            "tolerance_as_recorded_by_audit": issue.get("tolerance", source_pair.get("prior_summary_tolerance", "")),
            "key_matching_path": "same row in sec33_gset_summary.csv; prior check applies formula to rounded components; full-precision source pair is recorded separately",
            "audit_issue_label": label if issue else source_pair.get("prior_formula_status", "not_listed_in_C2-07_issue_text"),
            "audit_status": check.get("status", ""),
            "right_value_origin": "S5_C2_07_gset_source_pairing.csv:prior_summary_formula_gset",
        })
    a_row = next((r for r in rule_rows if r.get("rule_family") == "A" and r.get("k") == "1"), None)
    b_row = next((r for r in rule_rows if r.get("rule_family") == "B" and r.get("k") == "1"), None)
    output.append({
        "identity": "Rule_B_k1_equals_Rule_A_k1",
        "left_quantity": "Rule A k=1 kge_median", "left_source_file": rule_rel,
        "left_source_column": "kge_median", "left_key": "rule_family=A;k=1",
        "left_value": a_row.get("kge_median", "") if a_row else "",
        "right_quantity": "Rule B k=1 kge_median", "right_source_file": rule_rel,
        "right_source_column": "kge_median", "right_key": "rule_family=B;k=1",
        "right_value_as_recorded_by_audit": b_row.get("kge_median", "") if b_row else "",
        "tolerance_as_recorded_by_audit": "exact string equality; no numeric tolerance",
        "key_matching_path": "first row satisfying rule_family=A and k=1; first row satisfying rule_family=B and k=1; no other selector",
        "audit_status": check.get("status", ""),
        "right_value_origin": "sec33_rules_ladder_unified.csv",
    })
    for pair in pairing_rows:
        output.append({
            "identity": "G_set_source_row_pairing",
            "left_quantity": "gset_from_regret", "left_source_file": pair.get("verified_left_source_file", ""),
            "left_source_column": "gset_from_regret", "left_key": pair.get("verified_left_row_key", ""),
            "left_value": pair.get("verified_left_value", ""),
            "right_quantity": "G_set_reference", "right_source_file": pair.get("verified_right_source_file", ""),
            "right_source_column": "G_set", "right_key": pair.get("verified_right_row_key", ""),
            "right_value_as_recorded_by_audit": pair.get("verified_right_value", ""),
            "tolerance_as_recorded_by_audit": pair.get("verified_tolerance", ""),
            "key_matching_path": "same case, seed, k, and N; per-basin regret-derived G_set compared to the source G_set reference row",
            "audit_issue_label": pair.get("source_pairing_status", ""),
            "audit_status": "PASS" if pair.get("source_pairing_status") == "PASS_SAME_CASE_SEED_K_N" else "FAIL",
            "right_value_origin": "S5_C2_07_gset_source_pairing.csv:verified_right_value",
        })
    return output


def selected_by_registry_key(fields, rows, registry_row):
    key = registry_row.get("source_row_key", "")
    if not key or key.startswith(("aggregate_expression=", "unique_source_tuple=", "unresolved_estimand=")):
        return []
    selectors = []
    for part in key.split("|"):
        if "=" in part:
            field, value = part.split("=", 1)
            if field in fields:
                selectors.append((field, value))
    selected = list(rows)
    for field, value in selectors:
        selected = [r for r in selected if r.get(field, "").strip() == value.strip()]
    return selected


def literal_population_explanations(registry_row, n_value):
    source = registry_row.get("source_file", "")
    if not source:
        return ""
    fields, rows = read_relative_table(source)
    selected = selected_by_registry_key(fields, rows, registry_row)
    if not selected:
        selected = rows
    explanations = set()
    target = to_decimal(n_value)
    for field in POPULATION_KEYS:
        if field not in fields:
            continue
        for source_row in selected:
            value = source_row.get(field, "").strip()
            numeric = to_decimal(value)
            if target is not None and numeric == target:
                row_key = "|".join(f"{key}={source_row[key]}" for key in ("condition", "type", "k", "estimand", "entity_type") if source_row.get(key, ""))
                explanations.add(f"{field}={value}" + (f"[{row_key}]" if row_key else ""))
    return ";".join(sorted(explanations)) if explanations else "无"


def build_e5(registry, n_inventory, master_by_name):
    output = []
    inventory_values = sorted({r.get("population_N", "").strip() for r in list(n_inventory) + list(registry) if to_decimal(r.get("population_N", "").strip()) is not None}, key=lambda x: (to_decimal(x) or Decimal(0), x))
    for n_value in inventory_values:
        for claim in registry:
            if n_value not in re.findall(r"\d+", claim.get("population_N", "")):
                continue
            output.append({
                "record_type": "resolved_population_N_reference", "population_N_value": n_value,
                "population_N_text": claim.get("population_N", ""), "claim_id": claim.get("claim_id", ""),
                "section": claim.get("section", ""), "estimand_name": claim.get("estimand_name", ""),
                "source_file": claim.get("source_file", ""), "source_column": claim.get("source_column", ""),
                "source_table_literal_explanation": literal_population_explanations(claim, n_value),
                "source_table_row_count": "", "source_population_columns": "", "source_population_values": "",
                "source_row_status": claim.get("source_row_status", ""), "source_row_key": claim.get("source_row_key", ""),
                "manifest_notes": master_by_name.get(claim.get("estimand_name", ""), {}).get("notes", ""),
            })
    for claim in registry:
        if claim.get("population_N", "").strip() and claim.get("population_N", "").strip().lower() != "unresolved":
            continue
        source = claim.get("source_file", "")
        fields, rows = read_relative_table(source) if source else ([], [])
        present = [key for key in POPULATION_KEYS if key in fields]
        declared = []
        for key in present:
            values = sorted({r.get(key, "").strip() for r in rows if r.get(key, "").strip()})
            if values:
                declared.append(f"{key}={'|'.join(values)}")
            else:
                declared.append(f"{key}=<blank>")
        output.append({
            "record_type": "C2-09_unresolved_population_N", "population_N_value": "",
            "population_N_text": "", "claim_id": claim.get("claim_id", ""),
            "section": claim.get("section", ""), "estimand_name": claim.get("estimand_name", ""),
            "source_file": source, "source_column": claim.get("source_column", ""),
            "source_table_literal_explanation": "无", "source_table_row_count": len(rows),
            "source_population_columns": "|".join(present) if present else "无",
            "source_population_values": ";".join(declared) if declared else "无",
            "source_row_status": claim.get("source_row_status", ""), "source_row_key": claim.get("source_row_key", ""),
            "manifest_notes": master_by_name.get(claim.get("estimand_name", ""), {}).get("notes", ""),
        })
    return output


def build_e6(registry):
    r4 = [{key: row.get(key, "") for key in ("claim_id", "claim_slug", "estimand_name", "value", "source_file")} for row in registry if row.get("section") == "R4"]
    r123 = [{key: row.get(key, "") for key in ("section", "claim_id", "claim_slug")} for row in registry if row.get("section") in {"R1", "R2", "R3"}]
    return r4, r123


def manifest_row_for_claim(claim, citable, excluded, master_by_name):
    name = claim.get("estimand_name", "")
    entry = next((r for r in citable + excluded if r.get("estimand", "") == name), None)
    if entry:
        return entry
    return master_by_name.get(name, {})


def source_row_for_claim(claim, source_fields, source_rows):
    selected = selected_by_registry_key(source_fields, source_rows, claim)
    return selected[0] if len(selected) == 1 else None


def missing_reason(field, claim, manifest, source_fields, source_rows):
    source_row = source_row_for_claim(claim, source_fields, source_rows)
    aliases = {
        "sign_convention": SIGN_KEYS,
        "better_direction": DIRECTION_KEYS,
        "primary_or_secondary": ROLE_KEYS,
        "population_N": POPULATION_KEYS,
    }.get(field, ())
    if aliases:
        available = [key for key in aliases if key in source_fields]
        if field == "primary_or_secondary" and not available:
            token = re.search(r"(?:^|[_-])(primary|secondary)(?:[_-]|$)", claim.get("estimand_name", ""), re.I)
            if token:
                return "存在但未被当前行解析"
        if not available:
            note = manifest.get("notes", "")
            note_pattern = re.search(r"\b[Nn]\s*=\s*[0-9][0-9,]*", note)
            if field == "population_N" and note_pattern:
                return "来源行/总体口径未定位"
            return "字段在来源表中不存在"
        if source_row:
            values = {source_row.get(key, "").strip() for key in available if source_row.get(key, "").strip()}
        else:
            values = {row.get(key, "").strip() for row in source_rows for key in available if row.get(key, "").strip()}
        if len(values) > 1:
            return "存在多值"
        if not values:
            return "存在但未填"
        return "来源行缺失或总体未唯一"
    if field == "source_row_key" and claim.get("source_row_status") == "unresolved":
        col = claim.get("source_column", "")
        if col not in source_fields:
            return "字段在来源表中不存在"
        return "来源行缺失"
    source_name_fields = {
        "section": ("results_section",), "claim_id": ("estimand", "quantity_name"),
        "claim_slug": ("estimand", "quantity_name"), "estimand_name": ("estimand", "quantity_name"),
        "value": ("value",), "source_file": ("file_path",), "source_column": ("column_or_key",),
        "status": ("status",), "freeze_phase": ("freeze_phase",),
    }.get(field, ())
    if source_name_fields:
        found = [name for name in source_name_fields if name in manifest]
        if not found:
            return "字段在来源表中不存在"
        if not any(manifest.get(name, "").strip() for name in found):
            return "存在但未填"
    if field == "source_row_key":
        return "来源行缺失"
    return "字段在来源表中不存在"


def build_e7(registry, citable, excluded, master_by_name):
    fields = list(registry[0].keys()) if registry else []
    output = []
    for field in fields:
        unparseable = []
        reason_to_claims: Dict[str, List[str]] = defaultdict(list)
        for claim in registry:
            value = claim.get(field, "").strip()
            if field == "source_row_key":
                parseable = claim.get("source_row_status", "") in {"matched", "aggregate_expression", "unique_source_tuple"}
            elif field == "source_row_keys":
                if str(claim.get("is_composite", "false")).strip().lower() == "true":
                    components = [part.strip() for part in value.split(";") if part.strip()]
                    parseable = bool(components)
                else:
                    parseable = value == claim.get("source_row_key", "") and claim.get("source_row_status", "") in {"matched", "aggregate_expression", "unique_source_tuple"}
            else:
                parseable = bool(value)
            if parseable:
                continue
            unparseable.append(claim.get("claim_id", ""))
            manifest = manifest_row_for_claim(claim, citable, excluded, master_by_name)
            source_fields, source_rows = read_relative_table(claim.get("source_file", "")) if claim.get("source_file") else ([], [])
            reason = missing_reason(field, claim, manifest, source_fields, source_rows)
            reason_to_claims[reason].append(claim.get("claim_id", ""))
        if reason_to_claims:
            reasons = ";".join(f"{reason}={len(ids)}" for reason, ids in sorted(reason_to_claims.items()))
            detail = ";".join(f"{reason}:{'|'.join(sorted(ids))}" for reason, ids in sorted(reason_to_claims.items()))
        else:
            reasons, detail = "无不可解析项", ""
        output.append({
            "registry_column": field, "total_claims": len(registry),
            "parseable_claims": len(registry) - len(unparseable), "unparseable_claims": len(unparseable),
            "unparseable_root_cause_counts": reasons,
            "unparseable_claim_ids_by_root_cause": detail,
        })
    return output


def main() -> None:
    builder_config = {
        "TABLE_PATHS": literal_assignment(BUILDER_PATH, "TABLE_PATHS"),
        "REQUIRED_NAME_CHECKS": literal_assignment(BUILDER_PATH, "REQUIRED_NAME_CHECKS"),
    }
    registry = read_rows(REGISTRY_PATH)
    audit_rows = read_rows(AUDIT_PATH)
    audit_by_id = {r.get("check_id", ""): r for r in audit_rows}
    multi = read_rows(MULTI_PATH)
    n_inventory = read_rows(N_INVENTORY_PATH)
    master = read_rows(MASTER_PATH)
    citable = read_rows(CITABLE_PATH)
    excluded = read_rows(EXCLUDED_PATH)
    master_by_name = {r.get("quantity_name", ""): r for r in master}
    all_manifests = table_manifest_rows(master, citable, excluded)

    e1 = build_e1(multi, registry)
    e2 = build_e2(audit_by_id, registry, builder_config, all_manifests, master, citable, excluded)
    e3 = build_e3(ANALYSIS_ROOT, builder_config["TABLE_PATHS"], audit_by_id, master, citable, excluded)
    e4 = build_e4(ANALYSIS_ROOT, audit_by_id)
    e5 = build_e5(registry, n_inventory, master_by_name)
    e6_r4, e6_r123 = build_e6(registry)
    e7 = build_e7(registry, citable, excluded, master_by_name)

    write_rows(OUT_ROOT / "E1_multi_specification_members.csv", [
        "family_id", "estimand_name", "value", "source_file", "source_column", "status", "freeze_phase",
        "claim_id_references", "primary_or_secondary_existing", "role_source_field", "role_source_value",
        "manuscript_selected_existing", "source_row_key", "registry_reference_count",
    ], e1)
    write_rows(OUT_ROOT / "E2_unmapped_and_unresolved_sources.csv", [
        "evidence_item", "claim_id", "section", "claim_slug", "estimand_name", "requested_gap_id",
        "required_name_pattern", "mapping_failure_reason", "candidate_rank", "candidate_match_score",
        "candidate_selection_method",
        "candidate_estimand_name", "candidate_value", "candidate_source_file", "candidate_source_column",
        "candidate_status", "candidate_freeze_phase", "candidate_manifest_sources", "source_table_fields",
        "candidate_source_row_key", "manifest_source_column", "manifest_value",
    ], e2)
    write_rows(OUT_ROOT / "E3_C2_06_bounds_evidence.csv", [
        "record_type", "comparison", "key_matching_path", "condition", "basin_id", "left_quantity", "left_source_file",
        "left_source_column", "left_value", "left_underlying_source_file", "left_underlying_source_column",
        "right_quantity", "right_source_file", "right_source_column", "right_value", "right_underlying_source_file",
        "right_underlying_source_column", "q05_threshold_source_file", "q05_threshold_source_column", "q05_threshold_value",
        "q05_underlying_source_file", "q05_underlying_source_column", "violation_excess_source_file",
        "violation_excess_source_column", "violation_excess_value", "summary_source_file", "summary_count_column",
        "summary_count_value", "summary_denominator_column", "summary_denominator_value", "summary_fraction_column",
        "summary_fraction_value", "summary_ratio_median_column", "summary_ratio_median_value",
        "summary_ratio_max_column", "summary_ratio_max_value", "scheme", "check_input_class",
        "per_basin_violation_values_available_in_C2_06_source", "detail_table_manifest_status",
        "dispersion_input_manifest_status", "penalty_input_manifest_status", "check_summary_manifest_status",
        "related_superseded_summary_status", "idjoin_alternative_file_not_used_by_C2_06",
        "idjoin_alternative_manifest_status", "idjoin_alternative_join_method",
        "idjoin_alternative_summary_file_not_used_by_C2_06", "idjoin_alternative_summary_manifest_status",
        "provenance_evidence",
    ], e3)
    write_rows(OUT_ROOT / "E4_C2_07_identity_evidence.csv", [
        "identity", "left_quantity", "left_source_file", "left_source_column", "left_key", "left_value",
        "right_quantity", "right_source_file", "right_source_column", "right_key",
        "right_value_as_recorded_by_audit", "formula_input_mean_best_R", "formula_input_mean_best_F",
        "formula_input_mean_VBS", "tolerance_as_recorded_by_audit", "key_matching_path",
        "audit_issue_label", "audit_status", "right_value_origin",
    ], e4)
    write_rows(OUT_ROOT / "E5_population_N_evidence.csv", [
        "record_type", "population_N_value", "population_N_text", "claim_id", "section", "estimand_name",
        "source_file", "source_column", "source_table_literal_explanation", "source_table_row_count",
        "source_population_columns", "source_population_values", "source_row_status", "source_row_key",
        "manifest_notes",
    ], e5)
    write_rows(OUT_ROOT / "E6_R4_claim_inventory.csv", ["claim_id", "claim_slug", "estimand_name", "value", "source_file"], e6_r4)
    write_rows(OUT_ROOT / "E6_R123_claim_ids.csv", ["section", "claim_id", "claim_slug"], e6_r123)
    write_rows(OUT_ROOT / "E7_field_availability.csv", [
        "registry_column", "total_claims", "parseable_claims", "unparseable_claims",
        "unparseable_root_cause_counts", "unparseable_claim_ids_by_root_cause",
    ], e7)

    print(f"e1_rows={len(e1)};e2_rows={len(e2)};e3_rows={len(e3)};e4_rows={len(e4)};e5_rows={len(e5)}")
    print(f"e6_r4_rows={len(e6_r4)};e6_r123_rows={len(e6_r123)};e7_fields={len(e7)};output_files=8")
    print("audit_and_registry_rebuilt=false;source_tables_modified=false")


if __name__ == "__main__":
    main()
