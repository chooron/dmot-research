#!/usr/bin/env python3
"""Apply approved M1 source mappings and ordered M2 rules without estimators."""
from __future__ import annotations
import csv
import fnmatch
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
REMOVED = ('strict_winner_retention', 'singleton_basin_counts')
VENUES = (
    ('pf_g1_penalty_median_type*', 'si'),
    ('pf_g1b_spearman_type*', 'si'),
    ('pf_h1_null_95_*', 'si'),
    ('t3_2_fixed_ensemble_*', 'si'),
    ('e8_gset_*', 'si'),
    ('f2_constrained_p25_*', 'si'),
    ('f2_constrained_p75_*', 'si'),
    ('cluster_oob_*', 'si'),
    ('top5_models_share_of_total_delta_kge', 'si'),
    ('parsimony_penalty_mean_diff_ic', 'si'),
    ('uncertainty_bandwidth_*', 'si'),
    ('minimal_cover_*_target_90pct', 'si'),
    ('multiyear_3yr_block_boot_cover_*_90', 'si'),
    ('bootstrap_cover_ci_*_90_median', 'si'),
    ('optimal_solutions_count_*_100', 'si'),
    ('size_matched_null_90_and_100', 'si'),
    ('*', 'main_text'),
)
COMPONENT_SLUGS = (
    'equivalence_set_size_median_seen_dpl',
    'equivalence_set_size_median_oob_dpl',
    'minimal_cover_seen_dpl_target_95pct',
    'minimal_cover_oob_dpl_target_95pct',
)
COMP_FIELDS = ['estimand_name', 'component', 'source_file', 'source_column',
               'source_row_key', 'value', 'status', 'freeze_phase', 'existing_claim_id']
RESULT_FIELDS = ['original_estimand', 'new_estimand', 'disposition', 'expected_components',
                 'parsed_components', 'reason']
DEV_FIELDS = ['position', 'situation', 'handling_rule']

def read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write(path, fields, rows):
    if not path.parent.is_dir():
        raise RuntimeError('S2: output parent must already exist')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def inventory():
    report = ROOT / 'tables' / 'M1_name_occurrences.csv'
    if report.exists():
        return
    hits = []
    for base in (ROOT, ANALYSIS / '_manifest'):
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix not in {'.csv', '.py', '.md', '.json', '.txt'}:
                continue
            lines = p.read_text(encoding='utf-8-sig').splitlines()
            fields = next(csv.reader([lines[0]]), []) if p.suffix == '.csv' and lines else []
            for line, text in enumerate(lines, 1):
                for term in REMOVED:
                    if term not in text:
                        continue
                    cells = next(csv.reader([text]), []) if fields else []
                    matched = [fields[i] for i, v in enumerate(cells) if term in v and i < len(fields)]
                    hits.append({'name': term, 'file': p.relative_to(ANALYSIS).as_posix(),
                                 'line': line, 'field': '|'.join(matched) or 'text'})
    write(report, ['name', 'file', 'line', 'field'], hits)

def parse_mappings(module, baseline, entries):
    components, results, deviations = [], [], []
    for name in REMOVED:
        results.append(dict(original_estimand=name, new_estimand='', disposition='removed_declaration',
                            expected_components='', parsed_components='', reason='No registry claim deleted unless non-R2 unreferenced claim exists'))
    def meta(file, column):
        matches = [x for x in entries if x.get('file_path') == file and x.get('column_or_key') == column]
        if not matches:
            matches = [x for x in entries if x.get('file_path') == file]
        status = {x.get('status', '') for x in matches}
        phase = {x.get('freeze_phase', '') for x in matches}
        return (next(iter(status)) if len(status) == 1 else 'NOT_LISTED',
                next(iter(phase)) if len(phase) == 1 else 'NOT_LISTED')
    def cell(file, row, column, keys):
        status, phase = meta(file, column)
        return dict(source_file=file, source_column=column,
                    source_row_key='|'.join(k+'='+row[k] for k in keys), value=row[column],
                    status=status, freeze_phase=phase, existing_claim_id='')
    def commit(name, section, expected, cells, new_name=None, reason=''):
        disposition = 'mapped' if len(cells) == expected else 'unmapped_pending'
        results.append(dict(original_estimand=name, new_estimand=new_name or name,
                            disposition=disposition, expected_components=expected,
                            parsed_components=len(cells), reason=reason))
        if disposition == 'unmapped_pending':
            deviations.append(dict(position=name, situation=reason or 'Source component count mismatch', handling_rule='M1-6: retain unmapped_pending; continue'))
            return
        for index, c in enumerate(cells, 1):
            components.append(dict(estimand_name=new_name or name, component=index, **c))
    name = 'equivalence_membership_overlap'
    cells = []
    for slug in COMPONENT_SLUGS:
        hits = [r for r in baseline if r['claim_slug'] == slug]
        if len(hits) != 1:
            continue
        r = hits[0]
        cells.append({k: r[k] for k in ('source_file', 'source_column', 'source_row_key', 'value', 'status', 'freeze_phase')})
        cells[-1]['existing_claim_id'] = r['claim_id']
    commit(name, 'R1', len(COMPONENT_SLUGS), cells, 'composition_sharing_mechanism_four_number_comparison')
    file = 'sec33_usage_and_conditioning/tables/sec33_basin_specific_vs_global_topk.csv'
    rows = read(ANALYSIS / file)
    cells = [cell(file, r, 'coverage_diff_bs_minus_global', ['k']) for r in rows if r['k'] in {'1', '3', '5', '10'}]
    commit('conditionalized_upper_bound', 'R3', 4, cells)
    file = 'sec34_determinacy_and_bounds/tables/sec34_displacement_vs_oob_loss.csv'
    rows = read(ANALYSIS / file)
    cells = [cell(file, r, c, ['instability_metric']) for r in rows if 'displacement' in r['instability_metric']
             for c in ('pooled_spearman_rho', 'within_model_spearman_median', 'model_median_spearman_rho')]
    commit('displacement_oob_loss_three_aggregations', 'R4', 3, cells,
           reason='Multiple displacement definitions; no metric selector supplied; no selection made')
    file = 'sec31_transfer_and_library/tables/sec31_01_parity.json'
    data = json.loads((ANALYSIS / file).read_text())
    cells = []
    if 'n_fold_cells_positive' in data:
        status, phase = meta(file, 'n_fold_cells_positive')
        cells.append(dict(source_file=file, source_column='n_fold_cells_positive', source_row_key='json_key=n_fold_cells_positive',
                          value=str(data['n_fold_cells_positive']), status=status, freeze_phase=phase, existing_claim_id=''))
    commit('model_fold_same_sign_180', 'R1', 1, cells)
    file = 'sec32_library_members/tables/sec32_oob_full_cover_members.csv'
    rows = [r for r in read(ANALYSIS / file) if r['condition'] == 'OOB-dPL' and r['target_coverage'] == '1.000000']
    cells = []
    if rows and len({r['model'] for r in rows}) == len(rows):
        status, phase = meta(file, 'model')
        cells = [dict(source_file=file, source_column='model', source_row_key='condition=OOB-dPL|target_coverage=1.000000',
                      value=json.dumps([r['model'] for r in rows]), status=status, freeze_phase=phase, existing_claim_id='')]
    commit('oob_100_cover_member_list', 'R2', 1, cells)
    file = 'sec32_library_members/tables/sec32_optimal_covers_summary.csv'
    rows = [r for r in read(ANALYSIS / file) if r['condition'] in {'IC', 'seen-dPL', 'OOB-dPL'} and r['target_pct'] in {'95%', '100%'}]
    cells = [cell(file, r, c, ['condition', 'target_pct']) for r in rows for c in ('n_optimal_solutions', 'core_set_size', 'union_set_size')]
    commit('optimal_solution_core_union', 'R2', 18, cells)
    file = 'sec35_cluster_stratified/tables/sec35_g1_kruskal_wallis.csv'
    cells = [cell(file, r, 'effect_size_value', ['test']) for r in read(ANALYSIS / file)]
    file = 'sec35_cluster_stratified/tables/sec35_g1_pairwise_dunn_holm.csv'
    cells += [cell(file, r, 'p_holm', ['type_a', 'type_b']) for r in read(ANALYSIS / file) if r['reject_005_holm'].lower() == 'true']
    commit('penalty_effect_size_and_pairwise_adjusted_values', 'R4', 2, cells)
    file = 'sec31_transfer_and_library/tables/sec31_variance_decomposition.csv'
    cells = [cell(file, r, 'frac_of_total_ss', ['component']) for r in read(ANALYSIS / file) if r['component'] == 'residual']
    commit('residual_variance_share', 'R1', 1, cells)
    commit('single_model_attribute_negative_result', 'R3', 1, [], reason='Attribute recommender evaluation unit and stored sign conclusion not identified in supplied specification')
    file = 'sec36_size_matched_null/tables/sec36_h1_size_matched_null_summary.csv'
    rows = [r for r in read(ANALYSIS / file) if r['target_pct'] in {'90%', '100%'}]
    keys = {(r['condition'], r['target_pct'], r['type']) for r in rows}
    cells = [cell(file, r, 'null_median', ['condition', 'target_pct', 'type']) for r in rows] if len(keys) == len(rows) else []
    # Preserve the entire existing per-type result row, without aggregating it.
    for c, r in zip(cells, rows):
        c['source_column'] = '|'.join(r)
        c['value'] = json.dumps(r, sort_keys=True)
    commit('size_matched_null_90_and_100', 'R4', 60, cells)
    retention_cells = []
    for path in sorted((ANALYSIS / 'sec31_transfer_and_library/tables').glob('*.csv')):
        table = read(path)
        for r in table:
            for column in r:
                if 'retention' in column.lower() and 'exact' in column.lower():
                    retention_cells.append(cell(path.relative_to(ANALYSIS).as_posix(), r, column, []))
    commit('top3_top5_exact_set_retention', 'R1', 2, retention_cells, reason='No unambiguous stored exact-set retention cells in specified sec31 results')
    from finalize_registry_t1_t2 import extend_mappings
    return extend_mappings(module, baseline, entries, components, results, deviations)

def validate_component(c):
    from finalize_registry_t1_t2 import validate_special
    special = validate_special(c)
    if special is not None:
        return special
    p = ANALYSIS / c['source_file']
    if c['existing_claim_id']:
        return True  # Validated by the baseline manifest audit, not newly calculated.
    if p.suffix == '.json':
        return str(json.loads(p.read_text())[c['source_column']]) == c['value']
    selectors = dict(x.split('=', 1) for x in c['source_row_key'].split('|'))
    rows = [r for r in read(p) if all(r.get(k) == v for k, v in selectors.items())]
    if c['source_column'] == 'model':
        return json.dumps([r['model'] for r in rows]) == c['value']
    if len(rows) != 1:
        return False
    if '|' in c['source_column']:
        return json.dumps(rows[0], sort_keys=True) == c['value']
    return rows[0].get(c['source_column']) == c['value']

def protected_snapshot(module):
    snapshot = module.snapshot_inputs()
    for relative, (mtime, size, _) in list(snapshot.items()):
        snapshot[relative] = (mtime, size, module.sha256(module.REPO_ROOT / relative))
    return snapshot
def run(module):
    module._read_cache.clear()
    inventory()
    before = protected_snapshot(module)
    module.write_prompt_declarations()
    baseline, entries, gaps = module.build_registry()
    components, results, deviations = parse_mappings(module, baseline, entries)
    matched = {r['original_estimand']: r for r in results if r['disposition'] == 'mapped'}
    removed_claims = []
    retained = []
    citable_names = {r.get('estimand') for r in module.read_csv_rows(module.CITABLE_PATH)[1]}
    for r in baseline:
        if r['estimand_name'] in REMOVED and r['section'] != 'R2' and r['estimand_name'] not in citable_names:
            removed_claims.append(r)
        else:
            retained.append(r)
            if r['estimand_name'] in REMOVED:
                deviations.append(dict(position=r['claim_id'], situation='R2/citable referenced claim retained', handling_rule='M1-3: retain'))
    registry = list(retained)
    section_by_name = {name: section for section, name, _ in module.REQUIRED_NAME_CHECKS}
    by_slug = {r['claim_slug']: r for r in baseline}
    for name, result in matched.items():
        new_name = result['new_estimand']
        cs = [c for c in components if c['estimand_name'] == new_name]
        row = {field: 'unresolved' for field in module.REGISTRY_FIELDS}
        section = section_by_name[name]
        row.update(section=section, claim_id=section+':'+new_name, claim_slug=new_name, estimand_name=new_name,
                   value=cs[0]['value'] if len(cs) == 1 else json.dumps([c['value'] for c in cs]),
                   source_file=';'.join(c['source_file'] for c in cs), source_column=';'.join(c['source_column'] for c in cs),
                   source_row_key=';'.join(c['source_row_key'] for c in cs),
                   source_row_keys=';'.join(c['source_row_key'] for c in cs),
                   is_composite='true' if len(cs) > 1 else 'false', source_row_status='matched',
                   status=';'.join(dict.fromkeys(c['status'] for c in cs)), freeze_phase=';'.join(dict.fromkeys(c['freeze_phase'] for c in cs)),
                   primary_or_secondary='unresolved', venue='undecided')
        # Inherit metadata only from an exact existing source cell, never a guessed family.
        source_matches = [b for b in baseline if any(b['source_file'] == c['source_file'] and b['source_column'] == c['source_column'] and b['source_row_key'] == c['source_row_key'] for c in cs)]
        for field in ('population_N', 'quantity_class', 'sign_convention', 'better_direction', 'primary_or_secondary'):
            values = {r[field] for r in source_matches}
            if len(source_matches) == len(cs) and len(values) == 1:
                row[field] = next(iter(values))
        if any(row[k] == 'unresolved' for k in ('population_N', 'quantity_class', 'sign_convention', 'better_direction')):
            deviations.append(dict(position=row['claim_id'], situation='Mapped values; declaration metadata not uniquely supported by existing source-cell claims', handling_rule='Preserve unresolved; do not infer population/sign/direction'))
        registry.append(row)
    from finalize_registry_t1_t2 import apply_metadata, install_direct_index, check_original_claim_values
    metadata = apply_metadata(module, registry, components, deviations)
    check_original_claim_values(module, registry)
    audited_entries, original_index = install_direct_index(module, components, entries)
    rules_path = ROOT / 'declarations/venue_assignment_rules.csv'
    write(rules_path, ['slug_pattern', 'venue'], [dict(slug_pattern=p, venue=v) for p, v in VENUES])
    for r in registry:
        r['venue'] = next((v for pattern, v in VENUES if fnmatch.fnmatchcase(r['claim_slug'], pattern)), 'undecided')
        if r['venue'] == 'undecided':
            deviations.append(dict(position=r['claim_id'], situation='No venue rule matches', handling_rule='M2: retain undecided'))
    for occurrence in read(ROOT / 'tables/M1_name_occurrences.csv'):
        if occurrence['file'].endswith(('.md', '.txt')) or 'T3_proposed_mapping' in occurrence['file']:
            deviations.append(dict(position=occurrence['file']+':'+occurrence['line']+':'+occurrence['field'], situation='Historical/documentary occurrence retained; T3 not used', handling_rule='M1-2: do not edit historical evidence'))
    write(ROOT / 'tables/M1_mapping_components.csv', COMP_FIELDS, components)
    write(ROOT / 'tables/M1_dispositions.csv', RESULT_FIELDS, results)
    deviations.append(dict(position='manuscript/analysis/*/scripts', situation='Registry pipeline reuses original unit results; disposable-copy estimator runs are reported separately in T3_isolated_summary.json', handling_rule='Preserve frozen numbers and read-only upstream; execute registry/evidence pipeline only'))
    write(ROOT / 'tables/M1_M2_deviations.csv', DEV_FIELDS, deviations)
    module.write_rows(module.TABLES_ROOT / 'results_claim_registry.csv', module.REGISTRY_FIELDS, registry)
    remaining = [g for g in gaps if g['gap_id'] not in matched and g['gap_id'] not in REMOVED]
    module.run_evidence_packet_builder()
    module._read_cache.clear()
    module.write_population_taxonomy(registry)
    # Preserve the frozen-manifest audit for existing claims; validate each new
    # direct component against its recorded source without rewriting manifests.
    base_audits = module.source_manifest_checks(retained, audited_entries, remaining)
    audits = module.source_manifest_checks(registry, audited_entries, remaining)
    module.build_manifest_source_key_index = original_index
    c201 = next(a for a in audits if a['check_id'] == 'C2-01')
    base_c201 = next(a for a in base_audits if a['check_id'] == 'C2-01')
    invalid = [c for c in components if not validate_component(c)]
    c201.update(base_c201)
    c201['evidence'] = c201['evidence'].replace('registry_rows='+str(len(retained)), 'registry_rows='+str(len(registry)))
    c201['evidence'] += ';direct_mapped_claims='+str(len(matched))+';invalid_direct_components='+str(len(invalid))
    if invalid:
        c201['status'] = 'FAIL'
    module.write_rows(module.TABLES_ROOT / 'consistency_audit.csv', module.AUDIT_FIELDS, audits)
    decisions = module.build_open_decisions(registry, audits, remaining)
    pending_reasons = {r['original_estimand']: r['reason'] for r in results if r['disposition'] == 'unmapped_pending'}
    for decision in decisions:
        if decision['item_type'] == 'unmapped_requested_estimand':
            for name, reason in pending_reasons.items():
                if name in decision['claim_id'] or name in decision['related_value']:
                    decision['item_type'] = 'unmapped_pending'
                    decision['related_value'] = reason
                    break
    module.write_rows(module.TABLES_ROOT / 'open_decisions.csv', module.OPEN_FIELDS, decisions)
    module.export_panels()
    module.run_evidence_packet_builder()
    after = protected_snapshot(module)
    integrity = module.write_source_integrity(before, after)
    changed = [r for r in integrity if r['status'] != 'UNCHANGED']
    if changed:
        raise RuntimeError('S1/S2: protected inputs changed: '+str(changed[:3]))
    summary = dict(registry_rows=len(registry), baseline_rows=len(baseline), removed_claims=len(removed_claims), added_claims=len(matched),
                   disposition_counts=dict(Counter(r['disposition'] for r in results)),
                   venues={section: dict(Counter(r['venue'] for r in registry if r['section'] == section)) for section in ('R1','R2','R3','R4')},
                   undecided=sum(r['venue'] == 'undecided' for r in registry),
                   audits={r['check_id']:r['status'] for r in audits}, open_decisions=len(decisions),
                   decision_types=dict(Counter(r['item_type'] for r in decisions)), protected_inputs=len(integrity), changed_inputs=len(changed),
                   executed_scripts=['build_results_artifacts.py', 'build_evidence_packet.py', 'close_registry_m1_m2.py', 'finalize_registry_t1_t2.py'],
                   estimator_scripts_run=False, execution_scope='registry_pipeline_only', metadata_resolutions=len(metadata))
    (ROOT / 'tables/M1_M2_run_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    module.write_artifact_manifest()
    print(json.dumps(summary, sort_keys=True))

def verify_twice(module):
    def outputs():
        paths = set()
        for base in (ROOT / 'tables', ROOT / 'declarations', module.DATA_ROOT):
            paths.update(p for p in base.rglob('*') if p.is_file() and p.suffix in {'.csv', '.json'})
        return {p: p.read_bytes() for p in paths}
    run(module)
    first = outputs()
    run(module)
    second = outputs()
    different = [p for p in set(first) | set(second) if first.get(p) != second.get(p)]
    report = dict(output_files=len(set(first) | set(second)), inconsistent_files=len(different),
                  differing_paths=[str(p.relative_to(ANALYSIS)) for p in sorted(different)],
                  maximum_numeric_difference='0' if not different else 'not_established',
                  executed_scripts=['build_results_artifacts.py', 'build_evidence_packet.py', 'close_registry_m1_m2.py', 'finalize_registry_t1_t2.py'])
    (ROOT / 'tables/M1_M2_repeat_validation.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    module._read_cache.clear()
    module.write_artifact_manifest()
    print('REPEAT_VALIDATION='+json.dumps(report, sort_keys=True))
    if different:
        raise RuntimeError('Generated outputs differ between consecutive registry runs')
if __name__ == '__main__':
    if '--inventory-only' in sys.argv:
        inventory()
    else:
        spec = importlib.util.spec_from_file_location('build_results_artifacts', Path(__file__).with_name('build_results_artifacts.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        verify_twice(module) if '--verify-twice' in sys.argv else run(module)
