#!/usr/bin/env python3
"""Final source-backed mappings and metadata. No source tables are written."""
from __future__ import annotations
import csv
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
REPO = ANALYSIS.parents[3]
META_FIELDS = ['claim_id', 'population_N', 'population_category', 'population_meaning',
               'population_source_file', 'population_source_column', 'population_source_key',
               'quantity_class', 'sign_convention', 'better_direction', 'reason']
CLASSES = {
    'composition_sharing_mechanism_four_number_comparison': 'descriptive_count',
    'conditionalized_upper_bound': 'coverage_paired_difference',
    'model_fold_same_sign_180': 'descriptive_count',
    'oob_100_cover_member_list': 'categorical_identifier',
    'optimal_solution_core_union': 'descriptive_count',
    'penalty_effect_size_and_pairwise_adjusted_values': 'p_value_or_effect_size',
    'residual_variance_share': 'variance_or_share',
    'size_matched_null_90_and_100': 'null_composite_summary',
    'displacement_oob_loss_three_aggregations': 'rank_correlation',
    'single_model_attribute_negative_result': 'gain_sign_summary',
    'top3_top5_exact_set_retention': 'coverage_fraction',
}
NEW_DIRECTIONS = (
    ('coverage_paired_difference', 'basin_specific_minus_global', 'higher', 'Paired coverage difference as defined by the source column'),
    ('rank_correlation', 'n/a', 'none', 'Descriptive displacement/loss association; no optimization direction'),
    ('gain_sign_summary', 'n/a', 'none', 'Composite seed counts and maximum gain; maximum G alone has higher-is-better direction'),
    ('null_composite_summary', 'component_specific', 'none', 'Stored row dictionaries combine cover counts, null summaries and test probabilities; no scalar ordering of the whole object'),
)

def read(p):
    with p.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write(p, fields, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)

def rel(p):
    import os
    return os.path.relpath(p, ANALYSIS)

def source_rows(file):
    return read(ANALYSIS / file)

def derived_value(column, rows):
    if column == 'derived:unique_seed_count':
        return str(len({r['seed'] for r in rows}))
    if column == 'derived:negative_G_1_seed_count':
        return str(sum(Decimal(r['G_1']) < 0 for r in rows))
    if column == 'derived:maximum_G_1':
        return max(rows, key=lambda r: Decimal(r['G_1']))['G_1']
    raise ValueError(column)

def extend_mappings(module, baseline, entries, components, results, deviations):
    roots = [ANALYSIS, REPO / 'project/benchmarkpub/analysis', REPO / 'project/benchmark/analysis', REPO / 'project/benchmark/manuscript']
    search, candidates = [], []
    producers = {
        'topk_overlap_summary.csv': REPO / 'project/benchmarkpub/scripts/oob36_prelim/18_basin_rank_stability.py',
        'top3_top5_set_overlap.csv': REPO / 'project/benchmarkpub/scripts/oob36_prelim/22_candidate_set_transfer.py',
    }
    seen = set()
    for root in roots:
        scanned = 0
        for p in sorted(root.rglob('*.csv')):
            if p in seen or '_results_registry' in p.parts:
                continue
            seen.add(p); scanned += 1
            try:
                with p.open(newline='', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f); fields = reader.fieldnames or []
                    if 'full_set_retained_fraction' not in fields or 'intersection_distribution' not in fields:
                        if any(re.search(r'retain|retention|full_set', column, re.I) for column in fields):
                            search.append(dict(directory=str(root),file=str(p),result='retention-related header inspected',rows='',reason='No stored aggregate fraction plus intersection distribution schema',producer='',row_keys='|'.join(fields)))
                        continue
                    rows = list(reader)
            except (UnicodeError, csv.Error):
                continue
            hits = []
            for r in rows:
                if r.get('k') not in {'3', '5'}:
                    continue
                distribution = json.loads(r['intersection_distribution'])
                count = distribution.get(r['k'])
                n = sum(distribution.values())
                expected = {'3': 26, '5': 40}[r['k']]
                if n == 531 and count == expected:
                    hits.append(r)
            if len(hits) != 2:
                search.append(dict(directory=str(root), file=str(p), result='no unique qualifying Top-3/Top-5 pair', rows=len(rows), reason='N/count checks or pairing failed', producer='', row_keys=''))
                continue
            producer = producers.get(p.name)
            producer_text = producer.read_text() if producer and producer.is_file() else ''
            keyed = ('.pivot(' in producer_text and '.loc[' in producer_text) or 'seen_oob_matrices(' in producer_text
            accessor = 'basin_accessor' in producer_text and ('align_' in producer_text or 'canonical_basin_ids' in producer_text)
            keys = ['|'.join(k+'='+r[k] for k in ('variant', 'k') if k in r) for r in hits]
            candidates.append((p, hits, producer, keyed, accessor))
            search.append(dict(directory=str(root), file=str(p), result='qualifying candidate', rows=len(rows), reason='ID-keyed pivot='+str(keyed)+'; corrected accessor='+str(accessor), producer=str(producer or ''), row_keys=';'.join(keys)))
        json_scanned = 0
        def retention_paths(obj, prefix=''):
            found = []
            if isinstance(obj,dict):
                for name,value in obj.items():
                    key = prefix+'/'+name
                    if re.search(r'retain|retention|full_set',name,re.I): found.append(key)
                    found.extend(retention_paths(value,key))
            elif isinstance(obj,list):
                for index,value in enumerate(obj):
                    if isinstance(value,(dict,list)): found.extend(retention_paths(value,prefix+'/'+str(index)))
            return found
        for p in sorted(root.rglob('*.json')):
            if '_results_registry' in p.parts: continue
            json_scanned += 1
            try:
                paths = retention_paths(json.loads(p.read_text(encoding='utf-8-sig')))
            except (UnicodeError,json.JSONDecodeError):
                search.append(dict(directory=str(root),file=str(p),result='JSON inspection error',rows='',reason='Cannot parse stored JSON; no mapping inferred',producer='',row_keys=''))
                continue
            if paths:
                search.append(dict(directory=str(root),file=str(p),result='structured retention provenance inspected',rows=len(paths),reason='JSON retention fields recorded; numerical mapping requires a uniquely identified table/producer',producer='',row_keys=';'.join(paths)))
        search.append(dict(directory=str(root),file='',result='JSON search scope',rows=json_scanned,reason='Structured result/provenance keys inspected, including _verification',producer='',row_keys=''))
        search.append(dict(directory=str(root), file='', result='search scope', rows=scanned, reason='CSV headers inspected; registry/T3 candidates excluded', producer='', row_keys=''))
    write(ROOT / 'tables/T1_source_search.csv', ['directory', 'file', 'result', 'rows', 'reason', 'producer', 'row_keys'], search)
    qualifying = [c for c in candidates if c[3]]
    comparison_rows = []
    chosen = None
    decision = 'pending'
    decision_basis = 'No unique qualifying candidate'
    if len(qualifying) == 2:
        def by_k(hit_rows):
            return {r.get('k'): r for r in hit_rows}
        first, second = qualifying
        first_by_k, second_by_k = by_k(first[1]), by_k(second[1])
        common = ('intersection_distribution', 'full_set_retained_fraction', 'intersection_mean')
        numeric_aliases = {'recall_mean': 'retention_mean'}
        summary_equal = True
        for k in ('3', '5'):
            a, b = first_by_k[k], second_by_k[k]
            field_equal = {}
            for field in common:
                field_equal[field] = (a[field] == b[field])
            for field, alias in numeric_aliases.items():
                field_equal[field+'='+alias] = abs(Decimal(a[field]) - Decimal(b[alias])) <= Decimal('1e-15')
            summary_equal = summary_equal and all(field_equal.values())
            comparison_rows.append(dict(candidate_a=str(first[0]), candidate_b=str(second[0]), k=k,
                                        count_a=a['intersection_distribution'], count_b=b['intersection_distribution'],
                                        ratio_a=a['full_set_retained_fraction'], ratio_b=b['full_set_retained_fraction'],
                                        summary_fields_equal=all(field_equal.values()), field_details=json.dumps(field_equal, sort_keys=True),
                                        per_basin_markers_equal='', per_basin_marker_count='', rule_branch=''))
        def marker_map(item):
            p, hits, producer, keyed, accessor = item
            if p.name == 'topk_overlap_summary.csv':
                marker_file = p.parent / 'basin_rank_stability.csv'
                rows = read(marker_file)
                return {r['basin_id'].zfill(8): (r['top3_full_set_retained'].lower() == 'true', r['top5_full_set_retained'].lower() == 'true') for r in rows}, marker_file
            source = REPO / 'project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv'
            source_rows_ = read(source)
            matrix = {}
            for r in source_rows_:
                basin = r['basin_id'].zfill(8); matrix.setdefault(basin, {})[r['model']] = (float(r['kge_seenfolds_median4']), float(r['kge_heldout']))
            models = sorted(next(iter(matrix.values())))
            out = {}
            for basin, vals in matrix.items():
                seen_order = sorted(models, key=lambda m: (-vals[m][0], m))
                oob_order = sorted(models, key=lambda m: (-vals[m][1], m))
                out[basin] = (set(seen_order[:3]) == set(oob_order[:3]), set(seen_order[:5]) == set(oob_order[:5]))
            return out, source
        first_markers, first_marker_file = marker_map(first)
        second_markers, second_marker_file = marker_map(second)
        marker_equal = first_markers == second_markers and len(first_markers) == 531
        for row in comparison_rows:
            row['per_basin_markers_equal'] = marker_equal
            row['per_basin_marker_count'] = len(first_markers)
        if summary_equal and marker_equal:
            chosen = max(qualifying, key=lambda item: item[0].stat().st_mtime_ns)
            decision = 'b'
            decision_basis = 'Both candidates use ID-keyed pivot/.loc alignment rather than _shared/basin_accessor.py; counts, proportions and 531 per-basin retained markers are identical; later generated candidate selected.'
            for row in comparison_rows: row['rule_branch'] = 'b'
        else:
            decision_basis = 'Candidates differ in stored summary or per-basin markers; no new recomputation was authorized by the current implementation.'
    if chosen is None and len(qualifying) == 1:
        chosen = qualifying[0]
        decision = 'a'
        decision_basis = 'Exactly one qualifying candidate.'
    write(ROOT / 'tables/F1_candidate_comparison.csv',
          ['candidate_a','candidate_b','k','count_a','count_b','ratio_a','ratio_b','summary_fields_equal','field_details','per_basin_markers_equal','per_basin_marker_count','rule_branch'], comparison_rows)
    search.append(dict(directory=str(ROOT), file='', result='F1 decision', rows=len(comparison_rows), reason='rule_branch='+decision+'; '+decision_basis, producer=str(chosen[2]) if chosen else '', row_keys='k=3;k=5'))
    def replace(name, cells, reason='', expected=None):
        nonlocal components
        components = [c for c in components if c['estimand_name'] != name]
        result = next(r for r in results if r['original_estimand'] == name)
        result.update(disposition='mapped' if cells else 'unmapped_pending', parsed_components=len(cells),
                      expected_components=len(cells) if expected is None and cells else expected or result['expected_components'], reason=reason)
        for i, c in enumerate(cells, 1):
            components.append(dict(estimand_name=name, component=i, **c))
        deviations[:] = [d for d in deviations if d['position'] != name]
        if reason:
            deviations.append(dict(position=name, situation=reason, handling_rule='T1: use source evidence; otherwise retain pending'))
    if chosen:
        p, hits, producer, _, _ = chosen
        cells = [dict(source_file=rel(p), source_column='full_set_retained_fraction',
                      source_row_key='|'.join(k+'='+r[k] for k in ('variant','k') if k in r),
                      value=r['full_set_retained_fraction'], status='NOT_LISTED', freeze_phase='NOT_LISTED', existing_claim_id='') for r in hits]
        replace('top3_top5_exact_set_retention', cells)
    else:
        replace('top3_top5_exact_set_retention', [], 'Multiple validated result tables; no unique corrected-accessor producer established; see T1_source_search.csv')
    file = 'sec34_determinacy_and_bounds/tables/sec34_displacement_vs_oob_loss.csv'
    hits = [r for r in source_rows(file) if 'displacement' in r['instability_metric']]
    groups = {
        'pooled': ('pooled_spearman_rho','pooled_spearman_p_permutation','pooled_n'),
        'within_model': ('within_model_spearman_median','within_model_spearman_p_permutation','within_models_used'),
        'model_level': ('model_median_spearman_rho','model_median_spearman_p_permutation','model_median_n'),
    }
    cells, details = [], []
    for r in hits:
        for aggregation,(statistic,pvalue,n_column) in groups.items():
            key = 'instability_metric='+r['instability_metric']
            cells.append(dict(source_file=file, source_column=statistic, source_row_key=key,
                              value=r[statistic], status='NOT_LISTED', freeze_phase='NOT_LISTED', existing_claim_id=''))
            details.append(dict(source_file=file, source_row_key=key, aggregation=aggregation,
                                statistic_column=statistic, statistic=r[statistic], p_column=pvalue if pvalue in r else '', p=r.get(pvalue,''),
                                p_status='stored' if pvalue in r else 'not_stored', N_column=n_column, N=r[n_column],
                                positive_model_count=r.get('within_model_positive_model_count',r.get('within_model_spearman_n_positive','')) if aggregation=='within_model' else '',
                                fraction_negative=r.get('within_model_spearman_frac_negative','') if aggregation=='within_model' else '',
                                fraction_p_lt_005=r.get('within_model_spearman_frac_p_lt_005','') if aggregation=='within_model' else ''))
    write(ROOT / 'tables/T1_displacement_aggregations.csv', ['source_file','source_row_key','aggregation','statistic_column','statistic','p_column','p','p_status','N_column','N','positive_model_count','fraction_negative','fraction_p_lt_005'], details)
    replace('displacement_oob_loss_three_aggregations', cells, 'All stored displacement definitions retained across the three aggregations; no definition selected. Within-model p and strict positive-count fields are not stored; fractions are preserved without inferring positive counts.')
    # Stored RF single-model evaluation: the producer and its log certify the SBS reference.
    p = REPO / 'project/benchmark/analysis/chapter4_final_supplement_20260925/tables/f5_seed_per_seed.csv'
    producer = p.parents[1] / 'scripts/f5_seed_robustness.py'
    log = p.parents[1] / 'logs/f5.log'
    rows = read(p) if p.exists() else []
    source_code = producer.read_text() if producer.exists() else ''
    log_text = log.read_text() if log.exists() else ''
    if rows and len(rows) == len({r['seed'] for r in rows}) == 10 and 'RandomForestRegressor' in source_code and "SBS models = ['hbv96']" in log_text:
        columns = ('derived:unique_seed_count', 'derived:negative_G_1_seed_count', 'derived:maximum_G_1')
        cells = [dict(source_file=rel(p), source_column=c, source_row_key='aggregate=all_unique_seeds|gain_column=G_1',
                      value=derived_value(c, rows), status='NOT_LISTED', freeze_phase='NOT_LISTED', existing_claim_id='') for c in columns]
        replace('single_model_attribute_negative_result', cells)
        write(ROOT / 'tables/T1_single_model_gain_summary.csv', ['source_file','producer_script','source_row_key','component','value','quantity_class','sign_convention','better_direction'],
              [dict(source_file=rel(p),producer_script=rel(producer),source_row_key=c['source_row_key'],component=c['source_column'],value=c['value'],
                    quantity_class='gain_level' if c['source_column']=='derived:maximum_G_1' else 'descriptive_count',
                    sign_convention='mean(KGE_RF-KGE_HBV96)/mean(KGE_VBS-KGE_HBV96)' if c['source_column']=='derived:maximum_G_1' else 'n/a',
                    better_direction='higher' if c['source_column']=='derived:maximum_G_1' else 'none') for c in cells])
        search.append(dict(directory=str(p.parents[1]), file=str(p), result='RF single-model gain source validated', rows=len(rows),
                           reason='Producer uses RF; log certifies fixed HBV96; unique seeds verified', producer=str(producer), row_keys='seed'))
    else:
        replace('single_model_attribute_negative_result', [], 'RF/unique-seed/fixed-HBV96 provenance not established in scanned result directories')
    write(ROOT / 'tables/T1_source_search.csv', ['directory', 'file', 'result', 'rows', 'reason', 'producer', 'row_keys'], search)
    return components, results, deviations

def validate_special(component):
    column = component['source_column']
    if column.startswith('derived:'):
        rows = source_rows(component['source_file'])
        return len(rows) == len({r['seed'] for r in rows}) and derived_value(column, rows) == component['value']
    return None

def apply_metadata(module, registry, components, deviations):
    mapped_names = {c['estimand_name'] for c in components}
    rules = read(ROOT / 'declarations/claim_quantity_class_map.csv')
    extra_rules = [dict(slug_pattern=name, quantity_class=cls) for name,cls in CLASSES.items() if name in mapped_names]
    rules = extra_rules + [r for r in rules if r['slug_pattern'] not in CLASSES]
    module.write_rows(ROOT / 'declarations/claim_quantity_class_map.csv', ['slug_pattern','quantity_class'], rules)
    direction_rows = read(ROOT / 'declarations/sign_and_direction.csv')
    direction_rows += [dict(quantity_class=c, subtraction_order=s, better_direction=d, note=n) for c,s,d,n in NEW_DIRECTIONS if c not in {r['quantity_class'] for r in direction_rows}]
    module.write_rows(ROOT / 'declarations/sign_and_direction.csv', ['quantity_class','subtraction_order','better_direction','note'], direction_rows)
    directions = {r['quantity_class']:r for r in direction_rows}
    records = []
    basin_file = 'sec32_library_members/tables/sec32_basin_singleton_status.csv'
    basin_rows = source_rows(basin_file)
    canonical = set(module.canonical_basin_ids())
    by_condition = {c:[r for r in basin_rows if r['condition']==c] for c in ('IC','seen-dPL','OOB-dPL')}
    for rows in by_condition.values():
        assert len(rows) == len({r['basin_id'].zfill(8) for r in rows}) and {r['basin_id'].zfill(8) for r in rows} == canonical
    for row in registry:
        name = row['estimand_name']
        if name not in mapped_names or name not in CLASSES:
            continue
        category, meaning, file, column, key = 'population','full basin domain',basin_file,'basin_id','condition=OOB-dPL; unique ID-keyed rows'
        n = str(len(by_condition['OOB-dPL']))
        if name == 'conditionalized_upper_bound':
            p = module.S.DIR_OOB36_TRANSFER / 'tables/candidate_set_transfer_by_basin.csv'
            rows = read(p)
            assert {r['basin_id'].zfill(8) for r in rows} == canonical and len(rows) == len(canonical)
            n, file, key = str(len(rows)), rel(p), 'unique basin_id rows'
        elif name == 'model_fold_same_sign_180':
            file = 'sec31_transfer_and_library/tables/sec31_model_fold_delta_kge.csv'
            rows = source_rows(file)
            assert len(rows) == len({(r['model'],r['fold']) for r in rows})
            n, category, meaning, column, key = str(len(rows)), 'analysis_unit_count','model×fold cells','model|fold','unique model×fold rows'
        elif name == 'penalty_effect_size_and_pairwise_adjusted_values':
            file = 'sec35_cluster_stratified/tables/sec35_g1_kruskal_wallis.csv'
            rows = source_rows(file); assert len(rows)==1
            n, column, key, meaning = rows[0]['n_total'], 'n_total','test=Kruskal-Wallis','type-matched basin domain; pair subgroup Ns remain source-specific'
        elif name == 'size_matched_null_90_and_100':
            file = 'sec36_size_matched_null/tables/sec36_h1_run_meta.csv'
            rows = source_rows(file); assert len(rows)==1
            n, column, key, meaning = rows[0]['matched'],'matched','unique run metadata row','type-matched sampling population; each type N remains in source component'
        elif name == 'residual_variance_share':
            file = 'sec31_transfer_and_library/tables/sec31_pooled_delta_kge.csv'
            rows = source_rows(file); assert len(rows)==1
            totals = {r['component']:r for r in source_rows('sec31_transfer_and_library/tables/sec31_variance_decomposition.csv')}
            assert Decimal(rows[0]['n_cells']) == Decimal(totals['total']['df']) + 1
            n, column, key, meaning = rows[0]['n_cells'],'n_cells','metric='+rows[0]['metric'],'same model×basin population; source N checked against variance total df'
        elif name == 'displacement_oob_loss_three_aggregations':
            cs = [c for c in components if c['estimand_name']==name and c['source_column']=='pooled_spearman_rho']
            keys = {c['source_row_key'] for c in cs}
            file = cs[0]['source_file']
            rows = [r for r in source_rows(file) if 'instability_metric='+r['instability_metric'] in keys]
            values = {r['pooled_n'] for r in rows}; assert len(values)==1
            n, column, key, meaning = values.pop(),'pooled_n',';'.join(sorted(keys)),'pooled model×basin population; model/within-model analysis Ns remain source-specific'
        elif name == 'single_model_attribute_negative_result':
            cs = [c for c in components if c['estimand_name']==name and c['source_column']=='derived:unique_seed_count']; assert len(cs)==1
            n, file, column, key, category, meaning = cs[0]['value'],cs[0]['source_file'],'seed',cs[0]['source_row_key'],'analysis_unit_count','seed replicates'
        cls = CLASSES[name]
        direction_cls = {'cover_count':'cover_count / library_size','p_value_or_effect_size':'p_value / effect_size','coverage_fraction':'spearman / retention / coverage_fraction'}.get(cls,cls)
        declaration = directions.get(direction_cls)
        row['population_N'],row['quantity_class'] = n,cls
        row['sign_convention'] = declaration['subtraction_order'] if declaration else ''
        row['better_direction'] = declaration['better_direction'] if declaration else ''
        records.append(dict(claim_id=row['claim_id'], population_N=n, population_category=category, population_meaning=meaning,
                            population_source_file=file, population_source_column=column, population_source_key=key,
                            quantity_class=cls, sign_convention=row['sign_convention'], better_direction=row['better_direction'], reason=''))
    # Resolve the original alias from the producer's metric order, not a symmetric T3 pool.
    legacy = [r for r in registry if r['claim_slug']=='inter_metric_spearman_fold_instability_3caed92d']
    if len(legacy)==1:
        file = 'sec34_determinacy_and_bounds/tables/sec34_displacement_metrics_correlation_matrix.csv'
        producer = ANALYSIS / 'sec34_determinacy_and_bounds/scripts/sec34_08_displacement_correlation_recompute_e7.py'
        code = producer.read_text()
        rows = [r for r in source_rows(file) if r['instability_metric']=='param_fold_instability_median']
        if len(rows)==1 and '"param_displacement_median": float(np.median(disp_rms))' in code:
            value = rows[0]['param_displacement_median']; old = legacy[0]
            frozen = Decimal(old['value']); exact = Decimal(value)
            quantum = Decimal(1).scaleb(frozen.as_tuple().exponent)
            agrees = exact.quantize(quantum) == frozen
            key = 'instability_metric=param_fold_instability_median'
            write(ROOT / 'tables/T2_source_alias_resolution.csv', ['claim_id','source_file','producer_script','source_column','source_row_key','frozen_claim_value','exact_source_value','absolute_display_difference','source_rounds_to_frozen','handling'],
                  [dict(claim_id=old['claim_id'],source_file=file,producer_script=rel(producer),source_column='param_displacement_median',source_row_key=key,
                        frozen_claim_value=old['value'],exact_source_value=value,absolute_display_difference=str(abs(exact-frozen)),source_rounds_to_frozen=agrees,
                        handling='Preserve frozen claim value; direct producer-defined source identity, not symmetric-value matching')])
            if agrees:
                old.update(source_column='param_displacement_median',source_row_key=key,source_row_keys=key,source_row_status='matched')
                components.append(dict(estimand_name=old['estimand_name'], component=len(components)+1, source_file=file,
                                       source_column='param_displacement_median',source_row_key=key,value=old['value'],status=old['status'],freeze_phase=old['freeze_phase'],existing_claim_id=old['claim_id']))
                deviations.append(dict(position=old['claim_id'], situation='Alias resolved from producer metric order and RMS definition; source cell rounds to frozen claim precision; original value retained (see T2_source_alias_resolution.csv)',handling_rule='T2: direct source lookup; no T3 candidate used'))
            else:
                deviations.append(dict(position=old['claim_id'], situation='Producer-defined source cell disagrees even at frozen display precision',handling_rule='T2: retain unresolved source key; do not change frozen value'))
        else:
            deviations.append(dict(position=legacy[0]['claim_id'],situation='Alias definition not established',handling_rule='T2: retain unresolved'))
    deviations[:] = [d for d in deviations if not (d['position'] in {r['claim_id'] for r in records} and 'metadata not uniquely supported' in d['situation'])]
    write(ROOT / 'tables/T2_metadata_resolution.csv', META_FIELDS, records)
    module._read_cache.clear()
    return records

def check_original_claim_values(module, registry):
    baseline_path = ROOT / 'tables/T2_original_claim_value_baseline.csv'
    if not baseline_path.exists():
        original = read(ROOT / 'tables/results_claim_registry.csv')
        write(baseline_path,['claim_id','value'],[{k:r[k] for k in ('claim_id','value')} for r in original])
    original = read(baseline_path)
    current = {r['claim_id']:r['value'] for r in registry}
    checks = [dict(claim_id=r['claim_id'],original_value=r['value'],current_value=current.get(r['claim_id'],''),unchanged=current.get(r['claim_id'])==r['value']) for r in original]
    if any(not r['unchanged'] for r in checks):
        raise RuntimeError('BLOCKED(S1): pre-existing claim value would change')
    write(ROOT / 'tables/T2_original_claim_value_integrity.csv',['claim_id','original_value','current_value','unchanged'],checks)

def install_direct_index(module, components, entries):
    original = module.build_manifest_source_key_index
    virtual_entries = []
    for c in components:
        virtual_entries.append(dict(estimand=c['estimand_name'],file_path=c['source_file'],column_or_key=c['source_column'],
                                    value=c['value'],status=c['status'],freeze_phase=c['freeze_phase']))
    def index(all_entries):
        by_file, by_key = original(entries)
        for c,e in zip(components, virtual_entries):
            record = dict(entry=e,source_row_status='matched',source_row_key=c['source_row_key'])
            by_file[(c['source_file'],c['source_row_key'])].append(record)
            by_key[c['source_row_key']].append(record)
        return by_file,by_key
    module.build_manifest_source_key_index = index
    return entries+virtual_entries, original
