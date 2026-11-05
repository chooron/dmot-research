# Data freeze

- Freeze date: 20261005
- Freeze record: `project/benchmarkpub/manuscript/analysis/_freeze/freeze_record_20261005.json`
- Files recorded: 311
- Registry entries: 190

## Frozen scope

- `project/benchmark/analysis/chapter4_final_supplement_20260925/logs/`
- `project/benchmark/analysis/chapter4_final_supplement_20260925/scripts/`
- `project/benchmark/analysis/chapter4_final_supplement_20260925/tables/`
- `project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_candidate_set_transfer_20260930/tables/`
- `project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables/`
- `project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_supplemental_rank_extreme_20260930/tables/`
- `project/benchmarkpub/manuscript/analysis/_figures/`
- `project/benchmarkpub/manuscript/analysis/_figures/data/`
- `project/benchmarkpub/manuscript/analysis/_manifest/scripts/`
- `project/benchmarkpub/manuscript/analysis/_manifest/tables/`
- `project/benchmarkpub/manuscript/analysis/_results_registry/declarations/`
- `project/benchmarkpub/manuscript/analysis/_results_registry/scripts/`
- `project/benchmarkpub/manuscript/analysis/_results_registry/tables/`
- `project/benchmarkpub/manuscript/analysis/_results_registry/tables/evidence/`
- `project/benchmarkpub/manuscript/analysis/_verification/scripts/`
- `project/benchmarkpub/manuscript/analysis/_verification/tables/`
- `project/benchmarkpub/manuscript/analysis/sec31_transfer_and_library/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec31_transfer_and_library/tables/`
- `project/benchmarkpub/manuscript/analysis/sec32_library_members/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec32_library_members/tables/`
- `project/benchmarkpub/manuscript/analysis/sec33_usage_and_conditioning/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec33_usage_and_conditioning/tables/`
- `project/benchmarkpub/manuscript/analysis/sec34_determinacy_and_bounds/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec34_determinacy_and_bounds/tables/`
- `project/benchmarkpub/manuscript/analysis/sec35_cluster_stratified/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec35_cluster_stratified/tables/`
- `project/benchmarkpub/manuscript/analysis/sec36_size_matched_null/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec36_size_matched_null/tables/`
- `project/benchmarkpub/manuscript/analysis/sec_si_scope/scripts/`
- `project/benchmarkpub/manuscript/analysis/sec_si_scope/tables/`
- `project/benchmarkpub/scripts/oob36_prelim/`

## Verification

Run the following command from the repository root:

```bash
.venv/bin/python -c "import hashlib,json,pathlib; r=json.load(open('project/benchmarkpub/manuscript/analysis/_freeze/freeze_record_20261005.json')); bad=[x['path'] for x in r['files'] if hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()!=x['sha256'] or pathlib.Path(x['path']).stat().st_size!=x['bytes']]; print('hash_mismatches='+str(len(bad))); raise SystemExit(bool(bad))"
```

## Rule

After this freeze, any numerical change requires a new versioned freeze directory and a written reason; frozen files must not be modified in place.
