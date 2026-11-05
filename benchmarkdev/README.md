# benchmarkdev — GMD model-library verification workspace

Only this workspace is writable. `dmotpy`, `project/benchmark`, `releases`, and
`data` are imported/read-only. Existing results are never repaired or overwritten.
The audit explicitly distinguishes full runtime reproduction from syntax/header
checks; PARTIAL outputs are not manuscript-ready verification claims.

## Layout

- `config/config.json`: immutable pre-run tolerances, seed, paths and contracts.
- `src/benchmarkdev/core.py`: public model, dataset, IC/checkpoint, attribute and KGE adapters.
- `tests/test_contracts.py`: registry, minimal forward, original/new parity, provenance.
- `analyses/run.py`: C0–C9 serial analysis entry points and script-generated report.
- `manuscript/data`, `tables`, `logs`: machine-generated evidence.
- `manuscript/figures`: intentionally empty; no figures requested here.
- `cache`: completion markers and local test dependencies; no original environments changed.

## Environment and order

Use the repository's existing `.venv/bin/python`. Disable bytecode writes. CPU
threads are bounded. Do not start concurrent GPU workers. Local pytest dependencies
are isolated under `cache/test_dependencies`.

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/project/benchmarkdev/src:$PWD/project/benchmarkdev/cache/test_dependencies:$PWD"
.venv/bin/python -B project/benchmarkdev/analyses/run.py --all
.venv/bin/python -B -m pytest -p no:cacheprovider project/benchmarkdev/tests -q \
  --junitxml=project/benchmarkdev/cache/pytest.xml
.venv/bin/python -B project/benchmarkdev/analyses/run.py --report
# Serial continuation with phase deadlines, optional audited Heun and final JUnit summary:
.venv/bin/python -B project/benchmarkdev/analyses/serial_finish.py
# Rebuild the eight-section report from completed machine artifacts:
.venv/bin/python -B project/benchmarkdev/analyses/finalize.py
```

Resume uses stage completion markers. Incomplete scientific coverage is documented
with PARTIAL status; a marker means the attempted stage ended, not that every
requested configuration passed. Individual stage invocation uses `--stage C4`.

## Metric contracts

Public verification KGE delegates to `dmotpy.losses.KgeLoss` (sample std,
observation-valid mask, nonfinite predictions rejected, public epsilon).
Historical dPL losses differ in epsilon, population moments, negative-flow masking
and square-root safeguards. Never substitute one for another when checking archived
training diagnostics. The P2 summary imports the original selector directly.

## Current limitations

See machine-generated `manuscript/REPORT.md` and `data/analysis_status.csv`.
C4 refines only models with audited dimensional contracts; unsupported daily
routing/calendar/interception semantics are explicitly excluded. C6 implements
serial GPU/CPU, dtype and backend configurations plus the permitted single-restart
calibration fallback; measured coverage, not implementation presence, determines
completion. The historical Euler harness cannot be used unchanged because its
return interpretation conflicts with
the current core API. Skipped analyses are not represented as passed tests.
