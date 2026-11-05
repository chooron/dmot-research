# `oob36_prelim` — code for task `ch4-oob36-download-prelim-20260930`

Download of the OOB-36 results to the local machine plus descriptive statistics.
**No training, no re-inference, no GPU work, no recommendation / candidate-set /
equivalence analysis.**

```
ROOT=/home/jingxin/code/dmg-research/project/benchmarkpub
DATA=$ROOT/data/oob36_20260930
OUT=$ROOT/analysis/chapter4_oob36_prelim_20260930
CODE=$ROOT/scripts/oob36_prelim          # this directory
```

All paths are defined **once** in `common.py`; no other file hardcodes a path.

## Run order

| # | script | reads | writes | notes |
|---|---|---|---|---|
| 01 | `01_remote_inventory.py` | X99 run dirs (read-only, `python -` over stdin) + local existing8 run dirs + frozen manifests | `$OUT/remote_inventory.csv`, `$OUT/remote_inventory.json` | 180 rows (36 models x 5 folds); no file is created on X99 |
| 02 | `02_download.sh` | X99 via `rsync` (`--partial`, **never** `--delete`) and the local frozen tree | `$DATA/{runs,qsim,obs,manifests}`, `$DATA/rename_map.csv` | idempotent; snapshots are renamed to `last.pt`; safe to re-run |
| 03 | `03_verify_download.py` | `$DATA`, frozen manifests, X99 `sha256sum` | `$OUT/download_verification*.csv`, `$OUT/download_verification_summary.json` | strict; `--no-remote-sha` is a non-authoritative local-only mode |
| 04 | `04_build_basin_table.py` | `$DATA`, frozen metric modules, frozen paired seen/IC KGE table | `$OUT/tables/oob36_basin_metrics.{parquet,csv}`, `$OUT/tables/seen_ic_oob_merged.{parquet,csv}`, `$OUT/metric_selfcheck.json`, `$OUT/table_build_summary.json` | step 2.2 self-check must pass (<= 1e-8) before anything else runs |
| 05 | `05_descriptive_stats.py` | 04 outputs, `OOB36_TRAINING_SUMMARY.csv`, frozen `v1_parsimony.csv`, frozen `oob_common.py` | `$OUT/tables/S1..S4*.csv`, `$OUT/stats_summary.json` | S1 hard-stops unless `foldMed`/`pooled` are **bit-equal** to the frozen summary |
| 06 | `06_figures.py` | 04 + 05 outputs | `$OUT/figures/F1..F3.{pdf,png}` | both formats, as required |
| 07 | `07_artifact_manifest.py` | `$DATA`, `$OUT`, `$CODE` | `$OUT/ARTIFACT_MANIFEST.csv` | run last |

`run_when_x99_up.sh` is an operational helper (not part of the numbered pipeline):
it waits for X99's sshd to accept connections again and then re-runs `02` + `03`.
It exists only because X99 refused the key exchange during execution.

## Frozen code that is imported, never re-implemented

| what | module | git blob |
|---|---|---|
| KGE (independent, [T,B]) | `benchmark/analysis/chapter4_oob_pipeline_audit_20260926/scripts/minimal_kge.py` `kge_minimal` | `43a2bce5dc9be0de4a2f5d39e6a7920071e7e954` |
| KGE (pipeline) | `benchmark/src/objective.py` `streaming_kge` | `d2b0e17ad71e1e4ade3e318a8e93c77f8f8b0429` |
| KGE accumulator used for the frozen `foldMed`/`pooled` | `benchmark/scripts/oob/oob_common.py` `kge_numpy`, `kge_from_statistics`, `combine_statistics` | see `metric_sources.md` |
| logKGE + FHV (v2 freeze) | `benchmark/analysis/chapter4_final_closure_20260925/scripts/s2_supplementary_metrics_recommend.py` `compute_logkge_point`, `compute_fhv_score` | `256bea5604e33fcae0c66b630a48d5214725c291` |
| logKGE + FHV (rb2 definition) | `benchmark/analysis/chapter4_robustness_supplement_20260925/scripts/rb2_metrics.py` `per_year_stats`, `kge_from_sums` | `555df1e31b724b0c2aeeefc5ff80b997b98c8e02` |

Full provenance, including the two discrepancies between the task description and
the actual file locations, is in `$OUT/metric_sources.md`.

## Conventions

* basin IDs are `int64` internally and are joined on the integer value, never on row
  position; `gauge_id_str` is the 8-digit zero-padded export form (audit A4).
* bootstrap: resampling unit = basin, `B = 2000`, seed `20260930`, 95 % percentile CI.
* permutation: `N = 10000`, seed `20260930`.
* `numpy.std(ddof=1)`, `numpy.mean`; every table carries `n`.
* S2/S3 outputs carry the literal first line
  `EXPLORATORY — not a pre-registered test`.

## Determinism

Every script is re-runnable from scratch and produces identical outputs: all
randomness is seeded from `common.py`, and no network dependency exists apart from
X99 reads in `01`/`02`/`03`.
