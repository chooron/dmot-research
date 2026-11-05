# Asset Lineage and Provenance Ledger

This document records the provenance, migration mapping, exclusion rationale, data dependencies, and licensing status for the independent repository `/home/jingxin/code/dmot-research`.

---

## 1. Upstream Source Baseline

- **Upstream Repository**: `/home/jingxin/code/dmg-research`
- **Source Branch**: `revision-2026-freeze`
- **Source Commit**: `23461ec129fd081b1a7c6b9015a2098ab42aef5c`
- **Execution Date**: 2026-11-05
- **Constraint**: `/home/jingxin/code/dmg-research` remained strictly read-only throughout migration. Zero files were added, modified, or deleted in the upstream tree.

---

## 2. Component Migration Mapping

### 2.1 `dmotpy` Core Library
- **Source**: `/home/jingxin/code/dmg-research/project/benchmarkdev/tmp/dmotpy_zenodo_candidate_20261102/dmotpy-1.0.0rc1/`
- **Destination**: `/home/jingxin/code/dmot-research/dmotpy`
- **Included Content**:
  - `dmotpy/models/core/`: 36 conceptual rainfall-runoff models in PyTorch. All 36 model module hashes match the final closeout matrix `MODEL_36_FREEZE_MATRIX.csv` (including HBV96 and MODHYDROLOG zero-floor boundary fixes).
  - `dmotpy/models/flux/`: Shared physical process fluxes (evaporation, snowmelt, baseflow, recharge, saturation).
  - `dmotpy/models/unithydro/`: Differentiable causal unit hydrographs (half, full, triangular, gamma, exponential, uniform, delay).
  - `dmotpy/neural_networks/`: Catchment parameterizer networks for differentiable parameter learning (dPL).
  - `dmotpy/trainers/`: Training loops with finite transaction guards.
  - `dmotpy/losses.py`: Differentiable KGE and penalty loss implementations.
  - `dmotpy/data_contract.py`: Forcing layout, unit conversions, and calendar invariants.
  - `dmotpy/tests/`: 30 test modules covering water balance, gradient stability, autodiff gradcheck, and unit hydrographs.
  - `docs/`: In-depth documentation on differences from MARRMoT, numerical implementations, paper reproduction identities, and release readiness.
  - `examples/`: Minimal forward/autograd smoke example (`quickstart.py`).
  - Packaging files: `pyproject.toml`, `MANIFEST.in`, `RIGHTS_REVIEW.md`, `CITATION_DRAFT.md`.
- **Excluded**:
  - Full historical `dmotpy/validation_results/` tree (54 directories) except the 12 fixtures required by unit tests.
  - Legacy unregistered experimental models (`lascam.py`, `sacramento.py`).

### 2.2 `benchmarkparam` Benchmark Infrastructure
- **Source**: `/home/jingxin/code/dmg-research/project/benchmark/`
- **Destination**: `/home/jingxin/code/dmot-research/benchmarkparam`
- **Included Content**:
  - `src/` (17 files): `batched_cmaes.py`, `checkpoint_guard.py`, `checkpointing.py`, `compile_utils.py`, `convergence.py`, `data_selection.py`, `evotorch_reference.py`, `gpu_batch_planner.py`, `model_adapter.py`, `model_registry.py`, `objective.py`, `parameter_transform.py`, `production_config.py`, `reporting.py`, `solver_coordinator.py`, `streaming_evaluator.py`, `__init__.py`.
  - `dpl/` (13 files): `attributes.py`, `build_caravan_671_matrix.py`, `compare_attributes.py`, `nn_parameterizer.py`, `optimizer_transaction.py`, `train_dpl.py`, and test suite.
  - `configs/` (4 files): `default.yaml`, `full_run_10starts_300gen_dpl_aligned_1980_1995.yaml`, `full_run_10starts_300gen_warm1980_1981x5.yaml`, `oob_primary8_5fold_20260902.yaml`.
  - `scripts/` (8 formal entrypoints):
    - `run_36model_benchmark.py`: Unified CMA-ES optimizer for 36 models.
    - `run_dpl_benchmark_dmg_native.py`: Native PyTorch dPL trainer.
    - `run_dpl_benchmark_20ep.py`: 20-epoch dPL calibration runner.
    - `evaluate_benchmark_metrics.py`: Independent calibration evaluation.
    - `evaluate_ic_aligned_gen300.py`: Generation 300 test-period evaluator.
    - `validate_full300_config.py`: Full300 configuration validator.
    - `run_parallel_models.py`: Parallel model launcher.
    - `consolidate_final_checkpoints.py`: Atomic checkpoint consolidator.
- **Excluded**:
  - `checkpoints/` and `results/`: Large binary training weights and intermediate logs.
  - `scripts/diagnostics/` and `scripts/ablation/`: Exploratory, one-off diagnostic scripts.

### 2.3 `benchmarkpub` Publication & Manuscript Analyses
- **Source**: `/home/jingxin/code/dmg-research/project/benchmarkpub/`
- **Destination**: `/home/jingxin/code/dmot-research/benchmarkpub`
- **Included Content**:
  - `basin_split.py`, `specialization.py`, `__init__.py`, `tests/`.
  - `manuscript/analysis/_freeze/`: Data freeze ledger `freeze_record_20261005.json` (311 tracked files) and `FROZEN.md`.
  - `manuscript/analysis/`: All 70 frozen analytical modules:
    - `sec31_transfer_and_library/`: Out-of-basin transfer and library coverage.
    - `sec32_library_members/`: Optimal subset enumeration and bottleneck analyses.
    - `sec33_usage_and_conditioning/`: Usage rules, cluster typologies, regret bounds.
    - `sec34_determinacy_and_bounds/`: Complexity vs. instability, multi-seed dispersion.
    - `sec35_cluster_stratified/`: Hydro-climatic cluster stratified penalties.
    - `sec36_size_matched_null/`: Null model statistical comparisons.
    - `sec_si_scope/`: Supplementary information scope and cross-metric checks.
    - `_verification/`: Abstract numbers independent recomputation and provenance.
    - `_manifest/`: Master manifest and citable numbers compilation.
    - `_results_registry/`: Results registry declarations, evidence packets, and isolated reproduction.
    - `_shared/`: Shared data loaders, basin accessors, and path helpers.
    - `verify_reproducibility.py`: Automated bit-for-bit output reproduction tester.
  - `manuscript/scripts/`: Figure generation scripts (Figures 1–6, `hydrofig.py`, `rebuild_root_figures.py`, `rebuild_v6_figures.py`, `figures_v6/`).
- **Excluded**:
  - Unfrozen raw simulation outputs, ad-hoc draft figure iterations.

### 2.4 `benchmarkdev` Formal Verification Suite
- **Source**: `/home/jingxin/code/dmg-research/project/benchmarkdev/`
- **Destination**: `/home/jingxin/code/dmot-research/benchmarkdev`
- **Included Content**:
  - `src/benchmarkdev/`:
    - `balance.py`: Water balance conservation observers with UH causal convolution.
    - `core.py`: Public HydrologyModel adapters, parameter readers, read-only protection hooks.
    - `open_balance.py`: Open boundary observers for models with deep percolation sinks.
    - `substeps.py`: Explicit subcycling time-scale scaling tests.
  - `config/config.json`: Configuration defining numerical tolerances and test parameters.
  - `tests/`: `test_contracts.py` (36-model parity and loss identity), `test_substeps_contract.py`.
  - `INVENTORY.md`, `README.md`.
- **Excluded**:
  - `analyses/`: 46 exploratory research scripts.
  - `_worktrees/`, `cache/`, `tmp/`: Disposable workspace directories.

---

## 3. Data Dependencies & Symbolic Links

To conserve storage and prevent accidental modification of source datasets, large or canonical external data files are linked via symbolic links from `/home/jingxin/code/dmg-research/`:

| Symbolic Link Path | Target Source Path | Description | Size |
|---|---|---|---|
| `data/531sub_id.txt` | `dmg-research/data/531sub_id.txt` | 531 CAMELS basin ID inventory | 4.8 KB |
| `data/gage_id.npy` | `dmg-research/data/gage_id.npy` | Reference gauge IDs array | 5.5 KB |
| `data/camels_dataset` | `dmg-research/data/camels_dataset` | CAMELS daily forcing and flow pickle | 266.8 MB |
| `data/caravan_671_attributes.npy` | `dmg-research/data/caravan_671_attributes.npy` | Caravan 35 physical attributes array | 188.0 KB |
| `data/caravan_671_attributes.csv` | `dmg-research/data/caravan_671_attributes.csv` | Caravan catchment attributes CSV | 180.2 KB |
| `data/basin_groups/` | `dmg-research/data/basin_groups/` | 17 hydro-climatic region partition arrays | 12.9 KB |
| `data/gauge_cluster.csv` | `dmg-research/data/gauge_cluster.csv` | Gauge cluster classifications | 39.6 KB |
| `benchmarkpub/data/oob36_20260930` | `dmg-research/project/benchmarkpub/data/oob36_20260930` | Out-of-basin Qsim and obs data | 2.07 GB |
| `benchmarkpub/manuscript/cache/assets` | `dmg-research/project/benchmarkpub/manuscript/cache/assets` | Cartographic basemaps, relief, and fonts | 2.4 MB |
| `benchmarkpub/manuscript/tables/figures/gauge_coordinates_531.csv` | `dmg-research/project/benchmarkpub/manuscript/tables/figures/gauge_coordinates_531.csv` | 531 basin coordinates for map plotting | 15.1 KB |

---

## 4. Rights, Licensing, and Release Status

1. **MARRMoT Licensing (GPL-3.0)**:
   - Upstream MARRMoT (Modular Assessment of Rainfall-Runoff Models Toolbox; Knoben et al., 2019) is licensed under the GNU General Public License v3.0 (GPL-3.0).
   - `dmotpy` implements translated PyTorch formulations of 36 model structures. Legal attribution and license compatibility review must be completed before any public redistribution.
2. **dMoTpy Authorship & Copyright**:
   - `RIGHTS_REVIEW.md` and `CITATION_DRAFT.md` document that authorship, copyright ownership, and formal public release authorizations are unresolved.
3. **Public Release Disposition**:
   - **NO-GO for public Zenodo deposit / open-source release.**
   - This repository is designated exclusively for internal research, verification, and reproducible manuscript artifact compilation.
