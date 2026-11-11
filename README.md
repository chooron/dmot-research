# dMoT Research: Differentiable Conceptual Hydrological Modeling

`dmot-research` is a standalone scientific research codebase consolidating **dMoT** (differentiable Modeling of Topography / differentiable rainfall-runoff modeling in PyTorch), Python implementations of **36 MARRMoT** conceptual model variants, and formal benchmark/publication pipelines.

---

## Repository Structure

```
dmot-research/
├── dmotpy/              # Core differentiable hydrological model library
│   ├── models/          # 36 hydrological model variants + UH routing
│   │   ├── core/        # Core model ODE/difference formulations
│   │   ├── flux/        # Process flux equations (evap, melt, etc.)
│   │   └── unithydro/   # Unit hydrograph convolution kernels
│   ├── neural_networks/ # Parameter learning networks (dPL parameterizers)
│   ├── trainers/        # PyTorch trainers with finite transaction guards
│   ├── losses.py        # Differentiable KGE and custom hydrologic losses
│   └── data_contract.py # Forcing/target schema, calendars, and invariants
├── benchmarkparam/     # Large-scale benchmarking pipelines
│   ├── src/             # CMA-ES optimizer, streaming evaluator, batch planner
│   ├── dpl/             # Differentiable parameter learning infrastructure
│   ├── configs/         # Production YAML configurations (Full300, dPL)
│   └── scripts/         # Formal execution entrypoints
├── benchmarkpub/        # Publication analysis & reproducibility
│   ├── manuscript/
│   │   ├── analysis/    # 70 frozen analysis scripts (sec31–sec36, verification)
│   │   │   └── _freeze/ # Data freeze record (20261005) and SHA-256 manifests
│   │   └── scripts/     # Figure generation scripts (Fig. 1–6, Hydrofig)
│   ├── basin_split.py   # Spatial cross-validation / OOB fold partitioning
│   └── specialization.py# Hydrological model specialization metrics
├── benchmarkdev/        # Formal verification & contract testing
│   ├── src/benchmarkdev/# Water balance observers, substep scaling engine
│   ├── config/          # Verification tolerances, seeds, evaluation windows
│   └── tests/           # 36-model parity and conservation contract tests
├── results/             # IC, dPL, OOB formal evaluation metrics & registry
├── releases/            # Publication figures and release metadata
├── data/                # Symbolic links to read-only benchmark datasets
├── docs/                # Mathematical specifications, MARRMoT differences, audits
├── examples/            # Minimal forward/autograd smoke examples
├── pyproject.toml       # Package build specification (dmotpy)
├── PROVENANCE.md        # Complete asset lineage and source mapping
├── RIGHTS_REVIEW.md     # Intellectual property & licensing review
└── CITATION_DRAFT.md    # Draft citation metadata
```

---

## 36 Hydrological Models

`dmotpy` implements 36 conceptual rainfall-runoff models in PyTorch with full autograd support:

| Family | Models |
|---|---|
| Alpine | `alpine1` (4p), `alpine2` (6p) |
| Australian / Collie | `australia` (8p), `collie1` (1p), `collie2` (4p), `collie3` (6p) |
| FLEX | `flexb` (9p), `flexi` (10p), `flexis` (12p) |
| Standard Benchmarks | `gr4j` (4p), `gsfb` (8p), `hbv96` (15p), `hillslope` (7p), `hymod` (5p), `ihacres` (6p) |
| Large / Multi-Store | `modhydrolog` (15p), `plateau` (8p), `simhyd` (7p), `smar` (8p), `tank` (12p) |
| MOPEX | `mopex1` (5p), `mopex2` (7p), `mopex3` (8p), `mopex4` (10p), `mopex5` (12p) |
| Regional Variants | `newzealand1` (6p), `newzealand2` (8p), `penman` (4p), `susannah1` (6p), `susannah2` (6p), `tcm` (6p), `topmodel` (7p), `us1` (5p), `vic` (10p), `wetland` (4p), `xinanjiang` (12p) |

---

## Installation & Environment

### Prerequisites
- Python >= 3.10
- PyTorch >= 2.0 (tested on PyTorch 2.9.1)
- NumPy, SciPy, pandas, matplotlib, pyyaml

```bash
# In the dmot-research repository root:
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Verification Tests
```bash
# Run dmotpy unit tests:
python -m pytest dmotpy/tests -q
# Run benchmark contracts and conservation tests:
python -m pytest benchmarkparam/tests -q
python -m pytest benchmarkdev/tests -q

# Run benchmarkpub test suite:
python -m pytest benchmarkpub/tests -q
```

---

## Provenance & Attribution

- **Lineage**: Code assets are curated from `/home/jingxin/code/dmg-research`. Detailed source paths and SHA-256 references are recorded in `PROVENANCE.md`.
- **MARRMoT Comparison**: Model equations derive from the Modular Assessment of Rainfall-Runoff Models Toolbox (MARRMoT, Knoben et al., GPL-3.0). Differentiable approximations and explicit sequential differences are documented in `docs/differences_from_marrmot.md`.
- **Datasets**: CAMELS (Newman et al.) and Caravan (Kratzert et al.) basin attributes and forcings are linked via `data/` and remain read-only.
