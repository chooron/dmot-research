# dMoTpy 1.0.0rc1 — local release candidate

> **Status: not a public release.** This is an isolated source-package candidate prepared for review. It has no assigned open-source license, verified author/copyright metadata, public archive DOI, or publication authorization. Do not redistribute or upload it until the release blockers in [`docs/release_readiness.md`](docs/release_readiness.md) are cleared.

## Scope

dMoTpy provides PyTorch implementations of 36 conceptual rainfall–runoff model variants and shared process/flux helpers. The package supports differentiable daily simulation; the model equations, daily process ordering, smooth thresholds, parameter mappings, and selected routing implementations are not asserted to be numerically identical to MARRMoT or to every original paper formulation. See [`docs/differences_from_marrmot.md`](docs/differences_from_marrmot.md) and [`docs/model_reference.csv`](docs/model_reference.csv).

This candidate contains source, tests, documentation, one small example, and only the 12 pre-existing validation artifacts required by the included artifact-integrity tests. It excludes forcing/observation series, checkpoints, optimizer states, training outputs, bulk validation caches, Git metadata, MATLAB/Octave MARRMoT source, and unrelated projects.
One included Euler status CSV is a prior result fixture required by two legacy tests. It is copied verbatim from the development tree, not regenerated, and is not presented as scientific validation for this candidate; source generations remain separately documented.

## Installation (candidate only)

Requires Python 3.10 or later and PyTorch 2.9.1. From this directory:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Optional extras are declared in `pyproject.toml`: `analysis` supplies NumPy/SciPy/pandas, `training` supplies the umbrella project's `dmg` integration (its public distribution identity still needs confirmation), and `dev` supplies the test dependencies.

## Minimal forward/gradient example

```python
import torch
from dmotpy.models.hydrology_model import HydrologyModel

model = HydrologyModel(
    config={"model_name": "gr4j", "warm_up": 1,
            "parameter_mapping": "auto", "backend": "none"},
    device=torch.device("cpu"),
)
# [time, basin, channels], with channels ordered P, T, PET.
forcing = torch.tensor(
    [[[4.0, 8.0, 1.5]], [[0.0, 7.0, 1.0]], [[2.0, 9.0, 1.2]]],
    dtype=torch.float64,
)
raw_parameters = torch.full((1, 4), 0.5, dtype=torch.float64, requires_grad=True)
result = model({"x_phy": forcing}, (None, raw_parameters))
q = result["streamflow"]
q.sum().backward()
print(q.shape, raw_parameters.grad)
```

This is an API smoke example, not a scientific validation case or a reported experiment.

## Tests

Install the development extra, then run the complete included suite (do not omit the artifact-integrity tests):

```bash
python -m pip install '.[dev]'
PYTHONPATH=dmotpy python -m pytest dmotpy/tests -q
```

The `PYTHONPATH` is retained for legacy tests that import the `models` namespace directly. Test outcomes for this candidate are recorded outside the package in the accompanying release assessment.

## Paper and source identity

The GMD v1.0 experiment audit identifies the executed source snapshot by SHA-256 `57211c13310c29d8271b39369d85eeb6f5f2960add9eeb71a9259930cd7c5963`. This candidate is not that snapshot: GSFB, Plateau, HBV96, and MODHYDROLOG have post-freeze source differences with measured frozen-IC KGE impacts. Published results must remain attributed to their archived executed snapshot; this package must not be described as producing every published result. See [`docs/paper_reproduction.md`](docs/paper_reproduction.md).
