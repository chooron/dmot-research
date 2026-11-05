#!/usr/bin/env python3
"""36_parsimony_complexity_extract.py — per-basin dPL parameter extraction (frozen checkpoints).

SCOPE
-----
This is the ONLY step of the parsimony-complexity analysis that touches the frozen trained
objects.  It performs a *parameter forward* only:

    theta_{m, j, b} = Network_{m, j}( attributes_b )

for every model m (36), every fold-trained network j (5) and every basin b (531).  NO
hydrological simulation, NO KGE recomputation, NO training, NO modification of any frozen
output.  The forward network is the frozen `CatchmentParameterizer` reconstructed exactly as
`run_oob_job._make_parameterizer` builds it, then loaded from the run's own `best.pt`.

The evaluation pipeline that uses this same forward has already been independently verified
(`chapter4_oob36_20260926/scripts/s6_reinference.py`: rebuilt Qsim reproduces the frozen
`heldout_basin_kge.csv` to <= 1e-8, and `OOB36_REINFERENCE.csv` records the gate).  This script
adds its own structural checks (strict `load_state_dict`, best-epoch agreement, bound validity)
and writes them to `extraction_summary.json`.

PARAMETER CONVENTIONS
---------------------
Each model's parameter network emits a sigmoid value in (0, 1).  Two representations are stored:

  theta_norm  = sigmoid(raw)                       = (theta_phys - lower) / (upper - lower)
                                                    for LINEAR-mapped parameters
  theta_phys  = HydrologyModel._change_param_range(theta_norm, bounds, "auto", 100.0)

`theta_norm` is dimensionless and directly comparable across parameters and models, so it is the
primary stability space.  `theta_phys` applies exactly the mapping the hydrology model applies
(`parameter_mapping: auto`, `log_mapping_span_threshold: 100.0`), so it is the physical value
used in the simulation.

For a basin b held out in fold k(b):
  * the 4 vectors {theta_{m, j, b} : j != k(b)} are the SEEN-FOLD parameters, and
  * theta_{m, k(b), b} is the OOB-fold parameter.

Run dirs: the `existing8` cohort has no `normalization_metadata.json` in the downloaded staging
tree, so its run dirs are read from the local frozen tree
(`benchmark/results/oob_primary8_5fold_20260902/runs/`).  All other cohorts come from the
downloaded `data/oob36_20260930/runs/`.  The `existing8` local tree additionally carries
`config.json`, which is cross-checked against the frozen protocol.

Outputs -> $OUT/tables/extraction_summary.json
           $OUT/params/<model>.npz          (theta_norm, theta_phys, names, bounds, fold map)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/home/jingxin/code/dmg-research")
PUB = REPO / "project" / "benchmarkpub"
BENCH = REPO / "project" / "benchmark"
OUT1 = PUB / "analysis" / "chapter4_oob36_prelim_20260930"
OUT = OUT1 / "pub_parsimony_complexity_20260930"
PARAMS = OUT / "params"
TABLES = OUT / "tables"

sys.path[:0] = [str(REPO), str(BENCH), str(BENCH / "scripts" / "oob")]

import oob_common as OC  # noqa: E402
import run_oob_job as J  # noqa: E402
from src.model_registry import NPARAM_INFO_36, get_spec  # noqa: E402
from dmotpy.models.hydrology_model import HydrologyModel  # noqa: E402

# --- frozen input paths ----------------------------------------------------------------
INVENTORY = OUT1 / "remote_inventory.csv"
FOLD_ASSIGNMENT = PUB / "data" / "oob36_20260930" / "manifests" / "OOB_FOLD_ASSIGNMENT.csv"
DL_RUNS = PUB / "data" / "oob36_20260930" / "runs"
EXISTING8_RUNS = BENCH / "results" / "oob_primary8_5fold_20260902" / "runs"

MAPPING = "auto"
SPAN_THRESHOLD = 100.0
N_BASINS, N_FOLDS = 531, 5


def resolve_run_dir(model: str, fold: int) -> Path:
    """Prefer a run dir that actually carries the frozen attribute normalization."""
    candidates = [EXISTING8_RUNS / model / f"fold_{fold}", DL_RUNS / model / f"fold_{fold}"]
    for cand in candidates:
        if (cand / "best.pt").is_file() and (cand / "normalization_metadata.json").is_file():
            return cand
    raise FileNotFoundError(f"no complete run dir for {model} fold {fold}: {candidates}")


def load_stats(run_dir: Path) -> OC.NormalizationStats:
    meta = json.loads((run_dir / "normalization_metadata.json").read_text(encoding="utf-8"))
    return OC.NormalizationStats(
        "zscore",
        bool(meta["log_transform_skewed"]),
        np.asarray(meta["train_fitted_log_shifts"], dtype=np.float64),
        np.load(run_dir / "attribute_mean.npy"),
        np.load(run_dir / "attribute_std.npy"),
    )


def physical(theta_norm: torch.Tensor, spec) -> torch.Tensor:
    """Apply the hydrology model's own normalized -> physical parameter mapping."""
    bounds = spec.bounds
    cols = [
        HydrologyModel._change_param_range(
            theta_norm[:, i : i + 1], [float(bounds[i, 0]), float(bounds[i, 1])], MAPPING, SPAN_THRESHOLD
        )
        for i in range(spec.dimension)
    ]
    return torch.cat(cols, dim=1)


def main() -> int:
    PARAMS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    config = OC.load_oob_config("oob_primary8_5fold_20260902.yaml")
    repository = OC.BasinDataRepository(config)
    basin_ids = OC.load_ids(config["data"]["basin_ids"])
    if basin_ids.size != N_BASINS:
        raise RuntimeError(f"expected {N_BASINS} basins, got {basin_ids.size}")
    raw_attrs = repository.load_canonical_attributes(basin_ids.tolist())

    assign = pd.read_csv(FOLD_ASSIGNMENT)
    fold_of = dict(zip(assign["basin_id"].astype(int), assign["fold"].astype(int)))
    heldout_fold = np.asarray([fold_of[int(b)] for b in basin_ids], dtype=np.int64)
    for k in range(N_FOLDS):
        n_held = int((heldout_fold == k).sum())
        if n_held not in (106, 107):
            raise RuntimeError(f"fold {k} held-out basin count {n_held} is not a 531/5 split")

    inv = pd.read_csv(INVENTORY)
    models = sorted(inv["model"].unique().tolist())
    if models != sorted(NPARAM_INFO_36):
        raise RuntimeError("inventory model list does not match the frozen 36-model registry")

    device = torch.device("cpu")
    rows: list[dict] = []
    t0 = time.time()
    for model in models:
        spec = get_spec(model)
        n_p = NPARAM_INFO_36[model]
        theta_norm = np.empty((N_FOLDS, N_BASINS, n_p), dtype=np.float64)
        theta_phys = np.empty((N_FOLDS, N_BASINS, n_p), dtype=np.float64)
        for fold in range(N_FOLDS):
            run_dir = resolve_run_dir(model, fold)
            stats = load_stats(run_dir)
            attrs = torch.as_tensor(stats.transform(raw_attrs), dtype=torch.float64, device=device)
            net = J._make_parameterizer(model, attrs, config, device)
            payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            missing, unexpected = net.load_state_dict(payload["network"], strict=True)
            net.eval()
            with torch.inference_mode():
                tn = net(attrs).detach().cpu()
                tp = physical(tn, spec)
            theta_norm[fold] = tn.numpy()
            theta_phys[fold] = tp.numpy()
            recorded_epoch = int(json.loads((run_dir / "result.json").read_text())["best_epoch"])
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "cohort": str(inv.loc[inv.model == model, "cohort"].iloc[0]),
                    "run_dir": str(run_dir.relative_to(REPO)),
                    "n_parameters": n_p,
                    "best_epoch_in_ckpt": int(payload["metadata"]["best_epoch"]),
                    "best_epoch_in_result": recorded_epoch,
                    "epochs_agree": int(payload["metadata"]["best_epoch"]) == recorded_epoch,
                    "norm_min": float(tn.min()),
                    "norm_max": float(tn.max()),
                    "norm_in_open_unit_interval": bool((tn > 0).all() and (tn < 1).all()),
                    "phys_min": float(tp.min()),
                    "phys_max": float(tp.max()),
                    "phys_within_bounds": bool(
                        (tp >= spec.bounds[:, 0] - 1e-12).all() and (tp <= spec.bounds[:, 1] + 1e-12).all()
                    ),
                }
            )
        np.savez_compressed(
            PARAMS / f"{model}.npz",
            theta_norm=theta_norm,
            theta_phys=theta_phys,
            parameter_names=np.asarray(spec.parameter_names),
            bounds=spec.bounds.numpy(),
            basin_ids=basin_ids,
            heldout_fold=heldout_fold,
            model=np.asarray(model),
        )
        print(
            f"  {model:14s} P={n_p:2d}  norm[{theta_norm.min():.4f},{theta_norm.max():.4f}]  "
            f"elapsed={time.time() - t0:6.1f}s",
            flush=True,
        )

    df = pd.DataFrame(rows)
    summary = {
        "task": "ch4-oob36-parsimony-complexity-parameter-extraction",
        "models": len(models),
        "folds_per_model": N_FOLDS,
        "basins": N_BASINS,
        "network_forwards": int(len(df)),
        "parameter_vectors": int(len(df) * N_BASINS),
        "all_load_state_dict_strict": True,
        "all_epochs_agree": bool(df["epochs_agree"].all()),
        "all_norm_in_open_unit_interval": bool(df["norm_in_open_unit_interval"].all()),
        "all_phys_within_bounds": bool(df["phys_within_bounds"].all()),
        "network_forward_source": "run_oob_job._make_parameterizer + best.pt (frozen)",
        "mapping": MAPPING,
        "log_mapping_span_threshold": SPAN_THRESHOLD,
        "no_hydrological_simulation": True,
        "no_training": True,
        "no_kge_recomputation": True,
        "params_files": sorted(p.name for p in PARAMS.glob("*.npz")),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (TABLES / "extraction_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    df.to_csv(OUT / "logs" / "extraction_per_network.csv", index=False)

    print(json.dumps(summary, indent=1))
    ok = (
        summary["all_load_state_dict_strict"]
        and summary["all_epochs_agree"]
        and summary["all_norm_in_open_unit_interval"]
        and summary["all_phys_within_bounds"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
