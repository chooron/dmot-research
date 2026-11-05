#!/usr/bin/env python3
"""39_parsimony_complexity_manifest.py — sha256 manifest + provenance for the new directory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("/home/jingxin/code/dmg-research")
OUT = REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_parsimony_complexity_20260930"
SCRIPTS = REPO / "project/benchmarkpub/scripts/oob36_prelim"
NAMES = [
    "36_parsimony_complexity_extract.py",
    "37_parsimony_complexity_analysis.py",
    "38_parsimony_complexity_figures.py",
    "39_parsimony_complexity_manifest.py",
]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    import pandas as pd

    rows = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    for n in NAMES:
        f = SCRIPTS / n
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{n}", "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(OUT / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-parsimony-complexity-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "no_training": True,
        "no_hydrological_simulation": True,
        "no_kge_recomputation": True,
        "no_gpu": True,
        "project_benchmark_modified": False,
        "previous_parsimony_directory_modified": False,
        "primary_complexity_metric": "n_parameters",
        "secondary_complexity_metric": "n_states",
        "not_comparable": ["n_fluxes", "n_process_modules", "NUMBER_INFO"],
        "parameter_forward": {
            "network": "project/benchmark/dpl/nn_parameterizer.py::CatchmentParameterizer (legacy, hidden [256,256])",
            "built_by": "project/benchmark/scripts/oob/run_oob_job.py::_make_parameterizer",
            "checkpoints": "each run's own frozen best.pt (180 = 36 models x 5 folds)",
            "attributes": "data/caravan_671_attributes.npy (35 attributes, Caravan/531 basins)",
            "normalization": "each run's frozen attribute_mean/attribute_std/normalization_metadata.json",
            "space": "theta_norm = sigmoid(raw) in (0,1); theta_phys = HydrologyModel._change_param_range(theta_norm, bounds, 'auto', 100.0)",
            "same_forward_verified_elsewhere": "chapter4_oob36_20260926/scripts/s6_reinference.py recomputes Qsim from this forward and reproduces the frozen heldout_basin_kge.csv to <= 1e-8 (OOB36_REINFERENCE.csv)",
        },
        "run_dir_resolution": {
            "existing8": "project/benchmark/results/oob_primary8_5fold_20260902/runs (local; the download staging tree lacks normalization_metadata.json for this cohort)",
            "v4_new8_and_v5_new20": "project/benchmarkpub/data/oob36_20260930/runs",
        },
        "frozen_inputs": {
            "kge": "pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv",
            "adequacy": "pub_relaxed_adequacy_20260930/tables/relaxed_adequacy_model_basin.csv",
            "complexity": "pub_parsimony_20260930/tables/model_complexity_inventory.csv",
            "fold_assignment": "data/oob36_20260930/manifests/OOB_FOLD_ASSIGNMENT.csv",
        },
        "statistics": {
            "model_level": "Spearman rho + 10000-permutation two-sided p (seed 20260930) + Pearson (supplementary)",
            "basin_level": "pooled Spearman, within-model Spearman, model-median Spearman reported separately",
            "bands": "descriptive rank tertiles of n_parameters (12/12/12), no tuned cut-off",
        },
        "carried_forward_caveats": [
            "SEEN is the cross-pipeline canonical dPL store; same-pipeline 4-fold in-sample hydrographs would need new inference",
            "4 of the 10 adequacy signatures are convention-unverified (TOSSH absent)",
            "Q95 exact simulated zeros are counted as adequacy failures by design",
            "complexity is measured only as a parameter count; n_fluxes and n_process_modules are not comparable",
            "fold-to-fold parameter variability mixes the effect of different training basins with different train-fitted attribute normalization",
        ],
        "reproduce": [
            "python project/benchmarkpub/scripts/oob36_prelim/36_parsimony_complexity_extract.py",
            "python project/benchmarkpub/scripts/oob36_prelim/37_parsimony_complexity_analysis.py",
            "python project/benchmarkpub/scripts/oob36_prelim/38_parsimony_complexity_figures.py",
            "python project/benchmarkpub/scripts/oob36_prelim/39_parsimony_complexity_manifest.py",
        ],
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {OUT/'ARTIFACT_MANIFEST.csv'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
