#!/usr/bin/env python3
"""35_parsimony_manifest.py — sha256 manifest + provenance for the parsimony stage."""

from __future__ import annotations

import hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

SCRIPTS = PC.REPO / "project" / "benchmarkpub" / "scripts" / "oob36_prelim"
NAMES = ["33_parsimony.py", "34_parsimony_figures.py", "35_parsimony_manifest.py", "pub_common.py"]


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
        return subprocess.run(["git", *a], cwd=PC.REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    import pandas as pd
    rows = []
    for p in sorted(PC.OUT8.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(PC.OUT8)),
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    for n in NAMES:
        f = SCRIPTS / n
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{n}",
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(PC.OUT8 / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-parsimony-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "no_training": True, "no_inference": True, "no_torch": True, "no_gpu": True,
        "kge_recomputed": False, "project_benchmark_modified": False,
        "primary_parsimony_metric": "n_parameters",
        "metric_name_for_paper": "parameter-count parsimony (NOT universal structural simplicity)",
        "secondary_metric": "n_states",
        "not_comparable": {
            "n_fluxes": "no flux registry in dmotpy (FLUX_INFO absent)",
            "n_process_modules": "no module registry (MODULE_INFO absent); *_step_pre/_post exists "
                                 "for only 4 of 36 models",
            "NUMBER_INFO": "MARRMoT reference model number, not a count (frozen v3_common.py:21)",
        },
        "pipeline": "Accuracy Seen Top-5 -> Adequacy RELAXED_70 -> Parsimony simplest adequate",
        "tie_break": "min n_parameters, then SEEN KGE desc (rule-available); OOB KGE only brackets",
        "no_oob_truth_in_selection": True,
        "frozen_inputs": {
            "complexity": str(PC.BENCH / "analysis/chapter4_multicriteria_v3_20260925/tables/v1_parsimony.csv"),
            "kge": str(PC.OUT2 / "tables/pub_same_pipeline_model_basin.csv"),
            "relaxed_adequacy": str(PC.OUT1 / "pub_relaxed_adequacy_20260930/tables/relaxed_adequacy_model_basin.csv"),
            "frozen_candidate_sets": str(PC.OUT1 / "pub_adequacy_seen_oob_20260930/tables/candidate_set_adequacy_by_basin.csv"),
        },
        "carried_forward_caveats": [
            "SEEN is cross-pipeline (canonical dPL store); same-pipeline 4-fold in-sample "
            "hydrographs would need new inference",
            "4 of the 10 adequacy signatures are convention-unverified (TOSSH absent)",
            "Q95 exact simulated zeros are counted as adequacy failures by design",
            "all adequacy levels are threshold-dependent; only the ordering is stable",
        ],
        "reproduce": [f"python project/benchmarkpub/scripts/oob36_prelim/{n}" for n in NAMES[:2]],
    }
    (PC.OUT8 / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {PC.OUT8/'ARTIFACT_MANIFEST.csv'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
