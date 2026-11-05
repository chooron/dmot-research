#!/usr/bin/env python3
"""32_relaxed_adequacy_manifest.py — sha256 manifest + provenance for the relaxed-adequacy run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

SCRIPTS = PC.REPO / "project" / "benchmarkpub" / "scripts" / "oob36_prelim"
NAMES = ["30_relaxed_adequacy.py", "31_relaxed_adequacy_figures.py",
         "32_relaxed_adequacy_manifest.py", "pub_common.py"]


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
    for p in sorted(PC.OUT7.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(PC.OUT7)),
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    for n in NAMES:
        f = SCRIPTS / n
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{n}",
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(PC.OUT7 / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-relaxed-adequacy-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "no_model_execution": True, "no_inference": True, "no_torch": True,
        "no_gpu": True, "no_training": True, "kge_recomputed": False,
        "project_benchmark_modified": False,
        "prior_frozen_dir_modified": False,
        "definitions_pre_registered": ["STRICT_REFERENCE", "RELAXED_60", "RELAXED_70",
                                       "RELAXED_80", "GROUP_3OF5", "GROUP_4OF5"],
        "robustness_variants": ["SIG6_ALLPASS", "RELAXED_70_verified6",
                                "GROUP_3OF5_literal", "GROUP_3OF5_verified6"],
        "threshold_primary": 50.0, "threshold_sensitivity": [25.0, 50.0, 75.0],
        "recommended_definition": "RELAXED_70",
        "grouping_primary": "hydrologically correct (q95=low flow, q5=high flow)",
        "Q5_Q95_naming_collision": ("task text swaps Q5/Q95 relative to this repo's frozen "
                                    "v3_common definition; both groupings computed, the correct "
                                    "one is primary and the literal one is GROUP_3OF5_literal"),
        "frozen_inputs": {
            "adequacy_fold_level_long.parquet": str(
                PC.OUT1 / "pub_adequacy_seen_oob_20260930/tables/adequacy_fold_level_long.parquet"),
            "candidate_set_adequacy_by_basin.csv": str(
                PC.OUT1 / "pub_adequacy_seen_oob_20260930/tables/candidate_set_adequacy_by_basin.csv"),
            "pub_same_pipeline_model_basin.csv": str(
                PC.OUT2 / "tables/pub_same_pipeline_model_basin.csv"),
        },
        "carried_forward_caveats": [
            "SEEN is cross-pipeline (canonical dPL store); same-pipeline 4-fold in-sample "
            "hydrographs do not exist without new inference",
            "4 of the 10 signatures (LF/HF freq/dur) are convention-unverified (TOSSH absent); "
            "RELAXED_70_verified6 shows the recommendation is robust to excluding them",
            "Q95 exact simulated zeros are counted as failures and keep the frozen unfloored "
            "convention deliberately",
            "all levels are threshold-dependent; the ordering is not",
        ],
        "reproduce": [f"python project/benchmarkpub/scripts/oob36_prelim/{n}" for n in NAMES[:2]],
    }
    (PC.OUT7 / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {PC.OUT7/'ARTIFACT_MANIFEST.csv'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
