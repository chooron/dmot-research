#!/usr/bin/env python3
"""24_candidate_set_manifest.py — sha256 manifest + provenance for the candidate-set analysis."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

REPO = PC.REPO
SCRIPTS = REPO / "project" / "benchmarkpub" / "scripts" / "oob36_prelim"
NAMES = ["22_candidate_set_transfer.py", "23_candidate_set_figures.py",
         "24_candidate_set_manifest.py", "pub_common.py"]


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
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    import pandas as pd
    rows = []
    for p in sorted(PC.OUT4.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(PC.OUT4)),
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    for n in NAMES:
        f = SCRIPTS / n
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{n}",
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(PC.OUT4 / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-candidate-set-transfer-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "seed": 20260930, "bootstrap_B": 2000,
        "k_grid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 36],
        "no_training_executed": True, "no_parameter_update": True,
        "no_new_inference": True,
        "project_benchmark_modified": False,
        "no_equivalence_set": True, "no_candidate_probability_model": True,
        "frozen_inputs": {
            "model_basin_vectors": str(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv"),
            "merged_basin_table": str(PC.MERGED),
            "frozen_supplemental_overlap": str(PC.OUT3 / "tables" / "basin_rank_stability.csv"),
            "frozen_winner_retention": str(PC.OUT2 / "tables" / "winner_stability.json"),
            "seed_runs": "benchmark/results/dpl_canonical_v2_20260831/runs + "
                         "benchmark/manuscript/r3/dpl_multiseed_sensitivity_20260914/runs",
        },
        "validation_vs_frozen": {
            "k1_coverage_equals_winner_retention": "0.2523540489642185 (exact)",
            "k3_jaccard_median_abs_diff": 0.0,
            "k5_jaccard_median_abs_diff": 5.551115123125783e-17,
            "k1_seed_gap_matches_next_stage": "0.2065, CI [0.1569, 0.2536]",
        },
        "reproduce": [f"python project/benchmarkpub/scripts/oob36_prelim/{n}" for n in NAMES[:3]],
    }
    (PC.OUT4 / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {PC.OUT4/'ARTIFACT_MANIFEST.csv'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
