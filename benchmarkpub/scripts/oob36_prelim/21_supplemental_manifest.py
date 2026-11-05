#!/usr/bin/env python3
"""21_supplemental_manifest.py — sha256 manifest + provenance for the supplemental analysis.

Output: $OUT3/ARTIFACT_MANIFEST.csv and $OUT3/PROVENANCE.json
"""

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
NAMES = ["18_basin_rank_stability.py", "19_extreme_basin_diagnostics.py",
         "20_supplemental_figures.py", "21_supplemental_manifest.py", "pub_common.py"]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    import pandas as pd
    rows = []
    for p in sorted(PC.OUT3.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(PC.OUT3)),
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    for n in NAMES:
        f = SCRIPTS / n
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{n}",
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(PC.OUT3 / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-supplemental-rank-extreme-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "seed": 20260930, "bootstrap_B": 2000, "extreme_quantile": 0.95,
        "no_training_executed": True, "no_parameter_update": True,
        "no_new_inference": True,          # both 36-model vectors were already frozen
        "project_benchmark_modified": False,
        "frozen_inputs": {
            "model_basin_vectors": str(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv"),
            "basin_penalty_table": str(PC.OUT2 / "tables" / "basin_attribute_table.csv"),
            "basin_winner_table": str(PC.OUT2 / "tables" / "winner_margin_summary.csv"),
            "merged_basin_table": str(PC.MERGED),
            "observed_series": str(PC.PUB / "data" / "oob36_20260930" / "obs"),
            "kge_conventions_source": "project/benchmark/scripts/oob/oob_common.py",
        },
        "project_conventions_read": {"KGE_EPS": 0.1, "min_observation_std": 0.01,
                                     "min_valid_points": 30, "kge_numpy_min_count": 2},
        "reproduce": [f"python project/benchmarkpub/scripts/oob36_prelim/{n}" for n in NAMES[:4]],
    }
    (PC.OUT3 / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {PC.OUT3/'ARTIFACT_MANIFEST.csv'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
