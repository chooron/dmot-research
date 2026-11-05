#!/usr/bin/env python3
"""17_next_stage_manifest.py — sha256 manifest of the next-stage analysis directory.

Also records a hard provenance block so the whole result set can be re-derived:
the git HEAD, the seed, the frozen inputs consumed, and the fact that no training ran.

Output: $OUT2/ARTIFACT_MANIFEST.csv and $OUT2/PROVENANCE.json
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
    rows = []
    for p in sorted(PC.OUT2.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(PC.OUT2)),
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    for name in ("11_pub_insample_seen.py", "pub_common.py", "12_winner_stability.py",
                 "13_basin_attribute_screening.py", "14_delta_column_audit.py",
                 "15_pub_stats.py", "16_next_stage_figures.py",
                 "17_next_stage_manifest.py"):
        f = SCRIPTS / name
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{name}",
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(PC.OUT2 / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-next-stage-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "git_head_subject": git("log", "-1", "--format=%s"),
        "seed": 20260930,
        "bootstrap_B": 2000,
        "omnibus_permutations": 2000,
        "no_training_executed": True,
        "no_parameter_update": True,
        "project_benchmark_modified": False,
        "frozen_inputs": {
            "merged_basin_table": str(PC.MERGED),
            "fold_assignment": "project/benchmark/results/oob_primary8_5fold_20260902/OOB_FOLD_ASSIGNMENT.csv",
            "oob_config": "project/benchmark/configs/oob_primary8_5fold_20260902.yaml",
            "checkpoints_existing8": "project/benchmark/results/oob_primary8_5fold_20260902/runs",
            "checkpoints_other28": str(PC.PUB / "data" / "oob36_20260930" / "runs"),
            "observed_series": str(PC.PUB / "data" / "oob36_20260930" / "obs"),
            "basin_attributes": "data/caravan_671_attributes.csv",
            "seed_baseline": "project/benchmark/results/dpl_canonical_v2_20260831/runs + "
                             "project/benchmark/manuscript/r3/dpl_multiseed_sensitivity_20260914/runs",
            "b4_validation_table": "project/benchmark/analysis/chapter4_oob_pipeline_audit_20260926/"
                                   "tables/B4_attribute_permutation.csv",
        },
        "validation": {
            "b4_cells_reproduced": "80/80",
            "b4_max_abs_diff": 5.551115123125783e-17,
            "heldout_reproduction_max_abs_diff": {"existing8": 6.938893903907228e-13,
                                                  "v4_new8": 1.3711254120683e-14,
                                                  "v5_new20": 2.2339768923629322e-14},
            "seed42_vs_frozen_kge_seen_dpl_max_abs_diff": 4.99e-11,
        },
        "reproduce": [
            "python scripts/oob36_prelim/11_pub_insample_seen.py --resume "
            "--out analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/"
            "tables/pub_insample_seen_kge.csv",
            "python scripts/oob36_prelim/15_pub_stats.py",
            "python scripts/oob36_prelim/12_winner_stability.py",
            "python scripts/oob36_prelim/13_basin_attribute_screening.py",
            "python scripts/oob36_prelim/14_delta_column_audit.py",
            "python scripts/oob36_prelim/16_next_stage_figures.py",
        ],
    }
    (PC.OUT2 / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {PC.OUT2/'ARTIFACT_MANIFEST.csv'}")
    print(f"provenance -> {PC.OUT2/'PROVENANCE.json'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
