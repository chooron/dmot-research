#!/usr/bin/env python3
"""29_adequacy_manifest.py — sha256 manifest + provenance for the adequacy analysis."""

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
NAMES = ["26_adequacy_signatures.py", "27_adequacy_analyses.py",
         "28_adequacy_figures.py", "29_adequacy_manifest.py", "pub_common.py"]


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
    for p in sorted(PC.OUT6.rglob("*")):
        if not p.is_file() or p.name in ("ARTIFACT_MANIFEST.csv", "PROVENANCE.json"):
            continue
        rows.append({"relative_path": str(p.relative_to(PC.OUT6)),
                     "bytes": p.stat().st_size, "sha256": sha256(p)})
    for n in NAMES:
        f = SCRIPTS / n
        if f.exists():
            rows.append({"relative_path": f"scripts/oob36_prelim/{n}",
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(PC.OUT6 / "ARTIFACT_MANIFEST.csv", index=False)

    prov = {
        "task_id": "ch4-oob36-hydrological-adequacy-20260930",
        "git_head": git("rev-parse", "HEAD"),
        "reference": "Spieler, D. & Schuetze, N. (2024) WRR 60(7) e2023WR036199 "
                     "(Spieler-style adequacy analysis, NOT an exact replication)",
        "no_training": True, "no_inference": True, "no_torch": True,
        "kge_recomputed": False, "project_benchmark_modified": False,
        "scored_window": ["1995-10-01", "2010-09-30"], "n_days": 5479,
        "water_year_convention": "project convention: 15 WYs 1996..2010 inside the test window",
        "pct_threshold_primary": 50.0, "sensitivity": [25.0, 50.0, 75.0],
        "mhfd_error_metric": "circular wrapped day-fraction distance; threshold 0.25 (~91 days)",
        "undef_abs_eps": 1e-12,
        "verifiable_signatures": ["rc(RR)", "bfi", "fdc_slope", "q5", "q95", "mhfd"],
        "convention_unverified_signatures": ["lf_freq", "lf_freq", "lf_dur", "hf_dur"],
        "TOSSH_CONVENTION_VERIFIED": False,
        "signature_definitions": {
            "RR": "sum(Q)/sum(P) over valid days (v3_common 'rc')",
            "BFI": "Lyne-Hollick, alpha=0.925, 3 alternating passes (v3_common)",
            "slopeFDC": "(ln Q33 - ln Q66)/(0.66-0.33), i.e. the 33rd-66th percentile mid-segment",
            "Q5": "flow exceeded 5% of days", "Q95": "flow exceeded 95% of days",
            "MHFD": "circular mean over water years of the half-flow day fraction (v3_common)",
            "LF/HF freq & dur": "paper thresholds 0.2*mean(Q) / 9*median(Q); event convention "
                                "is repo-local and NOT TOSSH-verified",
        },
        "known_limitations": [
            "SEEN is the canonical dPL store (cross-pipeline); the same-pipeline 4-fold "
            "in-sample daily hydrographs are NOT_AVAILABLE without new inference",
            "LFfreq/HFfreq/LFdur/HFdur event convention cannot be verified: TOSSH absent",
            "the frozen v3 module floors flow quantiles at 1e-3*mean(obs); the primary "
            "percentage errors here are UNFLOORED, with the floored variant reported alongside",
        ],
        "frozen_inputs": {
            "seen_daily_q": str(PC.Q_DP_STORE / "qsim.zarr"),
            "ic_daily_q_and_obs": str(PC.Q_IC_STORE),
            "oob_daily_q": str(PC.DATA_STORE / "qsim"),
            "precip": "data/camels_forcing_v2.pkl via v3_common.load_precip",
            "signature_module": "project/benchmark/analysis/chapter4_multicriteria_v3_"
                                "20260925/scripts/v3_common.py (imported, not modified)",
            "kge_partners": str(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv"),
        },
        "reproduce": [f"python project/benchmarkpub/scripts/oob36_prelim/{n}" for n in NAMES[:3]],
    }
    (PC.OUT6 / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} files -> {PC.OUT6/'ARTIFACT_MANIFEST.csv'}")
    print(f"git HEAD: {prov['git_head']}")


if __name__ == "__main__":
    main()
