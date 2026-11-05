#!/usr/bin/env python3
"""sec31_04_equivalence_and_coverage.py — Format equivalence sets, size distributions, and coverage curves across IC, seen-dPL, and OOB-dPL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Approach (b): Load frozen verified artifacts from equiv_cover_t1t2_20261003
    src_dir = S.DIR_EQUIV_T1T2 / "tables"
    upstream_files = [
        src_dir / "t1_size_distribution.csv",
        src_dir / "t1_equivalence_sets.csv",
        src_dir / "t1_coverage_curve.csv",
        src_dir / "t1_greedy_curve.csv",
        src_dir / "t1_min_cover.csv",
        src_dir / "t1_model_coverage.csv",
    ]

    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    # 1. Size distribution
    df_size = pd.read_csv(src_dir / "t1_size_distribution.csv")
    S.write_tidy_table(df_size, TABLES / "sec31_equivalence_size_distribution.csv")

    # 2. Per-basin equivalence sets
    df_sets = pd.read_csv(src_dir / "t1_equivalence_sets.csv")
    S.write_tidy_table(df_sets, TABLES / "sec31_equivalence_sets_per_basin.csv")

    # 3. Exact coverage curves
    df_curve_exact = pd.read_csv(src_dir / "t1_coverage_curve.csv")
    S.write_tidy_table(df_curve_exact, TABLES / "sec31_coverage_curves_exact.csv")

    # 4. Greedy coverage curves
    df_curve_greedy = pd.read_csv(src_dir / "t1_greedy_curve.csv")
    S.write_tidy_table(df_curve_greedy, TABLES / "sec31_coverage_curves_greedy.csv")

    # 5. Minimal cover targets (90%, 95%, 100%)
    df_min = pd.read_csv(src_dir / "t1_min_cover.csv")
    S.write_tidy_table(df_min, TABLES / "sec31_minimal_cover_targets.csv")

    # 6. Model equivalence coverage
    df_cov = pd.read_csv(src_dir / "t1_model_coverage.csv")
    S.write_tidy_table(df_cov, TABLES / "sec31_model_equivalence_coverage.csv")

    parity_rec = {
        "script": "sec31_04_equivalence_and_coverage.py",
        "approach": "(b) Loaded frozen verified artifacts from equiv_cover_t1t2_20261003",
        "upstream_artifacts": upstream_meta,
        "n_conditions": df_size["condition"].nunique(),
        "targets_evaluated": df_min["target"].tolist(),
        "cover95_exact": dict(zip(df_min[df_min.target == 0.95]["condition"], df_min[df_min.target == 0.95]["n_models_exact"])),
        "cover100_exact": dict(zip(df_min[df_min.target == 1.00]["condition"], df_min[df_min.target == 1.00]["n_models_exact"])),
    }
    (TABLES / "sec31_04_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec31_04] PASS: formatted equivalence and coverage curves across 3 conditions ({len(upstream_files)} upstream files recorded)")


if __name__ == "__main__":
    main()
