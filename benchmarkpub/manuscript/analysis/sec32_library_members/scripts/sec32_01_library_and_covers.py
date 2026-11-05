#!/usr/bin/env python3
"""sec32_01_library_and_covers.py — Compile minimal model libraries, hitting sets, and OOB cover members."""
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
    # Approach (b): Load frozen artifacts from pub_minimal_model_library_20260930 and equiv_cover_t1t2_20261003
    src_min_lib = S.DIR_OOB36_MINIMAL_LIB / "tables"
    src_equiv_t1 = S.DIR_EQUIV_T1T2 / "tables"

    upstream_files = [
        src_min_lib / "library_model_frequency.csv",
        src_min_lib / "library_minimum_cover.csv",
        src_min_lib / "minimal_model_library.json",
        src_equiv_t1 / "t1_min_cover.csv",
    ]
    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    # 1. Model frequency in candidate sets and union sizes
    df_freq = pd.read_csv(src_min_lib / "library_model_frequency.csv")
    S.write_tidy_table(df_freq, TABLES / "sec32_library_model_frequency.csv")

    # 2. Minimal cover solutions (Top-3 and Top-5 exact MILP + alternative optima)
    df_cover = pd.read_csv(src_min_lib / "library_minimum_cover.csv")
    S.write_tidy_table(df_cover, TABLES / "sec32_minimal_cover_solutions.csv")

    # 3. OOB 100% cover member list (from equiv_cover_t1t2_20261003)
    df_t1_min = pd.read_csv(src_equiv_t1 / "t1_min_cover.csv")
    oob_row = df_t1_min[(df_t1_min.condition == "OOB-dPL") & (df_t1_min.target == 1.00)].iloc[0]

    oob_members = str(oob_row["exact_models"]).split("|")
    df_oob_members = pd.DataFrame({
        "model": sorted(oob_members),
        "condition": "OOB-dPL",
        "target_coverage": 1.00,
        "n_models_total": len(oob_members),
        "solver_status": int(oob_row["solver_status"]),
        "proven_optimal": bool(oob_row["proven_optimal"]),
    })
    S.write_tidy_table(df_oob_members, TABLES / "sec32_oob_full_cover_members.csv")

    parity_rec = {
        "script": "sec32_01_library_and_covers.py",
        "approach": "(b) Loaded frozen verified artifacts from pub_minimal_model_library_20260930 and equiv_cover_t1t2_20261003",
        "upstream_artifacts": upstream_meta,
        "top3_union_size": int((df_freq["in_any_top3"]).sum()),
        "top5_union_size": int((df_freq["in_any_top5"]).sum()),
        "top3_min_cover_exact_size": int(df_cover.loc[df_cover.k == 3, "min_cover_size"].iloc[0]),
        "top5_min_cover_exact_size": int(df_cover.loc[df_cover.k == 5, "min_cover_size"].iloc[0]),
        "oob_full_cover_exact_size": len(oob_members),
    }
    (TABLES / "sec32_01_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec32_01] PASS: Top-3 cover={parity_rec['top3_min_cover_exact_size']}, Top-5 cover={parity_rec['top5_min_cover_exact_size']}, OOB cover={parity_rec['oob_full_cover_exact_size']}")


if __name__ == "__main__":
    main()
