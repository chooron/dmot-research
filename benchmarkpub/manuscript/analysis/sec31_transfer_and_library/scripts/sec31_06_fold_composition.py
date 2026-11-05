#!/usr/bin/env python3
"""sec31_06_fold_composition.py — Distribution of geographic and hydroclimatic attributes across the 5 outer folds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ATTRIBUTE_COLS = [
    "lat",
    "lon",
    "arid",
    "f_snow",
    "runoff_ratio",
    "p_mean",
    "pet_mean",
    "p_seas",
    "elev",
    "slope",
    "area",
    "f_forest",
    "zero_flow_frac",
]


def main() -> None:
    # 1. Load basin IDs and fold assignments
    fa_df = pd.read_csv(S.DIR_OOB36_DATA / "manifests" / "OOB_FOLD_ASSIGNMENT.csv")
    fa_df["gauge_id_str"] = fa_df["basin_id"].astype(str).str.zfill(8)

    # 2. Load coordinates
    loc_file = ROOT / "project/flexmopex/archive/20261003_cleanup/legacy_training/results/formal_671_unified_nmul4/lambda_invariance/inputs/gauge_locations_671.csv"
    loc_df = pd.read_csv(loc_file)
    loc_df["gauge_id_str"] = loc_df["gage_id"].astype(str).str.zfill(8)

    # 3. Load hydroclimatic attributes from basin_attribute_table.csv
    attr_file = S.DIR_OOB36_NEXT_STAGE / "tables" / "basin_attribute_table.csv"
    attr_df = pd.read_csv(attr_file)
    attr_df["gauge_id_str"] = attr_df["basin_id"].astype(str).str.zfill(8)

    # Merge into single table
    merged = fa_df.merge(loc_df[["gauge_id_str", "lat", "lon"]], on="gauge_id_str", how="inner")
    merged = merged.merge(attr_df[["gauge_id_str", "arid", "f_snow", "runoff_ratio", "p_mean", "pet_mean", "p_seas", "elev", "slope", "area", "f_forest", "zero_flow_frac"]], on="gauge_id_str", how="inner")

    assert len(merged) == 531, f"Expected 531 basins, got {len(merged)}"

    # 4. Compute distributions per fold and overall
    rows = []
    scopes = [("all_531", merged)] + [(f"fold_{k}", merged[merged.fold == k]) for k in range(5)]

    for scope_name, sub in scopes:
        n_b = len(sub)
        for col in ATTRIBUTE_COLS:
            vals = sub[col].to_numpy(dtype=float)
            rows.append({
                "scope": scope_name,
                "n_basins": n_b,
                "attribute": col,
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "mean": float(np.mean(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })

    out_df = pd.DataFrame(rows)
    S.write_tidy_table(out_df, TABLES / "sec31_fold_composition.csv")

    parity_rec = {
        "script": "sec31_06_fold_composition.py",
        "approach": "(a) Recomputed from OOB_FOLD_ASSIGNMENT.csv, gauge_locations_671.csv, and basin_attribute_table.csv",
        "n_basins_total": len(merged),
        "attributes_included": ATTRIBUTE_COLS,
        "fold_counts": [int((merged.fold == k).sum()) for k in range(5)],
    }
    (TABLES / "sec31_06_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec31_06] PASS: computed fold composition for {len(ATTRIBUTE_COLS)} attributes across 5 folds ({parity_rec['fold_counts']})")


if __name__ == "__main__":
    main()
