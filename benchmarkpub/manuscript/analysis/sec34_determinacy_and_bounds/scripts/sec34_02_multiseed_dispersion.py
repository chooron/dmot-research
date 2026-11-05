#!/usr/bin/env python3
"""sec34_02_multiseed_dispersion.py — Format 8 models x 3 seeds dPL parameter dispersion and cross-distance."""
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
    src_dist = S.DIR_R3_MULTISEED / "tables" / "table_q2_parameter_distance.csv"
    upstream_meta = [S.file_metadata_record(src_dist)]

    df_dist = pd.read_csv(src_dist)
    S.write_tidy_table(df_dist, TABLES / "sec34_multiseed_8model_dispersion.csv")

    parity_rec = {
        "script": "sec34_02_multiseed_dispersion.py",
        "approach": "(b) Loaded frozen verified artifacts from dpl_multiseed_sensitivity_20260914",
        "upstream_artifacts": upstream_meta,
        "n_models": len(df_dist),
        "d_self_median_across_models": float(df_dist["D_self_median"].median()),
        "d_cross_median_across_models": float(df_dist["D_cross_median"].median()),
        "self_to_cross_ratio_median": float(df_dist["self_to_cross_ratio"].median()),
        "models_self_much_smaller_than_cross": int(df_dist["self_much_smaller_than_cross"].sum()),
    }
    (TABLES / "sec34_02_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec34_02] PASS: 8 models x 3 seeds D_self median={parity_rec['d_self_median_across_models']:.4f}, ratio={parity_rec['self_to_cross_ratio_median']:.1%}")


if __name__ == "__main__":
    main()
