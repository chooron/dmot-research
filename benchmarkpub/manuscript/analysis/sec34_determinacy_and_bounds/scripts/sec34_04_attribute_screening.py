#!/usr/bin/env python3
"""sec34_04_attribute_screening.py — Format 15 basin attributes screening against OOB penalty with BH-FDR."""
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
    src_screen = S.DIR_OOB36_NEXT_STAGE / "tables" / "basin_attribute_screening.csv"
    upstream_meta = [S.file_metadata_record(src_screen)]

    df = pd.read_csv(src_screen)
    df_sorted = df.sort_values("bh_fdr_q").reset_index(drop=True)
    S.write_tidy_table(df_sorted, TABLES / "sec34_attribute_screening_bh_fdr.csv")

    parity_rec = {
        "script": "sec34_04_attribute_screening.py",
        "approach": "(b) Loaded frozen verified artifacts from pub_next_stage_20260930",
        "upstream_artifacts": upstream_meta,
        "n_variables": len(df),
        "min_q_value": float(df["bh_fdr_q"].min()),
        "n_surviving_fdr_005": int((df["bh_fdr_q"] < 0.05).sum()),
        "top_attribute": str(df_sorted.loc[0, "variable"]),
        "top_attribute_rho": float(df_sorted.loc[0, "spearman_rho"]),
    }
    (TABLES / "sec34_04_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec34_04] PASS: {len(df)} attributes screened, min q={parity_rec['min_q_value']:.4f}, surviving q<0.05={parity_rec['n_surviving_fdr_005']}")


if __name__ == "__main__":
    main()
