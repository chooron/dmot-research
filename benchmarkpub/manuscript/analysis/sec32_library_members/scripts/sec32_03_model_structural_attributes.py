#!/usr/bin/env python3
"""sec32_03_model_structural_attributes.py — Compile MARRMoT model structural attributes."""
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
    src_ci = S.DIR_OOB36_PARSIMONY / "tables" / "model_complexity_inventory.csv"
    src_snow = S.DIR_GAP_CONDITIONAL / "tables" / "model_snow_annotation.csv"

    ci = pd.read_csv(src_ci)
    snow = pd.read_csv(src_snow)

    merged = ci.merge(snow[["model", "snow_module", "evidence_source", "evidence_import"]], on="model", how="left")
    merged = merged.rename(columns={
        "snow_module": "has_explicit_snow_routine",
    })

    out_df = merged[[
        "model",
        "n_parameters",
        "n_states",
        "complexity_band",
        "has_explicit_snow_routine",
        "evidence_source",
        "evidence_import",
    ]].sort_values("model").reset_index(drop=True)

    S.write_tidy_table(out_df, TABLES / "sec32_model_structural_attributes.csv")

    parity_rec = {
        "script": "sec32_03_model_structural_attributes.py",
        "approach": "(a) Compiled from model_complexity_inventory.csv and model_snow_annotation.csv",
        "n_models": len(out_df),
        "n_models_with_snow": int(out_df["has_explicit_snow_routine"].sum()),
        "min_params": int(out_df["n_parameters"].min()),
        "max_params": int(out_df["n_parameters"].max()),
        "median_params": float(out_df["n_parameters"].median()),
        "min_states": int(out_df["n_states"].min()),
        "max_states": int(out_df["n_states"].max()),
    }
    (TABLES / "sec32_03_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec32_03] PASS: compiled {len(out_df)} models ({parity_rec['n_models_with_snow']} with snow routine)")


if __name__ == "__main__":
    main()
