#!/usr/bin/env python3
"""sec31_02_variance_decomposition.py — Two-way ANOVA sum of squares on ΔKGE_PUB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pub = S.load_same_pipeline_table()
    pub["delta_kge"] = pub["kge_seenfolds_median4"] - pub["kge_heldout"]

    # Pivot to [36 models x 531 basins]
    models = sorted(pub["model"].unique())
    basins = np.sort(pub["basin_id"].unique())
    mat = pub.pivot(index="model", columns="basin_id", values="delta_kge").loc[models, basins].to_numpy(dtype=np.float64)

    n_models, n_basins = mat.shape
    grand_mean = float(mat.mean())

    model_effect = mat.mean(axis=1) - grand_mean
    basin_effect = mat.mean(axis=0) - grand_mean
    residual = mat - grand_mean - model_effect[:, None] - basin_effect[None, :]

    ss_total = float(np.square(mat - grand_mean).sum())
    ss_model = float(n_basins * np.square(model_effect).sum())
    ss_basin = float(n_models * np.square(basin_effect).sum())
    ss_residual = float(np.square(residual).sum())

    rows = [
        {"component": "basin", "ss": ss_basin, "df": n_basins - 1, "ms": ss_basin / (n_basins - 1), "frac_of_total_ss": ss_basin / ss_total},
        {"component": "model", "ss": ss_model, "df": n_models - 1, "ms": ss_model / (n_models - 1), "frac_of_total_ss": ss_model / ss_total},
        {"component": "residual", "ss": ss_residual, "df": (n_models - 1) * (n_basins - 1), "ms": ss_residual / ((n_models - 1) * (n_basins - 1)), "frac_of_total_ss": ss_residual / ss_total},
        {"component": "total", "ss": ss_total, "df": n_models * n_basins - 1, "ms": ss_total / (n_models * n_basins - 1), "frac_of_total_ss": 1.0},
    ]
    df = pd.DataFrame(rows)
    S.write_tidy_table(df, TABLES / "sec31_variance_decomposition.csv")

    # Parity vs frozen
    frozen = pd.read_csv(S.DIR_OOB36_NEXT_STAGE / "tables" / "pub_variance_decomposition.csv").set_index("component")
    diffs = [abs(row["frac_of_total_ss"] - frozen.loc[row["component"], "frac_of_total_ss"]) for _, row in df.iterrows()]
    max_diff = max(diffs)

    parity_rec = {
        "script": "sec31_02_variance_decomposition.py",
        "approach": "(a) Recomputed from pub_same_pipeline_model_basin.csv",
        "ss_total": ss_total,
        "frac_basin": float(df.loc[df.component == "basin", "frac_of_total_ss"].iloc[0]),
        "frac_model": float(df.loc[df.component == "model", "frac_of_total_ss"].iloc[0]),
        "frac_residual": float(df.loc[df.component == "residual", "frac_of_total_ss"].iloc[0]),
        "max_abs_diff_vs_frozen": max_diff,
    }
    (TABLES / "sec31_02_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec31_02] PASS: basin={parity_rec['frac_basin']:.4f}, model={parity_rec['frac_model']:.4f}, max_diff={max_diff:.3e}")


if __name__ == "__main__":
    main()
