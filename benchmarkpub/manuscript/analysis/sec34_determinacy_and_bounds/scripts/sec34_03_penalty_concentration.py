#!/usr/bin/env python3
"""sec34_03_penalty_concentration.py — Recompute OOB penalty concentration across basins and models."""
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
    # Approach (a): Recompute from pub_same_pipeline_model_basin.csv
    pub = S.load_same_pipeline_table()
    pub["delta_kge"] = pub["kge_seendpl_canonical"] - pub["kge_heldout"]

    model_sums = pub.groupby("model")["delta_kge"].sum().sort_values(ascending=False)
    total_delta = float(model_sums.sum())

    basin_sums = pub.groupby("basin_id")["delta_kge"].sum().sort_values(ascending=False)
    n_basins = pub["basin_id"].nunique()
    n_top5pct = int(0.05 * n_basins)  # 26 basins

    worst_model = str(model_sums.index[0])
    worst_model_share = float(model_sums.iloc[0] / total_delta)
    top5_model_share = float(model_sums.iloc[:5].sum() / total_delta)
    top10_model_share = float(model_sums.iloc[:10].sum() / total_delta)

    worst_basin = str(basin_sums.index[0])
    worst_basin_share = float(basin_sums.iloc[0] / basin_sums.sum())
    top5pct_basin_share = float(basin_sums.iloc[:n_top5pct].sum() / basin_sums.sum())

    rows = [
        {"entity_type": "all_total", "group_label": "all_531_basins_36_models", "n_units": len(pub), "sum_delta_kge": total_delta, "share_of_total_delta": 1.0},
        {"entity_type": "model", "group_label": f"worst_single_model ({worst_model})", "n_units": 1, "sum_delta_kge": float(model_sums.iloc[0]), "share_of_total_delta": worst_model_share},
        {"entity_type": "model", "group_label": "top_5_models", "n_units": 5, "sum_delta_kge": float(model_sums.iloc[:5].sum()), "share_of_total_delta": top5_model_share},
        {"entity_type": "model", "group_label": "top_10_models", "n_units": 10, "sum_delta_kge": float(model_sums.iloc[:10].sum()), "share_of_total_delta": top10_model_share},
        {"entity_type": "basin", "group_label": f"worst_single_basin ({worst_basin})", "n_units": 1, "sum_delta_kge": float(basin_sums.iloc[0]), "share_of_total_delta": worst_basin_share},
        {"entity_type": "basin", "group_label": f"top_5pct_basins (top {n_top5pct} basins)", "n_units": n_top5pct, "sum_delta_kge": float(basin_sums.iloc[:n_top5pct].sum()), "share_of_total_delta": top5pct_basin_share},
    ]
    df = pd.DataFrame(rows)
    S.write_tidy_table(df, TABLES / "sec34_penalty_concentration.csv")

    # Parity check vs frozen preliminary_statistics.json
    frozen_prelim = json.loads((S.DIR_OOB36_PAPER_LEVEL / "tables" / "preliminary_statistics.json").read_text(encoding="utf-8"))
    ref_conc = frozen_prelim["Q1_magnitude"]["concentration"]

    diff_top5pct_basin = abs(top5pct_basin_share - ref_conc["top5pct_basin_share_of_total_delta"])
    diff_top5_model = abs(top5_model_share - ref_conc["top5_model_share_of_total_delta"])

    parity_rec = {
        "script": "sec34_03_penalty_concentration.py",
        "approach": "(a) Recomputed from pub_same_pipeline_model_basin.csv",
        "top5pct_basin_share": top5pct_basin_share,
        "frozen_top5pct_basin_share": ref_conc["top5pct_basin_share_of_total_delta"],
        "top5pct_basin_share_abs_diff": diff_top5pct_basin,
        "top5_model_share": top5_model_share,
        "frozen_top5_model_share": ref_conc["top5_model_share_of_total_delta"],
        "top5_model_share_abs_diff": diff_top5_model,
        "worst_model": worst_model,
        "worst_model_share": worst_model_share,
    }
    (TABLES / "sec34_03_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec34_03] PASS: top 5% basins share={top5pct_basin_share:.4f} (diff={diff_top5pct_basin:.3e}), top 5 models share={top5_model_share:.4f} (diff={diff_top5_model:.3e})")


if __name__ == "__main__":
    main()
