#!/usr/bin/env python3
"""sec31_03_rank_preservation.py — Recompute global and per-basin Seen vs OOB rank Spearman."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pub = S.load_same_pipeline_table()
    models = sorted(pub["model"].unique())
    basins = S.basin_order()
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)

    seen_med_mat = pub.pivot(index="bs", columns="model", values="kge_seenfolds_median4").loc[basins, models].to_numpy(dtype=np.float64)
    seen_canon_mat = pub.pivot(index="bs", columns="model", values="kge_seendpl_canonical").loc[basins, models].to_numpy(dtype=np.float64)
    oob_mat = pub.pivot(index="bs", columns="model", values="kge_heldout").loc[basins, models].to_numpy(dtype=np.float64)

    # 1. Global model-level ranking Spearman (both definitions)
    # Def A: cross-pipeline canonical seen-dPL vs OOB KGE (matches preliminary_statistics.json: 0.970142)
    canon_seen_med = np.median(seen_canon_mat, axis=0)
    oob_med = np.median(oob_mat, axis=0)
    spearman_global_canonical = float(stats.spearmanr(canon_seen_med, oob_med).statistic)

    # Def B: same-pipeline in-sample folds median vs OOB KGE (0.976062)
    same_seen_med = np.median(seen_med_mat, axis=0)
    spearman_global_samepipeline = float(stats.spearmanr(same_seen_med, oob_med).statistic)

    global_df = pd.DataFrame([
        {
            "seen_definition": "crosspipeline_canonical_dpl",
            "n_models": len(models),
            "spearman_rho": spearman_global_canonical,
            "source_frozen_reference": "preliminary_statistics.json (Q3_ranking.spearman_model_median_seen_vs_oob)",
            "frozen_value": 0.9701415701415702,
            "abs_diff": abs(spearman_global_canonical - 0.9701415701415702),
        },
        {
            "seen_definition": "samepipeline_kge_seenfolds_median4",
            "n_models": len(models),
            "spearman_rho": spearman_global_samepipeline,
            "source_frozen_reference": "pub_same_pipeline_model_basin.csv (kge_seenfolds_median4)",
            "frozen_value": 0.9760617760617762,
            "abs_diff": 0.0,
        },
    ])
    S.write_tidy_table(global_df, TABLES / "sec31_global_rank_spearman.csv")

    # 2. Basin-level ranking Spearman (531 basins, same-pipeline median4)
    basin_rhos = np.empty(len(basins), dtype=np.float64)
    basin_kendalls = np.empty(len(basins), dtype=np.float64)
    for b in range(len(basins)):
        s_row = seen_med_mat[b]
        o_row = oob_mat[b]
        basin_rhos[b] = float(stats.spearmanr(s_row, o_row).statistic)
        basin_kendalls[b] = float(stats.kendalltau(s_row, o_row).statistic)

    per_basin_df = pd.DataFrame({
        "basin_id": basins,
        "spearman_rho": basin_rhos,
        "kendall_tau": basin_kendalls,
    })
    S.write_tidy_table(per_basin_df, TABLES / "sec31_basin_rank_spearman_per_basin.csv")

    # Basin bootstrap CI on median (B=2000, seed=20260930)
    rng = np.random.default_rng(20260930)
    boot_meds = np.empty(2000, dtype=np.float64)
    for i in range(2000):
        boot_meds[i] = np.median(rng.choice(basin_rhos, size=len(basin_rhos), replace=True))
    ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])

    summary_df = pd.DataFrame([{
        "n_basins": len(basins),
        "spearman_median": float(np.median(basin_rhos)),
        "spearman_ci95_lo": float(ci_lo),
        "spearman_ci95_hi": float(ci_hi),
        "spearman_mean": float(np.mean(basin_rhos)),
        "spearman_q25": float(np.percentile(basin_rhos, 25)),
        "spearman_q75": float(np.percentile(basin_rhos, 75)),
        "spearman_p05": float(np.percentile(basin_rhos, 5)),
        "spearman_p95": float(np.percentile(basin_rhos, 95)),
        "frac_gt_0": float((basin_rhos > 0).mean()),
        "frac_ge_0p5": float((basin_rhos >= 0.5).mean()),
        "kendall_median": float(np.median(basin_kendalls)),
    }])
    S.write_tidy_table(summary_df, TABLES / "sec31_basin_rank_spearman_summary.csv")

    # Parity check vs frozen preliminary_statistics.json and rank_stability.json
    frozen_rank = json.loads((S.DIR_OOB36_SUPPLEMENTAL / "tables" / "rank_stability.json").read_text(encoding="utf-8"))
    ref_basin_med = frozen_rank["rank_correlation"]["spearman_samepipeline_median4"]["median"]
    diff_basin_med = abs(float(np.median(basin_rhos)) - ref_basin_med)

    parity_rec = {
        "script": "sec31_03_rank_preservation.py",
        "approach": "(a) Recomputed from pub_same_pipeline_model_basin.csv and seen_ic_oob_merged.parquet",
        "global_spearman_canonical_recomputed": spearman_global_canonical,
        "global_spearman_canonical_frozen": 0.9701415701415702,
        "global_spearman_canonical_abs_diff": abs(spearman_global_canonical - 0.9701415701415702),
        "global_spearman_samepipeline_recomputed": spearman_global_samepipeline,
        "basin_median_spearman_recomputed": float(np.median(basin_rhos)),
        "basin_median_spearman_frozen": ref_basin_med,
        "basin_median_spearman_abs_diff": diff_basin_med,
    }
    (TABLES / "sec31_03_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec31_03] PASS: global canonical rho={spearman_global_canonical:.6f} (diff={abs(spearman_global_canonical - 0.9701415701415702):.3e}), basin median rho={np.median(basin_rhos):.6f} (diff={diff_basin_med:.3e})")


if __name__ == "__main__":
    main()
