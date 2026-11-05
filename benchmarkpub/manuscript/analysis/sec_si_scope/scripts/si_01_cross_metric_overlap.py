#!/usr/bin/env python3
"""si_01_cross_metric_overlap.py — Compile cross-metric candidate set overlap (evaluation-only, no retraining)."""
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
    src_overlap = S.DIR_C4_ROBUSTNESS / "tables" / "rb2_cross_metric_overlap.csv"
    src_supp = S.DIR_C4_FINAL_CLOSURE / "tables" / "s2_supp_metrics_summary.json"

    upstream_files = [src_overlap, src_supp]
    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    df_overlap = pd.read_csv(src_overlap)
    df_overlap["evaluation_scope"] = "EVALUATION_ONLY_NO_TARGET_MATCHED_RETRAINING"
    S.write_tidy_table(df_overlap, TABLES / "sec_si_cross_metric_overlap.csv")

    supp_json = json.loads(src_supp.read_text(encoding="utf-8"))
    supp_rows = []
    for met, res in supp_json["results"].items():
        supp_rows.append({
            "metric": met,
            "F3_models": "|".join(res["F3_models"]),
            "SBS_model": res["SBS_model"],
            "G_set": float(res["G_set"]),
            "G_set_lo95": float(res["G_set_lo95"]),
            "G_set_hi95": float(res["G_set_hi95"]),
            "verdict": res["verdict"],
            "beats_permutation": res["beats_permutation"],
            "seeds_G_set_mean": float(res["seeds_G_set_mean"]),
            "seeds_ci_positive": int(res["seeds_G_set_n_ci_positive"]),
        })
    df_supp = pd.DataFrame(supp_rows)
    S.write_tidy_table(df_supp, TABLES / "sec_si_supplementary_metric_recommendation.csv")

    parity_rec = {
        "script": "si_01_cross_metric_overlap.py",
        "approach": "(b) Loaded frozen verified artifacts from chapter4_robustness_supplement_20260925 and chapter4_final_closure_20260925",
        "upstream_artifacts": upstream_meta,
        "dpl_logkge_top3_overlap": float(df_overlap.loc[(df_overlap.paradigm == "dPL") & (df_overlap.metric == "logKGE"), "top3_overlap_with_KGE"].iloc[0]),
        "dpl_fhv_top3_overlap": float(df_overlap.loc[(df_overlap.paradigm == "dPL") & (df_overlap.metric == "FHV"), "top3_overlap_with_KGE"].iloc[0]),
        "ic_logkge_top3_overlap": float(df_overlap.loc[(df_overlap.paradigm == "IC") & (df_overlap.metric == "logKGE"), "top3_overlap_with_KGE"].iloc[0]),
        "ic_fhv_top3_overlap": float(df_overlap.loc[(df_overlap.paradigm == "IC") & (df_overlap.metric == "FHV"), "top3_overlap_with_KGE"].iloc[0]),
    }
    (TABLES / "si_01_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[si_01] PASS: dPL logKGE overlap={parity_rec['dpl_logkge_top3_overlap']:.4f}, FHV overlap={parity_rec['dpl_fhv_top3_overlap']:.4f}")


if __name__ == "__main__":
    main()
