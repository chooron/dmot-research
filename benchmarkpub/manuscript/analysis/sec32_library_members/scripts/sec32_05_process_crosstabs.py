#!/usr/bin/env python3
"""sec32_05_process_crosstabs.py — Process representation cross-tabulations of minimal library members vs full 36 library."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Load model attributes
    df_attr = pd.read_csv(TABLES / "sec32_model_structural_attributes.csv").set_index("model")
    models_36 = sorted(df_attr.index.tolist())

    # Load 21 OOB members
    df_oob = pd.read_csv(TABLES / "sec32_oob_full_cover_members.csv")
    models_oob21 = sorted(df_oob["model"].unique().tolist())

    # Load Top-3 (13) and Top-5 (9) minimal cover members
    df_min = pd.read_csv(TABLES / "sec32_minimal_cover_solutions.csv").set_index("k")
    models_top3_13 = sorted(df_min.loc[3, "min_cover_models"].split("|"))
    models_top5_9 = sorted(df_min.loc[5, "min_cover_models"].split("|"))

    subsets = {
        "full_library_36": models_36,
        "oob_100pct_cover_21": models_oob21,
        "seen_top3_min_cover_13": models_top3_13,
        "seen_top5_min_cover_9": models_top5_9,
    }

    # 1. Summary comparison across the 4 groups
    summary_rows = []
    for sname, mlist in subsets.items():
        sub = df_attr.loc[mlist]
        n_snow = int(sub["has_explicit_snow_routine"].sum())
        p = sub["n_parameters"].to_numpy(dtype=float)
        st = sub["n_states"].to_numpy(dtype=float)
        summary_rows.append({
            "subset_name": sname,
            "n_models": len(mlist),
            "n_with_snow_routine": n_snow,
            "pct_with_snow_routine": n_snow / len(mlist),
            "params_mean": float(p.mean()),
            "params_median": float(np.median(p)),
            "params_min": int(p.min()),
            "params_max": int(p.max()),
            "states_mean": float(st.mean()),
            "states_median": float(np.median(st)),
            "states_min": int(st.min()),
            "states_max": int(st.max()),
        })
    df_sum = pd.DataFrame(summary_rows)
    S.write_tidy_table(df_sum, TABLES / "sec32_process_representation_summary.csv")

    # 2. Parameter count distribution cross-tabulation
    all_param_counts = sorted(df_attr["n_parameters"].unique().tolist())
    p_rows = []
    for pc in all_param_counts:
        row = {"n_parameters": pc}
        for sname, mlist in subsets.items():
            sub = df_attr.loc[mlist]
            cnt = int((sub["n_parameters"] == pc).sum())
            pct = cnt / len(mlist)
            row[f"{sname}_count"] = cnt
            row[f"{sname}_pct"] = pct
        p_rows.append(row)
    df_p = pd.DataFrame(p_rows)
    S.write_tidy_table(df_p, TABLES / "sec32_process_crosstab_parameters.csv")

    # 3. Storage states count distribution cross-tabulation
    all_state_counts = sorted(df_attr["n_states"].unique().tolist())
    s_rows = []
    for sc in all_state_counts:
        row = {"n_states": sc}
        for sname, mlist in subsets.items():
            sub = df_attr.loc[mlist]
            cnt = int((sub["n_states"] == sc).sum())
            pct = cnt / len(mlist)
            row[f"{sname}_count"] = cnt
            row[f"{sname}_pct"] = pct
        s_rows.append(row)
    df_s = pd.DataFrame(s_rows)
    S.write_tidy_table(df_s, TABLES / "sec32_process_crosstab_states.csv")

    parity_rec = {
        "script": "sec32_05_process_crosstabs.py",
        "approach": "(a) Recomputed from sec32 tables",
        "n_subsets_evaluated": len(subsets),
        "full_36_snow_count": int(df_sum.loc[df_sum.subset_name == "full_library_36", "n_with_snow_routine"].iloc[0]),
        "oob21_snow_count": int(df_sum.loc[df_sum.subset_name == "oob_100pct_cover_21", "n_with_snow_routine"].iloc[0]),
        "top3_13_snow_count": int(df_sum.loc[df_sum.subset_name == "seen_top3_min_cover_13", "n_with_snow_routine"].iloc[0]),
        "top5_9_snow_count": int(df_sum.loc[df_sum.subset_name == "seen_top5_min_cover_9", "n_with_snow_routine"].iloc[0]),
    }
    (TABLES / "sec32_05_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec32_05] PASS: generated 3 cross-tabulations (snow counts: full36={parity_rec['full_36_snow_count']}, oob21={parity_rec['oob21_snow_count']}, top3_13={parity_rec['top3_13_snow_count']}, top5_9={parity_rec['top5_9_snow_count']})")


if __name__ == "__main__":
    import numpy as np
    main()
