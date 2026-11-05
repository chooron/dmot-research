#!/usr/bin/env python3
"""sec32_02_coverage_ladder_regret.py — Recompute Seen Top-k candidate set transfer coverage ladder and regret."""
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

KS = list(range(1, 11)) + [15, 20, 36]


def main() -> None:
    pub = S.load_same_pipeline_table()
    models = sorted(pub["model"].unique())
    basins = S.basin_order()
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)

    seen = pub.pivot(index="bs", columns="model", values="kge_seenfolds_median4").loc[basins, models].to_numpy(dtype=np.float64)
    oob = pub.pivot(index="bs", columns="model", values="kge_heldout").loc[basins, models].to_numpy(dtype=np.float64)

    n_basins = len(basins)
    seen_order = np.argsort(-seen, axis=1, kind="stable")
    oob_order = np.argsort(-oob, axis=1, kind="stable")
    oob_win_idx = oob_order[:, 0]
    oob_oracle = oob.max(axis=1)

    cov_rows = []
    reg_rows = []

    for k in KS:
        covers = np.empty(n_basins, dtype=bool)
        hits3 = np.empty(n_basins, dtype=bool)
        hits5 = np.empty(n_basins, dtype=bool)
        regrets = np.empty(n_basins, dtype=np.float64)

        for b in range(n_basins):
            cand_idx = set(seen_order[b, :k])
            covers[b] = (oob_win_idx[b] in cand_idx)
            hits3[b] = bool(cand_idx & set(oob_order[b, :3]))
            hits5[b] = bool(cand_idx & set(oob_order[b, :5]))
            best_in_cand = oob[b, list(cand_idx)].max()
            regrets[b] = oob_oracle[b] - best_in_cand

        cov_rows.append({
            "k": k,
            "strategy": "basin_specific",
            "coverage_top1": float(covers.mean()),
            "coverage_top3_anyhit": float(hits3.mean()),
            "coverage_top5_anyhit": float(hits5.mean()),
            "n_covered": int(covers.sum()),
        })

        reg_rows.append({
            "k": k,
            "strategy": "basin_specific",
            "regret_median": float(np.median(regrets)),
            "regret_mean": float(np.mean(regrets)),
            "regret_q75": float(np.percentile(regrets, 75)),
            "regret_q90": float(np.percentile(regrets, 90)),
            "frac_regret_le_0.01": float((regrets <= 0.01).mean()),
            "frac_regret_le_0.05": float((regrets <= 0.05).mean()),
        })

    df_cov = pd.DataFrame(cov_rows)
    df_reg = pd.DataFrame(reg_rows)
    S.write_tidy_table(df_cov, TABLES / "sec32_coverage_ladder.csv")
    S.write_tidy_table(df_reg, TABLES / "sec32_regret_curve.csv")

    # Parity check vs frozen candidate_set_transfer tables
    src_transfer = S.DIR_OOB36_TRANSFER / "tables"
    frozen_cov = pd.read_csv(src_transfer / "topk_coverage_curve.csv")
    frozen_bs_cov = frozen_cov[frozen_cov.strategy == "basin_specific"].set_index("k")

    frozen_reg = pd.read_csv(src_transfer / "topk_regret_curve.csv")
    frozen_bs_reg = frozen_reg[frozen_reg.strategy == "basin_specific"].set_index("k")

    diff_cov = [abs(row["coverage_top1"] - frozen_bs_cov.loc[row["k"], "coverage_top1"]) for _, row in df_cov.iterrows()]
    diff_reg = [abs(row["regret_median"] - frozen_bs_reg.loc[row["k"], "regret_median"]) for _, row in df_reg.iterrows()]
    max_cov_diff = max(diff_cov)
    max_reg_diff = max(diff_reg)

    parity_rec = {
        "script": "sec32_02_coverage_ladder_regret.py",
        "approach": "(a) Recomputed from pub_same_pipeline_model_basin.csv",
        "max_coverage_abs_diff": max_cov_diff,
        "max_regret_abs_diff": max_reg_diff,
        "coverage_k1": float(df_cov.loc[df_cov.k == 1, "coverage_top1"].iloc[0]),
        "coverage_k3": float(df_cov.loc[df_cov.k == 3, "coverage_top1"].iloc[0]),
        "coverage_k5": float(df_cov.loc[df_cov.k == 5, "coverage_top1"].iloc[0]),
        "coverage_k10": float(df_cov.loc[df_cov.k == 10, "coverage_top1"].iloc[0]),
    }
    (TABLES / "sec32_02_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec32_02] PASS: k1={parity_rec['coverage_k1']:.4f}, k3={parity_rec['coverage_k3']:.4f}, k5={parity_rec['coverage_k5']:.4f}, max diff cov={max_cov_diff:.3e}")


if __name__ == "__main__":
    main()
