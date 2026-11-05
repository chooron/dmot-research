#!/usr/bin/env python3
"""sec33_04_gset_and_regret.py — Compile G_set across Seen/A2d/A1b, paired regret differences, and candidate-set transfer tables."""
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
    gate_dir = S.DIR_OOB36_GATE / "tables"
    regret_dir = S.DIR_OOB36_PAIRED_REGRET / "tables"
    transfer_dir = S.DIR_OOB36_TRANSFER / "tables"

    upstream_files = [
        gate_dir / "gset_36_summary.json",
        gate_dir / "gset_36_A2d_per_seed.csv",
        gate_dir / "gset_36_A1b_per_seed.csv",
        gate_dir / "parity_p1_seen36_per_seed.csv",
        regret_dir / "paired_regret_summary.csv",
        regret_dir / "paired_regret_per_seed.csv",
        transfer_dir / "basin_specific_vs_global_topk.csv",
    ]
    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    # 1. G_set summary (Seen 36, A2d, A1b)
    gate_summary = json.loads((gate_dir / "gset_36_summary.json").read_text(encoding="utf-8"))
    p1_seeds = pd.read_csv(gate_dir / "parity_p1_seen36_per_seed.csv")

    seen_main = p1_seeds[p1_seeds.seed == S.SEED_MAIN].iloc[0]
    gset_rows = [
        {
            "estimand": "Seen-dPL (t2 baseline)",
            "role": "REFERENCE_CEILING",
            "k": 3,
            "n_basins": 531,
            "main_seed": S.SEED_MAIN,
            "G_set": float(seen_main["G_set"]),
            "lo95": float(seen_main["lo95"]),
            "hi95": float(seen_main["hi95"]),
            "mean_best_R": 0.744067,
            "mean_best_F": 0.739723,
            "mean_VBS": 0.761816,
            "numerator": 0.744067 - 0.739723,
            "denominator": 0.761816 - 0.739723,
            "G_set_min": float(p1_seeds["G_set"].min()),
            "G_set_max": float(p1_seeds["G_set"].max()),
            "G_set_median": float(p1_seeds["G_set"].median()),
            "G_set_mean": float(p1_seeds["G_set"].mean()),
            "n_seeds_positive": int((p1_seeds["G_set"] > 0).sum()),
            "n_seeds_ci_positive": int((p1_seeds["lo95"] > 0).sum()),
        },
        {
            "estimand": "A2d (Seen-trained, OOB-scored)",
            "role": "PRIMARY",
            "k": 3,
            "n_basins": gate_summary["A2d"]["n_basins"],
            "main_seed": S.SEED_MAIN,
            "G_set": gate_summary["A2d"]["G_set"],
            "lo95": gate_summary["A2d"]["lo95"],
            "hi95": gate_summary["A2d"]["hi95"],
            "mean_best_R": gate_summary["A2d"]["mean_best_R"],
            "mean_best_F": gate_summary["A2d"]["mean_best_F"],
            "mean_VBS": gate_summary["A2d"]["mean_VBS"],
            "numerator": gate_summary["A2d"]["numerator"],
            "denominator": gate_summary["A2d"]["denominator"],
            "G_set_min": gate_summary["A2d"]["G_set_min"],
            "G_set_max": gate_summary["A2d"]["G_set_max"],
            "G_set_median": gate_summary["A2d"]["G_set_median"],
            "G_set_mean": gate_summary["A2d"]["G_set_mean"],
            "n_seeds_positive": gate_summary["A2d"]["n_seeds_positive"],
            "n_seeds_ci_positive": gate_summary["A2d"]["n_seeds_ci_positive"],
        },
        {
            "estimand": "A1b (OOB-trained, OOB-scored)",
            "role": "SECONDARY_SENSITIVITY",
            "k": 3,
            "n_basins": gate_summary["A1b"]["n_basins"],
            "main_seed": S.SEED_MAIN,
            "G_set": gate_summary["A1b"]["G_set"],
            "lo95": gate_summary["A1b"]["lo95"],
            "hi95": gate_summary["A1b"]["hi95"],
            "mean_best_R": gate_summary["A1b"]["mean_best_R"],
            "mean_best_F": gate_summary["A1b"]["mean_best_F"],
            "mean_VBS": gate_summary["A1b"]["mean_VBS"],
            "numerator": gate_summary["A1b"]["numerator"],
            "denominator": gate_summary["A1b"]["denominator"],
            "G_set_min": gate_summary["A1b"]["G_set_min"],
            "G_set_max": gate_summary["A1b"]["G_set_max"],
            "G_set_median": gate_summary["A1b"]["G_set_median"],
            "G_set_mean": gate_summary["A1b"]["G_set_mean"],
            "n_seeds_positive": gate_summary["A1b"]["n_seeds_positive"],
            "n_seeds_ci_positive": gate_summary["A1b"]["n_seeds_ci_positive"],
        },
    ]
    df_gset = pd.DataFrame(gset_rows)
    S.write_tidy_table(df_gset, TABLES / "sec33_gset_summary.csv")

    # 2. G_set per seed table
    a2d_seeds = pd.read_csv(gate_dir / "gset_36_A2d_per_seed.csv")
    a2d_seeds["estimand"] = "A2d"
    a1b_seeds = pd.read_csv(gate_dir / "gset_36_A1b_per_seed.csv")
    a1b_seeds["estimand"] = "A1b"
    seen_seeds = p1_seeds.copy()
    seen_seeds["estimand"] = "Seen"
    seen_seeds["ci_positive"] = seen_seeds["lo95"] > 0
    df_gset_seeds = pd.concat([seen_seeds, a2d_seeds, a1b_seeds], ignore_index=True)
    S.write_tidy_table(df_gset_seeds, TABLES / "sec33_gset_per_seed.csv")

    # 3. Paired regret summary and per-seed
    df_regret_sum = pd.read_csv(regret_dir / "paired_regret_summary.csv")
    df_regret_seeds = pd.read_csv(regret_dir / "paired_regret_per_seed.csv")
    S.write_tidy_table(df_regret_sum, TABLES / "sec33_paired_regret_summary.csv")
    S.write_tidy_table(df_regret_seeds, TABLES / "sec33_paired_regret_per_seed.csv")

    # 4. Basin-specific vs Global Top-k (k=1..36)
    df_bs_vs_glob = pd.read_csv(transfer_dir / "basin_specific_vs_global_topk.csv")
    S.write_tidy_table(df_bs_vs_glob, TABLES / "sec33_basin_specific_vs_global_topk.csv")

    parity_rec = {
        "script": "sec33_04_gset_and_regret.py",
        "approach": "(b) Loaded frozen verified artifacts from oob36_gset_gate_20261001, oob36_paired_regret_20261003, pub_candidate_set_transfer_20260930",
        "upstream_artifacts": upstream_meta,
        "A2d_main_G_set": float(gate_summary["A2d"]["G_set"]),
        "A1b_main_G_set": float(gate_summary["A1b"]["G_set"]),
        "Seen_main_G_set": float(seen_main["G_set"]),
        "regret_diff_mean_A2d": float(df_regret_sum.loc[df_regret_sum.case == "A2d", "mean_diff_mean"].iloc[0]),
        "k3_bs_minus_glob_cov_diff": float(df_bs_vs_glob.loc[df_bs_vs_glob.k == 3, "coverage_diff_bs_minus_global"].iloc[0]),
    }
    (TABLES / "sec33_04_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec33_04] PASS: compiled G_set, paired regret, and transfer curves ({len(upstream_files)} files)")


if __name__ == "__main__":
    main()
