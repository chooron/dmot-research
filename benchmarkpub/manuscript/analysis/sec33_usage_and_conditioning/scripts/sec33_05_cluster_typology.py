#!/usr/bin/env python3
"""sec33_05_cluster_typology.py — Compile cluster leave-out validation and typology organization tables."""
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
    val_dir = S.DIR_CLUSTER_VALIDATION / "tables"
    prof_dir = S.DIR_CLUSTER_PROFILES / "tables"

    upstream_files = [
        val_dir / "cluster_vs_global_summary.csv",
        val_dir / "per_cluster_oob_validation.csv",
        prof_dir / "cluster_concentration_summary.csv",
        prof_dir / "cluster_seen_oob_consistency.csv",
    ]
    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    # 1. Cluster vs Global OOB validation summary (k=3 and k=5)
    df_val_sum = pd.read_csv(val_dir / "cluster_vs_global_summary.csv")
    S.write_tidy_table(df_val_sum, TABLES / "sec33_cluster_oob_validation_summary.csv")

    # 2. Per-cluster OOB validation breakdown
    df_val_cluster = pd.read_csv(val_dir / "per_cluster_oob_validation.csv")
    S.write_tidy_table(df_val_cluster, TABLES / "sec33_cluster_oob_per_cluster.csv")

    # 3. Cluster concentration and normalized entropy
    df_conc = pd.read_csv(prof_dir / "cluster_concentration_summary.csv")
    S.write_tidy_table(df_conc, TABLES / "sec33_cluster_concentration_entropy.csv")

    # 4. Seen -> OOB stability across 10 clusters
    df_stab = pd.read_csv(prof_dir / "cluster_seen_oob_consistency.csv")
    S.write_tidy_table(df_stab, TABLES / "sec33_cluster_seen_oob_stability.csv")

    parity_rec = {
        "script": "sec33_05_cluster_typology.py",
        "approach": "(b) Loaded frozen verified artifacts from cluster_oob_recommendation_validation_20261001 and cluster_model_recommendation_20261001",
        "upstream_artifacts": upstream_meta,
        "cluster_k3_coverage_diff": float(df_val_sum.loc[df_val_sum.k == 3, "diff_coverage"].iloc[0]),
        "cluster_k5_coverage_diff": float(df_val_sum.loc[df_val_sum.k == 5, "diff_coverage"].iloc[0]),
        "cluster_k5_regret_diff_mean": float(df_val_sum.loc[df_val_sum.k == 5, "diff_regret_mean"].iloc[0]),
        "cluster_k5_wilcoxon_p": float(df_val_sum.loc[df_val_sum.k == 5, "wilcoxon_regret_p_value"].iloc[0]),
        "n_clusters": df_val_cluster["cluster"].nunique(),
    }
    (TABLES / "sec33_05_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec33_05] PASS: compiled cluster typology and validation tables ({len(upstream_files)} files)")


if __name__ == "__main__":
    main()
