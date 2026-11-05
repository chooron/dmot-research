#!/usr/bin/env python3
"""sec34_01_complexity_vs_instability.py — Compile parameter count vs parameter instability and displacement vs OOB loss."""
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
    src_parsimony = S.DIR_OOB36_PARSIMONY / "tables"

    upstream_files = [
        src_parsimony / "complexity_vs_transfer.csv",
        src_parsimony / "parameter_instability_vs_oob_loss.csv",
    ]
    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    # 1. n_parameters vs instability metrics
    df_ct = pd.read_csv(src_parsimony / "complexity_vs_transfer.csv")
    instability_vars = [
        "param_fold_instability_median",
        "param_displacement_median",
        "param_displacement_euclid_median",
    ]
    sub_ct = df_ct[(df_ct.x_quantity == "n_parameters") & (df_ct.y_quantity.isin(instability_vars))].copy()
    sub_ct["interpretation"] = sub_ct["y_quantity"].map({
        "param_fold_instability_median": "fold-to-fold variability (per-parameter RMS)",
        "param_displacement_median": "seen->OOB displacement (per-parameter RMS)",
        "param_displacement_euclid_median": "seen->OOB displacement (total Euclidean)",
    })
    S.write_tidy_table(sub_ct, TABLES / "sec34_nparam_vs_instability.csv")

    # 2. Parameter displacement vs OOB loss
    df_loss = pd.read_csv(src_parsimony / "parameter_instability_vs_oob_loss.csv")
    S.write_tidy_table(df_loss, TABLES / "sec34_displacement_vs_oob_loss.csv")

    parity_rec = {
        "script": "sec34_01_complexity_vs_instability.py",
        "approach": "(b) Loaded frozen verified artifacts from pub_parsimony_complexity_20260930",
        "upstream_artifacts": upstream_meta,
        "nparam_vs_fold_instability_rho": float(sub_ct.loc[sub_ct.y_quantity == "param_fold_instability_median", "spearman_rho"].iloc[0]),
        "nparam_vs_fold_instability_perm_p": float(sub_ct.loc[sub_ct.y_quantity == "param_fold_instability_median", "spearman_p_permutation"].iloc[0]),
        "nparam_vs_disp_rms_rho": float(sub_ct.loc[sub_ct.y_quantity == "param_displacement_median", "spearman_rho"].iloc[0]),
        "nparam_vs_disp_rms_perm_p": float(sub_ct.loc[sub_ct.y_quantity == "param_displacement_median", "spearman_p_permutation"].iloc[0]),
        "nparam_vs_disp_euclid_rho": float(sub_ct.loc[sub_ct.y_quantity == "param_displacement_euclid_median", "spearman_rho"].iloc[0]),
        "nparam_vs_disp_euclid_perm_p": float(sub_ct.loc[sub_ct.y_quantity == "param_displacement_euclid_median", "spearman_p_permutation"].iloc[0]),
        "pooled_disp_vs_loss_rho": float(df_loss.loc[df_loss.instability_metric == "param_displacement_seen_mean_rms", "pooled_spearman_rho"].iloc[0]),
        "model_median_disp_vs_loss_rho": float(df_loss.loc[df_loss.instability_metric == "param_displacement_seen_mean_rms", "model_median_spearman_rho"].iloc[0]),
    }
    (TABLES / "sec34_01_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec34_01] PASS: fold_rho={parity_rec['nparam_vs_fold_instability_rho']:.3f}, disp_rho={parity_rec['nparam_vs_disp_rms_rho']:.3f}, euclid_rho={parity_rec['nparam_vs_disp_euclid_rho']:.3f}")


if __name__ == "__main__":
    main()
