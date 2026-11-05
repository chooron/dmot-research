#!/usr/bin/env python3
"""sec33_01_usage_rules_t2.py — Format T2 non-oracle usage rules A-E summary and per-basin outputs."""
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
    # Approach (b): Load frozen verified artifacts from equiv_cover_t1t2_20261003
    src_dir = S.DIR_EQUIV_T1T2 / "tables"
    upstream_files = [
        src_dir / "t2_rule_summary.csv",
        src_dir / "t2_rule_per_basin.csv",
        src_dir / "t2_set_construction_check.csv",
        src_dir / "t2_leakage_check.csv",
    ]
    upstream_meta = [S.file_metadata_record(f) for f in upstream_files]

    # 1. Summary across rules
    df_sum = pd.read_csv(src_dir / "t2_rule_summary.csv")
    S.write_tidy_table(df_sum, TABLES / "sec33_usage_rules_summary.csv")

    # 2. Per-basin outputs
    df_pb = pd.read_csv(src_dir / "t2_rule_per_basin.csv")
    S.write_tidy_table(df_pb, TABLES / "sec33_usage_rules_per_basin.csv")

    # 3. Set construction spot check
    df_chk = pd.read_csv(src_dir / "t2_set_construction_check.csv")
    S.write_tidy_table(df_chk, TABLES / "sec33_set_construction_spot_check.csv")

    # 4. Leakage check
    df_leak = pd.read_csv(src_dir / "t2_leakage_check.csv")
    S.write_tidy_table(df_leak, TABLES / "sec33_usage_rules_leakage_check.csv")

    parity_rec = {
        "script": "sec33_01_usage_rules_t2.py",
        "approach": "(b) Loaded frozen verified artifacts from equiv_cover_t1t2_20261003",
        "upstream_artifacts": upstream_meta,
        "rules_evaluated": sorted(df_sum["rule"].unique()),
        "k_values": [int(x) for x in df_sum["k"].unique()],
        "n_summary_rows": len(df_sum),
        "n_per_basin_rows": len(df_pb),
    }
    (TABLES / "sec33_01_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[sec33_01] PASS: formatted T2 rules A-E ({len(df_sum)} summary rows, {len(df_pb)} per-basin rows)")


if __name__ == "__main__":
    main()
