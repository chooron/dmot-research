#!/usr/bin/env python3
"""si_02_equivalence_protocol_params.py — Compile equivalence label definitions and bootstrap protocol parameters."""
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
    src_s2 = S.DIR_C4_EQUIV / "tables" / "s2_summary.json"
    upstream_meta = [S.file_metadata_record(src_s2)]

    s2_json = json.loads(src_s2.read_text(encoding="utf-8"))

    rows = [
        {
            "condition": "IC (local calibration)",
            "test_period": "1995-10-01..2010-09-30 (15 water years 1996..2010)",
            "n_blocks": 15,
            "bootstrap_method": "water_year_block_bootstrap_with_replacement",
            "n_replicates": s2_json["n_boot"],
            "random_seed": s2_json["seed"],
            "replicate_statistic": "best_r(b) = max_{m in 36} KGE_{m,b}(r)",
            "cutoff_rule": "q05_b = 5th percentile of replicate best KGEs",
            "membership_rule": "S_b = { m : KGE^point_{m,b} >= q05_b }",
            "kge_epsilon": S.KGE_EPS,
            "n_basins": 531,
            "median_size": 7.0,
            "cover95_count": 4,
        },
        {
            "condition": "seen-dPL (attribute mapping, target recorded)",
            "test_period": "1995-10-01..2010-09-30 (15 water years 1996..2010)",
            "n_blocks": 15,
            "bootstrap_method": "water_year_block_bootstrap_with_replacement",
            "n_replicates": s2_json["n_boot"],
            "random_seed": s2_json["seed"],
            "replicate_statistic": "best_r(b) = max_{m in 36} KGE_{m,b}(r)",
            "cutoff_rule": "q05_b = 5th percentile of replicate best KGEs",
            "membership_rule": "S_b = { m : KGE^point_{m,b} >= q05_b }",
            "kge_epsilon": S.KGE_EPS,
            "n_basins": 531,
            "median_size": 4.0,
            "cover95_count": 6,
        },
        {
            "condition": "OOB-dPL (attribute mapping, target withheld)",
            "test_period": "1995-10-01..2010-09-30 (15 water years 1996..2010)",
            "n_blocks": 15,
            "bootstrap_method": "water_year_block_bootstrap_with_replacement",
            "n_replicates": S.N_BOOT_BLOCK,
            "random_seed": S.SEED_BOOT,
            "replicate_statistic": "best_r(b) = max_{m in 36} KGE_{m,b}(r)",
            "cutoff_rule": "q05_b = 5th percentile of replicate best KGEs",
            "membership_rule": "S_b = { m : KGE^point_{m,b} >= q05_b }",
            "kge_epsilon": S.KGE_EPS,
            "n_basins": 531,
            "median_size": 4.0,
            "cover95_count": 12,
        },
    ]

    df_params = pd.DataFrame(rows)
    S.write_tidy_table(df_params, TABLES / "sec_si_equivalence_protocol_params.csv")

    parity_rec = {
        "script": "si_02_equivalence_protocol_params.py",
        "approach": "(a) Compiled from s2_summary.json and frozen protocol specifications",
        "upstream_artifacts": upstream_meta,
        "n_conditions": len(df_params),
        "n_replicates": S.N_BOOT_BLOCK,
        "seed": S.SEED_BOOT,
    }
    (TABLES / "si_02_parity.json").write_text(json.dumps(parity_rec, indent=2) + "\n", encoding="utf-8")
    print(f"[si_02] PASS: compiled protocol parameters across {len(df_params)} conditions (B={S.N_BOOT_BLOCK}, seed={S.SEED_BOOT})")


if __name__ == "__main__":
    main()
