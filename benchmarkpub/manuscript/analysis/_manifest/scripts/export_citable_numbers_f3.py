#!/usr/bin/env python3
"""export_citable_numbers_f3.py — F3: export the freeze-ready citable-numbers list from the master manifest.

Citable list: rows with status RECOMPUTED or RE-VERIFIED (never SUPERSEDED), with estimand name, value,
file, column/key, Results section, generating script and status.
Excluded list: rows with status SUPERSEDED, LOADED or SENSITIVITY, with the exclusion reason.
Both lists are generated from `manuscript_master_manifest.csv`; no value is hand-written or transcribed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
BASE = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"
MANIFEST = BASE / "_manifest"
TABLES = MANIFEST / "tables"

CITABLE_STATUSES = ["RECOMPUTED", "RE-VERIFIED", "POST-FREEZE"]
EXCLUDED_STATUSES = ["SUPERSEDED", "LOADED", "SENSITIVITY"]

# H3: the freeze phase of every row is carried explicitly and must not be erased or merged.
FREEZE_PHASE = {
    "RECOMPUTED": "pre_freeze",
    "RE-VERIFIED": "pre_freeze",
    "POST-FREEZE": "post_freeze",
    "SUPERSEDED": "pre_freeze",
    "LOADED": "pre_freeze",
    "SENSITIVITY": "pre_freeze",
}

EXCLUSION_REASON = {
    "SUPERSEDED": "SUPERSEDED: retained on disk for comparison only; superseded by an ID-joined recomputation",
    "LOADED": "LOADED: loaded from a verified frozen upstream artifact; not independently recomputed in this tree",
    "SENSITIVITY": "SENSITIVITY: exploratory variant; not part of the citable main-result set",
}


def main() -> None:
    df = pd.read_csv(TABLES / "manuscript_master_manifest.csv")

    cit = df[df.derivation_type.isin(CITABLE_STATUSES)].copy()
    cit["freeze_phase"] = cit["derivation_type"].map(FREEZE_PHASE)
    cit_out = cit[["quantity_name", "value", "file_path", "column_or_key",
                   "results_section", "generating_script", "derivation_type", "freeze_phase"]].rename(
        columns={"quantity_name": "estimand", "derivation_type": "status"})
    cit_out = cit_out.sort_values(["results_section", "estimand"]).reset_index(drop=True)
    cit_out.to_csv(TABLES / "manuscript_citable_numbers.csv", index=False)

    exc = df[df.derivation_type.isin(EXCLUDED_STATUSES)].copy()
    exc["exclusion_reason"] = exc["derivation_type"].map(EXCLUSION_REASON)
    exc["freeze_phase"] = exc["derivation_type"].map(FREEZE_PHASE)
    exc_out = exc[["quantity_name", "value", "file_path", "column_or_key",
                   "results_section", "generating_script", "derivation_type", "freeze_phase",
                   "exclusion_reason"]].rename(
        columns={"quantity_name": "estimand", "derivation_type": "status"})
    exc_out = exc_out.sort_values(["status", "results_section", "estimand"]).reset_index(drop=True)
    exc_out.to_csv(TABLES / "manuscript_excluded_numbers.csv", index=False)

    print(f"[F3] citable rows: {len(cit_out)} (status counts: {cit.derivation_type.value_counts().to_dict()})")
    print(f"[F3] citable freeze_phase counts: {cit_out.freeze_phase.value_counts().to_dict()}")
    print(f"[F3] citable Results sections: {cit_out.results_section.nunique()} -> {sorted(cit_out.results_section.unique())}")
    print(f"[F3] excluded rows: {len(exc_out)} (status counts: {exc.derivation_type.value_counts().to_dict()})")
    print(f"[F3] excluded freeze_phase counts: {exc_out.freeze_phase.value_counts().to_dict()}")
    print(f"[F3] total manifest rows: {len(df)}; citable + excluded = {len(cit_out) + len(exc_out)}")
    assert len(cit_out) + len(exc_out) == len(df), "citable + excluded must cover the whole manifest"


if __name__ == "__main__":
    main()
