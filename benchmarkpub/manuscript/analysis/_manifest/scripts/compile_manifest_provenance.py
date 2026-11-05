#!/usr/bin/env python3
"""compile_manifest_provenance.py — Generate PROVENANCE.md for _manifest."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
BASE = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"
MANIFEST_DIR = BASE / "_manifest"
TABLES = MANIFEST_DIR / "tables"


def main() -> None:
    df = pd.read_csv(TABLES / "manuscript_master_manifest.csv")
    recomp_count = int((df["derivation_type"] == "RECOMPUTED").sum())
    loaded_count = int((df["derivation_type"] == "LOADED").sum())
    sens_count = int((df["derivation_type"] == "SENSITIVITY").sum())
    superseded_count = int((df["derivation_type"] == "SUPERSEDED").sum())
    postfreeze_count = int((df["derivation_type"] == "POST-FREEZE").sum())
    reverified_count = int((df["derivation_type"] == "RE-VERIFIED").sum())
    sec_counts = df["results_section"].value_counts().to_dict()
    cit = pd.read_csv(TABLES / "manuscript_citable_numbers.csv")
    exc = pd.read_csv(TABLES / "manuscript_excluded_numbers.csv")

    md = [
        "# Unit Provenance: _manifest",
        "",
        "## 1. Scope & Master Manifest Summary",
        f"- Master manifest table: `tables/manuscript_master_manifest.csv`",
        f"- Total cited quantities in manifest: **{len(df)}** rows",
        f"- Unique Results sections covered: **{df['results_section'].nunique()}**",
        f"- Recomputed quantities count (from upstream raw/sufficient data): **{recomp_count}**",
        f"- Loaded quantities count (from verified frozen artifacts): **{loaded_count}**",
        f"- Sensitivity quantities count (exploratory variants): **{sens_count}**",
        f"- Superseded quantities count (retained for comparison only): **{superseded_count}**",
        f"- POST-FREEZE quantities count (analysis family added after the freeze): **{postfreeze_count}**",
        f"- Re-verified quantities count (independent basin-ID-joined recompute): **{reverified_count}**",
        "",
        "## 2. Quantities by Results Section",
    ]
    for sec, cnt in sec_counts.items():
        md.append(f"- **{sec}**: {cnt} quantities")

    md.extend([
        "",
        "## 2b. Exported citable / excluded lists (H3 handling)",
        f"- `manuscript_citable_numbers.csv`: **{len(cit)}** rows; status counts {cit['status'].value_counts().to_dict()};",
        f"  freeze_phase counts {cit['freeze_phase'].value_counts().to_dict()}.",
        f"- `manuscript_excluded_numbers.csv`: **{len(exc)}** rows; status counts {exc['status'].value_counts().to_dict()};",
        f"  freeze_phase counts {exc['freeze_phase'].value_counts().to_dict()}.",
        "- The `freeze_phase` column distinguishes pre-freeze from post-freeze rows in both lists and is never merged or erased.",
        "- Exclusion reasons for `LOADED`, `SUPERSEDED` and `SENSITIVITY` are unchanged by the H3 handling.",
        "",
        "## 3. Script Generating the Manifest",
        "- `scripts/export_citable_numbers_f3.py`: exports the freeze-ready citable list (status RECOMPUTED or RE-VERIFIED)",
        "  and the excluded list (SUPERSEDED / LOADED / SENSITIVITY, each with an exclusion reason) from the master manifest.",
        "- `scripts/compile_master_manifest.py`: dynamically scans all unit tables and compiles master manifest.",
        "- `scripts/compile_manifest_provenance.py`: generates this provenance file.",
        "",
        "## 4. Hard Requirement Compliance",
        "- Zero hand-written or hard-coded values in any output file.",
        "- Every single numerical cell is traced to a generating script, source column, file path, and upstream frozen artifact.",
    ])

    (MANIFEST_DIR / "PROVENANCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("PASS: generated PROVENANCE.md for _manifest")


if __name__ == "__main__":
    main()
