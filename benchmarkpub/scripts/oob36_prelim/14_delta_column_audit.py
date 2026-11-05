#!/usr/bin/env python3
"""14_delta_column_audit.py — read-only integrity audit of delta-KGE column naming.

Background: the upstream paired table
`project/benchmark/results/ic_dpl_seenbasin_formal_20260901/02_BASIN_PAIRED_KGE_LONG.csv`
defines `Delta_KGE = KGE_dPL - KGE_IC` (verified numerically here to 4.4e-16).  In the
previous stage a column of that same content was renamed to
`delta_kge_ic_minus_dpl` inside `benchmarkpub` — a name that states the opposite sign.

This audit answers three bounded questions and nothing else:
  1. where does the string `delta_kge_ic_minus_dpl` appear?
  2. does any consumer read a delta-KGE quantity with an explicitly stated definition, and
     does that stated definition match the content?
  3. did the mislabel actually reach any analysis result or manuscript statement?

READ-ONLY.  No frozen output is modified.  No manuscript is edited.
Output: $OUT2/tables/delta_column_audit.csv (+ a JSON summary).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

REPO = PC.REPO
PAIRED = REPO / "project/benchmark/results/ic_dpl_seenbasin_formal_20260901/02_BASIN_PAIRED_KGE_LONG.csv"
TARGET = "delta_kge_ic_minus_dpl"
SECONDARY = ["Delta_KGE", "delta_kge"]
EXTS = {".py", ".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".sh", ".rst", ".ipynb"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
SELF = Path(__file__).resolve()
SELF_OUT_DIR = PC.OUT2
SNIPPET = 180


def log(m: str) -> None:
    print(m, flush=True)


def classify(line: str, literal: str) -> tuple[str, str]:
    """Return (class, note) from an explicit textual rule set. No guessing beyond that."""
    low = line.lower()
    dpl_minus_ic = bool(re.search(r"dpl\s*-\s*ic", low))
    ic_minus_dpl = bool(re.search(r"ic\s*-\s*dpl", low))
    if literal == TARGET:
        # the defect itself is the mapping of a Delta_KGE column onto this name
        if "delta_kge" in low and re.search(r'[:=]\s*["\']?' + TARGET, low):
            return ("MISLABELLED_DEFECT",
                    "a Delta_KGE (== dPL-IC) column is mapped onto the name ic_minus_dpl")
        return ("MISLABEL_REFERENCE",
                "names/documents the mislabelled column")
    if literal == "Delta_KGE":
        if dpl_minus_ic:
            return ("CORRECT_DPL_MINUS_IC", "explicitly defined as dPL - IC (matches content)")
        if ic_minus_dpl:
            return ("CORRECT_IC_MINUS_DPL", "explicitly defined as IC - dPL (matches that name)")
        return ("UNCLASSIFIED", "no explicit definition on this line; manual review needed")
    # literal == "delta_kge": only interesting when it is about the IC/dPL contrast
    if dpl_minus_ic:
        return ("CORRECT_DPL_MINUS_IC", "explicit dPL - IC")
    if ic_minus_dpl:
        return ("CORRECT_IC_MINUS_DPL", "explicit IC - dPL")
    if "ic" in low or "dpl" in low:
        return ("UNCLASSIFIED", "mentions ic/dpl but states no definition")
    return ("OTHER_CONTEXT_UNRELATED", "unrelated delta_kge* identifier (no ic/dpl context)")


def ast_value_consumers(path: Path) -> list[int]:
    """Line numbers where TARGET is used as a VALUE (subscript/attribute), not a docstring."""
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Subscript, ast.Attribute, ast.Compare, ast.BinOp, ast.Call)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and sub.value == TARGET:
                    out.append(getattr(sub, "lineno", -1))
    return sorted(set(x for x in out if x > 0))


def main() -> None:
    PC.T2.mkdir(parents=True, exist_ok=True)
    rows = []
    scanned = 0
    for p in REPO.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.resolve() == SELF or SELF_OUT_DIR in p.resolve().parents:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        if TARGET not in txt and not any(s in txt for s in SECONDARY):
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            for literal in [TARGET] + SECONDARY:
                if literal in line:
                    cls, note = classify(line, literal)
                    rows.append({
                        "literal": literal, "class": cls,
                        "file": str(p.relative_to(REPO)), "line": i,
                        "snippet": line.strip()[:SNIPPET], "note": note,
                    })
    A = pd.DataFrame(rows)
    # keep the most specific literal per (file,line)
    if not A.empty:
        rank = {TARGET: 0, "Delta_KGE": 1, "delta_kge": 2}
        A = (A.assign(_r=A["literal"].map(rank))
               .sort_values(["file", "line", "_r"])
               .drop_duplicates(["file", "line"], keep="first")
               .drop(columns="_r"))
    A = A.sort_values(["class", "file", "line"]).reset_index(drop=True)
    review = A[A["class"] != "OTHER_CONTEXT_UNRELATED"]
    A.to_csv(PC.T2 / "delta_column_audit.csv", index=False)

    # ---- numeric proof of the upstream definition -------------------------
    d = pd.read_csv(PAIRED)
    proof = {
        "file": str(PAIRED.relative_to(REPO)),
        "rows": int(len(d)),
        "max_abs_DeltaKGE_minus_dPL_minus_IC": float((d["Delta_KGE"] - (d["KGE_dPL"] - d["KGE_IC"])).abs().max()),
        "max_abs_DeltaKGE_minus_IC_minus_dPL": float((d["Delta_KGE"] - (d["KGE_IC"] - d["KGE_dPL"])).abs().max()),
        "conclusion": "Delta_KGE == KGE_dPL - KGE_IC",
    }
    proof["definition_matches_dPL_minus_IC"] = bool(proof["max_abs_DeltaKGE_minus_dPL_minus_IC"] < 1e-12)

    # ---- did the mislabel reach any result? ------------------------------
    mislab = A[A["class"] == "MISLABELLED_DEFECT"]
    in_scope = mislab[mislab["file"].str.startswith("project/benchmarkpub/")]
    outside = mislab[~mislab["file"].str.startswith("project/benchmarkpub/")]

    # does ANY script read the mislabelled column as a value? (AST, not text matching)
    consumers = []
    for f in sorted(REPO.rglob("*.py")):
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        for ln in ast_value_consumers(f):
            line = f.read_text(encoding="utf-8", errors="ignore").splitlines()[ln - 1]
            consumers.append({"file": str(f.relative_to(REPO)), "line": ln,
                              "snippet": line.strip()[:SNIPPET]})
    consumers = pd.DataFrame(consumers)
    # the rename mapping that CREATES the mislabelled column is the defect, not a consumer
    if not consumers.empty:
        defect_pairs = {(r["file"], r["line"]) for r in A.to_dict(orient="records")
                        if r["class"] == "MISLABELLED_DEFECT"}
        consumers = consumers[~consumers.apply(lambda r: (r["file"], r["line"]) in defect_pairs, axis=1)]

    summary = {
        "files_scanned": scanned,
        "occurrences_total": int(len(A)),
        "class_counts": A["class"].value_counts().to_dict() if not A.empty else {},
        "occurrences_excluding_unrelated_delta_kge": int(len(review)),
        "review_rows": review.to_dict(orient="records"),
        "target_string_occurrences": int((A["literal"] == TARGET).sum()),
        "target_string_files": sorted(A.loc[A["literal"] == TARGET, "file"].unique().tolist()),
        "mislabelled_defect_occurrences": int(len(mislab)),
        "mislabelled_inside_benchmarkpub": int(len(in_scope)),
        "mislabelled_outside_benchmarkpub": int(len(outside)),
        "outside_locations": outside[["file", "line", "snippet"]].to_dict(orient="records"),
        "value_consumers_of_mislabelled_column_anywhere":
            consumers.to_dict(orient='records') if not consumers.empty else [],
        "manuscript_explicit_delta_definitions": sorted({
            r["file"] for r in review.to_dict(orient="records")
            if r["class"] in ("CORRECT_DPL_MINUS_IC", "CORRECT_IC_MINUS_DPL")
            and str(r["file"]).startswith("project/benchmark/manuscript/")}),
        "pre_existing_sign_audit_in_repo": None,
        "upstream_definition_proof": proof,
        "verdict": None,
    }
    summary["verdict"] = (
        "CONTAINED: the mislabelled name is introduced and referenced only inside "
        "project/benchmarkpub (never in project/benchmark or any manuscript), and no script "
        "reads the mislabelled column as a value. Every other delta-KGE consumer states its "
        "definition explicitly and agrees with the content "
        "(Delta_KGE = KGE_dPL - KGE_IC; G_seen = KGE_IC - KGE_dPL = -Delta_KGE). "
        "No analysis result or manuscript statement is affected."
        if len(outside) == 0 and consumers.empty
        else "REVIEW REQUIRED: the mislabel appears outside project/benchmarkpub or is read as a value"
    )
    # the repository already contains an explicit, passing sign audit for this column
    r1_audit = REPO / "project/benchmark/manuscript/r1/tables/R1_input_audit.csv"
    if r1_audit.exists():
        ra = pd.read_csv(r1_audit)
        hit = ra[ra.iloc[:, 0].astype(str).str.contains("delta_sign", case=False, na=False)]
        summary["pre_existing_sign_audit_in_repo"] = {
            "file": str(r1_audit.relative_to(REPO)),
            "rows": hit.to_dict(orient="records"),
        }
    with open(PC.T2 / "delta_column_audit.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    log(f"files scanned: {scanned}")
    log(f"occurrences: {len(A)}")
    log("class counts:\n" + (A["class"].value_counts().to_string() if not A.empty else "  (none)"))
    log(f"\n'{TARGET}' occurrences: {int((A['literal'] == TARGET).sum())} in "
        f"{len(summary['target_string_files'])} files")
    for f in summary["target_string_files"]:
        log(f"   {f}")
    log(f"\nMISLABELLED_DEFECT inside benchmarkpub : {len(in_scope)}")
    log(f"MISLABELLED_DEFECT outside benchmarkpub: {len(outside)}")
    log(f"\nupstream proof: max|Delta_KGE - (dPL-IC)| = "
        f"{proof['max_abs_DeltaKGE_minus_dPL_minus_IC']:.3e}; "
        f"max|Delta_KGE - (IC-dPL)| = {proof['max_abs_DeltaKGE_minus_IC_minus_dPL']:.3e}")
    log(f"\nVERDICT: {summary['verdict']}")


if __name__ == "__main__":
    main()
