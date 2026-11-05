#!/usr/bin/env python3
"""01_remote_inventory.py — step 1.2 of ch4-oob36-download-prelim-20260930.

Lists, for all 36 models x 5 folds = 180 runs, where each run lives and which
files are needed, and writes ``$OUT/remote_inventory.csv``.

Source of truth for cohort membership and Qsim filenames (no guessing):
  * OOB36_MODEL_LIST.csv                 -> cohort membership + n_params
  * B1B2_summary.json                    -> cohorts "existing8" (8) and "v4_new8" (8)
  * AUDIT_ARTIFACT_MANIFEST.csv          -> evidence/qsim/<cohort>__<model>__fold<k>__qsim.npy
  * logs/OOB36_QSIM_MANIFEST.csv         -> evidence/qsim/oob36__<model>__fold<k>__qsim.npy

Read-only with respect to X99: the remote step only reads files and runs python
from stdin (``python -``); it creates no file on X99.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EXISTING8, V4_NEW8, V5_NEW20, MODELS, FOLDS, QSIM_PREFIX,
    OUT, OUT_TABLES, RUN_ROOT_LOCAL_EXISTING8, RUN_ROOT_REMOTE_V4, RUN_ROOT_REMOTE_V5,
    QSIM_ROOT_LOCAL_AUDIT, QSIM_ROOT_REMOTE_V5, REMOTE_ROOT, ssh_python,
)

COHORT_OF = {}
for _m in EXISTING8:
    COHORT_OF[_m] = "existing8"
for _m in V4_NEW8:
    COHORT_OF[_m] = "v4_new8"
for _m in V5_NEW20:
    COHORT_OF[_m] = "v5_new20"


# --------------------------------------------------------------------------- #
# run/qsime path resolution
# --------------------------------------------------------------------------- #

def run_dir_for(model: str, fold: int) -> tuple[str, str]:
    """Return (run_dir, location) where location is 'local' or 'remote'."""
    c = COHORT_OF[model]
    if c == "existing8":
        return f"{RUN_ROOT_LOCAL_EXISTING8}/{model}/fold_{fold}", "local"
    if c == "v4_new8":
        return f"{RUN_ROOT_REMOTE_V4}/{model}/fold_{fold}", "remote"
    if c == "v5_new20":
        return f"{RUN_ROOT_REMOTE_V5}/{model}/fold_{fold}", "remote"
    raise KeyError(c)


def qsim_path_for(model: str, fold: int) -> tuple[str, str]:
    """Return (qsim_path, location) of the frozen Qsim .npy for (model, fold)."""
    c = COHORT_OF[model]
    pref = QSIM_PREFIX[c]
    name = f"{pref}__{model}__fold{fold}__qsim.npy"
    if c in ("existing8", "v4_new8"):
        return f"{QSIM_ROOT_LOCAL_AUDIT}/{name}", "local"
    return f"{QSIM_ROOT_REMOTE_V5}/{name}", "remote"


# --------------------------------------------------------------------------- #
# local probe (existing8)
# --------------------------------------------------------------------------- #

def probe_local(model: str, fold: int) -> dict:
    import torch  # local torch

    run_dir, loc = run_dir_for(model, fold)
    rec = {"model": model, "fold": fold, "cohort": COHORT_OF[model], "location": loc,
           "run_dir": run_dir}
    notes = []

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir missing: {run_dir}")

    bp = os.path.join(run_dir, "best.pt")
    rec["best_ckpt_path"] = bp if os.path.isfile(bp) else "MISSING"
    if rec["best_ckpt_path"] == "MISSING":
        notes.append("best.pt absent")

    rj = os.path.join(run_dir, "result.json")
    if os.path.isfile(rj):
        r = json.load(open(rj))
        rec["best_epoch"] = r.get("best_epoch", "MISSING")
        rec["status"] = r.get("status", "MISSING")
    else:
        raise FileNotFoundError(f"result.json missing: {rj}")

    th = os.path.join(run_dir, "training_health.json")
    if os.path.isfile(th):
        h = json.load(open(th))
        rec["stop_epoch"] = h.get("stop_epoch", "MISSING")
    else:
        raise FileNotFoundError(f"training_health.json missing: {th}")

    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshot_ep*.pt")),
                   key=lambda p: int(re.search(r"snapshot_ep(\d+)\.pt$", p).group(1)))
    if snaps:
        last = snaps[-1]
        rec["last_ckpt_path"] = last
        ck = torch.load(last, map_location="cpu", weights_only=False)
        rec["last_ckpt_epoch"] = ck.get("epoch", "MISSING")
    else:
        rec["last_ckpt_path"] = "MISSING"
        rec["last_ckpt_epoch"] = "MISSING"
        notes.append("no snapshot_ep*.pt present; run completed 2026-09-03, before the "
                     "2026-09-28 snapshot+resume runner change, and the runner writes best.pt "
                     "only when train loss improves -> no stop-epoch checkpoint exists")

    rec["qsim_path"], qloc = qsim_path_for(model, fold)
    if not os.path.isfile(rec["qsim_path"]):
        notes.append(f"qsim not found at expected local path (location={qloc})")
    rec["epoch_metrics_path"] = os.path.join(run_dir, "epoch_metrics.csv")
    if not os.path.isfile(rec["epoch_metrics_path"]):
        notes.append("epoch_metrics.csv missing")
    rec["notes"] = "; ".join(notes)
    return rec


# --------------------------------------------------------------------------- #
# remote probe (v4_new8 + v5_new20) — one ssh round-trip, no files written
# --------------------------------------------------------------------------- #

REMOTE_PROBE = r'''
import glob, json, os, re, sys
import torch

models_v4 = %(v4)s
models_v5 = %(v5)s
RUN_ROOTS = {"v4_new8": %(root_v4)r, "v5_new20": %(root_v5)r}
QSIM_V5_ROOT = %(qsim_v5)r
QSIM_PREFIX = {"v4_new8": "v4_new8", "v5_new20": "oob36"}
QSIM_LOCAL_AUDIT = None  # v4_new8 qsim lives on the LOCAL machine, not here

out = []
for cohort, models in (("v4_new8", models_v4), ("v5_new20", models_v5)):
    root = RUN_ROOTS[cohort]
    for model in models:
        for fold in range(5):
            d = os.path.join(root, model, "fold_%%d" %% fold)
            rec = {"model": model, "fold": fold, "cohort": cohort,
                   "location": "remote", "run_dir": d}
            notes = []
            if not os.path.isdir(d):
                rec["error"] = "run_dir missing"
                out.append(rec)
                continue
            bp = os.path.join(d, "best.pt")
            rec["best_ckpt_path"] = bp if os.path.isfile(bp) else "MISSING"
            if rec["best_ckpt_path"] == "MISSING":
                notes.append("best.pt absent")

            rj = os.path.join(d, "result.json")
            if not os.path.isfile(rj):
                rec["error"] = "result.json missing"
                out.append(rec)
                continue
            r = json.load(open(rj))
            rec["best_epoch"] = r.get("best_epoch", "MISSING")
            rec["status"] = r.get("status", "MISSING")

            th = os.path.join(d, "training_health.json")
            if os.path.isfile(th):
                rec["stop_epoch"] = json.load(open(th)).get("stop_epoch", "MISSING")
            else:
                rec["stop_epoch"] = "MISSING"
                notes.append("training_health.json missing")

            snaps = sorted(glob.glob(os.path.join(d, "snapshot_ep*.pt")),
                           key=lambda p: int(re.search(r"snapshot_ep(\d+)\.pt$", p).group(1)))
            if snaps:
                last = snaps[-1]
                rec["last_ckpt_path"] = last
                ck = torch.load(last, map_location="cpu", weights_only=False)
                rec["last_ckpt_epoch"] = ck.get("epoch", "MISSING")
            else:
                rec["last_ckpt_path"] = "MISSING"
                rec["last_ckpt_epoch"] = "MISSING"
                if cohort == "v4_new8":
                    notes.append("no snapshot_ep*.pt present; run completed 2026-09-25, "
                                 "before the 2026-09-28 snapshot+resume runner change, and the "
                                 "runner writes best.pt only when train loss improves")
                else:
                    notes.append("no snapshot_ep*.pt present for this run")

            if cohort == "v5_new20":
                rec["qsim_path"] = os.path.join(
                    QSIM_V5_ROOT, "oob36__%%s__fold%%d__qsim.npy" %% (model, fold))
            else:
                rec["qsim_path"] = "LOCAL"   # resolved locally by the caller
            rec["epoch_metrics_path"] = os.path.join(d, "epoch_metrics.csv")
            if not os.path.isfile(rec["epoch_metrics_path"]):
                notes.append("epoch_metrics.csv missing")
            rec["notes"] = "; ".join(notes)
            out.append(rec)

sys.stdout.write("@@JSON@@" + json.dumps(out))
'''


def probe_remote() -> list[dict]:
    code = REMOTE_PROBE % {
        "v4": repr(list(V4_NEW8)), "v5": repr(list(V5_NEW20)),
        "root_v4": RUN_ROOT_REMOTE_V4, "root_v5": RUN_ROOT_REMOTE_V5,
        "qsim_v5": QSIM_ROOT_REMOTE_V5,
    }
    raw = ssh_python(code, timeout=3600)
    marker = "@@JSON@@"
    if marker not in raw:
        raise RuntimeError(f"remote probe produced no {marker} payload; stdout head:\n{raw[:2000]}")
    payload = raw.split(marker, 1)[1]
    return json.loads(payload)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    os.makedirs(OUT_TABLES, exist_ok=True)

    records: list[dict] = []

    # remote cohorts first (single round-trip)
    remote_recs = probe_remote()
    by_key = {(r["model"], r["fold"]): r for r in remote_recs}
    expected_remote = {(m, k) for m in list(V4_NEW8) + list(V5_NEW20) for k in FOLDS}
    missing_remote = expected_remote - set(by_key)
    if missing_remote:
        raise RuntimeError(f"remote probe returned no record for {sorted(missing_remote)}")

    for m in MODELS:
        for k in FOLDS:
            if COHORT_OF[m] == "existing8":
                rec = probe_local(m, k)
            else:
                rec = by_key[(m, k)]
                if rec["cohort"] == "v4_new8":
                    qp, qloc = qsim_path_for(m, k)
                    rec["qsim_path"] = qp
                if rec.get("error"):
                    raise RuntimeError(f"remote record has error for {m}/fold_{k}: {rec['error']}")
            records.append(rec)

    assert len(records) == 180, len(records)
    assert len({(r["model"], r["fold"]) for r in records}) == 180

    cols = ["model", "fold", "run_dir", "best_ckpt_path", "best_epoch",
            "last_ckpt_path", "stop_epoch", "last_ckpt_epoch",
            "qsim_path", "epoch_metrics_path",
            "cohort", "location", "status", "notes"]

    # no blank cells: everything missing is the literal string MISSING with a reason
    for r in records:
        for c in cols:
            v = r.get(c, "MISSING")
            if v is None or (isinstance(v, str) and v.strip() == ""):
                r[c] = "MISSING"
                r["notes"] = (r.get("notes", "") + f"; blank {c}").strip("; ")

    csv_path = os.path.join(OUT, "remote_inventory.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({c: r.get(c, "MISSING") for c in cols})

    # ---- summary printed (not swallowable) ----
    n_missing_last = sum(1 for r in records if r["last_ckpt_path"] == "MISSING")
    n_missing_qsim = sum(1 for r in records if r["qsim_path"] == "MISSING")
    n_epoch_mismatch = sum(1 for r in records
                           if r["last_ckpt_path"] != "MISSING"
                           and str(r["last_ckpt_epoch"]) != str(r["stop_epoch"]))
    by_cohort = {}
    for r in records:
        by_cohort.setdefault(r["cohort"], [0, 0, 0])
        by_cohort[r["cohort"]][0] += 1
        by_cohort[r["cohort"]][1] += int(r["last_ckpt_path"] == "MISSING")
        by_cohort[r["cohort"]][2] += int(str(r["last_ckpt_epoch"]) != str(r["stop_epoch"])
                                         and r["last_ckpt_path"] != "MISSING")

    print(f"rows written: {len(records)}  -> {csv_path}")
    print(f"MISSING last_ckpt_path : {n_missing_last} / 180")
    print(f"MISSING qsim_path      : {n_missing_qsim} / 180")
    print(f"last_ckpt_epoch != stop_epoch : {n_epoch_mismatch} / 180")
    for c, (n, ml, em) in sorted(by_cohort.items()):
        print(f"  cohort {c:9s} n={n:3d}  last_ckpt MISSING={ml:3d}  epoch!=stop={em:3d}")

    # also dump the raw records as json for downstream use
    json_path = os.path.join(OUT, "remote_inventory.json")
    json.dump(records, open(json_path, "w"), indent=1)
    print(f"raw records: {json_path}")


if __name__ == "__main__":
    main()
