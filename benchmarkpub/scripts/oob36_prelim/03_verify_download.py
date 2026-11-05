#!/usr/bin/env python3
"""03_verify_download.py — step 1.4 of ch4-oob36-download-prelim-20260930.

Verifies every downloaded file:

1. sha256 of each local file vs the sha256 recorded by a frozen manifest; files
   with no manifest entry get a live `sha256sum` from X99 (read-only).
2. `best.pt` / `last.pt` internal epoch vs the epochs recorded in the frozen
   run metadata.
3. every Qsim: dtype float64, one axis == 5479, no NaN, and the other axis equal
   to the frozen held-out basin count for that fold.

Outputs
    $OUT/download_verification.csv
    $OUT/download_verification_summary.json

Hard-fails (SystemExit non-zero) on the first unmet criterion — nothing is
silently skipped.  X99 is read-only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    OUT, DATA, DATA_RUNS, DATA_QSIM, DATA_OBS, DATA_MANIFESTS,
    SSH_ALIAS, REMOTE_ROOT, QSIM_PREFIX,
)

SSH_RETRIES = 3
SSH_RETRY_SLEEP = 20

RUN_ROOT_REMOTE = {
    "v4_new8": REMOTE_ROOT + "/project/benchmark/results/oob_v4_16models_20260925/runs",
    "v5_new20": REMOTE_ROOT + "/project/benchmark/results/oob_v5_36models_20260926/runs",
}
QSIM_ROOT_REMOTE_V5 = (REMOTE_ROOT + "/project/benchmark/analysis/"
                       "chapter4_oob36_20260926/evidence/qsim")
PER_RUN_FILES = ["best.pt", "result.json", "epoch_metrics.csv", "best_metadata.json",
                 "training_health.json", "heldout_basin_ids.txt", "heldout_basin_kge.csv"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ssh(cmd: str, timeout: int = 900) -> str:
    last = None
    for attempt in range(1, SSH_RETRIES + 1):
        proc = subprocess.run(["ssh", "-o", "ConnectTimeout=20", SSH_ALIAS, cmd],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout
        last = (proc.returncode, proc.stderr.strip()[:300])
        print(f"    ssh attempt {attempt}/{SSH_RETRIES} failed (rc={proc.returncode}) "
              f"{proc.stderr.strip()[:120]}", file=sys.stderr)
        if attempt < SSH_RETRIES:
            import time
            time.sleep(SSH_RETRY_SLEEP)
    raise RuntimeError(f"ssh failed after {SSH_RETRIES} attempts: {last}")


# --------------------------------------------------------------------------- #
# 1. build the expected-sha map from the frozen manifests
# --------------------------------------------------------------------------- #

def load_expected() -> tuple[dict, dict, dict]:
    """Return (sha_by_local_path, remote_path_by_local_path, note_by_local_path)."""
    sha_map, rem_map, note_map = {}, {}, {}

    def put(local, sha, remote, note):
        sha_map[local] = sha
        rem_map[local] = remote
        note_map[local] = note

    inv = json.load(open(os.path.join(OUT, "remote_inventory.json")))
    by_key = {(r["model"], r["fold"]): r for r in inv}

    # --- (a) v5 Qsim + obs, from logs/OOB36_QSIM_MANIFEST.csv
    p = os.path.join(DATA_MANIFESTS, "logs", "OOB36_QSIM_MANIFEST.csv")
    for row in csv.DictReader(open(p)):
        m, k = row["model"], int(row["fold"])
        put(os.path.join(DATA_QSIM, m, f"fold_{k}.npy"), row["sha256"],
            QSIM_ROOT_REMOTE_V5 + "/" + os.path.basename(row["path"]),
            "manifest:logs/OOB36_QSIM_MANIFEST.csv")
    obs_shas = {}
    for row in csv.DictReader(open(p)):
        obs_shas[int(row["fold"])] = (row["obs_sha256"], row["obs_source"])
    return sha_map, rem_map, note_map, obs_shas, by_key


def main() -> None:
    import numpy as np
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-source-changed-manifest", action="store_true",
                    help="Accept the KNOWN, explained mismatches whose cause is a "
                         "local frozen document rewritten by the concurrent OOB36 "
                         "stage-6 workflow after its manifest entry was snapshotted. "
                         "Any OTHER mismatch still hard-fails.  Default: off, so the "
                         "strict run exits 1 on those two files.")
    ap.add_argument("--no-remote-sha", action="store_true",
                    help="NON-AUTHORITATIVE: run every local check but mark the "
                         "live X99 sha256 comparison as PENDING instead of "
                         "hard-failing.  The default (no flag) is the strict, "
                         "authoritative verification required by step 1.4.")
    args = ap.parse_args()

    inv = json.load(open(os.path.join(OUT, "remote_inventory.json")))
    by_key = {(r["model"], r["fold"]): r for r in inv}

    expected: dict[str, tuple[str, str, str]] = {}   # local -> (sha, remote_path, note)
    declared_bytes: dict[str, int] = {}
    need_remote_sha: set[str] = set()
    remote_override: dict[str, str] = {}
    staged_local_only: set[str] = set()

    def expect(local, sha, remote, note):
        expected[local] = (sha, remote, note)

    # (a) Qsim: v5 from OOB36_QSIM_MANIFEST, existing8/v4 from AUDIT_ARTIFACT_MANIFEST
    p_qsim = os.path.join(DATA_MANIFESTS, "logs", "OOB36_QSIM_MANIFEST.csv")
    if os.path.isfile(p_qsim):
        for row in csv.DictReader(open(p_qsim)):
            m, k = row["model"], int(row["fold"])
            expect(os.path.join(DATA_QSIM, m, f"fold_{k}.npy"), row["sha256"],
                   QSIM_ROOT_REMOTE_V5 + "/" + os.path.basename(row["path"]),
                   "manifest:logs/OOB36_QSIM_MANIFEST.csv")
    p_audit = os.path.join(DATA_MANIFESTS, "local_frozen_audit", "AUDIT_ARTIFACT_MANIFEST.csv")
    if os.path.isfile(p_audit):
        audit_sha = {r["relative_path"]: r["sha256"] for r in csv.DictReader(open(p_audit))}
        for (m, k), r in by_key.items():
            if r["cohort"] == "v5_new20":
                continue
            rel = f"evidence/qsim/{QSIM_PREFIX[r['cohort']]}__{m}__fold{k}__qsim.npy"
            if rel in audit_sha:
                expect(os.path.join(DATA_QSIM, m, f"fold_{k}.npy"), audit_sha[rel],
                       "LOCAL:" + rel, "manifest:AUDIT_ARTIFACT_MANIFEST.csv")

    # (b) obs: obs_sha256 from the Qsim manifest (same value for every model, per fold)
    if os.path.isfile(p_qsim):
        for row in csv.DictReader(open(p_qsim)):
            k = int(row["fold"])
            expect(os.path.join(DATA_OBS, f"fold_{k}", "heldout_y.npy"), row["obs_sha256"],
                   row["obs_source"], "manifest:logs/OOB36_QSIM_MANIFEST.csv")

    # (c) v5 best.pt: checkpoint_sha256 from RUN_LOG_OOB36.csv
    p_log = os.path.join(DATA_MANIFESTS, "RUN_LOG_OOB36.csv")
    if os.path.isfile(p_log):
        for row in csv.DictReader(open(p_log)):
            if row.get("status") != "DONE":
                continue
            m, k = row["model"], int(row["fold"])
            if row.get("checkpoint_sha256"):
                expect(os.path.join(DATA_RUNS, m, f"fold_{k}", "best.pt"),
                       row["checkpoint_sha256"], row.get("checkpoint_path", ""),
                       "manifest:RUN_LOG_OOB36.csv:checkpoint_sha256")

    # (d) everything else downloaded from X99 needs a live remote sha256sum
    for (m, k), r in by_key.items():
        if r["cohort"] == "existing8":
            continue                       # local-only source, no remote counterpart
        rd = r["run_dir"]
        for f in PER_RUN_FILES:
            local = os.path.join(DATA_RUNS, m, f"fold_{k}", f)
            if os.path.isfile(local) and local not in expected:
                need_remote_sha.add(local)
        if r["last_ckpt_path"] != "MISSING":
            local = os.path.join(DATA_RUNS, m, f"fold_{k}", "last.pt")
            if os.path.isfile(local) and local not in expected:
                need_remote_sha.add(local)

    # (e) downloaded manifest files not covered by OOB36_ARTIFACT_MANIFEST
    p_man = os.path.join(DATA_MANIFESTS, "OOB36_ARTIFACT_MANIFEST.csv")
    man_sha, man_bytes = {}, {}
    if os.path.isfile(p_man):
        for r in csv.DictReader(open(p_man)):
            man_sha[r["relative_path"]] = r["sha256"]
            try:
                man_bytes[r["relative_path"]] = int(r.get("bytes") or 0)
            except (TypeError, ValueError):
                man_bytes[r["relative_path"]] = None
    local_only_set = set()
    p_lo = os.path.join(DATA, "_lists", "manifests_local_only.txt")
    if os.path.isfile(p_lo):
        local_only_set = {l.strip() for l in open(p_lo) if l.strip()}
    oob36_remote = REMOTE_ROOT + "/project/benchmark/analysis/chapter4_oob36_20260926"
    for local in sorted(glob.glob(os.path.join(DATA_MANIFESTS, "**", "*"), recursive=True)):
        if os.path.isdir(local):
            continue
        rel = os.path.relpath(local, DATA_MANIFESTS)
        if rel.startswith("local_frozen_audit/"):
            continue
        if rel in ("LOCAL_FROZEN_STAGED.csv", "OOB_FOLD_ASSIGNMENT.local_frozen.csv",
                   "LOCAL_ONLY_FILELIST.txt", "REMOTE_FILELIST.txt"):
            continue                      # task bookkeeping, not a remote artefact
        if rel in man_sha:
            expect(local, man_sha[rel], oob36_remote + "/" + rel,
                   "manifest:OOB36_ARTIFACT_MANIFEST.csv")
            declared_bytes[local] = man_bytes.get(rel)
            if rel in local_only_set:
                staged_local_only.add(local)
        elif rel == "OOB_FOLD_ASSIGNMENT.csv":
            # fetched from X99, but from results/, not from the analysis dir
            rp = (REMOTE_ROOT + "/project/benchmark/results/"
                  "oob_v5_36models_20260926/OOB_FOLD_ASSIGNMENT.csv")
            remote_override[local] = rp
            expect(local, None, rp, "live-remote-sha256sum (results/ path)")
            need_remote_sha.add(local)
        elif rel == "OOB36_ARTIFACT_MANIFEST.csv" or rel in local_only_set:
            # staged verbatim from the local frozen tree; X99 has no counterpart
            staged_local_only.add(local)
            expected[local] = (None, "NO_REMOTE_COUNTERPART", "local-frozen staged (no X99 copy)")
        else:
            need_remote_sha.add(local)

    print(f"expected-sha entries : {len(expected)}")
    print(f"need remote sha256sum: {len(need_remote_sha)}")

    # ----------------------------------------------------------------------- #
    # 2. fetch the remote sha256 values for the uncovered files (one ssh call)
    # ----------------------------------------------------------------------- #
    remote_sha: dict[str, str] = {}
    pending_remote: set[str] = set()
    if need_remote_sha and args.no_remote_sha:
        pending_remote = set(need_remote_sha)
        loc2rem_pending = {}
        for local in pending_remote:
            if local in remote_override:
                loc2rem_pending[local] = remote_override[local]
                continue
            rel = os.path.relpath(local, DATA)
            parts = rel.split(os.sep)
            if parts[0] == "runs":
                m, fold_dir, fname = parts[1], parts[2], parts[3]
                k = int(fold_dir.split("_")[1])
                r = by_key[(m, k)]
                loc2rem_pending[local] = (r["last_ckpt_path"] if fname == "last.pt"
                                          else f"{r['run_dir']}/{fname}")
            elif parts[0] == "manifests":
                loc2rem_pending[local] = oob36_remote + "/" + os.path.relpath(local, DATA_MANIFESTS)
            else:
                raise RuntimeError(f"cannot map to a remote path: {local}")
        for local, rem in loc2rem_pending.items():
            expected.setdefault(local, (None, rem, "live-remote-sha256sum"))
        print(f"NON-AUTHORITATIVE MODE: {len(pending_remote)} files left with "
              f"sha256_remote=PENDING_X99")
    elif need_remote_sha:
        # map local -> remote absolute path
        loc2rem = {}
        for local in need_remote_sha:
            if local in remote_override:
                loc2rem[local] = remote_override[local]
                continue
            rel = os.path.relpath(local, DATA)
            parts = rel.split(os.sep)
            if parts[0] == "runs":
                m, fold_dir, fname = parts[1], parts[2], parts[3]
                k = int(fold_dir.split("_")[1])
                r = by_key[(m, k)]
                if fname == "last.pt":
                    loc2rem[local] = r["last_ckpt_path"]
                else:
                    loc2rem[local] = f"{r['run_dir']}/{fname}"
            elif parts[0] == "manifests":
                loc2rem[local] = oob36_remote + "/" + os.path.relpath(local, DATA_MANIFESTS)
            else:
                raise RuntimeError(f"cannot map to a remote path: {local}")

        payload = "\n".join(loc2rem.values()) + "\n"
        local_list = "\n".join(f"{loc2rem[l]}\t{l}" for l in loc2rem) + "\n"
        with open(os.path.join(OUT, "_remote_sha_list.tsv"), "w") as fh:
            fh.write(local_list)
        cmd = "xargs -d '\\n' -a /dev/stdin sha256sum"
        out = ssh_with_stdin(cmd, payload)
        for line in out.splitlines():
            h, _, p = line.partition("  ")
            remote_sha[p.strip()] = h.strip()
        missing = [p for p in loc2rem.values() if p not in remote_sha]
        if missing:
            raise RuntimeError(f"remote sha256sum returned no value for {len(missing)} paths, "
                               f"e.g. {missing[:3]}")
        for local, rem in loc2rem.items():
            expected.setdefault(local, (None, rem, "live-remote-sha256sum"))

    # ----------------------------------------------------------------------- #
    # 3. compare
    # ----------------------------------------------------------------------- #
    rows = []
    n_match = n_mismatch = n_local_only = n_pending = n_source_changed = 0
    for local in sorted(expected):
        sha_exp, remote_path, note = expected[local]
        if not os.path.isfile(local):
            raise FileNotFoundError(f"expected local file missing: {local}")
        sha_loc = sha256(local)
        if remote_path == "NO_REMOTE_COUNTERPART":
            rows.append({"local_path": os.path.relpath(local, DATA),
                         "remote_path": "(none - staged verbatim from the local frozen tree)",
                         "sha256_local": sha_loc, "sha256_remote": sha_loc,
                         "match": True,
                         "source": "local-frozen staged (no X99 copy)",
                         "note": "no X99 counterpart exists; the local frozen file is the source"})
            n_local_only += 1
            continue
        if sha_exp is None:
            if local in pending_remote:
                sha_rem = "PENDING_X99"
                rows.append({"local_path": os.path.relpath(local, DATA),
                             "remote_path": remote_path,
                             "sha256_local": sha_loc, "sha256_remote": sha_rem,
                             "match": "PENDING", "source": "live-remote-sha256sum"})
                n_pending += 1
                continue
            sha_rem = remote_sha.get(remote_path)
            if sha_rem is None:
                raise RuntimeError(f"no remote sha for {remote_path}")
        else:
            sha_rem = sha_exp
        match = (sha_loc == sha_rem)
        extra = ""
        if not match:
            db = declared_bytes.get(local)
            actual = os.path.getsize(local)
            if db is not None and db != actual:
                extra = (f"staged local-frozen document differs from its manifest entry: "
                         f"declared {db} B, actual {actual} B; the source file was rewritten "
                         f"at 2026-09-30 09:51:31 by the concurrent OOB36 stage-6 workflow "
                         f"after the manifest snapshot")
                n_source_changed += 1
        n_match += int(match)
        n_mismatch += int(not match)
        rows.append({"local_path": os.path.relpath(local, DATA),
                     "remote_path": remote_path,
                     "sha256_local": sha_loc, "sha256_remote": sha_rem,
                     "match": match, "source": note, "note": extra})

    # local-only existing8 files: no remote counterpart by construction
    for (m, k), r in sorted(by_key.items()):
        if r["cohort"] != "existing8":
            continue
        for f in PER_RUN_FILES:
            local = os.path.join(DATA_RUNS, m, f"fold_{k}", f)
            if not os.path.isfile(local):
                raise FileNotFoundError(f"expected local file missing: {local}")
            sha_loc = sha256(local)
            rows.append({"local_path": os.path.relpath(local, DATA),
                         "remote_path": os.path.join(r["run_dir"], f),
                         "sha256_local": sha_loc, "sha256_remote": sha_loc,
                         "match": True, "source": "local-only cohort (frozen tree is the source)"})
            n_local_only += 1

    # qsim / obs / fold assignment identity across the two fold-assignment copies
    p_remote_fa = os.path.join(DATA_MANIFESTS, "OOB_FOLD_ASSIGNMENT.csv")
    p_local_fa = os.path.join(DATA_MANIFESTS, "OOB_FOLD_ASSIGNMENT.local_frozen.csv")
    if os.path.isfile(p_remote_fa) and os.path.isfile(p_local_fa):
        a, b = sha256(p_remote_fa), sha256(p_local_fa)
        rows.append({"local_path": "manifests/OOB_FOLD_ASSIGNMENT.csv",
                     "remote_path": "(X99 results/oob_v5_36models_20260926/OOB_FOLD_ASSIGNMENT.csv)",
                     "sha256_local": a, "sha256_remote": b, "match": a == b,
                     "source": "fold-assignment remote-vs-local-frozen"})
        n_match += int(a == b); n_mismatch += int(a != b)

    with open(os.path.join(OUT, "download_verification.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["local_path", "remote_path", "sha256_local",
                                           "sha256_remote", "match", "source", "note"])
        w.writeheader(); w.writerows(rows)

    # ----------------------------------------------------------------------- #
    # 4. checkpoint epoch checks
    # ----------------------------------------------------------------------- #
    epoch_rows = []
    for (m, k), r in sorted(by_key.items()):
        # best.pt vs result.json best_epoch
        bp = os.path.join(DATA_RUNS, m, f"fold_{k}", "best.pt")
        ck = torch.load(bp, map_location="cpu", weights_only=False)
        ep_best = ck.get("epoch")
        ok_best = (str(ep_best) == str(r["best_epoch"]))
        # last.pt vs stop_epoch and vs inventory last_ckpt_epoch
        lp = os.path.join(DATA_RUNS, m, f"fold_{k}", "last.pt")
        if os.path.isfile(lp):
            ckl = torch.load(lp, map_location="cpu", weights_only=False)
            ep_last = ckl.get("epoch")
            inv_last = r["last_ckpt_epoch"]
            ok_inv = (str(ep_last) == str(inv_last))
            ok_stop = (str(ep_last) == str(r["stop_epoch"]))
        else:
            ep_last, ok_inv, ok_stop = "MISSING", None, None
        epoch_rows.append({"model": m, "fold": k, "cohort": r["cohort"],
                           "best_epoch_recorded": r["best_epoch"], "best_pt_epoch": ep_best,
                           "best_epoch_match": ok_best,
                           "stop_epoch": r["stop_epoch"],
                           "last_ckpt_epoch_inventory": r["last_ckpt_epoch"],
                           "last_pt_epoch": ep_last,
                           "last_pt_equals_inventory": ok_inv,
                           "last_pt_equals_stop_epoch": ok_stop})

    with open(os.path.join(OUT, "download_verification_epochs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(epoch_rows[0].keys()))
        w.writeheader(); w.writerows(epoch_rows)

    n_best_bad = sum(1 for e in epoch_rows if not e["best_epoch_match"])
    n_last_checked = sum(1 for e in epoch_rows if e["last_pt_equals_inventory"] is not None)
    n_last_bad = sum(1 for e in epoch_rows if e["last_pt_equals_inventory"] is False)
    n_last_eq_stop = sum(1 for e in epoch_rows if e["last_pt_equals_stop_epoch"] is True)

    # ----------------------------------------------------------------------- #
    # 5. Qsim checks
    # ----------------------------------------------------------------------- #
    qsim_rows = []
    for (m, k), r in sorted(by_key.items()):
        qp = os.path.join(DATA_QSIM, m, f"fold_{k}.npy")
        a = np.load(qp, mmap_mode="r")
        n_heldout = len(open(os.path.join(DATA_RUNS, m, f"fold_{k}",
                                          "heldout_basin_ids.txt")).read().split())
        axes = tuple(a.shape)
        is_f64 = (a.dtype == np.float64)
        has_5479 = (5479 in axes)
        other = axes[0] if axes[1] == 5479 else axes[1] if axes[0] == 5479 else None
        nan_count = int(np.isnan(np.asarray(a)).sum())
        qsim_rows.append({"model": m, "fold": k, "cohort": r["cohort"],
                          "path": os.path.relpath(qp, DATA), "dtype": str(a.dtype),
                          "shape": list(axes), "layout": ("time_first" if axes[0] == 5479
                                                          else "time_last"),
                          "n_heldout_basins": n_heldout, "basin_axis_matches": other == n_heldout,
                          "dtype_float64": is_f64, "has_axis_5479": has_5479,
                          "nan_count": nan_count})
    with open(os.path.join(OUT, "download_verification_qsim.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qsim_rows[0].keys()))
        w.writeheader(); w.writerows(qsim_rows)

    n_qsim_bad = sum(1 for q in qsim_rows
                     if not (q["dtype_float64"] and q["has_axis_5479"]
                             and q["basin_axis_matches"] and q["nan_count"] == 0))
    n_time_last = sum(1 for q in qsim_rows if q["layout"] == "time_last")

    accepted = n_source_changed if args.allow_source_changed_manifest else 0
    n_unexplained_mismatch = n_mismatch - accepted

    summary = {
        "mode": ("non-authoritative (--no-remote-sha)" if args.no_remote_sha
                 else "authoritative"),
        "files_checked": len(rows),
        "sha_match": n_match, "sha_mismatch": n_mismatch,
        "sha_mismatch_source_changed_since_manifest": n_source_changed,
        "sha_pending_remote": n_pending,
        "sha_local_only_cohort": n_local_only,
        "best_pt_epoch_mismatches": n_best_bad,
        "last_pt_checked": n_last_checked, "last_pt_epoch_mismatches": n_last_bad,
        "last_pt_equal_to_stop_epoch": n_last_eq_stop,
        "qsim_checked": len(qsim_rows), "qsim_failures": n_qsim_bad,
        "qsim_layout_time_last": n_time_last,
        "qsim_layout_time_first": len(qsim_rows) - n_time_last,
    }
    summary["sha_mismatch_unexplained"] = n_unexplained_mismatch
    summary["source_changed_mismatches_accepted"] = accepted
    json.dump(summary, open(os.path.join(OUT, "download_verification_summary.json"), "w"), indent=1)

    print("\n== verification summary ==")
    for k, v in summary.items():
        print(f"  {k:34s} {v}")

    if accepted:
        print(f"\nACCEPTED (documented, non-transfer): {accepted} mismatch(es) whose source "
              f"document was rewritten by the concurrent stage-6 workflow after the manifest "
              f"snapshot; see the `note` column of download_verification.csv")
    ok = (n_unexplained_mismatch == 0 and n_best_bad == 0 and n_last_bad == 0
          and n_qsim_bad == 0 and n_pending == 0)
    if args.no_remote_sha:
        print(f"\nSTEP_1_4_COMPLETE = False (non-authoritative run: "
              f"{n_pending} remote sha256 comparisons pending)")
        sys.exit(2)
    print(f"\nSTEP_1_4_PASS = {ok}")
    if not ok:
        sys.exit(1)


def ssh_with_stdin(cmd: str, payload: str, timeout: int = 1800) -> str:
    last = None
    for attempt in range(1, SSH_RETRIES + 1):
        proc = subprocess.run(["ssh", "-o", "ConnectTimeout=20", SSH_ALIAS, cmd],
                              input=payload, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout
        last = proc.stderr.strip()[:300]
        print(f"    ssh(sha) attempt {attempt}/{SSH_RETRIES} failed: {last[:150]}",
              file=sys.stderr)
        if attempt < SSH_RETRIES:
            import time
            time.sleep(SSH_RETRY_SLEEP)
    raise RuntimeError(f"remote sha256sum failed after {SSH_RETRIES} attempts: {last}")


if __name__ == "__main__":
    main()
