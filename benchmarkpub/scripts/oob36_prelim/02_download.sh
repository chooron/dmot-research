#!/usr/bin/env bash
# 02_download.sh — step 1.3 of ch4-oob36-download-prelim-20260930
#
# Downloads the OOB-36 results into
#     $DATA/runs/<model>/fold_<k>/{best.pt,last.pt,result.json,epoch_metrics.csv,
#                                  best_metadata.json,training_health.json,
#                                  heldout_basin_ids.txt,heldout_basin_kge.csv}
#     $DATA/qsim/<model>/fold_<k>.npy
#     $DATA/obs/fold_<k>/heldout_y.npy
#     $DATA/manifests/...
#
# X99 is READ-ONLY: only `rsync` (never --delete) and read-only `ssh` are used.
# Local-only cohorts (existing8) and the local-only artefacts (v4_new8 Qsim, the
# report/manifest files that exist only on this machine) are staged with `cp`
# from the frozen tree — no download involved, originals untouched.
#
# The file lists come from $OUT/remote_inventory.json (produced by
# 01_remote_inventory.py out of the frozen manifests); no path is guessed.
# Every manifest candidate is probed on X99 first, and falls back to the local
# frozen tree when X99 does not carry it.

set -euo pipefail

ROOT=/home/jingxin/code/dmg-research/project/benchmarkpub
DATA=$ROOT/data/oob36_20260930
OUT=$ROOT/analysis/chapter4_oob36_prelim_20260930
BENCH=/home/jingxin/code/dmg-research/project/benchmark
AUDIT_QSIM=$BENCH/analysis/chapter4_oob_pipeline_audit_20260926/evidence/qsim

SSH_ALIAS=remote-v100
REMOTE_ROOT=/home/wd500/jingxin/dmg-research
R_V4_RUNS=$REMOTE_ROOT/project/benchmark/results/oob_v4_16models_20260925/runs
R_V5_RUNS=$REMOTE_ROOT/project/benchmark/results/oob_v5_36models_20260926/runs
R_V5_QSIM=$REMOTE_ROOT/project/benchmark/analysis/chapter4_oob36_20260926/evidence/qsim
R_OBS=$REMOTE_ROOT/project/benchmark/results/oob_v4_16models_20260925/data_cache
R_OOB36=$REMOTE_ROOT/project/benchmark/analysis/chapter4_oob36_20260926

L_OOB36=$BENCH/analysis/chapter4_oob36_20260926
L_AUDIT=$BENCH/analysis/chapter4_oob_pipeline_audit_20260926

PY=/home/jingxin/code/dmg-research/.venv/bin/python

# rsync over a flaky link: 3 attempts, resumable.  Never --delete.
rsync_retry() {
  local n=0
  until rsync -a --partial --timeout=120 "$@"; do
    n=$((n+1))
    if [ "$n" -ge 3 ]; then
      echo "FATAL: rsync failed after 3 attempts: $*" >&2
      return 1
    fi
    echo "  rsync attempt $n failed; retrying in 10s ..." >&2
    sleep 10
  done
}

mkdir -p "$DATA"/{runs,qsim,obs,manifests} "$DATA/_lists"

if [ ! -f "$OUT/remote_inventory.json" ]; then
  echo "FATAL: $OUT/remote_inventory.json missing — run 01_remote_inventory.py first" >&2
  exit 2
fi

# --------------------------------------------------------------------------- #
# 1. file lists
# --------------------------------------------------------------------------- #
echo "== building file lists =="
"$PY" - "$OUT/remote_inventory.json" "$DATA/_lists" "$DATA" <<'PY'
import json, os, sys
inv = json.load(open(sys.argv[1])); lst = sys.argv[2]; DATA = sys.argv[3]
PER_RUN = ["best.pt", "result.json", "epoch_metrics.csv", "best_metadata.json",
           "training_health.json", "heldout_basin_ids.txt", "heldout_basin_kge.csv"]
v4, v5, qsim_v5, snap = [], [], [], []
for r in inv:
    m, k, c = r["model"], r["fold"], r["cohort"]
    if c == "existing8":
        continue
    rel = f"{m}/fold_{k}"
    for f in PER_RUN:
        (v4 if c == "v4_new8" else v5).append(f"{rel}/{f}")
    if c == "v5_new20":
        # idempotent: an already-staged Qsim is not re-fetched
        if not os.path.isfile(os.path.join(DATA, "qsim", m, f"fold_{k}.npy")):
            qsim_v5.append(f"oob36__{m}__fold{k}__qsim.npy")
        # idempotent: a snapshot already renamed to last.pt is not re-fetched
        if (r["last_ckpt_path"] != "MISSING"
                and not os.path.isfile(os.path.join(DATA, "runs", m, f"fold_{k}", "last.pt"))):
            snap.append((f"{rel}/{os.path.basename(r['last_ckpt_path'])}", m, k))
obs = [f"fold_{k}/heldout_y.npy" for k in range(5)]
def w(name, rows):
    open(os.path.join(lst, name), "w").write("\n".join(rows) + "\n")
    print(f"  {name}: {len(rows)}")
w("v4_runs.txt", v4); w("v5_runs.txt", v5); w("v5_qsim.txt", qsim_v5); w("obs.txt", obs)
with open(os.path.join(lst, "snapshots.tsv"), "w") as fh:
    for rel, m, k in snap:
        fh.write(f"{rel}\t{m}\t{k}\n")
print(f"  snapshots.tsv: {len(snap)}")
PY

# --------------------------------------------------------------------------- #
# 2. rsync remote file sets (resumable; already-present files are skipped)
# --------------------------------------------------------------------------- #
echo "== rsync v4_new8 runs (40 runs x 7 files) =="
rsync_retry --files-from="$DATA/_lists/v4_runs.txt" "$SSH_ALIAS:$R_V4_RUNS/" "$DATA/runs/"

echo "== rsync v5_new20 runs (100 runs x 7 files) =="
rsync_retry --files-from="$DATA/_lists/v5_runs.txt" "$SSH_ALIAS:$R_V5_RUNS/" "$DATA/runs/"

echo "== rsync v5_new20 snapshots =="
if [ -s "$DATA/_lists/snapshots.tsv" ]; then
  cut -f1 "$DATA/_lists/snapshots.tsv" > "$DATA/_lists/snapshots_paths.txt"
  rsync_retry --files-from="$DATA/_lists/snapshots_paths.txt" "$SSH_ALIAS:$R_V5_RUNS/" "$DATA/runs/"
fi

echo "== rsync v5_new20 Qsim -> staging (skips already-staged) =="
mkdir -p "$DATA/_qsim_stage"
if [ -s "$DATA/_lists/v5_qsim.txt" ]; then
  rsync_retry --files-from="$DATA/_lists/v5_qsim.txt" "$SSH_ALIAS:$R_V5_QSIM/" "$DATA/_qsim_stage/"
fi

echo "== rsync per-fold observations (5) =="
rsync_retry --files-from="$DATA/_lists/obs.txt" "$SSH_ALIAS:$R_OBS/" "$DATA/obs/"

# --------------------------------------------------------------------------- #
# 3. manifests: probe X99 first, fall back to the local frozen tree
# --------------------------------------------------------------------------- #
echo "== manifests: remote probe + rsync =="
"$PY" - "$R_OOB36" "$L_OOB36" "$L_AUDIT" "$DATA/_lists" <<'PY'
import os, sys, subprocess
R_OOB36, L_OOB36, L_AUDIT, LISTS = sys.argv[1:5]

# candidate relative paths under $R_OOB36 / $L_OOB36
CAND = [
 "OOB36_MODEL_LIST.csv", "OOB36_ARTIFACT_MANIFEST.csv", "OOB36_CONFIG_PARITY.csv",
 "OOB36_PREREGISTRATION.md", "OOB36_TRAINING_REPORT.md", "GATE_REPORT.md",
 "HUMAN_DECISIONS.json", "OPEN_QUESTIONS.md", "RECOVERY_SEMANTICS.md",
 "RUNNER_CHANGE_20260928_snapshot_resume.md", "RUN_LOG_OOB36.csv", "audit_summary.json",
 "OOB_FOLD_ASSIGNMENT.csv",
 "logs/ACTIVATION.json", "logs/BACKEND_FLAGS_POST_TRAINING.txt",
 "logs/OOB36_QSIM_MANIFEST.csv", "logs/RELAUNCH_AFTER_OUTAGE.json",
 "logs/S6_checks_1_2_3_4_6_7.json", "logs/S6_local_checks.json",
 "logs/S6_reinference_summary.json", "logs/s6_reinference.log",
 "logs/s6_reinference_gpu1.log", "logs/supervisor.log",
 "logs/supervisor_launch1_reboot_killed.log", "logs/watchdog.log",
 "logs/SUPERVISOR_STATUS.log", "logs/local_progress.csv", "logs/SCHEDULING.json",
 "tables/OOB36_REINFERENCE.csv", "tables/OOB36_TRAINING_SUMMARY.csv",
 "tables/OOB36_PER_MODEL_VALUES.json", "tables/OOB36_VALUES_v4_and_new20.json",
 "scripts/minimal_kge.py", "scripts/s6_checks.py", "scripts/s6_reference.py",
 "scripts/s6_reinference.py", "scripts/s6_local.py",
]
# probe existence on X99 in ONE ssh round-trip
probe = ("for f in " + " ".join(f"'{c}'" for c in CAND)
         + "; do [ -f \"" + R_OOB36 + "/$f\" ] && echo \"$f\"; done; exit 0")
out = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "remote-v100", probe],
                     capture_output=True, text=True, check=True).stdout.split()
remote_set = set(out)
local_only = [c for c in CAND if c not in remote_set]
open(os.path.join(LISTS, "manifests_remote.txt"), "w").write("\n".join(sorted(remote_set)) + "\n")
open(os.path.join(LISTS, "manifests_local_only.txt"), "w").write("\n".join(local_only) + "\n")
print(f"  exists on X99      : {len(remote_set)}")
print(f"  local-frozen only  : {len(local_only)}  -> {local_only}")
PY

rsync_retry --files-from="$DATA/_lists/manifests_remote.txt" "$SSH_ALIAS:$R_OOB36/" "$DATA/manifests/"

echo "== manifests: stage local-only files verbatim =="
"$PY" - "$L_OOB36" "$L_AUDIT" "$DATA/manifests" "$DATA/_lists" <<'PY'
import os, shutil, sys, hashlib, csv
L_OOB36, L_AUDIT, MAN, LISTS = sys.argv[1:5]
local_only = [l.strip() for l in open(os.path.join(LISTS, "manifests_local_only.txt")) if l.strip()]
rows = []
for rel in local_only:
    src = os.path.join(L_OOB36, rel)
    if not os.path.isfile(src):
        print(f"  (absent locally too: {rel})")
        continue
    dst = os.path.join(MAN, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    rows.append({"relative_path": rel, "source": "local_frozen:chapter4_oob36_20260926"})

# audit manifests (local only)
AUDIT = [
 "AUDIT_ARTIFACT_MANIFEST.csv", "logs/B1B2_summary.json",
 "tables/B1_reinference_parity.csv", "tables/B2_independent_kge.csv",
 "scripts/minimal_kge.py",
]
for rel in AUDIT:
    src = os.path.join(L_AUDIT, rel)
    if not os.path.isfile(src):
        print(f"  (absent locally: {rel})"); continue
    dst = os.path.join(MAN, "local_frozen_audit", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    rows.append({"relative_path": "local_frozen_audit/" + rel,
                 "source": "local_frozen:chapter4_oob_pipeline_audit_20260926"})

with open(os.path.join(MAN, "LOCAL_FROZEN_STAGED.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["relative_path", "source"]); w.writeheader(); w.writerows(rows)
print(f"  staged {len(rows)} local-only manifest files")
PY

echo "== frozen fold assignment =="
# The fold split lives under results/, not under the analysis dir, so it is fetched
# explicitly.  Frozen sha256 = fe9013bcd7155ea99ac85597e59ea5e417a13897dc08aaa69f2c63ddb6780085,
# the value recorded in RUN_LOG_OOB36.csv for every one of the 100 v5 runs.
rsync_retry \
  "$SSH_ALIAS:$REMOTE_ROOT/project/benchmark/results/oob_v5_36models_20260926/OOB_FOLD_ASSIGNMENT.csv" \
  "$DATA/manifests/OOB_FOLD_ASSIGNMENT.csv"
cp -p "$BENCH/results/oob_primary8_5fold_20260902/OOB_FOLD_ASSIGNMENT.csv" \
      "$DATA/manifests/OOB_FOLD_ASSIGNMENT.local_frozen.csv"
echo "  remote sha : $(sha256sum "$DATA/manifests/OOB_FOLD_ASSIGNMENT.csv" | cut -c1-16)"
echo "  local  sha : $(sha256sum "$DATA/manifests/OOB_FOLD_ASSIGNMENT.local_frozen.csv" | cut -c1-16)"

# --------------------------------------------------------------------------- #
# 4. stage local-only runs and Qsim (no download)
# --------------------------------------------------------------------------- #
echo "== stage local existing8 runs (40) =="
"$PY" - "$OUT/remote_inventory.json" "$DATA" <<'PY'
import json, os, shutil, sys
inv, DATA = json.load(open(sys.argv[1])), sys.argv[2]
PER_RUN = ["best.pt","result.json","epoch_metrics.csv","best_metadata.json",
           "training_health.json","heldout_basin_ids.txt","heldout_basin_kge.csv"]
n = 0
for r in inv:
    if r["cohort"] != "existing8":
        continue
    dst = os.path.join(DATA, "runs", r["model"], f"fold_{r['fold']}")
    os.makedirs(dst, exist_ok=True)
    for f in PER_RUN:
        s = os.path.join(r["run_dir"], f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, f)); n += 1
print(f"  staged {n} files")
PY

echo "== stage local Qsim for existing8 + v4_new8 (80) =="
"$PY" - "$OUT/remote_inventory.json" "$DATA" "$BENCH/analysis/chapter4_oob_pipeline_audit_20260926/evidence/qsim" <<'PY'
import json, os, shutil, sys
inv, DATA, SRC = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
n = 0
for r in inv:
    if r["cohort"] == "v5_new20":
        continue
    s = os.path.join(SRC, os.path.basename(r["qsim_path"]))
    dst = os.path.join(DATA, "qsim", r["model"])
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(s, os.path.join(dst, f"fold_{r['fold']}.npy")); n += 1
print(f"  staged {n} Qsim files")
PY

echo "== move staged v5 Qsim -> \$DATA/qsim/<model>/fold_<k>.npy =="
"$PY" - "$OUT/remote_inventory.json" "$DATA" <<'PY'
import json, os, shutil, sys
inv, DATA = json.load(open(sys.argv[1])), sys.argv[2]
stage = os.path.join(DATA, "_qsim_stage")
n = 0
for r in inv:
    if r["cohort"] != "v5_new20":
        continue
    s = os.path.join(stage, f"oob36__{r['model']}__fold{r['fold']}__qsim.npy")
    dst = os.path.join(DATA, "qsim", r["model"])
    os.makedirs(dst, exist_ok=True)
    shutil.move(s, os.path.join(dst, f"fold_{r['fold']}.npy")); n += 1
if os.path.isdir(stage) and not os.listdir(stage):
    os.rmdir(stage)
print(f"  moved {n} Qsim files")
PY

# --------------------------------------------------------------------------- #
# 5. rename snapshots -> last.pt; write rename_map.csv
# --------------------------------------------------------------------------- #
echo "== rename snapshots to last.pt =="
"$PY" - "$DATA" <<'PY'
import csv, os, sys
DATA = sys.argv[1]
rows = []
for line in open(os.path.join(DATA, "_lists", "snapshots.tsv")):
    rel, m, k = line.rstrip("\n").split("\t")
    src = os.path.join(DATA, "runs", rel)
    dst = os.path.join(DATA, "runs", m, f"fold_{k}", "last.pt")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"snapshot not downloaded: {src}")
    os.replace(src, dst)
    rows.append({"model": m, "fold": k, "remote_snapshot_path": rel,
                 "local_path": os.path.relpath(dst, DATA),
                 "renamed_from": os.path.basename(rel)})
with open(os.path.join(DATA, "rename_map.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model","fold","remote_snapshot_path","local_path","renamed_from"])
    w.writeheader(); w.writerows(rows)
print(f"  renamed {len(rows)} snapshots -> last.pt")
PY

rmdir "$DATA/_lists" 2>/dev/null || true

echo
echo "== download complete =="
echo "runs  : $(find "$DATA/runs" -type f | wc -l) files, $(find "$DATA/runs" -maxdepth 2 -type d -name 'fold_*' | wc -l) run dirs"
echo "qsim  : $(find "$DATA/qsim" -type f | wc -l) files"
echo "obs   : $(find "$DATA/obs" -type f | wc -l) files"
echo "manif : $(find "$DATA/manifests" -type f | wc -l) files"
du -sh "$DATA"
