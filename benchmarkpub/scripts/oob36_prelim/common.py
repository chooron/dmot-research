"""common.py — single place where every path constant, the basin-ID rule and the
frozen metric-function imports live.

Task: ch4-oob36-download-prelim-20260930
Import convention:  every other script does

    from common import *   # or explicit names

Nothing here computes anything; it only declares locations and re-exports the
frozen metric functions.  No path is hardcoded anywhere else.

Remote access
-------------
X99  = ssh alias ``remote-v100`` (hostname ``X99``), repo root ``/home/wd500/jingxin/dmg-research``.
X99 is READ-ONLY for this task: we only ever read it with ``ssh``/``rsync``.
"""

from __future__ import annotations

import os
import subprocess
import sys

# --------------------------------------------------------------------------- #
# 1. Local paths (defined ONCE)
# --------------------------------------------------------------------------- #

ROOT = "/home/jingxin/code/dmg-research/project/benchmarkpub"
DATA = os.path.join(ROOT, "data", "oob36_20260930")
OUT = os.path.join(ROOT, "analysis", "chapter4_oob36_prelim_20260930")
CODE = os.path.join(ROOT, "scripts", "oob36_prelim")

# frozen assets — READ ONLY, never written to
BENCH = "/home/jingxin/code/dmg-research/project/benchmark"
REPO = "/home/jingxin/code/dmg-research"

DATA_RUNS = os.path.join(DATA, "runs")
DATA_QSIM = os.path.join(DATA, "qsim")
DATA_OBS = os.path.join(DATA, "obs")
DATA_MANIFESTS = os.path.join(DATA, "manifests")

OUT_TABLES = os.path.join(OUT, "tables")
OUT_FIGURES = os.path.join(OUT, "figures")

# --------------------------------------------------------------------------- #
# 2. Remote (X99) constants
# --------------------------------------------------------------------------- #

SSH_ALIAS = "remote-v100"
REMOTE_ROOT = "/home/wd500/jingxin/dmg-research"

# --------------------------------------------------------------------------- #
# 3. Cohort layout
# --------------------------------------------------------------------------- #
# 36 models = 8 existing8 (local) + 8 v4_new8 (X99) + 20 v5_new20 (X99).
# Sources of truth:
#   * cohort membership  : project/benchmark/analysis/chapter4_oob36_20260926/OOB36_MODEL_LIST.csv
#   * existing8/v4_new8  : chapter4_oob_pipeline_audit_20260926/logs/B1B2_summary.json
#                          (cohorts named "existing8" and "v4_new8", 8x5 = 40 runs each)
#   * Qsim existing8/v4  : chapter4_oob_pipeline_audit_20260926/AUDIT_ARTIFACT_MANIFEST.csv
#                          (relative_path = evidence/qsim/<cohort>__<model>__fold<k>__qsim.npy)
#   * Qsim v5_new20      : chapter4_oob36_20260926/logs/OOB36_QSIM_MANIFEST.csv
#                          (path = evidence/qsim/oob36__<model>__fold<k>__qsim.npy)
#   * v5 run metadata    : chapter4_oob36_20260926/RUN_LOG_OOB36.csv

EXISTING8 = ("alpine2", "hbv96", "hillslope", "ihacres",
             "mopex4", "newzealand2", "us1", "xinanjiang")
V4_NEW8 = ("alpine1", "australia", "collie1", "gr4j",
           "mopex2", "mopex3", "mopex5", "simhyd")
V5_NEW20 = ("collie2", "collie3", "flexb", "flexi", "flexis", "gsfb", "hymod",
            "modhydrolog", "mopex1", "newzealand1", "penman", "plateau", "smar",
            "susannah1", "susannah2", "tank", "tcm", "topmodel", "vic", "wetland")
MODELS = EXISTING8 + V4_NEW8 + V5_NEW20          # 36
FOLDS = (0, 1, 2, 3, 4)                          # 5

# run-dir roots
RUN_ROOT_LOCAL_EXISTING8 = os.path.join(BENCH, "results", "oob_primary8_5fold_20260902", "runs")
RUN_ROOT_REMOTE_V4 = REMOTE_ROOT + "/project/benchmark/results/oob_v4_16models_20260925/runs"
RUN_ROOT_REMOTE_V5 = REMOTE_ROOT + "/project/benchmark/results/oob_v5_36models_20260926/runs"

# Qsim roots
QSIM_ROOT_LOCAL_AUDIT = os.path.join(
    BENCH, "analysis", "chapter4_oob_pipeline_audit_20260926", "evidence", "qsim")
QSIM_ROOT_REMOTE_V5 = (REMOTE_ROOT + "/project/benchmark/analysis/"
                       "chapter4_oob36_20260926/evidence/qsim")

# per-cohort Qsim filename prefix
QSIM_PREFIX = {"existing8": "existing8", "v4_new8": "v4_new8", "v5_new20": "oob36"}

# observations (shared across all cohorts, one file per fold)
OBS_REMOTE_DIR = REMOTE_ROOT + "/project/benchmark/results/oob_v4_16models_20260925/data_cache"

# frozen fold assignment
FOLD_ASSIGNMENT_REMOTE = (REMOTE_ROOT + "/project/benchmark/results/"
                          "oob_v5_36models_20260926/OOB_FOLD_ASSIGNMENT.csv")
FOLD_ASSIGNMENT_LOCAL_CANDIDATES = (
    os.path.join(BENCH, "results", "oob_primary8_5fold_20260902", "OOB_FOLD_ASSIGNMENT.csv"),
    os.path.join(BENCH, "analysis", "chapter4_oob36_20260926", "OOB_FOLD_ASSIGNMENT.csv"),
)

# frozen chapter-4 assets used by step 2.4
SEEN_DPL_KGE_DIR = os.path.join(REPO, "releases", "benchmark", "training_results", "dpl", "runs")
PAIRED_LONG_CSV = os.path.join(
    BENCH, "results", "ic_dpl_seenbasin_formal_20260901", "02_BASIN_PAIRED_KGE_LONG.csv")
FROZEN_EQUIV_DIR = os.path.join(BENCH, "analysis",
                                "chapter4_equivalence_recommendability_20260925")
ROBUSTNESS_DIR = os.path.join(BENCH, "analysis", "chapter4_robustness_supplement_20260925")
MULTICRIT_DIR = os.path.join(BENCH, "analysis", "chapter4_multicriteria_v3_20260925")
OOB36_DIR = os.path.join(BENCH, "analysis", "chapter4_oob36_20260926")

# --------------------------------------------------------------------------- #
# 4. Basin-ID rule (audit A4)
# --------------------------------------------------------------------------- #
# Internally: int64.  On export: gauge_id_str = zero-padded to 8 digits.
# All tables are joined on the integer ID, never on row position.

GAUGE_ID_WIDTH = 8


def gauge_id_str(gauge_id) -> str:
    """Canonical 8-digit zero-padded string form of an integer basin ID."""
    return str(int(gauge_id)).zfill(GAUGE_ID_WIDTH)


# --------------------------------------------------------------------------- #
# 5. Remote helpers (read-only)
# --------------------------------------------------------------------------- #

def ssh(cmd: str, *, check: bool = True, timeout: int = 600) -> str:
    """Run ``cmd`` on X99 over ssh and return stdout.  Never writes to X99."""
    full = ["ssh", "-o", "ConnectTimeout=20", SSH_ALIAS, cmd]
    proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ssh command failed (rc={proc.returncode}): {cmd[:200]}")
    return proc.stdout


def ssh_python(code: str, *, check: bool = True, timeout: int = 3600) -> str:
    """Run a python program on X99 via ``python -`` on stdin.

    No file is created on X99 (``-`` reads the program from stdin), so this stays
    read-only with respect to the remote filesystem.
    """
    remote = (f"{REMOTE_ROOT}/.venv/bin/python -")
    full = ["ssh", "-o", "ConnectTimeout=20", SSH_ALIAS, remote]
    proc = subprocess.run(full, input=code, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-8000:])
        raise RuntimeError(f"remote python failed (rc={proc.returncode})")
    return proc.stdout


# --------------------------------------------------------------------------- #
# 6. Bootstrap convention (stated once, used by 05)
# --------------------------------------------------------------------------- #

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260930
PERMUTATION_N = 10000
PERMUTATION_SEED = 20260930

EXPLORATORY_BANNER = "EXPLORATORY — not a pre-registered test"


# --------------------------------------------------------------------------- #
# 7. Frozen metric functions (step 2.2)
# --------------------------------------------------------------------------- #
# Filled in by 04_build_basin_table.py after metric_sources.md is produced.
# The import itself is done lazily there so that this module stays importable
# during step 1 (before metric_sources.md exists).
