"""Canonical paths, model lists, and frozen protocol constants for manuscript analyses."""
from __future__ import annotations

import os
from pathlib import Path

# Repository root
_here = Path(__file__).resolve()
REPO_ROOT = _here.parents[4] if (_here.parents[4] / "dmotpy").exists() else Path("/home/jingxin/code/dmg-research").resolve()

# Project roots
BENCHMARKPARAM = REPO_ROOT / "benchmarkparam" if (REPO_ROOT / "benchmarkparam").exists() else REPO_ROOT / "project" / "benchmark"
BENCHMARK = BENCHMARKPARAM
BENCHMARKPUB = REPO_ROOT / "benchmarkpub" if (REPO_ROOT / "benchmarkpub").exists() else REPO_ROOT / "project" / "benchmarkpub"
DATA_ROOT = REPO_ROOT / "data"

# Manuscript analysis root
MANUSCRIPT_ANALYSIS_ROOT = BENCHMARKPUB / "manuscript" / "analysis"

# Upstream frozen analysis directories
DIR_OOB36_PRELIM = BENCHMARKPUB / "analysis" / "chapter4_oob36_prelim_20260930"
DIR_OOB36_NEXT_STAGE = DIR_OOB36_PRELIM / "pub_next_stage_20260930"
DIR_OOB36_PAPER_LEVEL = DIR_OOB36_PRELIM / "paper_level_stats_20260930"
DIR_OOB36_SUPPLEMENTAL = DIR_OOB36_PRELIM / "pub_supplemental_rank_extreme_20260930"
DIR_OOB36_MINIMAL_LIB = DIR_OOB36_PRELIM / "pub_minimal_model_library_20260930"
DIR_OOB36_TRANSFER = DIR_OOB36_PRELIM / "pub_candidate_set_transfer_20260930"
DIR_OOB36_PARSIMONY = DIR_OOB36_PRELIM / "pub_parsimony_complexity_20260930"

DIR_OOB36_DATA = BENCHMARKPUB / "data" / "oob36_20260930"
DIR_EQUIV_T1T2 = BENCHMARKPUB / "analysis" / "equiv_cover_t1t2_20261003"
DIR_OOB36_GATE = BENCHMARKPUB / "analysis" / "oob36_gset_gate_20261001"
DIR_OOB36_PAIRED_REGRET = BENCHMARKPUB / "analysis" / "oob36_paired_regret_20261003"
DIR_CLUSTER_VALIDATION = BENCHMARKPUB / "analysis" / "cluster_oob_recommendation_validation_20261001"
DIR_CLUSTER_PROFILES = BENCHMARKPUB / "analysis" / "cluster_model_recommendation_20261001"

DIR_C4_EQUIV = BENCHMARK / "analysis" / "chapter4_equivalence_recommendability_20260925"
DIR_C4_FINAL_CLOSURE = BENCHMARK / "analysis" / "chapter4_final_closure_20260925"
DIR_C4_FINAL_SUPPLEMENT = BENCHMARK / "analysis" / "chapter4_final_supplement_20260925"
DIR_C4_ROBUSTNESS = BENCHMARK / "analysis" / "chapter4_robustness_supplement_20260925"
DIR_C4_TOPK = BENCHMARK / "analysis" / "topk_candidate_sets_20260925"
DIR_C4_MULTICRITERIA = BENCHMARK / "analysis" / "chapter4_multicriteria_v3_20260925"
DIR_C4_V31 = BENCHMARK / "analysis" / "chapter4_v31_parsimony_20260925"
DIR_GAP_CONDITIONAL = BENCHMARK / "analysis" / "ic_dpl_gap_conditional_20260923"
DIR_R3_MULTISEED = BENCHMARK / "manuscript" / "r3" / "dpl_multiseed_sensitivity_20260914"

# Model universe: canonical 36 models in alphabetical order
MODELS_36 = [
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96",
    "hillslope", "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2",
    "mopex3", "mopex4", "mopex5", "newzealand1", "newzealand2", "penman",
    "plateau", "simhyd", "smar", "susannah1", "susannah2", "tank",
    "tcm", "topmodel", "us1", "vic", "wetland", "xinanjiang",
]

# 8-model cohort
MODELS_8 = [
    "alpine2", "hbv96", "hillslope", "ihacres",
    "mopex4", "newzealand2", "us1", "xinanjiang",
]

# Protocol constants
KGE_EPS = 0.1
SNOW_FRAC_THRESHOLD = 0.30
FRAC_SNOW_IDX = 3

SEED_BOOT = 20260925
SEED_MAIN = 20260925
SEEDS_10 = [20260925 + i for i in range(10)]
N_BOOT_BLOCK = 1000
N_BOOT_BASIN = 10000

WATER_YEARS = list(range(1996, 2011))
WATER_YEAR_DAYS = [366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365]
TOTAL_TEST_DAYS = 5479
