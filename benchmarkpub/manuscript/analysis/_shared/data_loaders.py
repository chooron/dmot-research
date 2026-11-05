"""Canonical data loaders for manuscript analyses."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .paths import (
    DATA_ROOT,
    DIR_C4_EQUIV,
    DIR_EQUIV_T1T2,
    DIR_GAP_CONDITIONAL,
    DIR_OOB36_DATA,
    DIR_OOB36_NEXT_STAGE,
    FRAC_SNOW_IDX,
    KGE_EPS,
    MODELS_36,
    N_BOOT_BLOCK,
    SEED_BOOT,
    SNOW_FRAC_THRESHOLD,
    TOTAL_TEST_DAYS,
    WATER_YEAR_DAYS,
    WATER_YEARS,
)

C4_EQUIV_COMMON = Path("/home/jingxin/code/dmg-research/project/benchmark/analysis/chapter4_equivalence_recommendability_20260925/scripts/common.py")

# Load c4_common by explicit path
_spec = importlib.util.spec_from_file_location("c4_common_shared", C4_EQUIV_COMMON)
_c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c4)


def canonical_ids() -> List[int]:
    """Return the 531 canonical basin IDs from 531sub_id.txt as a list of integers."""
    text = (DATA_ROOT / "531sub_id.txt").read_text(encoding="utf-8").strip()
    return [int(x) for x in ast.literal_eval(text)]


def basin_order() -> List[str]:
    """Return the 531 canonical basin IDs sorted numerically as 8-character zero-padded strings."""
    return [str(b).zfill(8) for b in sorted(canonical_ids())]


def load_oob_folds() -> Dict[str, int]:
    """Return mapping of 8-character zero-padded basin_id -> heldout_fold (0..4)."""
    fa = pd.read_csv(DIR_OOB36_DATA / "manifests" / "OOB_FOLD_ASSIGNMENT.csv")
    return {str(b).zfill(8): int(f) for b, f in zip(fa.basin_id, fa.fold)}


def load_attributes(zscore: bool = False) -> np.ndarray:
    """Return [531, 35] array of Caravan physical attributes aligned to basin_order()."""
    return _c4.load_attributes(zscore=zscore)


def load_snow() -> Dict[str, bool]:
    """Return mapping of model -> bool indicating presence of explicit snow routine."""
    df = pd.read_csv(DIR_GAP_CONDITIONAL / "tables" / "model_snow_annotation.csv")
    return dict(zip(df.model, df.snow_module.astype(bool)))


def get_snow_allow_mask(raw_attributes: np.ndarray, models: List[str] = MODELS_36) -> np.ndarray:
    """Return boolean mask [531, len(models)] where False means model without snow is disallowed."""
    snow_map = load_snow()
    snowy = raw_attributes[:, FRAC_SNOW_IDX] >= SNOW_FRAC_THRESHOLD
    allow = np.ones((len(raw_attributes), len(models)), dtype=bool)
    for j, m in enumerate(models):
        if not snow_map[m]:
            allow[snowy, j] = False
    return allow


def load_same_pipeline_table() -> pd.DataFrame:
    """Return pub_same_pipeline_model_basin.csv as a DataFrame."""
    return pd.read_csv(DIR_OOB36_NEXT_STAGE / "tables" / "pub_same_pipeline_model_basin.csv")


def load_canonical_seen_kge_matrix() -> Tuple[List[str], List[str], np.ndarray]:
    """Return (models, basins, KGE_matrix [531, 36]) for canonical Seen-dPL test performance."""
    models, basins, years, ic, dp = _c4.stats_cube("test")
    seen = _c4.kge_from_sufficient(dp.sum(axis=2)).T  # [531, 36]
    return models, basins, seen


def load_oob_kge_matrix() -> Tuple[List[str], List[str], np.ndarray]:
    """Return (models, basins, KGE_matrix [531, 36]) for OOB-dPL held-out performance."""
    pub = load_same_pipeline_table()
    models = sorted(pub["model"].unique())
    basins = basin_order()
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)
    oob = (pub.pivot(index="bs", columns="model", values="kge_heldout")
           .reindex(index=basins, columns=models).to_numpy(np.float64))
    return models, basins, oob


def load_oob_cube() -> np.ndarray:
    """Load cached [36, 531, 15, 6] OOB sufficient statistics cube."""
    cache = DIR_EQUIV_T1T2 / "cache" / "oob_cube.npy"
    if not cache.exists():
        raise FileNotFoundError(f"Expected cached OOB cube at {cache}")
    return np.load(cache)


def load_year_counts(n_reps: int = N_BOOT_BLOCK, seed: int = SEED_BOOT) -> np.ndarray:
    """Generate water-year block bootstrap counts [n_reps, 15] using 依据1 protocol."""
    return _c4.make_year_counts(n_reps, len(WATER_YEARS), seed)
