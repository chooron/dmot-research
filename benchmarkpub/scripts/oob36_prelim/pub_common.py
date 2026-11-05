#!/usr/bin/env python3
"""pub_common.py — shared builder for the same-pipeline PUB penalty table.

Delta_KGE_PUB_{m,b} = aggregate over the 4 folds j != k(b) of KGE^seenFold_{m,j,b}
                      - KGE^heldout_{m,k(b),b}

where k(b) is the fold in which basin b is held out.  The seen side comes from
`tables/pub_insample_seen_kge.csv` (script 11, frozen best.pt, inference only); the
held-out side is the frozen `kge_oob` column of the phase-2 merged table (validated
in phase 2 to 9.99e-15 against the frozen `heldout_basin_kge.csv`).

Positive = the basin is modelled worse when it was withheld from training.

Both aggregations are returned; the task requires reporting mean-of-4 and median-of-4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jingxin/code/dmg-research")
PUB = REPO / "project" / "benchmarkpub"
OUT1 = PUB / "analysis" / "chapter4_oob36_prelim_20260930"
OUT2 = OUT1 / "pub_next_stage_20260930"
T2 = OUT2 / "tables"
MERGED = OUT1 / "tables" / "seen_ic_oob_merged.parquet"
INSAMPLE = T2 / "pub_insample_seen_kge.csv"

N_MODELS, N_BASINS, N_FOLDS = 36, 531, 5
N_CELLS = N_MODELS * N_FOLDS


def insample_status() -> dict:
    if not INSAMPLE.exists():
        return {"available": False, "reason": f"{INSAMPLE.name} does not exist yet",
                "cells": 0}
    d = pd.read_csv(INSAMPLE, usecols=["model", "fold"])
    cells = d.groupby(["model", "fold"]).ngroups
    return {"available": cells == N_CELLS, "cells": int(cells), "required": N_CELLS,
            "reason": "" if cells == N_CELLS else f"only {cells}/{N_CELLS} cells present"}


def load_merged() -> pd.DataFrame:
    df = pd.read_parquet(MERGED)
    if len(df) != N_MODELS * N_BASINS:
        raise ValueError(f"merged table has {len(df)} rows, expected {N_MODELS*N_BASINS}")
    return df


def build_pub(agg: str = "median") -> pd.DataFrame:
    """Return model, basin_id, fold, kge_seenfolds, kge_heldout, delta_kge_pub, n_seen_folds."""
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    st = insample_status()
    if not st["available"]:
        raise RuntimeError(f"in-sample table unavailable: {st['reason']}")
    ins = pd.read_csv(INSAMPLE)
    m = load_merged()[["model", "gauge_id", "fold", "kge_oob"]].rename(
        columns={"gauge_id": "basin_id", "fold": "heldout_fold", "kge_oob": "kge_heldout"})

    g = ins.groupby(["model", "basin_id"])["kge_insample"]
    seen = g.agg(agg).rename("kge_seenfolds").reset_index()
    n = g.size().rename("n_seen_folds").reset_index()
    seen = seen.merge(n, on=["model", "basin_id"])

    out = m.merge(seen, on=["model", "basin_id"], how="left", validate="one_to_one")
    if out["kge_seenfolds"].isna().any():
        raise RuntimeError("some (model,basin) rows have no seen-fold aggregate")
    if not (out["n_seen_folds"] == N_FOLDS - 1).all():
        raise RuntimeError("seen-fold aggregate does not use exactly 4 folds")
    # leakage guard: the held-out fold must not contribute an in-sample value
    leak = ins.merge(m[["model", "basin_id", "heldout_fold"]], on=["model", "basin_id"])
    if (leak["fold"] == leak["heldout_fold"]).any():
        raise RuntimeError("leakage: the held-out fold contributed an in-sample value")
    out["delta_kge_pub"] = out["kge_seenfolds"] - out["kge_heldout"]
    return out.sort_values(["model", "basin_id"]).reset_index(drop=True)


def build_existing() -> pd.DataFrame:
    """The cross-pipeline penalty used in the previous stage, for numeric comparison."""
    m = load_merged()
    return pd.DataFrame({
        "model": m["model"], "basin_id": m["gauge_id"], "heldout_fold": m["fold"],
        "kge_heldout": m["kge_oob"], "kge_seendpl_canonical": m["kge_seen_dpl"],
        "delta_kge_existing": m["kge_seen_dpl"] - m["kge_oob"],
    }).sort_values(["model", "basin_id"]).reset_index(drop=True)


def basin_penalty(pub: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-basin summary of a basin x model penalty table."""
    g = pub.groupby("basin_id")[value_col]
    out = pd.DataFrame({
        "basin_id": g.median().index,
        "penalty_median_over_models": g.median().to_numpy(),
        "penalty_mean_over_models": g.mean().to_numpy(),
        "penalty_q25_over_models": g.quantile(0.25).to_numpy(),
        "penalty_q75_over_models": g.quantile(0.75).to_numpy(),
        "penalty_sd_over_models": g.std(ddof=1).to_numpy(),
        "n_models": g.size().to_numpy(),
    })
    frac = pub.assign(pos=pub[value_col] > 0).groupby("basin_id")["pos"].mean()
    out["frac_models_positive"] = out["basin_id"].map(frac)
    out["gauge_id_str"] = [str(int(b)).zfill(8) for b in out["basin_id"]]
    return out


# --------------------------------------------------------------------------- #
# supplemental helpers (added for the rank-stability / extreme-basin analysis)
# --------------------------------------------------------------------------- #

OBS_DIR = PUB / "data" / "oob36_20260930" / "obs"
RUNS_DIR = PUB / "data" / "oob36_20260930" / "runs"
EXISTING8_RUNS = REPO / "project" / "benchmark" / "results" / "oob_primary8_5fold_20260902" / "runs"


def fold_basin_ids(fold: int) -> np.ndarray:
    """Held-out basin ids of a fold, in the order used by the frozen obs/qsim arrays.

    Verified in phase 2 and again in the next-stage analysis: recomputing KGE from
    (qsim, obs) reproduces each run's frozen `heldout_basin_kge.csv` to ~1e-14, so the
    column order of `obs/fold_<k>/heldout_y.npy` is this id order.
    """
    inv = pd.read_csv(OUT1 / "remote_inventory.csv")
    for cand in ([RUNS_DIR / m / f"fold_{fold}" / "heldout_basin_ids.txt"
                  for m in sorted(inv.model.unique())]
                 + [EXISTING8_RUNS / m / f"fold_{fold}" / "heldout_basin_ids.txt"
                    for m in sorted(inv.model.unique())]):
        if cand.exists():
            return np.array([int(x) for x in cand.read_text().split()], dtype=np.int64)
    raise RuntimeError(f"no heldout_basin_ids.txt found for fold {fold}")


def load_obs_full(ids: np.ndarray) -> np.ndarray:
    """Assemble [5479, n_ids] observed mm/day, ID-joined across the 5 fold files."""
    out = np.full((5479, len(ids)), np.nan, dtype=np.float64)
    pos = {int(b): i for i, b in enumerate(ids.tolist())}
    for k in range(5):
        fold_ids = fold_basin_ids(k)
        y = np.load(OBS_DIR / f"fold_{k}" / "heldout_y.npy")
        if y.shape[1] != fold_ids.size:
            raise RuntimeError(f"fold {k}: obs has {y.shape[1]} columns, ids {fold_ids.size}")
        for j, b in enumerate(fold_ids.tolist()):
            out[:, pos[b]] = y[:, j]
    if np.isnan(out).all(axis=0).any():
        raise RuntimeError("at least one basin has no observed series")
    return out


def seen_oob_matrices(seen_col: str = "kge_seenfolds_median4",
                      oob_col: str = "kge_heldout"):
    """Return (seen, oob) DataFrames indexed by basin_id with 36 model columns."""
    pub = pd.read_parquet(MERGED)[["model", "gauge_id", "fold"]].rename(
        columns={"gauge_id": "basin_id", "fold": "heldout_fold"})
    full = pd.read_csv(OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")
    need = {"model", "basin_id", "heldout_fold", seen_col, oob_col}
    missing = need - set(full.columns)
    if missing:
        raise RuntimeError(f"pub_same_pipeline_model_basin.csv lacks {sorted(missing)}")
    seen = full.pivot(index="basin_id", columns="model", values=seen_col)
    oob = full.pivot(index="basin_id", columns="model", values=oob_col)
    models = sorted(seen.columns)
    basins = np.sort(pub["basin_id"].unique())
    seen, oob = seen.loc[basins, models], oob.loc[basins, models]
    if seen.isna().any().any() or oob.isna().any().any():
        raise RuntimeError("seen/oob matrix contains NaN")
    return seen, oob


def basin_fold_map() -> dict:
    d = pd.read_parquet(MERGED)[["gauge_id", "fold"]].drop_duplicates()
    return dict(zip(d["gauge_id"].astype(int), d["fold"].astype(int)))


# supplemental analysis directory (read-only w.r.t. all frozen and next-stage outputs)
OUT3 = OUT1 / "pub_supplemental_rank_extreme_20260930"
T3 = OUT3 / "tables"
F3 = OUT3 / "figures"


# candidate-set transfer analysis directory
OUT4 = OUT1 / "pub_candidate_set_transfer_20260930"
T4 = OUT4 / "tables"
F4 = OUT4 / "figures"

# the 8 models that also have independent multi-seed retrainings (seed 42/123/2026)
SEED8 = ["alpine2", "hbv96", "xinanjiang", "newzealand2", "ihacres", "us1", "mopex4", "hillslope"]
SEED_DIRS = {
    42: REPO / "project" / "benchmark" / "results" / "dpl_canonical_v2_20260831" / "runs",
    123: REPO / "project" / "benchmark" / "manuscript" / "r3"
         / "dpl_multiseed_sensitivity_20260914" / "runs",
    2026: REPO / "project" / "benchmark" / "manuscript" / "r3"
          / "dpl_multiseed_sensitivity_20260914" / "runs",
}


def load_seed_matrix(seeds=(42, 123, 2026)):
    """(model, seed) -> per-basin test KGE for the 8 models with multi-seed runs.

    Returns (DataFrame indexed by basin_id with (model, seed) columns, list of missing paths).
    These are the canonical dPL v2 runs (all 531 basins trained); seed 42 files were verified
    in the next-stage analysis to equal the frozen `kge_seen_dpl` column to 4.99e-11.
    """
    cols, missing = {}, []
    for model in SEED8:
        for s in seeds:
            d = SEED_DIRS[s] / (f"{model}_seed_{s}" if s != 42 else model)
            f = d / "basin_test_kge.csv"
            if not f.exists():
                missing.append(str(f))
                continue
            t = pd.read_csv(f)
            idc = t.columns[0]
            kc = "kge" if "kge" in t.columns else t.columns[-1]
            cols[(model, s)] = pd.Series(t[kc].to_numpy(float),
                                         index=t[idc].astype(np.int64).to_numpy())
    if not cols:
        return pd.DataFrame(), missing
    return pd.DataFrame(cols).sort_index(), missing


def topk_sets(scores: pd.DataFrame, k: int) -> dict:
    """basin_id -> set of the k highest-scoring models (stable order, no tie ambiguity)."""
    arr = scores.to_numpy(np.float64)
    models = np.asarray(scores.columns)
    out = {}
    for i, b in enumerate(scores.index):
        idx = np.argsort(-arr[i], kind="stable")[:k]
        out[int(b)] = set(models[idx].tolist())
    return out


def coverage_and_regret(sets: dict, oob: pd.DataFrame, k: int):
    """Per-basin OOB coverage of the OOB top-1, top-3/top-5 capture, and OOB regret."""
    arr = oob.to_numpy(np.float64)
    models = np.asarray(oob.columns)
    oracle = arr.max(axis=1)
    order = np.argsort(-arr, axis=1, kind="stable")
    best1 = order[:, 0]
    top3 = [set(models[order[i, :3]].tolist()) for i in range(arr.shape[0])]
    top5 = [set(models[order[i, :5]].tolist()) for i in range(arr.shape[0])]
    rows = []
    for i, b in enumerate(oob.index):
        S = sets[int(b)]
        cols = [list(models).index(m) for m in S]
        best_in_set = float(arr[i, cols].max())
        rows.append({
            "basin_id": int(b),
            "covers_oob_top1": bool(models[best1[i]] in S),
            "hits_oob_top3_any": bool(len(S & top3[i]) > 0),
            "frac_oob_top3_covered": len(S & top3[i]) / 3,
            "hits_oob_top5_any": bool(len(S & top5[i]) > 0),
            "frac_oob_top5_covered": len(S & top5[i]) / 5,
            "oracle_kge": float(oracle[i]),
            "best_in_set_kge": best_in_set,
            "regret": float(oracle[i] - best_in_set),
        })
    return pd.DataFrame(rows).set_index("basin_id")


# minimal-model-library analysis directory
OUT5 = OUT1 / "pub_minimal_model_library_20260930"
T5 = OUT5 / "tables"


# hydrological-adequacy analysis directory
OUT6 = OUT1 / "pub_adequacy_seen_oob_20260930"
T6 = OUT6 / "tables"
F6 = OUT6 / "figures"

# frozen daily-Q stores (read-only) used by the chapter-4 signature work
Q_IC_STORE = PUB / "cache" / "phase05_gpu_reconstruction" / "daily_q"
Q_DP_STORE = PUB / "analysis" / "analysis_seen_dpl_daily_q_reconstruction_20260924" / "artifacts" / "daily_q"

# frozen OOB36 download root (Qsim, obs, run dirs)
DATA_STORE = PUB / "data" / "oob36_20260930"


# relaxed-adequacy analysis directory
OUT7 = OUT1 / "pub_relaxed_adequacy_20260930"
T7 = OUT7 / "tables"
F7 = OUT7 / "figures"


# parsimony analysis directory
OUT8 = OUT1 / "pub_parsimony_20260930"
T8 = OUT8 / "tables"
F8 = OUT8 / "figures"

# the frozen benchmark tree (read-only for every analysis here)
BENCH = REPO / "project" / "benchmark"
