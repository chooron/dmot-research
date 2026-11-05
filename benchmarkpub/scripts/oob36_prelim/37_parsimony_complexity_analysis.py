#!/usr/bin/env python3
"""37_parsimony_complexity_analysis.py — complexity vs performance / transfer / stability / adequacy.

READS (frozen, read-only)
-------------------------
  params/*.npz                                   script 36 (parameter forward only)
  pub_next_stage_20260930/tables/
      pub_same_pipeline_model_basin.csv          per model x basin seen/OOB KGE + ΔKGE_PUB
  pub_relaxed_adequacy_20260930/tables/
      relaxed_adequacy_model_basin.csv           per model x basin RELAXED_70 + signature fractions
  pub_parsimony_20260930/tables/
      model_complexity_inventory.csv             n_parameters / n_states (comparability flagged)

WRITES (new directory only)
---------------------------
  tables/model_complexity_inventory.csv
  tables/model_level_complexity_performance.csv
  tables/model_basin_parameter_stability.csv
  tables/parameter_level_stability.csv
  tables/complexity_vs_transfer.csv
  tables/complexity_vs_adequacy.csv
  tables/complexity_tradeoff_summary.csv
  tables/parameter_instability_vs_oob_loss.csv
  tables/parsimony_complexity.json

No training, no simulation, no KGE recomputation, no modification of frozen outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO = Path("/home/jingxin/code/dmg-research")
PUB = REPO / "project" / "benchmarkpub"
OUT1 = PUB / "analysis" / "chapter4_oob36_prelim_20260930"
OUT = OUT1 / "pub_parsimony_complexity_20260930"
TABLES = OUT / "tables"
PARAMS = OUT / "params"

N_BASINS, N_FOLDS = 531, 5
SEED = 20260930
N_PERM = 10000

NEAR_BOUND = 0.01      # theta_norm in (0,0.01) or (0.99,1)  -> "at the bound"
NEAR_BOUND_SOFT = 0.05  # theta_norm in (0,0.05) or (0.95,1)  -> "near the bound"

SEEN_CSV = OUT1 / "pub_next_stage_20260930" / "tables" / "pub_same_pipeline_model_basin.csv"
ADQ_CSV = OUT1 / "pub_relaxed_adequacy_20260930" / "tables" / "relaxed_adequacy_model_basin.csv"
INV_CSV = OUT1 / "pub_parsimony_20260930" / "tables" / "model_complexity_inventory.csv"


# --------------------------------------------------------------------------- stats helpers
def spearman_perm(x, y, n_perm: int = N_PERM, seed: int = SEED):
    """Spearman rho + analytic p + permutation p (pairwise-permutation, two-sided)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, np.nan, n
    rho, p_analytic = sstats.spearmanr(x, y)
    if not np.isfinite(rho):
        return np.nan, np.nan, np.nan, n
    rx = sstats.rankdata(x).astype(float)
    ry = sstats.rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    rx /= np.sqrt((rx ** 2).sum())
    rng = np.random.default_rng(seed)
    idx = rng.random((n_perm, n)).argsort(axis=1)
    ryp = ry[idx]
    ryp /= np.sqrt((ryp ** 2).sum(axis=1, keepdims=True))
    perm_rho = ryp @ rx
    p_perm = (np.count_nonzero(np.abs(perm_rho) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)
    return float(rho), float(p_analytic), float(p_perm), int(n)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.all(x[m] == x[m][0]) or np.all(y[m] == y[m][0]):
        return np.nan, np.nan
    r, p = sstats.pearsonr(x[m], y[m])
    return float(r), float(p)


def tertile_band(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"]).astype(str)


def pareto_front(df: pd.DataFrame, maximize: str, minimize: str) -> np.ndarray:
    """Boolean mask of non-dominated rows (>= max on `maximize`, <= min on `minimize`)."""
    a = df[maximize].to_numpy(float)
    b = df[minimize].to_numpy(float)
    n = len(df)
    keep = np.ones(n, bool)
    for i in range(n):
        dominated = (a >= a[i]) & (b <= b[i]) & ((a > a[i]) | (b < b[i]))
        if dominated.any():
            keep[i] = False
    return keep


# --------------------------------------------------------------------------- stability core
def stability_for_model(npz_path: Path) -> dict:
    """Return per-basin stability arrays for one model (all vectorised over 531 basins)."""
    z = np.load(npz_path, allow_pickle=False)
    basin_ids = z["basin_ids"]
    order = np.argsort(basin_ids)
    basin_ids = basin_ids[order]
    # stored [fold, basin, P] -> [basin, fold, P], sorted by basin_id
    T = np.transpose(z["theta_norm"], (1, 0, 2))[order]
    hf = z["heldout_fold"][order]
    names = [str(s) for s in z["parameter_names"]]
    bounds = z["bounds"]                     # [P, 2]
    n_basins, n_folds, P = T.shape
    assert n_basins == N_BASINS and n_folds == N_FOLDS
    b_idx = np.arange(n_basins)
    seen = np.empty((n_basins, N_FOLDS - 1, P), dtype=np.float64)
    for i in b_idx:
        seen[i] = np.delete(T[i], hf[i], axis=0)
    oob = T[b_idx, hf, :]                    # [531, P]

    # fold-to-fold variability over the 4 seen folds
    sd = seen.std(axis=1, ddof=1)            # [531, P]
    med = np.median(seen, axis=1)
    mad = np.median(np.abs(seen - med[:, None, :]), axis=1)  # [531, P]

    pairs = [(i, j) for i in range(N_FOLDS - 1) for j in range(i + 1, N_FOLDS - 1)]
    pr_rms_param = np.empty((n_basins, len(pairs)))
    pr_euclid = np.empty((n_basins, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        d = seen[:, i, :] - seen[:, j, :]
        pr_rms_param[:, k] = np.sqrt(np.mean(d ** 2, axis=1))
        pr_euclid[:, k] = np.sqrt(np.sum(d ** 2, axis=1))

    seen_mean = seen.mean(axis=1)
    seen_median = np.median(seen, axis=1)
    d_mean = seen_mean - oob
    d_median = seen_median - oob

    near_seen = ((seen < NEAR_BOUND) | (seen > 1 - NEAR_BOUND)).mean(axis=(1, 2))
    near_seen_soft = ((seen < NEAR_BOUND_SOFT) | (seen > 1 - NEAR_BOUND_SOFT)).mean(axis=(1, 2))
    near_oob = ((oob < NEAR_BOUND) | (oob > 1 - NEAR_BOUND)).mean(axis=1)
    near_oob_soft = ((oob < NEAR_BOUND_SOFT) | (oob > 1 - NEAR_BOUND_SOFT)).mean(axis=1)

    # per-parameter boundary-hit frequency over all (basin, fold) cells (for the param-level table)
    all_cells = T.reshape(-1, P)
    param_hit = ((all_cells < NEAR_BOUND) | (all_cells > 1 - NEAR_BOUND)).mean(axis=0)
    param_hit_soft = ((all_cells < NEAR_BOUND_SOFT) | (all_cells > 1 - NEAR_BOUND_SOFT)).mean(axis=0)
    seen_flat = seen.reshape(-1, P)
    param_sd_across_folds_mean = sd.mean(axis=0)
    # per-parameter seen->OOB displacement magnitude, averaged over basins
    param_disp_abs_mean = np.abs(d_mean).mean(axis=0)
    param_disp_rms = np.sqrt((d_mean ** 2).mean(axis=0))

    return {
        "basin_ids": basin_ids,
        "heldout_fold": hf,
        "P": P,
        "names": names,
        "bounds": bounds,
        "basin": {
            "fold_sd_mean_param": sd.mean(axis=1),
            "fold_sd_max_param": sd.max(axis=1),
            "fold_sd_rms_param": np.sqrt((sd ** 2).mean(axis=1)),
            "fold_mad_mean_param": mad.mean(axis=1),
            "fold_pairwise_rms_mean_param": pr_rms_param.mean(axis=1),
            "fold_pairwise_euclid_mean": pr_euclid.mean(axis=1),
            "disp_rms_mean_seen": np.sqrt(np.mean(d_mean ** 2, axis=1)),
            "disp_euclid_mean_seen": np.sqrt(np.sum(d_mean ** 2, axis=1)),
            "disp_rms_median_seen": np.sqrt(np.mean(d_median ** 2, axis=1)),
            "disp_euclid_median_seen": np.sqrt(np.sum(d_median ** 2, axis=1)),
            "frac_seen_near_bound": near_seen,
            "frac_seen_near_bound_soft": near_seen_soft,
            "frac_oob_near_bound": near_oob,
            "frac_oob_near_bound_soft": near_oob_soft,
        },
        "param": {
            "param_hit_rate": param_hit,
            "param_hit_rate_soft": param_hit_soft,
            "param_fold_sd_mean_basin": param_sd_across_folds_mean,
            "param_disp_abs_mean_basin": param_disp_abs_mean,
            "param_disp_rms_basin": param_disp_rms,
            "param_seen_mean_norm": seen_flat.mean(axis=0),
        },
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)

    seen = pd.read_csv(SEEN_CSV)
    adq = pd.read_csv(ADQ_CSV)
    inv = pd.read_csv(INV_CSV)
    inv = inv[["model", "n_parameters", "n_states"]].copy()

    dfb = seen[["model", "basin_id", "heldout_fold", "kge_heldout",
                "kge_seenfolds_median4", "kge_seenfolds_mean4",
                "delta_kge_pub_median4", "delta_kge_pub_mean4"]].copy()
    adq_cols = ["model", "basin_id", "seen_RELAXED_70", "oob_RELAXED_70",
                "signature_pass_fraction_seen", "signature_pass_fraction_oob",
                "median_abs_signature_pct_seen", "median_abs_signature_pct_oob"]
    dfb = dfb.merge(adq[adq_cols], on=["model", "basin_id"], how="left", validate="one_to_one")
    if dfb.isna().any().any():
        raise RuntimeError("merged model x basin table has NaN")
    if len(dfb) != 36 * N_BASINS:
        raise RuntimeError(f"expected {36 * N_BASINS} rows, got {len(dfb)}")
    dfb = dfb.sort_values(["model", "basin_id"]).reset_index(drop=True)

    # ---------------- per-basin stability, organised by model order used in dfb
    basin_frames, param_rows, model_extra = [], [], []
    for model, g in dfb.groupby("model", sort=True):
        st = stability_for_model(PARAMS / f"{model}.npz")
        g = g.sort_values("basin_id").reset_index(drop=True)
        if not np.array_equal(g["basin_id"].to_numpy(), st["basin_ids"]) or \
           not np.array_equal(g["heldout_fold"].to_numpy(), st["heldout_fold"]):
            raise RuntimeError(f"{model}: parameter/performance basin order mismatch")
        rec = {"model": model, "basin_id": st["basin_ids"], "heldout_fold": st["heldout_fold"],
               "n_parameters": st["P"]}
        rec.update(st["basin"])
        basin_frames.append(pd.DataFrame(rec))
        for i, name in enumerate(st["names"]):
            param_rows.append({
                "model": model,
                "parameter_name": name,
                "parameter_index": i,
                "n_parameters": st["P"],
                "lower": float(st["bounds"][i, 0]),
                "upper": float(st["bounds"][i, 1]),
                "bound_span": float(st["bounds"][i, 1] - st["bounds"][i, 0]),
                "fold_sd_mean_basin": float(st["param"]["param_fold_sd_mean_basin"][i]),
                "seen_to_oob_disp_abs_mean_basin": float(st["param"]["param_disp_abs_mean_basin"][i]),
                "seen_to_oob_disp_rms_basin": float(st["param"]["param_disp_rms_basin"][i]),
                "boundary_hit_rate": float(st["param"]["param_hit_rate"][i]),
                "boundary_hit_rate_soft": float(st["param"]["param_hit_rate_soft"][i]),
                "mean_seen_normalized_value": float(st["param"]["param_seen_mean_norm"][i]),
            })
        model_extra.append({
            "model": model,
            "n_parameters": st["P"],
            # hard boundary rate is computed above from the basin table; keep the soft variant here
            "param_boundary_rate_mean_soft": float(st["param"]["param_hit_rate_soft"].mean()),
        })
    stab = pd.concat(basin_frames, ignore_index=True)
    paramstab = pd.DataFrame(param_rows).sort_values(["model", "parameter_index"]).reset_index(drop=True)
    model_extra = pd.DataFrame(model_extra)

    mb = dfb.merge(stab, on=["model", "basin_id", "heldout_fold"], how="left", validate="one_to_one")
    if len(mb) != 36 * N_BASINS or mb.isna().any().any():
        raise RuntimeError("model x basin stability merge failed")
    mb.to_csv(TABLES / "model_basin_parameter_stability.csv", index=False)
    paramstab.to_csv(TABLES / "parameter_level_stability.csv", index=False)

    # ---------------- complexity inventory
    inventory = inv.copy()
    inventory["complexity_band"] = tertile_band(inventory["n_parameters"])
    inventory["n_parameters_comparability"] = "COMPARABLE"
    inventory["n_states_comparability"] = "COMPARABLE"
    inventory["primary_complexity_metric"] = "n_parameters"
    inventory = inventory.sort_values("model").reset_index(drop=True)
    inventory.to_csv(TABLES / "model_complexity_inventory.csv", index=False)

    # ---------------- model-level table
    rows = []
    for model, g in mb.groupby("model", sort=True):
        n_p = int(g["n_parameters"].iloc[0])
        rows.append({
            "model": model,
            "n_parameters": n_p,
            "n_states": int(inv.loc[inv.model == model, "n_states"].iloc[0]),
            "complexity_band": inventory.loc[inventory.model == model, "complexity_band"].iloc[0],
            "seen_kge_aggregate_mean": float(g["kge_seenfolds_median4"].mean()),
            "seen_kge_aggregate_mean4": float(g["kge_seenfolds_mean4"].mean()),
            "seen_kge_median": float(g["kge_seenfolds_median4"].median()),
            "oob_kge_aggregate_mean": float(g["kge_heldout"].mean()),
            "oob_kge_median": float(g["kge_heldout"].median()),
            "delta_kge_pub_median": float(g["delta_kge_pub_median4"].median()),
            "delta_kge_pub_mean": float(g["delta_kge_pub_median4"].mean()),
            "delta_kge_pub_q75": float(g["delta_kge_pub_median4"].quantile(0.75)),
            "delta_kge_pub_q90": float(g["delta_kge_pub_median4"].quantile(0.90)),
            "frac_delta_gt_0p05": float((g["delta_kge_pub_median4"] > 0.05).mean()),
            "frac_delta_gt_0p10": float((g["delta_kge_pub_median4"] > 0.10).mean()),
            "frac_delta_positive": float((g["delta_kge_pub_median4"] > 0).mean()),
            "param_fold_instability_median": float(g["fold_pairwise_rms_mean_param"].median()),
            "param_fold_instability_mean": float(g["fold_pairwise_rms_mean_param"].mean()),
            "param_fold_sd_median": float(g["fold_sd_mean_param"].median()),
            "param_displacement_median": float(g["disp_rms_mean_seen"].median()),
            "param_displacement_mean": float(g["disp_rms_mean_seen"].mean()),
            "param_displacement_euclid_median": float(g["disp_euclid_mean_seen"].median()),
            "param_displacement_euclid_mean": float(g["disp_euclid_mean_seen"].mean()),
            "param_boundary_rate_mean": float(g["frac_seen_near_bound"].mean()),
            "param_boundary_rate_oob_mean": float(g["frac_oob_near_bound"].mean()),
            "adequacy_relaxed70_seen_rate": float(g["seen_RELAXED_70"].mean()),
            "adequacy_relaxed70_oob_rate": float(g["oob_RELAXED_70"].mean()),
            "signature_pass_fraction_seen_mean": float(g["signature_pass_fraction_seen"].mean()),
            "signature_pass_fraction_oob_mean": float(g["signature_pass_fraction_oob"].mean()),
            "median_abs_signature_pct_seen": float(g["median_abs_signature_pct_seen"].median()),
            "median_abs_signature_pct_oob": float(g["median_abs_signature_pct_oob"].median()),
        })
    model = pd.DataFrame(rows)
    model = model.merge(model_extra, on=["model", "n_parameters"], how="left", validate="one_to_one")
    model = model.sort_values("model").reset_index(drop=True)
    model.to_csv(TABLES / "model_level_complexity_performance.csv", index=False)

    # ---------------- complexity vs performance / transfer relationships
    def rel(xname, yname, x, y):
        rho, pa, pp, n = spearman_perm(x, y)
        r, pr = pearson(x, y)
        return {
            "x_quantity": xname, "y_quantity": yname, "n_models": n,
            "spearman_rho": rho, "spearman_p_analytic": pa, "spearman_p_permutation": pp,
            "pearson_r": r, "pearson_p": pr,
            "direction": "positive" if rho > 0 else ("negative" if rho < 0 else "flat"),
        }

    perf_rels = [
        rel("n_parameters", "seen_kge_aggregate_mean", model.n_parameters, model.seen_kge_aggregate_mean),
        rel("n_parameters", "oob_kge_aggregate_mean", model.n_parameters, model.oob_kge_aggregate_mean),
        rel("n_parameters", "seen_kge_median", model.n_parameters, model.seen_kge_median),
        rel("n_parameters", "oob_kge_median", model.n_parameters, model.oob_kge_median),
        rel("n_states", "seen_kge_aggregate_mean", model.n_states, model.seen_kge_aggregate_mean),
        rel("n_states", "oob_kge_aggregate_mean", model.n_states, model.oob_kge_aggregate_mean),
    ]
    transfer_rels = [
        rel("n_parameters", "delta_kge_pub_median", model.n_parameters, model.delta_kge_pub_median),
        rel("n_parameters", "delta_kge_pub_mean", model.n_parameters, model.delta_kge_pub_mean),
        rel("n_parameters", "delta_kge_pub_q75", model.n_parameters, model.delta_kge_pub_q75),
        rel("n_parameters", "delta_kge_pub_q90", model.n_parameters, model.delta_kge_pub_q90),
        rel("n_parameters", "frac_delta_gt_0p05", model.n_parameters, model.frac_delta_gt_0p05),
        rel("n_parameters", "frac_delta_gt_0p10", model.n_parameters, model.frac_delta_gt_0p10),
        rel("n_parameters", "param_fold_instability_median", model.n_parameters, model.param_fold_instability_median),
        rel("n_parameters", "param_displacement_median", model.n_parameters, model.param_displacement_median),
        rel("n_parameters", "param_displacement_euclid_median", model.n_parameters, model.param_displacement_euclid_median),
        rel("n_parameters", "param_boundary_rate_mean", model.n_parameters, model.param_boundary_rate_mean),
        rel("n_states", "delta_kge_pub_median", model.n_states, model.delta_kge_pub_median),
    ]
    adequacy_rels = [
        rel("n_parameters", "adequacy_relaxed70_seen_rate", model.n_parameters, model.adequacy_relaxed70_seen_rate),
        rel("n_parameters", "adequacy_relaxed70_oob_rate", model.n_parameters, model.adequacy_relaxed70_oob_rate),
        rel("n_parameters", "signature_pass_fraction_seen_mean", model.n_parameters, model.signature_pass_fraction_seen_mean),
        rel("n_parameters", "signature_pass_fraction_oob_mean", model.n_parameters, model.signature_pass_fraction_oob_mean),
        rel("n_parameters", "median_abs_signature_pct_seen", model.n_parameters, model.median_abs_signature_pct_seen),
        rel("n_parameters", "median_abs_signature_pct_oob", model.n_parameters, model.median_abs_signature_pct_oob),
        rel("n_states", "adequacy_relaxed70_oob_rate", model.n_states, model.adequacy_relaxed70_oob_rate),
    ]

    # descriptive complexity-band means for every relationship target
    band_means = {}
    for name in ["seen_kge_aggregate_mean", "oob_kge_aggregate_mean", "seen_kge_median", "oob_kge_median",
                 "delta_kge_pub_median", "delta_kge_pub_mean", "delta_kge_pub_q75", "delta_kge_pub_q90",
                 "frac_delta_gt_0p05", "frac_delta_gt_0p10", "param_fold_instability_median",
                 "param_displacement_median", "param_displacement_euclid_median", "param_boundary_rate_mean",
                 "adequacy_relaxed70_seen_rate", "adequacy_relaxed70_oob_rate",
                 "signature_pass_fraction_seen_mean", "signature_pass_fraction_oob_mean",
                 "median_abs_signature_pct_seen", "median_abs_signature_pct_oob"]:
        bm = model.groupby("complexity_band")[name].mean()
        band_means[name] = {f"band_{k}_mean": float(v) for k, v in bm.items()}

    def with_bands(rels):
        out = []
        for r in rels:
            out.append({**r, **band_means.get(r["y_quantity"], {})})
        return pd.DataFrame(out)

    transfer_df = with_bands(transfer_rels)
    transfer_df.to_csv(TABLES / "complexity_vs_transfer.csv", index=False)
    adequacy_df = with_bands(adequacy_rels)
    adequacy_df.to_csv(TABLES / "complexity_vs_adequacy.csv", index=False)

    # also store the four headline performance relations in the json (no dedicated required table)
    performance_rels_df = with_bands(perf_rels)
    performance_rels_df.to_csv(TABLES / "complexity_vs_performance.csv", index=False)

    # ---------------- parameter instability vs OOB loss (§7)
    pool_rows = []
    targets = [("param_displacement_seen_mean_rms", "disp_rms_mean_seen"),
               ("param_displacement_seen_euclid", "disp_euclid_mean_seen"),
               ("param_displacement_seen_median_rms", "disp_rms_median_seen"),
               ("param_fold_variability_pairwise_rms", "fold_pairwise_rms_mean_param"),
               ("param_fold_variability_sd", "fold_sd_mean_param"),
               ("param_boundary_proximity_seen", "frac_seen_near_bound")]
    for label, col in targets:
        rho, pa, pp, n = spearman_perm(mb[col], mb["delta_kge_pub_median4"])
        r, pr = pearson(mb[col], mb["delta_kge_pub_median4"])
        within = []
        for m, g in mb.groupby("model", sort=True):
            if g[col].nunique() < 3:
                continue
            rw, pw = sstats.spearmanr(g[col], g["delta_kge_pub_median4"])
            within.append({"model": m, "rho": rw, "p": pw})
        wd = pd.DataFrame(within)
        mmed = model.set_index("model")
        model_series = mb.groupby("model")[col].median().reindex(mmed.index)
        mod_rho, mod_pa, mod_pp, mod_n = spearman_perm(
            model_series,
            mmed["delta_kge_pub_median"])
        pool_rows.append({
            "instability_metric": label,
            "column": col,
            "pooled_spearman_rho": rho, "pooled_spearman_p_permutation": pp, "pooled_n": n,
            "pooled_pearson_r": r,
            "within_model_spearman_median": float(wd["rho"].median()),
            "within_model_spearman_frac_negative": float((wd["rho"] < 0).mean()),
            "within_model_spearman_frac_p_lt_005": float((wd["p"] < 0.05).mean()),
            "within_models_used": int(len(wd)),
            "model_median_spearman_rho": mod_rho, "model_median_spearman_p_permutation": mod_pp,
            "model_median_n": mod_n,
        })
    pool_df = pd.DataFrame(pool_rows)
    pool_df.to_csv(TABLES / "parameter_instability_vs_oob_loss.csv", index=False)

    # ---------------- trade-off summary + Pareto flags
    trade = model[[
        "model", "n_parameters", "n_states", "complexity_band",
        "seen_kge_aggregate_mean", "oob_kge_aggregate_mean", "oob_kge_median",
        "delta_kge_pub_median", "frac_delta_gt_0p05",
        "adequacy_relaxed70_seen_rate", "adequacy_relaxed70_oob_rate",
        "signature_pass_fraction_oob_mean", "median_abs_signature_pct_oob",
        "param_fold_instability_median", "param_displacement_median",
        "param_displacement_euclid_median", "param_boundary_rate_mean",
    ]].copy()
    trade["oob_kge_per_parameter"] = trade["oob_kge_aggregate_mean"] / trade["n_parameters"]
    trade["pareto_kge_vs_nparam"] = pareto_front(trade, "oob_kge_aggregate_mean", "n_parameters")
    trade["pareto_adequacy_vs_nparam"] = pareto_front(trade, "adequacy_relaxed70_oob_rate", "n_parameters")
    trade["pareto_seenkge_vs_nparam"] = pareto_front(trade, "seen_kge_aggregate_mean", "n_parameters")
    # a "cheap" flag: above-median OOB KGE and below-median n_parameters
    med_kge = trade["oob_kge_aggregate_mean"].median()
    med_np = trade["n_parameters"].median()
    trade["high_value_low_complexity"] = (trade["oob_kge_aggregate_mean"] >= med_kge) & \
                                         (trade["n_parameters"] <= med_np)
    trade = trade.sort_values("n_parameters").reset_index(drop=True)
    trade.to_csv(TABLES / "complexity_tradeoff_summary.csv", index=False)

    # ---------------- json bundle
    band_table = (model.groupby("complexity_band")
                  [["n_parameters", "seen_kge_aggregate_mean", "oob_kge_aggregate_mean", "delta_kge_pub_median",
                    "adequacy_relaxed70_seen_rate", "adequacy_relaxed70_oob_rate",
                    "param_fold_instability_median", "param_displacement_median",
                    "param_displacement_euclid_median", "param_boundary_rate_mean"]]
                  .mean().round(6).to_dict(orient="index"))
    band_counts = model.groupby("complexity_band")["model"].apply(list).to_dict()

    # upper-performance envelope: how many high-complexity models sit in the top-k OOB
    oob_sorted = model.sort_values("oob_kge_aggregate_mean", ascending=False)
    top_k = {k: oob_sorted.head(k)["complexity_band"].value_counts().to_dict() for k in (5, 9, 12)}
    param_sorted = model.sort_values("n_parameters", ascending=False)
    high12 = set(param_sorted.head(12)["model"])

    bundle = {
        "task": "ch4-oob36-parsimony-complexity",
        "n_models": int(len(model)),
        "n_basins": N_BASINS,
        "n_folds": N_FOLDS,
        "primary_complexity_metric": "n_parameters",
        "secondary_complexity_metric": "n_states",
        "not_comparable_metrics": ["n_fluxes", "n_process_modules"],
        "stability_space": "theta_norm = (theta_phys - lower) / (upper - lower) from frozen best.pt",
        "near_bound_definition": NEAR_BOUND,
        "n_permutations": N_PERM,
        "seed": SEED,
        "performance_relationships": performance_rels_df.to_dict(orient="records"),
        "complexity_band_means": band_table,
        "complexity_band_members": band_counts,
        "oob_top_k_band_composition": top_k,
        "high_complexity_models_top12_by_nparams": sorted(high12),
        "pareto_kge_vs_nparam": trade.loc[trade.pareto_kge_vs_nparam, ["model", "n_parameters", "oob_kge_aggregate_mean"]].to_dict(orient="records"),
        "pareto_adequacy_vs_nparam": trade.loc[trade.pareto_adequacy_vs_nparam, ["model", "n_parameters", "adequacy_relaxed70_oob_rate"]].to_dict(orient="records"),
        "high_value_low_complexity_models": trade.loc[trade.high_value_low_complexity, "model"].tolist(),
        "n_models_positive_delta": int((model.delta_kge_pub_median > 0).sum()),
        "median_delta_over_models": float(model.delta_kge_pub_median.median()),
        "extraction_summary": json.loads((TABLES / "extraction_summary.json").read_text()),
    }
    (TABLES / "parsimony_complexity.json").write_text(json.dumps(bundle, indent=1, default=str) + "\n", encoding="utf-8")

    # ---------------- console summary
    print("model-level relationships with n_parameters (n=36):")
    for r in performance_rels_df.to_dict(orient="records"):
        print(f"  {r['y_quantity']:32s} rho={r['spearman_rho']:+.3f}  p_perm={r['spearman_p_permutation']:.4f}  "
              f"pearson={r['pearson_r']:+.3f}")
    print("\ntransfer relationships:")
    for r in transfer_df.to_dict(orient="records"):
        print(f"  {r['y_quantity']:34s} rho={r['spearman_rho']:+.3f}  p_perm={r['spearman_p_permutation']:.4f}")
    print("\nadequacy relationships:")
    for r in adequacy_df.to_dict(orient="records"):
        print(f"  {r['y_quantity']:36s} rho={r['spearman_rho']:+.3f}  p_perm={r['spearman_p_permutation']:.4f}")
    print("\ninstability vs delta:")
    print(pool_df[["instability_metric", "pooled_spearman_rho", "within_model_spearman_median",
                   "within_model_spearman_frac_negative", "model_median_spearman_rho"]].to_string(index=False))
    print("\nbands:")
    print(model.groupby("complexity_band")[["n_parameters", "oob_kge_aggregate_mean", "delta_kge_pub_median",
                                            "adequacy_relaxed70_oob_rate", "param_fold_instability_median",
                                            "param_displacement_median"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
