#!/usr/bin/env python3
"""13_basin_attribute_screening.py — Analysis 4: small exploratory basin screening.

Question: which basin types show a larger PUB penalty?

This is a BOUNDED exploratory screen, not a mechanism study:
  * only attributes with a clear provenance already in the repository are used
    (`data/caravan_671_attributes.csv`, the canonical named 35-attribute Caravan
    matrix contract) plus direct summaries of the observed discharge series that the
    KGE is computed against (mean flow, zero-flow fraction, runoff ratio, variability);
  * each variable gets ONE Spearman correlation with the basin penalty plus a
    permutation p-value, all reported (BH-FDR over the small declared family);
  * grouped quartile summaries and an extreme-basin listing;
  * a missingness check.
NO machine learning, NO feature selection, NO multivariate model fishing, and NO
causal interpretation.

The KGE-instability distinction is addressed explicitly: a basin can show a large
penalty either because it is genuinely hard to generalise to, or because KGE itself is
unstable there (near-zero observed variance, long zero-flow runs).  The observed-flow
moments and the across-model KGE dispersion are reported alongside, and the extreme
basins are flagged rather than interpreted.

Outputs -> $OUT2/tables/basin_attribute_screening.csv and friends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

REPO = PC.REPO
DATA = REPO / "project" / "benchmarkpub" / "data" / "oob36_20260930"
ATTR_CSV = REPO / "data" / "caravan_671_attributes.csv"
N_PERM = 10000
SEED = 20260930

# (column, label, source) — small, physically interpretable, provenance-clear
ATTRS = [
    ("arid", "aridity index (PET/P)", "caravan_671_attributes"),
    ("f_snow", "snowfall fraction", "caravan_671_attributes"),
    ("p_mean", "mean precipitation (mm/day)", "caravan_671_attributes"),
    ("pet_mean", "mean PET (mm/day)", "caravan_671_attributes"),
    ("p_seas", "precipitation seasonality", "caravan_671_attributes"),
    ("elev", "mean elevation (m)", "caravan_671_attributes"),
    ("area", "catchment area (km2)", "caravan_671_attributes"),
    ("slope", "mean slope", "caravan_671_attributes"),
    ("f_forest", "forest fraction", "caravan_671_attributes"),
    ("s_cond", "soil conductivity", "caravan_671_attributes"),
    ("q_mean", "mean observed discharge (mm/day)", "observed series"),
    ("runoff_ratio", "runoff ratio q_mean/p_mean", "observed series + attribute"),
    ("zero_flow_frac", "fraction of days with zero discharge", "observed series"),
    ("q_cv", "coefficient of variation of discharge", "observed series"),
    ("q_log_sd", "SD of log(Q + eps)", "observed series"),
]


def log(m: str) -> None:
    print(m, flush=True)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n, float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        r = n - rank + 1
        val = min(prev, p[idx] * n / r)
        q[idx] = val
        prev = val
    return q


def perm_spearman(x, y, n_perm: int = N_PERM, seed: int = SEED):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    rx = sstats.rankdata(x); ry = sstats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    rho = float(np.dot(rx, ry) / n)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(float(np.dot(rx, rng.permutation(ry)) / n)) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1), n


def load_obs_full(ids: np.ndarray) -> np.ndarray:
    """Assemble [5479, 531] observed mm/day, ID-joined across the 5 fold files."""
    out = np.full((5479, len(ids)), np.nan, dtype=np.float64)
    pos = {int(b): i for i, b in enumerate(ids.tolist())}
    inv = pd.read_csv(REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930"
                             "/remote_inventory.csv")
    for k in range(5):
        # any run dir of fold k gives the fold's held-out basin order
        cand = [DATA / "runs" / m / f"fold_{k}" / "heldout_basin_ids.txt" for m in sorted(inv.model.unique())]
        cand += [REPO / "project/benchmark/results/oob_primary8_5fold_20260902/runs" / m / f"fold_{k}"
                 / "heldout_basin_ids.txt" for m in sorted(inv.model.unique())]
        src = next((c for c in cand if c.exists()), None)
        if src is None:
            raise RuntimeError(f"no heldout_basin_ids.txt for fold {k}")
        fold_ids = np.array([int(x) for x in src.read_text().split()], dtype=np.int64)
        y = np.load(DATA / "obs" / f"fold_{k}" / "heldout_y.npy")
        if y.shape[1] != fold_ids.size:
            raise RuntimeError(f"fold {k}: obs has {y.shape[1]} columns, ids {fold_ids.size}")
        for j, b in enumerate(fold_ids.tolist()):
            out[:, pos[b]] = y[:, j]
    if np.isnan(out).all(axis=0).any():
        raise RuntimeError("at least one basin has no observed series")
    return out


def main() -> None:
    PC.T2.mkdir(parents=True, exist_ok=True)
    merged = PC.load_merged()
    ids = np.sort(merged["gauge_id"].unique())
    report = {"meta": {
        "seed": SEED, "n_perm": N_PERM, "n_basins": int(len(ids)),
        "attribute_source": "data/caravan_671_attributes.csv (canonical named Caravan contract, "
                            "531/531 coverage verified)",
        "observed_source": "data/oob36_20260930/obs/fold_<k>/heldout_y.npy, column order verified "
                           "against heldout_basin_ids.txt to ~1e-14 via re-derived KGE",
        "family_size": len(ATTRS),
        "scope": "bounded exploratory screening; no ML, no feature selection, no causal claim",
    }}

    # ---- basin penalty, both definitions -----------------------------------
    existing = PC.build_existing()
    top = PC.basin_penalty(existing, "delta_kge_existing")
    colmap = {"basin_id": "basin_id", "gauge_id_str": "gauge_id_str",
              "penalty_median_over_models": "penalty_median_existing",
              "penalty_sd_over_models": "penalty_sd_existing",
              "frac_models_positive": "frac_models_positive_existing"}
    BAS = top[list(colmap)].rename(columns=colmap)

    st = PC.insample_status()
    pub_available = st["available"]
    if pub_available:
        pub = PC.build_pub("median")
        b2 = PC.basin_penalty(pub, "delta_kge_pub")
        cm2 = {"basin_id": "basin_id", "penalty_median_over_models": "penalty_median_pub",
               "penalty_sd_over_models": "penalty_sd_pub",
               "frac_models_positive": "frac_models_positive_pub"}
        BAS = BAS.merge(b2[list(cm2)].rename(columns=cm2), on="basin_id", how="left")
        primary = "penalty_median_pub"
    else:
        primary = "penalty_median_existing"
    report["penalty_definition_used"] = {
        "primary": primary,
        "same_pipeline_available": pub_available,
        "reason_if_not": "" if pub_available else st["reason"],
    }

    # ---- basin attributes ---------------------------------------------------
    a = pd.read_csv(ATTR_CSV)
    a["basin_id"] = a["gauge_id"].astype(np.int64)
    BAS = BAS.merge(a[["basin_id"] + [c for c, _, _ in ATTRS if _ == "caravan_671_attributes"]],
                    on="basin_id", how="left", validate="one_to_one")

    obs = load_obs_full(ids)
    pos = {int(b): i for i, b in enumerate(ids.tolist())}
    order = np.array([pos[int(b)] for b in BAS["basin_id"]])
    O = obs[:, order]
    with np.errstate(invalid="ignore"):
        q_mean = np.nanmean(O, axis=0)
        q_sd = np.nanstd(O, axis=0, ddof=1)
        q_cv = q_sd / q_mean
        eps = 0.01 * q_mean
        q_log_sd = np.nanstd(np.log(np.maximum(O, 0) + eps[None, :]), axis=0, ddof=1)
        zero_frac = np.nanmean(O <= 0.0, axis=0)
    BAS["q_mean"] = q_mean
    BAS["q_cv"] = q_cv
    BAS["q_log_sd"] = q_log_sd
    BAS["zero_flow_frac"] = zero_frac
    BAS["runoff_ratio"] = BAS["q_mean"] / BAS["p_mean"]
    BAS["q_sd"] = q_sd
    BAS["obs_mean"] = q_mean
    BAS["obs_var"] = q_sd ** 2

    # across-model dispersion of KGE on that basin (poolability / metric sensitivity)
    kg = merged.groupby("gauge_id")["kge_oob"].agg(["median", "std"]).rename(
        columns={"median": "oob_kge_median", "std": "oob_kge_sd_over_models"})
    kg.index.name = "basin_id"
    BAS = BAS.merge(kg.reset_index(), on="basin_id", how="left")

    # ---- missingness check -------------------------------------------------
    miss = BAS[["basin_id"] + [c for c, _, _ in ATTRS]].isna().sum()
    report["missingness"] = {k: int(v) for k, v in miss.items()}

    # ---- correlations ------------------------------------------------------
    y = BAS[primary].to_numpy(float)
    rows = []
    for col, label, src in ATTRS:
        rho, p, n = perm_spearman(BAS[col].to_numpy(float), y)
        rows.append({"variable": col, "label": label, "source": src,
                     "spearman_rho": rho, "perm_p": p, "n_used": n})
    S = pd.DataFrame(rows)
    S["bh_fdr_q"] = bh_fdr(S["perm_p"].to_numpy())
    S["abs_rho"] = S["spearman_rho"].abs()
    S = S.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    S.to_csv(PC.T2 / "basin_attribute_screening.csv", index=False)
    report["screening"] = S.to_dict(orient="records")

    # ---- grouped quartile summaries ---------------------------------------
    groups = []
    for col, label, _ in ATTRS:
        v = BAS[col].to_numpy(float)
        ok = np.isfinite(v)
        try:
            q = pd.qcut(pd.Series(v[ok]), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        except ValueError:
            continue
        tmp = pd.DataFrame({"g": q.to_numpy(), "pen": y[ok]})
        g = tmp.groupby("g", observed=True)["pen"].agg(["median", "mean", "size"])
        for lvl, r in g.iterrows():
            groups.append({"variable": col, "label": label, "quartile": str(lvl),
                           "penalty_median": float(r["median"]), "penalty_mean": float(r["mean"]),
                           "n_basins": int(r["size"])})
    pd.DataFrame(groups).to_csv(PC.T2 / "basin_attribute_quartiles.csv", index=False)

    # ---- extreme basins ----------------------------------------------------
    BAS["penalty_primary"] = BAS[primary]
    ext = BAS.sort_values("penalty_primary", ascending=False)
    cols = ["gauge_id_str", "penalty_primary", "penalty_median_existing", "frac_models_positive_existing",
            "arid", "f_snow", "p_mean", "runoff_ratio", "q_mean", "zero_flow_frac", "q_cv",
            "q_log_sd", "elev", "area", "oob_kge_median", "oob_kge_sd_over_models"]
    if pub_available:
        cols.insert(3, "penalty_median_pub")
    ext = ext[[c for c in cols if c in ext.columns]]
    ext.head(40).to_csv(PC.T2 / "basin_extreme_high_penalty.csv", index=False)
    ext.tail(40).iloc[::-1].to_csv(PC.T2 / "basin_extreme_low_penalty.csv", index=False)

    dec = BAS["penalty_primary"] >= BAS["penalty_primary"].quantile(0.9)
    rest = ~dec
    def cmp(col):
        A = BAS.loc[dec, col].to_numpy(float); B = BAS.loc[rest, col].to_numpy(float)
        A = A[np.isfinite(A)]; B = B[np.isfinite(B)]
        if A.size < 3 or B.size < 3:
            return None, None, None
        return float(np.median(A)), float(np.median(B)), float(
            sstats.mannwhitneyu(A, B, alternative="two-sided").pvalue)
    extreme_cmp = {}
    for col in ["zero_flow_frac", "q_mean", "q_cv", "q_log_sd", "arid", "f_snow", "runoff_ratio",
                "obs_var", "oob_kge_sd_over_models", "area", "elev"]:
        a_, b_, p_ = cmp(col)
        extreme_cmp[col] = {"median_top_decile_penalty": a_, "median_rest": b_,
                           "mannwhitney_p": p_}
    report["top_decile_penalty_vs_rest"] = extreme_cmp

    # KGE-instability flags, reported not interpreted
    top50 = BAS.nlargest(50, "penalty_primary")
    flags = {
        "n_top50_penalty_basins": int(len(top50)),
        "frac_top50_zero_flow_frac_gt_0p1": float((top50["zero_flow_frac"] > 0.1).mean()),
        "frac_top50_zero_flow_frac_gt_0p25": float((top50["zero_flow_frac"] > 0.25).mean()),
        "frac_top50_q_mean_lt_0p5mm": float((top50["q_mean"] < 0.5).mean()),
        "frac_top50_q_log_sd_lt_0p5": float((top50["q_log_sd"] < 0.5).mean()),
        "frac_top50_oob_kge_median_lt_0": float((top50["oob_kge_median"] < 0).mean()),
        "frac_all_basins_zero_flow_frac_gt_0p1": float((BAS["zero_flow_frac"] > 0.1).mean()),
        "frac_all_basins_oob_kge_median_lt_0": float((BAS["oob_kge_median"] < 0).mean()),
        "median_oob_kge_sd_all_basins": float(BAS["oob_kge_sd_over_models"].median()),
        "median_oob_kge_sd_top50": float(top50["oob_kge_sd_over_models"].median()),
    }
    report["kge_instability_flags"] = flags

    BAS.to_csv(PC.T2 / "basin_attribute_table.csv", index=False)
    with open(PC.T2 / "basin_screening.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    log("== basin penalty used ==")
    log(f"   primary = {primary}"
        + ("" if pub_available else f"  (same-pipeline NOT available: {st['reason']})"))
    log(f"   n basins = {len(BAS)}; median = {BAS[primary].median():.4f}; "
        f"IQR = {BAS[primary].quantile(.75)-BAS[primary].quantile(.25):.4f}")
    log("\n== screening (|rho| sorted) ==")
    log(S[["variable", "spearman_rho", "perm_p", "bh_fdr_q", "n_used"]].round(4).to_string(index=False))
    log("\n== top-decile penalty vs rest ==")
    for k, v in extreme_cmp.items():
        log(f"   {k:26s} top10%={v['median_top_decile_penalty']}  rest={v['median_rest']}  "
            f"p={v['mannwhitney_p'] if v['mannwhitney_p'] is None else round(v['mannwhitney_p'],2)}")
    log("\n== KGE instability flags ==")
    for k, v in flags.items():
        log(f"   {k:44s} {v}")
    log(f"\nwrote tables -> {PC.T2}")


if __name__ == "__main__":
    main()
