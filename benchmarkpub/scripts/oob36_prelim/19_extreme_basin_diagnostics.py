#!/usr/bin/env python3
"""19_extreme_basin_diagnostics.py — Analysis 2: are the extreme PUB basins KGE-pathological?

The largest same-pipeline penalties might simply come from basins where KGE is numerically
ill-behaved: a near-zero observed mean or near-zero observed variance makes the KGE
beta/gamma ratios degenerate, because KGE adds a fixed epsilon to those denominators.

This checks that, using only provenance-clear quantities:

  * the frozen top-5% basin set by `penalty_median_pub` (defined in the next-stage analysis
    and NOT re-selected here);
  * observed-flow statistics computed directly from the same observed series the KGE is
    scored against (mean, SD, CV, zero-flow and near-zero-flow fractions, missing fraction);
  * the named Caravan attributes already used for the screening (runoff ratio, aridity);
  * the project's OWN KGE conventions, read from the frozen pipeline code:
      - `KGE_EPS = 0.1`  (added to the correlation denominator and to the obs mean/std in the
        beta and gamma ratios -> a basin with obs mean or SD well below 0.1 mm/day has a
        denominator dominated by eps, i.e. a pathological KGE ratio);
      - `min_observation_std = 0.01` (declared minimum informative variability for the
        training-window catalogue);
      - `min_valid_points = 30` and the `kge_numpy` invalid rule `count < 2`.

Then it re-reports the headline pooled median Delta_KGE_PUB after removing the extreme and
the pathological basins, with basin-clustered bootstrap CIs, to answer whether the headline
number depends on them.

This is a sanity check, not basin attribution: no ML, no feature selection, no causal claim.

Outputs -> $OUT3/tables/{extreme_basin_metric_diagnostics,extreme_basin_group_comparison,
                         pub_penalty_sensitivity}.csv  and extreme_basin.json
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

SEED = 20260930
B_BOOT = 2000
QUANTILE = 0.95

# frozen project conventions (read from project/benchmark/scripts/oob/oob_common.py)
KGE_EPS = 0.1                    # oob_common.py:48  KGE_EPS = 0.1
MIN_OBSERVATION_STD = 0.01       # oob_common.py:633 build_informative_window_catalog default
MIN_VALID_POINTS = 30            # oob_common.py:632 build_informative_window_catalog default
KGE_NUMPY_MIN_COUNT = 2          # oob_common.py:706 invalid = (count < 2.0) | ...


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


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(float(np.sum(x > b)) for x in a)
    lt = sum(float(np.sum(x < b)) for x in a)
    return float((gt - lt) / (a.size * b.size))


def boot_pooled_median(pairs: pd.DataFrame, value_col: str, B: int = B_BOOT, seed: int = SEED):
    """Pooled median over (basin, model) pairs, resampling BASINS with all models attached."""
    groups = [g[value_col].to_numpy(float) for _, g in pairs.groupby("basin_id", sort=True)]
    rng = np.random.default_rng(seed)
    n = len(groups)
    draws = np.array([np.median(np.concatenate([groups[i] for i in rng.integers(0, n, size=n)]))
                      for _ in range(B)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    PC.T3.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")
    if len(pairs) != 19116:
        raise ValueError(f"expected 19,116 pairs, got {len(pairs)}")
    B = pd.read_csv(PC.OUT2 / "tables" / "basin_attribute_table.csv")
    if len(B) != 531:
        raise ValueError(f"expected 531 basins, got {len(B)}")

    # ---- frozen extreme set (NOT re-selected here) -------------------------
    thr = float(B["penalty_median_pub"].quantile(QUANTILE))
    B["extreme_top5pct"] = B["penalty_median_pub"] >= thr

    # ---- observed-series diagnostics ---------------------------------------
    ids = np.sort(pairs["basin_id"].unique())
    obs = PC.load_obs_full(ids)
    pos = {int(b): i for i, b in enumerate(ids.tolist())}
    order = np.array([pos[int(b)] for b in B["basin_id"]])
    O = obs[:, order]
    with np.errstate(invalid="ignore"):
        obs_mean = np.nanmean(O, axis=0)
        obs_std = np.nanstd(O, axis=0, ddof=1)
        obs_cv = obs_std / obs_mean
        missing_frac = np.isnan(O).mean(axis=0)
        zero_frac = np.nanmean(O <= 0.0, axis=0)
        nearzero_0p01 = np.nanmean(O < 0.01, axis=0)
        nearzero_0p1 = np.nanmean(O < KGE_EPS, axis=0)      # 0.1 mm/day == KGE_EPS
    B["obs_mean_recomputed"] = obs_mean
    B["obs_std"] = obs_std
    B["obs_cv"] = obs_cv
    B["missing_frac"] = missing_frac
    B["zero_flow_frac_recomputed"] = zero_frac
    B["near_zero_flow_frac_lt_0p01"] = nearzero_0p01
    B["near_zero_flow_frac_lt_KGE_EPS_0p1"] = nearzero_0p1

    # per-basin minimum valid-pair count over the 36 models (distance from the invalid rule)
    nv = pairs.merge(
        pd.read_parquet(PC.MERGED)[["model", "gauge_id", "n_valid"]].rename(
            columns={"gauge_id": "basin_id"}), on=["model", "basin_id"], how="left")
    if nv["n_valid"].isna().any():
        raise ValueError("n_valid could not be joined for every pair")
    B["n_valid_min_over_models"] = B["basin_id"].map(
        nv.groupby("basin_id")["n_valid"].min())

    # ---- distance from the frozen KGE conventions --------------------------
    B["flag_obs_mean_lt_KGE_EPS"] = B["obs_mean_recomputed"] < KGE_EPS
    B["flag_obs_std_lt_KGE_EPS"] = B["obs_std"] < KGE_EPS
    B["flag_obs_std_lt_min_observation_std"] = B["obs_std"] < MIN_OBSERVATION_STD
    B["flag_count_lt_min_valid_points"] = B["n_valid_min_over_models"] < MIN_VALID_POINTS
    B["flag_count_lt_kge_numpy_min"] = B["n_valid_min_over_models"] < KGE_NUMPY_MIN_COUNT
    B["flag_zero_flow_gt_0p5"] = B["zero_flow_frac_recomputed"] > 0.5
    B["flag_missing_obs"] = B["missing_frac"] > 0.0
    B["flag_pathological_denominator"] = (B["flag_obs_mean_lt_KGE_EPS"]
                                          | B["flag_obs_std_lt_KGE_EPS"])

    B.to_csv(PC.T3 / "extreme_basin_metric_diagnostics.csv", index=False)

    # ---- group comparison: top 5% vs remaining 95% -------------------------
    ext = B["extreme_top5pct"].to_numpy()
    rest = ~ext
    DIAGS = ["obs_mean_recomputed", "obs_std", "obs_cv", "obs_var", "zero_flow_frac_recomputed",
             "near_zero_flow_frac_lt_0p01", "near_zero_flow_frac_lt_KGE_EPS_0p1",
             "missing_frac", "runoff_ratio", "arid", "q_log_sd", "p_mean",
             "oob_kge_median", "oob_kge_sd_over_models", "n_valid_min_over_models"]
    rows = []
    for c in DIAGS:
        a = B.loc[ext, c].to_numpy(float); b = B.loc[rest, c].to_numpy(float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        rows.append({
            "diagnostic": c,
            "extreme_median": float(np.median(a)), "extreme_iqr": float(
                np.percentile(a, 75) - np.percentile(a, 25)),
            "rest_median": float(np.median(b)), "rest_iqr": float(
                np.percentile(b, 75) - np.percentile(b, 25)),
            "cliffs_delta_extreme_vs_rest": cliffs_delta(a, b),
            "mannwhitney_p": float(sstats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
            "n_extreme": int(a.size), "n_rest": int(b.size),
        })
    G = pd.DataFrame(rows)
    G["bh_fdr_q"] = bh_fdr(G["mannwhitney_p"].to_numpy())
    G["abs_cliffs_delta"] = G["cliffs_delta_extreme_vs_rest"].abs()
    G = G.sort_values("abs_cliffs_delta", ascending=False).reset_index(drop=True)
    G.to_csv(PC.T3 / "extreme_basin_group_comparison.csv", index=False)

    flags = {}
    for c in [c for c in B.columns if c.startswith("flag_")]:
        m = B[c].to_numpy(bool)
        flags[c] = {"n_total": int(m.sum()),
                    "share_of_531": float(m.mean()),
                    "n_in_extreme_top5pct": int((m & ext).sum()),
                    "share_of_extreme_top5pct": float((m & ext).sum() / ext.sum()),
                    "share_of_rest": float((m & rest).sum() / rest.sum())}

    # ---- sensitivity of the headline PUB estimate --------------------------
    keep_ids = {"all basins (531)": np.ones(len(B), bool)}
    keep_ids["exclude extreme top 5% penalty basins"] = ~ext
    keep_ids["exclude KGE-pathological denominator (obs_mean<0.1 or obs_std<0.1)"] = \
        ~B["flag_pathological_denominator"].to_numpy()
    keep_ids["exclude obs_std < min_observation_std (0.01)"] = \
        ~B["flag_obs_std_lt_min_observation_std"].to_numpy()
    keep_ids["exclude zero-flow fraction > 0.5"] = ~B["flag_zero_flow_gt_0p5"].to_numpy()
    keep_ids["exclude basins with missing observations"] = ~B["flag_missing_obs"].to_numpy()
    keep_ids["exclude extreme top 5% AND pathological"] = \
        ~(ext | B["flag_pathological_denominator"].to_numpy())
    keep_ids["exclude union of all suspect basins"] = ~(
        ext | B["flag_pathological_denominator"].to_numpy()
        | B["flag_obs_std_lt_min_observation_std"].to_numpy()
        | B["flag_zero_flow_gt_0p5"].to_numpy()
        | B["flag_missing_obs"].to_numpy())

    sens_rows = []
    for label, mask in keep_ids.items():
        kept = set(B.loc[mask, "basin_id"].astype(int))
        sub = pairs[pairs["basin_id"].isin(kept)]
        for col, name in (("delta_kge_pub_median4", "median4"), ("delta_kge_pub_mean4", "mean4")):
            lo, hi = boot_pooled_median(sub, col)
            sens_rows.append({
                "scenario": label, "aggregation": name,
                "n_basins_kept": int(mask.sum()), "n_basins_dropped": int((~mask).sum()),
                "n_pairs_kept": int(len(sub)),
                "pooled_median": float(sub[col].median()),
                "pooled_mean": float(sub[col].mean()),
                "pooled_median_ci95_lo": lo, "pooled_median_ci95_hi": hi,
                "frac_pairs_positive": float((sub[col] > 0).mean()),
            })
    S = pd.DataFrame(sens_rows)
    S.to_csv(PC.T3 / "pub_penalty_sensitivity.csv", index=False)

    base_m = float(S[(S.scenario == "all basins (531)") & (S.aggregation == "median4")].pooled_median.iloc[0])
    worst_m = float(S[S.aggregation == "median4"].pooled_median.min())
    report = {
        "meta": {
            "seed": SEED, "bootstrap_B": B_BOOT, "quantile": QUANTILE,
            "n_extreme": int(ext.sum()), "n_basins": int(len(B)),
            "extreme_definition": "frozen penalty_median_pub >= its 95th percentile "
                                  "(next-stage table; NOT re-selected here)",
            "threshold_value": thr,
            "project_conventions_read": {
                "KGE_EPS": KGE_EPS,
                "min_observation_std": MIN_OBSERVATION_STD,
                "min_valid_points": MIN_VALID_POINTS,
                "kge_numpy_min_count": KGE_NUMPY_MIN_COUNT,
                "source": "project/benchmark/scripts/oob/oob_common.py",
            },
            "scope": "sanity check only; no ML, no feature selection, no causal claim",
        },
        "flags": flags,
        "group_comparison": G.to_dict(orient="records"),
        "sensitivity": sens_rows,
        "headline": {
            "all_basins_median4": base_m,
            "min_over_scenarios_median4": worst_m,
            "max_abs_change_vs_all": float(abs(worst_m - base_m)),
            "interpretation": ("if the pooled median stays near the all-basin value after "
                               "removing the extreme and pathological basins, the headline is "
                               "not driven by them"),
        },
    }
    with open(PC.T3 / "extreme_basin.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ---- console -----------------------------------------------------------
    log(f"== 2.1 extreme set (frozen) : {int(ext.sum())} basins at penalty >= {thr:.4f} ==")
    log("\n== 2.2 group comparison (top 5% vs remaining 95%) ==")
    log(G[["diagnostic", "extreme_median", "rest_median", "cliffs_delta_extreme_vs_rest",
           "mannwhitney_p", "bh_fdr_q"]].round(5).to_string(index=False))
    log("\n== 2.3 distance from the project's own KGE conventions ==")
    for k, v in flags.items():
        log(f"  {k:42s} total={v['n_total']:4d} ({v['share_of_531']:.3f} of 531)  "
            f"in_top5%={v['n_in_extreme_top5pct']:3d} "
            f"({v['share_of_extreme_top5pct']:.3f} of the top 5%)")
    log("\n== 2.4 headline sensitivity ==")
    log(S[S.aggregation == "median4"][["scenario", "n_basins_kept", "n_basins_dropped",
                                       "pooled_median", "pooled_median_ci95_lo",
                                       "pooled_median_ci95_hi"]].round(4).to_string(index=False))
    log(f"\n  all basins = {base_m:.4f}; worst scenario = {worst_m:.4f}; "
        f"max |change| = {abs(worst_m-base_m):.4f}")
    log(f"\nwrote tables -> {PC.T3}")


if __name__ == "__main__":
    main()
