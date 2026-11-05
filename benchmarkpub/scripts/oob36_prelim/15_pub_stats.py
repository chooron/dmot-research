#!/usr/bin/env python3
"""15_pub_stats.py — Analysis 1 (clean PUB estimate) and Analysis 2 (model heterogeneity).

Analysis 1:  the same-pipeline, basin-paired PUB penalty
    Delta_KGE_PUB_{m,b} = aggregate_{j != k(b)} KGE^seenFold_{m,j,b} - KGE^heldout_{m,k(b),b}
with both aggregations (median-of-4 primary, mean-of-4 secondary) reported side by side,
basin-clustered bootstrap uncertainty, per-model and per-basin summaries, leave-one-fold-out
and extreme-basin sensitivity.

Analysis 2:  is the model-to-model difference in penalty real and separable?
    basin-paired bootstrap of D_b = Delta_PUB_{m1,b} - Delta_PUB_{m2,b}, a Cliff's-delta
    effect size, a permutation omnibus test with basin-preserving model-label shuffling, and a
    balanced two-way variance partition (model / basin / residual).

Inference discipline: the 531 basins are the resampling unit everywhere.  The 180 model x fold
cells are never used as independent inferential units — `36/36 models positive` and
`180/180 cells positive` are reported as descriptive facts only.

Outputs -> $OUT2/tables/pub_same_pipeline_model_basin.csv, pub_model_summary.csv,
                     pub_basin_summary.csv, pub_model_pairwise_bootstrap.csv,
                     pub_variance_decomposition.csv, pub_stats.json
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
N_PERM_OMNIBUS = 2000


def log(m: str) -> None:
    print(m, flush=True)


def boot_stat(values_by_basin: list[np.ndarray], stat: str, B: int = B_BOOT, seed: int = SEED):
    """Percentile CI resampling BASINS (each entry is one basin's values)."""
    rng = np.random.default_rng(seed)
    n = len(values_by_basin)
    fn = np.median if stat == "median" else np.mean
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        out[b] = fn(np.concatenate([values_by_basin[i] for i in idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-based effect size in [-1, 1]; ties contribute 0.5/0.5."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    gt = sum(float(np.sum(x > b)) for x in a)
    lt = sum(float(np.sum(x < b)) for x in a)
    denom = a.size * b.size
    return float((gt - lt) / denom) if denom else float("nan")


def main() -> None:
    PC.T2.mkdir(parents=True, exist_ok=True)
    st = PC.insample_status()
    if not st["available"]:
        raise RuntimeError(f"in-sample table unavailable: {st['reason']}")
    log(f"in-sample cells: {st['cells']}/{st['required']}")

    pub_med = PC.build_pub("median")
    pub_mean = PC.build_pub("mean")
    ext = PC.build_existing()

    full = pub_med.rename(columns={"kge_seenfolds": "kge_seenfolds_median4",
                                  "delta_kge_pub": "delta_kge_pub_median4"})[
        ["model", "basin_id", "heldout_fold", "kge_heldout", "n_seen_folds",
         "kge_seenfolds_median4", "delta_kge_pub_median4"]]
    full = full.merge(
        pub_mean[["model", "basin_id", "kge_seenfolds", "delta_kge_pub"]].rename(
            columns={"kge_seenfolds": "kge_seenfolds_mean4",
                     "delta_kge_pub": "delta_kge_pub_mean4"}),
        on=["model", "basin_id"], validate="one_to_one")
    full = full.merge(
        ext[["model", "basin_id", "kge_seendpl_canonical", "delta_kge_existing"]],
        on=["model", "basin_id"], validate="one_to_one")
    full["gauge_id_str"] = [str(int(b)).zfill(8) for b in full["basin_id"]]
    full.to_csv(PC.T2 / "pub_same_pipeline_model_basin.csv", index=False)

    models = sorted(full["model"].unique())
    basins = np.sort(full["basin_id"].unique())
    D = full.pivot(index="basin_id", columns="model", values="delta_kge_pub_median4").loc[basins, models]
    Dm = full.pivot(index="basin_id", columns="model", values="delta_kge_pub_mean4").loc[basins, models]
    if D.isna().any().any():
        raise ValueError("pivot NaN")

    report = {"meta": {
        "seed": SEED, "bootstrap_B": B_BOOT, "omnibus_permutations": N_PERM_OMNIBUS,
        "n_models": len(models), "n_basins": len(basins), "n_rows": int(len(full)),
        "primary_aggregation": "median of the 4 seen-fold in-sample KGE values",
        "secondary_aggregation": "mean of the 4 seen-fold in-sample KGE values",
        "definition": "Delta_KGE_PUB = aggregate_{4 seen folds}(KGE) - KGE_heldout ; positive = OOB penalty",
        "source_seen": "tables/pub_insample_seen_kge.csv (script 11, frozen best.pt, inference only)",
        "source_heldout": "tables/seen_ic_oob_merged.parquet kge_oob (frozen, phase-2 validated)",
        "inference_unit": "basin (531); model x fold cells are descriptive only",
    }}

    # =====================================================================
    # Analysis 1
    # =====================================================================
    a1 = {}
    for tag, M in (("median4", D), ("mean4", Dm)):
        v = M.to_numpy().ravel()
        per_basin = [M.to_numpy()[i] for i in range(M.shape[0])]
        lo, hi = boot_stat(per_basin, "median")
        lo_m, hi_m = boot_stat(per_basin, "mean")
        a1[tag] = {
            "pooled_median": float(np.median(v)), "pooled_median_ci95": [lo, hi],
            "pooled_mean": float(np.mean(v)), "pooled_mean_ci95": [lo_m, hi_m],
            "pooled_q25": float(np.percentile(v, 25)), "pooled_q75": float(np.percentile(v, 75)),
            "pooled_q05": float(np.percentile(v, 5)), "pooled_q95": float(np.percentile(v, 95)),
            "frac_pairs_positive": float((v > 0).mean()),
            "frac_pairs_negative": float((v < 0).mean()),
        }
    report["analysis1"] = a1

    # per-model table
    rows = []
    for m in models:
        d = D[m].to_numpy(float)
        dm = Dm[m].to_numpy(float)
        # each basin contributes exactly one value for this model, so resampling basins
        # here is a plain bootstrap over that model's 531 basin values
        lo, hi = boot_stat([np.asarray([v]) for v in d], "median")
        rows.append({
            "model": m, "n_basins": int(d.size),
            "delta_pub_median": float(np.median(d)), "delta_pub_mean": float(np.mean(d)),
            "delta_pub_q25": float(np.percentile(d, 25)), "delta_pub_q75": float(np.percentile(d, 75)),
            "delta_pub_iqr": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "delta_pub_median_ci95_lo": lo, "delta_pub_median_ci95_hi": hi,
            "frac_positive": float((d > 0).mean()), "frac_negative": float((d < 0).mean()),
            "delta_pub_mean4_median": float(np.median(dm)),
            "delta_pub_mean4_mean": float(np.mean(dm)),
            "excess_mean_over_median": float(np.mean(d) - np.median(d)),
            "delta_existing_median": float(ext[ext.model == m]["delta_kge_existing"].median()),
            "delta_existing_mean": float(ext[ext.model == m]["delta_kge_existing"].mean()),
            "heldout_kge_median": float(full[full.model == m]["kge_heldout"].median()),
            "seenfolds_kge_median": float(pub_med[pub_med.model == m]["kge_seenfolds"].median()),
        })
    M1 = pd.DataFrame(rows).sort_values("delta_pub_median", ascending=False).reset_index(drop=True)
    M1["rank_by_median"] = np.arange(1, len(M1) + 1)
    M1.to_csv(PC.T2 / "pub_model_summary.csv", index=False)

    # per-basin table
    B1 = PC.basin_penalty(pub_med, "delta_kge_pub")
    B1 = B1.merge(PC.basin_penalty(pub_mean, "delta_kge_pub")[
        ["basin_id", "penalty_median_over_models"]].rename(
        columns={"penalty_median_over_models": "penalty_median_over_models_mean4"}), on="basin_id")
    B1 = B1.merge(PC.basin_penalty(ext, "delta_kge_existing")[
        ["basin_id", "penalty_median_over_models"]].rename(
        columns={"penalty_median_over_models": "penalty_median_over_models_existing"}), on="basin_id")
    B1.to_csv(PC.T2 / "pub_basin_summary.csv", index=False)

    # leave-one-fold-out
    lofo = []
    for k in range(5):
        keep = full["heldout_fold"] != k
        sub = full[keep]
        pv = sub.pivot(index="basin_id", columns="model", values="delta_kge_pub_median4")
        lofo.append({"excluded_fold": k, "n_basins_kept": int(pv.shape[0]),
                     "pooled_median": float(np.median(pv.to_numpy())),
                     "pooled_mean": float(np.mean(pv.to_numpy())),
                     "pooled_median_of_model_medians": float(np.median(pv.median(axis=0).to_numpy()))})
    L = pd.DataFrame(lofo)
    report["analysis1"]["leave_one_fold_out"] = L.to_dict(orient="records")

    # extreme-basin sensitivity
    pbm = B1.set_index("basin_id")["penalty_median_over_models"]
    sens = []
    for label, drop in (
        ("none", []),
        ("drop worst 1 basin", [pbm.idxmax()]),
        ("drop worst 5 basins", pbm.nlargest(5).index.tolist()),
        ("drop worst 5% basins", pbm.nlargest(int(0.05 * len(pbm))).index.tolist()),
        ("drop worst 10% basins", pbm.nlargest(int(0.10 * len(pbm))).index.tolist()),
        ("drop top/bottom 5% by |median|", pd.concat([
            pbm.nlargest(int(0.025 * len(pbm))), pbm.nsmallest(int(0.025 * len(pbm)))]).index.tolist()),
    ):
        sub = full[~full["basin_id"].isin(drop)]
        sens.append({"scenario": label, "n_dropped_basins": len(drop),
                     "n_rows_kept": int(len(sub)),
                     "pooled_median": float(sub["delta_kge_pub_median4"].median()),
                     "pooled_mean": float(sub["delta_kge_pub_median4"].mean())})
    report["analysis1"]["extreme_basin_sensitivity"] = sens

    # descriptive universality facts (NOT inferential units)
    cell_med = full.groupby(["model", "heldout_fold"])["delta_kge_pub_median4"].median()
    report["analysis1"]["descriptive_universality"] = {
        "n_models_positive_median": int((M1["delta_pub_median"] > 0).sum()),
        "n_models_total": len(M1),
        "n_model_fold_cells_positive": int((cell_med > 0).sum()),
        "n_model_fold_cells_total": int(len(cell_med)),
        "note": ("descriptive facts only; the 180 model x fold cells are NOT independent "
                 "inferential units and no sign test is computed on them"),
    }

    # =====================================================================
    # Analysis 2
    # =====================================================================
    arr = D.to_numpy(np.float64)
    # omnibus permutation: shuffle model labels within each basin (basin structure preserved)
    obs_sd = float(np.std(np.median(arr, axis=0), ddof=1))
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM_OMNIBUS)
    for i in range(N_PERM_OMNIBUS):
        perm = np.argsort(rng.random(arr.shape), axis=1)
        shuffled = np.take_along_axis(arr, perm, axis=1)
        null[i] = np.std(np.median(shuffled, axis=0), ddof=1)
    p_omni = (float(np.sum(null >= obs_sd)) + 1) / (N_PERM_OMNIBUS + 1)

    # balanced two-way partition (orthogonal because the design is balanced)
    grand = float(np.mean(arr))
    row_m = arr.mean(axis=1, keepdims=True)   # basin effect
    col_m = arr.mean(axis=0, keepdims=True)   # model effect
    ss_total = float(np.sum((arr - grand) ** 2))
    ss_basin = float(arr.shape[1] * np.sum((row_m - grand) ** 2))
    ss_model = float(arr.shape[0] * np.sum((col_m - grand) ** 2))
    ss_err = float(ss_total - ss_basin - ss_model)
    vd = pd.DataFrame([
        {"component": "basin", "ss": ss_basin, "df": arr.shape[0] - 1,
         "ms": ss_basin / (arr.shape[0] - 1), "frac_of_total_ss": ss_basin / ss_total},
        {"component": "model", "ss": ss_model, "df": arr.shape[1] - 1,
         "ms": ss_model / (arr.shape[1] - 1), "frac_of_total_ss": ss_model / ss_total},
        {"component": "residual", "ss": ss_err, "df": (arr.shape[0] - 1) * (arr.shape[1] - 1),
         "ms": ss_err / ((arr.shape[0] - 1) * (arr.shape[1] - 1)), "frac_of_total_ss": ss_err / ss_total},
        {"component": "total", "ss": ss_total, "df": arr.size - 1,
         "ms": ss_total / (arr.size - 1), "frac_of_total_ss": 1.0},
    ])
    vd.to_csv(PC.T2 / "pub_variance_decomposition.csv", index=False)

    # robust analog: spread across models within a basin vs across basins within a model
    within_basin_iqr = np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)
    within_model_iqr = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)

    # pairwise basin-paired bootstrap (all pairs; CI + effect size, no p-values)
    pairs = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            d = arr[:, i] - arr[:, models.index(m2)]
            rng2 = np.random.default_rng(SEED)
            meds = np.median(d[rng2.integers(0, d.size, size=(B_BOOT, d.size))], axis=1)
            lo, hi = float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
            pairs.append({
                "model_a": m1, "model_b": m2, "n_basins": int(d.size),
                "median_diff_a_minus_b": float(np.median(d)),
                "ci95_lo": lo, "ci95_hi": hi,
                "separated": bool(lo > 0 or hi < 0),
                "frac_basins_a_gt_b": float((d > 0).mean()),
                "cliffs_delta_a_vs_b": cliffs_delta(arr[:, i], arr[:, models.index(m2)]),
            })
    P = pd.DataFrame(pairs)
    P.to_csv(PC.T2 / "pub_model_pairwise_bootstrap.csv", index=False)

    n_sep = P.groupby("model_a")["separated"].sum().add(
        P.groupby("model_b")["separated"].sum(), fill_value=0).astype(int)
    M1["n_models_separated_from"] = M1["model"].map(n_sep).fillna(0).astype(int)
    M1["n_models_total_minus_self"] = len(models) - 1
    M1.to_csv(PC.T2 / "pub_model_summary.csv", index=False)

    # Directional separation: for each model count how many others it is RELIABLY above / below.
    # A pair counts only when the paired basin bootstrap CI for (a - b) excludes 0.
    n_above = {}   # model -> # others it is reliably higher than
    n_below = {}   # model -> # others it is reliably lower than
    for m in models:
        n_above[m], n_below[m] = 0, 0
    for r in P.itertuples():
        if not r.separated:
            continue
        if r.median_diff_a_minus_b > 0:
            n_above[r.model_a] += 1
            n_below[r.model_b] += 1
        else:
            n_below[r.model_a] += 1
            n_above[r.model_b] += 1
    # n_above[m] = how many other models m is reliably HIGHER than (higher penalty)
    # n_below[m] = how many other models m is reliably LOWER  than (lower penalty)
    M1["n_others_reliably_below_this_model"] = M1["model"].map(n_above).astype(int)
    M1["n_others_reliably_above_this_model"] = M1["model"].map(n_below).astype(int)
    M1.to_csv(PC.T2 / "pub_model_summary.csv", index=False)

    # a model is a reliable extreme when it separates from a MAJORITY of the others in one direction
    thr = (len(models) - 1) / 2
    reliable_high = M1[M1["n_others_reliably_below_this_model"] > thr]["model"].tolist()
    reliable_low = M1[M1["n_others_reliably_above_this_model"] > thr]["model"].tolist()
    unresolved = M1[~M1["model"].isin(reliable_high + reliable_low)]["model"].tolist()

    report["analysis2"] = {
        "omnibus_permutation": {
            "statistic": "SD of the 36 model medians of Delta_KGE_PUB",
            "observed": obs_sd, "null_median": float(np.median(null)),
            "null_q95": float(np.percentile(null, 95)), "perm_p": p_omni,
            "n_perm": N_PERM_OMNIBUS,
            "note": "model labels shuffled within each basin, so basin structure is preserved",
        },
        "variance_decomposition": vd.to_dict(orient="records"),
        "variance_decomposition_ms_ratio": {
            "basin_ms_over_model_ms": float(
                (ss_basin / (arr.shape[0] - 1)) / (ss_model / (arr.shape[1] - 1))),
            "note": "mean-square ratio; the raw SS fractions above are not df-adjusted",
        },
        "robust_spread_analog": {
            "median_within_basin_across_model_iqr": float(np.median(within_basin_iqr)),
            "median_within_model_across_basin_iqr": float(np.median(within_model_iqr)),
            "ratio_basin_over_model": float(np.median(within_basin_iqr) / np.median(within_model_iqr)),
        },
        "pairwise": {
            "n_pairs": int(len(P)), "n_separated_ci_excludes_0": int(P["separated"].sum()),
            "frac_separated": float(P["separated"].mean()),
            "max_abs_cliffs_delta": float(P["cliffs_delta_a_vs_b"].abs().max()),
            "median_abs_cliffs_delta": float(P["cliffs_delta_a_vs_b"].abs().median()),
            "most_separated_pairs": P.assign(absdiff=P["median_diff_a_minus_b"].abs())
                                     .nlargest(10, "absdiff")[
                ["model_a", "model_b", "median_diff_a_minus_b", "ci95_lo", "ci95_hi",
                 "frac_basins_a_gt_b", "cliffs_delta_a_vs_b"]].to_dict(orient="records"),
        },
        "reliable_high_penalty_models": reliable_high,
        "reliable_low_penalty_models": reliable_low,
        "models_unresolved_overlapping": unresolved,
        "separation_threshold_n_models": float(thr),
        "claims_allowed": (
            "only models listed in reliable_high_penalty_models may be called reliably more "
            "degraded; every model in models_unresolved_overlapping must be reported as "
            "overlapping / not separable"),
    }

    with open(PC.T2 / "pub_stats.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ---------------- console ----------------
    log("\n=== Analysis 1: same-pipeline PUB penalty ===")
    for tag in ("median4", "mean4"):
        v = a1[tag]
        log(f"  [{tag}] pooled median = {v['pooled_median']:.4f} CI95 "
            f"[{v['pooled_median_ci95'][0]:.4f}, {v['pooled_median_ci95'][1]:.4f}] | "
            f"mean = {v['pooled_mean']:.4f} | q25/q75 = {v['pooled_q25']:.4f}/{v['pooled_q75']:.4f} | "
            f"frac>0 = {v['frac_pairs_positive']:.4f}")
    log(f"  descriptive: {report['analysis1']['descriptive_universality']['n_models_positive_median']}"
        f"/36 models positive; "
        f"{report['analysis1']['descriptive_universality']['n_model_fold_cells_positive']}"
        f"/180 cells positive (descriptive only)")
    log("  leave-one-fold-out pooled medians: " +
        ", ".join(f"f{int(r['excluded_fold'])}={r['pooled_median']:.4f}" for r in lofo))
    log("  extreme-basin sensitivity: " +
        "; ".join(f"{r['scenario']}={r['pooled_median']:.4f}" for r in sens))
    log(f"  model median range: {M1['delta_pub_median'].min():.4f} "
        f"({M1.iloc[-1]['model']}) .. {M1['delta_pub_median'].max():.4f} ({M1.iloc[0]['model']})")
    log(f"  cross-check vs previous cross-pipeline estimate: "
        f"existing pooled median = {ext['delta_kge_existing'].median():.4f}, "
        f"same-pipeline pooled median = {a1['median4']['pooled_median']:.4f}")

    log("\n=== Analysis 2: model heterogeneity ===")
    log(f"  omnibus permutation: observed SD = {obs_sd:.5f}, null q95 = "
        f"{np.percentile(null, 95):.5f}, p = {p_omni:.4f}")
    log("  variance partition (balanced two-way): " +
        ", ".join(f"{r['component']}={r['frac_of_total_ss']:.4f}" for r in vd.to_dict(orient="records")))
    log(f"  robust analog: median within-basin across-model IQR = "
        f"{np.median(within_basin_iqr):.4f} vs within-model across-basin IQR = "
        f"{np.median(within_model_iqr):.4f} (ratio {np.median(within_basin_iqr)/np.median(within_model_iqr):.2f})")
    log(f"  pairwise: {int(P['separated'].sum())}/{len(P)} pairs separated at 95% "
        f"({P['separated'].mean():.3f}); median |Cliff's delta| = "
        f"{P['cliffs_delta_a_vs_b'].abs().median():.3f}")
    log(f"  variance MS ratio basin/model = "
        f"{(ss_basin/(arr.shape[0]-1))/(ss_model/(arr.shape[1]-1)):.1f}")
    log(f"  reliable HIGH-penalty extremes ({len(reliable_high)}): {reliable_high}")
    log(f"  reliable LOW-penalty extremes ({len(reliable_low)}): {reliable_low}")
    log(f"  unresolved/overlapping ({len(unresolved)}): {unresolved}")
    log(f"\nwrote tables -> {PC.T2}")


if __name__ == "__main__":
    main()
