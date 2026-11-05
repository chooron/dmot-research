#!/usr/bin/env python3
"""38_parsimony_complexity_figures.py — PNG figures for the parsimony-complexity analysis.

Writes ONLY .png into the new analysis directory:
    parameter_count_vs_seen_oob_kge.png     n_parameters vs aggregate seen / OOB KGE
    parameter_count_vs_pub_gap.png          n_parameters vs ΔKGE_PUB level and tail fraction
    parameter_count_vs_parameter_instability.png
                                            n_parameters vs fold variability and seen->OOB displacement
    parameter_displacement_vs_pub_gap.png   parameter displacement vs ΔKGE_PUB (pooled / within / model)
    parameter_count_vs_adequacy.png         n_parameters vs RELAXED_70 and signature pass fraction
    complexity_tradeoff_overview.png        4-panel trade-off overview

Reads only the tables written by script 37.  No training, no simulation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO = Path("/home/jingxin/code/dmg-research")
OUT = REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_parsimony_complexity_20260930"
T = OUT / "tables"
F = OUT / "figures"

# Okabe-Ito derived, colour-blind safe; every series also carries a distinct marker/linestyle.
INK, BLU, LBL, ORG, RED, GRY, GRN, PUR = (
    "#2b2b2b", "#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#8a8a8a", "#009E73", "#CC79A7")
BAND_COLORS = {"low": "#56B4E9", "mid": "#E69F00", "high": "#D55E00"}


def save(fig, stem):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def ols_line(ax, x, y, color=GRY):
    if len(x) < 2:
        return
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 50)
    ax.plot(xs, a + b * xs, color=color, lw=1.2, ls="--", zorder=1)


def label_points(ax, x, y, labels, fontsize=6.2):
    for i, (xi, yi, name) in enumerate(zip(x, y, labels)):
        # deterministic alternating offset (no hash(): keeps labels reproducible across runs)
        right = (i % 2 == 0)
        ax.annotate(name, (xi, yi), textcoords="offset points",
                    xytext=(4 if right else -4, 3 if right else -8),
                    fontsize=fontsize, color=INK)

def band_shade(ax, model, band_boundaries, ymin, ymax):
    for lo, hi, name in band_boundaries:
        ax.axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.07, zorder=0)


def main() -> None:
    model = pd.read_csv(T / "model_level_complexity_performance.csv")
    trade = pd.read_csv(T / "complexity_tradeoff_summary.csv")
    mb = pd.read_csv(T / "model_basin_parameter_stability.csv")
    tr = pd.read_csv(T / "complexity_vs_transfer.csv")
    aq = pd.read_csv(T / "complexity_vs_adequacy.csv")
    # keep only the n_parameters rows so y_quantity is a unique index (n_states is a secondary sensitivity)
    tr = tr[tr["x_quantity"] == "n_parameters"].reset_index(drop=True)
    aq = aq[aq["x_quantity"] == "n_parameters"].reset_index(drop=True)
    # json bundle read below
    import json
    bundle = json.loads((T / "parsimony_complexity.json").read_text())
    prels = {r["y_quantity"]: r for r in bundle["performance_relationships"]
             if r["x_quantity"] == "n_parameters"}

    npv = model["n_parameters"].to_numpy(float)
    jit = np.random.default_rng(0).normal(0, 0.10, len(model))
    xj = npv + jit

    # ------------------------------------------------------------------ F1
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True)
    for ax, col, ttl, mark in [
        (axes[0], "seen_kge_aggregate_mean", "aggregate SEEN KGE", "o"),
        (axes[1], "oob_kge_aggregate_mean", "aggregate OOB KGE", "s"),
    ]:
        y = model[col].to_numpy(float)
        ax.scatter(xj, y, s=42, marker=mark, color=BLU, alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
        ols_line(ax, npv, y)
        r = prels[col]
        ax.set_title(f"(a) {ttl}\nSpearman ρ={r['spearman_rho']:+.3f}, permutation p={r['spearman_p_permutation']:.3f}, "
                     f"Pearson r={r['pearson_r']:+.3f}  (n=36 models)", fontsize=9.5)
        ax.set_xlabel("n_parameters  (free parameters of the hydrological model)")
        ax.set_ylabel(ttl)
        ax.set_xticks(sorted(model.n_parameters.unique()))
        ax.grid(alpha=0.25)
        label_points(ax, xj, y, model["model"])
    axes[0].set_title("(a) aggregate SEEN KGE vs n_parameters\n"
                      f"Spearman ρ={prels['seen_kge_aggregate_mean']['spearman_rho']:+.3f}, "
                      f"p_perm={prels['seen_kge_aggregate_mean']['spearman_p_permutation']:.3f}, "
                      f"Pearson r={prels['seen_kge_aggregate_mean']['pearson_r']:+.3f}", fontsize=9.5)
    axes[1].set_title("(b) aggregate OOB KGE vs n_parameters\n"
                      f"Spearman ρ={prels['oob_kge_aggregate_mean']['spearman_rho']:+.3f}, "
                      f"p_perm={prels['oob_kge_aggregate_mean']['spearman_p_permutation']:.3f}, "
                      f"Pearson r={prels['oob_kge_aggregate_mean']['pearson_r']:+.3f}", fontsize=9.5)
    fig.suptitle("Aggregate KGE is averaged over the 531 basins (per-basin SEEN = median of 4 in-sample folds; "
                 "per-basin OOB = the held-out fold). n=36 models, 531 basins.", fontsize=9, y=1.02)
    save(fig, "parameter_count_vs_seen_oob_kge")

    # ------------------------------------------------------------------ F2
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True)
    d = model["delta_kge_pub_median"].to_numpy(float)
    axes[0].scatter(xj, d, s=42, color=RED, alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
    ols_line(axes[0], npv, d)
    r = tr.set_index("y_quantity").loc["delta_kge_pub_median"]
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_title("(a) median ΔKGE$_{PUB}$ vs n_parameters\n"
                      f"Spearman ρ={r.spearman_rho:+.3f}, p_perm={r.spearman_p_permutation:.3f}; "
                      f"Q75 ρ={tr.set_index('y_quantity').loc['delta_kge_pub_q75','spearman_rho']:+.3f}", fontsize=9.5)
    axes[0].set_ylabel("median ΔKGE$_{PUB}$ = seen − OOB (per model, 531 basins)")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(alpha=0.25)
    label_points(axes[0], xj, d, model["model"])

    f05 = model["frac_delta_gt_0p05"].to_numpy(float)
    f10 = model["frac_delta_gt_0p10"].to_numpy(float)
    axes[1].scatter(xj, f05, s=40, marker="o", color=ORG, alpha=0.85, edgecolor="white",
                    lw=0.6, zorder=3, label="fraction with ΔKGE$_{PUB}$ > 0.05")
    axes[1].scatter(xj, f10, s=40, marker="^", color=PUR, alpha=0.85, edgecolor="white",
                    lw=0.6, zorder=3, label="fraction with ΔKGE$_{PUB}$ > 0.10")
    ols_line(axes[1], npv, f05, ORG)
    ols_line(axes[1], npv, f10, PUR)
    rt = tr.set_index("y_quantity")
    axes[1].set_title("(b) share of basins with large transfer loss\n"
                      f"ρ(frac>0.05)={rt.loc['frac_delta_gt_0p05','spearman_rho']:+.3f} "
                      f"(p_perm={rt.loc['frac_delta_gt_0p05','spearman_p_permutation']:.3f}); "
                      f"ρ(frac>0.10)={rt.loc['frac_delta_gt_0p10','spearman_rho']:+.3f} "
                      f"(p_perm={rt.loc['frac_delta_gt_0p10','spearman_p_permutation']:.3f})", fontsize=9.5)
    axes[1].set_ylabel("fraction of the 531 basins")
    axes[1].set_xlabel("n_parameters")
    axes[1].set_xticks(sorted(model.n_parameters.unique()))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_ylim(0.0, max(f05.max(), f10.max()) * 1.08)  # fraction has a natural zero
    fig.suptitle("ΔKGE$_{PUB}$ = same-pipeline seen (median of 4 in-sample folds) − OOB KGE; positive = loss under "
                 "basin transfer. Thresholds are descriptive, not tuned.", fontsize=9, y=1.02)
    save(fig, "parameter_count_vs_pub_gap")

    # ------------------------------------------------------------------ F3
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True)
    y1 = model["param_fold_instability_median"].to_numpy(float)          # per-parameter normalised RMS
    y2 = model["param_displacement_euclid_median"].to_numpy(float)       # total Euclidean (same units, all params)
    axes[0].scatter(xj, y1, s=40, marker="o", color=GRN, alpha=0.85, edgecolor="white", lw=0.6,
                    zorder=3, label="per-parameter normalised RMS")
    axes[0].scatter(xj, y2, s=40, marker="^", color=GRY, alpha=0.85, edgecolor="white", lw=0.6,
                    zorder=3, label="total Euclidean distance (all parameters)")
    ols_line(axes[0], npv, y1, GRN)
    ols_line(axes[0], npv, y2, GRY)
    rt = tr.set_index("y_quantity")
    axes[0].set_title("(a) fold-to-fold parameter variability (4 SEEN folds)\n"
                      f"per-param RMS ρ={rt.loc['param_fold_instability_median','spearman_rho']:+.3f} "
                      f"(p_perm={rt.loc['param_fold_instability_median','spearman_p_permutation']:.3f})", fontsize=9.5)
    axes[0].set_ylabel("normalised parameter distance θ$_{norm}$ (unitless)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")
    label_points(axes[0], xj, y1, model["model"])

    z1 = model["param_displacement_median"].to_numpy(float)
    z2 = model["param_displacement_euclid_median"].to_numpy(float)
    axes[1].scatter(xj, z1, s=40, marker="o", color=BLU, alpha=0.85, edgecolor="white", lw=0.6,
                    zorder=3, label="per-parameter normalised RMS")
    axes[1].scatter(xj, z2, s=40, marker="^", color=RED, alpha=0.85, edgecolor="white", lw=0.6,
                    zorder=3, label="total Euclidean distance (all parameters)")
    ols_line(axes[1], npv, z1, BLU)
    ols_line(axes[1], npv, z2, RED)
    axes[1].set_title("(b) SEEN → OOB parameter displacement (median over basins)\n"
                      f"per-param RMS ρ={rt.loc['param_displacement_median','spearman_rho']:+.3f} "
                      f"(p_perm={rt.loc['param_displacement_median','spearman_p_permutation']:.3f}); "
                      f"Euclidean ρ={rt.loc['param_displacement_euclid_median','spearman_rho']:+.3f} "
                      f"(p_perm={rt.loc['param_displacement_euclid_median','spearman_p_permutation']:.3f})", fontsize=9.5)
    axes[1].set_ylabel("normalised parameter distance θ$_{norm}$ (unitless)")
    axes[1].set_xlabel("n_parameters")
    axes[1].set_xticks(sorted(model.n_parameters.unique()))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("θ$_{norm}$=(θ$_{phys}$−lower)/(upper−lower) from each frozen best.pt. The total Euclidean distance "
                 "grows mechanically with dimension; the per-parameter normalised RMS does not.", fontsize=9, y=1.02)
    save(fig, "parameter_count_vs_parameter_instability")

    # ------------------------------------------------------------------ F4
    fig = plt.figure(figsize=(14.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    hb = a.hexbin(mb["disp_rms_mean_seen"], mb["delta_kge_pub_median4"], gridsize=45,
                  cmap="cividis", mincnt=1, bins="log")
    a.set_xlabel("per-basin SEEN → OOB parameter displacement (normalised RMS)")
    a.set_ylabel("per-basin ΔKGE$_{PUB}$")
    pl = pd.read_csv(T / "parameter_instability_vs_oob_loss.csv").set_index("instability_metric")
    a.set_title("(a) pooled 36 models × 531 basins (n=19116)\n"
                f"Spearman ρ={pl.loc['param_displacement_seen_mean_rms','pooled_spearman_rho']:+.3f}", fontsize=9.5)
    fig.colorbar(hb, ax=a, label="log10 count")
    a.grid(alpha=0.2)

    b = fig.add_subplot(gs[0, 1])
    m = model.merge(mb.groupby("model")["disp_rms_mean_seen"].median().rename("disp_med"), on="model")
    b.scatter(m["disp_med"], m["delta_kge_pub_median"], s=44, color=PUR, alpha=0.85,
              edgecolor="white", lw=0.6, zorder=3)
    ols_line(b, m["disp_med"], m["delta_kge_pub_median"], PUR)
    b.set_xlabel("model-median SEEN → OOB displacement")
    b.set_ylabel("model-median ΔKGE$_{PUB}$")
    b.set_title("(b) model level (n=36)\n"
                f"Spearman ρ={pl.loc['param_displacement_seen_mean_rms','model_median_spearman_rho']:+.3f} "
                f"(p_perm={pl.loc['param_displacement_seen_mean_rms','model_median_spearman_p_permutation']:.3f})",
                fontsize=9.5)
    b.grid(alpha=0.25)
    label_points(b, m["disp_med"], m["delta_kge_pub_median"], m["model"])

    c = fig.add_subplot(gs[0, 2])
    within = []
    for name, g in mb.groupby("model", sort=True):
        rw = np.corrcoef(g["disp_rms_mean_seen"].rank(), g["delta_kge_pub_median4"].rank())[0, 1]
        within.append(rw)
    within = np.asarray(within)
    c.hist(within, bins=14, color=BLU, alpha=0.85, edgecolor="white")
    c.axvline(0, color=INK, lw=1.0)
    c.axvline(np.median(within), color=RED, lw=1.4, ls="--",
              label=f"median = {np.median(within):+.3f}")
    c.set_xlabel("within-model Spearman ρ (displacement vs ΔKGE$_{PUB}$)")
    c.set_ylabel("number of models")
    c.set_title("(c) within-model correlations (36 models, 531 basins each)\n"
                f"{int((within<0).sum())}/36 negative; "
                f"{int((within>0).sum())}/36 positive", fontsize=9.5)
    c.legend(fontsize=8)
    c.grid(alpha=0.25)
    fig.suptitle("Parameter displacement vs OOB transfer loss. Pooled, model-level and within-model directions are "
                 "reported separately.", fontsize=9, y=1.02)
    save(fig, "parameter_displacement_vs_pub_gap")

    # ------------------------------------------------------------------ F5
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True)
    yseen = model["adequacy_relaxed70_seen_rate"].to_numpy(float)
    yoob = model["adequacy_relaxed70_oob_rate"].to_numpy(float)
    axes[0].scatter(xj, yseen, s=40, marker="o", color=ORG, alpha=0.85, edgecolor="white", lw=0.6,
                    zorder=3, label="SEEN RELAXED_70 pass rate")
    axes[0].scatter(xj, yoob, s=40, marker="^", color=GRN, alpha=0.85, edgecolor="white", lw=0.6,
                    zorder=3, label="OOB RELAXED_70 pass rate")
    ols_line(axes[0], npv, yseen, ORG)
    ols_line(axes[0], npv, yoob, GRN)
    aqi = aq.set_index("y_quantity")
    axes[0].set_title("(a) RELAXED_70 adequacy pass rate\n"
                      f"SEEN ρ={aqi.loc['adequacy_relaxed70_seen_rate','spearman_rho']:+.3f} "
                      f"(p_perm={aqi.loc['adequacy_relaxed70_seen_rate','spearman_p_permutation']:.3f}); "
                      f"OOB ρ={aqi.loc['adequacy_relaxed70_oob_rate','spearman_rho']:+.3f} "
                      f"(p_perm={aqi.loc['adequacy_relaxed70_oob_rate','spearman_p_permutation']:.3f})", fontsize=9.5)
    axes[0].set_ylabel("fraction of the 531 basins passing")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")
    label_points(axes[0], xj, yoob, model["model"])

    sseen = model["signature_pass_fraction_seen_mean"].to_numpy(float)
    soob = model["signature_pass_fraction_oob_mean"].to_numpy(float)
    axes[1].scatter(xj, sseen, s=40, marker="o", color=LBL, alpha=0.9, edgecolor="white", lw=0.6,
                    zorder=3, label="SEEN mean signature pass fraction")
    axes[1].scatter(xj, soob, s=40, marker="^", color=RED, alpha=0.9, edgecolor="white", lw=0.6,
                    zorder=3, label="OOB mean signature pass fraction")
    ols_line(axes[1], npv, sseen, LBL)
    ols_line(axes[1], npv, soob, RED)
    axes[1].set_title("(b) mean signature pass fraction\n"
                      f"SEEN ρ={aqi.loc['signature_pass_fraction_seen_mean','spearman_rho']:+.3f} "
                      f"(p_perm={aqi.loc['signature_pass_fraction_seen_mean','spearman_p_permutation']:.3f}); "
                      f"OOB ρ={aqi.loc['signature_pass_fraction_oob_mean','spearman_rho']:+.3f} "
                      f"(p_perm={aqi.loc['signature_pass_fraction_oob_mean','spearman_p_permutation']:.3f})", fontsize=9.5)
    axes[1].set_ylabel("mean fraction of evaluable signatures passed")
    axes[1].set_xlabel("n_parameters")
    axes[1].set_xticks(sorted(model.n_parameters.unique()))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Hydrological adequacy vs complexity. SEEN uses the cross-pipeline canonical dPL store and is not a "
                 "strict generalisation gap.", fontsize=9, y=1.02)
    save(fig, "parameter_count_vs_adequacy")

    # ------------------------------------------------------------------ F6
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4))
    npv2 = trade["n_parameters"].to_numpy(float)
    jit2 = np.random.default_rng(1).normal(0, 0.10, len(trade))

    ax = axes[0, 0]
    ax.scatter(npv2 + jit2, trade["oob_kge_aggregate_mean"], s=42, color=BLU, alpha=0.85,
               edgecolor="white", lw=0.6, zorder=3)
    pf = trade[trade.pareto_kge_vs_nparam].sort_values("n_parameters")
    ax.plot(pf["n_parameters"], pf["oob_kge_aggregate_mean"], color=RED, lw=1.4, marker="D",
            ms=5, zorder=4, label="Pareto front (max OOB KGE, min n_parameters)")
    label_points(ax, npv2 + jit2, trade["oob_kge_aggregate_mean"], trade["model"])
    ax.set_xlabel("n_parameters"); ax.set_ylabel("aggregate OOB KGE")
    ax.set_title("(a) accuracy vs complexity", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.scatter(npv2 + jit2, trade["adequacy_relaxed70_oob_rate"], s=42, color=GRN, alpha=0.85,
               edgecolor="white", lw=0.6, zorder=3)
    pf = trade[trade.pareto_adequacy_vs_nparam].sort_values("n_parameters")
    ax.plot(pf["n_parameters"], pf["adequacy_relaxed70_oob_rate"], color=RED, lw=1.4, marker="D",
            ms=5, zorder=4, label="Pareto front (max adequacy, min n_parameters)")
    label_points(ax, npv2 + jit2, trade["adequacy_relaxed70_oob_rate"], trade["model"])
    ax.set_xlabel("n_parameters"); ax.set_ylabel("OOB RELAXED_70 pass rate")
    ax.set_title("(b) adequacy vs complexity", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    rels = ([r for r in bundle["performance_relationships"] if r["x_quantity"] == "n_parameters"]
            + [r for r in tr.to_dict(orient="records") if r["y_quantity"] in
               ("delta_kge_pub_median", "frac_delta_gt_0p05", "param_fold_instability_median",
                "param_displacement_median", "param_displacement_euclid_median", "param_boundary_rate_mean")]
            + [r for r in aq.to_dict(orient="records") if r["y_quantity"] in
               ("adequacy_relaxed70_seen_rate", "adequacy_relaxed70_oob_rate",
                "signature_pass_fraction_seen_mean", "signature_pass_fraction_oob_mean")])
    seen_lbl = {"seen_kge_aggregate_mean": "seen KGE (agg)", "oob_kge_aggregate_mean": "OOB KGE (agg)",
                "seen_kge_median": "seen KGE (median)", "oob_kge_median": "OOB KGE (median)",
                "delta_kge_pub_median": "median ΔKGE_PUB", "frac_delta_gt_0p05": "frac ΔKGE>0.05",
                "param_fold_instability_median": "fold param variability",
                "param_displacement_median": "seen→OOB displacement (per-param RMS)",
                "param_displacement_euclid_median": "seen→OOB displacement (Euclidean)",
                "param_boundary_rate_mean": "parameter boundary-hit rate",
                "adequacy_relaxed70_seen_rate": "RELAXED_70 rate (SEEN)",
                "adequacy_relaxed70_oob_rate": "RELAXED_70 rate (OOB)",
                "signature_pass_fraction_seen_mean": "signature pass (SEEN)",
                "signature_pass_fraction_oob_mean": "signature pass (OOB)"}
    rels = [r for r in rels if r["x_quantity"] == "n_parameters" and r["y_quantity"] in seen_lbl]
    labs = [seen_lbl[r["y_quantity"]] for r in rels]
    rhos = [r["spearman_rho"] for r in rels]
    ps = [r["spearman_p_permutation"] for r in rels]
    yy = np.arange(len(rels))
    ax.barh(yy, rhos, color=[GRN if p < 0.05 else GRY for p in ps], alpha=0.9)
    ax.axvline(0, color=INK, lw=1.0)
    for i, (r_, p_) in enumerate(zip(rhos, ps)):
        ax.text(r_ + (0.012 if r_ >= 0 else -0.012), i, f"{r_:+.2f} (p={p_:.3f})",
                va="center", ha="left" if r_ >= 0 else "right", fontsize=7.2)
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("Spearman ρ with n_parameters (n=36 models)")
    ax.set_xlim(min(rhos) - 0.16, max(rhos) + 0.20)
    ax.set_title("(c) effect sizes vs n_parameters; green = permutation p < 0.05", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 1]
    bands = ["low", "mid", "high"]
    positions = np.arange(3)
    data2 = [trade.loc[trade.complexity_band == b, "param_displacement_median"].to_numpy() for b in bands]
    bp2 = ax.boxplot(data2, positions=positions, widths=0.5, patch_artist=True,
                     medianprops=dict(color=INK), showfliers=False)
    for patch, b in zip(bp2["boxes"], bands):
        patch.set_facecolor(BAND_COLORS[b]); patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{b}\n(n_p {trade.loc[trade.complexity_band==b,'n_parameters'].min()}"
                        f"–{trade.loc[trade.complexity_band==b,'n_parameters'].max()}; "
                        f"mean ΔKGE={trade.loc[trade.complexity_band==b,'delta_kge_pub_median'].mean():.3f})"
                        for b in bands], fontsize=8.5)
    ax.set_ylabel("model-median SEEN → OOB displacement\n(per-parameter normalised RMS)")
    ax.set_title("(d) complexity tertiles (12 models each); ΔKGE annotated, not co-plotted", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Complexity trade-off overview. n_parameters 1–15 (median 7). Bands are rank tertiles, not tuned "
                 "cut-offs.", fontsize=9.5, y=0.995)
    save(fig, "complexity_tradeoff_overview")


if __name__ == "__main__":
    main()
