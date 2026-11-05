#!/usr/bin/env python3
"""09_paper_level_figures.py — PNG figures for the paper-level preliminary OOB stats.

Writes ONLY .png (project plot-output rule) into
$OUT/paper_level_stats_20260930/figures/.

    PL_F1_ranked_oob_and_delta.png   per-model OOB median KGE (seen -> OOB dumbbell)
                                     + median delta_KGE with basin-clustered CI
    PL_F2_model_seen_vs_oob.png      model-level seen vs OOB median KGE, 1:1 line
    PL_F3_delta_by_fold.png          per-model median delta_KGE by fold (Q2 fold check)
    PL_F4_rank_shift.png             seen rank -> OOB rank slope graph
    PL_F5_nparam_vs_delta.png        complexity vs delta_KGE, OLS + Theil-Sen + LOO band
    PL_F6_basin_difficulty.png       basin-level median delta_KGE and model-agreement

Existing F1/F2/F3 in $OUT/figures/ remain the reference boxplot / seen-vs-OOB /
n_params figures for the row-level view; these new plots are the paper-level
(model-aggregated) companions.  No figure is overwritten.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402

OUT2 = os.path.join(OUT, "paper_level_stats_20260930")
T2 = os.path.join(OUT2, "tables")
F2 = os.path.join(OUT2, "figures")

INK = "#2b2b2b"
BLU = "#3b6ea5"
LBL = "#cfe3f3"
ORG = "#d1721b"
RED = "#b7332c"
GRY = "#8a8a8a"


def save(fig, stem):
    os.makedirs(F2, exist_ok=True)
    fig.savefig(os.path.join(F2, f"{stem}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    S1 = pd.read_csv(os.path.join(T2, "model_oob_summary.csv"))
    MF = pd.read_csv(os.path.join(T2, "model_fold_oob_summary.csv"))
    RS = pd.read_csv(os.path.join(T2, "model_rank_shift.csv"))
    BS = pd.read_csv(os.path.join(T2, "basin_oob_summary.csv"))
    LOO = pd.read_csv(os.path.join(T2, "leave_one_model_out_complexity.csv"))
    R = json.load(open(os.path.join(T2, "preliminary_statistics.json")))

    order = S1.sort_values("oob_kge_median", ascending=False)["model"].tolist()
    S1o = S1.set_index("model").loc[order]

    # ---------------------------------------------------------------- PL_F1
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                               gridspec_kw={"width_ratios": [1.0, 1.0]})
    y = np.arange(len(order))
    a.hlines(y, S1o["oob_kge_median"], S1o["seen_kge_median"], color=GRY, lw=1.3, zorder=1)
    a.scatter(S1o["oob_kge_median"], y, s=44, color=BLU, zorder=3, label="OOB (held-out basins)")
    a.scatter(S1o["seen_kge_median"], y, s=44, facecolor="white", edgecolor=ORG, lw=1.6,
              zorder=3, label="seen-basin dPL")
    a.set_yticks(y); a.set_yticklabels(order, fontsize=7.5)
    a.invert_yaxis()
    a.set_xlabel("median KGE over 531 OOB basins")
    a.set_title("(a) per-model median KGE: seen vs OOB\nmodels ordered by OOB median", fontsize=10)
    a.legend(fontsize=8, loc="lower right"); a.grid(axis="x", alpha=0.25)
    a.axvline(0, color="0.7", lw=0.8, ls="--")

    b.hlines(y, S1o["delta_median_ci95_lo"], S1o["delta_median_ci95_hi"], color=LBL, lw=4.5, zorder=1)
    b.scatter(S1o["delta_median"], y, s=40, color=RED, zorder=3)
    b.axvline(0, color=INK, lw=1.0)
    b.axvline(float(np.median(S1o["delta_median"])), color=BLU, lw=1.2, ls="--",
              label=f"cross-model median = {np.median(S1o['delta_median']):.4f}")
    b.set_yticks(y); b.set_yticklabels([]); b.invert_yaxis()
    b.set_xlabel("median $\\Delta$KGE = KGE$_{seen-dPL}$ - KGE$_{OOB}$   (>0 = OOB penalty)")
    b.set_title("(b) per-model median $\\Delta$KGE with basin-clustered 95% CI\n"
                "36/36 models positive; exact sign test p = 2.9e-11", fontsize=10)
    b.legend(fontsize=8, loc="lower right"); b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_F1 — magnitude and universality of the OOB penalty (36 models x 531 basins). "
                 "Descriptive; no recommendation content.", fontsize=10.5, y=1.01)
    save(fig, "PL_F1_ranked_oob_and_delta")

    # ---------------------------------------------------------------- PL_F2
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.plot([0.3, 0.72], [0.3, 0.72], color="0.6", lw=1.0, ls="--", zorder=1, label="1:1")
    ax.scatter(RS["seen_kge_median"], RS["oob_kge_median"], s=52, color=BLU,
               edgecolor="white", lw=0.6, zorder=3)
    lim = max(RS["seen_kge_median"].max(), RS["oob_kge_median"].max()) + 0.02
    for r in RS.itertuples():
        dx = 0.006 if r.model not in ("hbv96", "mopex5") else -0.006
        ha = "left" if dx > 0 else "right"
        ax.annotate(r.model, (r.seen_kge_median, r.oob_kge_median),
                    xytext=(r.seen_kge_median + dx, r.oob_kge_median), fontsize=7,
                    ha=ha, va="center", color=INK)
    ax.set_xlabel("median KGE on seen basins (dPL setting)")
    ax.set_ylabel("median KGE on 531 OOB basins")
    ax.set_title("PL_F2 — model-level seen vs OOB median KGE (n=36 models)\n"
                 f"Spearman $\\rho$ = {R['Q3_ranking']['spearman_model_median_seen_vs_oob']:.4f} "
                 f"(perm p = {R['Q3_ranking']['spearman_perm_p']:.4f}), "
                 f"Kendall $\\tau$ = {R['Q3_ranking']['kendall_tau_model_median_seen_vs_oob']:.4f}\n"
                 "all points below the 1:1 line = OOB penalty; ordering is nearly preserved",
                 fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    save(fig, "PL_F2_model_seen_vs_oob")

    # ---------------------------------------------------------------- PL_F3
    mord = S1.sort_values("delta_median", ascending=False)["model"].tolist()
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(mord))
    for i, m in enumerate(mord):
        g = MF[MF["model"] == m].sort_values("fold")
        ax.scatter(np.full(len(g), i) + np.linspace(-0.16, 0.16, len(g)),
                   g["median_delta_KGE"], s=16, color=GRY, zorder=2)
    pooled = S1.set_index("model").loc[mord, "delta_median"].to_numpy(float)
    lo = S1.set_index("model").loc[mord, "delta_median_ci95_lo"].to_numpy(float)
    hi = S1.set_index("model").loc[mord, "delta_median_ci95_hi"].to_numpy(float)
    ax.errorbar(x, pooled, yerr=[pooled - lo, hi - pooled], fmt="o", ms=6, color=RED,
                elinewidth=1.3, capsize=2.5, zorder=3, label="pooled median $\\Delta$KGE (95% CI, basin-clustered)")
    sd_between = R["Q2_heterogeneity"]["model_median_delta_distribution"]["sd_ddof1"]
    m0 = float(np.median(pooled))
    ax.axhspan(m0 - sd_between, m0 + sd_between, color=LBL, alpha=0.5, zorder=0,
               label=f"$\\pm$1 between-model SD = {sd_between:.4f}")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(mord, rotation=90, fontsize=7.5)
    ax.set_ylabel("median $\\Delta$KGE per fold  (grey dots = folds 0-4)")
    ax.set_title("PL_F3 — fold-to-fold variability of the penalty vs between-model spread\n"
                 f"all 180 model x fold cells positive; mean within-model fold SD = "
                 f"{MF.groupby('model')['median_delta_KGE'].std(ddof=1).mean():.4f} "
                 f"$\\approx$ between-model SD = {sd_between:.4f} -> model ordering is "
                 f"not robust to fold choice", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    save(fig, "PL_F3_delta_by_fold")

    # ---------------------------------------------------------------- PL_F4
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    big = RS[RS["abs_rank_shift"] >= 5]
    for r in RS.itertuples():
        hi_ = r.model in set(big["model"])
        ax.plot([0, 1], [r.seen_rank, r.oob_rank],
                color=RED if hi_ else GRY, lw=2.2 if hi_ else 1.0,
                alpha=0.95 if hi_ else 0.6, zorder=3 if hi_ else 2)
        ax.annotate(r.model, (-0.035, r.seen_rank), ha="right", va="center", fontsize=7,
                    color=RED if hi_ else INK)
        ax.annotate(r.model, (1.035, r.oob_rank), ha="left", va="center", fontsize=7,
                    color=RED if hi_ else INK)
    ax.set_xlim(-0.42, 1.42); ax.invert_yaxis()
    ax.set_xticks([0, 1]); ax.set_xticklabels(["seen-basin rank", "OOB rank"])
    ax.set_ylabel("rank among 36 models (1 = best median KGE)")
    ax.set_title("PL_F4 — does OOB re-rank the models?\n"
                 f"Spearman $\\rho$ = {R['Q3_ranking']['spearman_model_median_seen_vs_oob']:.4f}; "
                 f"median |shift| = {R['Q3_ranking']['rank_shift_stats']['median_abs_shift']:.0f}, "
                 f"max |shift| = {R['Q3_ranking']['rank_shift_stats']['max_abs_shift']}; "
                 f"top-5 set retained 5/5\n"
                 "red/red = the 4 models moving 5 or more places", fontsize=9.5)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "PL_F4_rank_shift")

    # ---------------------------------------------------------------- PL_F5
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    xnp = RS["n_params"].to_numpy(float)
    yd = RS["delta_median"].to_numpy(float)
    ax.scatter(xnp, yd, s=52, color=BLU, edgecolor="white", lw=0.6, zorder=3)
    for r in RS.itertuples():
        ax.annotate(r.model, (r.n_params, r.delta_median), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7, color=INK)
    xs = np.linspace(xnp.min(), xnp.max(), 50)
    ols = np.polyfit(xnp, yd, 1)
    ax.plot(xs, np.polyval(ols, xs), color=ORG, lw=1.8, label=f"OLS slope = {ols[0]:.5f} / param")
    ts = R["Q4_complexity"]["theil_sen_slope"]
    x0, y0 = float(np.median(xnp)), float(np.median(yd))
    ax.plot(xs, y0 + ts * (xs - x0), color=RED, lw=1.8, ls="--",
            label=f"Theil-Sen slope = {ts:.5f} / param")
    lr = R["Q4_complexity"]["leave_one_model_out_rho"]
    ax.set_xlabel("number of model parameters (frozen v1_parsimony.csv)")
    ax.set_ylabel("median $\\Delta$KGE over 531 OOB basins")
    ax.set_title("PL_F5 — complexity vs OOB penalty (exploratory, descriptive)\n"
                 f"Spearman $\\rho$ = {R['Q4_complexity']['spearman_rho']:.4f} "
                 f"(perm p = {R['Q4_complexity']['spearman_perm_p']:.3f}); "
                 f"Kendall $\\tau$ = {R['Q4_complexity']['kendall_tau']:.4f}\n"
                 f"leave-one-model-out $\\rho$ stays in [{lr['min']:.3f}, {lr['max']:.3f}] "
                 f"-> weak, directionally positive evidence only", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    save(fig, "PL_F5_nparam_vs_delta")

    # ---------------------------------------------------------------- PL_F6
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    a.hist(BS["delta_median"], bins=48, color=BLU, alpha=0.85, edgecolor="white", lw=0.4)
    a.axvline(0, color=INK, lw=1.0)
    a.axvline(float(BS["delta_median"].median()), color=RED, lw=1.4, ls="--",
              label=f"median = {BS['delta_median'].median():.4f}")
    a.set_xlabel("per-basin median $\\Delta$KGE over 36 models")
    a.set_ylabel("number of basins (of 531)")
    a.set_title("(a) basin-level penalty distribution\n"
                f"IQR = {R['B_basin_difficulty']['basin_median_delta_distribution']['iqr']:.4f}; "
                f"range {BS['delta_median'].min():.3f} to {BS['delta_median'].max():.3f}",
                fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    b.hist(BS["frac_models_delta_gt_0"] * 36, bins=np.arange(0, 38, 1), color=ORG, alpha=0.85,
           edgecolor="white", lw=0.4)
    b.axvline(36, color=RED, lw=1.4, ls="--",
              label=f"all 36 degrade: {R['B_basin_difficulty']['n_basins_all36_models_degrade']} basins")
    b.axvline(18, color=GRY, lw=1.4, ls=":",
              label=f"<=18 of 36 degrade: {R['B_basin_difficulty']['n_basins_le_half_models_degrade']}")
    b.set_xlabel("number of the 36 models with $\\Delta$KGE > 0 on that basin")
    b.set_ylabel("number of basins")
    b.set_title("(b) how universal is the penalty per basin?\n"
                f"245 basins with >=30/36 models degrading; 41 with all 36", fontsize=10)
    b.legend(fontsize=8); b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_F6 — basin-level heterogeneity of the OOB penalty (check B). "
                 "Descriptive; no basin attribution or mechanism analysis.", fontsize=10.5, y=1.03)
    save(fig, "PL_F6_basin_difficulty")

    print(f"\nfigures -> {F2}")


if __name__ == "__main__":
    main()
