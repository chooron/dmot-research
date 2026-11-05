#!/usr/bin/env python3
"""20_supplemental_figures.py — PNG figures for the supplemental analysis.

    basin_rank_correlation_distribution.png   within-basin Spearman/Kendall distributions
    top3_top5_jaccard_distribution.png        Top-3 / Top-5 overlap and intersection sizes
    winner_retention_vs_rank_stability.png    winner change vs the rest of the ordering
    extreme_penalty_metric_diagnostics.png    top-5% penalty basins vs the rest

Writes ONLY .png.  Reads only the tables written by scripts 18 and 19.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

INK, BLU, LBL, ORG, RED, GRY, GRN = "#2b2b2b", "#3b6ea5", "#cfe3f3", "#d1721b", "#b7332c", "#8a8a8a", "#2e7d5b"
KGE_EPS = 0.1


def save(fig, stem):
    PC.F3.mkdir(parents=True, exist_ok=True)
    fig.savefig(PC.F3 / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    RS = pd.read_csv(PC.T3 / "basin_rank_stability.csv")
    T = pd.read_csv(PC.T3 / "topk_overlap_summary.csv")
    WS = pd.read_csv(PC.T3 / "winner_vs_rank_stability.csv")
    G = pd.read_csv(PC.T3 / "extreme_basin_group_comparison.csv")
    S = pd.read_csv(PC.T3 / "pub_penalty_sensitivity.csv")
    D = pd.read_csv(PC.T3 / "extreme_basin_metric_diagnostics.csv")
    E = json.load(open(PC.T3 / "extreme_basin.json"))

    # ---------------------------------------------------------------- F1
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    rho = RS["spearman_rho"].to_numpy()
    tau = RS["kendall_tau"].to_numpy()
    a.hist(rho, bins=40, range=(-1, 1), color=BLU, alpha=0.8, edgecolor="white", lw=0.3,
           label=f"Spearman (median {np.median(rho):.3f})")
    a.hist(tau, bins=40, range=(-1, 1), color=ORG, alpha=0.6, edgecolor="white", lw=0.3,
           label=f"Kendall (median {np.median(tau):.3f})")
    a.axvline(0, color=INK, lw=1.0)
    a.axvline(np.median(rho), color=BLU, ls="--", lw=1.4)
    a.axvline(0.97, color=GRN, ls=":", lw=1.8,
              label="aggregate model-level $\\rho$ = 0.970")
    a.set_xlabel("within-basin rank correlation of the 36 model scores (seen vs OOB)")
    a.set_ylabel("number of basins (of 531)")
    a.set_title("(a) basin-level rank correlation\n"
                "bulk of the ordering is preserved, but far less than the aggregate 0.97", fontsize=10)
    a.legend(fontsize=8, loc="upper left"); a.grid(axis="y", alpha=0.25)

    xs = np.sort(rho)
    a2 = b
    a2.plot(xs, np.arange(1, xs.size + 1) / xs.size, color=BLU, lw=2.0, label="Spearman")
    xs2 = np.sort(tau)
    a2.plot(xs2, np.arange(1, xs2.size + 1) / xs2.size, color=ORG, lw=2.0, label="Kendall")
    for v, lab in ((0.5, "0.5"), (0.7, "0.7")):
        a2.axvline(v, color=GRY, ls=":", lw=1.2)
        a2.text(v + 0.01, 0.05, f"$\\rho$={lab}", fontsize=8, color=GRY)
    for q, c in ((0.25, RED), (0.75, GRN)):
        a2.axhline(q, color=c, ls="--", lw=0.9, alpha=0.7)
    a2.set_xlim(-0.2, 1.0); a2.set_ylim(0, 1)
    a2.set_xlabel("rank correlation"); a2.set_ylabel("cumulative fraction of basins")
    a2.set_title("(b) ECDF\n"
                f"$\\rho\\geq$0.5 in {float((rho>=0.5).mean())*100:.0f}% of basins; "
                f"$\\rho\\geq$0.7 in {float((rho>=0.7).mean())*100:.0f}%", fontsize=10)
    a2.legend(fontsize=8, loc="lower right"); a2.grid(alpha=0.25)
    fig.suptitle("PL_SUP_F1 — within-basin stability of the full 36-model ordering "
                 "(same-pipeline seen vs OOB held-out). Descriptive.", fontsize=10.5, y=1.02)
    save(fig, "basin_rank_correlation_distribution")

    # ---------------------------------------------------------------- F2
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    j3 = RS["top3_jaccard"].to_numpy(); j5 = RS["top5_jaccard"].to_numpy()
    bins = np.linspace(0, 1, 21)
    a.hist(j3, bins=bins, color=RED, alpha=0.7, edgecolor="white", lw=0.3,
           label=f"Top-3 (median {np.median(j3):.2f})")
    a.hist(j5, bins=bins, color=BLU, alpha=0.6, edgecolor="white", lw=0.3,
           label=f"Top-5 (median {np.median(j5):.2f})")
    a.set_xlabel("Jaccard similarity of the seen-top-k and OOB-top-k model sets")
    a.set_ylabel("number of basins")
    a.set_title("(a) Top-k set overlap\n"
                f"Top-3 sets are identical in only {float((j3==1).mean())*100:.1f}% of basins", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    w = 0.38
    for i, k in enumerate((3, 5)):
        row = T[(T.variant == "samepipeline_median4") & (T.k == k)].iloc[0]
        fracs = [row[f"frac_intersection_{j}"] for j in range(k + 1)]
        b.bar(np.arange(k + 1) + (i - 0.5) * w, fracs, width=w,
              color=[RED, BLU][i], alpha=0.85, label=f"Top-{k} (mean overlap {row['intersection_mean']:.2f}/{k})")
    b.set_xticks(range(6))
    b.set_xlabel("number of models shared between the seen and OOB top-k sets")
    b.set_ylabel("fraction of basins")
    b.set_title("(b) intersection size distribution\n"
                "the head of the ranking is much less transferable than the bulk", fontsize=10)
    b.legend(fontsize=8); b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_SUP_F2 — Top-3 / Top-5 model-set transferability. Descriptive; no "
                 "candidate or equivalence set is constructed.", fontsize=10.5, y=1.02)
    save(fig, "top3_top5_jaccard_distribution")

    # ---------------------------------------------------------------- F3
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1, 1.05]})
    order = ["winner RETAINED", "winner CHANGED", "decided AND changed (the hard cases)", "all basins"]
    labels = ["winner\nretained\n(n=134)", "winner\nchanged\n(n=397)",
              "decided AND\nchanged\n(n=181)", "all basins\n(n=531)"]
    merged = []
    for g in order:
        if g == "winner RETAINED":
            merged.append(WS.loc[WS.strict_winner_retained, "spearman_rho"].to_numpy())
        elif g == "winner CHANGED":
            merged.append(WS.loc[~WS.strict_winner_retained, "spearman_rho"].to_numpy())
        elif g.startswith("decided AND"):
            merged.append(WS.loc[WS.near_tie_decided & ~WS.strict_winner_retained, "spearman_rho"].to_numpy())
        else:
            merged.append(WS["spearman_rho"].to_numpy())
    bp = a.boxplot(merged, widths=0.55, showfliers=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.3))
    for p, c in zip(bp["boxes"], [GRN, RED, ORG, GRY]):
        p.set_facecolor(c); p.set_alpha(0.45); p.set_edgecolor(INK)
    a.axhline(float(RS["spearman_rho"].median()), color=INK, ls="--", lw=1.1,
              label=f"all-basin median = {RS['spearman_rho'].median():.3f}")
    a.set_xticklabels(labels, fontsize=8)
    a.set_ylabel("within-basin Spearman $\\rho$")
    a.set_title("(a) rank stability by winner outcome\n"
                "winner changes sit in somewhat less stable basins, but those basins still "
                "have median $\\rho\\approx$0.70", fontsize=10)
    a.legend(fontsize=8, loc="lower left"); a.grid(axis="y", alpha=0.25)

    b.scatter(WS["spearman_rho"], WS["top3_jaccard"], s=16,
              c=np.where(WS["strict_winner_retained"], GRN, RED), alpha=0.7)
    b.axvline(float(RS["spearman_rho"].median()), color=INK, ls="--", lw=1.0)
    b.axhline(float(RS["top3_jaccard"].median()), color=INK, ls=":", lw=1.0)
    b.set_xlabel("within-basin Spearman $\\rho$ (full 36-model ordering)")
    b.set_ylabel("Top-3 Jaccard")
    b.set_title("(b) bulk ordering vs the head of the ranking\n"
                "green = top-1 winner retained, red = changed", fontsize=10)
    b.grid(alpha=0.25)
    fig.suptitle("PL_SUP_F3 — does top-1 instability imply a broader ranking collapse? "
                 "Partly: the bulk holds, the head does not.", fontsize=10.5, y=1.02)
    save(fig, "winner_retention_vs_rank_stability")

    # ---------------------------------------------------------------- F4
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.1])

    a = fig.add_subplot(gs[0, 0])
    ext = D["extreme_top5pct"].to_numpy()
    for vals, c, lab, off in ((D.loc[ext, "obs_mean_recomputed"], RED, "top 5% penalty", -0.16),
                              (D.loc[~ext, "obs_mean_recomputed"], BLU, "remaining 95%", 0.16)):
        v = np.maximum(vals.to_numpy(float), 1e-4)
        a.scatter(np.full(v.size, off) + np.random.default_rng(0).normal(0, 0.045, v.size),
                  v, s=9, color=c, alpha=0.55, label=f"{lab} (median {np.median(vals):.2f})")
        a.hlines(np.median(vals), off - 0.11, off + 0.11, color=INK, lw=2.2)
    a.axhline(KGE_EPS, color=ORG, ls="--", lw=1.6, label=f"KGE_EPS = {KGE_EPS} mm/day")
    a.set_yscale("log"); a.set_xticks([0, 1]); a.set_xticklabels(["top 5%", "rest"], fontsize=9)
    a.set_ylabel("mean observed discharge (mm/day, log scale)")
    a.set_title("(a) extreme basins are drier, but mostly NOT below KGE_EPS\n"
                f"{int(E['flags']['flag_obs_mean_lt_KGE_EPS']['n_in_extreme_top5pct'])}/"
                f"{E['meta']['n_extreme']} of the top 5% are below it", fontsize=9.5)
    a.legend(fontsize=7.5, loc="lower left"); a.grid(axis="y", alpha=0.25)

    a = fig.add_subplot(gs[0, 1])
    Gs = G[G.diagnostic.isin(["oob_kge_median", "oob_kge_sd_over_models", "obs_std",
                              "obs_mean_recomputed", "runoff_ratio", "arid",
                              "near_zero_flow_frac_lt_0p01", "zero_flow_frac_recomputed",
                              "obs_cv", "q_log_sd", "missing_frac"])].copy()
    Gs = Gs.sort_values("cliffs_delta_extreme_vs_rest")
    colr = [RED if abs(v) > 0.5 else ORG if abs(v) > 0.25 else GRY
            for v in Gs["cliffs_delta_extreme_vs_rest"]]
    a.barh(np.arange(len(Gs)), Gs["cliffs_delta_extreme_vs_rest"], color=colr, alpha=0.9)
    a.axvline(0, color=INK, lw=1.0)
    for v, lab in ((0.5, ""), (-0.5, "")):
        a.axvline(v, color=GRY, ls=":", lw=1.0)
    a.set_yticks(np.arange(len(Gs)))
    a.set_yticklabels(Gs["diagnostic"], fontsize=7.5)
    a.set_xlabel("Cliff's $\\delta$ (top 5% vs the rest)")
    a.set_title("(b) which diagnostics separate the extreme set?\n"
                "model failure and model dispersion dominate, not KGE pathology", fontsize=9.5)
    a.grid(axis="x", alpha=0.25)

    a = fig.add_subplot(gs[0, 2])
    Sm = S[S.aggregation == "median4"].copy()
    Sm["short"] = (Sm["scenario"].str.replace("exclude ", "excl. ", regex=False)
                                    .str.replace(" basins", "", regex=False))
    y = np.arange(len(Sm))
    a.barh(y, Sm["pooled_median"], color=[GRY if s.startswith("all") else
                                          (RED if "extreme top 5%" in s or "union" in s else BLU)
                                          for s in Sm["scenario"]], alpha=0.9)
    a.errorbar(Sm["pooled_median"], y,
               xerr=[Sm["pooled_median"] - Sm["pooled_median_ci95_lo"],
                     Sm["pooled_median_ci95_hi"] - Sm["pooled_median"]],
               fmt="none", ecolor=INK, elinewidth=1.0, capsize=2)
    base = float(Sm[Sm.scenario.str.startswith("all")].pooled_median.iloc[0])
    a.axvline(base, color=INK, ls="--", lw=1.2, label=f"all basins = {base:.4f}")
    a.set_yticks(y); a.set_yticklabels(Sm["short"], fontsize=7.5); a.invert_yaxis()
    a.set_xlabel("pooled median $\\Delta$KGE$_{PUB}$ (95% CI)")
    a.set_title("(c) headline sensitivity\n"
                "KGE-pathological removal changes almost nothing; removing the "
                "extreme 5% costs ~0.005", fontsize=9.5)
    a.legend(fontsize=8); a.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_SUP_F4 — are the extreme-penalty basins a KGE numerical artefact? "
                 "Low-flow associated, but not pathologically so.", fontsize=10.5, y=1.03)
    save(fig, "extreme_penalty_metric_diagnostics")

    print(f"\nfigures -> {PC.F3}")


if __name__ == "__main__":
    main()
