#!/usr/bin/env python3
"""16_next_stage_figures.py — PNG figures for the OOB next-stage analysis.

Writes ONLY .png (project plot-output rule) into $OUT2/figures/:

    pub_model_penalty.png              per-model same-pipeline PUB penalty + separation
    pub_basin_penalty_distribution.png per-basin penalty distribution and fold breakdown
    model_vs_basin_heterogeneity.png   variance partition and the robust spread analog
    winner_margin_and_retention.png    winner margins, near-tie sensitivity, seed baseline
    basin_attribute_screening.png      attribute screen and grouped quartile summaries

Reads only the tables written by scripts 11-15.
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

F = PC.OUT2 / "figures"
INK, BLU, LBL, ORG, RED, GRY, GRN = "#2b2b2b", "#3b6ea5", "#cfe3f3", "#d1721b", "#b7332c", "#8a8a8a", "#2e7d5b"


def save(fig, stem):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    M = pd.read_csv(PC.T2 / "pub_model_summary.csv")
    B = pd.read_csv(PC.T2 / "pub_basin_summary.csv")
    WM = pd.read_csv(PC.T2 / "winner_margin_summary.csv")
    RS = pd.read_csv(PC.T2 / "winner_retention_sensitivity.csv")
    SB = pd.read_csv(PC.T2 / "seed_noise_winner_baseline.csv")
    SC = pd.read_csv(PC.T2 / "basin_attribute_screening.csv")
    Q = pd.read_csv(PC.T2 / "basin_attribute_quartiles.csv")
    BT = pd.read_csv(PC.T2 / "basin_attribute_table.csv")
    S = json.load(open(PC.T2 / "pub_stats.json"))
    W = json.load(open(PC.T2 / "winner_stability.json"))

    # ---------------------------------------------------------------- F1
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    o = M.sort_values("delta_pub_median", ascending=False).reset_index(drop=True)
    x = np.arange(len(o))
    a.errorbar(x, o["delta_pub_median"],
               yerr=[o["delta_pub_median"] - o["delta_pub_median_ci95_lo"],
                     o["delta_pub_median_ci95_hi"] - o["delta_pub_median"]],
               fmt="o", ms=6, color=RED, elinewidth=1.3, capsize=2.5,
               label="same-pipeline $\\Delta$KGE$_{PUB}$ (median of 4 seen folds, 95% CI)")
    a.plot(x, o["delta_existing_median"], "s", ms=4.5, color=BLU, alpha=0.85,
           label="previous cross-pipeline estimate (canonical seen-dPL $-$ OOB)")
    a.axhline(float(S["analysis1"]["median4"]["pooled_median"]), color=INK, lw=1.1, ls="--",
              label=f"pooled median = {S['analysis1']['median4']['pooled_median']:.4f}")
    a.set_xticks(x); a.set_xticklabels(o["model"], rotation=90, fontsize=7.5)
    a.set_ylabel("$\\Delta$KGE$_{PUB}$  (>0 = OOB penalty)")
    a.set_title("(a) per-model same-pipeline PUB penalty\n"
                f"36/36 positive; previous pooled median 0.0442 vs new 0.0434", fontsize=10)
    a.legend(fontsize=8, loc="upper right"); a.grid(axis="y", alpha=0.25)

    sep = M.set_index("model").loc[o["model"]]
    b.barh(x, sep["n_others_reliably_below_this_model"], color=RED, alpha=0.85,
           label="# models this one is reliably ABOVE (higher penalty)")
    b.barh(x, -sep["n_others_reliably_above_this_model"], color=BLU, alpha=0.85,
           label="# models this one is reliably BELOW (lower penalty)")
    b.axvline(0, color=INK, lw=1.0)
    b.axvline(17.5, color=GRY, ls=":", lw=1.2); b.axvline(-17.5, color=GRY, ls=":", lw=1.2)
    b.set_yticks(x); b.set_yticklabels(o["model"], fontsize=7.5); b.invert_yaxis()
    b.set_xlabel("number of other models (of 35) separable at 95% paired CI")
    b.set_title("(b) how separable is each model?\n"
                f"only {len(S['analysis2']['reliable_high_penalty_models'])} reliable high and "
                f"{len(S['analysis2']['reliable_low_penalty_models'])} reliable low extremes; "
                f"{len(S['analysis2']['models_unresolved_overlapping'])} unresolved", fontsize=10)
    b.legend(fontsize=7.5, loc="lower right"); b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_PUB_F1 — same-pipeline PUB penalty per model (36 models x 531 basins). "
                 "Descriptive; no recommendation content.", fontsize=10.5, y=1.01)
    save(fig, "pub_model_penalty")

    # ---------------------------------------------------------------- F2
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    a.hist(B["penalty_median_over_models"], bins=48, color=BLU, alpha=0.85, edgecolor="white", lw=0.4)
    a.axvline(0, color=INK, lw=1.1)
    a.axvline(B["penalty_median_over_models"].median(), color=RED, ls="--", lw=1.4,
              label=f"median = {B['penalty_median_over_models'].median():.4f}")
    a.set_xlabel("per-basin median $\\Delta$KGE$_{PUB}$ over 36 models")
    a.set_ylabel("number of basins (of 531)")
    a.set_title("(a) basin-level PUB penalty\n"
                f"IQR = {B['penalty_median_over_models'].quantile(.75)-B['penalty_median_over_models'].quantile(.25):.4f}, "
                f"{float((B['penalty_median_over_models']>0).mean())*100:.1f}% of basins positive", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    folds = pd.read_parquet(PC.MERGED)[["gauge_id", "fold"]].drop_duplicates()
    BB = B.merge(folds.rename(columns={"gauge_id": "basin_id"}), on="basin_id", how="left")
    data = [BB.loc[BB["fold"] == k, "penalty_median_over_models"].to_numpy() for k in range(5)]
    bp = b.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.2))
    for p in bp["boxes"]:
        p.set_facecolor(LBL); p.set_edgecolor(BLU)
    b.axhline(0, color=INK, lw=1.0)
    b.set_xticklabels([f"fold {k}\nn={len(d)}" for k, d in enumerate(data)], fontsize=8)
    b.set_ylabel("per-basin median $\\Delta$KGE$_{PUB}$")
    b.set_title("(b) by held-out fold\nleave-one-fold-out pooled medians span only 0.0405-0.0452", fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PUB_F2 — basin-level distribution of the same-pipeline PUB penalty. "
                 "Descriptive.", fontsize=10.5, y=1.02)
    save(fig, "pub_basin_penalty_distribution")

    # ---------------------------------------------------------------- F3
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    vd = pd.DataFrame(S["analysis2"]["variance_decomposition"])
    vd = vd[vd["component"] != "total"]
    a.bar(range(len(vd)), vd["frac_of_total_ss"].astype(float), color=[GRN, ORG, GRY])
    for i, v in enumerate(vd["frac_of_total_ss"].astype(float)):
        a.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    a.set_xticks(range(len(vd))); a.set_xticklabels(vd["component"], fontsize=9)
    a.set_ylabel("fraction of total sum of squares")
    a.set_ylim(0, 0.75)
    a.set_title("(a) balanced two-way partition of $\\Delta$KGE$_{PUB}$\n"
                f"basin {vd.iloc[0]['frac_of_total_ss']:.3f} vs model {vd.iloc[1]['frac_of_total_ss']:.3f} "
                f"(MS ratio {S['analysis2']['variance_decomposition_ms_ratio']['basin_ms_over_model_ms']:.1f})",
                fontsize=10)
    a.grid(axis="y", alpha=0.25)

    D = pd.read_csv(PC.T2 / "pub_same_pipeline_model_basin.csv")
    pv = D.pivot(index="basin_id", columns="model", values="delta_kge_pub_median4")
    arr = pv.to_numpy()
    wb = np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)   # across models, within basin
    wm = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)   # across basins, within model
    a2 = b.boxplot([wb, wm], widths=0.55, showfliers=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.2))
    for p, c in zip(a2["boxes"], [GRN, ORG]):
        p.set_facecolor(c); p.set_alpha(0.5); p.set_edgecolor(INK)
    b.set_xticklabels([f"across models\nwithin a basin\n(n={len(wb)})",
                       f"across basins\nwithin a model\n(n={len(wm)})"], fontsize=8.5)
    b.set_ylabel("IQR of $\\Delta$KGE$_{PUB}$")
    b.set_title("(b) robust spread analog\n"
                f"median within-model across-basin spread is "
                f"{np.median(wm)/np.median(wb):.2f}x the within-basin across-model spread", fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PUB_F3 — basin heterogeneity dominates model heterogeneity "
                 "(and the model effect is still real: permutation p = "
                 f"{S['analysis2']['omnibus_permutation']['perm_p']:.4f}).", fontsize=10.5, y=1.02)
    save(fig, "model_vs_basin_heterogeneity")

    # ---------------------------------------------------------------- F4
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.15])
    a = fig.add_subplot(gs[0, 0])
    same = WM["same_winner"].to_numpy()
    a.scatter(WM.loc[same, "seen_margin12"], WM.loc[same, "oob_margin12"], s=13, color=BLU,
              alpha=0.75, label=f"winner retained (n={int(same.sum())})")
    a.scatter(WM.loc[~same, "seen_margin12"], WM.loc[~same, "oob_margin12"], s=13, color=RED,
              alpha=0.65, label=f"winner changed (n={int((~same).sum())})")
    a.set_xlabel("seen top1$-$top2 margin"); a.set_ylabel("OOB top1$-$top2 margin")
    a.set_title("(a) winner margins per basin\nmany swaps occur at LARGE margins", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.25)

    a = fig.add_subplot(gs[0, 1])
    ch = WM[WM["winner_changed"]]
    a.hist(ch["oob_gap_old_new_winner"], bins=44, color=RED, alpha=0.85, edgecolor="white", lw=0.3)
    a.axvline(ch["oob_gap_old_new_winner"].median(), color=INK, ls="--", lw=1.4,
              label=f"median = {ch['oob_gap_old_new_winner'].median():.4f}")
    a.axvline(WM["oob_margin12"].median(), color=BLU, ls=":", lw=1.4,
              label=f"typical top1$-$top2 margin = {WM['oob_margin12'].median():.4f}")
    a.set_xlabel("OOB KGE gap between outgoing and incoming winner")
    a.set_ylabel("number of changed basins")
    a.set_title("(b) are the swaps cheap?\n"
                f"only {float((ch['oob_gap_old_new_winner']<0.01).mean())*100:.1f}% are below 0.01 KGE",
                fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    a = fig.add_subplot(gs[0, 2])
    seed3 = float(SB.loc[SB["seed_a"] == "all", "retention"].iloc[0])
    seedpair = float(SB.loc[SB["seed_a"] != "all", "retention"].mean())
    pc = W.get("step3d_paired_contrast", {})
    sp8 = pc.get("same_pipeline_oob_agreement_mean", np.nan)
    labels = ["seed only\n(3-way)", "seed only\n(pair mean)", "same-pipeline\nOOB (8 models)",
              "same-pipeline\nOOB (36 models)"]
    vals = [seed3, seedpair, sp8, W["step3c_same_pipeline_winner"]["median"]["strict_retention"]]
    cols = [GRY, GRY, RED, RED]
    a.bar(range(4), vals, color=cols, alpha=0.9)
    for i, v in enumerate(vals):
        a.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8)
    a.set_ylabel("fraction of basins with the same winner")
    a.set_ylim(0, 0.78)
    a.set_title("(c) winner agreement vs a seed-only baseline\n"
                f"excess instability = {pc.get('difference', float('nan')):.3f} "
                f"CI95 [{pc.get('ci95', [np.nan, np.nan])[0]:.3f}, {pc.get('ci95', [np.nan, np.nan])[1]:.3f}]",
                fontsize=10)
    a.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PUB_F4 — basin-wise winner stability with near-tie and seed-noise control. "
                 "Descriptive; no equivalence set was defined.", fontsize=10.5, y=1.03)
    save(fig, "winner_margin_and_retention")

    # ---------------------------------------------------------------- F5
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    s = SC.sort_values("spearman_rho")
    yv = np.arange(len(s))
    cols = [RED if (q < 0.05 and r > 0) else (BLU if q < 0.05 else GRY)
            for q, r in zip(s["bh_fdr_q"], s["spearman_rho"])]
    a.barh(yv, s["spearman_rho"], color=cols, alpha=0.9)
    a.axvline(0, color=INK, lw=1.0)
    a.set_yticks(yv); a.set_yticklabels(s["variable"], fontsize=8)
    a.set_xlabel("Spearman $\\rho$ with per-basin median $\\Delta$KGE$_{PUB}$")
    a.set_title("(a) basin-attribute screen (n=531, BH-FDR over 15 variables)\n"
                f"nothing survives q<0.05 (smallest q = {s['bh_fdr_q'].min():.3f}); "
                "all |rho| <= 0.12", fontsize=10)
    a.grid(axis="x", alpha=0.25)

    top3 = SC.reindex(SC["spearman_rho"].abs().sort_values(ascending=False).index)["variable"].head(3).tolist()
    for i, var in enumerate(top3):
        qq = Q[Q["variable"] == var].sort_values("quartile")
        a2 = b if i == 0 else None
        if i == 0:
            xs = np.arange(len(qq))
            b.bar(xs, qq["penalty_median"], color=ORG, alpha=0.9)
            for j, v in enumerate(qq["penalty_median"]):
                b.text(j, v + 0.0015, f"{v:.4f}", ha="center", fontsize=8)
            b.set_xticks(xs)
            b.set_xticklabels([f"{q}\nn={int(n)}" for q, n in zip(qq["quartile"], qq["n_basins"])], fontsize=8)
            b.set_ylabel("median per-basin $\\Delta$KGE$_{PUB}$")
            b.set_title(f"(b) grouped summary: {var}\nquartile medians (descriptive)", fontsize=10)
            b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PUB_F5 — basin-level PUB heterogeneity screening: weak, "
                 "directionally consistent (drier / lower-runoff basins penalised slightly more).",
                 fontsize=10.5, y=1.02)
    save(fig, "basin_attribute_screening")

    print(f"\nfigures -> {F}")


if __name__ == "__main__":
    main()
