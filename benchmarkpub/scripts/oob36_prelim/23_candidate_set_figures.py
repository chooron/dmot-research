#!/usr/bin/env python3
"""23_candidate_set_figures.py — PNG figures for the candidate-set transfer analysis.

    topk_oob_coverage_curve.png        coverage vs k, marginal gain, pool-matched seed ceiling
    topk_regret_curve.png              regret quantiles and regret-threshold curves
    basin_specific_vs_global_topk.png  paired basin-specific minus global comparison
    top3_top5_overlap_distribution.png same-k set overlap (Top-3 / Top-5) and recall curve

Writes ONLY .png.  Reads only the tables written by script 22.
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
KSMALL = np.arange(1, 11)


def save(fig, stem):
    PC.F4.mkdir(parents=True, exist_ok=True)
    fig.savefig(PC.F4 / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    C = pd.read_csv(PC.T4 / "topk_coverage_curve.csv")
    R = pd.read_csv(PC.T4 / "topk_regret_curve.csv")
    MG = pd.read_csv(PC.T4 / "topk_marginal_gain.csv")
    CMP = pd.read_csv(PC.T4 / "basin_specific_vs_global_topk.csv")
    O = pd.read_csv(PC.T4 / "top3_top5_set_overlap.csv")
    SB = pd.read_csv(PC.T4 / "seed_candidate_set_baseline.csv")

    bs = C[C.strategy == "basin_specific"].sort_values("k")
    gl = C[C.strategy == "global"].sort_values("k")

    # ---------------------------------------------------------------- F1
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.0, 5.0),
                                  gridspec_kw={"width_ratios": [1.25, 1.0, 1.05]})
    a.fill_between(bs["k"], bs["coverage_top1_ci95_lo"], bs["coverage_top1_ci95_hi"],
                   color=LBL, alpha=0.7)
    a.plot(bs["k"], bs["coverage_top1"], "o-", color=BLU, lw=2.0, ms=5,
           label="basin-specific Seen Top-k")
    a.fill_between(gl["k"], gl["coverage_top1_ci95_lo"], gl["coverage_top1_ci95_hi"],
                   color="#f2e0cf", alpha=0.7)
    a.plot(gl["k"], gl["coverage_top1"], "s-", color=ORG, lw=2.0, ms=5,
           label="global Top-k (fixed set)")
    a.axhline(1.0, color=GRY, ls=":", lw=1.0)
    for k, v in ((1, bs.loc[bs.k == 1, "coverage_top1"].iloc[0]),
                 (3, bs.loc[bs.k == 3, "coverage_top1"].iloc[0]),
                 (5, bs.loc[bs.k == 5, "coverage_top1"].iloc[0]),
                 (10, bs.loc[bs.k == 10, "coverage_top1"].iloc[0])):
        a.annotate(f"k={k}\n{v:.3f}", (k, v), textcoords="offset points", xytext=(4, 8),
                   fontsize=8, color=INK)
    a.set_xlabel("candidate-set size k"); a.set_ylabel("P(OOB top-1 $\\in$ Seen Top-k)")
    a.set_ylim(0, 1.05); a.set_xlim(0.5, 36.5); a.set_xticks([1, 3, 5, 10, 15, 20, 36])
    a.set_title("(a) OOB top-1 coverage vs candidate-set size\n"
                "k=1 0.252 → k=3 0.503 → k=5 0.627 → k=10 0.804", fontsize=10)
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.25)

    m = MG[MG.strategy == "basin_specific"].copy()
    m["lab"] = [f"{int(r.k_from)}→{int(r.k_to)}" for r in m.itertuples()]
    cols = [RED if v >= 0.05 else ORG if v >= 0.03 else GRY for v in m["coverage_gain"]]
    b.barh(np.arange(len(m)), m["coverage_gain"], color=cols, alpha=0.9)
    b.set_yticks(np.arange(len(m))); b.set_yticklabels(m["lab"], fontsize=7.5); b.invert_yaxis()
    b.axvline(0.03, color=GRY, ls=":", lw=1.1)
    b.text(0.031, len(m) - 1.4, "3 pp", fontsize=7.5, color=GRY)
    b.set_xlabel("marginal OOB top-1 coverage gain")
    b.set_title("(b) marginal gain per extra model\ndiminishing returns set in after k≈5–6",
                fontsize=10)
    b.grid(axis="x", alpha=0.25)

    s = SB[SB.reference.str.startswith("seed->seed")]
    o8 = SB[SB.reference == "seen->OOB (same 8-model pool)"]
    o36 = SB[SB.reference == "seen->OOB (full 36-model pool)"]
    c.plot(s["k"], s["coverage_top1"], "o-", color=GRY, lw=2.0, ms=5,
           label="seed→seed (8 models, seed noise only)")
    c.plot(o8["k"], o8["coverage_top1"], "s-", color=RED, lw=2.0, ms=5,
           label="seen→OOB (same 8-model pool)")
    c.plot(o36["k"], o36["coverage_top1"], "^--", color=BLU, lw=1.6, ms=5,
           label="seen→OOB (36-model pool, not pool-matched)")
    for k in (1, 3, 5):
        g = float(s.loc[s.k == k, "coverage_top1"].iloc[0]) - float(o8.loc[o8.k == k, "coverage_top1"].iloc[0])
        c.annotate(f"gap {g*100:.1f}pp", (k, (float(s.loc[s.k == k, "coverage_top1"].iloc[0])
                                              + float(o8.loc[o8.k == k, "coverage_top1"].iloc[0])) / 2),
                   textcoords="offset points", xytext=(6, -2), fontsize=7.5, color=INK)
    c.set_xlabel("candidate-set size k"); c.set_ylabel("OOB top-1 coverage")
    c.set_ylim(0, 1.05); c.set_xticks(range(1, 9))
    c.set_title("(c) how much of the missing coverage is seed noise?\nthe seed-vs-transfer gap "
                "shrinks from 20.7 pp (k=1) to 4.0 pp (k=5)", fontsize=10)
    c.legend(fontsize=7.5, loc="lower right"); c.grid(alpha=0.25)
    fig.suptitle("PL_CS_F1 — candidate-set transfer: OOB top-1 coverage. Descriptive; no "
                 "candidate probability model and no equivalence set.", fontsize=10.5, y=1.02)
    save(fig, "topk_oob_coverage_curve")

    # ---------------------------------------------------------------- F2
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    rb = R[R.strategy == "basin_specific"].sort_values("k")
    rg = R[R.strategy == "global"].sort_values("k")
    a.plot(rb["k"], rb["regret_median"], "o-", color=BLU, lw=2.0, ms=5,
           label="basin-specific: median")
    a.fill_between(rb["k"], 0, rb["regret_q90"], color=LBL, alpha=0.6,
                   label="basin-specific: Q90")
    a.plot(rg["k"], rg["regret_median"], "s-", color=ORG, lw=2.0, ms=5,
           label="global: median")
    a.plot(rg["k"], rg["regret_q90"], "s--", color=ORG, lw=1.3, ms=4, alpha=0.7,
           label="global: Q90")
    a.axhline(0, color=INK, lw=1.0)
    a.set_xlabel("candidate-set size k"); a.set_ylabel("OOB regret (KGE)")
    a.set_xlim(0.5, 10.5); a.set_xticks(KSMALL)
    a.set_title("(a) regret: median hits exactly 0 at k=3 (basin-specific)\n"
                "the seen set usually already contains an OOB-optimal model", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.25)

    for t, cl in zip((0.01, 0.02, 0.05), (GRN, ORG, RED)):
        b.plot(rb["k"], rb[f"frac_regret_le_{t}"], "o-", color=cl, lw=2.0, ms=4.5,
               label=f"basin-specific: regret $\\leq$ {t}")
        b.plot(rg["k"], rg[f"frac_regret_le_{t}"], "s--", color=cl, lw=1.3, ms=4, alpha=0.65)
    b.axvline(3, color=INK, ls=":", lw=1.2)
    b.text(3.1, 0.06, "k=3", fontsize=8, color=INK)
    b.set_xlabel("candidate-set size k"); b.set_ylabel("fraction of basins")
    b.set_ylim(0, 1.0); b.set_xlim(0.5, 10.5); b.set_xticks(KSMALL)
    b.set_title("(b) fraction of basins with regret below a labelled threshold\n"
                "solid = basin-specific, dashed = global; thresholds are descriptive only",
                fontsize=10)
    b.legend(fontsize=8, loc="lower right"); b.grid(alpha=0.25)
    fig.suptitle("PL_CS_F2 — performance loss from using a seen candidate set instead of the OOB "
                 "oracle. No equivalence delta is defined.", fontsize=10.5, y=1.02)
    save(fig, "topk_regret_curve")

    # ---------------------------------------------------------------- F3
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    cm = CMP.sort_values("k")
    ks = cm["k"].to_numpy()
    a.errorbar(ks, cm["coverage_diff_bs_minus_global"] * 100,
               yerr=[(cm["coverage_diff_bs_minus_global"] - cm["coverage_diff_ci95_lo"]) * 100,
                     (cm["coverage_diff_ci95_hi"] - cm["coverage_diff_bs_minus_global"]) * 100],
               fmt="o", ms=6, color=BLU, elinewidth=1.4, capsize=3,
               label="basin-specific $-$ global (95% CI, basin bootstrap)")
    a.axhline(0, color=INK, lw=1.1)
    nsc = (cm["coverage_diff_ci95_lo"] <= 0).to_numpy()
    a.scatter(ks[nsc], cm["coverage_diff_bs_minus_global"].to_numpy()[nsc] * 100, s=120,
              facecolor="none", edgecolor=RED, lw=1.6, zorder=5,
              label="CI includes 0 (no established advantage)")
    a.set_xlabel("candidate-set size k"); a.set_ylabel("coverage difference (percentage points)")
    a.set_xlim(0.5, 20.5); a.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 15, 20])
    a.set_title("(a) is a basin-specific set better than a fixed global one?\n"
                "no at k=1 (+2.6 pp, CI includes 0); clear from k=3 (+7.9 pp)", fontsize=10)
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.25)

    k = 3
    x = np.arange(4)
    vals = [int(cm.loc[cm.k == k, c].iloc[0]) for c in ("n_both", "n_bs_only", "n_global_only", "n_neither")]
    b.bar(x, vals, color=[GRN, BLU, ORG, GRY], alpha=0.9)
    for i, v in enumerate(vals):
        b.text(i, v + 4, str(v), ha="center", fontsize=9)
    b.set_xticks(x)
    b.set_xticklabels(["covered by\nboth", "only by\nbasin-specific", "only by\nglobal",
                       "covered by\nneither"], fontsize=8.5)
    b.set_ylabel(f"number of basins (k={k})")
    b.set_ylim(0, max(vals) * 1.18)
    b.set_title(f"(b) per-basin coverage at k={k}\n"
                f"McNemar exact p = {float(cm.loc[cm.k==k,'mcnemar_exact_p'].iloc[0]):.1e}; "
                f"mean regret {float(cm.loc[cm.k==k,'mean_regret_basin_specific'].iloc[0]):.4f} vs "
                f"{float(cm.loc[cm.k==k,'mean_regret_global'].iloc[0]):.4f}", fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_CS_F3 — basin-specific candidate sets vs a simple global Top-k baseline.",
                 fontsize=10.5, y=1.02)
    save(fig, "basin_specific_vs_global_topk")

    # ---------------------------------------------------------------- F4
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    for kk, cl in ((3, RED), (5, BLU)):
        j = O.loc[O.k == kk].iloc[0]
        xx = []
        yy = []
        for i in range(kk + 1):
            n = json.loads(j["intersection_distribution"]).get(str(i), 0)
            xx.append(i); yy.append(n / 531)
        a.bar(np.array(xx) + (0.2 if kk == 5 else -0.2), yy, width=0.4, color=cl, alpha=0.85,
              label=f"k={kk} (Jaccard median {j['jaccard_median']:.2f}, full set {j['full_set_retained_fraction']*100:.1f}%)")
    a.set_xticks(range(6))
    a.set_xlabel("number of models shared between the seen Top-k and the OOB Top-k")
    a.set_ylabel("fraction of basins")
    a.set_title("(a) same-k set overlap distribution\n"
                "the head of the ranking transfers poorly, but overlap grows with k", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    o = O.sort_values("k")
    b.plot(o["k"], o["recall_mean"], "o-", color=BLU, lw=2.0, ms=5,
           label="recall  |seen Top-k $\\cap$ OOB Top-k| / k")
    b.plot(o["k"], o["jaccard_mean"], "s-", color=ORG, lw=2.0, ms=5, label="Jaccard (mean)")
    b.plot(o["k"], o["jaccard_median"], "^--", color=ORG, lw=1.3, ms=4, alpha=0.7,
           label="Jaccard (median)")
    for kk in (3, 5):
        b.annotate(f"k={kk}: recall {float(o.loc[o.k==kk,'recall_mean'].iloc[0]):.3f}",
                   (kk, float(o.loc[o.k == kk, "recall_mean"].iloc[0])),
                   textcoords="offset points", xytext=(5, -12), fontsize=8, color=INK)
    b.set_xlabel("k"); b.set_ylabel("set overlap")
    b.set_ylim(0, 1.0); b.set_xlim(0.5, 10.5); b.set_xticks(KSMALL)
    b.set_title("(b) overlap vs k\n"
                "recall rises 0.252 (k=1) → 0.392 (k=3) → 0.507 (k=5) → 0.632 (k=10)", fontsize=10)
    b.legend(fontsize=8); b.grid(alpha=0.25)
    fig.suptitle("PL_CS_F4 — same-k set overlap between the seen and held-out Top-k model sets "
                 "(complements the candidate-set coverage curves).", fontsize=10.5, y=1.02)
    save(fig, "top3_top5_overlap_distribution")

    print(f"\nfigures -> {PC.F4}")


if __name__ == "__main__":
    main()
