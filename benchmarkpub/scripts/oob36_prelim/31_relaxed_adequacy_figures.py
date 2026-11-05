#!/usr/bin/env python3
"""31_relaxed_adequacy_figures.py — PNG figures for the relaxed adequacy analysis.

    relaxed_definition_pass_rates.png        pass rate SEEN vs OOB and models adequate per basin
    candidate_count_before_after_adequacy.png  how much each definition actually shrinks Top-3/5
    zero_candidate_rate.png                  the sparse/filter trade-off
    regret_after_adequacy_filter.png         what the filtering costs in predictive regret

Writes ONLY .png.  Reads only the tables written by script 30.
"""

from __future__ import annotations

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
PRIMARY = ["STRICT_REFERENCE", "RELAXED_60", "RELAXED_70", "RELAXED_80", "GROUP_3OF5", "GROUP_4OF5"]
SHORT = {"STRICT_REFERENCE": "STRICT\n(reference)", "RELAXED_60": "RELAXED_60\n(≥60%)",
         "RELAXED_70": "RELAXED_70\n(≥70%)", "RELAXED_80": "RELAXED_80\n(≥80%)",
         "GROUP_3OF5": "GROUP\n3of5", "GROUP_4OF5": "GROUP\n4of5"}


def save(fig, stem):
    PC.F7.mkdir(parents=True, exist_ok=True)
    fig.savefig(PC.F7 / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    DS = pd.read_csv(PC.T7 / "relaxed_definition_summary.csv").set_index("definition")
    C = pd.read_csv(PC.T7 / "candidate_set_relaxed_adequacy.csv")
    TS = pd.read_csv(PC.T7 / "relaxed_threshold_sensitivity.csv")
    DS = DS.loc[PRIMARY + ["RELAXED_70_verified6", "GROUP_3OF5_verified6", "SIG6_ALLPASS"]]

    # ---------------------------------------------------------------- F1
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    show = PRIMARY
    x = np.arange(len(show))
    sv = [DS.loc[d, "seen_pass_rate"] for d in show]
    ov = [DS.loc[d, "oob_pass_rate"] for d in show]
    a.bar(x - 0.2, sv, width=0.4, color=BLU, alpha=0.9, label="SEEN")
    a.bar(x + 0.2, ov, width=0.4, color=ORG, alpha=0.9, label="OOB")
    for i, (s_, o_) in enumerate(zip(sv, ov)):
        a.text(i - 0.2, s_ + .012, f"{s_:.2f}", ha="center", fontsize=7.5)
        a.text(i + 0.2, o_ + .012, f"{o_:.2f}", ha="center", fontsize=7.5)
        a.text(i, max(s_, o_) + .055, f"−{(s_-o_)*100:.1f} pp", ha="center", fontsize=7,
               color=RED if (s_ - o_) > 0.05 else GRY)
    a.set_xticks(x); a.set_xticklabels([SHORT[d] for d in show], fontsize=8)
    a.set_ylabel("adequacy pass rate (model $\\times$ basin cells)")
    a.set_ylim(0, 1.0)
    a.set_title("(a) pass rate, primary threshold $\\pm$50%\n"
                "the strict reference is 25-40x sparser than any relaxed rule", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    y = np.arange(len(show))
    med = [DS.loc[d, "oob_median_adequate_models_per_basin"] for d in show]
    mean = [DS.loc[d, "oob_mean_adequate_models_per_basin"] for d in show]
    b.barh(y - 0.19, mean, height=0.38, color=ORG, alpha=0.9, label="mean (OOB)")
    b.barh(y + 0.19, med, height=0.38, color=GRN, alpha=0.9, label="median (OOB)")
    b.axvline(36, color=INK, ls="--", lw=1.0)
    b.text(36, -0.6, " all 36", fontsize=7.5)
    for i, (m_, me_) in enumerate(zip(mean, med)):
        b.text(m_ + 0.6, i - 0.19, f"{m_:.1f}", va="center", fontsize=7.5)
        b.text(me_ + 0.6, i + 0.19, f"{me_:.0f}", va="center", fontsize=7.5)
    b.set_yticks(y); b.set_yticklabels([SHORT[d].replace("\n", " ") for d in show], fontsize=8)
    b.invert_yaxis(); b.set_xlim(0, 40)
    b.set_xlabel("adequate models per basin (of 36), OOB")
    b.set_title("(b) how selective is each rule across the 36-model pool?\n"
                "GROUP_3OF5 keeps ~29 of 36 models — barely a filter", fontsize=10)
    b.legend(fontsize=8, loc="lower right"); b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_RA_F1 — relaxed adequacy definitions vs the strict Spieler reference. "
                 "All definitions are pre-registered; none was tuned.", fontsize=10.5, y=1.02)
    save(fig, "relaxed_definition_pass_rates")

    # ---------------------------------------------------------------- F2
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for ax, k in ((a, 3), (b, 5)):
        sub = C[C.k == k].set_index("definition").loc[show]
        before = sub["mean_candidates_before"].to_numpy(float)
        after = sub["mean_candidates_after"].to_numpy(float)
        xx = np.arange(len(show))
        ax.bar(xx, before, width=0.62, color=LBL, edgecolor=BLU, alpha=0.9,
               label=f"before filtering (Top-{k})")
        ax.bar(xx, after, width=0.62, color=ORG, alpha=0.95, label="after adequacy filter (OOB)")
        for i, (bf, af) in enumerate(zip(before, after)):
            ax.text(i, af + 0.06, f"{af:.2f}", ha="center", fontsize=8)
            ax.text(i + 0.36, bf - 0.05, f"−{100*(1-af/bf):.0f}%", ha="center", fontsize=7,
                    color=RED)
        ax.set_xticks(xx); ax.set_xticklabels([SHORT[d] for d in show], fontsize=8)
        ax.set_ylabel("mean candidates per basin")
        ax.set_ylim(0, k * 1.22)
        ax.set_title(f"({'ab'[k==5]}) Seen Top-{k}: candidates before and after the filter",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_RA_F2 — does each relaxed definition actually shrink the accuracy-based "
                 "candidate set?  GROUP_3OF5 removes only ~11%.", fontsize=10.5, y=1.02)
    save(fig, "candidate_count_before_after_adequacy")

    # ---------------------------------------------------------------- F3
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    y = np.arange(len(show))
    a.barh(y, [DS.loc[d, "oob_frac_basins_zero_adequate"] for d in show], color=RED, alpha=0.9,
           label="no adequate model in the whole 36-model pool")
    a.barh(y, [-DS.loc[d, "oob_frac_basins_zero_adequate"] for d in show], color="none")
    a.set_yticks(y); a.set_yticklabels([SHORT[d].replace("\n", " ") for d in show], fontsize=8)
    a.invert_yaxis(); a.set_xlim(0, 0.85)
    a.axvline(0.10, color=GRY, ls=":", lw=1.2)
    a.text(0.105, -0.55, "10%", fontsize=7.5, color=GRY)
    a.set_xlabel("fraction of basins with zero adequate models (OOB)")
    a.set_title("(a) sparsity of the definition itself\n"
                "STRICT leaves 75% of basins with nothing", fontsize=10)
    a.legend(fontsize=8, loc="lower right"); a.grid(axis="x", alpha=0.25)

    for k, cl, mk in ((3, BLU, "o"), (5, ORG, "s")):
        sub = C[C.k == k].set_index("definition").loc[show]
        b.plot(sub["frac_basins_contains_ge1_oob_adequate"], y, mk + "-", color=cl, lw=1.8,
               ms=6, label=f"Top-{k}: ≥1 OOB-adequate candidate")
    b.set_yticks(y); b.set_yticklabels([]); b.invert_yaxis()
    b.set_xlim(0, 1.0); b.set_xlabel("fraction of basins")
    b.set_title("(b) does the filtered candidate set survive?\n"
                "RELAXED_70 keeps a candidate in 74% (Top-3) / 80% (Top-5) of basins",
                fontsize=10)
    b.legend(fontsize=8, loc="lower left"); b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_RA_F3 — the sparsity / selectivity trade-off.  A definition is only useful "
                 "if it is neither empty nor toothless.", fontsize=10.5, y=1.02)
    save(fig, "zero_candidate_rate")

    # ---------------------------------------------------------------- F4
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    sub = C[C.k == 3].set_index("definition").loc[show]
    xx = np.arange(len(show))
    a.bar(xx - 0.2, sub["median_predictive_regret_after_filter"], width=0.4, color=BLU,
          alpha=0.9, label="after adequacy filter (OOB)")
    a.bar(xx + 0.2, sub["median_predictive_regret_unfiltered_topk"], width=0.4, color=GRY,
          alpha=0.9, label="unfiltered Top-3")
    a.set_xticks(xx); a.set_xticklabels([SHORT[d] for d in show], fontsize=8)
    a.set_ylabel("median predictive regret vs the OOB oracle (KGE)")
    a.set_title("(a) median regret, Top-3 base set\n"
                "filtering costs at most ~0.03 KGE in the median", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    for t, cl in ((0.01, GRN), (0.05, ORG)):
        a2 = b
        a2.plot(sub[f"frac_regret_le_{t:g}_after".replace("0.01", "0p01").replace("0.05", "0p05")],
                xx, "o-", color=cl, lw=1.8, ms=6, label=f"filtered: regret ≤ {t}")
        a2.plot(sub[f"frac_regret_le_{t:g}_unfiltered_topk".replace("0.01", "0p01").replace("0.05", "0p05")],
                xx, "s--", color=cl, lw=1.2, ms=5, alpha=0.6, label=f"unfiltered: regret ≤ {t}")
    b.set_xticks(xx); b.set_xticklabels([SHORT[d] for d in show], fontsize=8)
    b.set_ylabel("fraction of basins")
    b.set_ylim(0, 1.0)
    b.set_title("(b) retained predictive quality, Top-3\n"
                "solid = after filter (conditional on a non-empty set), dashed = unfiltered",
                fontsize=10)
    b.legend(fontsize=7.5, loc="lower left"); b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_RA_F4 — what the adequacy filter costs in predictive regret (Top-3 base set).",
                 fontsize=10.5, y=1.02)
    save(fig, "regret_after_adequacy_filter")

    # ---------------------------------------------------------------- F5 (threshold sensitivity)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for d, cl in (("RELAXED_70", BLU), ("GROUP_3OF5", ORG)):
        s = TS[TS.definition == d].sort_values("threshold_pct")
        ax.plot(s["threshold_pct"], s["top3_mean_kept"], "o-", color=cl, lw=2.0, ms=6,
                label=f"{d}: mean candidates kept (Top-3)")
        ax.plot(s["threshold_pct"], s["top3_contains_ge1"], "s--", color=cl, lw=1.3, ms=5,
                alpha=0.65, label=f"{d}: ≥1 candidate retained")
    ax.axhline(3.0, color=INK, ls=":", lw=1.0)
    ax.text(25.5, 3.02, "no filtering (3 of 3)", fontsize=8, color=INK)
    ax.set_xlabel("signature percentage-error threshold (%)")
    ax.set_ylabel("Top-3 base set")
    ax.set_xticks([25, 50, 75]); ax.set_ylim(0, 3.35)
    ax.set_title("PL_RA_F5 — threshold sensitivity of the two candidate definitions\n"
                 "the ordering and the qualitative picture hold at every threshold",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.25)
    save(fig, "threshold_sensitivity_relaxed")

    print(f"\nfigures -> {PC.F7}")


if __name__ == "__main__":
    main()
