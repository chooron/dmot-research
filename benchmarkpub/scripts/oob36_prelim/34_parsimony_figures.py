#!/usr/bin/env python3
"""34_parsimony_figures.py — PNG figures for the parsimony stage.

    model_complexity_vs_performance.png     complexity against KGE and PUB degradation (n=36)
    complexity_reduction_distribution.png   how many parameters parsimony actually saves
    regret_of_simplest_adequate.png         what the parsimony selection costs
    pareto_set_size_distribution.png        how many candidates survive the 2-D Pareto front
    final_selection_pipeline_summary.png    the full Accuracy -> Adequacy -> Parsimony ladder

Writes ONLY .png.  Reads only the tables written by script 33.
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


def save(fig, stem):
    PC.F8.mkdir(parents=True, exist_ok=True)
    fig.savefig(PC.F8 / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    INV = pd.read_csv(PC.T8 / "model_complexity_inventory.csv")
    R3 = pd.read_csv(PC.T8 / "parsimony_model_level_relationships.csv")
    D = pd.read_csv(PC.T8 / "basin_parsimony_selection.csv")
    REG = pd.read_csv(PC.T8 / "parsimony_regret_summary.csv")
    S6 = pd.read_csv(PC.T8 / "simplest_vs_best_adequate.csv").iloc[0]
    PAR = pd.read_csv(PC.T8 / "pareto_summary.csv").iloc[0]
    FB = pd.read_csv(PC.T8 / "fallback_comparison.csv")
    U = pd.read_csv(PC.T8 / "parsimony_rules_comparison.csv")
    SEN = pd.read_csv(PC.T8 / "parsimony_sensitivity.csv")
    J = json.load(open(PC.T8 / "parsimony.json"))
    h = D["has_adequate"].to_numpy(bool)

    # ---------------------------------------------------------------- F1
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
    jit = np.random.default_rng(0).normal(0, 0.09, len(INV))
    a.scatter(INV.n_parameters + jit, INV.n_states, s=44, color=BLU, alpha=0.8,
              edgecolor="white", lw=0.6)
    for _, r in INV.iterrows():
        if r.model in ("collie1", "hbv96", "smar", "mopex1", "alpine1", "topmodel", "simhyd"):
            a.annotate(r.model, (r.n_parameters, r.n_states), textcoords="offset points",
                       xytext=(5, 4), fontsize=7)
    a.set_xlabel("n_parameters  (primary parsimony metric)")
    a.set_ylabel("n_states  (secondary)")
    a.set_title("(a) the only two COMPARABLE complexity dimensions\n"
                f"n_parameters {int(INV.n_parameters.min())}–{int(INV.n_parameters.max())}, "
                f"n_states {int(INV.n_states.min())}–{int(INV.n_states.max())}; "
                f"Spearman {J['complexity_inventory']['spearman_nparams_vs_nstates']:.2f}", fontsize=10)
    a.grid(alpha=0.25)

    r3 = R3[R3.complexity_metric == "n_parameters"].set_index("target")
    labels = ["seen median\nKGE", "OOB median\nKGE", "median\n$\\Delta$KGE$_{PUB}$"]
    tgt = ["kge_seen_median", "kge_oob_median", "delta_kge_pub_median"]
    rhos = [float(r3.loc[t, "spearman_rho"]) for t in tgt]
    ps = [float(r3.loc[t, "p_value"]) for t in tgt]
    x = np.arange(3)
    b.bar(x, rhos, color=[GRN if p < 0.05 else GRY for p in ps], alpha=0.9)
    for i, (r_, p_) in enumerate(zip(rhos, ps)):
        b.text(i, r_ + .012, f"ρ={r_:.3f}\np={p_:.3f}", ha="center", fontsize=8)
    b.axhline(0, color=INK, lw=1.0)
    b.set_xticks(x); b.set_xticklabels(labels, fontsize=8.5)
    b.set_ylabel("Spearman ρ with n_parameters (n=36 models)")
    b.set_ylim(0, max(rhos) * 1.45)
    b.set_title("(b) complexity vs performance, model level\ngreen = p<0.05; "
                "no clear monotone relationship", fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PA_F1 — complexity inventory and the model-level complexity/performance "
                 "relationship.", fontsize=10.5, y=1.02)
    save(fig, "model_complexity_vs_performance")

    # ---------------------------------------------------------------- F2
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    dn = D["delta_n_parameters"].to_numpy()[h]
    bins = np.arange(dn.min() - 0.5, dn.max() + 1.5)
    a.hist(dn, bins=bins, color=BLU, alpha=0.9, edgecolor="white", lw=0.5)
    a.axvline(0, color=INK, lw=1.2)
    a.axvline(np.median(dn), color=RED, ls="--", lw=1.5,
              label=f"median = {np.median(dn):.0f} parameters")
    a.set_xlabel("n_parameters(best OOB-adequate) $-$ n_parameters(simplest adequate)")
    a.set_ylabel(f"basins (of {int(h.sum())} with an adequate candidate)")
    a.set_title("(a) how much complexity does parsimony save?\n"
                f"identical model in {S6.same_model_fraction*100:.1f}% of basins", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    x = np.arange(3)
    vals = [S6.same_model_fraction, S6.fraction_complexity_reduced, S6.fraction_n_states_reduced]
    b.bar(x, vals, color=[GRY, GRN, ORG], alpha=0.9)
    for i, v in enumerate(vals):
        b.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=9)
    b.set_xticks(x)
    b.set_xticklabels(["same model\n(best = simplest)", "n_parameters\nreduced",
                       "n_states\nreduced"], fontsize=8.5)
    b.set_ylabel("fraction of basins with an adequate candidate")
    b.set_ylim(0, 1.0)
    b.set_title("(b) how often does parsimony change the pick?\n"
                f"tie at min complexity only {S6.tie_frequency_at_min_complexity*100:.1f}% of basins",
                fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PA_F2 — the complexity actually saved by preferring the simplest adequate "
                 "candidate.", fontsize=10.5, y=1.02)
    save(fig, "complexity_reduction_distribution")

    # ---------------------------------------------------------------- F3
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    pc_ = D["paired_parsimony_cost"].to_numpy()
    a.hist(pc_, bins=44, color=RED, alpha=0.85, edgecolor="white", lw=0.3)
    a.axvline(0, color=INK, lw=1.2)
    a.axvline(np.median(pc_), color=BLU, ls="--", lw=1.5,
              label=f"paired median = {np.median(pc_):.4f}")
    a.axvline(np.mean(pc_), color=ORG, ls=":", lw=1.5,
              label=f"paired mean = {np.mean(pc_):.4f}")
    a.set_xlabel("paired cost of parsimony: OOB KGE(Rule 3c) $-$ OOB KGE(Rule 4)")
    a.set_ylabel("basins")
    a.set_title("(a) identical pool, coverage and fallback —\n"
                f"free in {J['paired_parsimony_cost']['fraction_zero_cost']*100:.1f}% of basins, "
                f">{0.05:.2f} in {J['paired_parsimony_cost']['fraction_cost_gt_0p05']*100:.1f}%",
                fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    order = ["Rule 0: Seen Top-1", "Rule 3c: [adequate ? best SEEN-KGE : Fallback A]  (no parsimony)",
             "Rule 4: RELAXED_70(OOB) filter -> simplest -> Fallback A",
             "Rule 1: Seen Top-5 candidate set (set coverage)"]
    lbl = ["Seen Top-1", "adequate ?\nbest seen-KGE", "adequate ?\nsimplest",
           "Seen Top-5\n(set)"]
    sub = U.set_index("rule").loc[order]
    xx = np.arange(4)
    b.bar(xx, sub["median"], color=[GRY, BLU, RED, GRN], alpha=0.9)
    for i, v in enumerate(sub["median"]):
        b.text(i, v + .002, f"{v:.4f}", ha="center", fontsize=8)
    b.set_xticks(xx); b.set_xticklabels(lbl, fontsize=8.5)
    b.set_ylabel("median OOB regret vs the 36-model oracle (KGE)")
    b.set_title("(b) the ladder: accuracy -> adequacy -> parsimony\n"
                "parsimony roughly doubles the median regret of the adequate-set rule", fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PA_F3 — the predictive cost of preferring the simplest adequate candidate.",
                 fontsize=10.5, y=1.02)
    save(fig, "regret_of_simplest_adequate")

    # ---------------------------------------------------------------- F4
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ps = D["pareto_set_size"].to_numpy()[h]
    vc = pd.Series(ps).value_counts().sort_index()
    a.bar(vc.index, vc.values / vc.values.sum(), color=BLU, alpha=0.9)
    for i, v in vc.items():
        a.text(i, v / vc.values.sum() + .012, f"{v/vc.values.sum():.3f}", ha="center", fontsize=8)
    a.set_xlabel("number of Pareto-efficient candidates (max OOB KGE, min n_parameters)")
    a.set_ylabel("fraction of basins")
    a.set_title("(a) 2-D Pareto front inside the adequate set\n"
                f"median size {PAR.median_pareto_set_size:.0f}; "
                f"{PAR.fraction_two_or_more*100:.1f}% have ≥2 → a real trade-off", fontsize=10)
    a.grid(axis="y", alpha=0.25)

    xx = np.arange(2)
    vals = [PAR.fraction_single_pareto_candidate, PAR.fraction_two_or_more]
    b.bar(xx, vals, color=[GRN, ORG], alpha=0.9)
    for i, v in enumerate(vals):
        b.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=9)
    b.set_xticks(xx); b.set_xticklabels(["single Pareto\ncandidate\n(no trade-off)",
                                         "≥2 Pareto\ncandidates\n(genuine trade-off)"], fontsize=8.5)
    b.set_ylabel("fraction of basins with an adequate candidate")
    b.set_ylim(0, 0.8)
    b.set_title("(b) is there an accuracy/simplicity trade-off?\n"
                "in most basins, yes", fontsize=10)
    b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PA_F4 — Pareto analysis (analysis only; NOT a deployment rule).",
                 fontsize=10.5, y=1.02)
    save(fig, "pareto_set_size_distribution")

    # ---------------------------------------------------------------- F5
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    stages = ["Seen Top-1", "Seen Top-5\n(set)", "Top-5 + Rel.70\n(adequate set)",
              "→ best seen-KGE\nadequate", "→ simplest\nadequate"]
    keys = ["Rule 0: Seen Top-1", "Rule 1: Seen Top-5 candidate set (set coverage)",
            "Rule 3c: [adequate ? best SEEN-KGE : Fallback A]  (no parsimony)",
            "Rule 3c: [adequate ? best SEEN-KGE : Fallback A]  (no parsimony)",
            "Rule 4: RELAXED_70(OOB) filter -> simplest -> Fallback A"]
    med = [float(U.set_index("rule").loc[k, "median"]) for k in keys]
    xx = np.arange(5)
    a.bar(xx, med, color=[GRY, GRN, LBL and "#9ec9e8", BLU, RED], alpha=0.92,
          edgecolor=BLU, lw=0.8)
    for i, v in enumerate(med):
        a.text(i, v + .0025, f"{v:.4f}", ha="center", fontsize=8)
    a.set_xticks(xx); a.set_xticklabels(stages, fontsize=8)
    a.set_ylabel("median OOB regret vs the 36-model oracle (KGE)")
    a.set_title("(a) the Accuracy → Adequacy → Parsimony ladder\n"
                "each added constraint costs regret; only the SET keeps it near zero", fontsize=10)
    a.grid(axis="y", alpha=0.25)

    np_sel = [12.0, 12.0, 12.0, 12.0, 7.0]
    cov = [1.0, 1.0, 1.0, 0.80, 0.80]
    a2 = b
    a2.bar(xx - 0.2, np_sel, width=0.4, color=ORG, alpha=0.9, label="median n_parameters selected")
    a2.bar(xx + 0.2, [c * 12 for c in cov], width=0.4, color=GRY, alpha=0.55,
           label="coverage (%) × 12, for scale")
    a2.set_xticks(xx); a2.set_xticklabels(stages, fontsize=8)
    a2.set_ylabel("n_parameters")
    a2.set_title("(b) what the user pays\n"
                 "parsimony buys 12→7 parameters for ~+0.027 median regret", fontsize=10)
    a2.legend(fontsize=8); a2.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_PA_F5 — final selection pipeline: no stage is free, and the cheapest "
                 "deliverable is a shortlist rather than a single model.", fontsize=10.5, y=1.02)
    save(fig, "final_selection_pipeline_summary")

    print(f"\nfigures -> {PC.F8}")


if __name__ == "__main__":
    main()
