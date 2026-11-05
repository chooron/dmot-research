#!/usr/bin/env python3
"""28_adequacy_figures.py — PNG figures for the adequacy analysis.

    seen_oob_adequacy_rates.png         component pass rates, SEEN vs OOB
    signature_failure_seen_vs_oob.png   per-signature failure rates, ranked, with the shift
    adequacy_transfer_seen_vs_oob.png   continuity of errors vs instability of the pass/fail
    candidate_set_after_adequacy_filter.png  how many candidates survive the adequacy filter
    kge_vs_adequacy.png                 KGE against adequacy, and the KGE winner's rank

Writes ONLY .png.  Reads only the tables written by scripts 26-27.
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
VERIF = ["RR", "BFI", "Q5", "Q95", "MHFD", "slopeFDC"]
UNVERIF = ["LFfreq", "HFfreq", "LFdur", "HFdur"]


def save(fig, stem):
    PC.F6.mkdir(parents=True, exist_ok=True)
    fig.savefig(PC.F6 / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"   wrote figures/{stem}.png")


def main() -> None:
    L = pd.read_parquet(PC.T6 / "adequacy_fold_level_long.parquet")
    L["basin_id"] = L["basin_id"].astype(np.int64)
    FR = pd.read_csv(PC.T6 / "adequacy_signature_failure_rates.csv")
    BW = pd.read_csv(PC.T6 / "adequacy_seen_oob_transfer.csv")
    CW = pd.read_csv(PC.T6 / "candidate_set_adequacy_summary.csv")
    PB = pd.read_csv(PC.T6 / "candidate_set_adequacy_by_basin.csv")
    EV = pd.read_csv(PC.T6 / "kge_vs_adequacy.csv")
    T1 = pd.read_csv(PC.T6 / "kge_top1_vs_adequacy.csv")
    TW = pd.read_csv(PC.T6 / "adequacy_threshold_sensitivity.csv")
    A1 = pd.read_csv(PC.T6 / "accurate_set_adequacy_summary.csv")

    seen = L[L.realization == "SEEN_canonical_dPL"]
    oob = L[L.realization == "OOB_heldout"]

    # ---------------------------------------------------------------- F1
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1, 1.05]})
    comps = [("Qmonth", "qm_pass"), ("Qyear", "qy_pass"),
             ("signature (6)", "sig_pass"), ("STRICT", "strict_pass"),
             ("signature floored", "sig_pass_floored6"), ("STRICT floored", "strict_pass_floored")]
    x = np.arange(len(comps))
    sv = [seen[c].mean() for _, c in comps]
    ov = [oob[c].mean() for _, c in comps]
    a.bar(x - 0.2, sv, width=0.4, color=BLU, alpha=0.9, label="SEEN")
    a.bar(x + 0.2, ov, width=0.4, color=ORG, alpha=0.9, label="OOB")
    for i, (s_, o_) in enumerate(zip(sv, ov)):
        a.text(i - 0.2, s_ + .008, f"{s_:.3f}", ha="center", fontsize=7.5)
        a.text(i + 0.2, o_ + .008, f"{o_:.3f}", ha="center", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels([c for c, _ in comps], fontsize=8, rotation=18, ha="right")
    a.set_ylabel("adequacy pass rate (all tests in the component)")
    a.set_ylim(0, 0.5)
    a.set_title("(a) component pass rates, |pct| $\\leq$ 50%\n"
                "strict all-pass is very sparse; OOB < SEEN everywhere", fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    pt = FR.pivot_table(index="signature", columns="condition", values="per_test_failure_rate")
    pt = pt.reindex(VERIF + UNVERIF + ["Qmonth", "Qyear"]).dropna(how="all")
    y = np.arange(len(pt))
    b.barh(y - 0.2, pt["SEEN"], height=0.4, color=BLU, alpha=0.9, label="SEEN (per test)")
    b.barh(y + 0.2, pt["OOB"], height=0.4, color=ORG, alpha=0.9, label="OOB (per test)")
    b.set_yticks(y); b.set_yticklabels(pt.index, fontsize=8); b.invert_yaxis()
    b.axvline(0.5, color=GRY, ls=":", lw=1.0)
    b.set_xlabel("per-test failure rate  (Qmonth = per-month, Qyear = per-year)")
    b.set_title("(b) the same result per individual test\n"
                "the component rates in (a) are inflated by multiplicity (12, 15, 6 tests)",
                fontsize=10)
    b.legend(fontsize=8); b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_AD_F1 — Spieler-style adequacy: SEEN (canonical dPL) vs OOB (held-out fold), "
                 "36 models x 531 basins.", fontsize=10.5, y=1.02)
    save(fig, "seen_oob_adequacy_rates")

    # ---------------------------------------------------------------- F2
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fr = FR[FR.key.isin(["rc", "bfi", "q5", "q95", "mhfd", "fdc_slope",
                         "lf_freq", "hf_freq", "lf_dur", "hf_dur"])].copy()
    lab = {"rc": "RR", "bfi": "BFI", "q5": "Q5", "q95": "Q95", "mhfd": "MHFD",
           "fdc_slope": "slopeFDC", "lf_freq": "LFfreq*", "hf_freq": "HFfreq*",
           "lf_dur": "LFdur*", "hf_dur": "HFdur*"}
    fr["label"] = fr["key"].map(lab)
    s = fr[fr.condition == "SEEN"].set_index("label")["failure_rate_among_evaluable"]
    o = fr[fr.condition == "OOB"].set_index("label")["failure_rate_among_evaluable"]
    order = s.sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))
    cols = [GRN if "*" not in k else GRY for k in order]
    a.barh(y, s.reindex(order), height=0.62, color=cols, alpha=0.85)
    a.scatter(o.reindex(order), y, s=42, color=RED, zorder=5, label="OOB")
    a.set_yticks(y); a.set_yticklabels(order, fontsize=9); a.invert_yaxis()
    a.axvline(0.5, color=GRY, ls=":", lw=1.0)
    a.set_xlabel("failure rate among evaluable cases (SEEN bars, OOB dots)")
    a.set_title("(a) which behaviours are missed\n"
                "green = verifiable in-repo definition, grey = *convention not TOSSH-verified",
                fontsize=10)
    a.legend(fontsize=8); a.grid(axis="x", alpha=0.25)

    d = (o - s).reindex(order)
    b.barh(y, d, color=[RED if v > 0 else BLU for v in d], alpha=0.9)
    b.axvline(0, color=INK, lw=1.0)
    b.set_yticks(y); b.set_yticklabels(order, fontsize=9); b.invert_yaxis()
    b.set_xlabel("OOB failure rate $-$ SEEN failure rate")
    b.set_title("(b) the OOB shift\n"
                "every behaviour degrades; high-flow and Q5 degrade most", fontsize=10)
    b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_AD_F2 — signature failure pattern, SEEN vs OOB.  Descriptive; no causal claim.",
                 fontsize=10.5, y=1.02)
    save(fig, "signature_failure_seen_vs_oob")

    # ---------------------------------------------------------------- F3
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    tr = BW.dropna(subset=["retention_of_seen_pass"]).copy()
    tr["lab"] = tr["component"].str.replace("signature ", "", regex=False)
    keep = ["RR", "BFI", "Q5", "Q95", "slopeFDC", "MHFD", "verified 6)", "Qmonth", "Qyear",
            "STRICT overall"]
    tr = tr[tr["lab"].isin([k.replace("signature ", "") for k in keep]) |
            tr["component"].isin(["signature (verified 6)", "STRICT overall", "Qmonth", "Qyear"])]
    tr = tr.sort_values("retention_of_seen_pass")
    y = np.arange(len(tr))
    a.barh(y - 0.19, tr["retention_of_seen_pass"], height=0.38, color=GRN, alpha=0.9,
           label="retention  P(OOB pass | SEEN pass)")
    a.barh(y + 0.19, tr["agreement"], height=0.38, color=GRY, alpha=0.9, label="agreement")
    a.set_yticks(y); a.set_yticklabels(tr["component"], fontsize=8); a.invert_yaxis()
    a.axvline(1.0, color=INK, ls="--", lw=1.0)
    a.set_xlabel("rate"); a.set_xlim(0, 1.05)
    a.set_title("(a) does the pass/fail verdict transfer?\n"
                "retention is low for the bundled criteria, high for individual signatures",
                fontsize=10)
    a.legend(fontsize=8, loc="lower left"); a.grid(axis="x", alpha=0.25)

    sp = BW.dropna(subset=["spearman_seen_vs_oob_abs_error"]).copy()
    sp = sp.sort_values("spearman_seen_vs_oob_abs_error")
    y2 = np.arange(len(sp))
    cols = [RED if "convention-unverified" in c else BLU for c in sp["component"]]
    b.barh(y2, sp["spearman_seen_vs_oob_abs_error"], color=cols, alpha=0.9)
    b.set_yticks(y2); b.set_yticklabels(sp["component"], fontsize=8); b.invert_yaxis()
    b.set_xlabel("Spearman(SEEN error, OOB error)")
    b.set_xlim(0, 1.0)
    b.set_title("(b) but the ERROR MAGNITUDE transfers strongly (0.65-0.86)\n"
                "binary thresholds are unstable, the underlying error is not", fontsize=10)
    b.grid(axis="x", alpha=0.25)
    fig.suptitle("PL_AD_F3 — adequacy transfer SEEN -> OOB: a threshold-crossing problem, not a "
                 "loss of information.", fontsize=10.5, y=1.02)
    save(fig, "adequacy_transfer_seen_vs_oob")

    # ---------------------------------------------------------------- F4
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.0, 5.0))
    m3 = PB[PB.k == 3]; m5 = PB[PB.k == 5]
    labels = ["0", "1", "2", "3"]
    x = np.arange(4)
    a.bar(x - 0.2, [np.mean(m3.n_adequate_seen == i) for i in range(4)], width=0.4,
          color=BLU, alpha=0.9, label="Top-3, SEEN adequacy")
    a.bar(x + 0.2, [np.mean(m3.n_adequate_oob == i) for i in range(4)], width=0.4,
          color=ORG, alpha=0.9, label="Top-3, OOB adequacy")
    a.set_xticks(x); a.set_xticklabels(labels)
    a.set_xlabel("number of adequate models inside the Seen Top-3")
    a.set_ylabel("fraction of basins")
    a.set_title("(a) Seen Top-3 after the adequacy filter\n"
                f"mean {CW.iloc[0]['mean_adequate_candidates_oob']:.2f} adequate (OOB) of 3",
                fontsize=10)
    a.legend(fontsize=8); a.grid(axis="y", alpha=0.25)

    b.bar(x - 0.2, [np.mean(m5.n_adequate_seen == i) for i in range(4)], width=0.4,
          color=BLU, alpha=0.9, label="Top-5, SEEN adequacy")
    b.bar(x + 0.2, [np.mean(m5.n_adequate_oob == i) for i in range(4)], width=0.4,
          color=ORG, alpha=0.9, label="Top-5, OOB adequacy")
    b.set_xticks(x); b.set_xticklabels(labels)
    b.set_xlabel("number of adequate models inside the Seen Top-5")
    b.set_ylabel("fraction of basins")
    b.set_title("(b) Seen Top-5 after the adequacy filter\n"
                f"mean {CW.iloc[1]['mean_adequate_candidates_oob']:.2f} adequate (OOB) of 5",
                fontsize=10)
    b.legend(fontsize=8); b.grid(axis="y", alpha=0.25)

    ks = ["Top-3", "Top-5"]
    cov_sig = [float(CW.iloc[0]["frac_basins_with_ge1_oob_adequate"]),
               float(CW.iloc[1]["frac_basins_with_ge1_oob_adequate"])]
    cov_str = [float(CW.iloc[0]["frac_basins_with_ge1_oob_strict_adequate"]),
               float(CW.iloc[1]["frac_basins_with_ge1_oob_strict_adequate"])]
    cov_str_seen = [float(CW.iloc[0]["frac_basins_with_ge1_seen_strict_adequate"]),
                    float(CW.iloc[1]["frac_basins_with_ge1_seen_strict_adequate"])]
    xx = np.arange(2)
    c.bar(xx - 0.22, cov_sig, width=0.22, color=BLU, alpha=0.9, label="OOB, signature criterion")
    c.bar(xx, cov_str, width=0.22, color=ORG, alpha=0.9, label="OOB, STRICT criterion")
    c.bar(xx + 0.22, cov_str_seen, width=0.22, color=GRY, alpha=0.9, label="SEEN, STRICT criterion")
    for i, v in enumerate(cov_sig):
        c.text(i - 0.22, v + .012, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(cov_str):
        c.text(i, v + .012, f"{v:.2f}", ha="center", fontsize=8)
    c.set_xticks(xx); c.set_xticklabels(ks)
    c.set_ylabel("fraction of basins with $\\geq$1 adequate candidate")
    c.set_ylim(0, 0.65)
    c.set_title("(c) does the candidate set contain an adequate model?\n"
                "strict adequacy leaves very few basins", fontsize=10)
    c.legend(fontsize=7.5); c.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_AD_F4 — can the adequacy filter refine the accuracy-based candidate sets? "
                 "|pct| $\\leq$ 50%.", fontsize=10.5, y=1.03)
    save(fig, "candidate_set_after_adequacy_filter")

    # ---------------------------------------------------------------- F5
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    for cond, col, cl in (("SEEN", BLU, BLU), ("OOB", ORG, ORG)):
        sub = L[L.realization == ("SEEN_canonical_dPL" if cond == "SEEN" else "OOB_heldout")]
        k = sub["kge"].to_numpy(float); n = sub["sig_n_pass"].to_numpy(float)
        m = np.isfinite(k)
        binx = np.linspace(np.nanmin(k), np.nanmax(k), 19)
        idx = np.digitize(k[m], binx)
        med, lo, hi, ctr = [], [], [], []
        for i in range(1, len(binx)):
            w = n[m][idx == i]
            if w.size >= 60:
                ctr.append((binx[i - 1] + binx[i]) / 2)
                med.append(np.median(w))
                q1, q3 = np.percentile(w, [25, 75])
                lo.append(q1); hi.append(q3)
        a.plot(ctr, med, "o-", color=cl, lw=1.8, ms=4, label=f"{cond} (median)")
        a.fill_between(ctr, lo, hi, color=cl, alpha=0.15)
    a.set_xlabel("KGE"); a.set_ylabel("signatures passed (of 6)")
    a.set_title("(a) KGE vs adequacy (median and IQR per KGE bin)\n"
                f"Spearman {EV.iloc[0]['spearman_KGE_vs_n_signatures_passed']:.2f} (SEEN), "
                f"{EV.iloc[1]['spearman_KGE_vs_n_signatures_passed']:.2f} (OOB) "
                "-> related, not equivalent", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.25)

    x = np.arange(2)
    tied = [float(T1.iloc[0]["frac_KGE_top1_tied_for_most_adequate"]),
            float(T1.iloc[1]["frac_KGE_top1_tied_for_most_adequate"])]
    worse = [float(T1.iloc[0]["frac_KGE_top1_strictly_less_adequate_than_best"]),
             float(T1.iloc[1]["frac_KGE_top1_strictly_less_adequate_than_best"])]
    above = [float(T1.iloc[0]["frac_KGE_top1_adequate_at_or_above_median"]),
             float(T1.iloc[1]["frac_KGE_top1_adequate_at_or_above_median"])]
    b.bar(x - 0.25, tied, width=0.25, color=GRN, alpha=0.9, label="KGE winner tied for most adequate")
    b.bar(x, worse, width=0.25, color=RED, alpha=0.9, label="KGE winner strictly less adequate than best")
    b.bar(x + 0.25, above, width=0.25, color=GRY, alpha=0.9, label="KGE winner at/above median adequacy")
    for i in range(2):
        for off, v in ((-0.25, tied[i]), (0.0, worse[i]), (0.25, above[i])):
            b.text(i + off, v + .015, f"{v:.2f}", ha="center", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(["SEEN", "OOB"])
    b.set_ylim(0, 1.05); b.set_ylabel("fraction of basins")
    b.set_title("(b) is the KGE winner the most adequate candidate?\n"
                "yes in only ~1/3 (SEEN) and ~1/4 (OOB) of basins", fontsize=10)
    b.legend(fontsize=7.5); b.grid(axis="y", alpha=0.25)
    fig.suptitle("PL_AD_F5 — KGE accuracy vs hydrological adequacy.  No causal claim.",
                 fontsize=10.5, y=1.02)
    save(fig, "kge_vs_adequacy")

    print(f"\nfigures -> {PC.F6}")


if __name__ == "__main__":
    main()
