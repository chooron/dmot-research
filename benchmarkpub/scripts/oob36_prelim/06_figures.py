#!/usr/bin/env python3
"""06_figures.py — F1-F3 of ch4-oob36-download-prelim-20260930.

F1  36-model OOB KGE boxplot, sorted by median
F2  per-model seen-dPL median vs OOB median, 1:1 line, model labels
F3  per-model median delta_KGE vs n_parameters, mopex series highlighted,
    title marked "exploratory"

Each figure is written as BOTH .pdf and .png (task requirement).
Only descriptive plots: no recommendation / candidate-set / equivalence content.

Inputs : $OUT/tables/seen_ic_oob_merged.parquet, $OUT/tables/S1_model_oob_summary.csv,
         $OUT/tables/S2_seen_to_oob_change.csv
Outputs: $OUT/figures/F1_oob_kge_box.{pdf,png}, F2_seen_vs_oob.{pdf,png},
         F3_delta_vs_nparam.{pdf,png}
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, OUT_TABLES, OUT_FIGURES, EXPLORATORY_BANNER  # noqa: E402


def save(fig, stem: str) -> None:
    os.makedirs(OUT_FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGURES, f"{stem}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   wrote figures/{stem}.pdf and .png")


def main() -> None:
    tbl = pd.read_parquet(os.path.join(OUT_TABLES, "seen_ic_oob_merged.parquet"))
    S1 = pd.read_csv(os.path.join(OUT_TABLES, "S1_model_oob_summary.csv"))

    # ---------------- F1 ----------------
    order = S1.sort_values("kge_oob_median", ascending=False)["model"].tolist()
    data = [tbl.loc[tbl["model"] == m, "kge_oob"].to_numpy(dtype=float) for m in order]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bp = ax.boxplot(data, vert=True, widths=0.62, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f3")
        patch.set_edgecolor("#3b6ea5")
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("OOB KGE (held-out basins, pooled over 5 folds)")
    ax.set_title(f"F1 — per-model OOB KGE, 36 models x 531 basins (n=531 per model)\n"
                 f"descriptive only", fontsize=10)
    save(fig, "F1_oob_kge_box")

    # ---------------- F2 ----------------
    med = (tbl.groupby("model")[["kge_seen_dpl", "kge_oob"]]
              .median().reset_index())
    med = med.sort_values("model")
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    x = med["kge_seen_dpl"].to_numpy(float)
    y = med["kge_oob"].to_numpy(float)
    ax.scatter(x, y, s=28, color="#3b6ea5", zorder=3)
    lim = [float(min(x.min(), y.min())) - 0.05, float(max(x.max(), y.max())) + 0.05]
    ax.plot(lim, lim, color="0.4", lw=0.9, ls="--", zorder=2, label="1:1")
    for xi, yi, m in zip(x, y, med["model"]):
        ax.annotate(m, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("median seen-dPL KGE (in-sample basins)")
    ax.set_ylabel("median OOB KGE (held-out basins)")
    ax.set_title("F2 — seen-dPL vs OOB, per model (n=531 basins each)\ndescriptive only",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    save(fig, "F2_seen_vs_oob")

    # ---------------- F3 ----------------
    S2 = pd.read_csv(os.path.join(OUT_TABLES, "S2_seen_to_oob_change.csv"), comment="#")
    d = S2[["model", "delta_kge_median", "n_params"]].dropna()
    is_mopex = d["model"].str.startswith("mopex").to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(d["n_params"][~is_mopex], d["delta_kge_median"][~is_mopex], s=30,
               color="#3b6ea5", label="non-MOPEX", zorder=3)
    ax.scatter(d["n_params"][is_mopex], d["delta_kge_median"][is_mopex], s=46,
               color="#d1701a", marker="s", label="MOPEX series", zorder=4)
    for _, r in d.iterrows():
        ax.annotate(r["model"], (r["n_params"], r["delta_kge_median"]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel("n parameters (frozen chapter-4 v3 table)")
    ax.set_ylabel("median $\\Delta$KGE = KGE$_{seen-dPL}$ - KGE$_{OOB}$")
    ax.set_title(f"F3 — seen→OOB change vs model complexity\n{EXPLORATORY_BANNER}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    save(fig, "F3_delta_vs_nparam")


if __name__ == "__main__":
    main()
