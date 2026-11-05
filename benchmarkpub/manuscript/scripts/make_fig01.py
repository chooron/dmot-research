"""Fig. 1 — The required library size depends on how parameters are obtained.   (7.2 x 4.5 in)

(a) HERO  exact coverage curves for IC, seen-dPL and OOB-dPL (3-yr blocks dashed),
          K* at 95 % (filled) and 100 % (open), Knoben et al. (2025) rings, 95 % resampling interval of K*
(b)       share of basins in which each model is near-optimal, IC -> seen -> OOB (core models in black)
(c–e)     maps of the per-basin equivalence-set size |S_b|

Inputs: recomputed T1, three-year bootstrap, basin bootstrap and optimal-cover tables.
Output: manuscript/figures/fig1_library_size.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import hydrofig as hf

hf.setup()
from pathlib import Path

MANUSCRIPT = Path(__file__).resolve().parents[1]
TABLES = MANUSCRIPT / "tables/figures"
T1 = MANUSCRIPT.parent / "analysis/equiv_cover_t1t2_20261003/tables"
S31 = MANUSCRIPT / "analysis/sec31_transfer_and_library/tables"
S32 = MANUSCRIPT / "analysis/sec32_library_members/tables"
labels = {"IC": "IC", "seen-dPL": "seen", "OOB-dPL": "OOB"}
curve1 = pd.read_csv(T1 / "t1_coverage_curve.csv")
curve3 = pd.read_csv(S31 / "sec31_multiyear_block_bootstrap_coverage_curve.csv")
for frame, scheme in ((curve1, "1yr"), (curve3, "3yr")):
    frame["condition"] = frame.condition.map(labels)
    frame["scheme"] = scheme
    frame.rename(columns={"n_models": "K", "frac_covered_exact": "coverage_frac"}, inplace=True)
cov = pd.concat([curve1, curve3], ignore_index=True)
if len(cov) != 216 or cov.condition.isna().any():
    raise ValueError("Expected all 36 budgets for three conditions and two bootstrap schemes")
minimum = pd.read_csv(T1 / "t1_min_cover.csv")
boot = pd.read_csv(S32 / "sec32_bootstrap_cover_ci_summary.csv")
kst = minimum.merge(boot, on=["condition", "target"], validate="one_to_one")
kst["condition"] = kst.condition.map(labels)
kst["level"] = (100 * kst.target).round().astype(int)
kst = kst.rename(columns={"n_models_exact": "K_star", "median": "boot_median",
                          "ci95_lo": "boot_lo", "ci95_hi": "boot_hi"})
eq = pd.read_csv(T1 / "t1_equivalence_sets.csv", dtype={"basin_id": str})
eq = eq.rename(columns={"basin_id": "gauge_id", "n_equivalent": "set_size"})
eq["condition"] = eq.condition.map(labels)
bas = pd.read_csv(TABLES / "gauge_coordinates_531.csv", dtype={"gauge_id": str})
if len(eq) != 3 * 531 or len(bas) != 531 or bas.gauge_id.duplicated().any():
    raise ValueError("Invalid basin coverage or coordinates")
summary = pd.read_csv(S32 / "sec32_optimal_covers_summary.csv")
if summary.loc[np.isclose(summary.target, .95), "is_truncated"].any():
    raise ValueError("Cannot identify core models from truncated enumeration")
core = pd.read_csv(S32 / "sec32_optimal_covers_model_frequencies.csv")
core = core[np.isclose(core.target, .95)][["condition", "model", "in_core_set"]]
frq = pd.read_csv(T1 / "t1_model_coverage.csv").merge(core, on=["condition", "model"], validate="one_to_one")
frq["condition"] = frq.condition.map(labels)
frq["equiv_pct_basins"] = 100 * frq.fraction
frq = frq.rename(columns={"in_core_set": "is_core"})
ranks = pd.read_csv(TABLES / "appendix_a1/fig_a1_model_summary.csv")[["model", "ic_rank"]]
frq = frq.merge(ranks, on="model", validate="many_to_one").rename(columns={"ic_rank": "global_rank"})
cov.to_csv(TABLES / "fig01_coverage_curves_current.csv", index=False)
kst.to_csv(TABLES / "fig01_kstar_current.csv", index=False)
eq.to_csv(TABLES / "fig01_equiv_set_size_current.csv", index=False)
frq.to_csv(TABLES / "fig01_model_equiv_freq_current.csv", index=False)
KNOBEN = {95: 4, 100: 10}            # Knoben et al. (2025), 555 basins
CONDS = ["IC", "seen", "OOB"]
RED = hf.COND["OOB"]

W, H = hf.FIG_W, 4.50
fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------ (a) hero
ax = hf.axes_in(fig, 0.50, 0.10, 3.95, 2.30)
KMAX = 24
for lvl in (95, 100):
    ax.axhline(lvl, color="#C8C8C8", lw=0.8, ls=(0, (4, 3)), zorder=0)
for c in CONDS:
    col = hf.COND[c]
    d1 = cov[(cov.condition == c) & (cov.scheme == "1yr")]
    d3 = cov[(cov.condition == c) & (cov.scheme == "3yr")]
    ax.plot(d1.K, 100 * d1.coverage_frac, color=col, lw=2.2, zorder=3, solid_capstyle="round")
    ax.plot(d3.K, 100 * d3.coverage_frac, color=col, lw=1.0, ls=(0, (2.5, 1.8)), zorder=2)
    for lvl, filled in ((95, True), (100, False)):
        k = int(kst[(kst.condition == c) & (kst.level == lvl)].K_star.iloc[0])
        ax.plot([k, k], [40, lvl], color=col, lw=0.7, ls=":", zorder=1)
        ax.plot(k, lvl, marker=hf.COND_MARKER[c], ms=7 if filled else 6, mfc=col if filled else "white",
                mec=col, mew=1.5, zorder=5)
        ax.text(k + 0.22, 40.8, str(k), ha="left", va="bottom", fontsize=hf.FS if filled else hf.FS_SMALL,
                fontweight="bold" if filled else "normal", color=col, zorder=6, path_effects=hf.halo())
for lvl, k in KNOBEN.items():
    ax.plot(k, lvl, marker="o", ms=14, mfc="none", mec="#7F7F7F", mew=1.0, zorder=4)
for i, c in enumerate(CONDS):                      # K*(95 %) basin-resampling interval
    r = kst[(kst.condition == c) & (kst.level == 95)].iloc[0]
    y = 48.0 + 3.4 * i
    ax.plot([r.boot_lo, r.boot_hi], [y, y], color=hf.COND[c], lw=4.2, solid_capstyle="round", alpha=0.85)
    ax.plot(r.boot_median, y, "o", ms=3.6, color="white", zorder=6)
ax.text(13.6, 51.4, "Resampling interval of $K^{*}$ (95 %)", fontsize=hf.FS_SMALL, color="#555555",
        va="center", path_effects=hf.halo(3))
ax.set_xlim(0.5, KMAX + 0.5); ax.set_ylim(40, 106)
ax.set_xticks([1, 4, 8, 12, 16, 20, 24]); ax.set_yticks([40, 60, 80, 95, 100])
ax.set_xlabel("Library size $K$ (number of models)")
ax.set_ylabel("Basins covered (%)")
leg = [Line2D([], [], color=hf.COND[c], lw=2.2, marker=hf.COND_MARKER[c], ms=5.5, label=hf.COND_LABEL[c])
       for c in CONDS]
leg += [Line2D([], [], color="#555555", lw=1.0, ls=(0, (2.5, 1.8)), label="3-yr blocks"),
        Line2D([], [], color="none", marker="o", ms=10, mfc="none", mec="#7F7F7F", label="Knoben et al. (2025)")]
ax.legend(handles=leg, loc="center right", bbox_to_anchor=(1.0, 0.55), handlelength=1.9, labelspacing=0.35)
hf.panel_label(ax, "a")

# ------------------------------------------------------------ (b) slope chart
axb = hf.axes_in(fig, 5.02, 0.10, 2.16, 2.30)
xpos = {c: i for i, c in enumerate(CONDS)}
core_sets = {c: set(frq.loc[(frq.condition == c) & frq.is_core, "model"]) for c in CONDS}
CORE_INTERSECTION = set.intersection(*core_sets.values())
core_lab = []
for mod, g in frq.groupby("model"):
    g = g.set_index("condition").loc[CONDS]
    core = mod in CORE_INTERSECTION
    col = hf.SEQ_MAG(0.12 + 0.82 * (36 - float(g.global_rank.iloc[0])) / 35)
    axb.plot([xpos[c] for c in CONDS], g.equiv_pct_basins, color=col, lw=2.0 if core else 0.8,
             alpha=1 if core else 0.7, zorder=4 if core else 2, marker="o", ms=3.6 if core else 1.8)
    if core:
        core_lab.append((mod, float(g.equiv_pct_basins.iloc[-1])))
ymax = frq.equiv_pct_basins.max()
if core_lab:
    for mod, yy in core_lab:
        rank = float(frq.loc[frq.model == mod, "global_rank"].iloc[0])
        axb.annotate(hf.model_label(mod), xy=(2, yy), xytext=(0, 5), textcoords="offset points",
                     va="bottom", ha="center", fontsize=hf.FS_SMALL,
                     color=hf.SEQ_MAG(0.12 + 0.82 * (36 - rank) / 35),
                     fontweight="bold", path_effects=hf.halo(), zorder=6)
axb.set_xticks(range(3)); axb.set_xticklabels(["IC", "seen", "OOB"])
for t in axb.get_xticklabels():
    t.set_color(hf.INK); t.set_fontweight("bold")
axb.set_xlim(-0.28, 2.40)
axb.set_ylim(0, ymax * 1.22)
axb.set_yticks([0, 20, 40, 60, 80])
axb.set_ylabel("Near-optimal basins (%)")
axb.spines["bottom"].set_bounds(0, 2)
hf.panel_label(axb, "b")

# ------------------------------------------------------------ (c–e) maps of |S_b|
bounds = [0.5, 1.5, 2.5, 5.5, 9.5, 36.5]
cmap = mcolors.ListedColormap(hf.SEQ_RHO(np.linspace(0.05, 0.78, len(bounds) - 1)))
norm = mcolors.BoundaryNorm(bounds, cmap.N)
x, y = hf.albers(bas.lon, bas.lat)
pos = dict(zip(bas.gauge_id, zip(x, y)))
PAD = 300
mw = 2.08; mh = hf.map_h(mw, PAD); top = 2.94
for j, c in enumerate(CONDS):
    axm = hf.axes_in(fig, 0.04 + j * (mw + 0.08), top, mw, mh)
    hf.basemap(axm, pad_top=PAD)
    d = eq[eq.condition == c].copy()
    d["x"] = [pos[g][0] for g in d.gauge_id]; d["y"] = [pos[g][1] for g in d.gauge_id]
    d = d.sort_values("set_size", ascending=True)        # large sets first, singletons on top
    sc = axm.scatter(d.x, d.y, c=d.set_size, cmap=cmap, norm=norm, s=9, lw=0.25, ec="#6E6E6E", zorder=3)
    hf.panel_label(axm, "cde"[j], extra=hf.COND_LABEL[c], extra_color=hf.INK)
hf.map_cbar(fig, axm, sc, "$|S_b|$", loc=(1.04, 0.10, 0.045, 0.76), orientation="vertical",
            ticks=[1, 2, 4, 7.5, 23], ticklabels=["1", "2", "3–5", "6–9", "≥10"])
hf.save(fig, "fig1_library_size", outdir=str(MANUSCRIPT / "figures"), png_only=True)
