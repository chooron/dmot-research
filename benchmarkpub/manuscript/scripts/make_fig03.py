"""Fig. 3 — A few core models are always needed; the interchangeable remainder grows with transfer.   (7.2 x 4.5 in)

(a) HERO  model x condition bubble matrix: bubble area = share of the 95 % optimal libraries that contain the model;
          shaded columns = models present in every optimal library of all three conditions;
          strip above = number of parameters (bar) and snow routine (*)
(b)       map (OOB-dPL): basins equivalently covered by the core models alone vs basins needing another model;
          stars = the single basin that forces the last model into each 100 % library
(c)       hydrographs of the three bottleneck basins with the KGE of both models and the equivalence threshold

Inputs: ./data/*.csv (DATA_SPEC.md).  Output: ./output/fig03.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import hydrofig as hf

hf.setup()
mods = pd.read_csv("data/models.csv").set_index("model")
sol = pd.read_csv("data/optimal_solutions.csv")
cov = hf.normalize_gauge_ids(pd.read_csv("data/basin_cover_status_oob.csv", dtype={"gauge_id": str, "bottleneck": str}))
hyd = hf.normalize_gauge_ids(pd.read_csv("data/bottleneck_hydrographs.csv", parse_dates=["date"], dtype={"gauge_id": str}))
bk = hf.normalize_gauge_ids(pd.read_csv("data/bottleneck_kge.csv", dtype={"gauge_id": str})).set_index("condition")
CONDS = ["IC", "seen", "OOB"]

W, H = hf.FIG_W, 4.55
fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------ (a) bubble matrix
s95 = sol[sol.level == 95]
share, nsol = {}, {}
for c in CONDS:
    g = s95[s95.condition == c]
    nsol[c] = g.solution_id.nunique()
    share[c] = g.groupby("model").solution_id.nunique() / nsol[c]
S = pd.DataFrame(share).fillna(0.0)
core_all = S.index[(S[CONDS] >= 1.0).all(axis=1)].tolist()
order = S.assign(core=S.index.isin(core_all), r=mods.loc[S.index, "global_rank"].values) \
         .sort_values(["core", "OOB", "seen", "IC", "r"], ascending=[False, False, False, False, True]).index.tolist()
M = len(order)
ax = hf.axes_in(fig, 0.05, 0.20, 7.10, 1.15)
ROWY = {"IC": 2, "seen": 1, "OOB": 0}
# Faint grid: one vertical rule per model column, one horizontal rule per condition row.
for j in range(M):
    ax.axvline(j, color="#F0F0F0", lw=0.6, zorder=0)
for c in CONDS:
    ax.plot([-.5, M - .5], [ROWY[c], ROWY[c]], color="#E8E8E8", lw=0.7, zorder=0)
for j, m in enumerate(order):
    if m in core_all:
        ax.axvspan(j - 0.45, j + 0.45, ymin=0, ymax=(2.5 + 1.1) / (3.9 + 1.1), color="#ECECEC", lw=0, zorder=0)
    for c in CONDS:
        f = S.loc[m, c]
        if f > 0:
            ax.scatter(j, ROWY[c], s=8 + 150 * f, color=hf.COND[c], alpha=1.0 if f >= 1 else 0.72,
                       ec=hf.INK if f >= 1 else "white", lw=0.8 if f >= 1 else 0.4, zorder=3)
    hb = 0.75 * mods.loc[m, "n_params"] / 15
    ax.add_patch(plt.Rectangle((j - 0.28, 2.75), 0.56, hb, color="#9E9E9E", lw=0, zorder=2))
    if mods.loc[m, "has_snow"]:
        ax.scatter(j, 3.68, marker=(6, 2, 0), s=22, lw=0.9, color=hf.INK, zorder=3)
for c in CONDS:
    ax.text(-0.75, ROWY[c], f"{hf.COND_LABEL[c]} ({nsol[c]})", ha="right", va="center", color=hf.COND[c],
            fontsize=hf.FS_SMALL, fontweight="bold")
pmax = float(mods.loc[order, "n_params"].max())
px = -5.2
ax.plot([px, px], [2.75, 2.75 + .75 * pmax / 15], color=hf.MUTED, lw=.6)
for pv in (0, pmax):
    py = 2.75 + .75 * pv / 15
    ax.plot([px - .15, px], [py, py], color=hf.MUTED, lw=.6)
    ax.text(px - .25, py, f"{pv:g}", ha="right", va="center", fontsize=hf.FS_SMALL)
ax.text(-0.75, 3.05, "Parameters", ha="right", va="center", fontsize=hf.FS_SMALL, color=hf.MUTED)
ax.text(-0.75, 3.62, "Snow routine", ha="right", va="center", fontsize=hf.FS_SMALL, color=hf.MUTED)
XL = -6.9
for i, f in enumerate([0.25, 1.0]):
    cx = -6.30 + 3.05 * i
    ax.scatter(cx, -0.78, s=8 + 150 * f, color="#9E9E9E", lw=0)
    ax.text(cx + 0.72, -0.78, f"{hf.fmt_half_up(f * 100, 0)} %", ha="left", va="center",
            fontsize=hf.FS_SMALL, color=hf.MUTED)
ax.set_xlim(XL, M - 0.4); ax.set_ylim(-1.1, 3.9)
# Tick rule under the parameter strip and a ticked baseline for the model columns.
ax.plot([-0.5, M - 0.5], [2.62, 2.62], color="#B5B5B5", lw=0.7, zorder=1)
ax.plot([-0.5, M - 0.5], [-0.55, -0.55], color="#B5B5B5", lw=0.7, zorder=1)
ax.set_xticks(range(M)); ax.set_xticklabels([hf.model_label(m) for m in order], rotation=90, fontsize=hf.FS_SMALL)
for t, m in zip(ax.get_xticklabels(), order):
    if m in core_all or m in {"hbv96", "mopex5"}:
        t.set_fontweight("bold")
ax.tick_params(axis="x", length=3.0, width=0.7, color="#B5B5B5", pad=2)
ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
hf.panel_label(ax, "a", x=0.0, y=1.12)

# ------------------------------------------------------------ (b) map
PAD = 250
mw = 3.15; mh = hf.map_h(mw, PAD); TOP2 = 2.50
axb = hf.axes_in(fig, 0.03, TOP2 - 0.20, mw, mh)
hf.basemap(axb, pad_top=PAD)
cv = cov[cov.bottleneck.isna() | (cov.bottleneck == "")]
x, y = hf.albers(cv.lon, cv.lat)
k = cv.covered_by_core.values.astype(bool)
axb.scatter(x[k], y[k], s=8, color="#B5B5B5", lw=0.25, ec="white", zorder=3, label="Core models suffice")
axb.scatter(x[~k], y[~k], s=10, color=hf.COND["OOB"], lw=0.25, ec="white", zorder=4, label="Needs another model")
bn = cov[cov.bottleneck.isin(CONDS)].set_index("bottleneck").loc[CONDS]
bx, by = hf.albers(bn.lon, bn.lat)
for i, (c, xx, yy) in enumerate(zip(CONDS, bx, by)):
    axb.scatter(xx, yy, marker="*", s=150, color=hf.COND[c], ec=hf.INK, lw=0.6, zorder=6)
    dx, dy = ((90, 100), (100, 140), (100, -180))[i]
    axb.text(xx + dx, yy + dy, str(i + 1), fontsize=hf.FS, fontweight="bold", color=hf.INK, zorder=7,
             path_effects=hf.halo())
lg = axb.legend(loc="upper left", bbox_to_anchor=(0.10, 1.02), ncol=2, handletextpad=0.2,
                columnspacing=0.8, borderaxespad=0, markerscale=1.4, frameon=False, fontsize=hf.FS_SMALL)
hf.panel_label(axb, "b")

# ------------------------------------------------------------ (c) bottleneck hydrographs
rh, gap, L, Wd = 0.40, 0.16, 3.72, 3.40
for i, c in enumerate(CONDS):
    axe = hf.axes_in(fig, L, TOP2 + 0.30 + i * (rh + gap), Wd, rh)
    d = hyd[hyd.condition == c]
    r = bk.loc[c]
    axe.plot(d.date, d.q_obs, color=hf.INK, lw=0.9)
    axe.plot(d.date, d.q_core_best, color=hf.MUTED, lw=1.0, ls="--")
    axe.plot(d.date, d.q_forced, color=hf.COND[c], lw=0.9, alpha=0.95)
    axe.set_ylim(0, np.nanpercentile(d.q_obs, 99.7) * 1.45)
    lab = f"{i + 1} · {d.gauge_id.iloc[0]}"
    if i == 0:
        hf.panel_label(axe, "c", x=-0.14, y=1.90)
    axe.text(0.015, 1.03, lab, transform=axe.transAxes, va="bottom", fontsize=hf.FS, fontweight="bold",
             path_effects=hf.halo(3), zorder=50)
    axe.text(0.99, 1.03, f"KGE {hf.fmt_half_up(r.kge_core, 2)} → {hf.fmt_half_up(r.kge_forced, 2)}  (q05 {hf.fmt_half_up(r.q05_best, 2)})",
             transform=axe.transAxes, ha="right", va="bottom", fontsize=hf.FS_SMALL,
             color=hf.COND[c], fontweight="bold", path_effects=hf.halo(2.6), zorder=50)
    axe.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, steps=[1, 2, 5, 10]))
    ticks = axe.yaxis.get_major_locator().tick_values(*axe.get_ylim())
    axe.set_yticks(ticks[(ticks >= 0) & (ticks <= axe.get_ylim()[1])])
    axe.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axe.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axe.tick_params(axis="y", labelsize=hf.FS_SMALL)
    if i < 2:
        axe.tick_params(labelbottom=False)
    if i == 1:
        axe.set_ylabel("Q (mm d$^{-1}$)", labelpad=6)
leg = fig.legend(handles=[Line2D([], [], color=hf.INK, lw=0.9), Line2D([], [], color=hf.MUTED, lw=1.0, ls="--"),
                          Line2D([], [], color=hf.MUTED, lw=1.0)],
                 labels=["Observed", "Core model", "Non-core model"], ncol=3, loc="upper left",
                 bbox_to_anchor=((L - 0.05) / W, 1 - (TOP2 - 0.06) / H), handlelength=1.6, columnspacing=1.0,
                 fontsize=hf.FS_SMALL, borderaxespad=0)
hf.save(fig, "fig03")
