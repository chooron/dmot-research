"""Fig. 5 — Potential coverage is not realized without local streamflow (outer-fold-isolated rules).  7.2 x 4.7 in

(a) HERO  realization gap: share of withheld basins near-optimal (KGE >= q05_b) for each rule, against the
          95 % coverage of the post-hoc 12-model library and the retention of the best seen model
(b)       ridgeline of paired differences B(k) - A
(c)       map of the regret of B(2) relative to the best single OOB-dPL model; worst 5 % circled
(d)       example basin 01451800, water year 1996
Inputs: ./data (figure_redraw_source_bundle_20261007), ./stats.json.  Output: ./output/fig5_usage_rules.{png,pdf}
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import hydrofig as hf

hf.setup()
S = json.load(open("stats.json"))
d = pd.read_csv("data/L1_rule_per_basin_r3_with_gauge_id.csv", dtype={"gauge_id": str})
K = d.pivot(index="gauge_id", columns="rule", values="kge")
ref = pd.read_csv("data/oob_reference_by_gauge_id.csv", dtype={"gauge_id": str}).set_index("gauge_id")
co = pd.read_csv("data/gauge_coordinates_531.csv", dtype={"gauge_id": str}).set_index("gauge_id")
hy = pd.read_csv("data/example_hydrograph.csv", parse_dates=["date"], dtype={"gauge_id": str})
K = K.loc[ref.index]
rng = np.random.default_rng(20260925)

W, H = hf.FIG_W, 4.70
fig = plt.figure(figsize=(W, H))
ORANGE_NONE = None

# ------------------------------------------------------------ (a) realization gap
axa = hf.axes_in(fig, 1.30, 0.06, 2.14, 2.36)
R = S["rules"]; SD = S["seeds"]
rows = [  # label, value(%), range(%) or None, style
    ("E: best of 36", 100 * R["E"]["near"], None, "ref"),
    ("D: best of top 3", 100 * R["D"]["near"], [100 * v for v in SD["D"]["near"]], "ref"),
    ("Best member of B(2)", 100 * 353 / 531, None, "ref"),
    ("C(2)", 100 * R["C(2)"]["near"], [100 * v for v in SD["C(2)"]["near"]], "C"),
    ("B(2)", 100 * R["B(2)"]["near"], None, "B"),
    ("C(3)", 100 * R["C(3)"]["near"], [100 * v for v in SD["C(3)"]["near"]], "C"),
    ("C(5)", 100 * R["C(5)"]["near"], [100 * v for v in SD["C(5)"]["near"]], "C"),
    ("B(3)", 100 * R["B(3)"]["near"], None, "B"),
    ("A", 100 * R["A"]["near"], None, "B"),
    ("B(5)", 100 * R["B(5)"]["near"], None, "B"),
    ("B(10)", 100 * R["B(10)"]["near"], None, "B"),
    ("Parsimony (≤ 7)", 100 * R["P(7)"]["near"], None, "X"),
    ("Structural (2)", 100 * R["Structural(2)"]["near"], None, "X"),
]
n = len(rows)
ys = np.arange(n)[::-1].astype(float)
ys[:3] += 0.45                                             # gap between reference block and realizable rules
axa.axvspan(53.3, 56.9, ymax=0.65, color=hf.COND["seen"], alpha=0.16, lw=0, zorder=0)
axa.axvline(95, color=hf.INK, lw=1.0, ls=(0, (4, 2)), zorder=1)
col = {"B": hf.INK, "C": hf.BLUE, "X": hf.RED, "ref": hf.MUTED}
for (lab, v, rg, st), y in zip(rows, ys):
    axa.plot([20, v if v < 93 else v - 18], [y, y], color="#E3E3E3", lw=0.8, zorder=1)
    if rg:
        axa.plot(rg, [y, y], color=col[st], lw=3.2, alpha=0.35, solid_capstyle="round", zorder=2)
    if st == "ref":
        axa.scatter(v, y, marker="D", s=30, fc="white", ec=hf.MUTED, lw=1.1, zorder=3)
    else:
        axa.scatter(v, y, s=30, color=col[st], ec="white", lw=0.6, zorder=3)
    axa.text(v + 1.8 if v < 93 else v - 7.5, y, f"{v:.0f}", ha="left" if v < 93 else "right", va="center",
             fontsize=hf.FS_SMALL, color=col[st], path_effects=hf.halo(2))
axa.set_yticks(ys); axa.set_yticklabels([r[0] for r in rows], fontsize=hf.FS_SMALL)
for tl, r in zip(axa.get_yticklabels(), rows):
    tl.set_color(col[r[3]] if r[3] != "ref" else hf.MUTED)
axa.tick_params(axis="y", length=0, pad=2)
axa.spines["left"].set_visible(False)
axa.set_xlim(20, 102); axa.set_ylim(-0.7, ys[0] + 1.25)
axa.set_xticks([20, 40, 60, 80, 100])
axa.set_xlabel("Withheld basins near-optimal (%)", labelpad=1)
yb2 = ys[4]
axa.annotate("", xy=(95, yb2), xytext=(63, yb2),
             arrowprops=dict(arrowstyle="<->", lw=0.9, color=hf.RED, shrinkA=0, shrinkB=0))
axa.text(79, yb2 + 0.16, f"{95 - R['B(2)']['near'] * 100:.0f} pp",
         ha="center", va="bottom", fontsize=hf.FS_SMALL, color=hf.RED, path_effects=hf.halo(2))
axa.text(93.4, ys[-1] - 0.3, "12-model library", ha="right", va="bottom", rotation=90, fontsize=hf.FS_SMALL,
         color=hf.INK, path_effects=hf.halo(2))
axa.text(87.0, (ys[-2] + ys[-1]) / 2, "Best seen model\nretained", ha="right", va="center", fontsize=hf.FS_SMALL,
         color=hf.COND["seen"], linespacing=0.95, path_effects=hf.halo(2))
axa.text(20, (ys[2] + ys[3]) / 2, "above: requires local streamflow", ha="left", va="center", fontsize=hf.FS_SMALL,
         color=hf.MUTED, style="italic")
hf.panel_label(axa, "a")

# ------------------------------------------------------------ (b) ridgeline B(k) - A
KS = [2, 3, 4, 5, 7, 10]
axb = hf.axes_in(fig, 0.50, 2.84, 2.94, 1.42)
diffs = {k: (K[f"B({k})"] - K["A"]).values for k in KS}
XL = 0.12
grid = np.linspace(-XL, XL, 400)
dnorm = mcolors.TwoSlopeNorm(vmin=-0.015, vcenter=0, vmax=0.015)
step, RH = 1.0, 1.25
pts = []
for i, k in enumerate(KS):
    x = diffs[k]
    dens = hf.kde(x, grid); dens = dens / dens.max() * RH
    base = (len(KS) - 1 - i) * step
    md = np.median(x)
    bs = np.median(x[rng.integers(0, len(x), (1000, len(x)))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    z = 10 + 2 * i
    axb.fill_between(grid, base, base + dens, color=hf.SEQ_MAG(0.05 + 0.9 * i / (len(KS) - 1)), lw=0, zorder=z)
    axb.plot(grid, base + dens, color="white", lw=1.4, zorder=z + 0.5)
    axb.plot(grid, base + dens, color=hf.INK, lw=0.5, zorder=z + 0.6)
    axb.plot([-XL, XL], [base, base], color="#9E9E9E", lw=0.5, zorder=z + 0.6)
    axb.text(-XL - 0.004, base + 0.12, f"k = {k}", ha="right", va="center", fontsize=hf.FS_SMALL)
    pts.append((md, lo, hi, base + 0.16))
for md, lo, hi, yy in pts:
    axb.plot([lo, hi], [yy, yy], color=hf.INK, lw=1.5, zorder=40, solid_capstyle="round")
axb.plot([p[0] for p in pts], [p[3] for p in pts], color=hf.INK, lw=1.0, zorder=41)
axb.scatter([p[0] for p in pts], [p[3] for p in pts], s=20, color="white", ec=hf.INK, lw=1.0, zorder=42)
axb.axvline(0, color=hf.INK, lw=0.7, ls=(0, (3, 2)), zorder=39)
axb.set_xlim(-XL, XL); axb.set_ylim(-0.1, (len(KS) - 1) * step + RH + 0.1)
axb.set_yticks([]); axb.spines["left"].set_visible(False)
axb.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
axb.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.2f}".replace("-", "−")))
axb.set_xlabel("KGE of B(k) − KGE of A", labelpad=1)
hf.panel_label(axb, "b")

# ------------------------------------------------------------ (c) regret map of B(2)
PAD = 150
mw = 3.40; mh = hf.map_h(mw, PAD)
axc = hf.axes_in(fig, 3.76, 0.06, mw, mh)
hf.basemap(axc, pad_top=PAD, neighbours=False)
reg = (ref.oob_best_single_kge - K["B(2)"])
c2 = co.reindex(reg.index)
ok = c2.lat.notna().values
x, y = hf.albers(c2.lon.values[ok], c2.lat.values[ok])
v = reg.values[ok]
vm = 0.3
o = np.argsort(v)
sc = axc.scatter(x[o], y[o], c=np.clip(v[o], 0, None), cmap=hf.SEQ_MAG, norm=mcolors.Normalize(0, vm), s=10, lw=0.25,
                 ec="#6E6E6E", zorder=3)
worst = reg.sort_values().index[-int(np.ceil(0.05 * len(reg))):]
w2 = co.reindex(worst).dropna()
xw, yw = hf.albers(w2.lon.values, w2.lat.values)
axc.scatter(xw, yw, s=46, fc="none", ec=hf.INK, lw=0.9, zorder=4)
neg = v < -1e-9
axc.scatter(x[neg], y[neg], s=10, fc="white", ec=hf.BLUE, lw=0.8, zorder=3.5)
hf.map_cbar(fig, axc, sc, "B(2) regret", loc=(0.02, -0.12, 0.24, 0.035), ticks=[0, .1, .2, .3],
            extend="max")
axc.scatter([], [], s=46, fc="none", ec=hf.INK, lw=0.9, label="worst 5 %")
axc.scatter([], [], s=10, fc="white", ec=hf.BLUE, lw=0.8, label="B(2) > best single model")
axc.legend(loc="center left", bbox_to_anchor=(0.29, -0.11), ncol=2, frameon=False,
           handlelength=0.6, columnspacing=0.4, handletextpad=0.1, borderaxespad=0, fontsize=hf.FS_SMALL)
hf.panel_label(axc, "c")

# ------------------------------------------------------------ (d) example hydrograph
axd = hf.axes_in(fig, 4.20, 0.06 + mh + 0.62, 2.95, H - (0.06 + mh + 0.62) - 0.46)
ens = 0.5 * (hy.q_member1 + hy.q_member2)
axd.plot(hy.date, hy.q_obs, color=hf.INK, lw=1.0, label="Observed", zorder=4)
axd.plot(hy.date, hy.q_member1, color="#8C8C8C", lw=0.8, label="HBV-96", zorder=2)
axd.plot(hy.date, hy.q_member2, color="#C8C8C8", lw=0.8, label="MOPEX-5", zorder=1)
axd.plot(hy.date, ens, color=hf.BLUE, lw=1.3, label="B(2) mean", zorder=3)
axd.set_ylim(0, np.nanmax(hy[["q_obs", "q_member1", "q_member2"]].values) * 1.1)
axd.set_ylabel("Q (mm d$^{-1}$)", labelpad=1)
axd.yaxis.set_major_locator(plt.MaxNLocator(3))
axd.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axd.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axd.set_xlabel("Water year 1996", labelpad=1)
axd.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, fontsize=hf.FS_SMALL,
           labelspacing=0.15, borderaxespad=0.1)
hf.panel_label(axd, "d", extra=hy.gauge_id.iloc[0], extra_color=hf.INK)

hf.save(fig, "fig5_usage_rules")
