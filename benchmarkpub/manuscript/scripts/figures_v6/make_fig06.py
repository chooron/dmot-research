"""Fig. 6 — Selection criteria without local streamflow (outer-fold-isolated).  7.2 x 3.85 in

(a) sorted per-basin paired differences against B(2): C(2) (reference seed, 10-seed envelope), C(5),
    Structural(2) and Parsimony(<= 7) (thin grey: caps 5.75 and 9.25)
(b) map of C(5) - B(5) (reference seed)
(c) set-level gain G_set: random-assignment null, seen-label diagnostic, withheld k = 2, 3, 5
Inputs: ./data, ./stats.json.  Output: ./output/fig6_selection_criteria.{png,pdf}
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hydrofig as hf

hf.setup()
S = json.load(open("stats.json"))
d = pd.read_csv("data/L1_rule_per_basin_r3_with_gauge_id.csv", dtype={"gauge_id": str})
p1 = pd.read_csv("data/P1_parsimony_rule_per_basin_with_gauge_id.csv", dtype={"gauge_id": str})
ref = pd.read_csv("data/oob_reference_by_gauge_id.csv", dtype={"gauge_id": str}).set_index("gauge_id")
co = pd.read_csv("data/gauge_coordinates_531.csv", dtype={"gauge_id": str}).set_index("gauge_id")
K = d.pivot(index="gauge_id", columns="rule", values="kge")
for r in ["P(5.75)", "P(7)", "P(9.25)"]:
    K[r] = p1[p1.rule == r].set_index("gauge_id").kge
K = K.loc[ref.index]
SEEDS = [f"seed={s}" for s in range(20260925, 20260935)]
REF = "seed=20260925"
N = len(K)

W, H = hf.FIG_W, 3.85
fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------ (a) sorted paired differences vs B(2)
axa = hf.axes_in(fig, 0.62, 0.10, 2.78, 3.20)
YL = 0.40
q = (np.arange(N) + 0.5) / N
dif = lambda r: np.sort((K[r] - K["B(2)"]).values)
env = np.array([dif(f"C(2)|{s}") for s in SEEDS])
axa.fill_between(q, env.min(0), env.max(0), color=hf.BLUE, alpha=0.3, lw=0, zorder=2)
for r in ["P(5.75)", "P(9.25)"]:
    axa.plot(q, dif(r), color="#B5B5B5", lw=0.7, zorder=1)
lines = [  # key, label, colour, ls, lw
    (f"C(2)|{REF}", "C(2)", hf.BLUE, "-", 1.6),
    (f"C(5)|{REF}", "C(5)", "#86B1D4", (0, (4, 1.5)), 1.3),
    ("P(7)", "Parsimony (≤ 7)", hf.RED, (0, (4, 1.5)), 1.3),
    ("Structural(2)", "Structural (2)", hf.RED, "-", 1.6),
]
for key, lab, c, ls, lw in lines:
    axa.plot(q, np.clip(dif(key), -YL, YL), color=c, ls=ls, lw=lw, zorder=4, label=lab)
axa.plot([], [], color="#B5B5B5", lw=0.7, label="Parsimony (≤ 5.75, ≤ 9.25)")
axa.fill_between([], [], [], color=hf.BLUE, alpha=0.3, lw=0, label="C(2), 10 seeds")
axa.axhline(0, color=hf.INK, lw=0.7, zorder=3)
axa.set_xlim(0, 1); axa.set_ylim(-YL, YL)
axa.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); axa.set_xticklabels(["0", "25", "50", "75", "100"])
axa.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
axa.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.1f}".replace("-", "−")))
axa.set_xlabel("Withheld basins, sorted (%)", labelpad=1)
axa.set_ylabel("KGE of rule − KGE of B(2)", labelpad=1)
# off-scale counts
nlo = int((dif("Structural(2)") < -YL).sum() + 0)
axa.text(0.99, -0.36, f"Structural (2): {nlo} basins < −0.4", ha="right", va="top",
         fontsize=hf.FS_SMALL, color=hf.RED, path_effects=hf.halo(2))
# lose / same / gain shares
P = S["paired"]
txt = [("C(2)", P[f"C(2)|{REF} - B(2)"], hf.BLUE), ("Structural (2)", P["Structural(2) - B(2)"], hf.RED)]
for i, (lab, v, c) in enumerate(txt):
    axa.text(0.985, -0.20 - 0.05 * i, f"{lab}: {v['lose']*100:.0f} / {v['same']*100:.0f} / {v['gain']*100:.0f} %",
             ha="right", va="top", fontsize=hf.FS_SMALL, color=c, path_effects=hf.halo(2))
axa.text(0.985, -0.30, "share of basins worse / equal / better", ha="right", va="top", fontsize=hf.FS_SMALL,
         color=hf.MUTED, style="italic")
axa.legend(loc="upper left", fontsize=hf.FS_SMALL, handlelength=1.6, labelspacing=0.2, borderaxespad=0.3)
hf.panel_label(axa, "a", x=0.05, y=1.02)

# ------------------------------------------------------------ (b) map C(5) - B(5)
PAD = 150
mw = 3.42; mh = hf.map_h(mw, PAD)
axb = hf.axes_in(fig, 3.74, 0.06, mw, mh)
hf.basemap(axb, pad_top=PAD, neighbours=False)
dv = K[f"C(5)|{REF}"] - K["B(5)"]
c2 = co.reindex(dv.index)
ok = c2.lat.notna().values
x, y = hf.albers(c2.lon.values[ok], c2.lat.values[ok])
v = dv.values[ok]
vm = hf.robust_max(np.abs(v), 95)
o = np.argsort(np.abs(v))
map_div = mcolors.LinearSegmentedColormap.from_list("condition_blue_white_red", [hf.BLUE, "#FFFFFF", hf.RED])
sc = axb.scatter(x[o], y[o], c=v[o], cmap=map_div, norm=mcolors.Normalize(-vm, vm), s=10, lw=0.25,
                 ec="#6E6E6E", zorder=3)
pc = P[f"C(5)|{REF} - B(5)"]
hf.map_cbar(fig, axb, sc, "C(5) − B(5) (KGE)", loc=(0.035, 0.09, 0.32, 0.045), ticks=[-vm, 0, vm],
            ticklabels=[f"−{vm:.2f}", "0", f"+{vm:.2f}"], extend="both")
axb.text(0.50, -0.12, f"worse {pc['lose']*100:.0f} %  ·  equal {pc['same']*100:.0f} %  ·  better {pc['gain']*100:.0f} %",
         transform=axb.transAxes, ha="center", va="bottom", fontsize=hf.FS_SMALL, color=hf.INK, path_effects=hf.halo(2))
hf.panel_label(axb, "b")

# ------------------------------------------------------------ (c) G_set
gs = pd.read_csv("data/gset_summary.csv").set_index("estimand")
gseeds = pd.read_csv("data/gset_seeds.csv")
gnull = pd.read_csv("data/gset_null.csv").null_gset.values
lf = pd.read_csv("data/LEAKFREE_DIGEST_gset_by_seed.csv")
axc = hf.axes_in(fig, 5.02, 0.06 + mh + 0.30, 2.13, H - (0.06 + mh + 0.30) - 0.42)
rows = ["Shuffled (null)", "Seen labels, 36 models", "Withheld, k = 2", "Withheld, k = 3", "Withheld, k = 5"]
ys = np.array([4.0, 3.0, 1.8, 0.9, 0.0])
lo, hi = np.percentile(gnull, [5, 95])
axc.plot([lo, hi], [ys[0]] * 2, color="#BDBDBD", lw=3.2, solid_capstyle="round", zorder=2)
axc.scatter(np.median(gnull), ys[0], s=22, color=hf.MUTED, ec="white", lw=0.6, zorder=3)
sv = gseeds[gseeds.estimand == "seen_k36"].gset.values
axc.scatter(sv, np.full(len(sv), ys[1]) + 0.25, s=6, color=hf.COND["seen"], alpha=0.55, lw=0, zorder=2)
axc.plot([gs.loc["seen_k36", "boot_lo"], gs.loc["seen_k36", "boot_hi"]], [ys[1]] * 2, color=hf.COND["seen"], lw=1.4, zorder=2)
axc.scatter(gs.loc["seen_k36", "gset"], ys[1], s=24, color=hf.COND["seen"], ec="white", lw=0.6, zorder=3)
for k, yy in zip([2, 3, 5], ys[2:]):
    t = lf[lf.k == k]
    r = t[t.is_reference_seed].iloc[0]
    axc.scatter(t.G_set, np.full(len(t), yy) + 0.25, s=6, color=hf.RED, alpha=0.55, lw=0, zorder=2)
    axc.plot([r.ci95_lo, r.ci95_hi], [yy, yy], color=hf.RED, lw=1.4, zorder=2)
    axc.scatter(r.G_set, yy, s=24, color=hf.RED, ec="white", lw=0.6, zorder=3)
axc.axvline(0, color=hf.INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
axc.set_yticks(ys); axc.set_yticklabels(rows, fontsize=hf.FS_SMALL)
for tl, c in zip(axc.get_yticklabels(), [hf.MUTED, hf.COND["seen"], hf.RED, hf.RED, hf.RED]):
    tl.set_color(c)
axc.tick_params(axis="y", length=0, pad=2)
axc.spines["left"].set_visible(False)
axc.set_ylim(-0.45, 4.55); axc.set_xlim(-0.55, 0.32)
axc.set_xticks([-0.4, -0.2, 0, 0.2])
axc.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:.1f}".replace("-", "−")))
axc.set_xlabel("Set-level gain $G_{\\mathrm{set}}$", labelpad=1)
hf.panel_label(axc, "c", x=(3.74 - 5.02) / 2.13, y=1.15)

hf.save(fig, "fig6_selection_criteria")
