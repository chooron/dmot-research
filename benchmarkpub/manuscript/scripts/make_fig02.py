"""Fig. 2 — Transfer penalty across models and basins.   (7.2 x 3.2 in)

(a) HERO  CONUS map of the 36-model median transfer penalty (hexagonal bins, median of basins), worst 5 % ringed
(b)       dot plot: 36 models x 5 folds, with zero reference
(c)       variance partition of the penalty (basin / residual / model)
(d)       concentration curves: cumulative share of the total penalty, basins vs models
          (signed sums; the basin curve exceeds 100 % because some basins gain when held out)

Inputs: recomputed pub_same_pipeline_model_basin.csv and pub_variance_decomposition.csv.
Panels a, b and d use canonical seen-dPL; panel c uses median-of-four same-pipeline seen folds.
Output: manuscript/figures/fig2_transfer_penalty.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hydrofig as hf

hf.setup()
from pathlib import Path

MANUSCRIPT = Path(__file__).resolve().parents[1]
SOURCE = MANUSCRIPT.parents[0] / "analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables"
TABLES = MANUSCRIPT / "tables/figures"
full = pd.read_csv(SOURCE / "pub_same_pipeline_model_basin.csv", dtype={"gauge_id_str": str})
if (len(full) != 36 * 531 or full.duplicated(["model", "gauge_id_str"]).any()
        or full.model.nunique() != 36 or full.gauge_id_str.nunique() != 531):
    raise ValueError("Expected 36 unique model scores for each of 531 basins")
pairs = full.rename(columns={"gauge_id_str": "gauge_id", "heldout_fold": "fold",
                             "delta_kge_existing": "delta_kge_pub"})
if not np.isfinite(pairs[["kge_seendpl_canonical", "kge_heldout", "delta_kge_pub"]].to_numpy()).all():
    raise ValueError("Non-finite scores in the recomputed analysis")
pb = pairs.groupby("gauge_id").delta_kge_pub.agg(median_penalty="median", sum_penalty="sum").reset_index()
coords = pd.read_csv(TABLES / "gauge_coordinates_531.csv", dtype={"gauge_id": str})
pb = pb.merge(coords[["gauge_id", "lon", "lat"]], on="gauge_id", validate="one_to_one")
if len(pb) != 531 or pb[["lon", "lat"]].isna().any().any():
    raise ValueError("Missing or duplicate basin coordinates")
n_top = int(0.05 * len(pb))
worst = pb.nlargest(n_top, "sum_penalty")
pb["is_worst5pct"] = pb.gauge_id.isin(worst.gauge_id)
mf = pairs.groupby(["model", "fold"], as_index=False).delta_kge_pub.median().rename(columns={"delta_kge_pub": "median_penalty"})
variance = pd.read_csv(SOURCE / "pub_variance_decomposition.csv").set_index("component")
vp = 100 * variance.frac_of_total_ss
display = pd.Series({"top5pct_basin_share": worst.sum_penalty.sum() / pb.sum_penalty.sum(),
                     "positive_fold_count": (mf.median_penalty > 0).sum(), "fold_count": len(mf)})
pb.to_csv(TABLES / "fig02_penalty_basin_current.csv", index=False)
mf.to_csv(TABLES / "fig02_penalty_model_fold_current.csv", index=False)
variance.to_csv(TABLES / "fig02_variance_partition_current.csv")
pairs[["gauge_id", "model", "fold", "delta_kge_pub"]].to_csv(TABLES / "fig02_kge_pairs_current.csv", index=False)
RED = hf.COND["seen"]  # retained local alias; panels b/d now use the seen green

W, H = hf.FIG_W, 3.48
fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------ (a) hero map
PAD = 150
mw = 3.95; mh = hf.map_h(mw, PAD)
axm = hf.axes_in(fig, 0.03, 0.04, mw, mh)
clip = hf.basemap(axm, pad_top=PAD)
x, y = hf.albers(pb.lon, pb.lat)
norm = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.3)
hb = axm.hexbin(x, y, C=pb.median_penalty, reduce_C_function=np.median, gridsize=(34, 15),
                extent=(*hf.MAP_XLIM, *hf.MAP_YLIM), cmap="RdBu_r", norm=norm,
                linewidths=0.4, edgecolors="white", zorder=2, mincnt=1)
hb.set_clip_path(clip)
w = pb.is_worst5pct.values.astype(bool)
axm.scatter(x[w], y[w], s=24, facecolor="none", edgecolor=hf.INK, lw=1.0, zorder=20)   # above every layer
hf.map_cbar(fig, axm, hb, "Transfer penalty ΔKGE", loc=(0.12, -0.10, 0.70, 0.04), extend="max",
            ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3])
hf.panel_label(axm, "a")

# ------------------------------------------------------------ (b) vertical model x fold dot plot
axb = hf.axes_in(fig, 4.55, 0.06, 2.62, 1.24)
order = mf.groupby("model").median_penalty.median().sort_values().index.tolist()
model_x = {m: i for i, m in enumerate(order)}
for mod, g in mf.groupby("model"):
    xx = model_x[mod]
    vals = g.sort_values("fold").median_penalty.to_numpy(float)
    axb.plot([xx, xx], [vals.min(), vals.max()], color=RED, lw=0.9, zorder=1)
    axb.scatter(np.full(len(vals), xx), vals, s=11, color=RED, edgecolor="white", linewidth=0.25, zorder=3)
axb.axhline(0, color=hf.INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
axb.set_xlim(-0.8, len(order) - 0.2)
ylo, yhi = float(mf.median_penalty.min()), float(mf.median_penalty.max())
pad = 0.10 * (yhi - ylo)
axb.set_ylim(ylo - pad, yhi + pad)
axb.set_yticks([0, 0.05, 0.10, 0.15])
axb.set_xticks([])
axb.set_xlabel("Models × folds\n(ordered by median penalty)", fontsize=hf.FS_SMALL, labelpad=1)
axb.set_ylabel("ΔKGE", labelpad=3)
axb.text(0.75, 0.96, f"{int(display['positive_fold_count'])}/{int(display['fold_count'])} > 0", transform=axb.transAxes, ha="right", va="top",
         fontsize=hf.FS_SMALL, color=RED, fontweight="bold", path_effects=hf.halo())
hf.panel_label(axb, "b")

# ------------------------------------------------------------ (c) horizontal stacked variance partition
axc = hf.axes_in(fig, 4.55, 1.68, 2.62, 0.40)
segs = [("basin", "Basin", hf.GREEN), ("residual", "Residual", hf.LIGHT),
        ("model", "Model", hf.INK)]
left = 0.0
for key, lab, col in segs:
    v = float(vp[key])
    axc.barh(0.42, v, left=left, height=0.44, color=col, edgecolor="white", lw=0.6, zorder=2)
    mid = left + v / 2
    txt = f"{lab} {hf.fmt_half_up(v, 0)} %"
    if v >= 12:                                            # wide enough to be labelled inside
        axc.text(mid, 0.42, txt, ha="center", va="center", fontsize=hf.FS_SMALL,
                 color="white" if key == "basin" else hf.INK, zorder=4)
    else:                                                  # sliver: label above the right end of the bar
        axc.annotate(txt, xy=(left + v - 0.5, 0.66), xytext=(100, 0.94), ha="right", va="bottom",
                     fontsize=hf.FS_SMALL, color=hf.INK,
                     arrowprops=dict(arrowstyle="-", lw=0.55, color=hf.INK))
    left += v
axc.set_xlim(0, 100); axc.set_ylim(0, 1.3)
axc.set_xticks([]); axc.set_yticks([])
for sp in axc.spines.values():
    sp.set_visible(False)
hf.panel_label(axc, "c", x=0.0, y=1.02)

# ------------------------------------------------------------ (d) concentration curves
axd = hf.axes_in(fig, 4.55, 2.18, 2.62, 0.92)
bsum = pairs.groupby("gauge_id").delta_kge_pub.sum().sort_values(ascending=False).values
msum = pairs.groupby("model").delta_kge_pub.sum().sort_values(ascending=False).values
peak = 100
for arr, col, lw, ls, lab in ((bsum, RED, 2.0, "-", "Basins"), (msum, hf.INK, 1.4, (0, (3, 2)), "Models")):
    xs = np.r_[0, np.arange(1, len(arr) + 1) / len(arr)] * 100
    ys = np.r_[0, np.cumsum(arr) / arr.sum()] * 100
    peak = max(peak, ys.max())
    axd.plot(xs, ys, color=col, lw=lw, ls=ls)
    i = int(len(xs) * 0.62)
    axd.text(xs[i], ys[i] - (12 if lab == "Basins" else 8), lab, color=col, fontsize=hf.FS_SMALL,
             ha="left", va="bottom" if lab == "Basins" else "top",
             fontweight="bold", path_effects=hf.halo())
axd.axhline(100, color="#BDBDBD", lw=0.7, ls=(0, (3, 2)))
share = 100 * float(display["top5pct_basin_share"])
axd.plot([5, 5], [0, share], color=RED, lw=0.8, ls=":")
axd.plot(5, share, "o", ms=4, color=RED)
axd.text(14, peak * 1.14, f"top 5 % of basins: {hf.fmt_half_up(display['top5pct_basin_share'] * 100, 0)} %", color=RED, fontsize=hf.FS_SMALL, va="center",
         fontweight="bold", path_effects=hf.halo())
axd.set_xlim(0, 106); axd.set_ylim(0, peak * 1.28); axd.set_xticks([0, 20, 40, 60, 80, 100])
axd.set_xlabel("Basins or models ranked by penalty (%)", labelpad=1)
axd.set_ylabel("Share (%)")
axd.set_yticks([0, 50, 100])
hf.panel_label(axd, "d")
hf.save(fig, "fig2_transfer_penalty", outdir=str(MANUSCRIPT / "figures"), png_only=True)
