"""Fig. 4 — Model rankings transfer in aggregate; the identity of the local winner does not.   (7.2 x 3.7 in)

(a) HERO  alluvial diagram: best model per basin under seen-dPL (left) -> under OOB-dPL (right);
          8 most frequent winners + "other"; flows that keep the same winner drawn dark
(b)       map of the per-basin Spearman correlation between the seen-dPL and OOB-dPL model rankings
(c)       distribution of the per-basin correlation; solid line = correlation of the global ranking
(d)       coverage ladder: share of basins whose OOB winner lies within their own seen Top-k

Inputs: recomputed same-pipeline model/basin scores and current IC rank table.
Panels a and d use canonical seen-dPL; panels b and c use median-of-four seen folds.
Output: manuscript/figures/fig4_identity_transfer.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
import hydrofig as hf

hf.setup()
from pathlib import Path as FilePath
from scipy.stats import spearmanr

MANUSCRIPT = FilePath(__file__).resolve().parents[1]
TABLES = MANUSCRIPT / "tables/figures"
SOURCE = MANUSCRIPT.parent / "analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables"
full = pd.read_csv(SOURCE / "pub_same_pipeline_model_basin.csv", dtype={"gauge_id_str": str})
if (len(full) != 36 * 531 or full.duplicated(["model", "gauge_id_str"]).any()
        or full.model.nunique() != 36 or full.gauge_id_str.nunique() != 531):
    raise ValueError("Expected 36 unique model scores for each of 531 basins")
mods = pd.read_csv(TABLES / "appendix_a1/fig_a1_model_summary.csv").set_index("model").rename(columns={"ic_rank": "global_rank"})
coords = pd.read_csv(TABLES / "gauge_coordinates_531.csv", dtype={"gauge_id": str})
rows = []
for basin, group in full.groupby("gauge_id_str", sort=True):
    scores = group.set_index("model").sort_index()
    canonical = scores.kge_seendpl_canonical.sort_values(ascending=False, kind="stable")
    best_oob = scores.kge_heldout.idxmax()
    rows.append({"gauge_id": basin, "winner_seen": canonical.index[0], "winner_oob": best_oob,
                 "rank_spearman": spearmanr(scores.kge_seenfolds_median4, scores.kge_heldout).statistic,
                 "oob_winner_seen_rank": canonical.index.get_loc(best_oob) + 1})
bt = pd.DataFrame(rows).merge(coords[["gauge_id", "lon", "lat"]], on="gauge_id", validate="one_to_one")
if len(bt) != 531 or bt[["lon", "lat", "rank_spearman"]].isna().any().any():
    raise ValueError("Incomplete basin coordinates or rank correlations")
seen = full.groupby("model").kge_seenfolds_median4.median()
oob = full.groupby("model").kge_heldout.median()
gr = pd.DataFrame({"value": {"global_rank_spearman": spearmanr(seen, oob).statistic,
                              "basin_rank_spearman_median": bt.rank_spearman.median(),
                              "winner_retention": (bt.winner_seen == bt.winner_oob).mean()}})
lad = pd.DataFrame({"k": range(1, 11)})
lad["coverage"] = [(bt.oob_winner_seen_rank <= k).mean() for k in lad.k]
bt.to_csv(TABLES / "fig04_basin_transfer_current.csv", index=False)
gr.to_csv(TABLES / "fig04_granularity_current.csv", index_label="metric")
lad.to_csv(TABLES / "fig04_coverage_ladder_current.csv", index=False)
NTOP = 8

W, H = hf.FIG_W, 3.86
fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------ (a) alluvial
axa = hf.axes_in(fig, 0.03, 0.06, 2.78, 3.32)
counts = pd.concat([bt.winner_seen, bt.winner_oob]).value_counts()
top = sorted(counts.index[:NTOP], key=lambda m: mods.loc[m, "global_rank"])
cat = lambda s: s.where(s.isin(top), "other")
L, R = cat(bt.winner_seen), cat(bt.winner_oob)
cats = top + ["other"]
n = len(bt); gap = 0.012 * n; bw = 0.07
def stack(series):
    pos, y = {}, n + gap * (len(cats) - 1)
    for c in cats:
        h = int((series == c).sum()); pos[c] = [y - h, y]; y -= h + gap
    return pos
pl, pr = stack(L), stack(R)
rank_by_model = mods.global_rank.to_dict()
def model_color(c):
    if c == "other":
        return "#F2F2F2"
    u = (float(rank_by_model[c]) - 1.0) / 35.0
    return hf.SEQ_MAG(0.30 + 0.55 * u)
col = {c: model_color(c) for c in cats}
curl = {c: pl[c][1] for c in cats}; curr = {c: pr[c][1] for c in cats}
flows = pd.crosstab(L, R).reindex(index=cats, columns=cats, fill_value=0)
x0, x1 = bw, 1 - bw
for a in cats:
    for b in cats:
        w = flows.loc[a, b]
        if w == 0:
            continue
        ya1 = curl[a]; ya0 = ya1 - w; curl[a] = ya0
        yb1 = curr[b]; yb0 = yb1 - w; curr[b] = yb0
        keep = (a == b) and a != "other"
        verts = [(x0, ya1), (0.5, ya1), (0.5, yb1), (x1, yb1), (x1, yb0),
                 (0.5, yb0), (0.5, ya0), (x0, ya0), (x0, ya1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        axa.add_patch(PathPatch(Path(verts, codes), fc=col[a], ec="none",
                                alpha=1.0 if keep else 0.32, zorder=3 if keep else 1))
for c in cats:
    for xx, p in ((0, pl), (1 - bw, pr)):
        y0, y1 = p[c]
        axa.add_patch(Rectangle((xx, y0), bw, y1 - y0, fc=col[c], ec="white", lw=0.5, zorder=4))
ytop = n + gap * len(cats)
for side, p, edge, tx, ha in (("left", pl, x0, -0.02, "right"), ("right", pr, x1, 1.02, "left")):
    mids = [(p[c][0] + p[c][1]) / 2 for c in cats]
    reps = hf.spread(mids, min_gap=0.12 * n, lo=0.02 * n, hi=ytop - 0.02 * n)
    idx = np.argsort(reps)
    for j in range(1, len(idx)):
        reps[idx[j]] = max(reps[idx[j]], reps[idx[j - 1]] + 0.12 * n)
    reps -= max(0, reps.max() - (ytop - 0.02 * n))
    for c, mid, rep in zip(cats, mids, reps):
        y0, y1 = p[c]
        axa.plot([edge, tx], [mid, rep], color=col[c], lw=0.45, zorder=5)
        axa.text(tx, rep, hf.model_label(c), ha=ha, va="center", fontsize=hf.FS_SMALL,
                 color=hf.INK if c != "other" else hf.MUTED, zorder=6)
axa.set_xlim(-0.82, 1.33); axa.set_ylim(-0.08 * n, ytop + 0.10 * n)
axa.set_xticks([]); axa.set_yticks([])
for spine in axa.spines.values():
    spine.set_visible(False)
axa.text(bw / 2, ytop + 0.025 * n, "seen-dPL", ha="center", va="bottom", color=hf.COND["seen"], fontweight="bold")
axa.text(1 - bw / 2, ytop + 0.025 * n, "OOB-dPL", ha="center", va="bottom", color=hf.COND["OOB"], fontweight="bold")
same = 100 * float(gr.loc["winner_retention", "value"])
axa.text(0.5, -0.035 * n, f"Same best model in {hf.fmt_half_up(gr.loc['winner_retention', 'value'] * 100, 0)} % of basins", ha="center", va="top", fontsize=hf.FS_SMALL)
hf.panel_label(axa, "a", x=0.12, y=1 - (0.10 - 0.06) / 3.32)

# ------------------------------------------------------------ (b) map of per-basin rank correlation
PAD = 220
mw = 3.60; mh = hf.map_h(mw, PAD)
axb = hf.axes_in(fig, 3.42, 0.06, mw, mh)
hf.basemap(axb, pad_top=PAD)
x, y = hf.albers(bt.lon, bt.lat)
norm = mcolors.Normalize(-0.4, 1)
o = np.argsort(-bt.rank_spearman.values)              # low correlations drawn on top
sc = axb.scatter(x[o], y[o], c=bt.rank_spearman.values[o], cmap=hf.SEQ_RHO, norm=norm, s=10,
                 lw=0.25, ec="#6E6E6E", zorder=3)
hf.map_cbar(fig, axb, sc, "Rank correlation ρ", loc=(0.035, 0.09, 0.32, 0.045), ticks=[-0.4, 0, 0.5, 1],
            card=True)
hf.panel_label(axb, "b", y=1 - (0.10 - 0.06) / mh)

# ------------------------------------------------------------ (c) distribution of basin correlation
TOP2 = 0.06 + mh + 0.10
axc = hf.axes_in(fig, 3.42, TOP2, 1.62, 0.78)
bins = np.linspace(-0.4, 1.0, 29)
cnt, edges = np.histogram(bt.rank_spearman.clip(-0.4, 1.0), bins=bins)
cmid = 0.5 * (edges[1:] + edges[:-1])
axc.bar(cmid, cnt, width=np.diff(edges), color=hf.SEQ_RHO(0.6), ec="white", lw=0.3)
g = gr.loc["global_rank_spearman", "value"]; med = gr.loc["basin_rank_spearman_median", "value"]
axc.axvline(g, color=hf.INK, lw=1.5)
axc.axvline(med, color=hf.INK, lw=1.0, ymax=0.72, ls=(0, (3, 2)))
axc.text(g - 0.08, cnt.max() * 1.42, f"global {g:.2f}", ha="right", va="top", fontsize=hf.FS_SMALL,
         path_effects=hf.halo(4.0))
axc.text(med - 0.08, cnt.max() * 1.10, f"median {med:.2f}", ha="right", va="top", fontsize=hf.FS_SMALL,
         path_effects=hf.halo(4.0))
axc.set_xlim(-0.4, 1.05); axc.set_ylim(0, cnt.max() * 1.48); axc.set_xticks([0, 0.5, 1])
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.set_xlabel("Rank correlation ρ", labelpad=1)
hf.panel_label(axc, "c", y=1.17)

# ------------------------------------------------------------ (d) coverage ladder
axd = hf.axes_in(fig, 5.68, TOP2, 1.34, 0.78)
c = hf.COND["seen"]
axd.fill_between(lad.k, 0, 100 * lad.coverage, color=c, alpha=0.14, lw=0)
axd.plot(lad.k, 100 * lad.coverage, color=c, lw=1.8, marker="o", ms=3)
axd.set_ylim(0, 100); axd.set_xlim(0.6, lad.k.max() + 0.4); axd.set_xticks([1, 3, 5, 10]); axd.set_yticks([0, 50, 100])
axd.set_xlabel("Own seen Top-$k$", labelpad=1)
axd.set_ylabel("Basins (%)", labelpad=-2)
hf.panel_label(axd, "d", x=0.0, y=1.17)

hf.save(fig, "fig4_identity_transfer", outdir=str(MANUSCRIPT / "figures"), png_only=True)
