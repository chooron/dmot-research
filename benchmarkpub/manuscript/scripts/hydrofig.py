"""hydrofig v2 — shared visual system for the HESS technical note figures.

One file, copied into every figure folder (sh _tools/sync.sh).

Rules enforced here
  * Font: Carlito (SIL OFL; installed or bundled in the project assets).
  * Condition colours remain fixed: blue = IC, green = seen-dPL, red = OOB-dPL.
  * One-signed magnitudes use SEQ_MAG (larger = darker); correlations use SEQ_RHO
    (smaller = darker); diverging values use BrBG via DIV.
  * Models are greys by global rank; snow routine is a black marker, never a colour.
  * Panel letters sit outside the axes at the upper-left.
"""
import json, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ sizes & fonts
FIG_W = 7.2            # inches = 183 mm, full page width
FS = 11                # base font size (pt), Carlito
FS_SMALL = 10
FS_LABEL = 12.5        # panel letters


def fmt_half_up(value, digits=2):
    """Format a stored/frozen value with decimal ROUND_HALF_UP."""
    quantum = Decimal(1).scaleb(-digits)
    return format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), f".{digits}f")



def fmt_signed(value, digits=3):
    """ROUND_HALF_UP with explicit sign and a fixed zero representation."""
    d = Decimal(str(value))
    s = fmt_half_up(abs(d), digits)
    if d == 0:
        return "0." + ("0" * digits)
    return ("+" if d > 0 else "−") + s

def fmt_pct(value, digits=0):
    return fmt_half_up(100 * Decimal(str(value)), digits) + " %"


def tick_label(value):
    """Compact tick labels (never apply claim display rounding to axes)."""
    if not np.isfinite(value):
        return str(value)
    if abs(value) < 1e-12:
        return "0"
    return np.format_float_positional(float(value), precision=8, trim="-").replace("-", "−")


def normalize_gauge_ids(frame, column="gauge_id"):
    """Normalize basin/gauge identifiers for every figure-side join."""
    if column in frame.columns:
        values = frame[column].astype("string").str.strip()
        frame[column] = values.str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return frame


MODEL_DISPLAY = {
    "alpine1": "Alpine1", "alpine2": "Alpine2", "australia": "Australia",
    "collie1": "Collie1", "collie2": "Collie2", "collie3": "Collie3",
    "flexb": "FLEX-B", "flexi": "FLEX-I", "flexis": "FLEXIS",
    "gr4j": "GR4J", "gsfb": "GSFB", "hbv96": "HBV-96",
    "hillslope": "Hillslope", "hymod": "HyMOD", "ihacres": "IHACRES",
    "modhydrolog": "MODHYDROLOG", "mopex1": "MOPEX-1", "mopex2": "MOPEX-2",
    "mopex3": "MOPEX-3", "mopex4": "MOPEX-4", "mopex5": "MOPEX-5",
    "newzealand1": "New Zealand1", "newzealand2": "New Zealand2",
    "penman": "Penman", "plateau": "Plateau", "simhyd": "SIMHYD",
    "smar": "SMAR", "susannah1": "SUSANNAH1", "susannah2": "SUSANNAH2",
    "tank": "Tank", "tcm": "TCM", "topmodel": "TOPMODEL", "us1": "US1",
    "vic": "VIC", "wetland": "Wetland", "xinanjiang": "Xinanjiang",
}


def model_label(model):
    """Return the manuscript display name for a canonical model identifier."""
    key = str(model)
    return MODEL_DISPLAY.get(key.lower(), key)
def _font_family():
    d = os.path.join(os.environ.get("TN_ASSET_DIR", os.path.join(HERE, "assets")), "fonts")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(d, f))
    return "Carlito"

FONT = None


def setup():
    global FONT
    FONT = _font_family()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
        "mathtext.bf": FONT + ":bold", "mathtext.sf": FONT,
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#222222",
        "axes.labelpad": 2.5, "legend.frameon": False, "legend.handletextpad": 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "figure.dpi": 100, "axes.unicode_minus": True,
    })
    return FONT

# ------------------------------------------------------------------ colours (3 hue families)
BLUE, GREEN, RED = "#3A78AD", "#3F9A7B", "#C4553D"
COND = {"IC": BLUE, "seen": GREEN, "OOB": RED}
COND_LABEL = {"IC": "IC", "seen": "seen-dPL", "OOB": "OOB-dPL"}
COND_MARKER = {"IC": "o", "seen": "s", "OOB": "D"}
COND_LS = {"IC": "-", "seen": (0, (4, 1.5)), "OOB": "-"}
INK, MUTED, LIGHT = "#222222", "#8C8C8C", "#D9D9D9"
ORACLE = "#9A9A9A"
BLUE_LIGHT, RED_LIGHT = "#C9DBEB", "#F0CABE"

_ramp = mcolors.LinearSegmentedColormap.from_list
SEQ_MAG = ListedColormap(sns.color_palette("mako_r", 256)[25:235], name="mako_mag")
SEQ_RHO = ListedColormap(sns.color_palette("mako", 256)[20:230], name="mako_rho")
DIV = plt.get_cmap("BrBG")
# Backward-compatible aliases; the eight final figures use SEQ_MAG, SEQ_RHO, or DIV explicitly.
SEQ = SEQ_MAG
DIVERGING = DIV
GREYS = _ramp("greys", ["#F2F2F2", "#BDBDBD", "#6E6E6E", "#262626"])
CM_RANKGREY = _ramp("rankgrey", ["#1F1F1F", "#CFCFCF"])


def rank_grey(rank, n=36):
    """Global-rank shade for a model: rank 1 (best) darkest."""
    return CM_RANKGREY((np.asarray(rank) - 1) / max(n - 1, 1))


def robust_max(v, q=95):
    """Upper colour limit from a quantile, rounded to 2 significant digits (avoids saturation by outliers)."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    m = np.nanpercentile(np.abs(v), q)
    if m <= 0:
        return 1.0
    p = 10 ** np.floor(np.log10(m))
    return float(np.ceil(m / p * 2) / 2 * p)

# ------------------------------------------------------------------ projection
_R = 6371.0
_L1, _L2, _L0, _LON0 = np.radians([29.5, 45.5, 23.0, -96.0])
_N = (np.sin(_L1) + np.sin(_L2)) / 2
_C = np.cos(_L1) ** 2 + 2 * _N * np.sin(_L1)
_RHO0 = _R * np.sqrt(_C - 2 * _N * np.sin(_L0)) / _N


def albers(lon, lat):
    """CONUS Albers equal-area (spherical), km."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    rho = _R * np.sqrt(_C - 2 * _N * np.sin(lat)) / _N
    th = _N * (lon - _LON0)
    return rho * np.sin(th), _RHO0 - rho * np.cos(th)

# ------------------------------------------------------------------ basemap
_BM = None


def _load_bm():
    global _BM
    if _BM is None:
        d = os.environ.get("TN_ASSET_DIR", os.path.join(HERE, "assets"))
        bm = json.load(open(os.path.join(d, "conus_basemap.json")))
        rel = np.load(os.path.join(d, "conus_relief.npz"))
        bm["relief"], bm["relief_ext"] = rel["relief"], rel["extent"]
        _BM = bm
    return _BM


MAP_XLIM = (-2380, 2290)
MAP_YLIM = (230, 3220)


def us_outline_path():
    bm = _load_bm()
    verts, codes = [], []
    for poly in bm["outline"]["albers"]:
        for ring in poly:
            r = np.asarray(ring)
            verts += r.tolist(); codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def basemap(ax, relief=True, rivers=True, neighbours=False, states=True, relief_alpha=0.28, pad_top=0):
    """Draw the CONUS base map in Albers km coordinates. Returns the outline patch for clipping.
    pad_top (km) adds empty space above the northern border, used to keep panel letters off the data."""
    bm = _load_bm()
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    path = us_outline_path()
    ax.add_patch(PathPatch(path, fc="white", ec="none", zorder=0.5))
    clip = PathPatch(path, fc="none", ec="none", transform=ax.transData)
    ax.add_patch(clip)
    if relief:
        rel = np.ma.masked_equal(bm["relief"], 255)
        im = ax.imshow(rel, extent=bm["relief_ext"], origin="lower", cmap="Greys_r",
                       vmin=-200, vmax=290, alpha=relief_alpha, zorder=0.6, interpolation="bilinear")
        im.set_clip_path(clip)
    if states:
        ax.add_collection(LineCollection(bm["states"], colors="#FFFFFF", linewidths=0.55, zorder=0.8))
    if rivers:
        ax.add_collection(LineCollection(bm["rivers"], colors="#A9C6DC", linewidths=0.45, zorder=0.9))
    ax.add_patch(PathPatch(path, fc="none", ec="#6E6E6E", lw=0.6, zorder=0.95))
    ax.set_xlim(*MAP_XLIM); ax.set_ylim(MAP_YLIM[0], MAP_YLIM[1] + pad_top)
    ax.set_axis_off()
    ax._is_basin_map = True
    return clip


def map_cbar(fig, ax, mappable, label, loc=(0.03, 0.05, 0.28, 0.04), ticks=None, ticklabels=None,
             extend="neither", fmt=None, card=False, orientation="horizontal"):
    """Frameless horizontal or vertical colorbar; check map decorations before exporting."""
    cax = ax.inset_axes(loc)
    cax.patch.set_visible(False)
    ax._map_decorations = getattr(ax, "_map_decorations", []) + [cax]
    cax._map_location = loc
    cax.set_zorder(46)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, extend=extend)
    cb.outline.set_linewidth(0.5)
    cax.tick_params(labelsize=FS_SMALL, length=2, pad=1.5)
    if ticks is not None:
        if fmt and ticklabels is None:
            limit = max(abs(float(v)) for v in ticks)
            rough = limit / 3
            power = 10 ** np.floor(np.log10(rough)) if rough else 1
            step = next((m * power for m in (1, 2, 5, 10) if m * power >= rough), power)
            lo, hi = min(ticks), max(ticks)
            ticks = np.arange(np.ceil(lo / step), np.floor(hi / step) + 1) * step
            ticks = np.round(ticks, 8)
        cb.set_ticks(ticks)
        (cax.xaxis if orientation == "horizontal" else cax.yaxis).set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: tick_label(v)))
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cax.set_title(label, fontsize=FS_SMALL, pad=3, loc="left", path_effects=halo(2.6))
    for t in (cax.get_xticklabels() if orientation == "horizontal" else cax.get_yticklabels()):
        t.set_path_effects(halo(2.2))
    return cb

# ------------------------------------------------------------------ layout & annotation
def axes_in(fig, left, top, w, h, **kw):
    """Add axes with geometry in inches: `left` from the left edge, `top` from the top edge."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1 - (top + h) / H, w / W, h / H], **kw)


def map_h(w, pad_top=0):
    """Height (in) of a basemap axes of width w (in) at the fixed CONUS extent (+ pad_top km)."""
    return w * (MAP_YLIM[1] + pad_top - MAP_YLIM[0]) / (MAP_XLIM[1] - MAP_XLIM[0])


def halo(width=2.6):
    return [pe.withStroke(linewidth=width, foreground="white")]


def panel_label(ax, letter, x=0.0, y=1.02, extra=None, extra_color=None):
    """Panel letter outside the axes at upper left, optionally followed by a short coloured tag."""
    t = ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FS, fontweight="bold",
                va="top", ha="left", color=INK, zorder=50, clip_on=False, path_effects=halo(3))
    if extra:
        ax.annotate(extra, xy=(1, 0.5), xycoords=t, xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=FS, fontweight="bold", color=extra_color or INK,
                    zorder=50, path_effects=halo(3))
    return t


def tag(ax, text, color, x=0.98, y=0.97, ha="right", va="top", size=None, weight="bold"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=size or FS,
            fontweight=weight, path_effects=halo(), zorder=40)


def spread(values, min_gap, lo=None, hi=None):
    """Label positions close to `values` but at least `min_gap` apart (1-D declutter)."""
    v = np.asarray(values, float)
    order = np.argsort(v)
    pos = v[order].copy()
    for _ in range(300):
        moved = False
        for i in range(1, len(pos)):
            gap = pos[i] - pos[i - 1]
            if gap < min_gap:
                s = (min_gap - gap) / 2
                pos[i - 1] -= s; pos[i] += s; moved = True
        if lo is not None and pos[0] < lo:
            pos += lo - pos[0]
        if hi is not None and pos[-1] > hi:
            pos -= pos[-1] - hi
        if not moved:
            break
    out = np.empty_like(pos); out[order] = pos
    return out


def kde(x, grid, bw=None):
    """Gaussian KDE with a robust Silverman bandwidth (min of SD and IQR/1.34) — heavy tails do not oversmooth."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if bw is None:
        s = min(x.std(), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) or x.std()
        bw = 0.9 * s * len(x) ** (-0.2)
    return np.exp(-0.5 * ((np.asarray(grid)[:, None] - x[None]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def text_overlaps(fig):
    """Number of overlapping pairs among visible text artists (layout self-check)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_alpha() != 0:
            try:
                bb = t.get_window_extent(r)
            except Exception:
                continue
            if bb.width > 1 and bb.height > 1:
                boxes.append((bb, t.get_text()))
    n, pairs = 0, []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][0], boxes[j][0]
            if a.overlaps(b):
                ix = min(a.x1, b.x1) - max(a.x0, b.x0); iy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ix > 1.5 and iy > 1.5:
                    n += 1; pairs.append((boxes[i][1][:18], boxes[j][1][:18]))
    return n, pairs

def layout_overlaps(fig):
    """Check axis-label/tick vs neighboring axes and text vs legends/colorbars."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [a for a in fig.axes if a.get_visible()]

    def overlap(a, b):
        if not a.overlaps(b):
            return False
        return (min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and
                min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5)

    def contains(a, b):
        return a.x0 <= b.x0 and a.y0 <= b.y0 and a.x1 >= b.x1 and a.y1 >= b.y1

    axis_hits = []
    for ax in axes:
        if hasattr(ax, "_colorbar"):
            continue
        if not ax.axison:
            continue
        items = [ax.xaxis.label, ax.yaxis.label]
        items += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in items:
            if not (t.get_visible() and t.get_text().strip()):
                continue
            bb = t.get_window_extent(renderer)
            for other in axes:
                if other is ax or contains(ax.bbox, other.bbox) or contains(other.bbox, ax.bbox):
                    continue
                if overlap(bb, other.bbox):
                    axis_hits.append((t.get_text()[:24], ax.get_label(), other.get_label()))

    legends = []
    legend_texts = set()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            legends.append((leg.get_window_extent(renderer), ax))
            legend_texts.update(leg.get_texts())

    colorbars = []
    for ax in axes:
        if ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar"):
            colorbars.append((ax.bbox, ax))

    asset_hits = []
    for t in fig.findobj(mpl.text.Text):
        if not (t.get_visible() and t.get_text().strip() and t.get_alpha() != 0):
            continue
        if t in legend_texts:
            continue
        owner = getattr(t, "axes", None)
        bb = t.get_window_extent(renderer)
        for box, owner_ax in legends:
            if owner is owner_ax:
                continue
            if overlap(bb, box):
                asset_hits.append((t.get_text()[:24], "legend"))
        for box, cax in colorbars:
            if owner is cax:
                continue
            if overlap(bb, box):
                asset_hits.append((t.get_text()[:24], "colorbar"))
    return {"axis_adjacent": axis_hits, "legend_colorbar": asset_hits}


def nudge_labels_off_markers(fig):
    """Move text annotations a small screen-space distance when scatter points obscure them."""
    from matplotlib.collections import PathCollection
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        points = []
        for col in ax.collections:
            if isinstance(col, PathCollection) and col.get_visible() and len(col.get_offsets()):
                xy = col.get_offset_transform().transform(col.get_offsets())
                sizes = col.get_sizes()
                rad = np.sqrt(sizes if len(sizes) == len(xy) else np.repeat(sizes[:1], len(xy))) * fig.dpi / 144
                points.append((xy, rad))
        if not points:
            continue
        for t in ax.texts:
            if not (t.get_visible() and t.get_text().strip()):
                continue
            origin = np.asarray(t.get_transform().transform(t.get_position()), float)
            original = t.get_position()
            for dx, dy in [(0, 0), (0, 18), (0, -18), (22, 0), (-22, 0),
                           (32, 20), (-32, 20), (32, -20), (-32, -20),
                           (0, 42), (42, 0), (-42, 0), (0, -42),
                           (65, 35), (-65, 35), (65, -35), (-65, -35)]:
                t.set_position(t.get_transform().inverted().transform(origin + [dx, dy]))
                box = t.get_window_extent(r)
                if (box.x0 < fig.bbox.x0 or box.y0 < fig.bbox.y0 or
                        box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1):
                    continue
                if any(np.any(((xy[:, 0] + rad > box.x0) & (xy[:, 0] - rad < box.x1) &
                               (xy[:, 1] + rad > box.y0) & (xy[:, 1] - rad < box.y1))) for xy, rad in points):
                    continue
                break
            else:
                t.set_position(original)
def g5_text_bounds_and_markers(fig):
    """Check every visible Text against the canvas, and annotations against scatter markers."""
    from matplotlib.collections import PathCollection
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside, markers = [], []
    legend_texts = {t for ax in fig.axes for leg in [ax.get_legend()] if leg is not None for t in leg.get_texts()}
    rendered = set(fig.texts)
    for ax in fig.axes:
        rendered.update(ax.texts)
        rendered.update(legend_texts)
        if ax.axison:
            rendered.update([ax.title, ax.xaxis.label, ax.yaxis.label])
            rendered.update(ax.get_xticklabels())
            rendered.update(ax.get_yticklabels())
    for t in rendered:
        if not (t.get_visible() and t.get_text().strip() and t.get_alpha() != 0):
            continue
        if t.get_text().lstrip().startswith("("):
            continue
        box = t.get_window_extent(renderer)
        if box.width <= 1 or box.height <= 1:
            continue
        if (box.x0 < canvas.x0 - 0.5 or box.y0 < canvas.y0 - 0.5 or
                box.x1 > canvas.x1 + 0.5 or box.y1 > canvas.y1 + 0.5):
            outside.append(t.get_text()[:32])
        ax = t.axes
        if ax is None or t in legend_texts or t not in ax.texts:
            continue
        # Scattered markers only; continuous curves and area fills are not markers.
        for col in ax.collections:
            if not isinstance(col, PathCollection) or not col.get_visible():
                continue
            offsets = col.get_offsets()
            if len(offsets) == 0:
                continue
            pts = col.get_offset_transform().transform(offsets)
            sizes = col.get_sizes()
            radii = np.sqrt(sizes if len(sizes) == len(pts) else np.repeat(sizes[:1], len(pts))) * fig.dpi / 144
            inside = ((pts[:, 0] + radii > box.x0) & (pts[:, 0] - radii < box.x1) &
                      (pts[:, 1] + radii > box.y0) & (pts[:, 1] - radii < box.y1))
            if np.any(inside):
                markers.append((t.get_text()[:32], int(inside.sum())))
                break
    return {"outside": outside, "text_marker": markers}


def check_occlusion(ax, artists, xy):
    """Count basin centers in the union bounding rectangle of map decorations.

    xy is in map data coordinates. Counts use drawn centers, not polygon areas.
    """
    from matplotlib.transforms import Bbox
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = []
    for artist in artists:
        if artist is None or not artist.get_visible():
            continue
        box = artist.get_tightbbox(renderer) if isinstance(artist, mpl.axes.Axes) else artist.get_window_extent(renderer)
        if box is not None and box.width > 0 and box.height > 0:
            boxes.append(box)
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not boxes or not len(points):
        return 0
    box = Bbox.union(boxes)
    screen = ax.transData.transform(points)
    return int(np.sum((screen[:, 0] >= box.x0) & (screen[:, 0] <= box.x1) &
                      (screen[:, 1] >= box.y0) & (screen[:, 1] <= box.y1)))


def check_text(fig):
    """Audit visible text bounds, text pairs, and annotation/data intersections.

    Lines use screen-space segment/rectangle intersection; scatter uses centers;
    bars use their screen-space rectangles. Filled ribbons are not data lines.
    """
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, overlaps, data_hits = [], [], []
    texts = []
    rendered = set(fig.texts)
    for ax in fig.findobj(mpl.axes.Axes):
        rendered.update(ax.texts)
        rendered.update([ax.title, ax._left_title, ax._right_title])
        if ax.axison:
            rendered.update([ax.xaxis.label, ax.yaxis.label])
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks():
                    if lo - 1e-10 <= tick.get_loc() <= hi + 1e-10:
                        rendered.update([tick.label1, tick.label2])
        if ax.get_legend() is not None:
            rendered.update(ax.get_legend().get_texts())
    for legend in fig.legends:
        rendered.update(legend.get_texts())
    for text in sorted(rendered, key=lambda t: (t.get_text(), id(t))):
        if not text.get_visible() or not text.get_text().strip() or text.get_alpha() == 0:
            continue
        box = text.get_window_extent(renderer)
        if box.width <= 1 or box.height <= 1:
            continue
        texts.append((text, box))
        if box.x0 < fig.bbox.x0 - .5 or box.y0 < fig.bbox.y0 - .5 or box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5:
            outside.append(text.get_text())
    for i, (text, box) in enumerate(texts):
        for other, otherbox in texts[i + 1:]:
            if min(box.x1, otherbox.x1) - max(box.x0, otherbox.x0) > 1.5 and min(box.y1, otherbox.y1) - max(box.y0, otherbox.y0) > 1.5:
                overlaps.append([text.get_text(), other.get_text()])
        ax = text.axes
        if ax is None or text not in ax.texts:
            continue
        kinds = []
        for line in ax.lines:
            if not line.get_visible() or not len(line.get_xdata()):
                continue
            path = line.get_transform().transform_path(line.get_path())
            if path.intersects_bbox(box, filled=False):
                kinds.append("line")
                break
        for collection in ax.collections:
            if isinstance(collection, PathCollection) and collection.get_visible():
                points = collection.get_offset_transform().transform(collection.get_offsets())
                if np.any((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1) & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)):
                    kinds.append("point")
                    break
        for patch in [p for container in ax.containers if isinstance(container, mpl.container.BarContainer) for p in container.patches]:
            if isinstance(patch, mpl.patches.Rectangle) and patch.get_visible() and patch.get_window_extent(renderer).overlaps(box):
                kinds.append("bar")
                break
        if kinds:
            data_hits.append({"text": text.get_text(), "kinds": kinds})
    return {"outside": outside, "overlaps": overlaps, "data_hits": data_hits}


def audit_maps(fig):
    """Check all map decorations; extend south by <=15%, then move outside."""
    from matplotlib.collections import PathCollection, PolyCollection
    rows = []
    for ax in fig.axes:
        if not getattr(ax, "_is_basin_map", False):
            continue
        centers = []
        for col in ax.collections:
            if isinstance(col, PathCollection) and len(col.get_offsets()):
                centers.extend(np.asarray(col.get_offsets()).tolist())
            elif isinstance(col, PolyCollection) and len(col.get_offsets()) > 1:
                centers.extend(np.asarray(col.get_offsets()).tolist())
        xy = np.unique(np.asarray(centers, float).reshape(-1, 2), axis=0) if centers else np.empty((0, 2))
        artists = list(getattr(ax, "_map_decorations", []))
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            artists.append(leg)
        before = check_occlusion(ax, artists, xy)
        if before:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - .15 * (hi - lo), hi)
        if check_occlusion(ax, artists, xy):
            for cax in getattr(ax, "_map_decorations", []):
                x, y, w, h = cax._map_location
                cax.set_axes_locator(mpl.axes._base._TransformedBoundsLocator((x, -.09, w, h), ax.transAxes))
            if leg is not None:
                leg.set_bbox_to_anchor((.5, -.20))
        rows.append({"map": len(rows) + 1, "centers": len(xy), "occluded": check_occlusion(ax, artists, xy), "before": before})
    return rows

def keep_annotations_in_canvas(fig):
    """Translate annotations (not ticks or data) just enough to fit the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            box = text.get_window_extent(renderer)
            dx = max(0, fig.bbox.x0 + 2 - box.x0) - max(0, box.x1 - fig.bbox.x1 + 2)
            dy = max(0, fig.bbox.y0 + 2 - box.y0) - max(0, box.y1 - fig.bbox.y1 + 2)
            if dx or dy:
                position = text.get_transform().transform(text.get_position())
                text.set_position(text.get_transform().inverted().transform(position + [dx, dy]))

def save(fig, stem, outdir=None, check=True, png_only=False):
    outdir = outdir or os.environ.get("TN_PLOTS_DIR", os.getcwd())
    os.makedirs(outdir, exist_ok=True)
    maps = audit_maps(fig)
    keep_annotations_in_canvas(fig)
    audit = {"figure": os.environ.get("TN_FIGURE_NAME", stem), "maps": maps, "text": check_text(fig)}
    audit_path = os.environ.get("TN_AUDIT_LOG")
    if audit_path:
        with open(audit_path, "a") as stream:
            stream.write(json.dumps(audit) + "\n")
    print("FINAL_AUDIT", json.dumps(audit))
    if check:
        n, pairs = text_overlaps(fig)
        g5 = layout_overlaps(fig)
        g5_n = len(g5["axis_adjacent"]) + len(g5["legend_colorbar"])
        print(f"{stem}: text overlaps = {n}" + (f"  {pairs[:6]}" if n else ""))
        print(f"{stem}: G5 axis/asset overlaps = {g5_n}" + (f"  {g5}" if g5_n else ""))
        g5_extra = g5_text_bounds_and_markers(fig)
        print(f"{stem}: G5 canvas/marker overlaps = "
              f"{len(g5_extra['outside'])}/{len(g5_extra['text_marker'])}" +
              (f"  {g5_extra}" if any(g5_extra.values()) else ""))
    name = os.environ.get("TN_FIGURE_NAME", stem)
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, dpi=300)
    if not png_only:
        pdf_path = os.path.join(outdir, name + ".pdf")
        fig.savefig(pdf_path)
        print("saved", pdf_path)
    print("saved", path, "| font:", FONT, "| size: %.2f x %.2f in" % tuple(fig.get_size_inches()))
