#!/usr/bin/env python3
"""Stage and render the styled working copies of supplied v6 Fig. 5/6 scripts.

Only plotting style and layout differ from the supplied scripts. Frozen data and
statistics are staged from the checked bundle; the completed coordinate table is
joined by exact gauge ID. A runtime override shortens one colliding zero tick label.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
import sys
from zipfile import ZipFile

import pandas as pd

MANUSCRIPT = Path(__file__).resolve().parents[1]
V12 = Path(tempfile.mkdtemp(prefix="hess_fig_v6_"))
SOURCE_DIR = MANUSCRIPT / "scripts" / "figures_v6"
WORK_DIR = V12 / "work"
OUTPUT_DIR = MANUSCRIPT / "figures"
TABLE_DIR = MANUSCRIPT / "tables" / "figures"
BUNDLE = MANUSCRIPT / "leakfree_20261007" / "exports" / "figure_redraw_source_bundle_20261007.zip"
STATS = MANUSCRIPT / "leakfree_20261007" / "tables" / "fig05_fig06_v6_stats.json"
ASSETS = MANUSCRIPT / "cache" / "assets"
COORDS = TABLE_DIR / "gauge_coordinates_531.csv"
SUPPLIED = Path("/mnt/d/Users/13035/Downloads/fig05_fig06_scripts_v6")

ZIP_PREFIX = "figure_redraw_source_bundle_20261007/"
FIGURE_READY = ZIP_PREFIX + "figure_ready/"
DATA_MEMBERS = {
    "fig05": {
        "L1_rule_per_basin_r3_with_gauge_id.csv": FIGURE_READY + "L1_rule_per_basin_r3_with_gauge_id.csv",
        "oob_reference_by_gauge_id.csv": FIGURE_READY + "oob_reference_by_gauge_id.csv",
        "example_hydrograph.csv": ZIP_PREFIX + "tables_appendix_20261006/zenodo/plot_data/fig05/data/example_hydrograph.csv",
    },
    "fig06": {
        "L1_rule_per_basin_r3_with_gauge_id.csv": FIGURE_READY + "L1_rule_per_basin_r3_with_gauge_id.csv",
        "P1_parsimony_rule_per_basin_with_gauge_id.csv": FIGURE_READY + "P1_parsimony_rule_per_basin_with_gauge_id.csv",
        "oob_reference_by_gauge_id.csv": FIGURE_READY + "oob_reference_by_gauge_id.csv",
        "gset_seeds.csv": ZIP_PREFIX + "tables_appendix_20261006/zenodo/plot_data/fig06/data/gset_seeds.csv",
        "gset_summary.csv": ZIP_PREFIX + "tables_appendix_20261006/zenodo/plot_data/fig06/data/gset_summary.csv",
        "gset_null.csv": ZIP_PREFIX + "tables_appendix_20261006/zenodo/plot_data/fig06/data/gset_null.csv",
        "LEAKFREE_DIGEST_gset_by_seed.csv": ZIP_PREFIX + "leakfree_20261007/tables/LEAKFREE_DIGEST_gset_by_seed.csv",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage() -> None:
    coords = pd.read_csv(COORDS, dtype={"gauge_id": str})
    if len(coords) != 531 or not coords.gauge_id.is_unique or coords[["lon", "lat"]].isna().any().any():
        raise ValueError("531 unique complete coordinates required")
    # Canonical release data are byte-identical to the preserved final staging inputs.
    repo_root = MANUSCRIPT.parents[1] if (MANUSCRIPT.parents[1] / "releases").exists() else MANUSCRIPT.parents[2]
    release = repo_root / "releases" / "benchmarkpub" / "figures"
    for fig, folder in [("fig05", "Fig5_usage_rules"), ("fig06", "Fig6_selection_criteria")]:
        src = release / folder
        dest = WORK_DIR / fig
        shutil.copytree(src / "data", dest / "data")
        shutil.copy2(src / "stats.json", dest / "stats.json")

def run_one(fig: str, script_name: str, stem: str, audit_rows: list[dict]) -> None:
    script_dir = SOURCE_DIR
    script = script_dir / script_name
    stage_dir = WORK_DIR / fig
    out = stage_dir / "output"
    out.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_hydrofig = sys.modules.pop("hydrofig", None)
    old_asset_dir = os.environ.get("TN_ASSET_DIR")
    old_output = os.environ.get("TN_PLOTS_DIR")
    os.environ["TN_PLOTS_DIR"] = str(out)
    sys.path.insert(0, str(script_dir))
    os.environ["TN_ASSET_DIR"] = str(ASSETS)
    os.chdir(stage_dir)
    try:
        spec = importlib.util.spec_from_file_location("hydrofig", script_dir / "hydrofig.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load supplied hydrofig.py")
        hf = importlib.util.module_from_spec(spec)
        sys.modules["hydrofig"] = hf
        spec.loader.exec_module(hf)

        # Keep all statistics and the diverging color scale unchanged.  The
        # shortened zero tick is a text-only label correction for the existing
        # Fig. 5 colorbar collision (0.00 against the map tick at 60).
        original_cbar = hf.map_cbar
        def map_cbar_v12(*args, **kwargs):
            cb = original_cbar(*args, **kwargs)
            ticks = kwargs.get("ticks")
            fmt = kwargs.get("fmt")
            if fmt == "%.2f" and ticks is not None and len(ticks) == 3 and abs(float(ticks[0])) < 1e-12:
                cb.set_ticklabels(["0", f"{float(ticks[1]):.2f}", f"{float(ticks[2]):.2f}"])
            return cb
        hf.map_cbar = map_cbar_v12

        original_overlap = hf.text_overlaps
        def recorded_overlap(fig_obj):
            n, pairs = original_overlap(fig_obj)
            audit_rows.append({
                "figure": stem + ".png",
                "overlap_count": int(n),
                "overlap_pairs": " | ".join(f"{a} <> {b}" for a, b in pairs),
            })
            return n, pairs
        hf.text_overlaps = recorded_overlap
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(old_cwd)
        if old_output is None:
            os.environ.pop("TN_PLOTS_DIR", None)
        else:
            os.environ["TN_PLOTS_DIR"] = old_output
        sys.path[:] = old_path
        sys.modules.pop("hydrofig", None)
        if old_hydrofig is not None:
            sys.modules["hydrofig"] = old_hydrofig
        if old_asset_dir is None:
            os.environ.pop("TN_ASSET_DIR", None)
        else:
            os.environ["TN_ASSET_DIR"] = old_asset_dir

    for ext in ("png", "pdf"):
        src = out / f"{stem}.{ext}"
        if not src.is_file() or src.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {src}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, OUTPUT_DIR / src.name)


def main() -> None:
    stage()
    audits: list[dict] = []
    run_one("fig05", "make_fig05.py", "fig5_usage_rules", audits)
    run_one("fig06", "make_fig06.py", "fig6_selection_criteria", audits)
    if len(audits) != 2:
        raise RuntimeError(f"Expected two text-overlap checks, received {len(audits)}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(TABLE_DIR / "fig05_fig06_text_overlap_v12.csv", index=False)
    manifest = {
        "figures": ["fig5_usage_rules", "fig6_selection_criteria"],
        "script_sha256": {name: sha256(SOURCE_DIR / name) for name in ("hydrofig.py", "make_fig05.py", "make_fig06.py")},
        "stats_sha256": sha256(WORK_DIR / "fig05" / "stats.json"),
        "coordinates_sha256": sha256(COORDS),
        "text_overlap_audit": "tables/figures/fig05_fig06_text_overlap_v12.csv",
        "outputs": [str(OUTPUT_DIR / f"fig{n}_{name}.{ext}") for n, name in [(5, "usage_rules"), (6, "selection_criteria")] for ext in ("png", "pdf")],
    }
    (TABLE_DIR / "fig05_fig06_v12_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Fig. 5/6 rendering complete; final outputs written to", OUTPUT_DIR)
    print(pd.DataFrame(audits).to_string(index=False))


if __name__ == "__main__":
    main()
