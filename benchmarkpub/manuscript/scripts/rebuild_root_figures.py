#!/usr/bin/env python3
"""Stage and rebuild manuscript final Figures 1–4 using temporary staging paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
import sys

import pandas as pd
import numpy as np

MANUSCRIPT = Path(__file__).resolve().parents[1]
V12 = Path(tempfile.mkdtemp(prefix="hess_fig_root_"))
WORK = V12 / "work"
OUTPUT = MANUSCRIPT / "figures"
TABLES = MANUSCRIPT / "tables" / "figures"
SCRIPTS = MANUSCRIPT / "scripts"
COORD_PATH = TABLES / "gauge_coordinates_531.csv"
COORDS = pd.read_csv(COORD_PATH, dtype={"gauge_id": str}).set_index("gauge_id")
ASSETS = MANUSCRIPT / "cache" / "assets"

# The cache inputs remain untouched.  Each mapped table is copied into v12/work
# and only its missing lon/lat values are filled by exact gauge_id lookup.
MAP_TABLES = {
    "fig02": "penalty_basin.csv",
    "fig03": "basin_cover_status_oob.csv",
    "fig04": "basin_transfer.csv",
}
RENDER = [
    ("fig01", "make_fig01.py", "fig1_library_size"),
    ("fig02", "make_fig02.py", "fig2_transfer_penalty"),
    ("fig03", "make_fig03.py", "fig3_library_members"),
    ("fig04", "make_fig04.py", "fig4_identity_transfer"),
]


def prepare_inputs() -> list[dict]:
    audits = []
    for fig in ("fig01", "fig02", "fig03", "fig04"):
        src = MANUSCRIPT / "cache" / fig / "data"
        dst = WORK / fig / "data"
        if not src.is_dir():
            raise FileNotFoundError(src)
        if dst.parent.exists():
            shutil.rmtree(dst.parent)
        shutil.copytree(src, dst)

    if len(COORDS) != 531 or not COORDS.index.is_unique or COORDS[["lon", "lat"]].isna().any().any():
        raise ValueError("v12 coordinate table must have 531 unique, complete gauge IDs")
    central = COORDS.reset_index()
    central_ids = set(central.gauge_id)

    # Fig. 1 and Fig. 8 coordinate-only map tables omit 16 IDs entirely.
    # Use the complete coordinate table after verifying existing values are unchanged.
    p = WORK / "fig01" / "data" / "basins.csv"
    old = pd.read_csv(p, dtype={"gauge_id": str})
    old["gauge_id"] = old.gauge_id.str.zfill(8)
    common = old.merge(central, on="gauge_id", suffixes=("_old", "_new"), validate="one_to_one")
    for col in ("lon", "lat"):
        if not np.allclose(common[f"{col}_old"], common[f"{col}_new"], rtol=0, atol=1e-8):
            raise ValueError(f"Fig. 1 existing {col} values differ from the completed coordinate table")
    added = sorted(central_ids - set(old.gauge_id))
    if len(added) != 16:
        raise ValueError(f"Fig. 1 expected 16 coordinate-only rows to add, found {len(added)}")
    central.to_csv(p, index=False, float_format="%.5f")
    audits.append({"figure": "fig01", "input_table": str(p.relative_to(V12)), "table_rows": len(central),
                   "unique_gauge_ids": len(central_ids), "rows_with_missing_coordinates_before": 0,
                   "unique_ids_completed": 0, "unique_ids_added": len(added),
                   "remaining_missing_coordinates": 0, "completed_ids": "|".join(added),
                   "unmatched_type_ids": ""})

    # Fig. 2, 3, and 4 have all basin records, with 16 blank coordinate pairs.
    for fig, filename in MAP_TABLES.items():
        path = WORK / fig / "data" / filename
        frame = pd.read_csv(path, dtype={"gauge_id": str})
        if "gauge_id" not in frame or not {"lon", "lat"}.issubset(frame.columns):
            raise ValueError(f"Expected gauge_id/lon/lat in {path}")
        frame["gauge_id"] = frame.gauge_id.str.zfill(8)
        n_ids = frame.gauge_id.nunique()
        if n_ids != 531:
            raise ValueError(f"{path} covers {n_ids}, not 531, unique gauge IDs")
        missing_before = frame[["lon", "lat"]].isna().any(axis=1)
        filled_ids = sorted(frame.loc[missing_before, "gauge_id"].unique())
        for col in ("lon", "lat"):
            known = frame[col].notna()
            expected = frame.loc[known, "gauge_id"].map(COORDS[col]).to_numpy(float)
            if not np.allclose(frame.loc[known, col].to_numpy(float), expected, rtol=0, atol=1e-8):
                raise ValueError(f"Existing {col} values in {path} disagree with the coordinate source")
            missing = frame[col].isna()
            frame.loc[missing, col] = frame.loc[missing, "gauge_id"].map(COORDS[col])
        if frame[["lon", "lat"]].isna().any().any():
            missing_ids = sorted(frame.loc[frame[["lon", "lat"]].isna().any(axis=1), "gauge_id"].unique())
            raise ValueError(f"Unresolved coordinates in {path}: {missing_ids}")
        frame.to_csv(path, index=False)
        audits.append({"figure": fig, "input_table": str(path.relative_to(V12)), "table_rows": len(frame),
                       "unique_gauge_ids": n_ids, "rows_with_missing_coordinates_before": int(missing_before.sum()),
                       "unique_ids_completed": len(filled_ids), "unique_ids_added": 0,
                       "remaining_missing_coordinates": 0, "completed_ids": "|".join(filled_ids),
                       "unmatched_type_ids": ""})

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(TABLES / "coordinate_completion_by_map.csv", index=False)
    return audits


def render_all() -> list[dict]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import hydrofig as hf

    output_rows: list[dict] = []
    old_overlap = hf.text_overlaps
    current_name = {"value": ""}

    def recorded_overlap(fig_obj):
        n, pairs = old_overlap(fig_obj)
        output_rows.append({
            "figure": current_name["value"] + ".png",
            "overlap_count": int(n),
            "overlap_pairs": " | ".join(f"{a} <> {b}" for a, b in pairs),
        })
        return n, pairs

    hf.text_overlaps = recorded_overlap
    old_cwd = Path.cwd()
    old_output = os.environ.get("TN_PLOTS_DIR")
    old_assets = os.environ.get("TN_ASSET_DIR")
    old_name = os.environ.get("TN_FIGURE_NAME")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["TN_PLOTS_DIR"] = str(OUTPUT)
    os.environ["TN_ASSET_DIR"] = str(ASSETS)
    try:
        for fig, script_name, filename in RENDER:
            current_name["value"] = filename
            os.environ["TN_FIGURE_NAME"] = filename
            os.chdir(WORK / fig)
            runpy.run_path(str(SCRIPTS / script_name), run_name="__main__")
    finally:
        os.chdir(old_cwd)
        hf.text_overlaps = old_overlap
        for key, old in (("TN_PLOTS_DIR", old_output), ("TN_ASSET_DIR", old_assets), ("TN_FIGURE_NAME", old_name)):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    expected = [f"{name}.png" for _, _, name in RENDER]
    actual = [row["figure"] for row in output_rows]
    if actual != expected:
        raise RuntimeError(f"Text-overlap checks did not match rendered figures: {actual}")
    pd.DataFrame(output_rows).to_csv(TABLES / "root_figures_text_overlap_v12.csv", index=False)
    return output_rows


def main() -> None:
    coords = prepare_inputs()
    overlaps = render_all()
    manifest = {
        "coordinate_table": str(COORD_PATH.relative_to(MANUSCRIPT)),
        "coordinate_audit": str((TABLES / "coordinate_completion_by_map.csv").relative_to(MANUSCRIPT)),
        "figure_overlap_audit": str((TABLES / "root_figures_text_overlap_v12.csv").relative_to(MANUSCRIPT)),
        "figures": [name for _, _, name in RENDER],
        "outputs": [str(OUTPUT / f"{name}.{ext}") for _, _, name in RENDER for ext in ("png", "pdf")],
    }
    (TABLES / "root_figures_v12_run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Map coordinate completion:")
    print(pd.DataFrame(coords)[["figure", "table_rows", "unique_gauge_ids", "rows_with_missing_coordinates_before", "remaining_missing_coordinates"]].to_string(index=False))
    print("\nText-overlap checks:")
    print(pd.DataFrame(overlaps).to_string(index=False))
    print("\nRoot figure outputs written to", OUTPUT)


if __name__ == "__main__":
    main()
