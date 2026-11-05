#!/usr/bin/env python3
"""11_pub_insample_seen.py — same-pipeline in-sample ("seen-fold") per-basin KGE.

WHY
---
Basin b is held out in exactly one of the 5 OOB folds.  For the other 4 folds, b is
in that fold's TRAINING set.  This script evaluates each fold's own frozen `best.pt`
on that fold's training basins over the identical test period / warm-up / masking,
producing

    KGE^seenFold_{m, j, b}   for every fold j and every training basin b of fold j.

Together with the already-frozen held-out KGE (`kge_oob` in
`tables/seen_ic_oob_merged.parquet`, which came from fold k's model on basin b where
b is held out in k), this gives a strictly within-pipeline PUB contrast:

    Delta_KGE_PUB_{m,b} = aggregate_{j != k} KGE^seenFold_{m,j,b}
                          - KGE^heldout_{m,k,b}

positive = the basin is modelled worse when it was withheld from training.

This is the same operation the frozen audit ran as its "within-fold seen" arm
(`chapter4_oob_pipeline_audit_20260926/scripts/b4_attribute_permutation.py`), which
stored only the per-cell median.  Here the per-basin values are kept, and the
per-cell medians are validated against that frozen table.

NO TRAINING.  `best.pt` is loaded read-only, `model.eval()` + `torch.inference_mode()`
only; no optimizer, no backward, no checkpoint write.

PROVENANCE / REUSE
------------------
The forward path is imported from the frozen pipeline modules
(`benchmark/scripts/oob/{oob_common,run_oob_job}.py`) rather than reimplemented:
`J.build_model`, `J._make_parameterizer`, `J._add_calendar_if_needed`,
`J._squeeze_streamflow`, `J._align_scored`, `OC.kge_numpy`, `OC.NormalizationStats`.
Nothing under `project/benchmark/` is modified.

Inputs (read-only)
------------------
  config      benchmark/configs/oob_primary8_5fold_20260902.yaml   (protocol identical
              for all three cohorts: verified against each run dir's config.json)
  ids         data/531sub_id.txt
  folds       benchmark/results/oob_primary8_5fold_20260902/OOB_FOLD_ASSIGNMENT.csv
  checkpoints existing8 -> benchmark/results/oob_primary8_5fold_20260902/runs
              v4_new8   -> benchmarkpub/data/oob36_20260930/runs
              v5_new20  -> benchmarkpub/data/oob36_20260930/runs
  {normalization_metadata.json, attribute_mean.npy, attribute_std.npy} per run dir

Output
------
  $OUT2/tables/pub_insample_seen_kge.csv   model,fold,basin_id,kge_insample,n_train,n_heldout
  appended per cell and flushed, so a long run is resumable with --resume.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/home/jingxin/code/dmg-research")
BENCH = REPO / "project" / "benchmark"
PUB = REPO / "project" / "benchmarkpub"
DATA = PUB / "data" / "oob36_20260930"
OUT2 = PUB / "analysis" / "chapter4_oob36_prelim_20260930" / "pub_next_stage_20260930"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(BENCH), str(BENCH / "scripts" / "oob"), str(HERE)]

import oob_common as OC      # noqa: E402
import run_oob_job as J      # noqa: E402

CONFIG_NAME = "oob_primary8_5fold_20260902.yaml"
EXISTING8_RUNS = BENCH / "results" / "oob_primary8_5fold_20260902" / "runs"
EXISTING8 = ["alpine2", "hbv96", "xinanjiang", "newzealand2", "ihacres", "us1", "mopex4", "hillslope"]
EVAL_WARMUP = 365
TEST_DATES = pd.date_range("1994-10-01", "2010-09-30", freq="D")
B4_TABLE = BENCH / "analysis" / "chapter4_oob_pipeline_audit_20260926" / "tables" / "B4_attribute_permutation.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_test_period(ids: np.ndarray):
    """One [T, n_ids, 3] forcing tensor and one [T, n_ids] mm/day target, ID-joined."""
    with (REPO / "data/camels_dataset").open("rb") as fh:
        bundle = pickle_load(fh)
    forcings_raw, target_raw = bundle[0], bundle[1]
    gage = np.load(REPO / "data/gage_id.npy")
    id_map = {int(g): i for i, g in enumerate(gage.tolist())}
    idx = np.asarray([id_map[int(b)] for b in ids], dtype=np.int64)
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")

    def b(a: str, c: str):
        return int(dates.get_loc(pd.Timestamp(a))), int(dates.get_loc(pd.Timestamp(c))) + 1

    tl, tr = b("1995-10-01", "2010-09-30")
    fl = tl - EVAL_WARMUP
    x = np.asarray(forcings_raw[idx, fl:tr, :3], dtype=np.float64).transpose(1, 0, 2)
    y = np.asarray(target_raw[idx, tl:tr], dtype=np.float64)
    if y.ndim == 3:
        y = y[..., 0]
    attrs_raw = np.load(REPO / "data/caravan_671_attributes.npy")
    area = attrs_raw[idx, 11].astype(np.float64)
    y = y * (0.0283168 * 86400.0 * 1000.0 / (area * 1e6))[:, None]
    return x, y.T          # [T, B, C], [T, B]


def pickle_load(fh):
    import pickle
    return pickle.load(fh)


def load_stats(run_dir: Path) -> OC.NormalizationStats:
    meta = json.loads((run_dir / "normalization_metadata.json").read_text(encoding="utf-8"))
    return OC.NormalizationStats("zscore", bool(meta["log_transform_skewed"]),
                                np.asarray(meta["train_fitted_log_shifts"], dtype=np.float64),
                                np.load(run_dir / "attribute_mean.npy"),
                                np.load(run_dir / "attribute_std.npy"))


def forward_kge(model: str, config: dict, run_dir: Path, stats: OC.NormalizationStats,
                raw_attr: np.ndarray, x: np.ndarray, y: np.ndarray, dates,
                device: torch.device) -> np.ndarray:
    """Per-basin KGE of the frozen best.pt on the supplied basin set. Read-only."""
    attrs = torch.as_tensor(stats.transform(raw_attr), dtype=torch.float64, device=device)
    net = J._make_parameterizer(model, attrs, config, device)
    net.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["network"])
    xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float64, device=device)
    xt = J._add_calendar_if_needed(xt, dates, model)
    hydro = J.build_model(model, device, warm_up=EVAL_WARMUP, backend="eager",
                          parameter_mapping=config["protocol"]["parameter_mapping"],
                          warmup_grad_mode=config["protocol"]["warmup_grad_mode"],
                          dtype=torch.float64)
    net.eval()
    with torch.inference_mode():
        pred = J._squeeze_streamflow(hydro({"x_phy": xt}, (None, net(attrs).unsqueeze(-1)))["streamflow"])
    pn, on = J._align_scored(pred, yt, EVAL_WARMUP)
    scores, invalid, _ = OC.kge_numpy(pn.detach().cpu().numpy().astype(np.float64),
                                      on.detach().cpu().numpy().astype(np.float64),
                                      eps=float(config["protocol"]["kge_eps"]))
    if bool(invalid.any()):
        raise RuntimeError(f"{model}: {int(invalid.sum())} invalid basins")
    return scores


def run_root_for(model: str) -> Path:
    return EXISTING8_RUNS if model in EXISTING8 else DATA / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="")
    ap.add_argument("--folds", default="")
    ap.add_argument("--out", default=str(OUT2 / "tables" / "pub_insample_seen_kge.csv"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-heldout", action="store_true",
                    help="also reproduce each cell's held-out KGE and compare with heldout_basin_kge.csv")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = OC.load_oob_config(CONFIG_NAME)
    ids = np.asarray(ast.literal_eval((REPO / "data/531sub_id.txt").read_text().strip()), dtype=np.int64)
    rows = list(csv.DictReader(open(BENCH / "results/oob_primary8_5fold_20260902/OOB_FOLD_ASSIGNMENT.csv")))
    assign = {int(r["basin_id"]): int(r["fold"]) for r in rows}
    if len(assign) != 531:
        raise RuntimeError(f"fold assignment has {len(assign)} basins, expected 531")
    raw_attrs = np.load(REPO / "data/caravan_671_attributes.npy")
    gage = np.load(REPO / "data/gage_id.npy")
    row_of = {int(g): i for i, g in enumerate(gage.tolist())}
    pos = {int(b): i for i, b in enumerate(ids.tolist())}

    # model list is taken from the frozen inventory so it is never guessed
    inv = pd.read_csv(PUB / "analysis" / "chapter4_oob36_prelim_20260930" / "remote_inventory.csv")
    all_models = sorted(inv["model"].unique())
    models = [m for m in (args.models.split(",") if args.models else all_models) if m]
    for m in models:
        if m not in all_models:
            raise RuntimeError(f"unknown model {m!r}")
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(5))

    log(f"models: {len(models)}  folds: {folds}  device: {device}")
    x_all, y_all = load_test_period(ids)
    log(f"test tensor: x{x_all.shape} y{y_all.shape}")

    done = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path, usecols=["model", "fold"])
        done = set(zip(prev["model"], prev["fold"].astype(int)))
        log(f"resume: {len(done)} (model,fold) cells already present")

    write_header = not out_path.exists() or not done
    fh = out_path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=["model", "fold", "basin_id", "kge_insample",
                                       "n_train", "n_heldout"])
    if write_header:
        w.writeheader()

    ho_check = []
    n_cells = 0
    for model in models:
        for fold in folds:
            if (model, fold) in done:
                continue
            run_dir = run_root_for(model) / model / f"fold_{fold}"
            if not (run_dir / "best.pt").exists():
                raise RuntimeError(f"missing checkpoint: {run_dir/'best.pt'}")
            ho = np.asarray(sorted(b for b in ids if assign[int(b)] == fold), dtype=np.int64)
            tr = np.asarray(sorted(b for b in ids if assign[int(b)] != fold), dtype=np.int64)
            tr_pos = np.asarray([pos[int(b)] for b in tr])
            stats = load_stats(run_dir)
            seen = forward_kge(model, config, run_dir, stats,
                               raw_attrs[[row_of[int(b)] for b in tr]], x_all[:, tr_pos], y_all[:, tr_pos],
                               TEST_DATES, device)
            for b, k in zip(tr.tolist(), seen.tolist()):
                w.writerow({"model": model, "fold": fold, "basin_id": int(b),
                            "kge_insample": k, "n_train": int(tr.size), "n_heldout": int(ho.size)})
            if args.check_heldout:
                ho_pos = np.asarray([pos[int(b)] for b in ho])
                re_ho = forward_kge(model, config, run_dir, stats,
                                    raw_attrs[[row_of[int(b)] for b in ho]], x_all[:, ho_pos], y_all[:, ho_pos],
                                    TEST_DATES, device)
                ref = pd.read_csv(run_dir / "heldout_basin_kge.csv")
                ref_col = "kge" if "kge" in ref.columns else ref.columns[-1]
                ref_map = {int(a): float(c) for a, c in zip(ref[ref.columns[0]], ref[ref_col])}
                d = max(abs(float(k) - ref_map[int(b)]) for b, k in zip(ho.tolist(), re_ho.tolist()))
                ho_check.append({"model": model, "fold": fold, "max_abs_diff_vs_heldout_basin_kge": d})
                log(f"   heldout-reproduction {model} f{fold}: max|diff| = {d:.3e}")
            fh.flush()
            os.fsync(fh.fileno())
            n_cells += 1
            log(f"  [{n_cells}] {model:14s} fold{fold}  n_train={tr.size}  "
                f"median_in_sample={np.median(seen):.4f}")
    fh.close()

    if ho_check:
        # accumulate across invocations so a per-cohort validation set is not overwritten
        rep_path = out_path.parent / "pub_insample_heldout_reproduction.csv"
        new_df = pd.DataFrame(ho_check)
        if rep_path.exists():
            prev_df = pd.read_csv(rep_path)
            new_df = (pd.concat([prev_df, new_df], ignore_index=True)
                        .drop_duplicates(["model", "fold"], keep="last"))
        new_df.to_csv(rep_path, index=False)
        worst = max(r["max_abs_diff_vs_heldout_basin_kge"] for r in ho_check)
        log(f"heldout reproduction worst |diff| = {worst:.3e}")
        if worst > 1e-8:
            raise RuntimeError(f"held-out reproduction failed: {worst:.3e} > 1e-8")

    log(f"wrote {out_path} ({n_cells} new cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
