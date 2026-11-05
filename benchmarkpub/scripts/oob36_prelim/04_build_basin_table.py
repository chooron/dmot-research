#!/usr/bin/env python3
"""04_build_basin_table.py — steps 2.1-2.4 of ch4-oob36-download-prelim-20260930.

2.2  import the frozen metric functions (see metric_sources.md) and self-check on
     one model x one fold against the run's `heldout_basin_kge.csv` and the
     `OOB36_REINFERENCE.csv` record.
2.3  build the OOB basin table (36 models x 531 basins = 19,116 rows).
2.4  merge with the frozen chapter-4 seen-dPL / IC paired KGE table.

All metric arithmetic is done by the imported frozen functions; nothing is
re-implemented.  Joins are on the integer basin ID, never on row position.

Outputs
    $OUT/tables/oob36_basin_metrics.parquet   (+ .csv)
    $OUT/tables/seen_ic_oob_merged.parquet    (+ .csv)
    $OUT/table_build_summary.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    OUT, OUT_TABLES, DATA_QSIM, DATA_OBS, DATA_RUNS, DATA_MANIFESTS,
    BENCH, PAIRED_LONG_CSV, gauge_id_str,
)

FROZEN = {
    "minimal_kge": BENCH + "/analysis/chapter4_oob_pipeline_audit_20260926/scripts/minimal_kge.py",
    "rb2_metrics": BENCH + "/analysis/chapter4_robustness_supplement_20260925/scripts/rb2_metrics.py",
    "s2_supp": BENCH + "/analysis/chapter4_final_closure_20260925/scripts/s2_supplementary_metrics_recommend.py",
}


def load_module(name: str, path: str):
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def init_metrics():
    return (load_module("fz_minimal_kge", FROZEN["minimal_kge"]),
            load_module("fz_s2_supp", FROZEN["s2_supp"]),
            load_module("fz_rb2", FROZEN["rb2_metrics"]))


def load_fold_assignment() -> dict[int, int]:
    cands = [os.path.join(DATA_MANIFESTS, "OOB_FOLD_ASSIGNMENT.csv"),
             os.path.join(DATA_MANIFESTS, "OOB_FOLD_ASSIGNMENT.local_frozen.csv")]
    p = next((c for c in cands if os.path.isfile(c)), None)
    if p is None:
        raise FileNotFoundError("no OOB_FOLD_ASSIGNMENT*.csv in $DATA/manifests")
    out = {}
    for r in csv.DictReader(open(p)):
        gid = int(r["basin_id"])
        if gid in out:
            raise ValueError(f"duplicate basin in fold assignment: {gid}")
        out[gid] = int(r["fold"])
    if len(out) != 531:
        raise ValueError(f"fold assignment has {len(out)} basins, expected 531")
    return out


def canonical_fold_ids(fold_map: dict[int, int], k: int) -> list[int]:
    return sorted(g for g, f in fold_map.items() if f == k)


def load_run_ids(model: str, fold: int) -> list[int]:
    p = os.path.join(DATA_RUNS, model, f"fold_{fold}", "heldout_basin_ids.txt")
    return [int(x) for x in open(p).read().split()]


def hf_mask(obs_tB: np.ndarray) -> np.ndarray:
    """Top-2% observed-flow days, [B, T] — the frozen rb2/s2 definition."""
    o = obs_tB.T
    thr = np.nanquantile(o, 0.98, axis=1)
    return np.isfinite(o) & (o >= thr[:, None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--selfcheck-model", default="collie2")
    ap.add_argument("--selfcheck-fold", type=int, default=0)
    args = ap.parse_args()

    M_MIN, M_S2, M_RB2 = init_metrics()
    os.makedirs(OUT_TABLES, exist_ok=True)

    inv = json.load(open(os.path.join(OUT, "remote_inventory.json")))
    fold_map = load_fold_assignment()
    if len({r["model"] for r in inv}) != 36:
        raise ValueError("expected 36 models in the inventory")

    # ------------------------------------------------------------------ #
    # step 2.2 self-check
    # ------------------------------------------------------------------ #
    m0, k0 = args.selfcheck_model, args.selfcheck_fold
    q0 = np.load(os.path.join(DATA_QSIM, m0, f"fold_{k0}.npy"))
    o0 = np.load(os.path.join(DATA_OBS, f"fold_{k0}", "heldout_y.npy"))
    ids0 = load_run_ids(m0, k0)
    kge0, pairs0 = M_MIN.kge_minimal(q0, o0)

    ref = pd.read_csv(os.path.join(DATA_RUNS, m0, f"fold_{k0}", "heldout_basin_kge.csv"))
    key_col = ref.columns[0]
    ref = ref.set_index(key_col)
    d_csv = max(abs(float(ref.loc[i].iloc[0]) - kge0[j])
                for j, i in enumerate(ids0) if i in ref.index)

    d_reinf = None
    rc_path = os.path.join(DATA_MANIFESTS, "tables", "OOB36_REINFERENCE.csv")
    if os.path.isfile(rc_path):
        rc = pd.read_csv(rc_path)
        row = rc[(rc["model"] == m0) & (rc["fold"] == k0)]
        if len(row) == 1:
            d_reinf = float(row.iloc[0]["max_abs_diff_perbasin_kge_vs_frozen_csv"])

    import torch
    if os.path.join(BENCH, "src") not in sys.path:
        sys.path.insert(0, os.path.join(BENCH, "src"))
    from objective import streaming_kge  # noqa: E402
    with torch.no_grad():
        sk, _ = streaming_kge(torch.as_tensor(q0, dtype=torch.float64).unsqueeze(-1).unsqueeze(-1),
                              torch.as_tensor(o0, dtype=torch.float64), eps=0.1)
    d_pipeline = float(np.max(np.abs(sk[:, 0, 0].numpy() - kge0)))

    selfcheck = {
        "model": m0, "fold": k0, "n_basins": int(q0.shape[1]),
        "max_abs_diff_vs_heldout_basin_kge_csv": d_csv,
        "max_abs_diff_implied_by_OOB36_REINFERENCE_record": d_reinf,
        "max_abs_diff_minimal_vs_streaming_kge": d_pipeline,
        "pass_csv_le_1e-8": bool(d_csv <= 1e-8),
        "pass_pipeline_le_1e-8": bool(d_pipeline <= 1e-8),
    }
    print("== step 2.2 self-check ==")
    for k, v in selfcheck.items():
        print(f"   {k:46s} {v}")
    json.dump(selfcheck, open(os.path.join(OUT, "metric_selfcheck.json"), "w"), indent=1)
    if not (selfcheck["pass_csv_le_1e-8"] and selfcheck["pass_pipeline_le_1e-8"]):
        raise SystemExit("step 2.2 self-check FAILED (max abs diff > 1e-8)")
    if args.selfcheck_only:
        return

    # ------------------------------------------------------------------ #
    # step 2.3 OOB basin table
    # ------------------------------------------------------------------ #
    frames, per_run = [], []
    for r in sorted(inv, key=lambda x: (x["model"], x["fold"])):
        m, k = r["model"], r["fold"]
        q = np.load(os.path.join(DATA_QSIM, m, f"fold_{k}.npy"))        # [T,B]
        o = np.load(os.path.join(DATA_OBS, f"fold_{k}", "heldout_y.npy"))
        ids = load_run_ids(m, k)

        if q.shape != o.shape:
            raise ValueError(f"{m}/fold_{k}: qsim {q.shape} vs obs {o.shape}")
        if len(ids) != q.shape[1]:
            raise ValueError(f"{m}/fold_{k}: {len(ids)} ids vs {q.shape[1]} columns")
        if ids != canonical_fold_ids(fold_map, k):
            raise ValueError(f"{m}/fold_{k}: basin order/ids differ from the frozen fold assignment")

        kge, pairs = M_MIN.kge_minimal(q, o)
        eps_b = 0.01 * np.nanmean(o, axis=0)                            # [B]
        logkge = M_S2.compute_logkge_point(q.T[None], o.T, eps_b)[:, 0]  # [B]
        _s, err = M_S2.compute_fhv_score(q.T[None], o.T, hf_mask(o))     # [B,1]
        fhv = err[:, 0]

        frames.append(pd.DataFrame({
            "model": m, "fold": k,
            "gauge_id": np.asarray(ids, dtype=np.int64),
            "kge_oob": kge, "logkge_oob": logkge, "fhv_oob": fhv,
            "n_valid": pairs.astype(np.int64),
        }))
        per_run.append({"model": m, "fold": k, "n_basins": int(q.shape[1]),
                        "mean_obs_hf_days": float(hf_mask(o).sum(axis=1).mean())})

    tbl = pd.concat(frames, ignore_index=True)
    tbl["gauge_id_str"] = [gauge_id_str(g) for g in tbl["gauge_id"]]
    tbl = tbl[["model", "gauge_id", "gauge_id_str", "fold",
               "kge_oob", "logkge_oob", "fhv_oob", "n_valid"]]

    if len(tbl) != 19116:
        raise ValueError(f"expected 19,116 rows, got {len(tbl)}")
    if tbl.duplicated(["model", "gauge_id"]).any():
        raise ValueError("duplicate (model, gauge_id) rows")
    if tbl.groupby("model")["gauge_id"].nunique().min() != 531:
        raise ValueError("not every model covers 531 basins")
    if tbl.groupby("gauge_id")["fold"].nunique().max() != 1:
        raise ValueError("a basin belongs to more than one fold")
    n_missing = int(tbl[["kge_oob", "logkge_oob", "fhv_oob"]].isna().sum().sum())
    print(f"\n== step 2.3 ==")
    print(f"   rows={len(tbl)}  models={tbl.model.nunique()}  basins={tbl.gauge_id.nunique()}")
    print(f"   non-finite metric cells = {n_missing}")
    if n_missing:
        raise ValueError(f"{n_missing} non-finite metric cells in the OOB table")

    tbl.to_parquet(os.path.join(OUT_TABLES, "oob36_basin_metrics.parquet"), index=False)
    tbl.to_csv(os.path.join(OUT_TABLES, "oob36_basin_metrics.csv"), index=False)

    # ------------------------------------------------------------------ #
    # step 2.4 merge with the frozen seen / IC table
    # ------------------------------------------------------------------ #
    paired = pd.read_csv(PAIRED_LONG_CSV)
    paired["gauge_id"] = paired["basin_id"].astype(np.int64)
    keep = ["model", "gauge_id", "KGE_IC", "KGE_dPL", "Delta_KGE"]
    for extra in ("logKGE_IC", "logKGE_dPL", "FHV_IC", "FHV_dPL"):
        if extra in paired.columns:
            keep.append(extra)
    merged = tbl.merge(paired[keep], on=["model", "gauge_id"], how="left",
                       validate="one_to_one", indicator=True)
    n_unmatched = int((merged["_merge"] != "both").sum())
    merged = merged.drop(columns=["_merge"])
    merged = merged.rename(columns={"KGE_IC": "kge_ic", "KGE_dPL": "kge_seen_dpl",
                                    "Delta_KGE": "delta_kge_ic_minus_dpl"})
    merged["delta_kge_seen_minus_oob"] = merged["kge_seen_dpl"] - merged["kge_oob"]
    print(f"\n== step 2.4 ==")
    print(f"   merged rows={len(merged)}  unmatched={n_unmatched}")
    print(f"   seen/IC logKGE/FHV columns present: "
          f"{[c for c in ('logKGE_IC','logKGE_dPL','FHV_IC','FHV_dPL') if c in paired.columns]}")
    if n_unmatched:
        bad = merged.loc[merged["kge_seen_dpl"].isna(), ["model", "gauge_id"]]
        raise ValueError(f"{n_unmatched} unmatched rows, e.g. {bad.head().values.tolist()}")

    merged.to_parquet(os.path.join(OUT_TABLES, "seen_ic_oob_merged.parquet"), index=False)
    merged.to_csv(os.path.join(OUT_TABLES, "seen_ic_oob_merged.csv"), index=False)

    summary = {
        "selfcheck": selfcheck,
        "oob_table_rows": int(len(tbl)),
        "oob_table_models": int(tbl.model.nunique()),
        "oob_table_basins": int(tbl.gauge_id.nunique()),
        "merged_rows": int(len(merged)),
        "merged_unmatched": n_unmatched,
        "seen_logkge_fhv_available": [c for c in ("logKGE_IC", "logKGE_dPL", "FHV_IC", "FHV_dPL")
                                      if c in paired.columns],
        "per_run_high_flow_days": per_run,
    }
    json.dump(summary, open(os.path.join(OUT, "table_build_summary.json"), "w"), indent=1)
    print("\n   wrote tables/oob36_basin_metrics.{parquet,csv}, "
          "tables/seen_ic_oob_merged.{parquet,csv}")


if __name__ == "__main__":
    main()
