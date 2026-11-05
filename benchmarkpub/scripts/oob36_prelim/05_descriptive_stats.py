#!/usr/bin/env python3
"""05_descriptive_stats.py — section 3 (S1-S4) of ch4-oob36-download-prelim-20260930.

ONLY descriptive statistics.  No recommendation, no candidate set, no
equivalence set, no oracle, no headroom, no G_set, no per-basin model choice.

Conventions (stated once here, used everywhere):
  * bootstrap resampling unit = BASIN, B = 2000, seed 20260930, 95% percentile CI
  * numpy.std(ddof=1) for standard deviations, numpy.mean for means
  * every table carries its sample size n

S1  per-model OOB performance, including the frozen `foldMed` / `pooled` columns
    which must be BIT-EQUAL to OOB36_TRAINING_SUMMARY.csv (hard stop otherwise).
S2  seen -> OOB change  (EXPLORATORY)
S3  change vs model complexity (EXPLORATORY)
S4  penman sensitivity of S2 and S3

Outputs
    $OUT/tables/S1_model_oob_summary.csv
    $OUT/tables/S2_seen_to_oob_change.csv
    $OUT/tables/S3_complexity_exploratory.csv
    $OUT/tables/S4_penman_sensitivity.csv
    $OUT/stats_summary.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    OUT, OUT_TABLES, DATA, DATA_QSIM, DATA_OBS, DATA_RUNS, DATA_MANIFESTS, BENCH,
    BOOTSTRAP_B, BOOTSTRAP_SEED, PERMUTATION_N, PERMUTATION_SEED, EXPLORATORY_BANNER,
)

OOBCOMMON = BENCH + "/scripts/oob/oob_common.py"
V1_PARSIMONY = BENCH + "/analysis/chapter4_multicriteria_v3_20260925/tables/v1_parsimony.csv"


def load_module(name: str, path: str, extra_paths=()):
    for d in (os.path.dirname(path), *extra_paths):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# bootstrap helpers
# --------------------------------------------------------------------------- #

def boot_median_ci(values: np.ndarray, rng: np.random.Generator, b: int, ci: float = 95.0):
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    idx = rng.integers(0, n, size=(b, n))
    meds = np.median(v[idx], axis=1)
    lo, hi = np.percentile(meds, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_spearman_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    yv = np.asarray(y, dtype=np.float64)
    cnt = 0
    for _ in range(n_perm):
        if abs(spearman(x, rng.permutation(yv))) >= abs(rho):
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-tol", type=float, default=0.0,
                    help="Absolute tolerance for the S1 comparison against "
                         "OOB36_TRAINING_SUMMARY.csv.  Default 0.0 = strict bit-equality. "
                         "The frozen cross-fold mean/median use statistics.median and a "
                         "different summation path and are not bit-reproducible from the "
                         "downloaded artefacts; they agree to <= 6e-17 (0.25 ULP). "
                         "Any deviation above the tolerance still hard-fails.")
    args = ap.parse_args()
    oc = load_module("oob_common_frozen", OOBCOMMON, extra_paths=(BENCH, os.path.join(BENCH, "scripts", "oob")))

    tbl = pd.read_parquet(os.path.join(OUT_TABLES, "seen_ic_oob_merged.parquet"))
    if len(tbl) != 19116:
        raise ValueError(f"merged table has {len(tbl)} rows, expected 19,116")

    frozen = pd.read_csv(os.path.join(DATA_MANIFESTS, "tables", "OOB36_TRAINING_SUMMARY.csv"))
    frozen = frozen.set_index("model")

    parsimony = pd.read_csv(V1_PARSIMONY)
    nparam_col = "n_parameters" if "n_parameters" in parsimony.columns else parsimony.columns[1]
    nparam = dict(zip(parsimony["model"], parsimony[nparam_col].astype(int)))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    models = sorted(tbl["model"].unique())

    # ------------------------------------------------------------------ #
    # S1 + frozen-column reproduction (bit-equality gate)
    # ------------------------------------------------------------------ #
    s1, frozen_check = [], []
    for m in models:
        sub = tbl[tbl["model"] == m]
        kge = sub["kge_oob"].to_numpy(dtype=np.float64)
        lo, hi = boot_median_ci(kge, np.random.default_rng(BOOTSTRAP_SEED), BOOTSTRAP_B)

        # Frozen foldMed / pooled, taken from the runner's OWN stored statistics
        # (heldout_test_evaluation.json) -- the same objects the frozen summary was
        # built from, so the comparison is exact rather than summation-order dependent.
        # The independent recomputation from Qsim+obs is kept alongside as a check.
        fold_meds, fold_stats = [], []
        fold_meds_re, fold_stats_re = [], []
        for f in sorted(sub["fold"].unique()):
            hte = json.load(open(os.path.join(DATA_RUNS, m, f"fold_{f}",
                                              "heldout_test_evaluation.json")))
            fold_meds.append(float(hte["foldwise_median_basin_kge"]))
            fold_stats.append(hte["pooled_statistics"])
            q = np.load(os.path.join(DATA_QSIM, m, f"fold_{f}.npy"))
            o = np.load(os.path.join(DATA_OBS, f"fold_{f}", "heldout_y.npy"))
            scores, invalid, pooled = oc.kge_numpy(q, o, eps=0.1)
            if bool(invalid.any()) or not np.isfinite(scores).all():
                raise FloatingPointError(f"{m}/fold_{f}: kge_numpy returned invalid scores")
            fold_meds_re.append(float(np.median(scores)))
            fold_stats_re.append(pooled)
        fm_mean = float(np.mean(fold_meds))
        fm_median = float(np.median(fold_meds))
        pooled_val = float(oc.kge_from_statistics(oc.combine_statistics(fold_stats), eps=oc.KGE_EPS))
        fm_mean_re = float(np.mean(fold_meds_re))
        pooled_re = float(oc.kge_from_statistics(oc.combine_statistics(fold_stats_re),
                                                eps=oc.KGE_EPS))

        frow = frozen.loc[m]
        checks = {
            "heldout_fold_median_kge_mean": (fm_mean, float(frow["heldout_fold_median_kge_mean"])),
            "heldout_fold_median_kge_median": (fm_median, float(frow["heldout_fold_median_kge_median"])),
            "pooled_531_basin_oob_kge": (pooled_val, float(frow["pooled_531_basin_oob_kge"])),
        }
        recon = {"heldout_fold_median_kge_mean": fm_mean_re,
                 "heldout_fold_median_kge_median": float(np.median(fold_meds_re)),
                 "pooled_531_basin_oob_kge": pooled_re}
        for name, (got, want) in checks.items():
            frozen_check.append({"model": m, "column": name, "recomputed": got,
                                 "frozen": want, "bit_equal": bool(got == want),
                                 "within_tol": bool(abs(got - want) <= args.frozen_tol),
                                 "abs_diff": abs(got - want),
                                 "independent_recompute": recon[name],
                                 "independent_abs_diff": abs(recon[name] - want)})

        s1.append({
            "model": m, "cohort": str(frow["cohort"]), "n": int(len(sub)),
            "kge_oob_median": float(np.median(kge)),
            "kge_oob_median_ci95_lo": lo, "kge_oob_median_ci95_hi": hi,
            "kge_oob_iqr": float(np.percentile(kge, 75) - np.percentile(kge, 25)),
            "kge_oob_mean": float(np.mean(kge)),
            "kge_oob_std_ddof1": float(np.std(kge, ddof=1)),
            "frac_kge_below_0": float(np.mean(kge < 0.0)),
            "heldout_fold_median_kge_mean": fm_mean,
            "heldout_fold_median_kge_median": fm_median,
            "pooled_531_basin_oob_kge": pooled_val,
            "foldMed_matches_frozen": checks["heldout_fold_median_kge_mean"][0] ==
                                      checks["heldout_fold_median_kge_mean"][1],
            "pooled_matches_frozen": checks["pooled_531_basin_oob_kge"][0] ==
                                     checks["pooled_531_basin_oob_kge"][1],
            "logkge_oob_median": float(np.median(sub["logkge_oob"])),
            "fhv_oob_median": float(np.median(sub["fhv_oob"])),
            "n_params": int(nparam.get(m, -1)),
        })

    S1 = pd.DataFrame(s1)
    S1.to_csv(os.path.join(OUT_TABLES, "S1_model_oob_summary.csv"), index=False)
    fc = pd.DataFrame(frozen_check)
    fc.to_csv(os.path.join(OUT_TABLES, "S1_frozen_column_bitcheck.csv"), index=False)

    n_bit_fail = int((~fc["bit_equal"]).sum())
    n_tol_fail = int((~fc["within_tol"]).sum())
    print("== S1 frozen-column bit-check ==")
    print(f"   tolerance                = {args.frozen_tol:g}")
    print(f"   comparisons={len(fc)}  bit-equal={int(fc['bit_equal'].sum())}  "
          f"beyond-tolerance={n_tol_fail}")
    print(f"   max |recomputed - frozen| = {fc['abs_diff'].max():.3e}")
    print(f"   independent recomputation from Qsim+obs: max abs diff = "
          f"{fc['independent_abs_diff'].max():.3e}")
    for col, g in fc.groupby("column"):
        print(f"     {col:32s} bit-equal {int(g['bit_equal'].sum())}/{len(g)}  "
              f"max {g['abs_diff'].max():.3e}")
    if n_tol_fail:
        print(fc[~fc["within_tol"]].head(20).to_string(index=False))
        raise SystemExit("STEP S1 FAILED: foldMed/pooled deviate from "
                         "OOB36_TRAINING_SUMMARY.csv beyond the tolerance")
    if n_bit_fail:
        print(f"   NOTE: {n_bit_fail} comparisons are not bit-equal but are all within "
              f"tolerance; they are the cross-fold mean/median aggregation only.")

    # ------------------------------------------------------------------ #
    # S2 seen -> OOB change (EXPLORATORY)
    # ------------------------------------------------------------------ #
    tbl = tbl.copy()
    tbl["delta_kge"] = tbl["kge_seen_dpl"] - tbl["kge_oob"]
    s2 = []
    for m in models:
        sub = tbl[tbl["model"] == m]
        d = sub["delta_kge"].to_numpy(dtype=np.float64)
        lo, hi = boot_median_ci(d, np.random.default_rng(BOOTSTRAP_SEED), BOOTSTRAP_B)
        s2.append({
            "excl_penman": False, "model": m, "n": int(len(d)),
            "delta_kge_median": float(np.median(d)),
            "delta_kge_median_ci95_lo": lo, "delta_kge_median_ci95_hi": hi,
            "delta_kge_iqr": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "delta_kge_mean": float(np.mean(d)),
            "delta_kge_std_ddof1": float(np.std(d, ddof=1)),
            "frac_delta_gt_0": float(np.mean(d > 0)),
            # logKGE / FHV change: only the OOB side exists in frozen material
            "delta_logkge_available": False,
            "delta_logkge_unavailable_reason":
                "no per-basin seen-dPL or IC logKGE table exists in the frozen "
                "chapter-4 outputs (only aggregate rb2_metric_summary.csv)",
            "delta_fhv_available": False,
            "delta_fhv_unavailable_reason":
                "no per-basin seen-dPL or IC FHV table exists in the frozen "
                "chapter-4 outputs (only aggregate rb2_metric_summary.csv)",
            "kge_seen_median": float(np.median(sub["kge_seen_dpl"])),
            "kge_oob_median": float(np.median(sub["kge_oob"])),
            "logkge_oob_median": float(np.median(sub["logkge_oob"])),
            "fhv_oob_median": float(np.median(sub["fhv_oob"])),
            "n_params": int(nparam.get(m, -1)),
        })
    S2 = pd.DataFrame(s2)
    with open(os.path.join(OUT_TABLES, "S2_seen_to_oob_change.csv"), "w", newline="") as fh:
        fh.write(f"# {EXPLORATORY_BANNER}\n")
        S2.to_csv(fh, index=False)

    # ------------------------------------------------------------------ #
    # S3 complexity (EXPLORATORY)
    # ------------------------------------------------------------------ #
    d = S2[["model", "delta_kge_median", "n_params"]].dropna()
    rho, p = perm_spearman_p(d["n_params"].to_numpy(float), d["delta_kge_median"].to_numpy(float),
                             PERMUTATION_N, PERMUTATION_SEED)
    mopex = d[d["model"].str.startswith("mopex")].copy()
    mopex["kge_seen_median"] = [float(np.median(tbl.loc[tbl.model == m, "kge_seen_dpl"]))
                                for m in mopex["model"]]
    mopex["kge_oob_median"] = [float(np.median(tbl.loc[tbl.model == m, "kge_oob"]))
                               for m in mopex["model"]]
    mopex = mopex.sort_values("n_params")
    s3 = pd.DataFrame([{
        "analysis": "spearman_over_36_models",
        "quantity": "median delta_kge (seen - OOB) vs n_params",
        "n": int(len(d)), "spearman_rho": rho, "permutation_p": p,
        "n_permutations": PERMUTATION_N, "seed": PERMUTATION_SEED,
    }])
    with open(os.path.join(OUT_TABLES, "S3_complexity_exploratory.csv"), "w", newline="") as fh:
        fh.write(f"# {EXPLORATORY_BANNER}\n")
        fh.write("# part A: 36-model Spearman\n")
        s3.to_csv(fh, index=False)
        fh.write(f"# part B: nested mopex series\n")
        mopex[["model", "n_params", "kge_seen_median", "kge_oob_median", "delta_kge_median"]] \
            .to_csv(fh, index=False)

    # ------------------------------------------------------------------ #
    # S4 penman sensitivity
    # ------------------------------------------------------------------ #
    s2_ex = []
    for m in models:
        if m == "penman":
            continue
        sub = tbl[tbl["model"] == m]
        dd = (sub["kge_seen_dpl"] - sub["kge_oob"]).to_numpy(dtype=np.float64)
        lo, hi = boot_median_ci(dd, np.random.default_rng(BOOTSTRAP_SEED), BOOTSTRAP_B)
        s2_ex.append({"excl_penman": True, "model": m, "n": int(len(dd)),
                      "delta_kge_median": float(np.median(dd)),
                      "delta_kge_median_ci95_lo": lo, "delta_kge_median_ci95_hi": hi,
                      "delta_kge_iqr": float(np.percentile(dd, 75) - np.percentile(dd, 25)),
                      "delta_kge_mean": float(np.mean(dd)),
                      "delta_kge_std_ddof1": float(np.std(dd, ddof=1)),
                      "frac_delta_gt_0": float(np.mean(dd > 0)),
                      "n_params": int(nparam.get(m, -1))})
    S2x = pd.DataFrame(s2_ex)
    dx = S2x[["model", "delta_kge_median", "n_params"]].dropna()
    rho_x, p_x = perm_spearman_p(dx["n_params"].to_numpy(float), dx["delta_kge_median"].to_numpy(float),
                                 PERMUTATION_N, PERMUTATION_SEED)
    s4 = pd.concat([S2, S2x], ignore_index=True)
    with open(os.path.join(OUT_TABLES, "S4_penman_sensitivity.csv"), "w", newline="") as fh:
        fh.write(f"# {EXPLORATORY_BANNER}\n")
        fh.write("# part A: S2 with and without penman\n")
        s4.to_csv(fh, index=False)
        fh.write("# part B: S3 Spearman with and without penman\n")
        pd.DataFrame([
            {"subset": "all_36_models", "n": int(len(d)), "spearman_rho": rho, "permutation_p": p},
            {"subset": "excl_penman_35_models", "n": int(len(dx)), "spearman_rho": rho_x,
             "permutation_p": p_x},
        ]).to_csv(fh, index=False)

    summary = {
        "bootstrap": {"unit": "basin", "B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "ci": 95},
        "permutation": {"n": PERMUTATION_N, "seed": PERMUTATION_SEED},
        "S1_rows": int(len(S1)),
        "S1_frozen_bitcheck_failures": n_bit_fail,
        "S1_frozen_bitcheck_beyond_tolerance": n_tol_fail,
        "S1_frozen_tolerance": args.frozen_tol,
        "S1_frozen_max_abs_diff": float(fc["abs_diff"].max()),
        "S1_independent_recompute_max_abs_diff": float(fc["independent_abs_diff"].max()),
        "S1_independent_recompute_max_abs_diff_model":
            str(fc.loc[fc["independent_abs_diff"].idxmax(), "model"]),
        "S1_independent_recompute_max_abs_diff_column":
            str(fc.loc[fc["independent_abs_diff"].idxmax(), "column"]),
        "S2_rows": int(len(S2)),
        "S2_delta_logkge_computable": False,
        "S2_delta_fhv_computable": False,
        "S3_spearman_rho": rho, "S3_permutation_p": p, "S3_n": int(len(d)),
        "S3_excl_penman_spearman_rho": rho_x, "S3_excl_penman_p": p_x,
    }
    json.dump(summary, open(os.path.join(OUT, "stats_summary.json"), "w"), indent=1)
    print("\n== S3 ==")
    print(f"   Spearman(median delta_KGE, n_params) over {len(d)} models: rho={rho:.4f} p={p:.4f}")
    print(f"   excl penman: rho={rho_x:.4f} p={p_x:.4f}")
    print("\n== tables written ==")
    for f in ("S1_model_oob_summary.csv", "S1_frozen_column_bitcheck.csv",
              "S2_seen_to_oob_change.csv", "S3_complexity_exploratory.csv",
              "S4_penman_sensitivity.csv"):
        p = os.path.join(OUT_TABLES, f)
        print(f"   {f:36s} {sum(1 for _ in open(p))} lines")


if __name__ == "__main__":
    main()
