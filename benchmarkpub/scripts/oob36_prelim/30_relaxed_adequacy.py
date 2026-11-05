#!/usr/bin/env python3
"""30_relaxed_adequacy.py — relaxed, pre-registered adequacy definitions on top of the frozen
Spieler-style adequacy results.

MOTIVATION
----------
The strict Spieler all-pass rule is far too sparse to act as the second stage of
Accuracy -> Adequacy -> Parsimony: only 3.0% (SEEN) / 2.2% (OOB) of model x basin cells pass,
and only 11-13% of basins retain a strict candidate.  This script keeps that rule untouched as
`STRICT_REFERENCE` and adds six FIXED, pre-registered relaxed definitions, then measures whether
any of them can shrink the frozen Top-3 / Top-5 accuracy candidate sets without leaving most
basins with no candidate and without degrading predictive regret.

DEFINITIONS (all fixed before looking at the results; nothing is tuned)
----------------------------------------------------------------------
  STRICT_REFERENCE   the frozen Spieler rule: Qmonth AND Qyear AND all 6 verifiable signatures
                     within +/-50%   (read straight from the frozen parquet, unmodified)
  RELAXED_60         >= 60% of evaluable signatures pass +/-50%      (>= 6/10 when all evaluable)
  RELAXED_70         >= 70%                                          (>= 7/10)
  RELAXED_80         >= 80%                                          (>= 8/10)
  GROUP_3OF5         >= 3 of 5 hydrological-function groups pass
  GROUP_4OF5         >= 4 of 5 groups pass

A group passes when >= 50% of its evaluable signatures pass; a single-signature group (RR, BFI)
therefore requires that signature to pass.  A group with no evaluable signature is UNDEFINED and
is neither passed nor failed.  Undefined signatures are never silently counted as failures.

GROUPING — AND A NAMING COLLISION THAT MUST BE READ
---------------------------------------------------
The task text lists "Low-flow behaviour: Q5, LFfreq, LFdur" and "High-flow behaviour: Q95,
HFfreq, HFdur".  In THIS repository the frozen authority
(`chapter4_multicriteria_v3_20260925/scripts/v3_common.py`) defines

    q5  = quantile(1 - 5/100)  = 95th percentile = HIGH flow   (observed median 4.04 mm/day)
    q95 = quantile(1 - 95/100) =  5th percentile = LOW  flow   (observed median 0.09 mm/day)

i.e. the task's Q5/Q95 labels are swapped relative to this repo (and relative to the usual TOSSH
convention).  Both member sets are computed and reported:

    hydrologically correct grouping (PRIMARY): low_flow = {q95, LFfreq, LFdur},
                                               high_flow = {q5, HFfreq, HFdur}
    literal task grouping (SENSITIVITY):       low_flow = {q5, ...}, high_flow = {q95, ...}

Note that 4 of the 10 signatures (LFfreq, HFfreq, LFdur, HFdur) are the convention-unverified
threshold signatures carried over from the previous stage, so the low- and high-flow groups each
contain 2 unverified members.  A 6-verifiable-signature variant of every definition is reported
alongside as a robustness check.

DATA
----
Reads ONLY the frozen `pub_adequacy_seen_oob_20260930/` parquet and
`pub_next_stage_20260930/` tables.  No model is run, no hydrograph is recomputed, no torch, no
GPU, no training, and nothing under `project/benchmark/` or any frozen analysis directory is
modified.

Outputs -> $OUT7/tables/{relaxed_adequacy_model_basin,relaxed_definition_summary,
                         candidate_set_relaxed_adequacy,adequacy_filter_regret,
                         kge_vs_relaxed_adequacy,relaxed_threshold_sensitivity}.csv
           $OUT7/tables/relaxed_adequacy.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

SEED = 20260930
N_BOOT = 2000
MAIN_THRESHOLD = 50.0
THRESHOLDS = (25.0, 50.0, 75.0)
MHFD_MAX_DIST = 0.25          # day-fraction; the +/-50% analogue for a circular timing metric

SIG6 = ["rc", "bfi", "fdc_slope", "q5", "q95", "mhfd"]
SIG10 = SIG6 + ["lf_freq", "hf_freq", "lf_dur", "hf_dur"]
LABEL = {"rc": "RR", "bfi": "BFI", "fdc_slope": "slopeFDC", "q5": "Q5", "q95": "Q95",
         "mhfd": "MHFD", "lf_freq": "LFfreq", "hf_freq": "HFfreq",
         "lf_dur": "LFdur", "hf_dur": "HFdur"}
UNVERIFIED = ["lf_freq", "hf_freq", "lf_dur", "hf_dur"]

GROUPS_CORRECT = {
    "water_balance": ["rc"],
    "baseflow": ["bfi"],
    "low_flow": ["q95", "lf_freq", "lf_dur"],
    "high_flow": ["q5", "hf_freq", "hf_dur"],
    "timing_distribution": ["mhfd", "fdc_slope"],
}
GROUPS_LITERAL = {
    "water_balance": ["rc"],
    "baseflow": ["bfi"],
    "low_flow": ["q5", "lf_freq", "lf_dur"],
    "high_flow": ["q95", "hf_freq", "hf_dur"],
    "timing_distribution": ["mhfd", "fdc_slope"],
}
GROUPS_VERIFIED6 = {
    "water_balance": ["rc"], "baseflow": ["bfi"], "low_flow": ["q95"],
    "high_flow": ["q5"], "timing_distribution": ["mhfd", "fdc_slope"],
}
GROUPS = {"correct10": GROUPS_CORRECT, "literal10": GROUPS_LITERAL,
          "verified6": GROUPS_VERIFIED6}


def log(m: str) -> None:
    print(m, flush=True)


def boot_mean(x: np.ndarray, B: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float)[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pass_flags(D: pd.DataFrame, thr: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-signature pass flags at `thr`, plus (n_pass, n_eval) over the 10 signatures."""
    P, defn = {}, {}
    for k in SIG10:
        d = D[f"sig_defined_{k}"].to_numpy(bool)
        defn[k] = d
        if k == "mhfd":
            v = D["mhfd_dist"].to_numpy(float)
            P[k] = d & np.isfinite(v) & (v <= MHFD_MAX_DIST * thr / 50.0)
        else:
            p = np.abs(D[f"sig_pct_{k}"].to_numpy(float))
            P[k] = d & np.isfinite(p) & (p <= thr)
    n_pass = np.sum([P[k] for k in SIG10], axis=0).astype(int)
    n_eval = np.sum([defn[k] for k in SIG10], axis=0).astype(int)
    return P, n_pass, n_eval


def group_stats(P: dict, defn: dict, groups: dict):
    """(n_groups_pass, n_groups_defined, per-group pass flags dict)."""
    n_pass = np.zeros(len(next(iter(P.values()))), int)
    n_def = np.zeros_like(n_pass)
    per = {}
    for g, members in groups.items():
        ne = np.sum([defn[k] for k in members], axis=0).astype(int)
        np_ = np.sum([P[k] for k in members], axis=0).astype(int)
        ok = (ne > 0) & (np_ >= 0.5 * ne)          # size-1 group => must pass
        per[g] = {"pass": ok, "n_pass": np_, "n_eval": ne, "undefined": ne == 0}
        n_def += (ne > 0).astype(int)
        n_pass += ok.astype(int)
    return n_pass, n_def, per


def build_definitions(D: pd.DataFrame, thr: float) -> dict:
    """All fixed relaxed definitions at one threshold, as boolean arrays."""
    P, n_pass10, n_eval10 = pass_flags(D, thr)
    P6 = {k: P[k] for k in SIG6}
    defn6 = {k: D[f"sig_defined_{k}"].to_numpy(bool) for k in SIG6}
    n_pass6 = np.sum([P6[k] for k in SIG6], axis=0).astype(int)
    n_eval6 = np.sum([defn6[k] for k in SIG6], axis=0).astype(int)
    frac10 = np.where(n_eval10 > 0, n_pass10 / np.maximum(n_eval10, 1), np.nan)
    frac6 = np.where(n_eval6 > 0, n_pass6 / np.maximum(n_eval6, 1), np.nan)
    defn10 = {k: D[f"sig_defined_{k}"].to_numpy(bool) for k in SIG10}

    g3c, nd3c, _ = group_stats(P, defn10, GROUPS_CORRECT)
    g3l, nd3l, _ = group_stats(P, defn10, GROUPS_LITERAL)
    g3v, nd3v, _ = group_stats(P6, defn6, GROUPS_VERIFIED6)

    out = {
        # ---- the frozen literature baseline, read straight from the parquet
        "STRICT_REFERENCE": D["strict_pass"].to_numpy(bool),
        "SIG6_ALLPASS": D["sig_pass"].to_numpy(bool),
        # ---- relaxed pass-fraction rules on the 10 signatures
        "RELAXED_60": (n_eval10 > 0) & (frac10 >= 0.60),
        "RELAXED_70": (n_eval10 > 0) & (frac10 >= 0.70),
        "RELAXED_80": (n_eval10 > 0) & (frac10 >= 0.80),
        # ---- grouped rules
        "GROUP_3OF5": (nd3c > 0) & (g3c >= 3),
        "GROUP_4OF5": (nd3c > 0) & (g3c >= 4),
        # ---- robustness variants
        "RELAXED_70_verified6": (n_eval6 > 0) & (frac6 >= 0.70),
        "GROUP_3OF5_literal": (nd3l > 0) & (g3l >= 3),
        "GROUP_3OF5_verified6": (nd3v > 0) & (g3v >= 3),
    }
    diag = {
        "n_pass10": n_pass10, "n_eval10": n_eval10, "frac10": frac10,
        "n_pass6": n_pass6, "n_eval6": n_eval6, "frac6": frac6,
        "n_groups_pass_correct": g3c, "n_groups_defined_correct": nd3c,
        "n_groups_pass_literal": g3l, "n_groups_pass_verified6": g3v,
        "pass_flags": P, "defined_flags": defn10,
    }
    return out | {"__diag__": diag}


MAIN_DEFS = ["STRICT_REFERENCE", "RELAXED_60", "RELAXED_70", "RELAXED_80",
             "GROUP_3OF5", "GROUP_4OF5"]
EXTRA_DEFS = ["SIG6_ALLPASS", "RELAXED_70_verified6", "GROUP_3OF5_literal",
              "GROUP_3OF5_verified6"]


def main() -> None:
    PC.T7.mkdir(parents=True, exist_ok=True)
    R = PC.OUT1
    A = R / "pub_adequacy_seen_oob_20260930"
    N = R / "pub_next_stage_20260930"

    L = pd.read_parquet(A / "tables" / "adequacy_fold_level_long.parquet")
    L["basin_id"] = L["basin_id"].astype(np.int64)
    if len(L) != 38232:
        raise ValueError(f"expected 38,232 rows, got {len(L)}")
    S = L[L.realization == "SEEN_canonical_dPL"].reset_index(drop=True)
    O = L[L.realization == "OOB_heldout"].reset_index(drop=True)
    if len(S) != 19116 or len(O) != 19116:
        raise ValueError("realizations must each cover 19,116 cells")
    # NOTE: S and O are NOT in the same row order — the OOB block is grouped by fold — so
    # every SEEN/OOB pairing below is done on the (model, basin_id) KEY, never by position.

    report = {"meta": {
        "task_id": "ch4-oob36-relaxed-adequacy-20260930",
        "n_cells_per_condition": int(len(S)), "threshold_primary": MAIN_THRESHOLD,
        "thresholds_sensitivity": list(THRESHOLDS), "mhfd_max_dist_day_fraction": MHFD_MAX_DIST,
        "definitions_primary": MAIN_DEFS, "definitions_robustness": EXTRA_DEFS,
        "grouping_primary": {k: [LABEL[x] for x in v] for k, v in GROUPS_CORRECT.items()},
        "grouping_literal_task_text": {k: [LABEL[x] for x in v] for k, v in GROUPS_LITERAL.items()},
        "Q5_Q95_naming_collision": (
            "The task text puts Q5 in the low-flow group and Q95 in the high-flow group. In this "
            "repository v3_common defines q5 = 95th percentile = HIGH flow (observed median "
            "4.04 mm/day) and q95 = 5th percentile = LOW flow (0.09 mm/day), so the task labels "
            "are swapped. The hydrologically correct grouping is primary; the literal grouping "
            "is reported as GROUP_3OF5_literal."),
        "unverified_signatures": UNVERIFIED,
        "no_model_execution": True, "no_torch": True, "project_benchmark_modified": False,
        "seen_caveat": ("SEEN is the cross-pipeline canonical dPL store; same-pipeline 4-fold "
                        "in-sample hydrographs do not exist, so the SEEN-OOB pass-rate "
                        "difference must not be read as a strict adequacy generalisation gap"),
    }}

    # ---------------- definitions at the primary threshold -------------------
    dS = build_definitions(S, MAIN_THRESHOLD)
    dO = build_definitions(O, MAIN_THRESHOLD)
    diagS, diagO = dS.pop("__diag__"), dO.pop("__diag__")

    # ---- validation 1: pass counts are reproducible from the per-signature flags
    chk = {}
    for tag, D, dg in (("SEEN", S, diagS), ("OOB", O, diagO)):
        chk[f"{tag}_sig_n_pass_matches_stored"] = bool(
            np.array_equal(dg["n_pass6"], D["sig_n_pass"].to_numpy(int)))
        chk[f"{tag}_sig10_n_pass_matches_stored"] = bool(
            np.array_equal(dg["n_pass10"], D["sig10_n_pass"].to_numpy(int)))
        chk[f"{tag}_strict_reproduced_from_components"] = bool(np.array_equal(
            (D["qm_pass"].to_numpy(bool) & D["qy_pass"].to_numpy(bool)
             & D["sig_pass"].to_numpy(bool)), D["strict_pass"].to_numpy(bool)))
        chk[f"{tag}_strict_rate_matches_frozen_report"] = bool(
            abs(float(D["strict_pass"].mean()) - (0.02977 if tag == "SEEN" else 0.02176)) < 5e-5)
    report["validation"] = chk
    if not all(chk.values()):
        raise AssertionError(f"pass-count reproduction failed: {chk}")

    # ---- validation 2: Top-k candidate sets match the frozen outputs
    F = pd.read_csv(N / "tables" / "pub_same_pipeline_model_basin.csv")
    models = sorted(F["model"].unique())
    basins = np.sort(F["basin_id"].unique())
    kp = F.pivot(index="basin_id", columns="model",
                 values="kge_seenfolds_median4").loc[basins, models].to_numpy()
    order = np.argsort(-kp, axis=1, kind="stable")
    frozen = pd.read_csv(A / "tables" / "candidate_set_adequacy_by_basin.csv")
    mism = 0
    for k in (3, 5):
        sub = frozen[frozen.k == k].set_index("basin_id").reindex(basins)
        for i, b in enumerate(basins):
            mine = sorted(models[j] for j in order[i, :k])
            theirs = sorted(str(sub.loc[int(b), "candidates"]).split("|"))
            mism += int(mine != theirs)
    report["validation"]["topk_membership_mismatches_vs_frozen"] = int(mism)
    if mism:
        raise AssertionError(f"{mism} Top-k memberships differ from the frozen candidate sets")

    # ---------------- model x basin table (joined on the KEY, never by position) --------
    KEY = ["model", "basin_id"]

    def side(D, defs, dg):
        t = pd.DataFrame({"model": D["model"], "basin_id": D["basin_id"],
                          "n_signatures_pass": dg["n_pass10"],
                          "n_signatures_evaluable": dg["n_eval10"],
                          "signature_pass_fraction": dg["frac10"],
                          "n_function_groups_passed": dg["n_groups_pass_correct"],
                          "n_function_groups_defined": dg["n_groups_defined_correct"],
                          "n_function_groups_passed_literal": dg["n_groups_pass_literal"],
                          "median_abs_signature_pct": D["sig10_median_abs_pct"].to_numpy(float),
                          "kge": D["kge"].to_numpy(float)})
        for name in MAIN_DEFS + EXTRA_DEFS:
            t[name] = defs[name]
        _, _, per = group_stats(dg["pass_flags"], dg["defined_flags"], GROUPS_CORRECT)
        for g in GROUPS_CORRECT:
            t[f"group_{g}"] = per[g]["pass"]
            t[f"group_{g}_n_pass"] = per[g]["n_pass"]
            t[f"group_{g}_n_eval"] = per[g]["n_eval"]
        return t.set_index(KEY)

    WS, WO = side(S, dS, diagS), side(O, dO, diagO)
    if len(WS) != 19116 or len(WO) != 19116:
        raise ValueError("per-condition tables must each have 19,116 cells")
    W = WS.add_prefix("seen_").join(WO.add_prefix("oob_"), how="outer", validate="one_to_one")
    if len(W) != 19116 or W.isna().any().any():
        raise ValueError(f"keyed join failed: {len(W)} rows, {int(W.isna().sum().sum())} NaN")
    W = W.reset_index()
    W["model"] = W["model"].astype(str)
    W["basin_id"] = W["basin_id"].astype(np.int64)
    W["gauge_id_str"] = [str(int(b)).zfill(8) for b in W["basin_id"]]
    # tidy column order: plain names without the seen_/oob_ prefix duplicate
    W = W.rename(columns={"seen_kge": "kge_seen", "oob_kge": "kge_oob",
                          "seen_signature_pass_fraction": "signature_pass_fraction_seen",
                          "oob_signature_pass_fraction": "signature_pass_fraction_oob",
                          "seen_n_signatures_pass": "n_signatures_pass_seen",
                          "oob_n_signatures_pass": "n_signatures_pass_oob",
                          "seen_n_signatures_evaluable": "n_signatures_evaluable_seen",
                          "oob_n_signatures_evaluable": "n_signatures_evaluable_oob",
                          "seen_n_function_groups_passed": "n_function_groups_passed_seen",
                          "oob_n_function_groups_passed": "n_function_groups_passed_oob",
                          "seen_median_abs_signature_pct": "median_abs_signature_pct_seen",
                          "oob_median_abs_signature_pct": "median_abs_signature_pct_oob"})
    W.to_csv(PC.T7 / "relaxed_adequacy_model_basin.csv", index=False)

    # cross-check the OOB KGE column against the frozen next-stage table
    kge_oob_frozen = F.set_index(KEY)["kge_heldout"]
    j = W.set_index(KEY)["kge_oob"].reindex(kge_oob_frozen.index)
    report["validation"]["kge_oob_max_abs_diff_vs_frozen"] = float(
        np.max(np.abs(j.to_numpy(float) - kge_oob_frozen.to_numpy(float))))
    if report["validation"]["kge_oob_max_abs_diff_vs_frozen"] > 1e-9:
        raise AssertionError("OOB KGE does not reproduce the frozen table")

    # ---------------- §5 definition comparison --------------------------------
    n_basins = len(basins)
    # W is a keyed join, so its row order differs from S/O; build the index maps explicitly
    # instead of ever assigning a per-condition array into W-ordered positions.
    Srow = {(m, int(b)): i for i, (m, b) in enumerate(zip(S["model"], S["basin_id"]))}
    Orow = {(m, int(b)): i for i, (m, b) in enumerate(zip(O["model"], O["basin_id"]))}
    wkeys = list(zip(W["model"], W["basin_id"].to_numpy(int)))
    s_of_w = np.asarray([Srow[k] for k in wkeys])
    o_of_w = np.asarray([Orow[k] for k in wkeys])
    wb = W.assign(bpos=W["basin_id"].map({int(b): i for i, b in enumerate(basins)}),
                  mpos=W["model"].map({m: i for i, m in enumerate(models)}))
    bidx = wb["bpos"].to_numpy(); midx = wb["mpos"].to_numpy()

    def grid(col):
        g = np.zeros((n_basins, len(models)), dtype=bool)
        g[bidx, midx] = W[col].to_numpy(bool)
        return g

    rowsDef, kept_by_def = [], {}
    for name in MAIN_DEFS + EXTRA_DEFS:
        gs, go = grid(f"seen_{name}"), grid(f"oob_{name}")
        kept_by_def[name] = (gs, go)
        cs = gs.sum(axis=1); co = go.sum(axis=1)
        lo_s, hi_s = boot_mean((co == 0).astype(float))
        rowsDef.append({
            "definition": name,
            "category": "primary" if name in MAIN_DEFS else "robustness",
            "seen_pass_rate": float(gs.mean()), "oob_pass_rate": float(go.mean()),
            "seen_mean_adequate_models_per_basin": float(cs.mean()),
            "seen_median_adequate_models_per_basin": float(np.median(cs)),
            "seen_q25_adequate_per_basin": float(np.percentile(cs, 25)),
            "seen_q75_adequate_per_basin": float(np.percentile(cs, 75)),
            "oob_mean_adequate_models_per_basin": float(co.mean()),
            "oob_median_adequate_models_per_basin": float(np.median(co)),
            "oob_q25_adequate_per_basin": float(np.percentile(co, 25)),
            "oob_q75_adequate_per_basin": float(np.percentile(co, 75)),
            "seen_frac_basins_zero_adequate": float((cs == 0).mean()),
            "oob_frac_basins_zero_adequate": float((co == 0).mean()),
            "oob_frac_basins_zero_ci95_lo": lo_s, "oob_frac_basins_zero_ci95_hi": hi_s,
            "oob_frac_basins_ge3_adequate": float((co >= 3).mean()),
        })
    DS = pd.DataFrame(rowsDef)
    DS.to_csv(PC.T7 / "relaxed_definition_summary.csv", index=False)
    report["definition_summary"] = DS.to_dict("records")

    # ---------------- §6 candidate sets --------------------------------------
    kge_oob_g = np.zeros((n_basins, len(models)))
    kge_oob_g[bidx, midx] = W["kge_oob"].to_numpy(float)
    oracle = kge_oob_g.max(axis=1)
    rr_i = np.arange(n_basins)[:, None]

    rowsCand = []
    for k in (3, 5):
        tk = order[:, :k]
        base_best = kge_oob_g[rr_i, tk].max(axis=1)
        for name in MAIN_DEFS + EXTRA_DEFS:
            gs, go = kept_by_def[name]
            keep = go[rr_i, tk]                       # [n_basins, k]
            n_keep = keep.sum(axis=1)
            has = n_keep >= 1
            best_kept = np.where(keep, kge_oob_g[rr_i, tk], -np.inf).max(axis=1)
            rowsCand.append({
                "candidate_set": f"Top-{k}", "k": k, "definition": name,
                "mean_candidates_before": float(k),
                "mean_candidates_after": float(n_keep.mean()),
                "median_candidates_after": float(np.median(n_keep)),
                "q25_candidates_after": float(np.percentile(n_keep, 25)),
                "q75_candidates_after": float(np.percentile(n_keep, 75)),
                "frac_basins_0_kept": float((n_keep == 0).mean()),
                "frac_basins_1_kept": float((n_keep == 1).mean()),
                "frac_basins_2_kept": float((n_keep == 2).mean()),
                "frac_basins_ge3_kept": float((n_keep >= 3).mean()),
                "frac_basins_contains_ge1_oob_adequate": float(has.mean()),
                "frac_removed_of_all_candidates": float(1 - n_keep.sum() / (n_basins * k)),
                "median_predictive_regret_after_filter": float(np.median(oracle[has] - best_kept[has]))
                    if has.any() else np.nan,
                "mean_predictive_regret_after_filter": float(np.mean(oracle[has] - best_kept[has]))
                    if has.any() else np.nan,
                "q75_predictive_regret_after_filter": float(np.percentile(oracle[has] - best_kept[has], 75))
                    if has.any() else np.nan,
                "q90_predictive_regret_after_filter": float(np.percentile(oracle[has] - best_kept[has], 90))
                    if has.any() else np.nan,
                "frac_regret_le_0p01_after": float(np.mean((oracle[has] - best_kept[has]) <= 0.01))
                    if has.any() else np.nan,
                "frac_regret_le_0p05_after": float(np.mean((oracle[has] - best_kept[has]) <= 0.05))
                    if has.any() else np.nan,
                "median_predictive_regret_unfiltered_topk": float(
                    np.median(oracle - base_best)),
                "frac_regret_le_0p01_unfiltered_topk": float(np.mean((oracle - base_best) <= 0.01)),
                "frac_regret_le_0p05_unfiltered_topk": float(np.mean((oracle - base_best) <= 0.05)),
            })
    CAND = pd.DataFrame(rowsCand)
    CAND.to_csv(PC.T7 / "candidate_set_relaxed_adequacy.csv", index=False)
    CAND.to_csv(PC.T7 / "adequacy_filter_regret.csv", index=False)
    report["candidate_set"] = CAND.to_dict("records")

    # ---------------- §8 adequacy beyond KGE ---------------------------------
    rowsK = []
    for k in (3, 5):
        tk = order[:, :k]
        # pooled within-candidate Spearman across all (basin, candidate) pairs
        sel_kge = kge_oob_g[rr_i, tk].ravel()
        gf = np.zeros((n_basins, len(models))); gf[bidx, midx] = W["signature_pass_fraction_oob"]
        gg = np.zeros((n_basins, len(models)), int); gg[bidx, midx] = W["n_function_groups_passed_oob"]
        gm = np.zeros((n_basins, len(models))); gm[bidx, midx] = W["median_abs_signature_pct_oob"]
        f_k = gf[rr_i, tk].ravel(); g_k = gg[rr_i, tk].ravel(); m_k = gm[rr_i, tk].ravel()
        rowsK.append({
            "candidate_set": f"Top-{k}", "k": k, "condition": "OOB",
            "spearman_KGE_vs_pass_fraction": float(sstats.spearmanr(sel_kge, f_k).statistic),
            "spearman_KGE_vs_n_groups_passed": float(sstats.spearmanr(sel_kge, g_k).statistic),
            "spearman_KGE_vs_median_abs_sig_pct": float(sstats.spearmanr(sel_kge, m_k).statistic),
        })
        # per-basin: is the best-KGE candidate also the most adequate?
        best_kge_col = np.argmax(kge_oob_g[rr_i, tk], axis=1)
        rows_i = np.arange(n_basins)
        for lab, G in (("highest_signature_pass_fraction", gf),
                       ("highest_n_function_groups_passed", gg.astype(float)),
                       ("lowest_median_abs_signature_pct", -gm)):
            sub = G[rr_i, tk]                       # [n_basins, k]
            assert sub.shape == (n_basins, k), sub.shape
            nx = np.argmax(sub, axis=1)
            kmax = sub.max(axis=1)
            top1_val = sub[rows_i, best_kge_col]    # positional, no broadcasting
            rowsK.append({"candidate_set": f"Top-{k}", "k": k, "condition": "OOB",
                          "most_adequate_by": lab,
                          "frac_KGE_top1_is_argmax": float(np.mean(nx == best_kge_col)),
                          "frac_KGE_top1_tied_for_most_adequate":
                              float(np.mean(np.isclose(top1_val, kmax))),
                          "frac_KGE_top1_strictly_worse_than_most_adequate":
                              float(np.mean(top1_val < kmax))})
    KG = pd.DataFrame(rowsK)
    KG.to_csv(PC.T7 / "kge_vs_relaxed_adequacy.csv", index=False)
    report["kge_vs_adequacy"] = KG.to_dict("records")

    # ---------------- §9 threshold sensitivity -------------------------------
    rowsT = []
    for thr in THRESHOLDS:
        dSt = build_definitions(S, thr); dOt = build_definitions(O, thr)
        dSt.pop("__diag__"); dOt.pop("__diag__")
        for name in ("STRICT_REFERENCE", "RELAXED_60", "RELAXED_70", "RELAXED_80",
                     "GROUP_3OF5", "GROUP_4OF5"):
            go = np.zeros((n_basins, len(models)), bool)
            go[bidx, midx] = dOt[name][o_of_w]        # O-ordered -> W-ordered, explicitly
            n_keep3 = go[rr_i, order[:, :3]].sum(axis=1)
            n_keep5 = go[rr_i, order[:, :5]].sum(axis=1)
            rowsT.append({
                "threshold_pct": thr, "definition": name,
                "seen_pass_rate": float(dSt[name].mean()),   # order-independent
                "seen_pass_rate_wcheck": float(dSt[name][s_of_w].mean()),
                "oob_pass_rate": float(dOt[name].mean()),
                "oob_mean_adequate_per_basin": float(go.sum(axis=1).mean()),
                "oob_frac_basins_zero_adequate": float((go.sum(axis=1) == 0).mean()),
                "top3_contains_ge1": float((n_keep3 >= 1).mean()),
                "top3_mean_kept": float(n_keep3.mean()),
                "top5_contains_ge1": float((n_keep5 >= 1).mean()),
                "top5_mean_kept": float(n_keep5.mean()),
            })
    TS = pd.DataFrame(rowsT)
    TS.to_csv(PC.T7 / "relaxed_threshold_sensitivity.csv", index=False)
    report["threshold_sensitivity"] = TS.to_dict("records")

    with open(PC.T7 / "relaxed_adequacy.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ---------------- console -------------------------------------------------
    log("== validation ==")
    for k, v in report["validation"].items():
        log(f"   {k}: {v}")
    log("\n== §5 definition comparison ==")
    log(DS[["definition", "seen_pass_rate", "oob_pass_rate",
            "oob_mean_adequate_models_per_basin", "oob_median_adequate_models_per_basin",
            "oob_frac_basins_zero_adequate"]].round(4).to_string(index=False))
    log("\n== §6/§7 candidate sets (OOB) ==")
    log(CAND[["candidate_set", "definition", "mean_candidates_after", "median_candidates_after",
              "frac_basins_0_kept", "frac_basins_contains_ge1_oob_adequate",
              "median_predictive_regret_after_filter", "frac_regret_le_0p01_after",
              "median_predictive_regret_unfiltered_topk"]].round(4).to_string(index=False))
    log("\n== §8 adequacy beyond KGE (Top-3) ==")
    log(KG[KG.k == 3].round(4).to_string(index=False))
    log("\n== §9 threshold sensitivity ==")
    log(TS[TS.definition.isin(["STRICT_REFERENCE", "RELAXED_70", "GROUP_3OF5"])]
        [["threshold_pct", "definition", "oob_pass_rate", "top3_mean_kept", "top3_contains_ge1",
          "top5_mean_kept", "top5_contains_ge1"]].round(4).to_string(index=False))
    log(f"\nwrote -> {PC.T7}")


if __name__ == "__main__":
    main()
