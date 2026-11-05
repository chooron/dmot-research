#!/usr/bin/env python3
"""27_adequacy_analyses.py — Analyses A-E on the adequacy table produced by script 26.

A1  Spieler-style accuracy set: within each basin and condition, the "accurate" models are
    those with KGE within 0.05 of the basin's best; how many of them are adequate?
A2  The frozen Seen Top-3 / Top-5 candidate sets: how many members are adequate, under SEEN
    and under OOB?
B   Does adequacy itself transfer seen -> OOB?  Contingency, retention, agreement, per-basin
    Jaccard of the adequate-model sets, and Spearman of the continuous errors.
C   Can adequacy refine the accuracy-based candidate sets, and does the seen Top-k contain at
    least one OOB-adequate model?
D   Which hydrological behaviours fail, ranked, and how the failure rate shifts seen -> OOB.
E   Is the KGE winner the most adequate candidate?

Threshold sensitivity at +/-25/50/75% is computed EXACTLY from the stored per-signature
percentage errors and per-component maxima, so no re-computation of the signatures is needed.

Reads only $OUT6/tables/adequacy_fold_level_long.parquet.  No inference, no torch.
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
ACC_DELTA_KGE = 0.05          # Spieler-style accuracy band
THRESHOLDS = (25.0, 50.0, 75.0)
MHFD_MAX_DIST = 0.25
SIG5 = ["rc", "bfi", "fdc_slope", "q5", "q95"]
SIG4 = ["lf_freq", "hf_freq", "lf_dur", "hf_dur"]
SIG_LABEL = {"rc": "RR", "bfi": "BFI", "fdc_slope": "slopeFDC", "q5": "Q5", "q95": "Q95",
             "mhfd": "MHFD", "lf_freq": "LFfreq", "hf_freq": "HFfreq",
             "lf_dur": "LFdur", "hf_dur": "HFdur", "qmonth": "Qmonth", "qyear": "Qyear"}


def log(m: str) -> None:
    print(m, flush=True)


def boot_mean(x: np.ndarray, B: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float)[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_median(x: np.ndarray, B: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(x, float)
    d = np.median(v[rng.integers(0, len(v), size=(B, len(v)))], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def derive_pass_at(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Component pass flags at an arbitrary threshold, exactly, from stored errors."""
    out = pd.DataFrame(index=df.index)
    out["qm_pass"] = (df["qm_n_evaluable"] > 0) & (df["qm_max_abs_pct"] <= thr)
    out["qy_pass"] = (df["qy_n_evaluable"] > 0) & (df["qy_max_abs_pct"] <= thr)
    n_eval = np.zeros(len(df), int)
    n_pass = np.zeros(len(df), int)
    for k in SIG5:
        d = df[f"sig_defined_{k}"].to_numpy(bool)
        p = df[f"sig_pct_{k}"].to_numpy(float)
        n_eval += d.astype(int)
        n_pass += (d & (np.abs(p) <= thr)).astype(int)
    mh_d = df["mhfd_defined"].to_numpy(bool)
    mh_v = df["mhfd_dist"].to_numpy(float)
    n_eval += mh_d.astype(int)
    n_pass += (mh_d & (mh_v <= MHFD_MAX_DIST * thr / 50.0)).astype(int)
    out["sig_n_eval"] = n_eval
    out["sig_n_pass"] = n_pass
    out["sig_pass"] = (n_eval > 0) & (n_pass == n_eval)
    out["strict_pass"] = out["qm_pass"] & out["qy_pass"] & out["sig_pass"]
    return out


def main() -> None:
    L = pd.read_parquet(PC.T6 / "adequacy_fold_level_long.parquet")
    L["basin_id"] = L["basin_id"].astype(np.int64)   # stored padded; normalise for joins
    if len(L) != 2 * 19116:
        raise ValueError(f"expected {2*19116} realization rows, got {len(L)}")
    seen = L[L.realization == "SEEN_canonical_dPL"].copy()
    oob = L[L.realization == "OOB_heldout"].copy()
    if len(seen) != 19116 or len(oob) != 19116:
        raise ValueError("each realization must cover all 19,116 model x basin cells")
    n_basins = L["basin_id"].nunique()
    models = sorted(L["model"].unique())
    basins = np.sort(L["basin_id"].unique())

    # ---------------- wide model x basin table ------------------------------
    key = ["model", "basin_id"]
    keep_seen = ["kge", "sig_pass", "sig_n_pass", "sig_n_evaluable", "strict_pass",
                 "sig_pass_floored6", "strict_pass_floored", "qm_pass", "qy_pass",
                 "sig_median_abs_pct", "sig10_pass", "signature_pass_fraction_verified6"]
    W = seen[key + keep_seen].rename(columns={c: f"seen_{c}" for c in keep_seen})
    W = W.merge(oob[key + ["fold"] + keep_seen].rename(
        columns={c: f"oob_{c}" for c in keep_seen + ["fold"]}), on=key, how="outer",
        validate="one_to_one")
    W["gauge_id_str"] = [str(int(b)).zfill(8) for b in W["basin_id"]]
    for k in SIG5:
        W[f"seen_defined_{k}"] = seen.set_index(key)[f"sig_defined_{k}"].reindex(
            pd.MultiIndex.from_frame(W[key])).to_numpy()
        W[f"oob_defined_{k}"] = oob.set_index(key)[f"sig_defined_{k}"].reindex(
            pd.MultiIndex.from_frame(W[key])).to_numpy()
    W.to_csv(PC.T6 / "adequacy_model_basin_seen_oob.csv", index=False)

    S = seen.set_index(key).loc[pd.MultiIndex.from_frame(W[key])]
    O = oob.set_index(key).loc[pd.MultiIndex.from_frame(W[key])]
    W = W.reset_index(drop=True)
    S = S.reset_index(drop=True)
    O = O.reset_index(drop=True)

    report = {"meta": {
        "n_models": len(models), "n_basins": int(n_basins), "n_model_basin": int(len(W)),
        "accuracy_band_delta_kge": ACC_DELTA_KGE,
        "pct_threshold_primary": 50.0, "thresholds_sensitivity": list(THRESHOLDS),
        "mhfd_threshold_day_fraction": MHFD_MAX_DIST,
        "mhfd_error_metric": "circular wrapped day-fraction distance",
        "seen_definition": "canonical dPL store (cross-pipeline seen); same-pipeline 4-fold "
                           "in-sample daily Q is NOT_AVAILABLE",
        "spieler_note": "Spieler-style adequacy analysis, not an exact replication of the paper",
    }}

    # ====================================================================== D
    # signature failure rates (before A/B/C so the reader sees the pattern first)
    rowsD = []
    for cond, D in (("SEEN", S), ("OOB", O)):
        for k in SIG5:
            d = D[f"sig_defined_{k}"].to_numpy(bool)
            p = np.abs(D[f"sig_pct_{k}"].to_numpy(float))
            fail = d & (p > 50.0)
            rowsD.append({"condition": cond, "signature": SIG_LABEL[k], "key": k,
                          "n_evaluable": int(d.sum()), "n_undefined": int((~d).sum()),
                          "n_fail": int(fail.sum()),
                          "failure_rate_among_evaluable": float(fail.sum() / max(d.sum(), 1)),
                          "per_test_failure_rate": float(fail.sum() / max(d.sum(), 1)),
                          "n_sim_exactly_zero_obs_positive": int(np.sum(
                              (D[f"sig_val_sim_{k}"].to_numpy(float) == 0)
                              & (np.abs(D[f"sig_val_obs_{k}"].to_numpy(float)) > 1e-12)))})
        d = D["mhfd_defined"].to_numpy(bool)
        v = D["mhfd_dist"].to_numpy(float)
        rowsD.append({"condition": cond, "signature": "MHFD", "key": "mhfd",
                      "n_evaluable": int(d.sum()), "n_undefined": int((~d).sum()),
                      "n_fail": int(np.sum(d & (v > MHFD_MAX_DIST))),
                      "failure_rate_among_evaluable": float(np.sum(d & (v > MHFD_MAX_DIST)) / max(d.sum(), 1)),
                      "per_test_failure_rate": float(np.sum(d & (v > MHFD_MAX_DIST)) / max(d.sum(), 1)),
                      "n_sim_exactly_zero_obs_positive": 0})
        for k in SIG4:
            d = D[f"sig_defined_{k}"].to_numpy(bool)
            p = np.abs(D[f"sig_pct_{k}"].to_numpy(float))
            fail = d & (p > 50.0)
            rowsD.append({"condition": cond, "signature": SIG_LABEL[k], "key": k,
                          "n_evaluable": int(d.sum()), "n_undefined": int((~d).sum()),
                          "n_fail": int(fail.sum()),
                          "failure_rate_among_evaluable": float(fail.sum() / max(d.sum(), 1)),
                          "per_test_failure_rate": float(fail.sum() / max(d.sum(), 1)),
                          "n_sim_exactly_zero_obs_positive": np.nan})
        rowsD.append({"condition": cond, "signature": "Qmonth", "key": "qmonth",
                      "n_evaluable": int(D["qm_n_evaluable"].sum()),
                      "n_undefined": int(D["qm_n_undefined"].sum()),
                      "n_fail": int(np.sum(~D["qm_pass"].to_numpy(bool) & (D["qm_n_evaluable"] > 0).to_numpy())),
                      "failure_rate_among_evaluable": float(
                          np.mean(~D["qm_pass"].to_numpy(bool)[(D["qm_n_evaluable"] > 0).to_numpy()])
                          if (D["qm_n_evaluable"] > 0).any() else np.nan),
                      "per_test_failure_rate": float(
                          1.0 - D["qm_n_pass"].sum() / max(D["qm_n_evaluable"].sum(), 1)),
                      "n_sim_exactly_zero_obs_positive": np.nan})
        rowsD.append({"condition": cond, "signature": "Qyear", "key": "qyear",
                      "n_evaluable": int(D["qy_n_evaluable"].sum()),
                      "n_undefined": int(D["qy_n_undefined"].sum()),
                      "n_fail": int(np.sum(~D["qy_pass"].to_numpy(bool) & (D["qy_n_evaluable"] > 0).to_numpy())),
                      "failure_rate_among_evaluable": float(
                          np.mean(~D["qy_pass"].to_numpy(bool)[(D["qy_n_evaluable"] > 0).to_numpy()])
                          if (D["qy_n_evaluable"] > 0).any() else np.nan),
                      "per_test_failure_rate": float(
                          1.0 - D["qy_n_pass"].sum() / max(D["qy_n_evaluable"].sum(), 1)),
                      "n_sim_exactly_zero_obs_positive": np.nan})
    FR = pd.DataFrame(rowsD)
    piv = FR.pivot(index="signature", columns="condition",
                   values="failure_rate_among_evaluable")
    delta = (piv["OOB"] - piv["SEEN"]).rename("delta_failure_rate_oob_minus_seen")
    FR = FR.merge(delta, left_on="signature", right_index=True, how="left")
    FR.to_csv(PC.T6 / "adequacy_signature_failure_rates.csv", index=False)
    report["signature_failure_rank_SEEN"] = (
        FR[FR.condition == "SEEN"].sort_values("failure_rate_among_evaluable", ascending=False)
        [["signature", "failure_rate_among_evaluable", "n_undefined"]].to_dict("records"))
    report["signature_failure_rank_OOB"] = (
        FR[FR.condition == "OOB"].sort_values("failure_rate_among_evaluable", ascending=False)
        [["signature", "failure_rate_among_evaluable", "n_undefined"]].to_dict("records"))

    # ====================================================================== A1
    # Spieler-style accuracy set: KGE within 0.05 of the basin's best
    A1 = []
    for cond in ("SEEN", "OOB"):
        kc = f"{cond.lower()}_kge"
        acc = np.zeros(len(W), bool)
        for b, g in W.groupby("basin_id"):
            m = g[kc].to_numpy(float)
            acc[g.index] = m >= (np.nanmax(m) - ACC_DELTA_KGE)
        W[f"{cond.lower()}_accurate"] = acc
        sub = W[acc]
        A1.append({
            "condition": cond, "n_accurate_cells": int(acc.sum()),
            "mean_accurate_per_basin": float(acc.sum() / n_basins),
            "qm_adequate_rate": float(sub[f"{cond.lower()}_qm_pass"].mean()),
            "qy_adequate_rate": float(sub[f"{cond.lower()}_qy_pass"].mean()),
            "sig_adequate_rate": float(sub[f"{cond.lower()}_sig_pass"].mean()),
            "strict_adequate_rate": float(sub[f"{cond.lower()}_strict_pass"].mean()),
            "sig_adequate_rate_floored": float(sub[f"{cond.lower()}_sig_pass_floored6"].mean()),
            "mean_sig_pass_fraction": float(sub[f"{cond.lower()}_signature_pass_fraction_verified6"].mean()),
            "mean_sig_n_pass_of6": float(sub[f"{cond.lower()}_sig_n_pass"].mean()),
        })
    A1w = pd.DataFrame(A1)
    A1w.to_csv(PC.T6 / "accurate_set_adequacy_summary.csv", index=False)
    report["A1_accurate_set"] = A1w.to_dict("records")

    # ====================================================================== A2 + C
    # frozen Seen Top-3 / Top-5 candidate sets = the k best seen-KGE models per basin
    F = pd.read_csv(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")
    kp = F.pivot(index="basin_id", columns="model",
                 values="kge_seenfolds_median4").loc[basins, models]
    order = np.argsort(-kp.to_numpy(), axis=1, kind="stable")
    bpos = {int(b): i for i, b in enumerate(basins)}
    mpos = {m: j for j, m in enumerate(models)}
    rb = W["basin_id"].map(bpos).to_numpy()
    cm = W["model"].map(mpos).to_numpy()

    def to_grid(col, dtype=bool):
        g = np.zeros((len(basins), len(models)), dtype=dtype)
        g[rb, cm] = W[col].to_numpy()
        return g

    G_sig_seen = to_grid("seen_sig_pass")
    G_sig_oob = to_grid("oob_sig_pass")
    G_sig_seen_fl = to_grid("seen_sig_pass_floored6")
    G_strict_seen = to_grid("seen_strict_pass")
    G_strict_oob = to_grid("oob_strict_pass")
    G_npass_oob = to_grid("oob_sig_n_pass", int)
    G_npass_seen = to_grid("seen_sig_n_pass", int)
    G_frac_oob = to_grid("oob_signature_pass_fraction_verified6", float)
    G_kge_seen = to_grid("seen_kge", float)
    G_kge_oob = to_grid("oob_kge", float)

    rowsC, rowsPB = [], []
    for k in (3, 5):
        tk = order[:, :k]                                     # [531, k]
        rows_i = np.arange(len(basins))[:, None]
        ok_seen = G_sig_seen[rows_i, tk].sum(axis=1)
        ok_oob = G_sig_oob[rows_i, tk].sum(axis=1)
        contains = G_sig_oob[rows_i, tk].any(axis=1)
        contains_strict = G_strict_oob[rows_i, tk].any(axis=1)
        npass_oob_i = G_npass_oob
        top1_np = npass_oob_i[np.arange(len(basins)), order[:, 0]]
        max_np = npass_oob_i.max(axis=1)
        top1_is_best = top1_np == max_np          # tie-tolerant: is the KGE winner tied for best?
        top1_strictly_worse = top1_np < max_np
        rowsC.append({
            "candidate_set": f"Seen Top-{k}", "k": k,
            "mean_candidates_before_filter": float(k),
            "mean_adequate_candidates_seen": float(ok_seen.mean()),
            "mean_adequate_candidates_oob": float(ok_oob.mean()),
            "frac_basins_0_adequate_seen": float((ok_seen == 0).mean()),
            "frac_basins_1_adequate_seen": float((ok_seen == 1).mean()),
            "frac_basins_2_adequate_seen": float((ok_seen == 2).mean()),
            "frac_basins_ge3_adequate_seen": float((ok_seen >= 3).mean()),
            "frac_basins_0_adequate_oob": float((ok_oob == 0).mean()),
            "frac_basins_1_adequate_oob": float((ok_oob == 1).mean()),
            "frac_basins_2_adequate_oob": float((ok_oob == 2).mean()),
            "frac_basins_ge3_adequate_oob": float((ok_oob >= 3).mean()),
            "frac_basins_with_ge1_oob_adequate": float(contains.mean()),
            "frac_basins_with_ge1_oob_strict_adequate": float(contains_strict.mean()),
            "frac_basins_with_ge1_seen_strict_adequate":
                float(G_strict_seen[rows_i, tk].any(axis=1).mean()),
            "kge_top1_is_tied_for_most_adequate_oob": float(top1_is_best.mean()),
            "kge_top1_strictly_less_adequate_than_best_oob": float(top1_strictly_worse.mean()),
        })
        for i, b in enumerate(basins):
            rowsPB.append({"basin_id": int(b), "gauge_id_str": str(int(b)).zfill(8), "k": k,
                           "candidates": "|".join(models[j] for j in tk[i]),
                           "n_adequate_seen": int(ok_seen[i]),
                           "n_adequate_oob": int(ok_oob[i]),
                           "contains_oob_adequate": bool(contains[i]),
                           "contains_oob_strict_adequate": bool(contains_strict[i]),
                           "kge_top1_is_tied_for_most_adequate": bool(top1_is_best[i]),
                           "kge_top1_strictly_less_adequate_than_best": bool(top1_strictly_worse[i])})
    Cw = pd.DataFrame(rowsC)
    Cw.to_csv(PC.T6 / "candidate_set_adequacy_summary.csv", index=False)
    PBC = pd.DataFrame(rowsPB)
    PBC.to_csv(PC.T6 / "candidate_set_adequacy_by_basin.csv", index=False)
    report["A2_and_C_candidate_sets"] = Cw.to_dict("records")

    # ====================================================================== B
    rowsB = []
    comps = [("sig_pass", "signature (verified 6)"),
             ("qm_pass", "Qmonth"), ("qy_pass", "Qyear"), ("strict_pass", "STRICT overall"),
             ("sig_pass_floored6", "signature (floored convention)")]
    for col, label in comps:
        a = S[col].to_numpy(bool)
        b_ = O[col].to_numpy(bool)
        pp = int(np.sum(a & b_)); pf = int(np.sum(a & ~b_))
        fp = int(np.sum(~a & b_)); ff = int(np.sum(~a & ~b_))
        both = int(np.sum(a & b_)); either = int(np.sum(a | b_))
        rowsB.append({"component": label, "key": col,
                      "seen_pass_oob_pass": pp, "seen_pass_oob_fail": pf,
                      "seen_fail_oob_pass": fp, "seen_fail_oob_fail": ff,
                      "retention_of_seen_pass": float(pp / max(pp + pf, 1)),
                      "agreement": float((pp + ff) / len(W)),
                      "jaccard": float(both / either) if either else np.nan,
                      "n_seen_pass": int(a.sum()), "n_oob_pass": int(b_.sum())})
    for k in SIG5:
        a = S[f"sig_pct_{k}"].to_numpy(float); b_ = O[f"sig_pct_{k}"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b_)
        rho = float(sstats.spearmanr(a[m], b_[m]).statistic)
        pp = int(np.sum((np.abs(a) <= 50) & m & (np.abs(b_) <= 50)))
        pf = int(np.sum((np.abs(a) <= 50) & m & (np.abs(b_) > 50)))
        fp = int(np.sum((np.abs(a) > 50) & m & (np.abs(b_) <= 50)))
        ff = int(np.sum((np.abs(a) > 50) & m & (np.abs(b_) > 50)))
        either = pp + pf + fp
        rowsB.append({"component": f"signature {SIG_LABEL[k]}", "key": k,
                      "seen_pass_oob_pass": pp, "seen_pass_oob_fail": pf,
                      "seen_fail_oob_pass": fp, "seen_fail_oob_fail": ff,
                      "retention_of_seen_pass": float(pp / max(pp + pf, 1)),
                      "agreement": float((pp + ff) / max(m.sum(), 1)),
                      "jaccard": float(pp / either) if either else np.nan,
                      "spearman_seen_vs_oob_abs_error": rho,
                      "n_seen_pass": pp + pf, "n_oob_pass": pp + fp})
    dm = np.isfinite(S["mhfd_dist"].to_numpy(float)) & np.isfinite(O["mhfd_dist"].to_numpy(float))
    a = S["mhfd_dist"].to_numpy(float); b_ = O["mhfd_dist"].to_numpy(float)
    rowsB.append({"component": "signature MHFD", "key": "mhfd",
                  "seen_pass_oob_pass": int(np.sum(dm & (a <= MHFD_MAX_DIST) & (b_ <= MHFD_MAX_DIST))),
                  "seen_pass_oob_fail": int(np.sum(dm & (a <= MHFD_MAX_DIST) & (b_ > MHFD_MAX_DIST))),
                  "seen_fail_oob_pass": int(np.sum(dm & (a > MHFD_MAX_DIST) & (b_ <= MHFD_MAX_DIST))),
                  "seen_fail_oob_fail": int(np.sum(dm & (a > MHFD_MAX_DIST) & (b_ > MHFD_MAX_DIST))),
                  "spearman_seen_vs_oob_abs_error": float(sstats.spearmanr(a[dm], b_[dm]).statistic)})
    for k in SIG4:
        a = S[f"sig_pct_{k}"].to_numpy(float); b_ = O[f"sig_pct_{k}"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b_)
        rowsB.append({"component": f"signature {SIG_LABEL[k]} (convention-unverified)", "key": k,
                      "spearman_seen_vs_oob_abs_error": float(sstats.spearmanr(a[m], b_[m]).statistic)
                      if m.sum() > 2 else np.nan})
    for col, lab in (("qm_median_abs_pct", "Qmonth median |pct|"),
                     ("qy_median_abs_pct", "Qyear median |pct|"),
                     ("sig_median_abs_pct", "signature median |pct|")):
        a = S[col].to_numpy(float); b_ = O[col].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b_)
        rowsB.append({"component": lab, "key": col,
                      "spearman_seen_vs_oob_abs_error": float(sstats.spearmanr(a[m], b_[m]).statistic)})
    Bw = pd.DataFrame(rowsB)
    Bw["retention_minus_agreement"] = Bw["retention_of_seen_pass"] - Bw["agreement"]
    Bw.to_csv(PC.T6 / "adequacy_seen_oob_transfer.csv", index=False)
    report["B_transfer"] = Bw.to_dict("records")

    # per-basin Jaccard of the adequate-model sets
    self_ok = G_sig_seen.sum(axis=1) > 0
    both = (G_sig_seen & G_sig_oob).sum(axis=1)
    either = (G_sig_seen | G_sig_oob).sum(axis=1)
    jac = np.where(either > 0, both / np.maximum(either, 1), np.nan)
    ret = both / np.maximum(G_sig_seen.sum(axis=1), 1)
    report["B_per_basin_jaccard_adequate_sets"] = {
        "median": float(np.nanmedian(jac)), "mean": float(np.nanmean(jac)),
        "q25": float(np.nanpercentile(jac, 25)), "q75": float(np.nanpercentile(jac, 75)),
        "n_basins_both_empty": int(np.sum(either == 0)),
        "median_retention_of_seen_adequate": float(np.median(ret[self_ok]))}
    pd.DataFrame({"basin_id": basins, "jaccard_seen_oob_adequate_sets": jac,
                  "retention_of_seen_adequate": ret}).to_csv(
        PC.T6 / "adequacy_per_basin_jaccard.csv", index=False)

    # ====================================================================== E
    rowsE = []
    for cond in ("SEEN", "OOB"):
        D = S if cond == "SEEN" else O
        kge = D["kge"].to_numpy(float)
        npass = D["sig_n_pass"].to_numpy(float)
        medabs = D["sig_median_abs_pct"].to_numpy(float)
        m = np.isfinite(kge) & np.isfinite(medabs)
        rowsE.append({"condition": cond, "n": int(m.sum()),
                      "spearman_KGE_vs_n_signatures_passed":
                          float(sstats.spearmanr(kge[m], npass[m]).statistic),
                      "spearman_KGE_vs_median_abs_signature_error":
                          float(sstats.spearmanr(kge[m], medabs[m]).statistic),
                      "pearson_KGE_vs_median_abs_signature_error":
                          float(np.corrcoef(kge[m], medabs[m])[0, 1])})
    Ew = pd.DataFrame(rowsE)
    Ew.to_csv(PC.T6 / "kge_vs_adequacy.csv", index=False)
    # top-1 vs the rest of the candidate set
    top1_rows = []
    for cond, Gk, Gn in (("SEEN", G_kge_seen, G_npass_seen), ("OOB", G_kge_oob, G_npass_oob)):
        top1 = np.nanargmax(Gk, axis=1)
        rows_i = np.arange(len(basins))
        top1_np = Gn[rows_i, top1]
        max_np = Gn.max(axis=1)
        rows_med = np.nanmedian(Gn, axis=1)
        top1_rows.append({"condition": cond,
                          "frac_KGE_top1_tied_for_most_adequate": float(np.mean(top1_np == max_np)),
                          "frac_KGE_top1_strictly_less_adequate_than_best":
                              float(np.mean(top1_np < max_np)),
                          "frac_KGE_top1_adequate_at_or_above_median":
                              float(np.mean(top1_np >= rows_med))})
    T1 = pd.DataFrame(top1_rows)
    T1.to_csv(PC.T6 / "kge_top1_vs_adequacy.csv", index=False)
    report["E_kge_vs_adequacy"] = {"global": Ew.to_dict("records"), "top1": T1.to_dict("records")}

    # ====================================================================== threshold sensitivity
    rowsT = []
    for thr in THRESHOLDS:
        ps = derive_pass_at(S, thr)
        po = derive_pass_at(O, thr)
        Gsig_oob_t = np.zeros((len(basins), len(models)), bool)
        Gsig_oob_t[rb, cm] = po["sig_pass"].to_numpy(bool)
        topk = {}
        for k in (3, 5):
            tk = order[:, :k]
            topk[k] = float(Gsig_oob_t[np.arange(len(basins))[:, None], tk].any(axis=1).mean())
        rowsT.append({
            "threshold_pct": thr,
            "seen_mean_signature_pass_fraction": float((ps["sig_n_pass"] / ps["sig_n_eval"].clip(lower=1)).mean()),
            "oob_mean_signature_pass_fraction": float((po["sig_n_pass"] / po["sig_n_eval"].clip(lower=1)).mean()),
            "seen_signature_adequate_rate": float(ps["sig_pass"].mean()),
            "oob_signature_adequate_rate": float(po["sig_pass"].mean()),
            "seen_strict_adequate_rate": float(ps["strict_pass"].mean()),
            "oob_strict_adequate_rate": float(po["strict_pass"].mean()),
            "top3_contains_oob_adequate": topk[3],
            "top5_contains_oob_adequate": topk[5],
        })
    Tw = pd.DataFrame(rowsT)
    Tw.to_csv(PC.T6 / "adequacy_threshold_sensitivity.csv", index=False)
    report["threshold_sensitivity"] = Tw.to_dict("records")

    with open(PC.T6 / "adequacy_analyses.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ====================================================================== console
    log("== D: signature failure rates (among evaluable) ==")
    log(FR.pivot_table(index="signature", columns="condition",
                       values="failure_rate_among_evaluable").round(4).to_string())
    log("\n== A1: Spieler-style accurate set ==")
    log(A1w.round(4).to_string(index=False))
    log("\n== A2/C: candidate sets ==")
    log(Cw.round(4).to_string(index=False))
    log("\n== B: seen->OOB transfer ==")
    log(Bw.round(4).to_string(index=False))
    log("\n== E: KGE vs adequacy ==")
    log(Ew.round(4).to_string(index=False))
    log(T1.round(4).to_string(index=False))
    log("\n== threshold sensitivity ==")
    log(Tw.round(4).to_string(index=False))
    log(f"\nwrote -> {PC.T6}")


if __name__ == "__main__":
    main()
