#!/usr/bin/env python3
"""26_adequacy_signatures.py — Spieler-style hydrological adequacy, SEEN and OOB.

WHAT THIS COMPUTES
------------------
For every (model, basin) realization, on exactly the same scored window used by the existing
KGE evaluation (1995-10-01..2010-09-30, 5479 days, NO warm-up), the Spieler & Schuetze (2024)
adequacy components:

    Qmonth   12 calendar-month climatological mean flows
    Qyear    one mean flow per water year of the evaluation window
    10 signatures: RR, BFI, Q5, Q95, LFfreq, HFfreq, LFdur, HFdur, MHFD, slopeFDC

and the Spieler percentage error  pct = 100 * (sim - obs) / obs  with |pct| <= 50% as the
acceptability threshold.

DEFINITIONS: REUSE, NOT RE-INVENTION
------------------------------------
Six signatures are taken from the frozen chapter-4 implementation
`benchmark/analysis/chapter4_multicriteria_v3_20260925/scripts/v3_common.py`
(imported, not re-implemented): rc (= RR, sum Q / sum P over valid days), bfi
(Lyne-Hollick, alpha 0.925, 3 alternating passes = the standard baseflow-package variant),
fdc_slope ((ln Q33 - ln Q66)/(0.66-0.33), i.e. exactly the paper's 33rd-66th percentile
mid-segment slope), q5, q95, mhfd (circular mean over the project's water years of the
half-flow day fraction).  Water years and precipitation also come from that module, so the
project's existing conventions are used rather than new ones.

THE ONE DELIBERATE DIVERGENCE FROM THE FROZEN MODULE
----------------------------------------------------
`v3_common.signatures_for_model` floors the flow quantiles at 1e-3 * mean(obs) before taking
logs/quantiles, and floors the *denominator* of its relative error.  This task explicitly
forbids epsilon-adjusting an observed value to keep a percentage error finite, so:

  * the frozen floored signature values are ALSO computed, for cross-checking, but
  * the adequacy percentage errors are formed from UNFLOORED signature values, and
  * an observed value that is 0 or non-finite makes that test UNDEFINED (never pass, never
    fail, always counted and reported).

The two are compared, and the number of cells where the floor changes the value is reported.

FOUR SIGNATURES ARE NOT VERIFIABLE IN THIS ENVIRONMENT
------------------------------------------------------
LFfreq/HFfreq/LFdur/HFdur have no implementation in the repository, and TOSSH is not
installed or vendored anywhere on this machine, so the paper's *event-counting* convention
(how an event is delimited, whether frequency is events/year or days/year, whether duration
is mean event length or total days) cannot be verified.  They are therefore computed as an
EXPLICITLY LABELLED extended block with the paper's thresholds (0.2 * mean(Q) for low flow,
9 * median(Q) for high flow) under two documented conventions, and they are reported
separately from the six verifiable signatures.  `TOSSH_CONVENTION_VERIFIED = False`.

DATA (all frozen, read-only; no inference, no torch)
----------------------------------------------------
  SEEN  pub benchmarkpub/analysis/analysis_seen_dpl_daily_q_reconstruction_20260924/
        artifacts/daily_q/qsim.zarr  [36, 531, 5479]  canonical dPL, all 531 basins trained
  OOB   benchmarkpub/data/oob36_20260930/qsim/<model>/fold_<k>.npy  [5479, n_fold]
  obs   Q_IC store qobs.npy [531, 5479] (verified bit-identical to obs/fold_<k>/heldout_y.npy)

KNOWN LIMITATION, RECORDED NOT WORKED AROUND
--------------------------------------------
The SEEN side is the canonical dPL store (a single all-531-basin training run per model), NOT
the same-pipeline 4-fold in-sample hydrographs (the model trained with that basin withheld).
Those 4 seen-fold daily series do not exist anywhere on disk and producing them would require
new inference, which this task forbids.  So the seen condition here is the CROSS-PIPELINE seen
used by the frozen `delta_kge_existing` / `kge_seendpl_canonical` column, and
`same_pipeline_seen_fold_realizations = NOT_AVAILABLE`.

Outputs -> $OUT6/tables/adequacy_fold_level_long.parquet
           $OUT6/tables/adequacy_model_basin_seen_oob.csv
           $OUT6/tables/adequacy_undefined_denominators.csv
           $OUT6/tables/adequacy_signature_values.npz   (compact, for re-derivation)
           $OUT6/tables/adequacy_compute_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

V3_SCRIPTS = PC.REPO / "project" / "benchmark" / "analysis" / "chapter4_multicriteria_v3_20260925" / "scripts"
TK_SCRIPTS = PC.REPO / "project" / "benchmark" / "analysis" / "topk_candidate_sets_20260925" / "scripts"
sys.path.insert(0, str(V3_SCRIPTS))
sys.path.insert(0, str(TK_SCRIPTS))
import v3_common as V  # noqa: E402  the frozen signature implementation

SEED = 20260930
PCT_THRESHOLD = 50.0          # Spieler primary; sensitivity run separately
UNDEF_ABS_EPS = 1e-12         # |obs| <= this -> percentage error undefined (documented)
NEAR_ZERO_BANDS = (1e-6, 1e-3, 1e-2)   # reported for visibility only
MOD_LF = 0.2                  # LFfreq/LFdur threshold = 0.2 * mean(Q)   (paper)
MOD_HF = 9.0                  # HFfreq/HFdur threshold = 9 * median(Q)   (paper)
MHFD_MAX_DIST = 0.25          # 50% of the max circular distance (0.5) ~ 91 days

SIG6 = list(V.SIGNATURE_NAMES)                       # ('rc','bfi','fdc_slope','q5','q95','mhfd')
SIG4 = ["lf_freq", "hf_freq", "lf_dur", "hf_dur"]    # extended, convention-unverified
SIG10 = SIG6 + SIG4
SIG_LABEL = {"rc": "RR", "bfi": "BFI", "fdc_slope": "slopeFDC", "q5": "Q5",
             "q95": "Q95", "mhfd": "MHFD", "lf_freq": "LFfreq", "hf_freq": "HFfreq",
             "lf_dur": "LFdur", "hf_dur": "HFdur"}


def log(m: str) -> None:
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# signature implementations
# --------------------------------------------------------------------------- #
def signatures_raw(q: np.ndarray, p: np.ndarray, valid: np.ndarray,
                   wy: list[int], wy_slices: list[slice]) -> dict[str, np.ndarray]:
    """The same six formulas as `v3_common.signatures_for_model`, with the flow floor REMOVED.

    Only the floor is dropped; the Lyne-Hollick filter, the quantile definitions, the fdc
    mid-segment expression and the circular water-year MHFD are byte-for-byte the frozen
    formulas.  Removing the floor means a zero/low-flow quantile can make `fdc_slope`
    non-finite, which is intended: that is a genuinely undefined observed value, not
    something to be repaired.
    """
    qf_removed = np.where(valid, q, np.nan)                   # no QFLOOR_FRACTION
    b = V._baseflow_lyne_hollick(q, valid)
    qv = np.where(valid, q, 0.0)
    pv = np.where(valid, p, 0.0)
    sum_p = pv.sum(axis=1)
    sum_q = qv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = np.where(sum_p > 0, sum_q / np.maximum(sum_p, 1e-12), np.nan)
    bv = np.where(valid, b, 0.0)
    qpos = np.where(valid, np.maximum(q, 0.0), 0.0)
    sum_qpos = qpos.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bfi = np.where(sum_qpos > 0, bv.sum(axis=1) / np.maximum(sum_qpos, 1e-12), np.nan)
    q33 = np.nanquantile(qf_removed, 1.0 - V.FDC_LO, axis=1)
    q66 = np.nanquantile(qf_removed, 1.0 - V.FDC_HI, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fdc_slope = (np.log(q33) - np.log(q66)) / (V.FDC_HI - V.FDC_LO)
    q5 = np.nanquantile(qf_removed, 1.0 - V.Q_HIGH_EXCEED / 100.0, axis=1)
    q95 = np.nanquantile(qf_removed, 1.0 - V.Q_LOW_EXCEED / 100.0, axis=1)
    # ---- MHFD: identical to the frozen implementation
    B, Tm = q.shape
    fracs = np.full((B, len(wy)), np.nan)
    for k, sl in enumerate(wy_slices):
        seg_q = np.where(valid[:, sl], q[:, sl], 0.0)
        n_valid = valid[:, sl].sum(axis=1)
        tot = seg_q.sum(axis=1)
        ok = (n_valid >= V.MIN_VALID_DAYS_WY) & (tot > 0)
        if not ok.any():
            continue
        c = np.cumsum(seg_q, axis=1)
        reached = c >= 0.5 * tot[:, None]
        d = reached.argmax(axis=1)
        frac = (d + 1) / seg_q.shape[1]
        fracs[ok, k] = frac[ok]
    ok_any = np.isfinite(fracs).any(axis=1)
    n_ok = np.isfinite(fracs).sum(axis=1)
    safe = np.where(np.isfinite(fracs), fracs, 0.0)
    re = np.where(np.isfinite(fracs), np.cos(2 * np.pi * safe), 0.0).sum(axis=1)
    im = np.where(np.isfinite(fracs), np.sin(2 * np.pi * safe), 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.arctan2(im / np.maximum(n_ok, 1), re / np.maximum(n_ok, 1)) / (2.0 * np.pi)
    mhfd = np.where(ok_any, np.mod(ang, 1.0), np.nan)
    return {"rc": rr, "bfi": bfi, "fdc_slope": fdc_slope, "q5": q5, "q95": q95, "mhfd": mhfd}


def _runs_along(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: (number of contiguous True runs, total True count)."""
    m = np.asarray(mask, dtype=bool)
    n_runs = np.zeros(m.shape[0], dtype=np.int64)
    prev = np.zeros(m.shape[0], dtype=bool)
    for t in range(m.shape[1]):
        cur = m[:, t]
        n_runs += (cur & ~prev).astype(np.int64)
        prev = cur
    return n_runs, m.sum(axis=1).astype(np.int64)


def signatures_threshold_extended(q: np.ndarray, valid: np.ndarray, wy_slices: list[slice],
                                  mean_obs: np.ndarray) -> dict[str, np.ndarray]:
    """LF/HF frequency and duration with the PAPER's thresholds under a DOCUMENTED convention.

    NOT TOSSH-verified: TOSSH is not available in this environment, so the event convention
    below is a repo-local, explicitly-stated choice and is reported as such.

      convention E (primary here): a low-flow event is a maximal contiguous run of days with
        Q < 0.2 * mean(Q); LFfreq = number of such events per water year;
        LFdur = mean event length in days.  High flow: Q > 9 * median(Q), same rules.
      convention D (reported as a sensitivity): LFfreq_frac = fraction of days below the
        threshold; LFdur_total = total days below the threshold per water year.
    """
    B = q.shape[0]
    qf = np.where(valid, q, np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(qf, axis=1)
    thr_lo = MOD_LF * mean_obs
    thr_hi = MOD_HF * np.where(np.isfinite(med), med, np.nan)
    lo = np.where(valid, q < thr_lo[:, None], False)
    hi = np.where(valid & np.isfinite(thr_hi)[:, None], q > np.broadcast_to(thr_hi[:, None], q.shape), False)
    n_lo, tot_lo = _runs_along(lo)
    n_hi, tot_hi = _runs_along(hi)
    n_wy = float(len(wy_slices))
    n_days_valid = np.maximum(valid.sum(axis=1), 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lf_freq = n_lo / n_wy
        hf_freq = n_hi / n_wy
        lf_dur = np.where(n_lo > 0, tot_lo / np.maximum(n_lo, 1), 0.0)
        hf_dur = np.where(n_hi > 0, tot_hi / np.maximum(n_hi, 1), 0.0)
        lf_frac_frac = tot_lo / n_days_valid
        hf_frac_frac = tot_hi / n_days_valid
        lf_dur_total = tot_lo / n_wy
        hf_dur_total = tot_hi / n_wy
    return {"lf_freq": lf_freq, "hf_freq": hf_freq, "lf_dur": lf_dur, "hf_dur": hf_dur,
            "lf_freq_frac": lf_frac_frac, "hf_freq_frac": hf_frac_frac,
            "lf_dur_total": lf_dur_total, "hf_dur_total": hf_dur_total}


def monthly_means(q: np.ndarray, valid: np.ndarray, month_idx: np.ndarray) -> np.ndarray:
    """[B, 12] climatological mean flow per calendar month over valid days."""
    B = q.shape[0]
    out = np.full((B, 12), np.nan)
    for m in range(12):
        sel = month_idx == m
        v = valid[:, sel]
        cnt = v.sum(axis=1)
        tot = np.where(v, q[:, sel], 0.0).sum(axis=1)
        out[:, m] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


def yearly_means(q: np.ndarray, valid: np.ndarray, wy_slices: list[slice]) -> np.ndarray:
    """[B, n_wy] mean flow per water year over valid days."""
    B = q.shape[0]
    out = np.full((B, len(wy_slices)), np.nan)
    for k, sl in enumerate(wy_slices):
        v = valid[:, sl]
        cnt = v.sum(axis=1)
        tot = np.where(v, q[:, sl], 0.0).sum(axis=1)
        out[:, k] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


def pct_error(sim: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(100*(sim-obs)/obs, defined_mask).  Observed zero/non-finite -> undefined, never fixed."""
    sim = np.asarray(sim, np.float64)
    obs = np.asarray(obs, np.float64)
    defined = np.isfinite(obs) & (np.abs(obs) > UNDEF_ABS_EPS) & np.isfinite(sim)
    out = np.full(obs.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[defined] = 100.0 * (sim[defined] - obs[defined]) / obs[defined]
    return out, defined


def component_summary(pct: np.ndarray, defined: np.ndarray, thr: float):
    """(all-pass, n_pass, n_evaluable, n_undefined, median|pct|, max|pct|) over the last axis."""
    n_eval = defined.sum(axis=1)
    a = np.where(defined, np.abs(pct), np.nan)
    n_pass = (a <= thr).sum(axis=1)
    passed = (n_eval > 0) & (n_pass == n_eval)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(a, axis=1)
        mx = np.nanmax(np.where(defined, np.abs(pct), np.nan), axis=1)
    return passed, n_pass, n_eval, (defined.shape[1] - n_eval), med, mx


def main() -> None:
    PC.T6.mkdir(parents=True, exist_ok=True)

    # ---------------- shared inputs -----------------------------------------
    root_ic = PC.Q_IC_STORE
    basins = [str(int(x)).zfill(8) for x in np.load(root_ic / "basin_ids.npy")]
    obs = np.asarray(np.load(root_ic / "qobs.npy"), np.float64)          # [531, 5479]
    valid = np.isfinite(obs)
    if obs.shape != (531, V.N_TEST_DAYS):
        raise ValueError(f"obs shape {obs.shape} != (531, {V.N_TEST_DAYS})")
    P = V.load_precip(basins)                                            # [531, 5479] mm/day
    mean_obs = np.nanmean(obs, axis=1)
    wy, wy_slices = V.water_year_slices()
    dates = V.test_dates()
    month_idx = np.asarray(pd.DatetimeIndex(dates).month) - 1            # 0..11
    if month_idx.size != V.N_TEST_DAYS:
        raise ValueError("month index length mismatch")

    models = sorted(pd.read_csv(PC.OUT1 / "remote_inventory.csv")["model"].unique())
    if len(models) != 36:
        raise ValueError(f"expected 36 models, got {len(models)}")

    # frozen fold assignment: basin -> held-out fold, and fold -> basin ids (per model dir)
    # frozen same-pipeline table: carries both KGE partners used by the adequacy analysis
    merged = pd.read_csv(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")[
        ["model", "basin_id", "heldout_fold", "kge_heldout", "kge_seendpl_canonical"]]
    bidx = {b: i for i, b in enumerate(basins)}
    gint = [int(b) for b in basins]
    kge_seen_p = merged.pivot(index="basin_id", columns="model",
                              values="kge_seendpl_canonical").loc[gint, models]
    kge_oob_p = merged.pivot(index="basin_id", columns="model",
                             values="kge_heldout").loc[gint, models]
    fold_p = merged.pivot(index="basin_id", columns="model",
                          values="heldout_fold").loc[gint, models]
    if kge_seen_p.isna().any().any() or kge_oob_p.isna().any().any():
        raise ValueError("KGE pivot contains NaN")

    # ---------------- observed side (identical for every model) -------------
    log("computing observed signatures ...")
    obs_sig = signatures_raw(obs, P, valid, wy, wy_slices)
    obs_sig_floored = V.signatures_for_model(obs.copy(), P, valid, mean_obs, wy, wy_slices)
    obs_ext = signatures_threshold_extended(obs, valid, wy_slices, mean_obs)
    obs_month = monthly_means(obs, valid, month_idx)
    obs_year = yearly_means(obs, valid, wy_slices)

    undef_rows = []
    for s in SIG6:
        n_undef = int((~np.isfinite(obs_sig[s])).sum())
        near = {f"n_obs_abs_lt_{b:g}": int((np.isfinite(obs_sig[s]) & (np.abs(obs_sig[s]) < b)).sum())
                for b in NEAR_ZERO_BANDS}
        undef_rows.append({"component": SIG_LABEL[s], "key": s, "n_obs_undefined": n_undef,
                           "n_obs_undefined_or_nonfinite": n_undef,
                           "n_obs_exactly_zero": int((obs_sig[s] == 0).sum()), **near})
    for s in SIG4:
        undef_rows.append({"component": SIG_LABEL[s], "key": s,
                           "n_obs_undefined": int((~np.isfinite(obs_ext[s])).sum()),
                           "n_obs_undefined_or_nonfinite": int((~np.isfinite(obs_ext[s])).sum()),
                           "n_obs_exactly_zero": int((obs_ext[s] == 0).sum()),
                           **{f"n_obs_abs_lt_{b:g}": int((np.isfinite(obs_ext[s]) & (np.abs(obs_ext[s]) < b)).sum())
                              for b in NEAR_ZERO_BANDS}})
    undef_rows.append({"component": "Qmonth", "key": "qmonth",
                       "n_obs_undefined": int((~np.isfinite(obs_month)).sum()),
                       "n_obs_undefined_or_nonfinite": int((~np.isfinite(obs_month)).sum()),
                       "n_obs_exactly_zero": int((obs_month == 0).sum()),
                       **{f"n_obs_abs_lt_{b:g}": int((np.isfinite(obs_month) & (np.abs(obs_month) < b)).sum())
                          for b in NEAR_ZERO_BANDS}})
    undef_rows.append({"component": "Qyear", "key": "qyear",
                       "n_obs_undefined": int((~np.isfinite(obs_year)).sum()),
                       "n_obs_undefined_or_nonfinite": int((~np.isfinite(obs_year)).sum()),
                       "n_obs_exactly_zero": int((obs_year == 0).sum()),
                       **{f"n_obs_abs_lt_{b:g}": int((np.isfinite(obs_year) & (np.abs(obs_year) < b)).sum())
                          for b in NEAR_ZERO_BANDS}})

    # ---------------- model loop -------------------------------------------
    zstore = zarr.open(str(PC.Q_DP_STORE / "qsim.zarr"), "r")
    zmodels = [str(x) for x in np.load(PC.Q_DP_STORE / "model_ids.npy")]
    zbasins = [str(int(x)).zfill(8) for x in np.load(PC.Q_DP_STORE / "basin_ids.npy")]
    zb_idx = [zbasins.index(b) for b in basins]

    rows, sig_store, floor_diff = [], {}, {s: 0 for s in SIG6}
    for mi, model in enumerate(models):
        # ---- SEEN: canonical dPL store, all 531 basins
        q_seen = np.asarray(zstore[zmodels.index(model)], np.float64)[zb_idx]      # [531, 5479]
        for s in SIG6:
            a, b_ = obs_sig_floored[s], obs_sig[s]
            floor_diff[s] += int(np.sum(~np.isclose(a, b_, rtol=0, atol=1e-12, equal_nan=True)))
        s6 = signatures_raw(q_seen, P, valid, wy, wy_slices)
        s6f = V.signatures_for_model(q_seen, P, valid, mean_obs, wy, wy_slices)
        s4 = signatures_threshold_extended(q_seen, valid, wy_slices, mean_obs)
        mth = monthly_means(q_seen, valid, month_idx)
        yr = yearly_means(q_seen, valid, wy_slices)
        sig_store.setdefault("seen_sig6", []).append(np.stack([s6[s] for s in SIG6], axis=1))
        sig_store.setdefault("seen_sig4", []).append(np.stack([s4[s] for s in SIG4], axis=1))
        sig_store.setdefault("seen_sig4_alt", []).append(
            np.stack([s4[k] for k in ("lf_freq_frac", "hf_freq_frac",
                                      "lf_dur_total", "hf_dur_total")], axis=1))
        sig_store.setdefault("seen_month", []).append(mth)
        sig_store.setdefault("seen_year", []).append(yr)
        band = kge_seen_p[model].to_numpy(np.float64)
        fidx_all = np.arange(len(basins))
        _emit(rows, model, basins, fidx_all, "SEEN_canonical_dPL", -1, band,
              obs_sig, obs_ext, obs_month, obs_year, s6, s4, mth, yr, s6f,
              obs_sig_floored, valid)

        # ---- OOB: held-out fold hydrographs
        for k in range(5):
            p = _qsim_path(model, k)
            fold_ids = _fold_ids(model, k)
            q = np.asarray(np.load(p), np.float64).T                       # [n_fold, 5479]
            if q.shape != (len(fold_ids), V.N_TEST_DAYS):
                raise ValueError(f"{model} fold {k}: qsim {q.shape} vs ids {len(fold_ids)}")
            # fold identity must agree with the frozen assignment
            expect = fold_p[model].loc[fold_ids].to_numpy(int)
            if not np.all(expect == k):
                raise ValueError(f"{model} fold {k}: basin ids disagree with frozen fold column")
            fidx = np.asarray([bidx[str(int(b)).zfill(8)] for b in fold_ids])
            v = valid[fidx]
            s6 = signatures_raw(q, P[fidx], v, wy, wy_slices)
            s6f = V.signatures_for_model(q, P[fidx], v, mean_obs[fidx], wy, wy_slices)
            s4 = signatures_threshold_extended(q, v, wy_slices, mean_obs[fidx])
            mth = monthly_means(q, v, month_idx)
            yr = yearly_means(q, v, wy_slices)
            band = kge_oob_p[model].loc[fold_ids].to_numpy(np.float64)
            _emit(rows, model, basins, fidx, "OOB_heldout", k, band,
                  obs_sig, obs_ext, obs_month, obs_year, s6, s4, mth, yr, s6f,
                  obs_sig_floored, v)
        if (mi + 1) % 6 == 0:
            log(f"  {mi+1}/{len(models)} models done")

    L = pd.concat(rows, ignore_index=True)
    L.to_parquet(PC.T6 / "adequacy_fold_level_long.parquet", index=False)

    np.savez_compressed(
        PC.T6 / "adequacy_signature_values.npz",
        models=np.asarray(models), basins=np.asarray(basins),
        obs_sig6=np.stack([obs_sig[s] for s in SIG6], axis=1),
        obs_sig4=np.stack([obs_ext[s] for s in SIG4], axis=1),
        obs_month=obs_month, obs_year=obs_year,
        seen_sig6=np.stack(sig_store["seen_sig6"]), seen_sig4=np.stack(sig_store["seen_sig4"]),
        seen_sig4_alt=np.stack(sig_store["seen_sig4_alt"]),
        seen_month=np.stack(sig_store["seen_month"]), seen_year=np.stack(sig_store["seen_year"]),
        sig6=SIG6, sig4=SIG4, sig4_alt=["lf_freq_frac", "hf_freq_frac", "lf_dur_total", "hf_dur_total"],
        water_years=np.asarray(wy),
    )
    pd.DataFrame(undef_rows).to_csv(PC.T6 / "adequacy_undefined_denominators.csv", index=False)

    summary = {
        "n_rows_fold_level": int(len(L)),
        "n_models": len(models), "n_basins": len(basins), "n_days": int(V.N_TEST_DAYS),
        "scored_window": [str(V.TEST_START), str(V.TEST_END)],
        "water_years": [int(wy[0]), int(wy[-1])], "n_water_years": len(wy),
        "pct_threshold_primary": PCT_THRESHOLD,
        "mhfd_max_distance_day_fraction": MHFD_MAX_DIST,
        "mhfd_error_metric": "circular wrapped day-fraction distance (percentage error invalid)",
        "undef_abs_eps": UNDEF_ABS_EPS,
        "verifiable_signatures": SIG6, "convention_unverified_signatures": SIG4,
        "TOSSH_CONVENTION_VERIFIED": False,
        "same_pipeline_seen_fold_realizations": "NOT_AVAILABLE (requires new inference)",
        "seen_source": str(PC.Q_DP_STORE / "qsim.zarr"),
        "oob_source": str(PC.DATA_STORE / "qsim"),
        "floor_diff_vs_frozen_impl": floor_diff,
        "thresholds": {"LF": f"{MOD_LF} * mean(Q)", "HF": f"{MOD_HF} * median(Q)",
                       "fdc_segment": [V.FDC_LO, V.FDC_HI], "bfi": f"Lyne-Hollick a={V.BFI_ALPHA}, {V.BFI_PASSES} passes"},
    }
    (PC.T6 / "adequacy_compute_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"fold-level rows: {len(L)}")
    log(f"undefined observed cells: " + ", ".join(
        f"{r['component']}={r['n_obs_undefined']}" for r in undef_rows if r['n_obs_undefined']))
    log(f"floor-vs-unfloored differences (obs): {floor_diff}")
    log(f"wrote -> {PC.T6}")


def _qsim_path(model: str, k: int):
    """Path of the frozen held-out-fold simulated discharge for one model/fold."""
    return PC.DATA_STORE / "qsim" / model / f"fold_{k}.npy"


def _fold_ids(model: str, k: int):
    cands = [PC.DATA_STORE / "runs" / model / f"fold_{k}" / "heldout_basin_ids.txt",
             PC.EXISTING8_RUNS / model / f"fold_{k}" / "heldout_basin_ids.txt"]
    for c in cands:
        if c.exists():
            return [int(x) for x in c.read_text().split()]
    raise FileNotFoundError(f"no heldout_basin_ids.txt for {model} fold {k}")


def mhfd_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular day-fraction distance in [0, 0.5] between two mean half-flow dates.

    MHFD is a timing metric on a circle, so a percentage error is invalid across the wrap
    point; the frozen chapter-4 implementation uses this wrapped distance and so do we.
    `MHFD_MAX_DIST = 0.25` day-fraction is the "50% of the maximum possible distance"
    analogue of the paper's +/-50%, i.e. about 91 days of a 365-day water year.
    """
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return np.minimum(d, 1.0 - d)


def sig6_adequacy(sim: dict, obs: dict, fidx, thr: float):
    """The 6-signature adequacy: 5 percentage-error tests (RR, BFI, slopeFDC, Q5, Q95)
    plus the circular MHFD distance test.  Returns the aggregate pass statistics."""
    n = len(fidx)
    names5 = ["rc", "bfi", "fdc_slope", "q5", "q95"]
    p5 = np.full((n, 5), np.nan)
    d5 = np.zeros((n, 5), bool)
    for j, k in enumerate(names5):
        p5[:, j], d5[:, j] = pct_error(sim[k], obs[k][fidx])
    mh_sim = np.asarray(sim["mhfd"], np.float64)
    mh_obs = obs["mhfd"][fidx]
    mh_def = np.isfinite(mh_sim) & np.isfinite(mh_obs)
    mh_dist = np.where(mh_def, mhfd_distance(mh_sim, mh_obs), np.nan)
    n_eval = d5.sum(axis=1) + mh_def.astype(int)
    a = np.where(d5, np.abs(p5), np.nan)
    n_pass = (a <= thr).sum(axis=1) + (mh_def & (mh_dist <= MHFD_MAX_DIST)).astype(int)
    passed = (n_eval > 0) & (n_pass == n_eval)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(a, axis=1)
        mx = np.nanmax(a, axis=1)
    return {"pass": passed, "n_pass": n_pass, "n_eval": n_eval,
            "n_undef": (6 - n_eval), "med_abs_pct": med, "max_abs_pct": mx,
            "mhfd_dist": mh_dist, "mhfd_dist_days": mh_dist * 365.25, "mhfd_defined": mh_def,
            "p5": p5, "d5": d5}


def sig10_adequacy(sim6: dict, obs6: dict, sim4: dict, obs4: dict, fidx, thr: float):
    """Extended 10-signature adequacy: the 6 above plus 4 convention-unverified signatures."""
    n = len(fidx)
    base = sig6_adequacy(sim6, obs6, fidx, thr)
    names4 = ["lf_freq", "hf_freq", "lf_dur", "hf_dur"]
    p4 = np.full((n, 4), np.nan)
    d4 = np.zeros((n, 4), bool)
    for j, k in enumerate(names4):
        p4[:, j], d4[:, j] = pct_error(sim4[k], obs4[k][fidx])
    n_eval = base["n_eval"] + d4.sum(axis=1)
    a4 = np.where(d4, np.abs(p4), np.nan)
    n_pass = base["n_pass"] + (a4 <= thr).sum(axis=1)
    passed = (n_eval > 0) & (n_pass == n_eval)
    with np.errstate(invalid="ignore"):
        med10 = np.nanmedian(np.concatenate([base["p5"], a4], axis=1), axis=1)
    return {"pass": passed, "n_pass": n_pass, "n_eval": n_eval,
            "n_undef": (10 - n_eval), "med_abs_pct": med10, "p4": p4, "d4": d4, "names4": names4}


def _emit(rows, model, basins, fidx, realization, fold, kge, obs_sig, obs_ext,
          obs_month, obs_year, s6, s4, mth, yr, s6f, obs_sig_floored, valid_sub):
    """Append one realization block (one row per basin) to the long table."""
    n = len(fidx)
    rec = {"model": [model] * n, "basin_id": [basins[i] for i in fidx],
           "realization": [realization] * n, "fold": [fold] * n, "kge": kge}
    pm, dm = pct_error(mth, obs_month[fidx])
    pm_p, pm_np_, pm_ne, pm_nu, pm_med, pm_mx = component_summary(pm, dm, PCT_THRESHOLD)
    py, dy = pct_error(yr, obs_year[fidx])
    py_p, py_np_, py_ne, py_nu, py_med, py_mx = component_summary(py, dy, PCT_THRESHOLD)

    a6 = sig6_adequacy(s6, obs_sig, fidx, PCT_THRESHOLD)
    a6f = sig6_adequacy(s6f, obs_sig_floored, fidx, PCT_THRESHOLD)
    a10 = sig10_adequacy(s6, obs_sig, s4, obs_ext, fidx, PCT_THRESHOLD)

    rec.update({
        "qm_pass": pm_p, "qm_n_pass": pm_np_, "qm_n_evaluable": pm_ne, "qm_n_undefined": pm_nu,
        "qm_median_abs_pct": pm_med, "qm_max_abs_pct": pm_mx,
        "qy_pass": py_p, "qy_n_pass": py_np_, "qy_n_evaluable": py_ne, "qy_n_undefined": py_nu,
        "qy_median_abs_pct": py_med, "qy_max_abs_pct": py_mx,
        "sig_pass": a6["pass"], "sig_n_pass": a6["n_pass"], "sig_n_evaluable": a6["n_eval"],
        "sig_n_undefined": a6["n_undef"], "sig_median_abs_pct": a6["med_abs_pct"],
        "sig_max_abs_pct": a6["max_abs_pct"],
        "mhfd_dist": a6["mhfd_dist"], "mhfd_dist_days": a6["mhfd_dist_days"],
        "mhfd_defined": a6["mhfd_defined"],
        "sig10_pass": a10["pass"], "sig10_n_pass": a10["n_pass"],
        "sig10_n_evaluable": a10["n_eval"], "sig10_n_undefined": a10["n_undef"],
        "sig10_median_abs_pct": a10["med_abs_pct"],
        "strict_pass": pm_p & py_p & a6["pass"],
        "strict10_pass": pm_p & py_p & a10["pass"],
        "n_undefined_total": pm_nu + py_nu + a10["n_undef"],
        "signature_pass_fraction_verified6": np.where(a6["n_eval"] > 0,
                                                      a6["n_pass"] / np.maximum(a6["n_eval"], 1), np.nan),
        "signature_pass_fraction_extended10": np.where(a10["n_eval"] > 0,
                                                       a10["n_pass"] / np.maximum(a10["n_eval"], 1), np.nan),
        # labelled sensitivity: the frozen module's floored convention
        "sig_pass_floored6": a6f["pass"], "sig_n_pass_floored6": a6f["n_pass"],
        "sig_n_evaluable_floored6": a6f["n_eval"], "sig_median_abs_pct_floored6": a6f["med_abs_pct"],
        "mhfd_dist_floored6": a6f["mhfd_dist"],
        "signature_pass_fraction_floored6": np.where(a6f["n_eval"] > 0,
                                                     a6f["n_pass"] / np.maximum(a6f["n_eval"], 1), np.nan),
        "strict_pass_floored": pm_p & py_p & a6f["pass"],
    })
    # per-signature detail
    for j, k in enumerate(["rc", "bfi", "fdc_slope", "q5", "q95"]):
        rec[f"sig_val_sim_{k}"] = s6[k]
        rec[f"sig_val_obs_{k}"] = obs_sig[k][fidx]
        rec[f"sig_pct_{k}"] = a6["p5"][:, j]
        rec[f"sig_defined_{k}"] = a6["d5"][:, j]
    rec["sig_val_sim_mhfd"] = s6["mhfd"]
    rec["sig_val_obs_mhfd"] = obs_sig["mhfd"][fidx]
    rec["sig_pct_mhfd"] = np.nan          # undefined by design: circular metric, see mhfd_dist
    rec["sig_defined_mhfd"] = a6["mhfd_defined"]
    for j, k in enumerate(a10["names4"]):
        rec[f"sig_val_sim_{k}"] = s4[k]
        rec[f"sig_val_obs_{k}"] = obs_ext[k][fidx]
        rec[f"sig_pct_{k}"] = a10["p4"][:, j]
        rec[f"sig_defined_{k}"] = a10["d4"][:, j]
    for k in ("lf_freq_frac", "hf_freq_frac", "lf_dur_total", "hf_dur_total"):
        rec[f"sig_val_sim_{k}"] = s4[k]
        rec[f"sig_val_obs_{k}"] = obs_ext[k][fidx]
    rows.append(pd.DataFrame(rec))


if __name__ == "__main__":
    main()
