#!/usr/bin/env python3
"""12_winner_stability.py — Analysis 3: basin-wise winner stability with near-tie control.

The headline "strict basin-wise winner retention = 128/531 (24.1%)" cannot be read as
"per-basin optima do not transfer" until we know how much of the change is near-tie /
seed noise.  This script answers that in three steps.

Step 1  WINNER MARGINS.  Per basin, in the seen and in the held-out (OOB) ranking:
        best / 2nd / 3rd KGE, top1-top2 and top1-top3 margins, and how many models sit
        within a labelled epsilon of the best.  This measures how many basins are
        near-ties to begin with.

Step 2  THRESHOLD SENSITIVITY (descriptive only — NOT an equivalence-set definition,
        and no delta is frozen anywhere):
          * strict top-1 retention;
          * retention restricted to basins that are "decided" (margin >= tau) in BOTH
            regimes, for a grid of labelled tau;
          * for the basins whose winner changed, the OOB KGE gap between the old and
            the new winner (are the swaps cheap?);
          * the OOB rank of the seen winner, and whether the seen winner stays inside
            the OOB top-k.

Step 3  RANDOMNESS BASELINE.  The repository holds a same-paradigm seed experiment:
        the canonical dPL v2 runs (seed 42) plus independent retrainings at seeds 123
        and 2026, for the 8 `existing8` models, each with a per-basin test KGE over the
        same 531 basins.  That gives a seed-only winner-instability baseline within a
        matched 8-model pool.  The seen->OOB retention is then recomputed on the SAME
        8-model pool so the two numbers are pool-matched.

NOTHING is trained, no checkpoint is touched, and no candidate/equivalence/oracle/
G_set quantity is produced.  `tau` values are labelled descriptive sensitivities.

Outputs -> $OUT2/tables/{winner_margin_summary,winner_retention_sensitivity,
                         seed_noise_winner_baseline,winner_matched_pool_comparison}.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO = Path("/home/jingxin/code/dmg-research")
BENCH = REPO / "project" / "benchmark"
PUB = REPO / "project" / "benchmarkpub"
OUT1 = PUB / "analysis" / "chapter4_oob36_prelim_20260930"
OUT2 = OUT1 / "pub_next_stage_20260930"
T2 = OUT2 / "tables"
MERGED = OUT1 / "tables" / "seen_ic_oob_merged.parquet"
INSAMPLE = T2 / "pub_insample_seen_kge.csv"

EXISTING8 = ["alpine2", "hbv96", "xinanjiang", "newzealand2", "ihacres", "us1", "mopex4", "hillslope"]
SEEDS = [42, 123, 2026]
SEED_DIRS = {
    42: BENCH / "results" / "dpl_canonical_v2_20260831" / "runs",
    123: BENCH / "manuscript" / "r3" / "dpl_multiseed_sensitivity_20260914" / "runs",
    2026: BENCH / "manuscript" / "r3" / "dpl_multiseed_sensitivity_20260914" / "runs",
}
TAUS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SEED = 20260930


def log(m: str) -> None:
    print(m, flush=True)


def winners_and_margins(scores: pd.DataFrame) -> pd.DataFrame:
    """scores: rows = basins, cols = models. Returns best/2nd/3rd, margins, winner."""
    v = np.sort(scores.to_numpy(np.float64), axis=1)[:, ::-1]
    win = scores.idxmax(axis=1)
    return pd.DataFrame({
        "basin_id": scores.index.to_numpy(),
        "winner": win.to_numpy(),
        "best": v[:, 0], "second": v[:, 1], "third": v[:, 2],
        "margin12": v[:, 0] - v[:, 1],
        "margin13": v[:, 0] - v[:, 2],
    }, index=scores.index)


def load_seed_matrix() -> pd.DataFrame:
    """(model, seed) -> per-basin test KGE, seen-basin canonical dPL regime."""
    rows, missing = {}, []
    for model in EXISTING8:
        for seed in SEEDS:
            base = SEED_DIRS[seed]
            d = base / (f"{model}_seed_{seed}" if seed != 42 else model)
            f = d / "basin_test_kge.csv"
            if not f.exists():
                missing.append(str(f))
                continue
            t = pd.read_csv(f)
            idc, kc = t.columns[0], ("kge" if "kge" in t.columns else t.columns[-1])
            rows[(model, seed)] = pd.Series(t[kc].to_numpy(float),
                                            index=t[idc].astype(np.int64).to_numpy())
    if missing:
        log(f"   MISSING seed files ({len(missing)}): " + "; ".join(missing[:4]))
    return pd.DataFrame(rows), missing


def main() -> None:
    T2.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MERGED)
    if len(df) != 19116:
        raise ValueError(f"expected 19,116 rows, got {len(df)}")
    ids = np.sort(df["gauge_id"].unique())
    models = sorted(df["model"].unique())

    seen = df.pivot(index="gauge_id", columns="model", values="kge_seen_dpl").loc[ids, models]
    oob = df.pivot(index="gauge_id", columns="model", values="kge_oob").loc[ids, models]
    if seen.isna().any().any() or oob.isna().any().any():
        raise ValueError("pivot produced NaN")

    report = {"meta": {
        "seed": SEED, "n_basins": int(len(ids)), "n_models": int(len(models)),
        "seen_definition": "canonical dPL v2 seed 42, seen-basin (KGE_dPL of "
                           "results/ic_dpl_seenbasin_formal_20260901)",
        "heldout_definition": "OOB fold model where the basin is held out (kge_oob)",
        "tau_grid": TAUS,
        "tau_status": "DESCRIPTIVE sensitivity grid only; no equivalence delta is frozen",
    }}

    # ---------------------------------------------------------------- step 1
    ms = winners_and_margins(seen)
    mo = winners_and_margins(oob)
    M = pd.DataFrame({
        "basin_id": ids,
        "seen_winner": ms["winner"], "oob_winner": mo["winner"],
        "seen_best": ms["best"], "oob_best": mo["best"],
        "seen_second": ms["second"], "oob_second": mo["second"],
        "seen_third": ms["third"], "oob_third": mo["third"],
        "seen_margin12": ms["margin12"], "oob_margin12": mo["margin12"],
        "seen_margin13": ms["margin13"], "oob_margin13": mo["margin13"],
    })
    M["same_winner"] = M["seen_winner"] == M["oob_winner"]
    M["winner_changed"] = ~M["same_winner"]

    # how many models within eps of the best (near-tie breadth)
    for eps in (0.01, 0.02, 0.05):
        M[f"seen_n_within_{eps}"] = (seen.to_numpy() >= ms["best"].to_numpy()[:, None] - eps).sum(axis=1)
        M[f"oob_n_within_{eps}"] = (oob.to_numpy() >= mo["best"].to_numpy()[:, None] - eps).sum(axis=1)

    # OOB gap between the outgoing and incoming winner on the basins that changed
    oob_arr = oob.to_numpy(np.float64)
    col_of = {m: i for i, m in enumerate(models)}
    gap = np.zeros(len(ids), dtype=np.float64)
    rank_of_seen_winner = np.zeros(len(ids), dtype=np.int64)
    sw_arr = M["seen_winner"].to_numpy()
    ow_arr = M["oob_winner"].to_numpy()
    changed = M["winner_changed"].to_numpy()
    for i in range(len(ids)):
        sw = sw_arr[i]
        if changed[i]:
            gap[i] = float(oob_arr[i, col_of[ow_arr[i]]] - oob_arr[i, col_of[sw]])
        rank_of_seen_winner[i] = int(1 + np.sum(oob_arr[i] > oob_arr[i, col_of[sw]]))
    M["oob_gap_old_new_winner"] = gap
    M["oob_rank_of_seen_winner"] = rank_of_seen_winner
    M.to_csv(T2 / "winner_margin_summary.csv", index=False)

    ch = M[M["winner_changed"]]
    report["step1_margins"] = {
        "strict_retention": int(M["same_winner"].sum()),
        "strict_retention_fraction": float(M["same_winner"].mean()),
        "n_changed": int(M["winner_changed"].sum()),
        "seen_margin12": {k: float(v) for k, v in {
            "median": M["seen_margin12"].median(), "q25": M["seen_margin12"].quantile(.25),
            "q75": M["seen_margin12"].quantile(.75), "min": M["seen_margin12"].min()}.items()},
        "oob_margin12": {k: float(v) for k, v in {
            "median": M["oob_margin12"].median(), "q25": M["oob_margin12"].quantile(.25),
            "q75": M["oob_margin12"].quantile(.75), "min": M["oob_margin12"].min()}.items()},
        "gap_old_new_winner_on_changed_basins": {
            "median": float(ch["oob_gap_old_new_winner"].median()),
            "q25": float(ch["oob_gap_old_new_winner"].quantile(.25)),
            "q75": float(ch["oob_gap_old_new_winner"].quantile(.75)),
            "max": float(ch["oob_gap_old_new_winner"].max()),
            "frac_gap_lt_0p01": float((ch["oob_gap_old_new_winner"] < 0.01).mean()),
            "frac_gap_lt_0p02": float((ch["oob_gap_old_new_winner"] < 0.02).mean()),
            "frac_gap_lt_0p05": float((ch["oob_gap_old_new_winner"] < 0.05).mean()),
        },
        "oob_rank_of_seen_winner": {
            "median": float(np.median(rank_of_seen_winner)),
            "frac_rank1": float(np.mean(np.array(rank_of_seen_winner) == 1)),
            "frac_rank_le3": float(np.mean(np.array(rank_of_seen_winner) <= 3)),
            "frac_rank_le5": float(np.mean(np.array(rank_of_seen_winner) <= 5)),
            "frac_rank_gt10": float(np.mean(np.array(rank_of_seen_winner) > 10)),
        },
        "near_tie_breadth": {
            f"seen_median_n_within_{e}": float(M[f"seen_n_within_{e}"].median()) for e in (0.01, 0.02, 0.05)
        } | {f"oob_median_n_within_{e}": float(M[f"oob_n_within_{e}"].median()) for e in (0.01, 0.02, 0.05)},
    }

    # ---------------------------------------------------------------- step 2
    sens = []
    n_all = len(M)
    for tau in TAUS:
        dec = (M["seen_margin12"] >= tau) & (M["oob_margin12"] >= tau)
        d = M[dec]
        row = {
            "tau_abs_kge_margin": tau,
            "n_basins_undecided_seen": int((M["seen_margin12"] < tau).sum()),
            "n_basins_undecided_oob": int((M["oob_margin12"] < tau).sum()),
            "n_basins_decided_both": int(dec.sum()),
            "frac_basins_decided_both": float(dec.mean()),
            "strict_retention_all_basins": float(M["same_winner"].mean()),
            "strict_retention_decided_both": float(d["same_winner"].mean()) if len(d) else None,
        }
        for k in (1, 2, 3, 5, 10):
            row[f"seen_winner_in_oob_top{k}"] = float((M["oob_rank_of_seen_winner"] <= k).mean())
        sens.append(row)
    S = pd.DataFrame(sens)
    S.to_csv(T2 / "winner_retention_sensitivity.csv", index=False)
    report["step2_threshold_sensitivity"] = S.to_dict(orient="records")
    report["step2_note"] = (
        "tau is an ABSOLUTE KGE top1-top2 margin used only to label near-ties. It is a "
        "descriptive sensitivity grid, not a frozen equivalence delta, and no candidate "
        "or equivalence set was constructed from it.")

    # ---------------------------------------------------------------- step 3
    seedmat, missing = load_seed_matrix()
    if seedmat.shape[1] == 0:
        pd.DataFrame().to_csv(T2 / "seed_noise_winner_baseline.csv", index=False)
        report["step3_seed_baseline"] = {"status": "NOT_AVAILABLE",
                                         "reason": "no per-basin seed files found"}
    else:
        got_models = sorted({c[0] for c in seedmat.columns})
        got_seeds = sorted({c[1] for c in seedmat.columns})
        seed_ids = np.sort(seedmat.index.to_numpy())
        if len(seed_ids) != 531 or set(seed_ids.tolist()) != set(ids.tolist()):
            raise ValueError(f"seed files cover {len(seed_ids)} basins, expected the same 531")
        rows = []
        win_by_seed = {}
        for s in got_seeds:
            sub = seedmat[[(m, s) for m in got_models if (m, s) in seedmat.columns]]
            sub.columns = [m for m, ss in seedmat.columns if ss == s]
            sub = sub.loc[seed_ids, got_models]
            win_by_seed[s] = sub.idxmax(axis=1)
            rows.append({"seed": s, "n_models": int(sub.shape[1]), "n_basins": int(sub.shape[0]),
                         "median_best_kge": float(np.median(sub.max(axis=1)))})
        pairs = [(a, b) for i, a in enumerate(got_seeds) for b in got_seeds[i + 1:]]
        base = []
        for a, b in pairs:
            same = (win_by_seed[a] == win_by_seed[b])
            base.append({"model_pool": len(got_models), "seed_a": a, "seed_b": b,
                         "n_basins": int(len(same)), "n_same_winner": int(same.sum()),
                         "retention": float(same.mean())})
        allagree = np.ones(len(seed_ids), dtype=bool)
        for s in got_seeds:
            allagree &= (win_by_seed[s] == win_by_seed[got_seeds[0]]).to_numpy()
        # per-basin mean pairwise seed agreement (for the paired contrast below)
        seed_agree_frac = np.zeros(len(seed_ids), dtype=float)
        for a_, b_ in pairs:
            seed_agree_frac += (win_by_seed[a_] == win_by_seed[b_]).to_numpy().astype(float)
        seed_agree_frac /= max(len(pairs), 1)
        base.append({"model_pool": len(got_models), "seed_a": "all", "seed_b": "all",
                     "n_basins": int(len(seed_ids)), "n_same_winner": int(allagree.sum()),
                     "retention": float(allagree.mean())})
        SB = pd.DataFrame(base)
        SB.to_csv(T2 / "seed_noise_winner_baseline.csv", index=False)

        # --- matched-pool comparison: same 8 models, seen->OOB vs seed->seed
        p8 = [m for m in got_models if m in models]
        s8 = seen.loc[ids, p8]
        o8 = oob.loc[ids, p8]
        w_s8 = s8.idxmax(axis=1)
        w_o8 = o8.idxmax(axis=1)
        same8 = (w_s8 == w_o8)
        # margins inside the 8-model pool
        v8 = np.sort(s8.to_numpy(), axis=1)[:, ::-1]
        vo8 = np.sort(o8.to_numpy(), axis=1)[:, ::-1]
        m8 = pd.DataFrame({"seen_margin12": v8[:, 0] - v8[:, 1], "oob_margin12": vo8[:, 0] - vo8[:, 1]},
                          index=ids)
        dec8 = (m8["seen_margin12"] >= 0.01) & (m8["oob_margin12"] >= 0.01)
        seed_ret = float(SB.loc[SB["seed_a"] == "all", "retention"].iloc[0])
        seed_pair_mean = float(SB.loc[SB["seed_a"] != "all", "retention"].mean())
        cmp_rows = [
            {"comparison": "seed-only (canonical dPL, 8 models, seeds 42/123/2026)",
             "n_models": len(p8), "n_basins": int(len(ids)),
             "variation_source": "random seed (same regime, all basins trained on)",
             "retention": seed_ret, "n_same": int(round(seed_ret * len(ids))),
             "note": "3-way agreement across all three seeds"},
            {"comparison": "seed-only, mean over seed pairs",
             "n_models": len(p8), "n_basins": int(len(ids)),
             "variation_source": "random seed (same regime)",
             "retention": seed_pair_mean, "n_same": None,
             "note": "mean of the 3 seed pairs"},
            {"comparison": "seen -> OOB (same 8-model pool)",
             "n_models": len(p8), "n_basins": int(len(ids)),
             "variation_source": "basin withheld from training (same pipeline family)",
             "retention": float(same8.mean()), "n_same": int(same8.sum()),
             "note": "pool-matched to the seed baseline"},
            {"comparison": "seen -> OOB, decided basins only (8-model pool, tau=0.01)",
             "n_models": len(p8), "n_basins": int(dec8.sum()),
             "variation_source": "basin withheld, near-ties removed",
             "retention": float(same8[dec8].mean()) if dec8.any() else None,
             "n_same": int(same8[dec8].sum()), "note": "tau=0.01 absolute KGE margin"},
            {"comparison": "seen -> OOB (full 36-model pool, for reference)",
             "n_models": len(models), "n_basins": int(len(ids)),
             "variation_source": "basin withheld",
             "retention": float(M["same_winner"].mean()), "n_same": int(M["same_winner"].sum()),
             "note": "NOT pool-matched to the seed baseline"},
        ]
        C = pd.DataFrame(cmp_rows)
        C.to_csv(T2 / "winner_matched_pool_comparison.csv", index=False)
        report["step3_seed_baseline"] = {
            "status": "AVAILABLE",
            "models": p8, "seeds": got_seeds, "n_basins": int(len(seed_ids)),
            "missing_files": missing,
            "seed_only_retention_3way": seed_ret,
            "seed_only_retention_pair_mean": seed_pair_mean,
            "seed_pair_detail": SB[SB["seed_a"] != "all"].to_dict(orient="records"),
            "seen_to_oob_retention_8model_pool": float(same8.mean()),
            "seen_to_oob_retention_8model_pool_decided_tau0p01": float(same8[dec8].mean()) if dec8.any() else None,
            "n_decided_8model_pool_tau0p01": int(dec8.sum()),
            "caveat": ("the seed runs are the canonical dPL v2 seen-basin regime (all 531 basins "
                       "trained), while the seen->OOB contrast withholds a basin; the pool (8 models) "
                       "and the basin set (531) are matched, the training-set composition is not. "
                       "8 models is a smaller pool than 36, so both numbers here are pool-matched to "
                       "each other but not to the full 36-model retention."),
        }

    # ------------------------------------------------------------------ step 3c
    # SAME-PIPELINE variant: the "seen" side is now the OOB pipeline's own in-sample
    # arm (the fold models that DID see the basin), so the only difference from the
    # held-out side is basin withholding, not a different training experiment.
    if INSAMPLE.exists():
        ins = pd.read_csv(INSAMPLE)
        if ins.groupby(["model", "fold"]).ngroups == 180:
            ins = ins.merge(pd.read_parquet(MERGED)[["model", "gauge_id", "fold"]]
                            .rename(columns={"gauge_id": "basin_id"}),
                            on=["model", "basin_id"], how="left")
            sp = {}
            for stat_col in ("median", "mean"):
                pv = (ins.groupby(["model", "basin_id"])["kge_insample"]
                      .agg(stat_col).unstack("model").loc[ids, models])
                w = pv.idxmax(axis=1)
                v = np.sort(pv.to_numpy(), axis=1)[:, ::-1]
                m12 = v[:, 0] - v[:, 1]
                oob_rank = np.array([
                    1 + int(np.sum(oob.to_numpy()[i] > oob.to_numpy()[i, models.index(w.iloc[i])]))
                    for i in range(len(ids))])
                same = (w == mo["winner"]).to_numpy()
                dec = (m12 >= 0.01) & (M["oob_margin12"].to_numpy() >= 0.01)
                # pool-matched to the 8-model seed baseline
                p8 = [m for m in EXISTING8 if m in models]
                pv8 = pv[p8]
                w8 = pv8.idxmax(axis=1)
                w8_oob = oob[p8].idxmax(axis=1)
                same8 = (w8 == w8_oob).to_numpy()
                sp[stat_col] = {
                    "aggregation": stat_col,
                    "strict_retention": float(same.mean()),
                    "n_same": int(same.sum()),
                    "strict_retention_decided_tau0p01": float(same[dec].mean()) if dec.any() else None,
                    "n_decided_tau0p01": int(dec.sum()),
                    "median_seen_margin12": float(np.median(m12)),
                    "frac_same_pipeline_winner_in_oob_top3": float((oob_rank <= 3).mean()),
                    "frac_same_pipeline_winner_in_oob_top5": float((oob_rank <= 5).mean()),
                    "median_oob_rank_of_same_pipeline_winner": float(np.median(oob_rank)),
                    "retention_8model_pool": float(same8.mean()),
                    "n_same_8model_pool": int(same8.sum()),
                }
                sp[stat_col]["same_pipeline_vs_seed_retention_8model_pool"] = {
                    "same_pipeline_seen_to_oob": float(same8.mean()),
                    "seed_only_3way": float(SB.loc[SB["seed_a"] == "all", "retention"].iloc[0]),
                    "seed_only_pair_mean": float(SB.loc[SB["seed_a"] != "all", "retention"].mean()),
                }
            report["step3c_same_pipeline_winner"] = sp
            pd.DataFrame([{**{"aggregation": k}, **{kk: vv for kk, vv in v.items()
                                                     if not isinstance(vv, dict)}}
                          for k, v in sp.items()]).to_csv(
                T2 / "winner_same_pipeline_comparison.csv", index=False)
        else:
            report["step3c_same_pipeline_winner"] = {"status": "NOT_AVAILABLE",
                                                    "reason": "in-sample table incomplete"}
    else:
        report["step3c_same_pipeline_winner"] = {"status": "NOT_AVAILABLE",
                                                "reason": "pub_insample_seen_kge.csv missing"}
    # ------- paired contrast: same-pipeline OOB change vs seed-only change -------
    spc = report.get("step3c_same_pipeline_winner", {})
    if isinstance(spc, dict) and "median" in spc and "seed_agree_frac" in dir():
        pass
    try:
        p8b = [m for m in EXISTING8 if m in models]
        spv = (pd.read_csv(INSAMPLE).merge(
                    pd.read_parquet(MERGED)[["model", "gauge_id"]].rename(
                        columns={"gauge_id": "basin_id"}), on=["model", "basin_id"], how="left")
               .groupby(["model", "basin_id"])["kge_insample"].median()
               .unstack("model").loc[ids, p8b])
        oob8 = df.pivot(index="gauge_id", columns="model", values="kge_oob").loc[ids, p8b]
        same_sp = (spv.idxmax(axis=1) == oob8.idxmax(axis=1)).to_numpy().astype(float)
        d_b = seed_agree_frac - same_sp          # per basin: seed agreement minus OOB agreement
        rng3 = np.random.default_rng(SEED)
        boots = d_b[rng3.integers(0, d_b.size, size=(2000, d_b.size))].mean(axis=1)
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        report["step3d_paired_contrast"] = {
            "unit": "basin (531)",
            "statistic": "mean_b(seed_agreement_b) - mean_b(oob_agreement_b), 8-model pool",
            "seed_agreement_frac_mean": float(seed_agree_frac.mean()),
            "same_pipeline_oob_agreement_mean": float(same_sp.mean()),
            "difference": float(d_b.mean()), "ci95": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_basins": int(d_b.size), "bootstrap_B": 2000,
            "interpretation": ("seed-only agreement exceeds same-pipeline OOB agreement by this "
                              "margin; a CI above 0 means withholding a basin destabilises the "
                              "per-basin winner beyond stochastic-training instability"),
            "caveat": ("the seed runs are the canonical all-531-basin dPL regime; the same-pipeline "
                       "arm is the 4/5-basin OOB fold regime. Both are neural dPL on the same 531 "
                       "basins with the same protocol, but the training-set composition differs."),
        }
    except Exception as exc:   # noqa: BLE001 - recorded, not swallowed
        report["step3d_paired_contrast"] = {"status": "NOT_AVAILABLE", "reason": repr(exc)}
    with open(T2 / "winner_stability.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # -------- console --------
    log("== step 1: winner margins ==")
    log(f"   strict retention = {M['same_winner'].sum()}/{len(M)} = {M['same_winner'].mean():.4f}")
    log(f"   seen margin12 median = {M['seen_margin12'].median():.4f}; "
        f"oob margin12 median = {M['oob_margin12'].median():.4f}")
    log(f"   on the {int(M['winner_changed'].sum())} changed basins, OOB gap old->new winner: "
        f"median {ch['oob_gap_old_new_winner'].median():.4f}, "
        f"{100*float((ch['oob_gap_old_new_winner']<0.01).mean()):.1f}% below 0.01")
    log(f"   OOB rank of seen winner: median {np.median(rank_of_seen_winner):.0f}, "
        f"{100*report['step1_margins']['oob_rank_of_seen_winner']['frac_rank_le3']:.1f}% within top-3")
    log("\n== step 2: threshold sensitivity (descriptive tau) ==")
    log(S.to_string(index=False))
    log("\n== step 3: seed baseline ==")
    if report["step3_seed_baseline"]["status"] == "AVAILABLE":
        log(f"   seed-only 3-way winner retention (8 models) = {seed_ret:.4f}")
        log(f"   seen->OOB retention, same 8-model pool      = {float(same8.mean()):.4f}")
        log(f"   seen->OOB retention, full 36-model pool     = {float(M['same_winner'].mean()):.4f}")
    else:
        log("   NOT_AVAILABLE")
    sp = report.get("step3c_same_pipeline_winner", {})
    log("\n== step 3c: same-pipeline winner (OOB in-sample vs OOB held-out) ==")
    if isinstance(sp, dict) and "median" in sp:
        for tag in ("median", "mean"):
            r = sp[tag]
            log(f"   [{tag}] strict retention = {r['strict_retention']:.4f} "
                f"({r['n_same']}/{len(ids)}); decided-only(tau=0.01) = "
                f"{r['strict_retention_decided_tau0p01']:.4f} (n={r['n_decided_tau0p01']})")
            log(f"        seen-winner in OOB top-3 = {r['frac_same_pipeline_winner_in_oob_top3']:.4f}; "
                f"8-model pool retention = {r['retention_8model_pool']:.4f} "
                f"vs seed-only = {r['same_pipeline_vs_seed_retention_8model_pool']['seed_only_3way']:.4f}")
    else:
        log(f"   {sp.get('status', 'unavailable')}: {sp.get('reason', '')}")
    pc = report.get("step3d_paired_contrast", {})
    log("\n== step 3d: paired contrast (seed-only vs same-pipeline OOB), 8-model pool ==")
    if "difference" in pc:
        log(f"   seed agreement = {pc['seed_agreement_frac_mean']:.4f}; "
            f"OOB agreement = {pc['same_pipeline_oob_agreement_mean']:.4f}; "
            f"difference = {pc['difference']:.4f} CI95 [{pc['ci95'][0]:.4f}, {pc['ci95'][1]:.4f}] "
            f"-> CI excludes 0: {pc['ci_excludes_zero']}")
    else:
        log(f"   {pc.get('status', 'unavailable')}: {pc.get('reason','')}")
    log(f"\nwrote tables -> {T2}")


if __name__ == "__main__":
    main()
