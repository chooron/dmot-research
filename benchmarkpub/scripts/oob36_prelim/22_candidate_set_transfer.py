#!/usr/bin/env python3
"""22_candidate_set_transfer.py — does a small seen candidate set still cover the good models
after spatial transfer to a held-out basin?

The paper's recommendation premise is: do not recommend the single top-1 model per basin, but a
small high-quality candidate set.  This script tests that premise directly.

For each basin b, two candidate-set strategies, both of size k:

    basin-specific Seen Top-k   the k best models under the basin's OWN seen score
    global Top-k                the k best models under aggregate seen performance over all
                                531 basins (one fixed set shared by every basin)

and for both:

    coverage   OOB top-1 covered?          (the headline metric)
               OOB top-3 / top-5 any-hit and mean fraction captured
    regret     max OOB KGE inside the set vs the OOB oracle max over all 36 models

Both 36-model vectors per basin are already frozen: seen = `kge_seenfolds_median4` (the
next-stage PRIMARY aggregation) and OOB = `kge_heldout`, in
`pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv`.  NO new inference.

`k` is never optimised or frozen: the whole curve k=1..10 is reported, plus k=36 as the
full-pool reference, and only the coverage/efficiency trade-off is described.

Outputs -> $OUT4/tables/{candidate_set_transfer_by_basin,candidate_set_grid_long,
                         topk_coverage_curve,topk_regret_curve,basin_specific_vs_global_topk,
                         top3_top5_set_overlap,seed_candidate_set_baseline}.csv
           $OUT4/tables/candidate_set_transfer.json
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
B_BOOT = 2000
KS = list(range(1, 11)) + [15, 20, 36]      # 36 = full pool reference
REGRET_THRESHOLDS = (0.01, 0.02, 0.05)      # descriptive sensitivity only


def log(m: str) -> None:
    print(m, flush=True)


def boot_mean(x: np.ndarray, B: int = B_BOOT, seed: int = SEED):
    """Percentile CI of the mean, resampling BASINS."""
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float)[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_median(x: np.ndarray, B: int = B_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    d = np.median(np.asarray(x, float)[rng.integers(0, len(x), size=(B, len(x)))], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    PC.T4.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")
    if len(F) != 19116:
        raise ValueError(f"expected 19,116 rows, got {len(F)}")

    models = sorted(F["model"].unique())
    basins = np.sort(F["basin_id"].unique())
    seen = F.pivot(index="basin_id", columns="model", values="kge_seenfolds_median4").loc[basins, models]
    oob = F.pivot(index="basin_id", columns="model", values="kge_heldout").loc[basins, models]
    seen_mean4 = F.pivot(index="basin_id", columns="model",
                         values="kge_seenfolds_mean4").loc[basins, models]
    if seen.isna().any().any() or oob.isna().any().any():
        raise ValueError("pivot produced NaN")

    # global aggregate seen performance -> one fixed ranking shared by all basins
    agg = seen.median(axis=0).sort_values(ascending=False)
    global_order = agg.index.tolist()
    # basin-specific ranking (order per basin, stable)
    seen_arr = seen.to_numpy(np.float64)
    local_order = np.argsort(-seen_arr, axis=1, kind="stable")
    local_sets = {k: {int(b): set(np.asarray(models)[local_order[i, :k]].tolist())
                      for i, b in enumerate(basins)} for k in KS}

    report = {"meta": {
        "seed": SEED, "bootstrap_B": B_BOOT, "n_basins": int(len(basins)),
        "n_models": int(len(models)), "k_grid": KS,
        "primary_seen_definition": "kge_seenfolds_median4 (the frozen next-stage primary)",
        "oob_definition": "kge_heldout (held-out fold checkpoint)",
        "global_definition": "models ranked by their median kge_seenfolds_median4 over all 531 basins",
        "global_ranking_top10": global_order[:10],
        "global_ranking_full": global_order,
        "regret_thresholds": list(REGRET_THRESHOLDS),
        "threshold_status": "descriptive sensitivity only; no equivalence set or delta is defined",
        "no_new_inference": True,
    }}

    # ------------------------------------------------------------------ curves
    grid_rows, cov_rows, reg_rows, overlap_rows = [], [], [], []
    by_basin = {k: {} for k in KS}
    # precompute OOB-derived per-basin quantities once per k via the helper
    for k in KS:
        glob_sets = {int(b): set(global_order[:k]) for b in basins}
        for tag, sets in (("basin_specific", local_sets[k]), ("global", glob_sets)):
            R = PC.coverage_and_regret(sets, oob, k)
            for b in basins:
                grid_rows.append({
                    "strategy": tag, "k": k, "basin_id": int(b),
                    "covers_oob_top1": bool(R.loc[b, "covers_oob_top1"]),
                    "hits_oob_top3_any": bool(R.loc[b, "hits_oob_top3_any"]),
                    "frac_oob_top3_covered": float(R.loc[b, "frac_oob_top3_covered"]),
                    "hits_oob_top5_any": bool(R.loc[b, "hits_oob_top5_any"]),
                    "frac_oob_top5_covered": float(R.loc[b, "frac_oob_top5_covered"]),
                    "regret": float(R.loc[b, "regret"]),
                })
            cov = R["covers_oob_top1"].to_numpy(float)
            lo, hi = boot_mean(cov)
            rg = R["regret"].to_numpy(float)
            rlo, rhi = boot_median(rg)
            row = {
                "strategy": tag, "k": k,
                "coverage_top1": float(cov.mean()), "coverage_top1_ci95_lo": lo,
                "coverage_top1_ci95_hi": hi,
                "n_covered": int(cov.sum()),
                "coverage_top3_anyhit": float(R["hits_oob_top3_any"].mean()),
                "coverage_top3_mean_frac": float(R["frac_oob_top3_covered"].mean()),
                "coverage_top5_anyhit": float(R["hits_oob_top5_any"].mean()),
                "coverage_top5_mean_frac": float(R["frac_oob_top5_covered"].mean()),
            }
            cov_rows.append(row)
            reg_rows.append({
                "strategy": tag, "k": k,
                "regret_median": float(np.median(rg)), "regret_mean": float(np.mean(rg)),
                "regret_median_ci95_lo": rlo, "regret_median_ci95_hi": rhi,
                "regret_q75": float(np.percentile(rg, 75)),
                "regret_q90": float(np.percentile(rg, 90)),
                "regret_q95": float(np.percentile(rg, 95)),
                "regret_max": float(rg.max()),
                **{f"frac_regret_le_{t}": float((rg <= t).mean()) for t in REGRET_THRESHOLDS},
            })
        # same-k set overlap (seen Top-k vs OOB Top-k), for the required k values
        oob_arr = oob.to_numpy(np.float64)
        oob_order = np.argsort(-oob_arr, axis=1, kind="stable")
        inter = np.array([len(local_sets[k][int(b)]
                              & set(np.asarray(models)[oob_order[i, :k]].tolist()))
                          for i, b in enumerate(basins)])
        overlap_rows.append({
            "k": k,
            "jaccard_median": float(np.median(inter / (2 * k - inter))),
            "jaccard_mean": float(np.mean(inter / (2 * k - inter))),
            "jaccard_q25": float(np.percentile(inter / (2 * k - inter), 25)),
            "jaccard_q75": float(np.percentile(inter / (2 * k - inter), 75)),
            "recall_mean": float(inter.mean() / k),
            "intersection_mean": float(inter.mean()),
            "intersection_distribution": json.dumps(
                {int(i): int((inter == i).sum()) for i in range(k + 1)}),
            "full_set_retained_fraction": float((inter == k).mean()),
        })

    G = pd.DataFrame(grid_rows)
    C = pd.DataFrame(cov_rows)
    Rg = pd.DataFrame(reg_rows)
    O = pd.DataFrame(overlap_rows)
    G.to_csv(PC.T4 / "candidate_set_grid_long.csv", index=False)
    C.to_csv(PC.T4 / "topk_coverage_curve.csv", index=False)
    Rg.to_csv(PC.T4 / "topk_regret_curve.csv", index=False)
    O.to_csv(PC.T4 / "top3_top5_set_overlap.csv", index=False)

    # marginal gain of top-1 coverage from k to k+1
    mg = []
    for tag in ("basin_specific", "global"):
        s = C[C.strategy == tag].sort_values("k")
        for i in range(len(s) - 1):
            mg.append({"strategy": tag, "k_from": int(s.iloc[i]["k"]), "k_to": int(s.iloc[i + 1]["k"]),
                       "coverage_gain": float(s.iloc[i + 1]["coverage_top1"] - s.iloc[i]["coverage_top1"])})
    MG = pd.DataFrame(mg)
    MG.to_csv(PC.T4 / "topk_marginal_gain.csv", index=False)

    # ------------------------------------------------- per-basin wide table
    wide = pd.DataFrame({"basin_id": basins})
    wide["gauge_id_str"] = [str(int(b)).zfill(8) for b in basins]
    fold = pd.read_parquet(PC.MERGED)[["gauge_id", "fold"]].drop_duplicates()
    wide["heldout_fold"] = wide["basin_id"].map(dict(zip(fold["gauge_id"].astype(int),
                                                         fold["fold"].astype(int))))
    wide["oob_winner"] = oob.idxmax(axis=1).to_numpy()
    wide["seen_winner"] = seen.idxmax(axis=1).to_numpy()
    wide["basin_specific_rank_of_oob_winner"] = [
        int(1 + np.where(np.asarray(models)[local_order[i]] == oob.idxmax(axis=1).iloc[i])[0][0])
        for i in range(len(basins))]
    for k in KS:
        bs = G[(G.strategy == "basin_specific") & (G.k == k)].set_index("basin_id")
        gl = G[(G.strategy == "global") & (G.k == k)].set_index("basin_id")
        wide[f"cov_bs_k{k}"] = bs.loc[basins, "covers_oob_top1"].to_numpy()
        wide[f"regret_bs_k{k}"] = bs.loc[basins, "regret"].to_numpy()
        if k in (1, 3, 5, 10):
            wide[f"cov_global_k{k}"] = gl.loc[basins, "covers_oob_top1"].to_numpy()
            wide[f"regret_global_k{k}"] = gl.loc[basins, "regret"].to_numpy()
        wide[f"cov3_bs_k{k}"] = bs.loc[basins, "hits_oob_top3_any"].to_numpy()
        wide[f"frac3_bs_k{k}"] = bs.loc[basins, "frac_oob_top3_covered"].to_numpy()
        wide[f"cov5_bs_k{k}"] = bs.loc[basins, "hits_oob_top5_any"].to_numpy()
        wide[f"frac5_bs_k{k}"] = bs.loc[basins, "frac_oob_top5_covered"].to_numpy()
    wide.to_csv(PC.T4 / "candidate_set_transfer_by_basin.csv", index=False)

    # ------------------------------------------------- basin-specific vs global
    cmp_rows = []
    for k in KS:
        bs = G[(G.strategy == "basin_specific") & (G.k == k)].set_index("basin_id").loc[basins]
        gl = G[(G.strategy == "global") & (G.k == k)].set_index("basin_id").loc[basins]
        a = bs["covers_oob_top1"].to_numpy(bool); b_ = gl["covers_oob_top1"].to_numpy(bool)
        diff_cov = a.astype(float) - b_.astype(float)
        lo, hi = boot_mean(diff_cov)
        ra = bs["regret"].to_numpy(float); rb = gl["regret"].to_numpy(float)
        dreg = ra - rb
        rlo, rhi = boot_median(dreg)
        cmp_rows.append({
            "k": k,
            "coverage_basin_specific": float(a.mean()), "coverage_global": float(b_.mean()),
            "coverage_diff_bs_minus_global": float(diff_cov.mean()),
            "coverage_diff_ci95_lo": lo, "coverage_diff_ci95_hi": hi,
            "n_bs_only": int((a & ~b_).sum()), "n_global_only": int((~a & b_).sum()),
            "n_both": int((a & b_).sum()), "n_neither": int((~a & ~b_).sum()),
            "mcnemar_exact_p": float(sstats.binomtest(int((a & ~b_).sum()),
                                                      int((a & ~b_).sum() + int((~a & b_).sum())),
                                                      0.5).pvalue)
            if int((a & ~b_).sum()) + int((~a & b_).sum()) > 0 else None,
            "regret_median_basin_specific": float(np.median(ra)),
            "regret_median_global": float(np.median(rb)),
            "regret_median_diff_bs_minus_global": float(np.median(ra) - np.median(rb)),
            "regret_median_diff_ci95_lo": rlo, "regret_median_diff_ci95_hi": rhi,
            "mean_regret_basin_specific": float(ra.mean()),
            "mean_regret_global": float(rb.mean()),
        })
    CMP = pd.DataFrame(cmp_rows)
    CMP.to_csv(PC.T4 / "basin_specific_vs_global_topk.csv", index=False)

    # ------------------------------------------------- seed-noise reference (§6)
    seed_df, missing = PC.load_seed_matrix()
    seed_rows = []
    if seed_df.shape[1] > 0:
        pool = PC.SEED8
        seed_df = seed_df.loc[basins, [(m, s) for m in pool for s in (42, 123, 2026)
                                       if (m, s) in seed_df.columns]]
        seeds = sorted({s for _, s in seed_df.columns})
        seen8 = seen[pool]; oob8 = oob[pool]
        for k in range(1, len(pool) + 1):
            # (i) seed -> seed, same regime, different seed
            hits, regs = [], []
            for a in seeds:
                for b_ in seeds:
                    if a == b_:
                        continue
                    S = seed_df[[(m, a) for m in pool]].copy(); S.columns = pool
                    T = seed_df[[(m, b_) for m in pool]].copy(); T.columns = pool
                    sets = PC.topk_sets(S, k)
                    r = PC.coverage_and_regret(sets, T, k)
                    hits.append(r["covers_oob_top1"].to_numpy(float))
                    regs.append(r["regret"].to_numpy(float))
            seed_rows.append({
                "pool": f"{len(pool)} models (seed experiment)", "k": k,
                "reference": "seed->seed (same regime, different seed)",
                "coverage_top1": float(np.mean([h.mean() for h in hits])),
                "coverage_top1_ci95_lo": np.nan, "coverage_top1_ci95_hi": np.nan,
                "regret_median": float(np.median(np.concatenate(regs))),
                "n_ordered_pairs": len(hits),
            })
            # (ii) seen -> OOB inside the SAME 8-model pool (pool-matched counterpart)
            l8 = {int(b): set(np.asarray(pool)[np.argsort(-seen8.to_numpy()[i], kind="stable")[:k]].tolist())
                  for i, b in enumerate(basins)}
            r8 = PC.coverage_and_regret(l8, oob8, k)
            seed_rows.append({
                "pool": f"{len(pool)} models (OOB experiment)", "k": k,
                "reference": "seen->OOB (same 8-model pool)",
                "coverage_top1": float(r8["covers_oob_top1"].mean()),
                "coverage_top1_ci95_lo": boot_mean(r8["covers_oob_top1"].to_numpy(float))[0],
                "coverage_top1_ci95_hi": boot_mean(r8["covers_oob_top1"].to_numpy(float))[1],
                "regret_median": float(r8["regret"].median()),
                "n_ordered_pairs": 1,
            })
            # (iii) full 36-model pool, for scale reference
            c36 = C[(C.strategy == "basin_specific") & (C.k == k)]
            if len(c36):
                seed_rows.append({
                    "pool": "36 models (OOB experiment, reference)", "k": k,
                    "reference": "seen->OOB (full 36-model pool)",
                    "coverage_top1": float(c36["coverage_top1"].iloc[0]),
                    "coverage_top1_ci95_lo": float(c36["coverage_top1_ci95_lo"].iloc[0]),
                    "coverage_top1_ci95_hi": float(c36["coverage_top1_ci95_hi"].iloc[0]),
                    "regret_median": float(Rg[(Rg.strategy == "basin_specific") & (Rg.k == k)]["regret_median"].iloc[0]),
                    "n_ordered_pairs": np.nan,
                })
    else:
        seed_rows.append({"pool": "NOT_AVAILABLE", "k": np.nan, "reference": "no seed files found",
                          "coverage_top1": np.nan, "coverage_top1_ci95_lo": np.nan,
                          "coverage_top1_ci95_hi": np.nan, "regret_median": np.nan,
                          "n_ordered_pairs": 0})
    SB = pd.DataFrame(seed_rows)
    SB.to_csv(PC.T4 / "seed_candidate_set_baseline.csv", index=False)

    # ------------------------------------------------- sensitivity: mean-of-4 seen
    sens = []
    lm = np.argsort(-seen_mean4.to_numpy(np.float64), axis=1, kind="stable")
    for k in (1, 3, 5, 10):
        sets = {int(b): set(np.asarray(models)[lm[i, :k]].tolist()) for i, b in enumerate(basins)}
        r = PC.coverage_and_regret(sets, oob, k)
        sens.append({"seen_definition": "mean-of-4", "k": k,
                     "coverage_top1": float(r["covers_oob_top1"].mean()),
                     "regret_median": float(r["regret"].median()),
                     "coverage_top3_anyhit": float(r["hits_oob_top3_any"].mean()),
                     "coverage_top5_anyhit": float(r["hits_oob_top5_any"].mean())})
    SEN = pd.DataFrame(sens)
    SEN.to_csv(PC.T4 / "candidate_set_seen_definition_sensitivity.csv", index=False)

    # ------------------------------------------------- validation vs frozen tables
    val = {"frozen_supplemental_top3_top5_overlap": {}}
    sup = pd.read_csv(PC.OUT3 / "tables" / "basin_rank_stability.csv")
    for k in (3, 5):
        mine = float(O.loc[O.k == k, "jaccard_median"].iloc[0])
        theirs = float(sup[f"top{k}_jaccard"].median())
        val["frozen_supplemental_top3_top5_overlap"][f"k{k}_jaccard_median"] = {
            "recomputed": mine, "frozen": theirs, "abs_diff": abs(mine - theirs)}
    val["k1_coverage_equals_frozen_winner_retention"] = {
        "recomputed": float(C[(C.strategy == "basin_specific") & (C.k == 1)]["coverage_top1"].iloc[0]),
        "frozen_same_pipeline_winner_retention": json.load(
            open(PC.OUT2 / "tables" / "winner_stability.json")
        )["step3c_same_pipeline_winner"]["median"]["strict_retention"],
    }

    report["coverage_curve"] = C.to_dict(orient="records")
    report["regret_curve"] = Rg.to_dict(orient="records")
    report["set_overlap"] = O.to_dict(orient="records")
    report["basin_specific_vs_global"] = CMP.to_dict(orient="records")
    report["marginal_gain"] = MG.to_dict(orient="records")
    report["seed_reference"] = SB.to_dict(orient="records")
    report["seen_definition_sensitivity"] = sens
    report["validation"] = val
    with open(PC.T4 / "candidate_set_transfer.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ------------------------------------------------- console
    log("== coverage / regret curve (basin-specific Seen Top-k) ==")
    log(C[C.strategy == "basin_specific"][
        ["k", "coverage_top1", "coverage_top1_ci95_lo", "coverage_top1_ci95_hi",
         "coverage_top3_anyhit", "coverage_top3_mean_frac",
         "coverage_top5_anyhit", "coverage_top5_mean_frac"]].round(4).to_string(index=False))
    log("\n== regret curve (basin-specific) ==")
    log(Rg[Rg.strategy == "basin_specific"][
        ["k", "regret_median", "regret_mean", "regret_q90", "regret_q95",
         "frac_regret_le_0.01", "frac_regret_le_0.02", "frac_regret_le_0.05"]].round(4).to_string(index=False))
    log("\n== basin-specific vs global ==")
    log(CMP.round(4).to_string(index=False))
    log("\n== marginal coverage gain ==")
    log(MG[MG.strategy == "basin_specific"].round(4).to_string(index=False))
    log("\n== seed reference (8-model pool) ==")
    log(SB.round(4).to_string(index=False))
    log("\n== validation vs frozen ==")
    log(json.dumps(val, indent=1))
    log(f"\nwrote tables -> {PC.T4}")


if __name__ == "__main__":
    main()
