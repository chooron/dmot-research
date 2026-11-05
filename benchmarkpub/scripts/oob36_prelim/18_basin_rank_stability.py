#!/usr/bin/env python3
"""18_basin_rank_stability.py — Analysis 1: basin-level 36-model rank stability.

Question: the strict top-1 winner changes in ~75% of basins.  Does that mean the WHOLE
within-basin ordering of the 36 models is unstable, or are these mostly local top-1 swaps
inside an otherwise well-preserved ordering?

For every basin b we compare two 36-model KGE vectors:

    seen  = same-pipeline in-sample KGE of the 4 fold checkpoints that trained on b
            (median over those 4 folds; mean-of-4 reported as a sensitivity)
    OOB   = held-out KGE of the fold checkpoint that never saw b

and compute the within-basin Spearman rho, Kendall tau, and Top-3 / Top-5 set overlap
(Jaccard, intersection size, retention).

The same-pipeline definition is primary.  The previous cross-pipeline seen side
(canonical dPL v2, all 531 basins trained) is computed alongside purely for comparison, so
the report can show that the conclusion does not depend on which seen definition is used.

NO inference, NO retraining: both 36-model vectors already exist in
`pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv`.  Basin-level
correlations are descriptive; the 531 basins are NOT treated as independent time replicates.

Outputs -> $OUT3/tables/{basin_rank_stability,topk_overlap_summary,winner_vs_rank_stability}.csv
           $OUT3/tables/rank_stability.json
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
KS = (3, 5)
# the frozen descriptor of the OOB contract: eps added to the KGE correlation denominator
# and to the obs mean / std in the beta and gamma ratios
KGE_EPS = 0.1


def log(m: str) -> None:
    print(m, flush=True)


def per_basin_rank_stats(seen: pd.DataFrame, oob: pd.DataFrame) -> pd.DataFrame:
    """Within-basin rank agreement of two basin x model KGE matrices."""
    models = list(seen.columns)
    rows = []
    for b in seen.index:
        s = seen.loc[b, models].to_numpy(np.float64)
        o = oob.loc[b, models].to_numpy(np.float64)
        rho = float(sstats.spearmanr(s, o).statistic)
        tau = float(sstats.kendalltau(s, o).statistic)
        rec = {"basin_id": int(b), "spearman_rho": rho, "kendall_tau": tau,
               "n_models": len(models),
               "seen_winner": models[int(np.argmax(s))],
               "oob_winner": models[int(np.argmax(o))]}
        for k in KS:
            s_top = set(np.argsort(-s, kind="stable")[:k].tolist())
            o_top = set(np.argsort(-o, kind="stable")[:k].tolist())
            inter = len(s_top & o_top)
            rec[f"top{k}_intersection"] = inter
            rec[f"top{k}_jaccard"] = inter / (2 * k - inter)
            rec[f"top{k}_retention"] = inter / k
            rec[f"top{k}_full_set_retained"] = bool(inter == k)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("basin_id")


def summarise(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return {
        "n": int(x.size), "median": float(np.median(x)), "mean": float(np.mean(x)),
        "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
        "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
        "min": float(x.min()), "max": float(x.max()),
        "frac_gt_0": float((x > 0).mean()),
        "frac_ge_0p3": float((x >= 0.3).mean()), "frac_ge_0p5": float((x >= 0.5).mean()),
        "frac_ge_0p7": float((x >= 0.7).mean()),
    }


def boot_median_ci(x: np.ndarray, B: int = B_BOOT, seed: int = SEED):
    """Percentile CI of the median, resampling BASINS."""
    rng = np.random.default_rng(seed)
    draws = np.median(np.asarray(x, float)[rng.integers(0, len(x), size=(B, len(x)))], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    PC.T3.mkdir(parents=True, exist_ok=True)

    # exact-tie guard: any tie at a top-k boundary would make set overlap ambiguous
    all_vals = []
    variants = {}
    for tag, seen_col in (("samepipeline_median4", "kge_seenfolds_median4"),
                          ("samepipeline_mean4", "kge_seenfolds_mean4"),
                          ("crosspipeline_canonical", "kge_seendpl_canonical")):
        seen, oob = PC.seen_oob_matrices(seen_col=seen_col, oob_col="kge_heldout")
        variants[tag] = per_basin_rank_stats(seen, oob)
        all_vals.append(seen.to_numpy())
    n_tie_pairs = int(sum(len(np.unique(v)) != v.size for v in all_vals))

    prim = variants["samepipeline_median4"]
    out = prim.copy()
    out["gauge_id_str"] = [str(int(b)).zfill(8) for b in out.index]
    out["heldout_fold"] = [PC.basin_fold_map()[int(b)] for b in out.index]
    for tag in ("samepipeline_mean4", "crosspipeline_canonical"):
        v = variants[tag]
        out[f"spearman_rho_{tag}"] = v["spearman_rho"]
        out[f"kendall_tau_{tag}"] = v["kendall_tau"]
        for k in KS:
            out[f"top{k}_jaccard_{tag}"] = v[f"top{k}_jaccard"]
            out[f"top{k}_retention_{tag}"] = v[f"top{k}_retention"]
    out = out.reset_index()
    out.to_csv(PC.T3 / "basin_rank_stability.csv", index=False)

    # ------------------------------------------------------------------ summaries
    report = {"meta": {
        "seed": SEED, "bootstrap_B": B_BOOT, "n_basins": int(len(prim)),
        "n_models": int(prim["n_models"].iloc[0]), "top_k": list(KS),
        "primary_seen_definition": "same-pipeline in-sample KGE, median over the 4 folds "
                                  "that trained on the basin (kge_seenfolds_median4)",
        "oob_definition": "held-out fold KGE (kge_heldout)",
        "kge_eps_reference": KGE_EPS,
        "note": "within-basin correlations are descriptive; basins are not independent "
                "time replicates and no threshold is frozen",
        "exact_tie_matrices": n_tie_pairs,
    }}
    report["rank_correlation"] = {}
    for tag, v in variants.items():
        d = summarise(v["spearman_rho"].to_numpy())
        lo, hi = boot_median_ci(v["spearman_rho"].to_numpy())
        report["rank_correlation"][f"spearman_{tag}"] = {**d, "median_ci95": [lo, hi]}
        report["rank_correlation"][f"kendall_{tag}"] = summarise(v["kendall_tau"].to_numpy())

    # ------------------------------------------------------------------ top-k overlap
    topk_rows = []
    for tag, v in variants.items():
        for k in KS:
            j = v[f"top{k}_jaccard"].to_numpy()
            inter = v[f"top{k}_intersection"].to_numpy()
            lo, hi = boot_median_ci(j)
            row = {"variant": tag, "k": k,
                   "jaccard_median": float(np.median(j)), "jaccard_mean": float(np.mean(j)),
                   "jaccard_q25": float(np.percentile(j, 25)),
                   "jaccard_q75": float(np.percentile(j, 75)),
                   "jaccard_p05": float(np.percentile(j, 5)),
                   "jaccard_p95": float(np.percentile(j, 95)),
                   "jaccard_median_ci95_lo": lo, "jaccard_median_ci95_hi": hi,
                   "full_set_retained_fraction": float(v[f"top{k}_full_set_retained"].mean()),
                   "intersection_mean": float(inter.mean()),
                   "retention_mean": float(v[f"top{k}_retention"].mean())}
            dist = {int(i): int((inter == i).sum()) for i in range(k + 1)}
            row["intersection_distribution"] = json.dumps(dist)
            for i in range(k + 1):
                row[f"frac_intersection_{i}"] = float((inter == i).mean())
            topk_rows.append(row)
    T = pd.DataFrame(topk_rows)
    T.to_csv(PC.T3 / "topk_overlap_summary.csv", index=False)
    report["topk_overlap"] = T.to_dict(orient="records")

    # ------------------------------------------------------------------ winner vs rank
    # same-pipeline strict winner, near-tie-adjusted winner, and the cross-pipeline winner
    seen4, oob = PC.seen_oob_matrices("kge_seenfolds_median4", "kge_heldout")
    seenc, _ = PC.seen_oob_matrices("kge_seendpl_canonical", "kge_heldout")

    def winners(M):
        return M.idxmax(axis=1)

    def margin12(M):
        v = np.sort(M.to_numpy(), axis=1)[:, ::-1]
        return v[:, 0] - v[:, 1]

    w_s4, w_o = winners(seen4), winners(oob)
    m_s4, m_o = margin12(seen4), margin12(oob)
    wc = winners(seenc)
    W = pd.DataFrame({
        "basin_id": seen4.index.astype(int),
        "strict_winner_retained": (w_s4 == w_o).to_numpy(),
        "seen_winner_samepipeline": w_s4.to_numpy(),
        "oob_winner": w_o.to_numpy(),
        "seen_margin12_samepipeline": m_s4,
        "oob_margin12": m_o,
        "near_tie_decided": (m_s4 >= 0.01) & (m_o >= 0.01),
        "strict_winner_retained_crosspipeline": (wc == w_o).to_numpy(),
    }).set_index("basin_id")
    W["near_tie_adjusted_retained"] = np.where(W["near_tie_decided"],
                                              W["strict_winner_retained"], np.nan)
    # prim already carries seen_winner / oob_winner; drop them before joining
    W = W.join(prim.drop(columns=[c for c in ("seen_winner", "oob_winner")
                                  if c in prim.columns]))
    W = W.reset_index()
    W.to_csv(PC.T3 / "winner_vs_rank_stability.csv", index=False)

    def grouped(mask, label):
        g = prim.loc[prim.index.isin(W.loc[mask, "basin_id"])]
        r = g["spearman_rho"].to_numpy()
        j3 = g["top3_jaccard"].to_numpy(); j5 = g["top5_jaccard"].to_numpy()
        return {"group": label, "n": int(len(g)),
                "spearman_median": float(np.median(r)), "spearman_mean": float(np.mean(r)),
                "spearman_frac_ge_0p5": float((r >= 0.5).mean()),
                "kendall_median": float(np.median(g["kendall_tau"])),
                "top3_jaccard_median": float(np.median(j3)),
                "top5_jaccard_median": float(np.median(j5)),
                "top3_full_set_retained": float((j3 == 1.0).mean()),
                "top5_full_set_retained": float((j5 == 1.0).mean())}

    ret = W["strict_winner_retained"].to_numpy()
    dec = W["near_tie_decided"].to_numpy()
    grp = [grouped(ret, "winner RETAINED"),
           grouped(~ret, "winner CHANGED"),
           grouped(dec, "decided basins (tau=0.01 both sides)"),
           grouped(~dec, "near-tie basins (tau<0.01 on either side)"),
           grouped(dec & ret, "decided AND retained"),
           grouped(dec & ~ret, "decided AND changed (the hard cases)"),
           grouped(np.ones(len(W), bool), "all basins")]
    G = pd.DataFrame(grp)
    G.to_csv(PC.T3 / "winner_vs_rank_stability_summary.csv", index=False)

    def mw(a, b):
        return float(sstats.mannwhitneyu(a, b, alternative="two-sided").pvalue)

    report["winner_vs_rank"] = {
        "strict_winner_retention": float(ret.mean()),
        "n_strict_retained": int(ret.sum()),
        "near_tie_decided_n": int(dec.sum()),
        "near_tie_adjusted_retention": float(W.loc[dec, "strict_winner_retained"].mean()),
        "groups": grp,
        "mannwhitney_spearman_retained_vs_changed": mw(
            prim.loc[W.loc[ret, "basin_id"], "spearman_rho"].to_numpy(),
            prim.loc[W.loc[~ret, "basin_id"], "spearman_rho"].to_numpy()),
        "mannwhitney_top3jaccard_retained_vs_changed": mw(
            prim.loc[W.loc[ret, "basin_id"], "top3_jaccard"].to_numpy(),
            prim.loc[W.loc[~ret, "basin_id"], "top3_jaccard"].to_numpy()),
        "spearman_of_spearman_vs_winner_retained": float(
            sstats.spearmanr(prim["spearman_rho"].to_numpy(), ret.astype(float)).statistic),
        "interpretation": ("compare the spearman_median of the winner-CHANGED group against "
                          "the all-basin value: if it stays high, top-1 swaps are local and the "
                          "within-basin ordering is largely preserved"),
    }
    with open(PC.T3 / "rank_stability.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ------------------------------------------------------------------ console
    log("== 1.1 within-basin rank correlation (36 models per basin, n=531) ==")
    for tag in ("samepipeline_median4", "samepipeline_mean4", "crosspipeline_canonical"):
        s = report["rank_correlation"][f"spearman_{tag}"]
        log(f"  [{tag}] spearman median={s['median']:.4f} CI95 "
            f"[{s['median_ci95'][0]:.4f}, {s['median_ci95'][1]:.4f}] | mean={s['mean']:.4f} | "
            f"q25/q75={s['q25']:.4f}/{s['q75']:.4f} | p05/p95={s['p05']:.4f}/{s['p95']:.4f}")
        log(f"      frac>0={s['frac_gt_0']:.3f}  >=0.3={s['frac_ge_0p3']:.3f}  "
            f">=0.5={s['frac_ge_0p5']:.3f}  >=0.7={s['frac_ge_0p7']:.3f}")
        k = report["rank_correlation"][f"kendall_{tag}"]
        log(f"      kendall median={k['median']:.4f} mean={k['mean']:.4f} "
            f"q25/q75={k['q25']:.4f}/{k['q75']:.4f}")
    log(f"  exact-tie matrices (ambiguity guard): {n_tie_pairs}")
    log("\n== 1.2 Top-k overlap ==")
    for tag in ("samepipeline_median4", "crosspipeline_canonical"):
        for k in KS:
            r = [x for x in topk_rows if x["variant"] == tag and x["k"] == k][0]
            log(f"  [{tag} k={k}] jaccard median={r['jaccard_median']:.4f} "
                f"mean={r['jaccard_mean']:.4f} q25/q75={r['jaccard_q25']:.4f}/{r['jaccard_q75']:.4f} "
                f"| full-set retained={r['full_set_retained_fraction']:.4f} "
                f"| mean intersection={r['intersection_mean']:.3f}/{k}")
            log(f"      intersection dist: {r['intersection_distribution']}")
    log("\n== 1.3 winner retention vs rank stability ==")
    log(G.round(4).to_string(index=False))
    log(f"\n  MW p (spearman, retained vs changed) = "
        f"{report['winner_vs_rank']['mannwhitney_spearman_retained_vs_changed']:.3e}")
    log(f"  MW p (top3 jaccard, retained vs changed) = "
        f"{report['winner_vs_rank']['mannwhitney_top3jaccard_retained_vs_changed']:.3e}")
    log(f"\nwrote tables -> {PC.T3}")


if __name__ == "__main__":
    main()
