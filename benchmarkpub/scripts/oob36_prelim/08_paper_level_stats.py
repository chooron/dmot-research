#!/usr/bin/env python3
"""08_paper_level_stats.py — paper-level preliminary OOB statistics.

Reads ONLY the frozen phase-2 artefacts under
$ROOT/analysis/chapter4_oob36_prelim_20260930/ and writes a new, separate
sub-directory paper_level_stats_20260930/.  No existing output is overwritten.

Scope (per the task): Q1 magnitude of the OOB penalty, Q2 heterogeneity across
models, Q3 stability of the relative ranking, Q4 complexity vs degradation,
plus the two "additional useful" checks (seen-performance dependence, basin
difficulty).  NO candidate set / equivalence set / oracle / headroom / G_set /
recommendation computation.

Definition used, verified from source before use
------------------------------------------------
Delta_KGE := KGE_seen-dPL - KGE_OOB-dPL        (computed here from the two columns)
so Delta_KGE > 0 means the model scored BETTER on the basins it was trained on
than on held-out basins, i.e. Delta_KGE is an OOB PENALTY.

Note (pre-existing label defect, NOT introduced here and NOT modified):
in `seen_ic_oob_merged.*` the upstream paired-table column `Delta_KGE`
(= KGE_dPL - KGE_IC, verified to 4.4e-16) was renamed to
`delta_kge_ic_minus_dpl`, which is the wrong name for its content.  That column
is not used anywhere in this script; KGE_IC and KGE_dPL are used directly.

Hierarchy: 36 models x 531 basins x 5 folds.  The 19,116 rows are NOT treated
as independent: basin-level bootstrap resamples BASINS (keeping all models for a
resampled basin together), and model-level rank inference uses permutation over
the 36 model labels.

Outputs (all under $OUT2):
    tables/model_oob_summary.csv
    tables/model_fold_oob_summary.csv
    tables/model_rank_shift.csv
    tables/basin_oob_summary.csv
    tables/pooled_delta_distribution.csv
    tables/leave_one_model_out_complexity.csv
    tables/seen_level_dependence.csv
    tables/preliminary_statistics.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, BENCH  # noqa: E402

OUT2 = os.path.join(OUT, "paper_level_stats_20260930")
T2 = os.path.join(OUT2, "tables")
MERGED = os.path.join(OUT, "tables", "seen_ic_oob_merged.parquet")
INVENTORY = os.path.join(OUT, "remote_inventory.csv")
V1_PARSIMONY = BENCH + "/analysis/chapter4_multicriteria_v3_20260925/tables/v1_parsimony.csv"

SEED = 20260930
B_BOOT = 2000
N_PERM = 10000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def boot_ci(values: np.ndarray, *, B: int = B_BOOT, seed: int = SEED, ci: float = 95.0):
    """Percentile CI of the median, resampling the supplied units."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, v.size, size=(B, v.size))], axis=1)
    return float(np.percentile(meds, (100 - ci) / 2)), float(np.percentile(meds, 100 - (100 - ci) / 2))


def clustered_boot(df: pd.DataFrame, value_col: str, *, cluster_col: str = "gauge_id",
                   stat: str = "median", B: int = B_BOOT, seed: int = SEED):
    """Bootstrap a pooled statistic, resampling CLUSTERS (basins) with replacement."""
    groups = [g[value_col].to_numpy(np.float64) for _, g in df.groupby(cluster_col, sort=True)]
    rng = np.random.default_rng(seed)
    n = len(groups)
    out = np.empty(B, dtype=np.float64)
    fn = np.median if stat == "median" else np.mean
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        out[b] = fn(np.concatenate([groups[i] for i in idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_spearman(x, y, *, n_perm: int = N_PERM, seed: int = SEED):
    """Spearman rho + two-sided permutation p over the supplied units.

    Permutation is over the units of `y` (model labels for model-level tests).
    Implemented as Pearson-on-ranks so the percentile can be vectorised.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = sstats.rankdata(x); ry = sstats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    n = rx.size
    rho = float(np.dot(rx, ry) / n)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        draws[i] = float(np.dot(rx, rng.permutation(ry)) / n)
    p = (float(np.sum(np.abs(draws) >= abs(rho) - 1e-12)) + 1) / (n_perm + 1)
    return rho, p


def theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(n) for j in range(i + 1, n) if x[j] != x[i]]
    return float(np.median(slopes))


def q(v, p):
    return float(np.percentile(np.asarray(v, float), p))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    os.makedirs(T2, exist_ok=True)

    df = pd.read_parquet(MERGED)
    if len(df) != 19116:
        raise ValueError(f"expected 19,116 rows, got {len(df)}")

    parsimony = pd.read_csv(V1_PARSIMONY)
    np_col = "n_parameters" if "n_parameters" in parsimony.columns else parsimony.columns[1]
    nparam = dict(zip(parsimony["model"], parsimony[np_col].astype(int)))

    inv = pd.read_csv(INVENTORY)
    cohort = dict(zip(inv["model"], inv["cohort"]))
    locn = dict(zip(inv["model"], inv["location"]))

    df = df.copy()
    df["delta_kge"] = df["kge_seen_dpl"] - df["kge_oob"]     # >0 = OOB penalty
    models = sorted(df["model"].unique())
    n_models = len(models)
    n_basins = df["gauge_id"].nunique()
    report = {"meta": {
        "n_models": n_models, "n_basins": n_basins, "n_rows": int(len(df)),
        "n_folds": int(df["fold"].nunique()),
        "seed": SEED, "bootstrap_B": B_BOOT, "permutations": N_PERM,
        "delta_definition": "KGE_seen_dPL - KGE_OOB_dPL  (>0 = OOB penalty)",
        "source_table": "analysis/chapter4_oob36_prelim_20260930/tables/seen_ic_oob_merged.parquet",
    }}

    # ======================================================================= #
    # Q1 — magnitude of the OOB penalty
    # ======================================================================= #
    rows = []
    for m in models:
        s = df[df["model"] == m]
        d = s["delta_kge"].to_numpy(np.float64)
        o = s["kge_oob"].to_numpy(np.float64)
        k = s["kge_seen_dpl"].to_numpy(np.float64)
        lo, hi = boot_ci(d)
        olo, ohi = boot_ci(o)
        rows.append({
            "model": m, "cohort": str(cohort.get(m, "")), "location": str(locn.get(m, "")),
            "n_params": int(nparam.get(m, -1)), "n_basins": int(len(s)),
            "n_folds": int(s["fold"].nunique()),
            "seen_kge_median": q(k, 50), "seen_kge_mean": float(np.mean(k)),
            "seen_kge_q25": q(k, 25), "seen_kge_q75": q(k, 75),
            "oob_kge_median": q(o, 50), "oob_kge_mean": float(np.mean(o)),
            "oob_kge_q25": q(o, 25), "oob_kge_q75": q(o, 75),
            "oob_kge_median_ci95_lo": olo, "oob_kge_median_ci95_hi": ohi,
            "delta_median": q(d, 50), "delta_mean": float(np.mean(d)),
            "delta_q25": q(d, 25), "delta_q75": q(d, 75),
            "delta_iqr": q(d, 75) - q(d, 25),
            "delta_median_ci95_lo": lo, "delta_median_ci95_hi": hi,
            "frac_delta_gt_0": float(np.mean(d > 0)),
            "frac_delta_lt_0": float(np.mean(d < 0)),
            "frac_delta_eq_0": float(np.mean(d == 0)),
            "frac_absdelta_lt_0p005": float(np.mean(np.abs(d) < 0.005)),
            "frac_absdelta_lt_0p01": float(np.mean(np.abs(d) < 0.01)),
            "frac_absdelta_lt_0p02": float(np.mean(np.abs(d) < 0.02)),
            "frac_absdelta_lt_0p05": float(np.mean(np.abs(d) < 0.05)),
        })
    S1 = pd.DataFrame(rows)
    # exact paired sign test per model (H0: median delta == 0), two-sided
    S1["sign_test_p_two_sided"] = [
        float(sstats.binomtest(int((df.loc[df["model"] == m, "delta_kge"] > 0).sum()),
                            int((df["model"] == m).sum()), 0.5).pvalue)
        for m in models]
    S1 = S1.sort_values("delta_median", ascending=False)
    S1.to_csv(os.path.join(T2, "model_oob_summary.csv"), index=False)

    # pooled distribution over all (model, basin) pairs
    d_all = df["delta_kge"].to_numpy(np.float64)
    pool_lo, pool_hi = clustered_boot(df, "delta_kge", stat="median")
    poolmean_lo, poolmean_hi = clustered_boot(df, "delta_kge", stat="mean")
    pooled_rows = [
        {"quantity": "median", "value": q(d_all, 50), "ci95_lo": pool_lo, "ci95_hi": pool_hi},
        {"quantity": "mean", "value": float(np.mean(d_all)), "ci95_lo": poolmean_lo, "ci95_hi": poolmean_hi},
        {"quantity": "q05", "value": q(d_all, 5), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "q25", "value": q(d_all, 25), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "q75", "value": q(d_all, 75), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "q95", "value": q(d_all, 95), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "frac_gt_0", "value": float(np.mean(d_all > 0)), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "frac_lt_0", "value": float(np.mean(d_all < 0)), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "frac_abs_lt_0p01", "value": float(np.mean(np.abs(d_all) < 0.01)), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "frac_abs_lt_0p02", "value": float(np.mean(np.abs(d_all) < 0.02)), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "frac_gt_0p5", "value": float(np.mean(d_all > 0.5)), "ci95_lo": None, "ci95_hi": None},
        {"quantity": "frac_gt_1p0", "value": float(np.mean(d_all > 1.0)), "ci95_lo": None, "ci95_hi": None},
    ]
    pd.DataFrame(pooled_rows).to_csv(os.path.join(T2, "pooled_delta_distribution.csv"), index=False)

    # concentration: is the pooled penalty driven by few models / few basins?
    model_sums = df.groupby("model")["delta_kge"].sum().sort_values(ascending=False)
    total = float(model_sums.sum())
    basin_sums = df.groupby("gauge_id")["delta_kge"].sum().sort_values(ascending=False)
    concentration = {
        "sum_delta_total": total,
        "worst_model": str(model_sums.index[0]),
        "worst_model_share_of_total_delta": float(model_sums.iloc[0] / total),
        "top5_model_share_of_total_delta": float(model_sums.iloc[:5].sum() / total),
        "top10_model_share_of_total_delta": float(model_sums.iloc[:10].sum() / total),
        "worst_basin_share_of_total_delta": float(basin_sums.iloc[0] / basin_sums.sum()),
        "top5pct_basin_share_of_total_delta": float(basin_sums.iloc[:int(0.05 * n_basins)].sum() / basin_sums.sum()),
        "median_delta_excluding_worst_model": float(
            df[df["model"] != model_sums.index[0]]["delta_kge"].median()),
        "median_delta_all_models": float(np.median(d_all)),
        "n_models_with_positive_median_delta": int((S1["delta_median"] > 0).sum()),
        "n_models_with_positive_mean_delta": int((S1["delta_mean"] > 0).sum()),
    }

    # exact paired sign tests at the two non-row levels of the hierarchy
    n_models_pos = int((S1["delta_median"] > 0).sum())
    n_foldcells_pos = int((df.groupby(["model", "fold"])["delta_kge"].median() > 0).sum())

    # where does the total delta budget come from?
    bsgn = df.groupby("gauge_id")["delta_kge"].median()
    pos_basins = bsgn.index[bsgn > 0]
    total_d = float(df["delta_kge"].sum())
    decomposition = {
        "sum_delta_total": total_d,
        "n_basins_median_delta_gt_0": int(len(pos_basins)),
        "n_basins_median_delta_lt_0": int((bsgn < 0).sum()),
        "sum_delta_from_majority_degrading_basins": float(
            df[df["gauge_id"].isin(pos_basins)]["delta_kge"].sum()),
        "share_of_total_delta_from_majority_degrading_basins": float(
            df[df["gauge_id"].isin(pos_basins)]["delta_kge"].sum() / total_d),
        "median_delta_drop_worst_basin": float(
            df[df["gauge_id"] != basin_sums.index[0]]["delta_kge"].median()),
        "median_delta_drop_top5pct_basins": float(
            df[~df["gauge_id"].isin(basin_sums.index[:int(0.05 * n_basins)])]["delta_kge"].median()),
    }

    report["Q1_magnitude"] = {
        "pooled": {r["quantity"]: r["value"] for r in pooled_rows},
        "concentration": concentration,
        "budget_decomposition": decomposition,
        "sign_tests": {
            "model_level_n_positive_of_36": n_models_pos,
            "model_level_exact_two_sided_p": float(sstats.binomtest(n_models_pos, n_models, 0.5).pvalue),
            "model_fold_level_n_positive_of_180": n_foldcells_pos,
            "model_fold_level_exact_two_sided_p": float(
                sstats.binomtest(n_foldcells_pos, n_models * int(df["fold"].nunique()), 0.5).pvalue),
            "per_model_sign_test_p_min": float(S1["sign_test_p_two_sided"].min()),
            "per_model_sign_test_p_median": float(S1["sign_test_p_two_sided"].median()),
            "per_model_sign_test_p_max": float(S1["sign_test_p_two_sided"].max()),
            "n_models_sign_test_p_lt_0p05": int((S1["sign_test_p_two_sided"] < 0.05).sum()),
        },
        "model_delta_median_range": [float(S1["delta_median"].min()), float(S1["delta_median"].max())],
        "top5_by_oob_median_kge": [[str(r.model), float(r.oob_kge_median)]
                                     for r in S1.nlargest(5, "oob_kge_median").itertuples()],
        "bottom5_by_oob_median_kge": [[str(r.model), float(r.oob_kge_median)]
                                       for r in S1.nsmallest(5, "oob_kge_median").itertuples()],
        "top5_by_delta_median": [[str(r.model), float(r.delta_median)]
                                  for r in S1.nlargest(5, "delta_median").itertuples()],
        "bottom5_by_delta_median": [[str(r.model), float(r.delta_median)]
                                     for r in S1.nsmallest(5, "delta_median").itertuples()],
    }

    # ======================================================================= #
    # Q2 — heterogeneity across models
    # ======================================================================= #
    mf_rows = []
    for m in models:
        s = df[df["model"] == m]
        for f in sorted(s["fold"].unique()):
            g = s[s["fold"] == f]
            mf_rows.append({
                "model": m, "fold": int(f), "n_basins": int(len(g)),
                "median_seen_KGE": float(np.median(g["kge_seen_dpl"])),
                "median_OOB_KGE": float(np.median(g["kge_oob"])),
                "median_delta_KGE": float(np.median(g["delta_kge"])),
                "mean_delta_KGE": float(np.mean(g["delta_kge"])),
                "sum_delta_KGE": float(g["delta_kge"].sum()),
                "frac_delta_gt_0": float(np.mean(g["delta_kge"] > 0)),
            })
    MF = pd.DataFrame(mf_rows)
    MF.to_csv(os.path.join(T2, "model_fold_oob_summary.csv"), index=False)

    fold_agree, fold_conc = [], []
    for m in models:
        g = MF[MF["model"] == m].sort_values("fold")
        meds = g["median_delta_KGE"].to_numpy(float)
        sums = g["sum_delta_KGE"].to_numpy(float)
        fold_agree.append({
            "model": m, "n_folds": int(len(meds)),
            "n_folds_median_delta_gt_0": int((meds > 0).sum()),
            "all_folds_positive": bool((meds > 0).all()),
            "fold_median_delta_sd": float(np.std(meds, ddof=1)),
            "fold_median_delta_range": float(meds.max() - meds.min()),
            "fold_median_delta_min": float(meds.min()),
            "fold_median_delta_max": float(meds.max()),
        })
        fold_conc.append({
            "model": m,
            "worst_fold_share_of_sum_delta": float(sums.max() / sums.sum()) if sums.sum() != 0 else None,
            "best_fold": int(g["fold"].iloc[int(np.argmax(sums))]),
        })
    FA = pd.DataFrame(fold_agree)
    FC = pd.DataFrame(fold_conc)

    d_med = S1["delta_median"].to_numpy(float)
    q2 = {
        "model_median_delta_distribution": {
            "min": float(d_med.min()), "q25": q(d_med, 25), "median": float(np.median(d_med)),
            "q75": q(d_med, 75), "max": float(d_med.max()),
            "iqr": q(d_med, 75) - q(d_med, 25),
            "range": float(d_med.max() - d_med.min()),
            "sd_ddof1": float(np.std(d_med, ddof=1)),
            "n_models": int(len(d_med)),
        },
        "least_degraded_model": str(S1.iloc[-1]["model"]),
        "least_degraded_delta_median": float(S1.iloc[-1]["delta_median"]),
        "most_degraded_model": str(S1.iloc[0]["model"]),
        "most_degraded_delta_median": float(S1.iloc[0]["delta_median"]),
        "ratio_most_over_least": float(S1.iloc[0]["delta_median"] / S1.iloc[-1]["delta_median"]),
        "n_models_all_folds_positive": int(FA["all_folds_positive"].sum()),
        "n_models_4of5_folds_positive": int((FA["n_folds_median_delta_gt_0"] >= 4).sum()),
        "fold_median_delta_sd_across_models": {
            "min": float(FA["fold_median_delta_sd"].min()),
            "median": float(FA["fold_median_delta_sd"].median()),
            "max": float(FA["fold_median_delta_sd"].max()),
        },
        "worst_fold_share_of_sum_delta": {
            "min": float(FC["worst_fold_share_of_sum_delta"].min()),
            "median": float(FC["worst_fold_share_of_sum_delta"].median()),
            "max": float(FC["worst_fold_share_of_sum_delta"].max()),
        },
    }
    # rank the models by delta_median and split into thirds to see if it is a continuum
    report["Q2_heterogeneity"] = q2

    # ======================================================================= #
    # Q3 — does OOB change the relative ranking?
    # ======================================================================= #
    ms = df.groupby("model")["kge_seen_dpl"].median().reindex(models)
    mo = df.groupby("model")["kge_oob"].median().reindex(models)
    rank_seen = ms.rank(ascending=False).astype(int)
    rank_oob = mo.rank(ascending=False).astype(int)
    rho_m, p_m = perm_spearman(ms.to_numpy(float), mo.to_numpy(float))
    tau_m = float(sstats.kendalltau(ms.to_numpy(float), mo.to_numpy(float)).statistic)

    rs = pd.DataFrame({
        "model": models, "n_params": [int(nparam.get(m, -1)) for m in models],
        "seen_kge_median": ms.to_numpy(float), "oob_kge_median": mo.to_numpy(float),
        "seen_rank": rank_seen.to_numpy(int), "oob_rank": rank_oob.to_numpy(int),
        "delta_median": S1.set_index("model")["delta_median"].reindex(models).to_numpy(float),
    })
    rs["rank_shift"] = rs["seen_rank"] - rs["oob_rank"]      # >0 = dropped in OOB ranking
    rs["abs_rank_shift"] = rs["rank_shift"].abs()
    rs = rs.sort_values("oob_rank")
    rs.to_csv(os.path.join(T2, "model_rank_shift.csv"), index=False)

    topk = {}
    for k in (3, 5, 10, 15):
        a = set(rs.nsmallest(k, "seen_rank")["model"])
        b = set(rs.nsmallest(k, "oob_rank")["model"])
        topk[str(k)] = {"overlap": len(a & b), "retention": len(a & b) / k,
                        "entered": sorted(b - a), "left": sorted(a - b)}

    # basin-level winners (descriptive only)
    def winner(col):
        return df.loc[df.groupby("gauge_id")[col].idxmax()].set_index("gauge_id")["model"]
    w_seen, w_oob = winner("kge_seen_dpl"), winner("kge_oob")
    w = pd.DataFrame({"seen_winner": w_seen, "oob_winner": w_oob})
    w["same"] = w["seen_winner"] == w["oob_winner"]
    oob_win_counts = w["oob_winner"].value_counts()
    seen_win_counts = w["seen_winner"].value_counts()
    trans = (w[w["same"] == False].groupby(["seen_winner", "oob_winner"]).size()
             .sort_values(ascending=False))
    winsummary = pd.DataFrame({
        "model": models,
        "n_basin_wins_seen": [int(seen_win_counts.get(m, 0)) for m in models],
        "n_basin_wins_oob": [int(oob_win_counts.get(m, 0)) for m in models],
    })
    winsummary["win_retention"] = winsummary["n_basin_wins_oob"] / winsummary["n_basin_wins_seen"].replace(0, np.nan)
    winsummary.to_csv(os.path.join(T2, "basin_winner_summary.csv"), index=False)
    (pd.DataFrame({"seen_winner": [i[0] for i in trans.index],
                   "oob_winner": [i[1] for i in trans.index],
                   "n_basins": trans.to_numpy(int)})
     .to_csv(os.path.join(T2, "basin_winner_transitions.csv"), index=False))

    report["Q3_ranking"] = {
        "spearman_model_median_seen_vs_oob": rho_m, "spearman_perm_p": p_m,
        "kendall_tau_model_median_seen_vs_oob": tau_m,
        "topk_retention": topk,
        "rank_shift_stats": {
            "max_abs_shift": int(rs["abs_rank_shift"].max()),
            "median_abs_shift": float(rs["abs_rank_shift"].median()),
            "q75_abs_shift": float(q(rs["abs_rank_shift"].to_numpy(float), 75)),
            "n_models_shift_ge_5": int((rs["abs_rank_shift"] >= 5).sum()),
            "n_models_shift_ge_10": int((rs["abs_rank_shift"] >= 10).sum()),
        },
        "basin_winner": {
            "n_basins": int(len(w)),
            "same_winner_fraction": float(w["same"].mean()),
            "n_basins_same_winner": int(w["same"].sum()),
            "n_distinct_models_winning_oob": int(w["oob_winner"].nunique()),
            "n_distinct_models_winning_seen": int(w["seen_winner"].nunique()),
            "top_oob_winner": str(oob_win_counts.index[0]),
            "top_oob_winner_basins": int(oob_win_counts.iloc[0]),
            "top_oob_winner_share": float(oob_win_counts.iloc[0] / len(w)),
            "most_common_transition": ([str(trans.index[0][0]), str(trans.index[0][1]), int(trans.iloc[0])]
                                       if len(trans) else None),
        },
    }

    # ======================================================================= #
    # Q4 — complexity vs degradation
    # ======================================================================= #
    x = np.array([nparam[m] for m in models], float)
    y = rs.set_index("model").reindex(models)["delta_median"].to_numpy(float)
    rho4, p4 = perm_spearman(x, y)
    tau4 = float(sstats.kendalltau(x, y).statistic)
    ols = np.polyfit(x, y, 1)
    ts = theil_sen(x, y)
    loo = []
    for i, m in enumerate(models):
        keep = np.arange(len(models)) != i
        r = float(sstats.spearmanr(x[keep], y[keep]).statistic)
        loo.append({"omitted_model": m, "n_params": float(x[i]),
                    "delta_median": float(y[i]), "spearman_rho": r})
    LOO = pd.DataFrame(loo)
    LOO.to_csv(os.path.join(T2, "leave_one_model_out_complexity.csv"), index=False)

    keep = np.array([m != "penman" for m in models])
    rho_np, p_np = perm_spearman(x[keep], y[keep])
    q4 = {
        "n_models": int(len(models)),
        "spearman_rho": rho4, "spearman_perm_p": p4,
        "kendall_tau": tau4,
        "ols_slope_delta_per_parameter": float(ols[0]), "ols_intercept": float(ols[1]),
        "theil_sen_slope": ts,
        "n_params_range": [float(x.min()), float(x.max())],
        "leave_one_model_out_rho": {
            "min": float(LOO["spearman_rho"].min()), "max": float(LOO["spearman_rho"].max()),
            "median": float(LOO["spearman_rho"].median()),
            "worst_case_model": str(LOO.loc[LOO["spearman_rho"].idxmin(), "omitted_model"]),
            "best_case_model": str(LOO.loc[LOO["spearman_rho"].idxmax(), "omitted_model"]),
            "sign_flips": int((np.sign(LOO["spearman_rho"]) != np.sign(rho4)).sum()),
        },
        "excluding_penman": {"n": int(keep.sum()), "spearman_rho": rho_np, "perm_p": p_np},
    }
    report["Q4_complexity"] = q4

    # ======================================================================= #
    # A — seen-performance dependence of the penalty
    # ======================================================================= #
    pooled_rho = float(sstats.spearmanr(df["kge_seen_dpl"], df["delta_kge"]).statistic)
    within_model = []
    for m in models:
        s = df[df["model"] == m]
        within_model.append({"model": m,
                             "spearman_rho_seen_vs_delta": float(
                                 sstats.spearmanr(s["kge_seen_dpl"], s["delta_kge"]).statistic)})
    WM = pd.DataFrame(within_model)
    within_basin = []
    for g, s in df.groupby("gauge_id"):
        if s["kge_seen_dpl"].nunique() > 1:
            within_basin.append(float(sstats.spearmanr(s["kge_seen_dpl"], s["delta_kge"]).statistic))
    rho_model_level, p_model_level = perm_spearman(ms.to_numpy(float), y)
    dep = pd.DataFrame([
        {"level": "model (n=36)", "quantity": "median seen KGE vs median delta_KGE",
         "statistic": "spearman", "value": rho_model_level, "p_perm": p_model_level},
        {"level": "pooled rows (n=19116, NOT independent)", "quantity": "seen KGE vs delta_KGE",
         "statistic": "spearman", "value": pooled_rho, "p_perm": None},
        {"level": "within model (36 x spearman)", "quantity": "seen KGE vs delta_KGE",
         "statistic": "median spearman", "value": float(WM["spearman_rho_seen_vs_delta"].median()),
         "p_perm": None},
        {"level": "within basin (531 x spearman)", "quantity": "seen KGE vs delta_KGE",
         "statistic": "median spearman", "value": float(np.median(within_basin)), "p_perm": None},
    ])
    dep.to_csv(os.path.join(T2, "seen_level_dependence.csv"), index=False)
    report["A_seen_dependence"] = {
        "model_level": {"spearman": rho_model_level, "perm_p": p_model_level},
        "pooled_rows_descriptive": {"spearman": pooled_rho, "n_rows": int(len(df))},
        "within_model_median_spearman": float(WM["spearman_rho_seen_vs_delta"].median()),
        "within_model_range": [float(WM["spearman_rho_seen_vs_delta"].min()),
                               float(WM["spearman_rho_seen_vs_delta"].max())],
        "within_basin_median_spearman": float(np.median(within_basin)),
        "within_basin_range": [float(np.min(within_basin)), float(np.max(within_basin))],
        "caveat": "regression to the mean is not controlled for; descriptive only",
    }

    # ======================================================================= #
    # B — basin difficulty
    # ======================================================================= #
    bs = df.groupby("gauge_id").agg(
        fold=("fold", "first"),
        n_models=("model", "nunique"),
        seen_kge_median=("kge_seen_dpl", "median"),
        oob_kge_median=("kge_oob", "median"),
        delta_median=("delta_kge", "median"),
        delta_mean=("delta_kge", "mean"),
        delta_q25=("delta_kge", lambda v: np.percentile(v, 25)),
        delta_q75=("delta_kge", lambda v: np.percentile(v, 75)),
        frac_models_delta_gt_0=("delta_kge", lambda v: float(np.mean(v > 0))),
        frac_models_delta_lt_0=("delta_kge", lambda v: float(np.mean(v < 0))),
        worst_model_delta=("delta_kge", "min"),
        best_model_delta=("delta_kge", "max"),
    ).reset_index()
    bs["gauge_id_str"] = [str(int(g)).zfill(8) for g in bs["gauge_id"]]
    bs = bs.sort_values("delta_median")
    bs.to_csv(os.path.join(T2, "basin_oob_summary.csv"), index=False)

    bd = bs["delta_median"].to_numpy(float)
    report["B_basin_difficulty"] = {
        "basin_median_delta_distribution": {
            "min": float(bd.min()), "q05": q(bd, 5), "q25": q(bd, 25), "median": float(np.median(bd)),
            "q75": q(bd, 75), "q95": q(bd, 95), "max": float(bd.max()),
            "iqr": q(bd, 75) - q(bd, 25), "sd_ddof1": float(np.std(bd, ddof=1)),
        },
        "n_basins": int(len(bs)),
        "n_basins_all36_models_degrade": int((bs["frac_models_delta_gt_0"] == 1.0).sum()),
        "n_basins_ge_30_of_36_degrade": int((bs["frac_models_delta_gt_0"] >= 30 / 36).sum()),
        "n_basins_ge_24_of_36_degrade": int((bs["frac_models_delta_gt_0"] >= 24 / 36).sum()),
        "n_basins_le_half_models_degrade": int((bs["frac_models_delta_gt_0"] <= 0.5).sum()),
        "largest_penalty_basin": {"gauge_id_str": str(bs.iloc[0]["gauge_id_str"]),
                                 "delta_median": float(bs.iloc[0]["delta_median"]),
                                 "oob_kge_median": float(bs.iloc[0]["oob_kge_median"]),
                                 "seen_kge_median": float(bs.iloc[0]["seen_kge_median"])},
        "smallest_penalty_basin": {"gauge_id_str": str(bs.iloc[-1]["gauge_id_str"]),
                                  "delta_median": float(bs.iloc[-1]["delta_median"]),
                                  "oob_kge_median": float(bs.iloc[-1]["oob_kge_median"]),
                                  "seen_kge_median": float(bs.iloc[-1]["seen_kge_median"])},
        "spread_largest_minus_smallest_penalty": float(
            bs.iloc[0]["delta_median"] - bs.iloc[-1]["delta_median"]),
        "n_basins_with_penalty_gt_0p5": int((bs["delta_median"] > 0.5).sum()),
        "n_basins_with_penalty_lt_0": int((bs["delta_median"] < 0).sum()),
    }
    # ======================================================================= #
    report["descriptive_thresholds_note"] = (
        "No frozen project convention for a 'near-zero Delta_KGE' exists; the "
        "near-zero bands reported here (|d| < 0.005 / 0.01 / 0.02 / 0.05 KGE) are "
        "DESCRIPTIVE ONLY and are reported alongside the exact sign counts.")
    report["outputs"] = {
        "model_oob_summary": "tables/model_oob_summary.csv",
        "model_fold_oob_summary": "tables/model_fold_oob_summary.csv",
        "model_rank_shift": "tables/model_rank_shift.csv",
        "basin_oob_summary": "tables/basin_oob_summary.csv",
        "pooled_delta_distribution": "tables/pooled_delta_distribution.csv",
        "leave_one_model_out_complexity": "tables/leave_one_model_out_complexity.csv",
        "seen_level_dependence": "tables/seen_level_dependence.csv",
    }
    with open(os.path.join(T2, "preliminary_statistics.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    # -------- console summary --------
    print("== Q1 magnitude ==")
    print(f"   pooled median delta = {q(d_all,50):.4f}  CI95 [{pool_lo:.4f}, {pool_hi:.4f}]  "
          f"(basin-clustered bootstrap B={B_BOOT})")
    print(f"   pooled mean   delta = {np.mean(d_all):.4f}  CI95 [{poolmean_lo:.4f}, {poolmean_hi:.4f}]")
    print(f"   pairs: frac>0 {np.mean(d_all>0):.4f}  frac<0 {np.mean(d_all<0):.4f}  "
          f"|d|<0.01 {np.mean(np.abs(d_all)<0.01):.4f}")
    print(f"   worst model {concentration['worst_model']} share of total delta = "
          f"{concentration['worst_model_share_of_total_delta']:.4f}; top5 share = "
          f"{concentration['top5_model_share_of_total_delta']:.4f}")
    print(f"   top5% basins share of total delta = {concentration['top5pct_basin_share_of_total_delta']:.4f}")
    st = report["Q1_magnitude"]["sign_tests"]
    print(f"   sign tests: model level {st['model_level_n_positive_of_36']}/36 p={st['model_level_exact_two_sided_p']:.2e}; "
          f"model x fold {st['model_fold_level_n_positive_of_180']}/180 p={st['model_fold_level_exact_two_sided_p']:.2e}")
    print(f"   per-model sign test: p max {st['per_model_sign_test_p_max']:.2e}, "
          f"{st['n_models_sign_test_p_lt_0p05']}/36 models p<0.05")
    print(f"   budget: {decomposition['n_basins_median_delta_gt_0']} basins with median delta>0 "
          f"carry {decomposition['share_of_total_delta_from_majority_degrading_basins']:.3f} of sum(delta)")
    print(f"   median delta after dropping worst basin: {decomposition['median_delta_drop_worst_basin']:.4f}; "
          f"after dropping top-5% basins: {decomposition['median_delta_drop_top5pct_basins']:.4f}")
    print("   top5 by OOB median KGE: " + ", ".join(
        f"{m}={v:.4f}" for m, v in report['Q1_magnitude']['top5_by_oob_median_kge']))
    print("   bottom5 by OOB median KGE: " + ", ".join(
        f"{m}={v:.4f}" for m, v in report['Q1_magnitude']['bottom5_by_oob_median_kge']))
    print("\n== Q2 heterogeneity ==")
    for k, v in q2["model_median_delta_distribution"].items():
        print(f"   model median delta {k:8s} {v:.4f}")
    print(f"   most degraded {q2['most_degraded_model']} {q2['most_degraded_delta_median']:.4f}; "
          f"least {q2['least_degraded_model']} {q2['least_degraded_delta_median']:.4f}; "
          f"ratio {q2['ratio_most_over_least']:.2f}")
    print(f"   all-folds-positive models: {q2['n_models_all_folds_positive']}/36;  "
          f"4-of-5: {q2['n_models_4of5_folds_positive']}/36")
    print(f"   worst-fold share of sum(delta): median {q2['worst_fold_share_of_sum_delta']['median']:.3f} "
          f"max {q2['worst_fold_share_of_sum_delta']['max']:.3f}")
    print("\n== Q3 ranking ==")
    print(f"   spearman {rho_m:.4f} (perm p={p_m:.4f})  kendall {tau_m:.4f}")
    print(f"   top-k retention: " + "  ".join(f"k={k}:{v['overlap']}/{k}" for k, v in topk.items()))
    print(f"   rank shift: max {rs['abs_rank_shift'].max()}  median {rs['abs_rank_shift'].median():.1f}  "
          f">=5: {(rs['abs_rank_shift']>=5).sum()}  >=10: {(rs['abs_rank_shift']>=10).sum()}")
    print(f"   basin winner retention = {w['same'].mean():.4f}  ({int(w['same'].sum())}/{len(w)}); "
          f"oob winners distinct = {w['oob_winner'].nunique()}")
    print("\n== Q4 complexity ==")
    print(f"   spearman {rho4:.4f} (perm p={p4:.4f})  kendall {tau4:.4f}")
    print(f"   OLS slope {ols[0]:.5f} per param; Theil-Sen {ts:.5f}")
    print(f"   leave-one-out rho range [{LOO['spearman_rho'].min():.4f}, {LOO['spearman_rho'].max():.4f}], "
          f"sign flips {(np.sign(LOO['spearman_rho'])!=np.sign(rho4)).sum()}")
    print(f"   excl penman: rho {rho_np:.4f} p {p_np:.4f} (n={int(keep.sum())})")
    print("\n== A seen dependence ==")
    print(f"   model level rho {rho_model_level:.4f} p {p_model_level:.4f}; "
          f"within-model median rho {np.median(WM['spearman_rho_seen_vs_delta']):.4f}; "
          f"within-basin median rho {np.median(within_basin):.4f}")
    print("\n== B basin difficulty ==")
    for k, v in report["B_basin_difficulty"]["basin_median_delta_distribution"].items():
        print(f"   basin median delta {k:6s} {v:.4f}")
    print(f"   basins where all 36 models degrade: {report['B_basin_difficulty']['n_basins_all36_models_degrade']}")
    print(f"   basins >=30/36 degrade: {report['B_basin_difficulty']['n_basins_ge_30_of_36_degrade']}; "
          f">=24/36: {report['B_basin_difficulty']['n_basins_ge_24_of_36_degrade']}; "
          f"<=18/36: {report['B_basin_difficulty']['n_basins_le_half_models_degrade']}")
    print(f"\nwrote tables -> {T2}")


if __name__ == "__main__":
    main()
