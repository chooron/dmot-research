#!/usr/bin/env python3
"""sec35_01_penalty_by_type_g1.py — G1: PUB-penalty differences across external basin types, and G1b descriptive single-attribute check.

G1: response is the per-basin median OOB penalty (`penalty_median_pub`: median over the 36 models of
    `delta_kge_pub_median4`, recomputed by basin-ID join). Total = 515 type-matched basins; the 16 unmatched
    basins are reported and excluded from testing. Kruskal-Wallis (10 types) with a rank-based effect size,
    per-type quantiles, and all 45 pairwise Dunn comparisons with Holm correction.
G1b: descriptive only — per-type Spearman rho between the smallest-q attribute of the 15-attribute screening
    (recomputed A4 table) and the same penalty. No test, no multiplicity correction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEC34 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec34_determinacy_and_bounds" / "tables"


def dunn_pairwise(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Dunn's z-test on average ranks with tie correction (two-sided), all pairs."""
    ranks = stats.rankdata(values)
    N = len(values)
    groups = sorted(set(labels.tolist()))
    # tie correction
    _, counts = np.unique(values, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    sigma2 = (N * (N + 1) / 12.0) - tie_term / (12.0 * (N - 1))
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            ma, mb = labels == a, labels == b
            ra, rb = ranks[ma].mean(), ranks[mb].mean()
            na, nb = int(ma.sum()), int(mb.sum())
            z = (ra - rb) / np.sqrt(sigma2 * (1.0 / na + 1.0 / nb))
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            rows.append({"type_a": a, "type_b": b, "n_a": na, "n_b": nb,
                         "mean_rank_a": float(ra), "mean_rank_b": float(rb),
                         "rank_diff": float(ra - rb), "z": float(z), "p_uncorrected": float(p)})
    df = pd.DataFrame(rows)
    # Holm correction over the whole family
    m = len(df)
    order = np.argsort(df["p_uncorrected"].to_numpy())
    p_sorted = df["p_uncorrected"].to_numpy()[order]
    adj = np.empty(m)
    running = 0.0
    for idx, p in enumerate(p_sorted):
        val = (m - idx) * p
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    p_holm = np.empty(m)
    p_holm[order] = adj
    df["p_holm"] = p_holm
    df["family_size"] = m
    df["correction"] = "Holm"
    df["reject_005_holm"] = df["p_holm"] < 0.05
    return df.sort_values(["type_a", "type_b"]).reset_index(drop=True)


def main() -> None:
    basins = S.canonical_basin_ids()
    pos = S.basin_position_map(basins)

    cf = pd.read_csv(ROOT / "data" / "gauge_cluster.csv")
    cf["bs"] = cf["gauge_index"].map(S.pad_basin_id)
    cmap = dict(zip(cf["bs"], cf["gauge_cluster"]))
    matched = [b for b in basins if b in cmap]
    unmatched = [b for b in basins if b not in cmap]

    pub = pd.read_csv(S.DIR_OOB36_PRELIM / "pub_next_stage_20260930" / "tables" / "pub_same_pipeline_model_basin.csv")
    pub["bs"] = pub["basin_id"].map(S.pad_basin_id)
    pen_series = pub.groupby("bs")["delta_kge_pub_median4"].median()
    penalty = S.align_series_by_id(pen_series.index, pen_series.to_numpy(), canonical=basins, name="penalty_median_pub")

    types = np.array([cmap[b] for b in matched])
    pen_m = np.array([penalty[pos[b]] for b in matched])
    assert len(matched) + len(unmatched) == len(basins) == 531

    # ---- G1: Kruskal-Wallis + per-type quantiles + pairwise Dunn/Holm ----
    groups = [pen_m[types == t] for t in sorted(set(types.tolist()))]
    H, p_kw = stats.kruskal(*groups)
    k = len(groups)
    n = len(pen_m)
    eps2 = (H - k + 1.0) / (n - k)          # rank epsilon-squared (Kerby form)

    type_rows = []
    for t in sorted(set(types.tolist())):
        v = pen_m[types == t]
        type_rows.append({"type": int(t), "n": int(v.size), "median": float(np.median(v)),
                          "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
                          "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                          "mean_rank": float(stats.rankdata(pen_m)[types == t].mean())})
    df_types = pd.DataFrame(type_rows)
    S.write_tidy_table(df_types, TABLES / "sec35_g1_penalty_by_type.csv")

    S.write_tidy_table(pd.DataFrame([{
        "test": "Kruskal-Wallis",
        "statistic": float(H), "df": int(k - 1), "p_value": float(p_kw), "p_value_scientific": f"{float(p_kw):.6e}",
        "n_total": int(n), "n_types": int(k),
        "effect_size_name": "rank epsilon-squared (epsilon^2)",
        "effect_size_definition": "epsilon^2 = (H - k + 1) / (n - k), H = Kruskal-Wallis statistic, k = number of types, n = number of basins",
        "effect_size_value": float(eps2),
    }]), TABLES / "sec35_g1_kruskal_wallis.csv")

    df_pairs = dunn_pairwise(pen_m, types)
    S.write_tidy_table(df_pairs, TABLES / "sec35_g1_pairwise_dunn_holm.csv")

    S.write_tidy_table(pd.DataFrame([{
        "n_basins_total": len(basins), "n_matched": len(matched), "n_unmatched": len(unmatched),
        "n_types": int(k), "unmatched_basin_ids": "|".join(unmatched),
        "type_counts": "|".join(f"{t}:{int((types == t).sum())}" for t in sorted(set(types.tolist()))),
        "grouping_source": "data/gauge_cluster.csv (external published classification), used as a grouping factor only",
    }]), TABLES / "sec35_g1_matched_summary.csv")

    # ---- G1b: descriptive single-attribute check (no test, no correction) ----
    a4 = pd.read_csv(SEC34 / "sec34_a4_attribute_screening_recompute.csv")
    qmin = a4.loc[a4["recomputed_bh_fdr_q"].idxmin()]
    attr = str(qmin["variable"])
    qval = float(qmin["recomputed_bh_fdr_q"])
    attr_raw = pd.read_csv(ROOT / "data" / "caravan_671_attributes.csv")
    attr_raw["bs"] = attr_raw["gauge_id"].map(S.pad_basin_id)
    if attr not in attr_raw.columns:
        # observed-series attribute: recompute from the frozen OOB observation files by basin ID
        obs = np.full((5479, len(basins)), np.nan)
        for f in range(5):
            ids_path = S.DIR_OOB36_DATA / "runs" / S.MODELS_36[0] / f"fold_{f}" / "heldout_basin_ids.txt"
            fold_ids = [int(x) for x in ids_path.read_text().split()]
            y = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
            for j, bid in enumerate(fold_ids):
                obs[:, pos[S.pad_basin_id(bid)]] = y[:, j]
        with np.errstate(invalid="ignore"):
            q_mean = np.nanmean(obs, axis=0)
            q_sd = np.nanstd(obs, axis=0, ddof=1)
            series = {"q_mean": q_mean, "q_cv": q_sd / q_mean,
                      "zero_flow_frac": np.nanmean(obs <= 0.0, axis=0)}
            eps = 0.01 * q_mean
            series["q_log_sd"] = np.nanstd(np.log(np.maximum(obs, 0) + eps[None, :]), axis=0, ddof=1)
        p_mean = S.align_series_by_id(attr_raw["bs"], attr_raw["p_mean"].to_numpy(), canonical=basins, name="p_mean")
        series["runoff_ratio"] = q_mean / p_mean
        attr_vals = series[attr]
    else:
        attr_vals = S.align_series_by_id(attr_raw["bs"], attr_raw[attr].to_numpy(), canonical=basins, name=attr)

    g1b_rows = []
    for t in sorted(set(types.tolist())):
        m = types == t
        x = np.array([attr_vals[pos[b]] for b in np.array(matched)[m]], dtype=float)
        y = pen_m[m]
        ok = np.isfinite(x) & np.isfinite(y)
        rho = float(stats.spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 2 and np.std(x[ok]) > 0 else np.nan
        g1b_rows.append({"type": int(t), "n": int(ok.sum()), "attribute": attr,
                         "attribute_global_bh_fdr_q": qval, "spearman_rho": rho,
                         "scope_note": "descriptive only; no significance test and no multiplicity correction"})
    S.write_tidy_table(pd.DataFrame(g1b_rows), TABLES / "sec35_g1b_attribute_penalty_spearman_by_type.csv")

    print(f"[G1] matched={len(matched)} unmatched={len(unmatched)} types={k}")
    print(f"[G1] Kruskal-Wallis H={H:.4f} df={k-1} p={p_kw:.6e} epsilon^2={eps2:.4f}")
    print(df_types.to_string(index=False))
    print(f"[G1] pairwise pairs={len(df_pairs)} significant(Holm<0.05)={int(df_pairs.reject_005_holm.sum())}")
    print(f"[G1b] attribute={attr} (global q={qval:.6f}); per-type rho:")
    print(pd.DataFrame(g1b_rows)[["type", "n", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
