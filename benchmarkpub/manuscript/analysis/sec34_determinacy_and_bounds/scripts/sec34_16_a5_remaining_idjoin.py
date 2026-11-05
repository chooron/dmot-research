#!/usr/bin/env python3
"""sec34_16_a5_remaining_idjoin.py — A5: ID-joined re-verification of remaining basin-joined items.

Recomputes with strict basin-ID joins and compares against existing values:
  1. External basin classification matching count (existing 515/531) and unmatched basin list.
  2. Cluster (type) leave-out validation coverage and regret at k=3, k=5 (global vs cluster sets).
  3. Fold composition table: 13 attributes x 5 folds + overall (median/Q25/Q75/mean/min/max).
  4. Penalty concentration: top 5% basin share of total Delta_KGE (existing 34.25%) and top-5-model share.
  5. Coverage ladder: Seen Top-k -> OOB Top-1 coverage at k=1,3,5,10 (existing 0.2524/0.5028/0.6271/0.8041)
     and candidate-set regret medians.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
SEC31 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec31_transfer_and_library" / "tables"
SEC32 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec32_library_members" / "tables"
SEC33 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec33_usage_and_conditioning" / "tables"

ATTR_COLS = ["lat", "lon", "arid", "f_snow", "runoff_ratio", "p_mean", "pet_mean", "p_seas",
             "elev", "slope", "area", "f_forest", "zero_flow_frac"]


def main() -> None:
    models = S.MODELS_36
    models_arr = np.array(models)
    basins = S.canonical_basin_ids()      # A6 accessor
    nb = len(basins)
    bi = S.basin_position_map(basins)     # A6 accessor

    pub = pd.read_csv(S.DIR_OOB36_PRELIM / "pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv")
    pub["bs"] = pub["basin_id"].astype(str).str.zfill(8)

    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])

    comp_rows = []

    # =====================================================================
    # 1. External basin classification matching (515/531)
    # =====================================================================
    cf = pd.read_csv(ROOT / "data" / "gauge_cluster.csv")
    cf["bs"] = cf["gauge_index"].map(S.pad_basin_id)   # A6 accessor
    cluster_map = dict(zip(cf["bs"], cf["gauge_cluster"]))
    matched = [b for b in basins if b in cluster_map]
    unmatched = [b for b in basins if b not in cluster_map]
    n_matched = len(matched)
    exist_audit = pd.read_csv(S.DIR_CLUSTER_PROFILES / "tables" / "cluster_join_audit.csv")
    e_matched = int(exist_audit[exist_audit.metric == "matched_basins_count"]["value"].iloc[0])
    e_unmatched_ids = str(exist_audit[exist_audit.metric == "unmatched_basin_ids"]["value"].iloc[0]).split(";")

    comp_rows.append({
        "item": "cluster_matching_count",
        "sub_item": "matched_basins",
        "old_value": e_matched,
        "new_value": n_matched,
        "abs_diff": abs(e_matched - n_matched),
    })
    comp_rows.append({
        "item": "cluster_matching_count",
        "sub_item": "unmatched_basins",
        "old_value": 531 - e_matched,
        "new_value": len(unmatched),
        "abs_diff": abs((531 - e_matched) - len(unmatched)),
    })
    comp_rows.append({
        "item": "cluster_matching_ids",
        "sub_item": "unmatched_id_set",
        "old_value": ";".join(e_unmatched_ids),
        "new_value": ";".join(unmatched),
        "abs_diff": 0 if set(e_unmatched_ids) == set(unmatched) else 1,
    })
    print(f"[sec34_16] cluster matching: matched={n_matched}/531 (existing {e_matched}), "
          f"unmatched_id_set_equal={set(e_unmatched_ids) == set(unmatched)}")

    # =====================================================================
    # 2. Cluster leave-out validation: coverage & regret (k=3, k=5)
    # =====================================================================
    seen = S.pivot_by_id(pub, "bs", "model", "kge_seenfolds_median4",
                         canonical=basins, columns_order=models, name="seen")   # A6 accessor
    oob = S.pivot_by_id(pub, "bs", "model", "kge_heldout",
                        canonical=basins, columns_order=models, name="oob")     # A6 accessor
    basin_clusters = np.array([cluster_map.get(b, -1) for b in basins])
    seen_order = np.argsort(-seen, axis=1, kind="stable")
    oob_order = np.argsort(-oob, axis=1, kind="stable")

    matched_idx = np.flatnonzero(basin_clusters >= 0)

    exist_cl = pd.read_csv(SEC33 / "sec33_cluster_oob_validation_summary.csv").set_index("k")

    for k in [3, 5]:
        ind_seen = np.zeros((nb, len(models)), dtype=float)
        for i in range(nb):
            ind_seen[i, seen_order[i, :k]] = 1.0

        glob_sets, clust_sets = {}, {}
        for f in range(5):
            tr_all = (folds != f)
            freq_g = ind_seen[tr_all].mean(axis=0)
            score_g = seen[tr_all].mean(axis=0)
            order_g = sorted(range(len(models)), key=lambda idx: (-freq_g[idx], -score_g[idx]))
            glob_sets[f] = set(models_arr[order_g[:k]])
            clust_sets[f] = {}
            for c in range(10):
                tr_c = (folds != f) & (basin_clusters == c)
                freq_c = ind_seen[tr_c].mean(axis=0)
                score_c = seen[tr_c].mean(axis=0)
                order_c = sorted(range(len(models)), key=lambda idx: (-freq_c[idx], -score_c[idx]))
                clust_sets[f][c] = set(models_arr[order_c[:k]])

        g_cov = np.empty(len(matched_idx), dtype=bool)
        c_cov = np.empty(len(matched_idx), dtype=bool)
        g_reg = np.empty(len(matched_idx), dtype=float)
        c_reg = np.empty(len(matched_idx), dtype=float)
        for j, idx in enumerate(matched_idx):
            b = basins[idx]
            f = folds[idx]
            c = int(basin_clusters[idx])
            oob_win = models[oob_order[idx, 0]]
            vbs = oob[idx].max()
            gset = glob_sets[f]
            cset = clust_sets[f][c]
            g_cov[j] = oob_win in gset
            c_cov[j] = oob_win in cset
            g_reg[j] = vbs - oob[idx, [models.index(m) for m in gset]].max()
            c_reg[j] = vbs - oob[idx, [models.index(m) for m in cset]].max()

        e = exist_cl.loc[k]
        for name, new, old in [
            ("global_coverage", float(g_cov.mean()), float(e["global_coverage"])),
            ("cluster_coverage", float(c_cov.mean()), float(e["cluster_coverage"])),
            ("diff_coverage", float(c_cov.mean() - g_cov.mean()), float(e["diff_coverage"])),
            ("global_regret_mean", float(g_reg.mean()), float(e["global_regret_mean"])),
            ("cluster_regret_mean", float(c_reg.mean()), float(e["cluster_regret_mean"])),
            ("diff_regret_mean", float(c_reg.mean() - g_reg.mean()), float(e["diff_regret_mean"])),
            ("global_coverage_count", float(g_cov.sum()), float(e["global_coverage_count"])),
            ("cluster_coverage_count", float(c_cov.sum()), float(e["cluster_coverage_count"])),
        ]:
            comp_rows.append({
                "item": f"cluster_validation_k{k}",
                "sub_item": name,
                "old_value": old,
                "new_value": new,
                "abs_diff": abs(old - new),
            })
        print(f"[sec34_16] cluster validation k={k}: global cov {g_cov.mean():.4f} (exist {float(e['global_coverage']):.4f}), "
              f"cluster cov {c_cov.mean():.4f} (exist {float(e['cluster_coverage']):.4f}), "
              f"diff regret {c_reg.mean()-g_reg.mean():+.5f} (exist {float(e['diff_regret_mean']):+.5f})")

    # =====================================================================
    # 3. Fold composition (13 attributes)
    # =====================================================================
    fa_df = pd.read_csv(S.DIR_OOB36_DATA / "manifests" / "OOB_FOLD_ASSIGNMENT.csv")
    fa_df["bs"] = fa_df["basin_id"].map(S.pad_basin_id)   # A6 accessor
    loc_file = ROOT / "project/flexmopex/archive/20261003_cleanup/legacy_training/results/formal_671_unified_nmul4/lambda_invariance/inputs/gauge_locations_671.csv"
    loc_df = pd.read_csv(loc_file)
    loc_df["bs"] = loc_df["gage_id"].map(S.pad_basin_id)   # A6 accessor
    attr_df = pd.read_csv(S.DIR_OOB36_NEXT_STAGE / "tables" / "basin_attribute_table.csv")
    attr_df["bs"] = attr_df["basin_id"].map(S.pad_basin_id)   # A6 accessor

    merged = fa_df.merge(loc_df[["bs", "lat", "lon"]], on="bs", how="inner")
    merged = merged.merge(attr_df[["bs", "arid", "f_snow", "runoff_ratio", "p_mean", "pet_mean",
                                   "p_seas", "elev", "slope", "area", "f_forest", "zero_flow_frac"]],
                          on="bs", how="inner")
    assert len(merged) == 531, f"fold composition merge produced {len(merged)} rows"

    exist_fc = pd.read_csv(SEC31 / "sec31_fold_composition.csv")
    new_fc_rows = []
    for scope_name, sub in [("all_531", merged)] + [(f"fold_{k}", merged[merged.fold == k]) for k in range(5)]:
        for col in ATTR_COLS:
            vals = sub[col].to_numpy(dtype=float)
            new_fc_rows.append({
                "scope": scope_name, "n_basins": len(sub), "attribute": col,
                "median": float(np.median(vals)), "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)), "mean": float(np.mean(vals)),
                "min": float(vals.min()), "max": float(vals.max()),
            })
    new_fc = pd.DataFrame(new_fc_rows)
    S.write_tidy_table(new_fc, TABLES / "sec34_a5_fold_composition_idjoin.csv")

    fc_cmp = exist_fc.merge(new_fc, on=["scope", "attribute"], suffixes=("_old", "_new"))
    fc_diffs = []
    for stat in ["median", "q25", "q75", "mean", "min", "max"]:
        fc_diffs.append(np.abs(fc_cmp[f"{stat}_old"] - fc_cmp[f"{stat}_new"]).to_numpy())
    fc_diffs = np.concatenate(fc_diffs)
    comp_rows.append({
        "item": "fold_composition",
        "sub_item": "all_6_stats_13_attrs_6_scopes",
        "old_value": np.nan,
        "new_value": np.nan,
        "abs_diff": float(fc_diffs.max()),
    })
    print(f"[sec34_16] fold composition: {len(exist_fc)} rows compared, max abs diff={fc_diffs.max():.3e}")

    # =====================================================================
    # 4. Penalty concentration (top 5% basins, top 5 models)
    # =====================================================================
    pub["delta_kge"] = pub["kge_seendpl_canonical"] - pub["kge_heldout"]
    basin_sums = pub.groupby("bs")["delta_kge"].sum().sort_values(ascending=False)
    model_sums = pub.groupby("model")["delta_kge"].sum().sort_values(ascending=False)
    n_top5pct = int(0.05 * pub["bs"].nunique())
    top5pct_basin_share = float(basin_sums.iloc[:n_top5pct].sum() / basin_sums.sum())
    top5_model_share = float(model_sums.iloc[:5].sum() / model_sums.sum())

    exist_conc = pd.read_csv(TABLES / "sec34_penalty_concentration.csv").set_index("group_label")
    e_basin = float(exist_conc.loc[f"top_5pct_basins (top {n_top5pct} basins)", "share_of_total_delta"])
    e_model = float(exist_conc.loc["top_5_models", "share_of_total_delta"])
    comp_rows.append({"item": "penalty_concentration", "sub_item": "top5pct_basin_share",
                      "old_value": e_basin, "new_value": top5pct_basin_share,
                      "abs_diff": abs(e_basin - top5pct_basin_share)})
    comp_rows.append({"item": "penalty_concentration", "sub_item": "top5_model_share",
                      "old_value": e_model, "new_value": top5_model_share,
                      "abs_diff": abs(e_model - top5_model_share)})
    print(f"[sec34_16] penalty concentration: top5% basins {top5pct_basin_share:.4f} (exist {e_basin:.4f}), "
          f"top5 models {top5_model_share:.4f} (exist {e_model:.4f})")

    # =====================================================================
    # 5. Coverage ladder (Seen Top-k -> OOB Top-1)
    # =====================================================================
    oob_win_idx = oob_order[:, 0]
    oob_oracle = oob.max(axis=1)
    exist_ladder = pd.read_csv(SEC32 / "sec32_coverage_ladder.csv").set_index("k")
    exist_regret = pd.read_csv(SEC32 / "sec32_regret_curve.csv").set_index("k")

    for k in [1, 3, 5, 10]:
        covers = np.empty(nb, dtype=bool)
        regrets = np.empty(nb, dtype=float)
        for b in range(nb):
            cand = set(seen_order[b, :k])
            covers[b] = oob_win_idx[b] in cand
            regrets[b] = oob_oracle[b] - oob[b, list(cand)].max()
        e_cov = float(exist_ladder.loc[k, "coverage_top1"])
        e_reg = float(exist_regret.loc[k, "regret_median"])
        comp_rows.append({"item": "coverage_ladder", "sub_item": f"coverage_top1_k{k}",
                          "old_value": e_cov, "new_value": float(covers.mean()),
                          "abs_diff": abs(e_cov - float(covers.mean()))})
        comp_rows.append({"item": "coverage_ladder", "sub_item": f"regret_median_k{k}",
                          "old_value": e_reg, "new_value": float(np.median(regrets)),
                          "abs_diff": abs(e_reg - float(np.median(regrets)))})
        print(f"[sec34_16] coverage ladder k={k}: coverage {covers.mean():.4f} (exist {e_cov:.4f}), "
              f"regret median {np.median(regrets):.4f} (exist {e_reg:.4f})")

    df_comp = pd.DataFrame(comp_rows)
    S.write_tidy_table(df_comp, TABLES / "sec34_a5_remaining_comparison.csv")

    # Summary of max abs diff per item
    summ = df_comp[df_comp.abs_diff.notna()].groupby("item")["abs_diff"].agg(
        n_checks="size", max_abs_diff="max").reset_index()
    S.write_tidy_table(summ, TABLES / "sec34_a5_remaining_summary.csv")
    print("\n[A5] summary:")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
