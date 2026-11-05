#!/usr/bin/env python3
"""sec33_10_rule_c_k2_k5_per_basin_f1.py — F1: persist Rule C per-basin KGE series at k=2 and k=5 and re-verify at basin level.

Uses the E8 persisted full ranking predictions (no recommender refit). For every seed and every basin:
  * C(k) KGE from the daily equal-weight ensemble mean series of the top-k recommended models
    (never by averaging member KGEs);
  * A (k=1) and B(2) KGE for the paired differences, using the same daily-ensemble protocol.
Writes a per-basin table in the exact column layout of `sec33_usage_rules_per_basin.csv`, recomputes the
paired differences vs A and vs B(k=2) with the T2/T3 protocol, compares them with the existing ladder values,
and re-checks the new table row-by-row by basin-ID join against raw inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

N_BOOT_BLOCK = S.N_BOOT_BLOCK
SEED_BOOT = S.SEED_BOOT
TRIAL = "--trial" in sys.argv


def wy_stats(series: np.ndarray, obs: np.ndarray, wy_idx: np.ndarray) -> np.ndarray:
    out = np.zeros((15, 6))
    for yi, y in enumerate(S.WATER_YEARS):
        sl = (wy_idx == y)
        s, o = series[sl], obs[sl]
        m = np.isfinite(s) & np.isfinite(o)
        ss = np.where(m, s, 0.0)
        oo = np.where(m, o, 0.0)
        out[yi] = [m.sum(), ss.sum(), oo.sum(), (ss * ss).sum(), (oo * oo).sum(), (ss * oo).sum()]
    return out


def paired_stats(diff: np.ndarray, counts: np.ndarray, rng_bs: np.random.Generator) -> Dict[str, float]:
    """T2/T3 paired protocol: Wilcoxon on non-zero diffs; basin bootstrap; water-year block proxy."""
    nb = len(diff)
    d_nz = diff[diff != 0]
    wilc_p = float(stats.wilcoxon(d_nz).pvalue) if len(d_nz) > 0 else 1.0
    draws_bs = np.empty(N_BOOT_BLOCK, dtype=np.float64)
    for r in range(N_BOOT_BLOCK):
        draws_bs[r] = diff[rng_bs.integers(0, nb, size=nb)].mean()
    bs_lo, bs_hi = np.percentile(draws_bs, [2.5, 97.5])
    draws_wy = np.empty(N_BOOT_BLOCK, dtype=np.float64)
    for r in range(N_BOOT_BLOCK):
        w = counts[r] / counts[r].sum()
        draws_wy[r] = np.mean([diff[b] * (1.0 + 0.5 * (w[b % 15] - 1.0 / 15.0)) for b in range(nb)])
    wy_lo, wy_hi = np.percentile(draws_wy, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "wilcoxon_p": wilc_p, "ci95_lo_basin_boot": float(bs_lo), "ci95_hi_basin_boot": float(bs_hi),
            "ci95_lo_wy_block_boot": float(wy_lo), "ci95_hi_wy_block_boot": float(wy_hi)}


def main() -> None:
    basins = S.canonical_basin_ids()          # SORTED (A6 accessor)
    nb = len(basins)
    pos = S.basin_position_map(basins)        # A6 accessor
    models = S.MODELS_36
    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])
    seeds = S.SEEDS_10 if not TRIAL else S.SEEDS_10[:1]

    # E8 persisted rankings
    z = np.load(TABLES / "sec33_recommender_full_rankings.npz")
    check_seeds = z["seeds"].tolist()
    assert check_seeds == S.SEEDS_10, check_seeds
    rank_seen = z["rank_seen_a2d"]            # [10, 531, 36]

    # Unconstrained B members per fold (existing protocol product)
    mem = pd.read_csv(TABLES / "sec33_fixed_ensemble_fold_members.csv")
    fixed_B = {}
    for f in sorted(set(folds)):
        fixed_B[f] = {}
        for k in [1, 2]:
            row = mem[(mem.fold == f) & (mem.k == k)].iloc[0]
            fixed_B[f][k] = [models.index(m) for m in row["members"].split("|")]

    wy_idx = np.concatenate([np.full(n, y) for n, y in zip(S.WATER_YEAR_DAYS, S.WATER_YEARS)])
    KS = [2, 3, 5]                            # 3 kept so the shared-RNG order matches sec33_09 exactly

    def compute_kge(rank_src, ks, b_members):
        """Daily equal-weight ensemble mean series KGE per basin, recomputed from raw Qsim/obs + rankings."""
        ens_c = {(s_idx, k): np.zeros((nb, 15, 6)) for s_idx in range(len(S.SEEDS_10)) for k in ks}
        ens_b = {k: np.zeros((nb, 15, 6)) for k in b_members}
        for f in sorted(set(folds)):
            ids = [str(x).zfill(8) for x in (S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{f}" / "heldout_basin_ids.txt").read_text().split()]
            obs = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
            Q = np.stack([np.load(S.DIR_OOB36_DATA / "qsim" / m / f"fold_{f}.npy") for m in models])
            for j, bid in enumerate(ids):
                b = pos[S.pad_basin_id(bid)]  # A6 accessor ID lookup
                for k in b_members:
                    idx_b = b_members[k][f]
                    series_b = Q[idx_b, :, j].mean(axis=0) if len(idx_b) > 1 else Q[idx_b[0], :, j]
                    ens_b[k][b] = wy_stats(series_b, obs[:, j], wy_idx)
                for s_idx in range(len(S.SEEDS_10)):
                    for k in ks:
                        idx_c = rank_src[s_idx, b, :k]
                        ens_c[(s_idx, k)][b] = wy_stats(Q[idx_c, :, j].mean(axis=0), obs[:, j], wy_idx)
            del Q
        kb = {k: S.kge_from_sufficient(ens_b[k].sum(axis=1)) for k in ens_b}
        kc = {(s_idx, k): S.kge_from_sufficient(ens_c[(s_idx, k)].sum(axis=1))
              for s_idx in range(len(S.SEEDS_10)) for k in ks}
        return kc, kb

    b_members = {k: {f: fixed_B[f][k] for f in sorted(set(folds))} for k in [1, 2]}
    kge_C, kge_B = compute_kge(rank_seen, KS, b_members)
    kge_A = kge_B[1]
    kge_B2 = kge_B[2]

    # ---- per-basin table (F1 deliverable), format identical to sec33_usage_rules_per_basin.csv ----
    rows = []
    for s_idx, s in enumerate(S.SEEDS_10):
        if s not in seeds:
            continue
        for k in [2, 5]:
            cur = kge_C[(s_idx, k)]
            for i, b in enumerate(basins):
                rows.append({"rule": "C_A2d", "k": k, "seed": s, "basin_id": int(b),
                             "fold": int(folds[i]), "kge": float(cur[i]),
                             "diff_vs_A": float(cur[i] - kge_A[i])})
    df_pb = pd.DataFrame(rows)
    out_name = "sec33_rule_c_k2_k5_per_basin.csv"
    S.write_tidy_table(df_pb, TABLES / out_name)
    print(f"[F1] wrote {out_name}: {len(df_pb)} rows, seeds={sorted(df_pb.seed.unique())}", flush=True)

    if TRIAL:
        print("[F1] trial mode: single seed; stopping before the full re-verification", flush=True)
        return

    # ---- paired-difference recompute (shared RNG order identical to sec33_09) ----
    counts = S.load_year_counts(n_reps=N_BOOT_BLOCK, seed=SEED_BOOT)
    rng_bs = np.random.default_rng(SEED_BOOT)
    new_rows = []
    for k in KS:
        for s_idx, s in enumerate(S.SEEDS_10):
            for bench, ref in (("A", kge_A), ("B2", kge_B2)):
                st = paired_stats(kge_C[(s_idx, k)] - ref, counts, rng_bs)
                new_rows.append({"rule": f"Rule_C_k{k}", "k": k, "seed": s, "benchmark": bench,
                                 "n_basins": nb, **st})
    df_new = pd.DataFrame(new_rows)

    # ---- compare with existing values ----
    exist_seed = pd.read_csv(TABLES / "sec33_rule_c_per_seed_ladder.csv")
    exist_ladder = pd.read_csv(TABLES / "sec33_rules_ladder_unified.csv")
    cmp_rows = []
    for k in [2, 5]:
        for s_idx, s in enumerate(S.SEEDS_10):
            e = exist_seed[(exist_seed.k == k) & (exist_seed.seed == s)].iloc[0]
            n_a = df_new[(df_new.k == k) & (df_new.seed == s) & (df_new.benchmark == "A")].iloc[0]
            n_b = df_new[(df_new.k == k) & (df_new.seed == s) & (df_new.benchmark == "B2")].iloc[0]
            for label, old, new in [
                ("vs_A_mean_diff", e["vs_A_mean_diff"], n_a["mean_diff"]),
                ("vs_A_median_diff", e["vs_A_median_diff"], n_a["median_diff"]),
                ("vs_A_wilcoxon_p", e["vs_A_wilcoxon_p"], n_a["wilcoxon_p"]),
                ("vs_A_ci95_lo_basin", e["vs_A_ci95_lo_basin"], n_a["ci95_lo_basin_boot"]),
                ("vs_A_ci95_hi_basin", e["vs_A_ci95_hi_basin"], n_a["ci95_hi_basin_boot"]),
                ("vs_A_ci95_lo_wy", e["vs_A_ci95_lo_wy"], n_a["ci95_lo_wy_block_boot"]),
                ("vs_A_ci95_hi_wy", e["vs_A_ci95_hi_wy"], n_a["ci95_hi_wy_block_boot"]),
                ("vs_B2_mean_diff", e["vs_B2_mean_diff"], n_b["mean_diff"]),
                ("vs_B2_median_diff", e["vs_B2_median_diff"], n_b["median_diff"]),
                ("vs_B2_wilcoxon_p", e["vs_B2_wilcoxon_p"], n_b["wilcoxon_p"]),
                ("vs_B2_ci95_lo_basin", e["vs_B2_ci95_lo_basin"], n_b["ci95_lo_basin_boot"]),
                ("vs_B2_ci95_hi_basin", e["vs_B2_ci95_hi_basin"], n_b["ci95_hi_basin_boot"]),
                ("vs_B2_ci95_lo_wy", e["vs_B2_ci95_lo_wy"], n_b["ci95_lo_wy_block_boot"]),
                ("vs_B2_ci95_hi_wy", e["vs_B2_ci95_hi_wy"], n_b["ci95_hi_wy_block_boot"]),
            ]:
                cmp_rows.append({"k": k, "seed": s, "quantity": label, "old_value": float(old),
                                 "new_value": float(new), "abs_diff": abs(float(old) - float(new))})
        # 10-seed mean rows of the unified ladder
        e_lad = exist_ladder[(exist_ladder.rule_family == "C") & (exist_ladder.k == k)].iloc[0]
        for label, col_old in [("vs_A_mean_diff", "vs_A_mean_diff"), ("vs_A_median_diff", "vs_A_median_diff"),
                               ("vs_A_wilcoxon_p", "vs_A_wilcoxon_p"), ("vs_B2_mean_diff", "vs_B2_mean_diff"),
                               ("vs_B2_median_diff", "vs_B2_median_diff"), ("vs_B2_wilcoxon_p", "vs_B2_wilcoxon_p")]:
            new_val = float(df_new[(df_new.k == k) & (df_new.benchmark == ("A" if label.startswith("vs_A") else "B2"))][
                {"vs_A_mean_diff": "mean_diff", "vs_A_median_diff": "median_diff", "vs_A_wilcoxon_p": "wilcoxon_p",
                 "vs_B2_mean_diff": "mean_diff", "vs_B2_median_diff": "median_diff", "vs_B2_wilcoxon_p": "wilcoxon_p"}[label]].mean())
            cmp_rows.append({"k": k, "seed": "10seed_mean", "quantity": label, "old_value": float(e_lad[col_old]),
                             "new_value": new_val, "abs_diff": abs(float(e_lad[col_old]) - new_val)})
    df_cmp = pd.DataFrame(cmp_rows)
    S.write_tidy_table(df_cmp, TABLES / "sec33_rule_c_k2_k5_recompute_comparison.csv")

    # full-precision agreement summary (the comparison table itself is written at the tree's 6-dp convention)
    cmp_sum = pd.DataFrame([{
        "quantity": q,
        "n_rows": int(len(g)),
        "max_abs_diff": float(g.abs_diff.max()),
        "max_abs_diff_sci": f"{float(g.abs_diff.max()):.3e}",
        "n_rows_agree_within_1e6": int((g.abs_diff < 1e-6).sum()),
    } for q, g in df_cmp.groupby("quantity")])
    S.write_tidy_table(cmp_sum, TABLES / "sec33_rule_c_k2_k5_comparison_summary.csv")

    # ---- row-by-row alignment check of the new per-basin table (A2-style) ----
    # Independent pass: re-read the written CSV from disk and recompute kge / diff_vs_A from raw inputs.
    pb_disk = pd.read_csv(TABLES / out_name)
    kge_C2, kge_B2_only = compute_kge(rank_seen, [2, 5], {k: {f: fixed_B[f][k] for f in sorted(set(folds))} for k in [1]})
    kge_A2 = kge_B2_only[1]
    chk_rows = []
    for k in [2, 5]:
        sub = pb_disk[pb_disk.k == k]
        for col, fn in [("kge", lambda s_idx, b: kge_C2[(s_idx, k)][b]),
                        ("diff_vs_A", lambda s_idx, b: kge_C2[(s_idx, k)][b] - kge_A2[b])]:
            diffs = []
            for _, r in sub.iterrows():
                s_idx = S.SEEDS_10.index(int(r["seed"]))
                b = pos[S.pad_basin_id(r["basin_id"])]      # A6 accessor ID lookup
                diffs.append(abs(fn(s_idx, b) - float(r[col])))
            diffs = np.array(diffs)
            chk_rows.append({"table": out_name, "slice": f"k={k}", "tested_column": col,
                             "n_rows_checked": len(sub), "n_mismatches": int((diffs > 1e-6).sum()),
                             "max_abs_diff": float(diffs.max()),
                             "max_abs_diff_sci": f"{float(diffs.max()):.3e}",
                             "status": "PASS" if int((diffs > 1e-6).sum()) == 0 else "FAIL_MISMATCH"})
    df_chk = pd.DataFrame(chk_rows)
    S.write_tidy_table(df_chk, TABLES / "sec33_rule_c_k2_k5_alignment_check.csv")

    print("[F1] per-basin rows:", len(df_pb))
    print(df_chk.to_string(index=False))
    print("[F1] comparison max abs diffs by quantity:")
    print(df_cmp.groupby("quantity")["abs_diff"].max().to_string())


if __name__ == "__main__":
    main()
