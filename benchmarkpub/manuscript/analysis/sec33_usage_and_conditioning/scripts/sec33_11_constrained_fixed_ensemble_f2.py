#!/usr/bin/env python3
"""sec33_11_constrained_fixed_ensemble_f2.py — F2: parameter-constrained fixed global ensembles.

Builds fixed global equal-weight ensembles restricted to models with n_parameters <= threshold, at the
P25 / median / P75 thresholds of the 36-model library, selecting members by the training-fold median OOB KGE
(the same ranking statistic as the existing unconstrained B(k) protocol; target-fold information is never used).
Compares each constrained ensemble with the unconstrained B(k) of the same size, reports member lists,
parameter counts, degeneracy, parameter savings, and a per-fold leakage check.
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
SEC32 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec32_library_members" / "tables"

N_BOOT_BLOCK = S.N_BOOT_BLOCK
SEED_BOOT = S.SEED_BOOT
TRIAL = "--trial" in sys.argv
KS = [2, 3]


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
    return {"mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)), "wilcoxon_p": wilc_p,
            "ci95_lo_basin_boot": float(bs_lo), "ci95_hi_basin_boot": float(bs_hi),
            "ci95_lo_wy_block_boot": float(wy_lo), "ci95_hi_wy_block_boot": float(wy_hi)}


def main() -> None:
    basins = S.canonical_basin_ids()
    nb = len(basins)
    pos = S.basin_position_map(basins)
    models = S.MODELS_36
    fa = S.load_oob_folds()
    folds = np.array([fa[b] for b in basins])
    fold_ids = sorted(set(folds.tolist()))

    attr = pd.read_csv(SEC32 / "sec32_model_structural_attributes.csv").set_index("model")
    npar = {m: int(attr.loc[m, "n_parameters"]) for m in models}
    par_vec = np.array([npar[m] for m in models], dtype=int)
    p25, p50, p75 = (float(np.percentile(par_vec, 25)), float(np.median(par_vec)), float(np.percentile(par_vec, 75)))
    tiers = [("P25", p25), ("median", p50), ("P75", p75)]
    if TRIAL:
        tiers = [("median", p50)]

    _, _, oob_mat = S.load_oob_kge_matrix()   # [531, 36] SORTED basin order, model order = S.MODELS_36

    # member selection: identical ranking statistic to the existing B(k) protocol (training-fold median OOB KGE)
    members = {}
    for tname, thr in tiers:
        eligible = [i for i in range(len(models)) if par_vec[i] <= thr]
        members[tname] = {"threshold": thr, "eligible": eligible, "folds": {}}
        for f in fold_ids:
            tr = np.flatnonzero(folds != f)
            med = np.median(oob_mat[tr, :], axis=0)
            order = [i for i in np.argsort(-med, kind="stable") if i in set(eligible)]
            members[tname]["folds"][f] = {k: [int(i) for i in order[:k]] for k in KS}
    unconstrained = {}
    for f in fold_ids:
        tr = np.flatnonzero(folds != f)
        med = np.median(oob_mat[tr, :], axis=0)
        order = np.argsort(-med, kind="stable")
        unconstrained[f] = {k: [int(i) for i in order[:k]] for k in KS}

    # cross-check unconstrained members against the existing frozen fold-members table
    exist_mem = pd.read_csv(TABLES / "sec33_fixed_ensemble_fold_members.csv")
    mem_check = []
    for f in fold_ids:
        for k in KS:
            row = exist_mem[(exist_mem.fold == f) & (exist_mem.k == k)].iloc[0]
            frozen = [models.index(m) for m in row["members"].split("|")]
            mem_check.append({"fold": f, "k": k, "recomputed": "|".join(models[i] for i in unconstrained[f][k]),
                              "existing": row["members"], "identical": frozen == unconstrained[f][k]})
    df_mem_check = pd.DataFrame(mem_check)
    assert df_mem_check.identical.all(), df_mem_check[~df_mem_check.identical].to_string()

    wy_idx = np.concatenate([np.full(n, y) for n, y in zip(S.WATER_YEAR_DAYS, S.WATER_YEARS)])
    ens_u = {k: np.zeros((nb, 15, 6)) for k in KS}
    ens_c = {(tname, k): np.zeros((nb, 15, 6)) for tname, _ in tiers for k in KS}
    for f in fold_ids:
        ids = [str(x).zfill(8) for x in (S.DIR_OOB36_DATA / "runs" / models[0] / f"fold_{f}" / "heldout_basin_ids.txt").read_text().split()]
        obs = np.load(S.DIR_OOB36_DATA / "obs" / f"fold_{f}" / "heldout_y.npy")
        Q = np.stack([np.load(S.DIR_OOB36_DATA / "qsim" / m / f"fold_{f}.npy") for m in models])
        for j, bid in enumerate(ids):
            b = pos[S.pad_basin_id(bid)]
            for k in KS:
                idx_u = unconstrained[f][k]
                ens_u[k][b] = wy_stats(Q[idx_u, :, j].mean(axis=0), obs[:, j], wy_idx)
            for tname, _ in tiers:
                for k in KS:
                    idx_c = members[tname]["folds"][f][k]
                    ens_c[(tname, k)][b] = wy_stats(Q[idx_c, :, j].mean(axis=0), obs[:, j], wy_idx)
        del Q
    kge_u = {k: S.kge_from_sufficient(ens_u[k].sum(axis=1)) for k in KS}
    kge_c = {(tname, k): S.kge_from_sufficient(ens_c[(tname, k)].sum(axis=1)) for tname, _ in tiers for k in KS}

    # paired differences (constrained minus unconstrained), T2/T3 protocol
    counts = S.load_year_counts(n_reps=N_BOOT_BLOCK, seed=SEED_BOOT)
    rng_bs = np.random.default_rng(SEED_BOOT)
    sum_rows, pb_rows = [], []
    for tname, thr in tiers:
        for k in KS:
            diff = kge_c[(tname, k)] - kge_u[k]
            st = paired_stats(diff, counts, rng_bs)
            sum_rows.append({"tier": tname, "threshold_n_parameters": thr, "k": k,
                             "n_eligible_models": len(members[tname]["eligible"]),
                             "n_basins": nb, "constrained_kge_median": float(np.median(kge_c[(tname, k)])),
                             "unconstrained_kge_median": float(np.median(kge_u[k])),
                             "identical_ensemble_all_folds": bool(all(
                                 members[tname]["folds"][f][k] == unconstrained[f][k] for f in fold_ids)),
                             "wilcoxon_p_scientific": f"{st['wilcoxon_p']:.6e}",
                             **st})
            for i, b in enumerate(basins):
                pb_rows.append({"tier": tname, "k": k, "basin_id": int(b),
                                "constrained_kge": float(kge_c[(tname, k)][i]),
                                "unconstrained_kge": float(kge_u[k][i]),
                                "diff_constrained_minus_unconstrained": float(diff[i])})
    df_sum = pd.DataFrame(sum_rows)
    S.write_tidy_table(df_sum, TABLES / "sec33_constrained_ensemble_summary.csv")
    S.write_tidy_table(pd.DataFrame(pb_rows), TABLES / "sec33_constrained_ensemble_per_basin.csv")

    # members, parameter counts, degeneracy, savings
    m_rows, leak_rows = [], []
    for f in fold_ids:
        leak_rows.append({"fold": f, "n_training_basins": int((folds != f).sum()),
                          "n_target_basins": int((folds == f).sum()),
                          "target_fold_in_construction": "NO",
                          "selection_statistic": "training-fold median OOB KGE"})
        for tname, thr in tiers:
            for k in KS:
                c_idx = members[tname]["folds"][f][k]
                u_idx = unconstrained[f][k]
                m_rows.append({"fold": f, "tier": tname, "threshold_n_parameters": thr, "k": k,
                               "constrained_members": "|".join(models[i] for i in c_idx),
                               "constrained_member_n_parameters": "|".join(str(npar[models[i]]) for i in c_idx),
                               "constrained_params_sum": int(sum(npar[models[i]] for i in c_idx)),
                               "constrained_params_mean": float(np.mean([npar[models[i]] for i in c_idx])),
                               "unconstrained_members": "|".join(models[i] for i in u_idx),
                               "unconstrained_member_n_parameters": "|".join(str(npar[models[i]]) for i in u_idx),
                               "unconstrained_params_sum": int(sum(npar[models[i]] for i in u_idx)),
                               "unconstrained_params_mean": float(np.mean([npar[models[i]] for i in u_idx])),
                               "members_identical": bool(c_idx == u_idx),
                               "params_saved_sum": int(sum(npar[models[i]] for i in u_idx) - sum(npar[models[i]] for i in c_idx)),
                               "params_saved_mean": float(np.mean([npar[models[i]] for i in u_idx]) - np.mean([npar[models[i]] for i in c_idx]))})
    df_members = pd.DataFrame(m_rows)
    S.write_tidy_table(df_members, TABLES / "sec33_constrained_ensemble_members.csv")
    df_leak = pd.DataFrame(leak_rows)
    S.write_tidy_table(df_leak, TABLES / "sec33_constrained_ensemble_leakage_check.csv")
    S.write_tidy_table(df_mem_check, TABLES / "sec33_constrained_ensemble_unconstrained_member_check.csv")

    print("[F2] thresholds (n_parameters): P25=%.2f median=%.2f P75=%.2f" % (p25, p50, p75))
    print("[F2] eligible model counts:", {t: len(members[t]["eligible"]) for t, _ in tiers})
    print(df_sum.to_string(index=False))
    print()
    print("[F2] degeneracy per (tier,k):")
    print(df_members.groupby(["tier", "k"])["members_identical"].agg(["sum", "size"]).to_string())
    print()
    print("[F2] parameter savings (sum) per tier/k:")
    print(df_members.groupby(["tier", "k"])["params_saved_sum"].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
