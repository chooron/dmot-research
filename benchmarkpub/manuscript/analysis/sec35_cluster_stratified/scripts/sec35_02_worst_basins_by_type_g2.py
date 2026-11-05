#!/usr/bin/env python3
"""sec35_02_worst_basins_by_type_g2.py — G2: type distribution of the 26 worst-penalty basins.

Definition (unchanged): per basin, sum of `delta_kge` (= kge_seendpl_canonical - kge_heldout) over the 36 models,
keeping negative values; take the top int(0.05 x 531) = 26 basins by descending sum.
Reports how many of the 26 are type-matched, the type x (in/out of top 26) contingency table with observed and
expected counts over the matched basins, and an exact test on the 2 x 10 table.
"""
from __future__ import annotations

import sys
from math import lgamma
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

N_TOP = int(0.05 * 531)          # 26
N_DRAWS = 100000
SEED = 20261005


def log_multivariate_hypergeom(x: np.ndarray, sizes: np.ndarray, R: int, N: int) -> float:
    """log P(X = x) for the multivariate hypergeometric with type sizes `sizes`, R draws from N."""
    val = lgamma(N + 1) - lgamma(R + 1) - lgamma(N - R + 1)
    for xi, ni in zip(x, sizes):
        val += lgamma(ni + 1) - lgamma(xi + 1) - lgamma(ni - xi + 1)
    return float(val)


def main() -> None:
    basins = S.canonical_basin_ids()
    pos = S.basin_position_map(basins)

    cf = pd.read_csv(ROOT / "data" / "gauge_cluster.csv")
    cf["bs"] = cf["gauge_index"].map(S.pad_basin_id)
    cmap = dict(zip(cf["bs"], cf["gauge_cluster"]))

    pub = pd.read_csv(S.DIR_OOB36_PRELIM / "pub_next_stage_20260930" / "tables" / "pub_same_pipeline_model_basin.csv")
    pub["bs"] = pub["basin_id"].map(S.pad_basin_id)
    pub["delta_kge"] = pub["kge_seendpl_canonical"] - pub["kge_heldout"]
    basin_sum = pub.groupby("bs")["delta_kge"].sum().sort_values(ascending=False)
    top_ids = basin_sum.index[:N_TOP].tolist()
    matched_ids = [b for b in basins if b in cmap]

    n_top_matched = int(sum(1 for b in top_ids if b in cmap))
    n_top_unmatched = N_TOP - n_top_matched

    types_sorted = sorted(set(cmap[b] for b in matched_ids))
    sizes = np.array([sum(1 for b in matched_ids if cmap[b] == t) for t in types_sorted])
    in_top = np.array([sum(1 for b in top_ids if b in cmap and cmap[b] == t) for t in types_sorted])
    N = int(sizes.sum())
    R = int(in_top.sum())

    rows = []
    for i, t in enumerate(types_sorted):
        expected = sizes[i] * R / N
        rows.append({"type": int(t), "n_matched": int(sizes[i]),
                     "n_in_top26": int(in_top[i]), "n_not_in_top26": int(sizes[i] - in_top[i]),
                     "expected_in_top26": float(expected),
                     "obs_minus_expected": float(in_top[i] - expected)})
    df_tab = pd.DataFrame(rows)
    S.write_tidy_table(df_tab, TABLES / "sec35_g2_worst26_contingency.csv")

    # exact test on the 2 x 10 table: Monte Carlo enumeration of the multivariate hypergeometric null
    # using Fisher's probability-ordering statistic
    type_index = np.array([types_sorted.index(cmap[b]) for b in matched_ids])
    obs_logp = log_multivariate_hypergeom(in_top, sizes, R, N)
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(N_DRAWS):
        idx = rng.choice(N, size=R, replace=False)
        x = np.bincount(type_index[idx], minlength=len(types_sorted))
        if log_multivariate_hypergeom(x, sizes, R, N) <= obs_logp + 1e-12:
            count += 1
    p_mc = (count + 1) / (N_DRAWS + 1)
    se_mc = float(np.sqrt(p_mc * (1.0 - p_mc) / (N_DRAWS + 1)))

    chi2, p_chi2, dof, expected_full = stats.chi2_contingency(np.vstack([in_top, sizes - in_top]))
    S.write_tidy_table(pd.DataFrame([{
        "n_top_defined": N_TOP, "n_top_matched": n_top_matched, "n_top_unmatched": n_top_unmatched,
        "n_matched_total": N, "n_types": len(types_sorted),
        "exact_test_implementation": "Fisher-type exact test by Monte Carlo enumeration of the multivariate hypergeometric null "
                                     "(rng.choice without replacement; probability-ordering statistic; no conditioning beyond the fixed margins)",
        "n_draws": N_DRAWS, "seed": SEED,
        "p_value_exact_monte_carlo": float(p_mc), "monte_carlo_standard_error": se_mc,
        "chi2_statistic_asymptotic": float(chi2), "chi2_dof_asymptotic": int(dof), "p_value_chi2_asymptotic": float(p_chi2),
        "observed_top26_counts": "|".join(f"{t}:{int(c)}" for t, c in zip(types_sorted, in_top)),
    }]), TABLES / "sec35_g2_fisher_exact.csv")

    print(f"[G2] top-{N_TOP}: matched={n_top_matched} unmatched={n_top_unmatched}")
    print(df_tab.to_string(index=False))
    print(f"[G2] exact MC p={p_mc:.6f} (se={se_mc:.2e}, draws={N_DRAWS}, seed={SEED}); chi2 p={p_chi2:.6f}")


if __name__ == "__main__":
    main()
