#!/usr/bin/env python3
"""sec34_14_bottleneck_singleton_idjoin.py — Extra item (after A4): Re-determine 100% coverage bottleneck basins and singleton counts by basin ID.

Recomputes by strict basin-ID joins from `t1_equivalence_sets.csv`:
  * 100% coverage bottleneck: exact minimum cover K* and maximum coverage at budget K*-1 (HiGHS MILP),
    listing the uncovered basins and their equivalence-set models for IC, seen-dPL, and OOB-dPL.
  * Singleton basin counts per model (|S_b| = 1) for the three conditions.
Compares with the existing values (bottleneck basins 14236200 / 11528700 / 09066300+09378170; singleton counts 36/47/63).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

# Shared module
ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def build_B_idjoin(eq_df: pd.DataFrame, condition: str, models: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build [nb, nm] boolean cover matrix with rows ordered by basin_id (string, zero-padded)."""
    sub = eq_df[eq_df.condition == condition].copy()
    sub["bs"] = sub["basin_id"].astype(int).astype(str).str.zfill(8)
    sub = sub.sort_values("bs").reset_index(drop=True)
    nb = len(sub)
    model_map = {m: i for i, m in enumerate(models)}
    B = np.zeros((nb, len(models)), dtype=bool)
    for i, em in enumerate(sub["equivalent_models"].astype(str)):
        for m in em.split("|"):
            if m in model_map:
                B[i, model_map[m]] = True
    return B, sub["bs"].tolist()


def solve_max_cov(B: np.ndarray, budget: int) -> Tuple[int, np.ndarray, int]:
    nb, nm = B.shape
    nv = nm + nb
    c = np.concatenate([np.zeros(nm), -np.ones(nb)])
    rows, lb, ub = [], [], []
    for i in range(nb):
        r = np.zeros(nv)
        r[nm + i] = 1.0
        r[:nm] = -B[i].astype(float)
        rows.append(r)
        lb.append(-np.inf)
        ub.append(0.0)
    b_row = np.zeros(nv)
    b_row[:nm] = 1.0
    rows.append(b_row)
    lb.append(budget)
    ub.append(budget)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return int(round(-res.fun)), (res.x[nm:] > 0.5), int(res.status)


def solve_min_cover(B: np.ndarray, needed: int) -> Tuple[int, int, int]:
    nm = B.shape[1]
    nb = B.shape[0]
    for q in range(1, nm + 1):
        cov, _, st = solve_max_cov(B, q)
        if cov >= needed:
            return q, cov, st
    return nm, nb, 0


def main() -> None:
    models = S.MODELS_36
    eq_df = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")

    SEC32 = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis" / "sec32_library_members" / "tables"
    exist_btn = pd.read_csv(SEC32 / "sec32_bottleneck_100pct_coverage.csv").set_index("condition")
    exist_sing = pd.read_csv(SEC32 / "sec32_singleton_model_counts.csv")

    btn_rows = []
    sing_rows = []

    for cond in ["IC", "seen-dPL", "OOB-dPL"]:
        B, basin_ids = build_B_idjoin(eq_df, cond, models)
        nb = len(B)

        # 100% cover minimum
        k_star, cov_star, st = solve_min_cover(B, nb)
        max_at_minus1, covered_mask, _ = solve_max_cov(B, k_star - 1)
        uncovered = [basin_ids[i] for i in np.flatnonzero(~covered_mask)]

        # equivalence models for uncovered basins (ID-joined)
        sub = eq_df[eq_df.condition == cond].copy()
        sub["bs"] = sub["basin_id"].astype(int).astype(str).str.zfill(8)
        sub = sub.set_index("bs")
        uncov_models = [str(sub.loc[b, "equivalent_models"]) for b in uncovered]
        forcing = sorted(set(m for em in uncov_models for m in em.split("|")))

        e = exist_btn.loc[cond]
        btn_rows.append({
            "condition": cond,
            "k_star_idjoin": k_star,
            "budget_minus_1": k_star - 1,
            "max_basins_covered_budget_minus_1": max_at_minus1,
            "uncovered_basin_ids_idjoin": "|".join(uncovered),
            "uncovered_basin_equivalent_models_idjoin": "; ".join(f"{b}: {m}" for b, m in zip(uncovered, uncov_models)),
            "bottleneck_forcing_models_idjoin": "|".join(forcing),
            "existing_uncovered_basin_ids": str(e["uncovered_basin_ids"]),
            "existing_bottleneck_forcing_models": str(e["bottleneck_forcing_models"]),
            "basin_ids_match": "|".join(uncovered) == str(e["uncovered_basin_ids"]),
            "forcing_models_match": "|".join(forcing) == str(e["bottleneck_forcing_models"]),
        })

        # Singletons
        sizes = B.sum(axis=1)
        singleton_rows = sub.loc[[b for b in basin_ids if sizes[basin_ids.index(b)] == 1]] if False else None
        # compute per-model singleton counts
        counts = {m: 0 for m in models}
        for i, b in enumerate(basin_ids):
            if sizes[i] == 1:
                counts[str(sub.loc[b, "equivalent_models"])] += 1
        for m in models:
            sing_rows.append({
                "condition": cond,
                "model": m,
                "singleton_basin_count_idjoin": counts[m],
            })

        print(f"[sec34_14] [{cond}] K*={k_star}, budget {k_star-1} covers {max_at_minus1}/{nb}; "
              f"uncovered={uncovered}; forcing={forcing} | match_ids={btn_rows[-1]['basin_ids_match']}, match_models={btn_rows[-1]['forcing_models_match']}")

    df_btn = pd.DataFrame(btn_rows)
    S.write_tidy_table(df_btn, TABLES / "sec34_bottleneck_idjoin_check.csv")

    df_sing = pd.DataFrame(sing_rows)
    # compare with existing
    exist_sing["singleton_basin_count"] = exist_sing["singleton_basin_count"].astype(int)
    merged = df_sing.merge(exist_sing[["condition", "model", "singleton_basin_count"]],
                           on=["condition", "model"], how="left")
    merged["abs_diff"] = (merged["singleton_basin_count_idjoin"] - merged["singleton_basin_count"]).abs()
    merged["match"] = merged["abs_diff"] == 0
    S.write_tidy_table(merged, TABLES / "sec34_singleton_idjoin_check.csv")

    # totals per condition
    tot = merged.groupby("condition").agg(
        total_singletons_idjoin=("singleton_basin_count_idjoin", "sum"),
        total_singletons_existing=("singleton_basin_count", "sum"),
    ).reset_index()
    tot["abs_diff"] = (tot["total_singletons_idjoin"] - tot["total_singletons_existing"]).abs()
    S.write_tidy_table(tot, TABLES / "sec34_singleton_totals_idjoin_check.csv")

    print("\nBottleneck comparison:")
    print(df_btn[["condition", "k_star_idjoin", "max_basins_covered_budget_minus_1",
                  "uncovered_basin_ids_idjoin", "bottleneck_forcing_models_idjoin",
                  "basin_ids_match", "forcing_models_match"]].to_string(index=False))
    print("\nSingleton totals:")
    print(tot.to_string(index=False))
    print(f"\nPer-model singleton mismatches: {int((~merged['match']).sum())}/{len(merged)}")


if __name__ == "__main__":
    main()
