#!/usr/bin/env python3
"""sec36_01_size_matched_null_h1.py — H1: size-matched null distribution for the per-type minimal cover size.

Question form: is a type's minimal cover size lower than the cover size obtained from a random draw of the SAME
number of basins from the same 515-basin matched pool?
Sampling pool: the 515 type-matched basins used by G1-G4 (same source as the types).
Main tier: 95% cover, three conditions (IC / seen-dPL / OOB-dPL) x 10 types, R=1000 draws without replacement per cell,
each draw solved exactly with the same MILP implementation (scipy.optimize.milp / HiGHS).
Secondary tier: 90% and 100% cover, same structure, R=200.
No equivalence rule is recomputed: the frozen `t1_equivalence_sets.csv` is used throughout.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
sys.path.insert(0, str(ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"))
import _shared as S

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["IC", "seen-dPL", "OOB-dPL"]
MAIN_TARGET = 0.95
MAIN_R = 1000
SECONDARY_TARGETS = [0.90, 1.00]
SECONDARY_R = 200
SEED = 20261006
TRIAL = "--trial" in sys.argv

DIRECTION = ("one-sided empirical percentile p_null_le_observed = (#null draws with cover <= observed + 1) / (R + 1); "
             "small values mean the type's observed cover is lower than the size-matched random draws")


SOLVE_COUNT = {"total": 0, "null_draws": 0}


def solve_min_cover_direct(Bsub: np.ndarray, needed: int, is_null_draw: bool = False):
    """Exact minimal number of models covering at least `needed` basins of the sampled subset (single MILP)."""
    SOLVE_COUNT["total"] += 1
    if is_null_draw:
        SOLVE_COUNT["null_draws"] += 1
    nb, nm = Bsub.shape
    nv = nm + nb
    c = np.concatenate([np.ones(nm), np.zeros(nb)])
    A1 = np.hstack([-Bsub.astype(np.float64), np.eye(nb)])
    A2 = np.concatenate([np.zeros(nm), np.ones(nb)])[None, :]
    A = np.vstack([A1, A2])
    lb = np.concatenate([np.full(nb, -np.inf), [float(needed)]])
    ub = np.concatenate([np.zeros(nb), [np.inf]])
    res = milp(c=c, constraints=LinearConstraint(A, lb=lb, ub=ub),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    if res.status != 0:
        return None, int(res.status)
    return int(round(res.fun)), int(res.status)


def main() -> None:
    t_start = time.time()
    basins = S.canonical_basin_ids()
    cf = pd.read_csv(ROOT / "data" / "gauge_cluster.csv")
    cf["bs"] = cf["gauge_index"].map(S.pad_basin_id)
    cmap = dict(zip(cf["bs"], cf["gauge_cluster"]))
    matched = [b for b in basins if b in cmap]
    types_sorted = sorted(set(cmap[b] for b in matched))
    type_n = {t: sum(1 for b in matched if cmap[b] == t) for t in types_sorted}
    pool = len(matched)
    models = S.MODELS_36
    midx = {m: i for i, m in enumerate(models)}

    eq = pd.read_csv(S.DIR_EQUIV_T1T2 / "tables" / "t1_equivalence_sets.csv")
    eq["bs"] = eq["basin_id"].map(S.pad_basin_id)
    eq = eq.set_index(["bs", "condition"])

    # pool cover matrices (frozen equivalence sets, basin-ID joins only)
    B = {}
    for c in CONDITIONS:
        M = np.zeros((pool, len(models)), dtype=bool)
        for i, b in enumerate(matched):
            for m in str(eq.loc[(b, c), "equivalent_models"]).split("|"):
                M[i, midx[m]] = True
        B[c] = M

    # observed per-type cover from the type's own basins (frozen sets, exact solve)
    observed = {}
    for c in CONDITIONS:
        for t in types_sorted:
            rows = [i for i, b in enumerate(matched) if cmap[b] == t]
            n_t = len(rows)
            k, st = solve_min_cover_direct(B[c][rows], int(np.ceil(MAIN_TARGET * n_t)))
            observed[(c, t, MAIN_TARGET)] = k

    if TRIAL:
        t_largest = max(types_sorted, key=lambda t: type_n[t])
        rng = np.random.default_rng(SEED)
        t0 = time.time()
        draws = []
        for _ in range(50):
            sub = rng.choice(pool, size=type_n[t_largest], replace=False)
            k, st = solve_min_cover_direct(B["IC"][sub], int(np.ceil(MAIN_TARGET * type_n[t_largest])))
            draws.append(k)
        dt = time.time() - t0
        print(f"[H1 TRIAL] condition=IC type={t_largest} N={type_n[t_largest]} R=50: "
              f"observed={observed[('IC', t_largest, MAIN_TARGET)]}, null median={np.median(draws)}, "
              f"min={min(draws)}, max={max(draws)}, {dt:.2f}s ({dt/50*1000:.1f} ms/solve)")
        print(f"[H1 TRIAL] projected full run: main {3*1000*50 and 3*len(types_sorted)*1000} solves + "
              f"secondary {2*3*len(types_sorted)*200} solves at the measured per-solve cost")
        return

    summary_rows, dist_rows = [], []
    nonoptimal = 0

    def run_tier(target: float, R: int, tier_name: str, rng: np.random.Generator) -> None:
        nonlocal nonoptimal
        for c in CONDITIONS:
            for t in types_sorted:
                n_t = type_n[t]
                need = int(np.ceil(target * n_t))
                obs = observed.get((c, t, target))
                if obs is None:
                    rows = [i for i, b in enumerate(matched) if cmap[b] == t]
                    obs, st = solve_min_cover_direct(B[c][rows], need)
                vals = []
                for _ in range(R):
                    sub = rng.choice(pool, size=n_t, replace=False)
                    k, st = solve_min_cover_direct(B[c][sub], need, is_null_draw=True)
                    if k is None:
                        nonoptimal += 1
                        continue
                    vals.append(k)
                    dist_rows.append({"tier": tier_name, "target": target, "condition": c, "type": int(t),
                                      "n_type": n_t, "draw_index": len(vals) - 1, "null_cover": int(k)})
                v = np.array(vals, dtype=float)
                p_le = float((np.sum(v <= obs) + 1) / (len(v) + 1)) if len(v) else np.nan
                summary_rows.append({
                    "tier": tier_name, "target": target, "target_pct": f"{int(target*100)}%",
                    "condition": c, "type": int(t), "n_type": n_t, "need_basins": need,
                    "observed_cover": int(obs),
                    "null_median": float(np.median(v)), "null_q25": float(np.quantile(v, 0.25)),
                    "null_q75": float(np.quantile(v, 0.75)), "null_p5": float(np.quantile(v, 0.05)),
                    "null_p95": float(np.quantile(v, 0.95)),
                    "null_min": float(v.min()), "null_max": float(v.max()), "null_mean": float(v.mean()),
                    "n_draws_used": int(len(v)), "n_nonoptimal_solves": int(R - len(v)),
                    "one_sided_empirical_percentile": p_le,
                    "direction_definition": DIRECTION,
                    "R": R, "seed": SEED,
                    "sampling_pool": f"{pool} type-matched basins (G1-G4 matched pool)",
                })
                print(f"[H1 {tier_name}] {c} type{t}: obs={obs} null_med={np.median(v):.1f} "
                      f"p5={np.quantile(v,0.05):.1f} p95={np.quantile(v,0.95):.1f} p_le={p_le:.3f} "
                      f"({time.time()-t_start:.0f}s)", flush=True)

    rng_main = np.random.default_rng(SEED)
    run_tier(MAIN_TARGET, MAIN_R, "main_95", rng_main)
    rng_sec = np.random.default_rng(SEED + 1)
    for target in SECONDARY_TARGETS:
        run_tier(target, SECONDARY_R, f"secondary_{int(target*100)}", rng_sec)

    S.write_tidy_table(pd.DataFrame(summary_rows), TABLES / "sec36_h1_size_matched_null_summary.csv")
    S.write_tidy_table(pd.DataFrame(dist_rows), TABLES / "sec36_h1_size_matched_null_distribution.csv")
    S.write_tidy_table(pd.DataFrame([{
        "sampling_pool": f"{pool} type-matched basins", "matched": pool, "unmatched": len(basins) - pool,
        "main_target": MAIN_TARGET, "main_R": MAIN_R, "secondary_targets": "|".join(f"{t:.2f}" for t in SECONDARY_TARGETS),
        "secondary_R": SECONDARY_R, "seed": SEED, "seed_secondary": SEED + 1,
        "solver": "scipy.optimize.milp (HiGHS), direct minimal-cover formulation, single MILP per draw",
        "nonoptimal_solves": nonoptimal,
        "total_null_draws": int(SOLVE_COUNT["null_draws"]),
        "total_observed_solves": int(SOLVE_COUNT["total"] - SOLVE_COUNT["null_draws"]),
        "total_milp_solves": int(SOLVE_COUNT["total"]),
        "direction_definition": DIRECTION,
        "timing_note": "wall-clock elapsed time is printed to stdout only; it is not persisted, so every output file is deterministic",
    }]), TABLES / "sec36_h1_run_meta.csv")
    print(f"[H1] PASS: {len(summary_rows)} cells, non-optimal solves = {nonoptimal}, "
          f"elapsed {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
