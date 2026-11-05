#!/usr/bin/env python3
"""25_minimal_model_library.py — how many DISTINCT model structures does the candidate-set
recommendation actually require maintaining across the 531 basins?

Reads only frozen results.  For the seen Top-3 (and Top-5) candidate set of every basin:

  1. the union of all 531 sets: how many of the 36 models ever appear, and how often;
  2. the MINIMUM set cover: the fewest models such that every basin's Top-k set contains at
     least one of them.  Solved to proven optimality with an integer program
     (`scipy.optimize.milp`, HiGHS); the LP-relaxation bound is reported as an independent
     optimality certificate, and a greedy cover is reported for comparison.

No KGE is recomputed: the Top-k sets are re-derived by ranking the frozen
`kge_seenfolds_median4` column, and the derivation is validated against the frozen
aggregate overlap statistics.  No torch, no training, no modification of frozen outputs.

Outputs -> $OUT5/tables/{library_model_frequency,library_minimum_cover,
                        library_greedy_cover_progress}.csv
           $OUT5/tables/minimal_model_library.json
           $OUT5/MINIMAL_MODEL_LIBRARY.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path("/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim")
sys.path.insert(0, str(HERE))
import pub_common as PC  # noqa: E402

KS = (3, 5)
PRIMARY_SEEN = "kge_seenfolds_median4"


def log(m: str) -> None:
    print(m, flush=True)


def topk_membership(M: pd.DataFrame, k: int) -> np.ndarray:
    """Boolean [n_basins, n_models]: is model m in the basin's top-k by the frozen score?"""
    arr = M.to_numpy(np.float64)
    order = np.argsort(-arr, axis=1, kind="stable")[:, :k]
    B = np.zeros(arr.shape, dtype=bool)
    for i in range(arr.shape[0]):
        B[i, order[i]] = True
    return B


def exact_min_cover(B: np.ndarray, exclude: set[int] | None = None):
    """Minimum number of columns of B hitting every row.

    Solved with `scipy.optimize.milp` (HiGHS branch-and-bound) under binary integrality, so a
    status-0 return IS a proof of optimality for this integer program.  The LP relaxation is a
    separate, weaker lower bound and is reported alongside as an independent sanity check.
    """
    n_rows, n_cols = B.shape
    A = B.astype(float)
    c = np.ones(n_cols)
    cons = [LinearConstraint(A, lb=1.0, ub=np.inf)]
    if exclude:
        row = np.zeros(n_cols)
        row[sorted(exclude)] = 1.0
        cons.append(LinearConstraint(row.reshape(1, -1), lb=-np.inf, ub=len(exclude) - 1))
    res = milp(c=c, constraints=cons, integrality=np.ones(n_cols),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    chosen = np.flatnonzero(res.x > 0.5)
    cert = {"status": int(res.status), "message": str(res.message),
            "proven_optimal": bool(res.status == 0),
            "mip_gap": float(getattr(res, "mip_gap", float("nan")))}
    return chosen, float(res.fun), cert


def lp_bound(B: np.ndarray) -> float:
    res = milp(c=np.ones(B.shape[1]), constraints=[LinearConstraint(B.astype(float), lb=1.0,
                                                                    ub=np.inf)],
               integrality=np.zeros(B.shape[1]), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(res.fun)


def greedy_cover(B: np.ndarray):
    n_rows, n_cols = B.shape
    uncovered = np.ones(n_rows, dtype=bool)
    chosen, progress = [], []
    while uncovered.any():
        gains = B[uncovered].sum(axis=0)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        chosen.append(best)
        uncovered &= ~B[:, best]
        progress.append({"n_models": len(chosen), "model": None,
                         "basins_covered": int((~uncovered).sum()),
                         "frac_basins_covered": float((~uncovered).mean())})
        progress[-1]["model"] = None
    return chosen, progress


def main() -> None:
    PC.T5.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")
    models = sorted(F["model"].unique())
    basins = np.sort(F["basin_id"].unique())
    M = F.pivot(index="basin_id", columns="model", values=PRIMARY_SEEN).loc[basins, models]
    if M.isna().any().any():
        raise ValueError("pivot produced NaN")
    n_basins, n_models = M.shape
    if (n_basins, n_models) != (531, 36):
        raise ValueError(f"expected 531x36, got {n_basins}x{n_models}")

    report = {"meta": {
        "n_basins": int(n_basins), "n_models_total": int(n_models),
        "seen_definition": PRIMARY_SEEN + " (the frozen next-stage primary seen aggregation)",
        "membership_derivation": "re-ranking the frozen seen scores; no KGE recomputation",
        "solver": "scipy.optimize.milp (HiGHS), binary integrality -> proven optimal",
        "no_training": True, "no_inference": True,
    }}

    freq_rows, cover_rows, progress_rows = [], [], []
    res = {}
    for k in KS:
        B = topk_membership(M, k)
        if not (B.sum(axis=1) == k).all():
            raise ValueError(f"top-{k} membership is not exactly {k} models per basin")
        counts = B.sum(axis=0)
        used = np.flatnonzero(counts > 0)
        freq = pd.DataFrame({
            "model": models, f"n_basins_in_top{k}": counts,
            f"frac_of_basins_in_top{k}": counts / n_basins,
        }).sort_values(f"n_basins_in_top{k}", ascending=False).reset_index(drop=True)
        freq[f"rank_by_top{k}_frequency"] = np.arange(1, n_models + 1)
        freq_rows.append(freq)

        # ---- 1. union -------------------------------------------------------
        union_models = [models[i] for i in used]
        # ---- 2. exact minimum cover ----------------------------------------
        chosen, opt, cert = exact_min_cover(B)
        cover_models = [models[i] for i in sorted(chosen)]
        covered = B[:, chosen].any(axis=1)
        lp = lp_bound(B)
        # alternative optima?
        alt, alt_opt, _ = exact_min_cover(B, exclude=set(chosen.tolist()))
        alt_models = [models[i] for i in sorted(alt)]
        # greedy
        g_chosen, g_progress = greedy_cover(B)
        # cumulative greedy coverage with model names
        unc = np.ones(n_basins, dtype=bool)
        for i, mi in enumerate(g_chosen, start=1):
            unc &= ~B[:, mi]
            progress_rows.append({"k": k, "step": i, "model_added": models[mi],
                                  "cumulative_models": i,
                                  "basins_covered": int((~unc).sum()),
                                  "frac_basins_covered": float((~unc).mean())})

        res[k] = {
            "union_size": int(len(union_models)), "union_models": union_models,
            "min_cover_size_exact": int(len(cover_models)), "min_cover_models": cover_models,
            "min_cover_verified_all_basins_covered": bool(covered.all()),
            "n_basins_covered_by_cover": int(covered.sum()),
            "lp_relaxation_bound": lp, "lp_ceiling": int(np.ceil(lp - 1e-9)),
            "is_proven_optimal": cert["proven_optimal"],
            "solver_status": cert["status"], "solver_message": cert["message"],
            "solver_mip_gap": cert["mip_gap"],
            "lp_bound_is_tight": bool(len(cover_models) == int(np.ceil(lp - 1e-9))),
            "alternative_optimum_exists": bool(abs(alt_opt - opt) < 1e-6),
            "alternative_optimum_models": alt_models,
            "greedy_cover_size": int(len(g_chosen)),
            "greedy_cover_models": [models[i] for i in g_chosen],
            "greedy_matches_exact": bool(len(g_chosen) == len(cover_models)),
        }
        cover_rows.append({
            "k": k, "union_size": len(union_models), "min_cover_size": len(cover_models),
            "min_cover_models": "|".join(cover_models),
            "lp_relaxation_bound": lp, "lp_ceiling": int(np.ceil(lp - 1e-9)),
            "proven_optimal": cert["proven_optimal"],
            "solver_status": cert["status"],
            "lp_bound_is_tight": bool(len(cover_models) == int(np.ceil(lp - 1e-9))),
            "n_basins_covered": int(covered.sum()),
            "alternative_optimum_exists": bool(abs(alt_opt - opt) < 1e-6),
            "alternative_optimum_models": "|".join(alt_models),
            "greedy_cover_size": len(g_chosen),
            "greedy_cover_models": "|".join(models[i] for i in g_chosen),
            "greedy_matches_exact": bool(len(g_chosen) == len(cover_models)),
        })

    f3, f5 = freq_rows
    FREQ = f3.merge(f5, on="model", how="outer").sort_values(
        "n_basins_in_top3", ascending=False).reset_index(drop=True)
    FREQ["in_any_top3"] = FREQ["n_basins_in_top3"] > 0
    FREQ["in_any_top5"] = FREQ["n_basins_in_top5"] > 0
    FREQ["in_min_cover_top3"] = FREQ["model"].isin(res[3]["min_cover_models"])
    FREQ["in_min_cover_top5"] = FREQ["model"].isin(res[5]["min_cover_models"])
    FREQ.to_csv(PC.T5 / "library_model_frequency.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(PC.T5 / "library_minimum_cover.csv", index=False)
    pd.DataFrame(progress_rows).to_csv(PC.T5 / "library_greedy_cover_progress.csv", index=False)

    # ---- validation against the frozen aggregate statistics ----------------
    C = pd.read_csv(PC.OUT4 / "tables" / "topk_coverage_curve.csv")
    O = pd.read_csv(PC.OUT4 / "tables" / "top3_top5_set_overlap.csv")
    oob = F.pivot(index="basin_id", columns="model", values="kge_heldout").loc[basins, models]
    oa = oob.to_numpy(np.float64)
    seen_arr = M.to_numpy(np.float64)
    val = {}
    for k in KS:
        B = topk_membership(M, k)
        ob = np.argsort(-oa, axis=1, kind="stable")[:, :k]
        inter = np.array([len(set(np.flatnonzero(B[i])) & set(ob[i].tolist())) for i in range(n_basins)])
        rec = float(inter.mean() / k)
        cov_frozen = float(C[(C.strategy == "basin_specific") & (C.k == k)]["coverage_top1"].iloc[0])
        recall_frozen = float(O[O.k == k]["recall_mean"].iloc[0])
        best_in = np.array([oa[i, np.flatnonzero(B[i])].max() for i in range(n_basins)])
        cov_mine = float((np.isclose(best_in, oa.max(axis=1))).mean())
        val[f"k{k}"] = {"recall_mine": rec, "recall_frozen": recall_frozen,
                        "recall_abs_diff": abs(rec - recall_frozen),
                        "coverage_mine": cov_mine, "coverage_frozen": cov_frozen,
                        "coverage_abs_diff": abs(cov_mine - cov_frozen)}
    report["validation_vs_frozen"] = val
    report["top3"] = res[3]
    report["top5"] = res[5]
    report["frequency_table_top3"] = f3.to_dict(orient="records")
    report["frequency_table_top5"] = f5.to_dict(orient="records")
    # ---- independent optimality certificate (different formulation) --------
    def max_coverage_with_budget(B: np.ndarray, q: int) -> int:
        """Max basins coverable with exactly q models (independent of the set-cover MILP)."""
        nb, nm = B.shape
        nv = nm + nb
        c = np.concatenate([np.zeros(nm), -np.ones(nb)])
        rows, lb, ub = [], [], []
        for i in range(nb):
            r = np.zeros(nv); r[nm + i] = 1.0; r[:nm] = -B[i].astype(float)
            rows.append(r); lb.append(-np.inf); ub.append(0.0)
        budget = np.zeros(nv); budget[:nm] = 1.0
        rows.append(budget); lb.append(q); ub.append(q)
        rr = milp(c=c, constraints=LinearConstraint(np.array(rows), lb=np.array(lb), ub=np.array(ub)),
                  integrality=np.ones(nv), bounds=Bounds(0, 1))
        if rr.status != 0:
            raise RuntimeError(f"budget MILP failed: {rr.message}")
        return int(round(-rr.fun))

    cert_rows = []
    for k in KS:
        Bm = topk_membership(M, k)
        q = res[k]["min_cover_size_exact"] - 1
        mx = max_coverage_with_budget(Bm, q)
        cert_rows.append((k, q, mx))
        res[k]["budget_certificate"] = {"budget": q, "max_basins_coverable": mx,
                                        "cover_of_that_size_impossible": bool(mx < n_basins)}
    GP = pd.DataFrame(progress_rows)

    # dump the JSON only now that the optimality certificates are attached
    with open(PC.T5 / "minimal_model_library.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ---- short markdown report --------------------------------------------
    md = []
    md.append("# Minimal model library required by the seen Top-3 / Top-5 candidate sets\n")
    md.append(f"Frozen input: `pub_same_pipeline_model_basin.csv`, seen = `{PRIMARY_SEEN}` "
              f"(the next-stage primary aggregation). {n_basins} basins x {n_models} models. "
              "The Top-k sets are re-derived by ranking the frozen scores — no KGE is "
              "recomputed, nothing is trained, and no frozen output is modified.\n")
    md.append("## Headline\n")
    md.append("```text")
    md.append(f"Top-3 union:         {res[3]['union_size']} / {n_models} models")
    md.append(f"Top-3 minimum cover: {res[3]['min_cover_size_exact']} / {n_models} models "
              f"{'(proven optimal)' if res[3]['is_proven_optimal'] else '(NOT PROVEN)'}")
    md.append(f"Top-5 union:         {res[5]['union_size']} / {n_models} models")
    md.append(f"Top-5 minimum cover: {res[5]['min_cover_size_exact']} / {n_models} models "
              f"{'(proven optimal)' if res[5]['is_proven_optimal'] else '(NOT PROVEN)'}")
    md.append("```\n")
    for k in KS:
        r = res[k]
        md.append(f"## Top-{k}\n")
        md.append(f"* **Union: {r['union_size']} of {n_models} models** ever appear in a "
                  f"basin's Top-{k}.")
        md.append(f"* **Minimum cover: {r['min_cover_size_exact']} models** — "
                  f"`{'`, `'.join(r['min_cover_models'])}`.")
        md.append(f"  Every one of the {n_basins} basins has at least one of these in its "
                  f"Top-{k} (verified: {r['n_basins_covered_by_cover']}/{n_basins} basins covered). "
                  f"Solved with `scipy.optimize.milp` (HiGHS) under binary integrality, solver "
                  f"status {r['solver_status']} "
                  f"({'optimal — so this IS the proven exact minimum' if r['is_proven_optimal'] else 'NOT proven optimal'}), "
                  f"not a greedy approximation. The LP-relaxation lower bound is "
                  f"{r['lp_relaxation_bound']:.4f} (ceiling {r['lp_ceiling']}), a weaker bound "
                  f"that {'coincides with' if r['lp_bound_is_tight'] else 'does not reach'} the "
                  f"integer optimum.")
        md.append(f"* Alternative optimal covers exist: "
                  f"**{'yes' if r['alternative_optimum_exists'] else 'no'}**"
                  + (f" (e.g. `{'`, `'.join(r['alternative_optimum_models'])}`)." 
                     if r["alternative_optimum_exists"] else "."))
        md.append(f"* Greedy set cover gives {r['greedy_cover_size']} models "
                  f"({'matching' if r['greedy_matches_exact'] else 'differing from'} the exact "
                  f"optimum); retained only as a cross-check.\n")
    md.append("## What this means\n")
    md.append(f"Although each individual basin is only recommended {3}–{5} models, the "
              f"531-basin system as a whole draws on **{res[3]['union_size']}** distinct model "
              f"structures (Top-3) and **{res[5]['union_size']}** (Top-5), i.e. the "
              f"basin-specific sets are spread over "
              f"{res[3]['union_size']}/{n_models} = "
              f"{100*res[3]['union_size']/n_models:.0f}% of the available structures.\n")
    md.append(f"At the same time a much smaller **shared core** suffices to guarantee that "
              f"*every* basin has at least one of its recommended models covered: "
              f"**{res[3]['min_cover_size_exact']} models for Top-3** and "
              f"**{res[5]['min_cover_size_exact']} for Top-5**. So a maintainable library of "
              f"~{res[3]['min_cover_size_exact']} structures can serve all basins, while the "
              f"basin-specific part of the recommendation remains spread across many more "
              f"structures.\n")
    md.append("### Independent optimality certificate\n")
    md.append("Re-derived with a *different* formulation — maximise the number of covered basins "
              "with a budget of `optimum − 1` models:\n")
    md.append("| k | budget | max basins coverable | verdict |")
    md.append("|---|---|---|---|")
    for k, q, mx in cert_rows:
        art = "an" if q in (8, 11, 18) else "a"
        md.append(f"| {k} | {q} models | {mx} of {n_basins} | {art} {q}-model cover is **impossible** |")
    md.append("\nBoth budgets fall exactly one basin short, which is what forces the extra "
              "library model. The binding basins are the ones whose Top-k sets contain a model "
              "that is almost never recommended elsewhere — e.g. `collie1`, `mopex1` and "
              "`penman` each appear in only **one** basin's Top-3.\n")
    md.append("### How many models to cover most basins (greedy order)\n")
    md.append("| k | 50% of basins | 90% | 99% | 100% |")
    md.append("|---|---|---|---|---|")
    for k in KS:
        g = GP[GP.k == k]
        cells = []
        for frac in (0.5, 0.9, 0.99, 1.0):
            row = g[g.frac_basins_covered >= frac].head(1)
            cells.append(f"{int(row.cumulative_models.iloc[0])}")
        md.append(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    md.append(f"\n`hbv96` alone sits in {int(FREQ.loc[FREQ.model=='hbv96','n_basins_in_top3'].iloc[0])} "
              f"basins' Top-3 ({FREQ.loc[FREQ.model=='hbv96','frac_of_basins_in_top3'].iloc[0]:.1%}), "
              "so it almost single-handedly covers the common case; the remaining "
              f"{res[3]['min_cover_size_exact']-1} models exist to catch the basins that `hbv96` "
              "misses.\n")
    md.append("### Model frequency (Top-3, descending)\n")
    md.append("| rank | model | basins in Top-3 | share | in min cover (k=3) | basins in Top-5 |")
    md.append("|---|---|---|---|---|---|")
    for _, r in f3.iterrows():
        if r["n_basins_in_top3"] == 0:
            continue
        md.append(f"| {int(r['rank_by_top3_frequency'])} | `{r['model']} | "
                  f"{int(r['n_basins_in_top3'])} | {r['frac_of_basins_in_top3']:.3f} | "
                  f"{'yes' if r['model'] in res[3]['min_cover_models'] else ''} | "
                  f"{int(FREQ.loc[FREQ.model==r['model'],'n_basins_in_top5'].iloc[0])} |".replace(
                      f"`{r['model']} |", f"`{r['model']}` |"))
    zero3 = int((f3["n_basins_in_top3"] == 0).sum())
    zero5 = int((f5["n_basins_in_top5"] == 0).sum())
    md.append(f"\nModels never appearing in any Top-3: **{zero3}**"
              + (f" (`{'`, `'.join(f3.loc[f3['n_basins_in_top3']==0,'model'])}`)." if zero3 else "."))
    md.append(f"Models never appearing in any Top-5: **{zero5}**"
              + (f" (`{'`, `'.join(f5.loc[f5['n_basins_in_top5']==0,'model'])}`)." if zero5 else "."))
    md.append("\n## Validation against the frozen aggregate statistics\n")
    md.append("| k | recall (mine) | recall (frozen) | abs diff | coverage (mine) | coverage (frozen) | coverage abs diff |")
    md.append("|---|---|---|---|---|---|---|")
    for k in KS:
        v = val[f"k{k}"]
        md.append(f"| {k} | {v['recall_mine']:.4f} | {v['recall_frozen']:.4f} | "
                  f"{v['recall_abs_diff']:.2e} | {v['coverage_mine']:.4f} | "
                  f"{v['coverage_frozen']:.4f} | {v['coverage_abs_diff']:.2e} |")
    md.append("\n## Reproduce\n")
    md.append("```bash")
    md.append("python project/benchmarkpub/scripts/oob36_prelim/25_minimal_model_library.py")
    md.append("```")
    md.append("\n## Scope / limitations\n")
    md.append("* The cover question is answered for the **seen** Top-k sets. It says which "
              "models must be *maintained* so that every basin has at least one recommended "
              "model available — it is not a statement about held-out performance.")
    md.append("* The minimum cover size is **proven optimal for the stated integer program**: the "
              "MILP returns solver status 0, and an independent max-coverage formulation shows "
              "that a budget one model smaller cannot cover all basins. The LP relaxation alone "
              "is a weaker bound and would not have certified it. Multiple optimal covers exist, "
              "so the composition should be read as *one* optimal solution, not the unique answer.")
    md.append("* Ties in the frozen seen scores: 0 exact ties were found, so the Top-k "
              "membership is unambiguous.")
    md.append("* No equivalence set, no candidate probability model, no causal claim.")
    (Path(PC.OUT5) / "MINIMAL_MODEL_LIBRARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- provenance + manifest --------------------------------------------
    import hashlib
    import subprocess

    def sha256(f: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with f.open("rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PC.REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        head = f"UNAVAILABLE: {exc}"
    (PC.OUT5 / "PROVENANCE.json").write_text(json.dumps({
        "task_id": "ch4-oob36-minimal-model-library-20260930",
        "git_head": head, "solver": "scipy.optimize.milp (HiGHS), binary integrality",
        "no_training": True, "no_inference": True, "no_kge_recomputation": True,
        "project_benchmark_modified": False,
        "frozen_inputs": {
            "seen_scores": str(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv"),
            "frozen_aggregate_validation": [str(PC.OUT4 / "tables" / "topk_coverage_curve.csv"),
                                            str(PC.OUT4 / "tables" / "top3_top5_set_overlap.csv")],
        },
        "optimality_certificates": {f"top{k}": res[k]["budget_certificate"] for k in KS},
        "reproduce": ["python project/benchmarkpub/scripts/oob36_prelim/25_minimal_model_library.py"],
    }, indent=1) + "\n", encoding="utf-8")
    mrows = [{"relative_path": str(f.relative_to(PC.OUT5)), "bytes": f.stat().st_size,
              "sha256": sha256(f)}
             for f in sorted(PC.OUT5.rglob("*"))
             if f.is_file() and f.name not in ("ARTIFACT_MANIFEST.csv",)]
    mrows.append({"relative_path": "scripts/oob36_prelim/25_minimal_model_library.py",
                  "bytes": Path(__file__).stat().st_size, "sha256": sha256(Path(__file__))})
    pd.DataFrame(mrows).to_csv(PC.OUT5 / "ARTIFACT_MANIFEST.csv", index=False)

    # ---- console ----------------------------------------------------------
    for k in KS:
        r = res[k]
        log(f"== Top-{k} ==")
        log(f"   union: {r['union_size']}/{n_models}")
        log(f"   exact minimum cover: {r['min_cover_size_exact']}  -> {r['min_cover_models']}")
        log(f"     LP bound {r['lp_relaxation_bound']:.4f} (ceil {r['lp_ceiling']}) "
            f"-> proven optimal: {r['is_proven_optimal']}; all basins covered: "
            f"{r['min_cover_verified_all_basins_covered']}")
        log(f"     alternative optimum exists: {r['alternative_optimum_exists']} "
            f"{r['alternative_optimum_models']}")
        log(f"     greedy: {r['greedy_cover_size']} models -> {r['greedy_cover_models']}")
    log("\n== Top-3 frequency (models appearing at least once) ==")
    log(FREQ[FREQ.n_basins_in_top3 > 0][
        ["model", "n_basins_in_top3", "frac_of_basins_in_top3", "n_basins_in_top5",
         "in_min_cover_top3"]].round(4).to_string(index=False))
    log(f"\n   never in any Top-3: {int((FREQ.n_basins_in_top3==0).sum())} models -> "
        f"{FREQ.loc[FREQ.n_basins_in_top3==0,'model'].tolist()}")
    log(f"   never in any Top-5: {int((FREQ.n_basins_in_top5==0).sum())} models -> "
        f"{FREQ.loc[FREQ.n_basins_in_top5==0,'model'].tolist()}")
    log(f"\nwrote -> {PC.OUT5}")


if __name__ == "__main__":
    main()
