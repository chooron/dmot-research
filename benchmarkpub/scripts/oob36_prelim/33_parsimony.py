#!/usr/bin/env python3
"""33_parsimony.py — stage 3: does parsimony resolve the ambiguity left by accuracy + adequacy?

FROZEN PIPELINE UNDER TEST
--------------------------
    Accuracy   : Seen Top-5  (frozen seen-KGE ranking, kge_seenfolds_median4)
    Adequacy   : RELAXED_70  (frozen relaxed-adequacy flags)
    Parsimony  : this script

CENTRAL QUESTION
----------------
Among candidates that are already accurate and hydrologically adequate, can we usually prefer the
SIMPLEST one at almost no cost in out-of-basin predictive performance?

COMPLEXITY INVENTORY (see §1 output)
------------------------------------
Only TWO structural dimensions in this repository are uniformly defined and verifiable for all 36
models, both frozen in `chapter4_multicriteria_v3_20260925/tables/v1_parsimony.csv`:

    n_parameters   dmotpy/models/core/__init__.py:314  NPARAM_INFO   (1..15)
    n_states       dmotpy/models/core/__init__.py:354  STATE_INFO    (1..6)

Everything else is NOT_COMPARABLE and is declared as such rather than approximated:

    n_fluxes           no flux registry exists (FLUX_INFO absent)
    n_process_modules  no module registry exists (MODULE_INFO absent); the *_step_pre /
                       *_step_post sub-step split exists for only 4 of 36 models
    NUMBER_INFO        is the MARRMoT reference MODEL NUMBER, not a count.  The frozen
                       v3_common.py:21 states it is deliberately not used as a complexity
                       measure, so it is excluded here too.

Consequently the primary metric is `n_parameters` and the result must be called
**parameter-count parsimony**, not universal structural simplicity.  `n_states` is the
pre-registered secondary sensitivity.

SELECTION RULE DISCIPLINE
-------------------------
No selection below ever uses OOB KGE.  The realisable rules use only the seen ranking, the
frozen adequacy flags and the frozen complexity metadata.  Where a complexity tie must be broken,
the primary tie-break is the SEEN KGE (rule-available); OOB KGE is used only to bracket the
tie-break's effect post hoc (`_optimistic` / `_conservative`) with the tie frequency reported.

NO training, NO inference, NO KGE recomputation, NO torch, NO modification of any frozen output.

Outputs -> $OUT8/tables/{model_complexity_inventory,parsimony_model_level_relationships,
                         basin_parsimony_selection,simplest_vs_best_adequate,
                         parsimony_regret_summary,pareto_summary,fallback_comparison,
                         parsimony_sensitivity}.csv
           $OUT8/tables/parsimony.json
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
N_BOOT = 2000
BIG = 1e9
REG_BANDS = (0.01, 0.02, 0.05)
V1 = PC.BENCH / "analysis" / "chapter4_multicriteria_v3_20260925" / "tables" / "v1_parsimony.csv"
REG_NOTE = ("minimise n_parameters among the adequate candidates; the primary tie-break is the "
            "SEEN KGE (rule-available). OOB KGE is used only to bracket the tie-break post hoc.")


def log(m: str) -> None:
    print(m, flush=True)


def boot_median(x: np.ndarray, B: int = N_BOOT, seed: int = SEED):
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = np.median(v[rng.integers(0, v.size, size=(B, v.size))], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def regret_stats(reg: np.ndarray, mask: np.ndarray | None = None) -> dict:
    r = np.asarray(reg, float)
    if mask is not None:
        r = r[mask]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {k: np.nan for k in ("n", "median", "mean", "q75", "q90", "q95")} | \
               {f"frac_le_{b}": np.nan for b in REG_BANDS}
    lo, hi = boot_median(r)
    return {"n": int(r.size), "median": float(np.median(r)), "mean": float(np.mean(r)),
            "q75": float(np.percentile(r, 75)), "q90": float(np.percentile(r, 90)),
            "q95": float(np.percentile(r, 95)),
            "median_ci95_lo": lo, "median_ci95_hi": hi,
            **{f"frac_le_{b}": float((r <= b).mean()) for b in REG_BANDS}}


def main() -> None:
    PC.T8.mkdir(parents=True, exist_ok=True)
    A2 = PC.OUT1 / "pub_adequacy_seen_oob_20260930"
    A7 = PC.OUT1 / "pub_relaxed_adequacy_20260930"

    # ------------------------------------------------------------------ inputs
    pars = pd.read_csv(V1)
    F = pd.read_csv(PC.OUT2 / "tables" / "pub_same_pipeline_model_basin.csv")
    RA = pd.read_csv(A7 / "tables" / "relaxed_adequacy_model_basin.csv")
    frozen_sets = pd.read_csv(A2 / "tables" / "candidate_set_adequacy_by_basin.csv")

    models = sorted(F["model"].unique())
    basins = np.sort(F["basin_id"].unique())
    nb, nm = len(basins), len(models)
    bpos = {int(b): i for i, b in enumerate(basins)}
    mpos = {m: j for j, m in enumerate(models)}
    if len(models) != 36 or nb != 531:
        raise ValueError("expected 36 models x 531 basins")

    report = {"meta": {
        "task_id": "ch4-oob36-parsimony-20260930",
        "n_models": nm, "n_basins": nb,
        "primary_parsimony_metric": "n_parameters",
        "metric_name_for_paper": "parameter-count parsimony (NOT universal structural simplicity)",
        "secondary_metric": "n_states",
        "not_comparable": {"n_fluxes": "no flux registry (FLUX_INFO absent)",
                           "n_process_modules": "no module registry (MODULE_INFO absent); "
                                                "*_step_pre/_post exists for only 4/36 models",
                           "NUMBER_INFO": "MARRMoT reference model number, not a count"},
        "pipeline": "Accuracy: Seen Top-5 -> Adequacy: RELAXED_70 -> Parsimony: simplest adequate",
        "tie_break": REG_NOTE,
        "no_oob_truth_in_selection": True,
        "no_training": True, "no_inference": True, "no_torch": True,
    }}

    # ================================================================== §1
    inv = pd.DataFrame({
        "model": models,
        "n_parameters": [int(pars.set_index("model").loc[m, "n_parameters"]) for m in models],
        "n_states": [int(pars.set_index("model").loc[m, "n_states"]) for m in models],
        "n_fluxes": np.nan, "n_process_modules": np.nan,
        "source_n_parameters": "dmotpy/models/core/__init__.py:314 NPARAM_INFO",
        "source_n_states": "dmotpy/models/core/__init__.py:354 STATE_INFO",
        "comparability_flag_n_parameters": "COMPARABLE",
        "comparability_flag_n_states": "COMPARABLE",
        "comparability_flag_n_fluxes": "NOT_COMPARABLE",
        "comparability_flag_n_process_modules": "NOT_COMPARABLE",
        "n_fluxes_note": "no flux registry in dmotpy (FLUX_INFO absent); NUMBER_INFO is the "
                         "MARRMoT model number and is not a count",
        "n_process_modules_note": "no module registry (MODULE_INFO absent); the *_step_pre/_post "
                                 "split exists for only 4 of 36 models",
        "frozen_source_file": "chapter4_multicriteria_v3_20260925/tables/v1_parsimony.csv",
    })
    inv = inv.sort_values(["n_parameters", "n_states", "model"]).reset_index(drop=True)
    inv.to_csv(PC.T8 / "model_complexity_inventory.csv", index=False)
    report["complexity_inventory"] = {
        "n_parameters": {"min": int(inv.n_parameters.min()), "max": int(inv.n_parameters.max()),
                         "median": float(inv.n_parameters.median()),
                         "n_distinct": int(inv.n_parameters.nunique()),
                         "argmin": inv.loc[inv.n_parameters.idxmin(), "model"],
                         "argmax": inv.loc[inv.n_parameters.idxmax(), "model"]},
        "n_states": {"min": int(inv.n_states.min()), "max": int(inv.n_states.max()),
                     "median": float(inv.n_states.median()),
                     "n_distinct": int(inv.n_states.nunique())},
        "spearman_nparams_vs_nstates": float(
            sstats.spearmanr(inv.n_parameters, inv.n_states).statistic),
    }

    # ------------------------------------------------------------- grids
    def grid(col, src, dtype=float, fill=np.nan):
        g = np.full((nb, nm), fill, dtype=dtype)
        bi = src["basin_id"].map(bpos).to_numpy()
        mi = src["model"].map(mpos).to_numpy()
        if bi.min() < 0 or mi.min() < 0 or len(src) != nb * nm:
            raise ValueError(f"grid({col}) key mapping failed")
        g[bi, mi] = src[col].to_numpy()
        return g

    G_oob = grid("kge_heldout", F)
    G_seen = grid("kge_seenfolds_median4", F)
    G_npar = np.array([pars.set_index("model").loc[m, "n_parameters"] for m in models],
                      float)[None, :].repeat(nb, 0)
    G_nst = np.array([pars.set_index("model").loc[m, "n_states"] for m in models],
                     float)[None, :].repeat(nb, 0)
    for c in ("oob_RELAXED_60", "oob_RELAXED_70", "oob_RELAXED_80",
              "seen_RELAXED_60", "seen_RELAXED_70", "seen_RELAXED_80"):
        if c not in RA.columns:
            raise ValueError(f"relaxed_adequacy_model_basin.csv lacks {c}")
    A60o = grid("oob_RELAXED_60", RA, bool, False)
    A70o = grid("oob_RELAXED_70", RA, bool, False)
    A80o = grid("oob_RELAXED_80", RA, bool, False)
    A60s = grid("seen_RELAXED_60", RA, bool, False)
    A70s = grid("seen_RELAXED_70", RA, bool, False)
    A80s = grid("seen_RELAXED_80", RA, bool, False)
    SFO = grid("signature_pass_fraction_oob", RA)
    SFS = grid("signature_pass_fraction_seen", RA)
    if not (np.isfinite(G_oob).all() and np.isfinite(G_seen).all()):
        raise ValueError("KGE grids contain non-finite values")
    oracle = G_oob.max(axis=1)
    rr = np.arange(nb)[:, None]

    # seen-KGE ranking -> candidate pools; validated against the frozen membership
    order = np.argsort(-G_seen, axis=1, kind="stable")
    mism = 0
    for k in (3, 5):
        sub = frozen_sets[frozen_sets.k == k].set_index("basin_id").reindex(basins)
        for i, b in enumerate(basins):
            if sorted(models[j] for j in order[i, :k]) != sorted(str(sub.loc[int(b), "candidates"]).split("|")):
                mism += 1
    report["validation"] = {"topk_membership_mismatches_vs_frozen": int(mism)}
    if mism:
        raise AssertionError("Top-k membership differs from the frozen candidate sets")

    def pool(k: int, col: int = -1) -> np.ndarray:
        p = np.zeros((nb, nm), bool)
        p[rr, order[:, :k]] = True
        return p

    # ------------------------------------------------- selection primitives
    def pick_simplest(mask: np.ndarray, tb_hi: np.ndarray | None = None,
                      tb_lo: np.ndarray | None = None):
        """Index of the min-n_parameters candidate in `mask`.

        Primary tie-break: `tb_hi` descending (defaults to the SEEN KGE, which is rule-available).
        `tb_lo` ascending is used only for the post-hoc conservative bracket.
        """
        hi = G_seen if tb_hi is None else tb_hi
        np_m = np.where(mask, G_npar, BIG)
        hi_m = np.where(mask, hi, -BIG)
        lo_m = np.where(mask, tb_lo if tb_lo is not None else -hi, BIG)
        if tb_lo is None:
            idx = np.lexsort((-hi_m, np_m), axis=1)[:, 0]
        else:
            idx = np.lexsort((lo_m, np_m), axis=1)[:, 0]
        return idx, mask[rr[:, 0], idx]

    def pick_best(mask: np.ndarray, score: np.ndarray):
        s = np.where(mask, score, -BIG)
        idx = np.argmax(s, axis=1)
        return idx, mask[rr[:, 0], idx]

    def kge_at(idx: np.ndarray) -> np.ndarray:
        return G_oob[rr[:, 0], idx]

    # ------------------------------------------------- per-basin selection (§4-6)
    top5 = pool(5)
    adeq = top5 & A70o                              # §4 pipeline: Top-5 AND OOB-RELAXED_70
    n_adeq = adeq.sum(axis=1)
    has = n_adeq >= 1

    i_simple, ok_simple = pick_simplest(adeq)
    i_simple_opt, _ = pick_simplest(adeq, tb_hi=G_oob, tb_lo=None)
    i_simple_con, _ = pick_simplest(adeq, tb_hi=None, tb_lo=G_oob)
    i_best, ok_best = pick_best(adeq, G_oob)          # oracle-within-adequate (upper bound)
    i_seenkge, _ = pick_best(adeq, G_seen)            # rule-available competitor (no OOB truth)
    i_top5best, _ = pick_best(top5, G_oob)
    npar_masked = np.where(adeq, G_npar, BIG)                 # [nb, nm]
    min_npar = npar_masked.min(axis=1)                        # [nb]
    n_min_tied = np.where(has, (npar_masked == min_npar[:, None]).sum(axis=1), 0)

    D = pd.DataFrame({
        "basin_id": basins,
        "gauge_id_str": [str(int(b)).zfill(8) for b in basins],
        "n_adequate_in_top5": n_adeq,
        "has_adequate": has,
        "idx_simplest_adequate": i_simple, "idx_best_oob_adequate": i_best,
        "model_simplest_adequate": [models[j] for j in i_simple],
        "model_best_oob_adequate": [models[j] for j in i_best],
        "model_best_oob_in_top5": [models[j] for j in i_top5best],
        "n_min_complexity_tied": n_min_tied,
        "n_parameters_simplest_adequate": G_npar[rr[:, 0], i_simple],
        "n_parameters_best_oob_adequate": G_npar[rr[:, 0], i_best],
        "n_states_simplest_adequate": G_nst[rr[:, 0], i_simple],
        "n_states_best_oob_adequate": G_nst[rr[:, 0], i_best],
        "kge_oob_simplest_adequate": kge_at(i_simple),
        "kge_oob_simplest_adequate_optimistic": kge_at(i_simple_opt),
        "kge_oob_simplest_adequate_conservative": kge_at(i_simple_con),
        "kge_oob_best_oob_adequate": kge_at(i_best),
        "model_best_seen_kge_adequate": [models[j] for j in i_seenkge],
        "kge_oob_best_seen_kge_adequate": kge_at(i_seenkge),
        "kge_oob_best_in_top5": kge_at(i_top5best),
        "kge_oob_oracle": oracle,
    })
    D["same_model_best_vs_simplest"] = D["model_best_oob_adequate"] == D["model_simplest_adequate"]
    D["delta_n_parameters"] = D["n_parameters_best_oob_adequate"] - D["n_parameters_simplest_adequate"]
    D["delta_n_states"] = D["n_states_best_oob_adequate"] - D["n_states_simplest_adequate"]
    D["regret_simplest_vs_oracle"] = D["kge_oob_oracle"] - D["kge_oob_simplest_adequate"]
    D["regret_best_adequate_vs_oracle"] = D["kge_oob_oracle"] - D["kge_oob_best_oob_adequate"]
    D["regret_simplest_vs_best_adequate"] = D["kge_oob_best_oob_adequate"] - D["kge_oob_simplest_adequate"]
    D["regret_top5best_vs_oracle"] = D["kge_oob_oracle"] - D["kge_oob_best_in_top5"]
    D["regret_simplest_optimistic_vs_oracle"] = D["kge_oob_oracle"] - D["kge_oob_simplest_adequate_optimistic"]
    D["regret_simplest_conservative_vs_oracle"] = D["kge_oob_oracle"] - D["kge_oob_simplest_adequate_conservative"]
    D["regret_best_seen_kge_adequate_vs_oracle"] = D["kge_oob_oracle"] - D["kge_oob_best_seen_kge_adequate"]
    D["regret_simplest_vs_best_seen_kge_adequate"] = (D["kge_oob_best_seen_kge_adequate"]
                                                      - D["kge_oob_simplest_adequate"])
    D.to_csv(PC.T8 / "basin_parsimony_selection.csv", index=False)

    # ================================================================== §3
    m_agg = pd.DataFrame({
        "model": models,
        "n_parameters": [int(pars.set_index("model").loc[m, "n_parameters"]) for m in models],
        "n_states": [int(pars.set_index("model").loc[m, "n_states"]) for m in models],
        "kge_seen_median": np.median(G_seen, axis=0),
        "kge_oob_median": np.median(G_oob, axis=0),
        "delta_kge_pub_median": np.median(G_seen - G_oob, axis=0),
    })
    rows3 = []
    for metrics_name in ("n_parameters", "n_states"):
        for tgt in ("kge_seen_median", "kge_oob_median", "delta_kge_pub_median"):
            rho, p = sstats.spearmanr(m_agg[metrics_name], m_agg[tgt])
            rows3.append({"complexity_metric": metrics_name, "target": tgt,
                          "spearman_rho": float(rho), "p_value": float(p), "n_models": len(m_agg)})
    R3 = pd.DataFrame(rows3)
    R3.to_csv(PC.T8 / "parsimony_model_level_relationships.csv", index=False)
    report["model_level_relationships"] = R3.to_dict("records")

    # ================================================================== §5/§7 regret
    rowsR = []
    for label, reg, msk in (
        ("Rule5_best_in_seen_top5", D["kge_oob_oracle"] - D["kge_oob_best_in_top5"], None),
        ("best_OOB_adequate (C)", D["regret_best_adequate_vs_oracle"], has),
        ("best_SEEN-KGE_adequate (rule-available)", D["regret_best_seen_kge_adequate_vs_oracle"], has),
        ("simplest_adequate (D)", D["regret_simplest_vs_oracle"], has),
        ("D_optimistic_tiebreak", D["regret_simplest_optimistic_vs_oracle"], has),
        ("D_conservative_tiebreak", D["regret_simplest_conservative_vs_oracle"], has),
    ):
        rowsR.append({"scenario": label, "regret_reference": "OOB oracle (all 36)",
                      **regret_stats(reg.to_numpy(), msk)})
    rowsR.append({"scenario": "simplest_adequate_cost (D vs C)",
                  "regret_reference": "best OOB-adequate candidate (upper bound)",
                  **regret_stats(D["regret_simplest_vs_best_adequate"].to_numpy(), has)})
    rowsR.append({"scenario": "simplest_adequate_cost (D vs best SEEN-KGE adequate)",
                  "regret_reference": "best SEEN-KGE adequate candidate (rule-available)",
                  **regret_stats(D["regret_simplest_vs_best_seen_kge_adequate"].to_numpy(), has)})
    REG = pd.DataFrame(rowsR)
    REG.to_csv(PC.T8 / "parsimony_regret_summary.csv", index=False)
    report["regret_summary"] = REG.to_dict("records")

    # ================================================================== §6/§7 complexity
    h = has
    same = D["same_model_best_vs_simplest"].to_numpy()
    dn = D["delta_n_parameters"].to_numpy()
    dns = D["delta_n_states"].to_numpy()
    red = h & (dn > 0)
    S6 = {
        "n_basins_with_adequate": int(h.sum()),
        "coverage_fraction": float(h.mean()),
        "same_model_fraction": float(same[h].mean()),
        "fraction_complexity_reduced": float((dn[h] > 0).mean()),
        "fraction_complexity_unchanged": float((dn[h] == 0).mean()),
        "median_delta_n_parameters": float(np.median(dn[h])),
        "mean_delta_n_parameters": float(np.mean(dn[h])),
        "q25_delta_n_parameters": float(np.percentile(dn[h], 25)),
        "q75_delta_n_parameters": float(np.percentile(dn[h], 75)),
        "median_delta_n_parameters_when_reduced": float(np.median(dn[red])) if red.any() else None,
        "median_delta_n_states": float(np.median(dns[h])),
        "fraction_n_states_reduced": float((dns[h] > 0).mean()),
        "median_n_parameters_simplest": float(np.median(D["n_parameters_simplest_adequate"].to_numpy()[h])),
        "median_n_parameters_best": float(np.median(D["n_parameters_best_oob_adequate"].to_numpy()[h])),
        "kge_cost_when_different_median": float(np.median(
            D["regret_simplest_vs_best_adequate"].to_numpy()[h & ~same])),
        "kge_cost_when_reduced_median": float(np.median(
            D["regret_simplest_vs_best_adequate"].to_numpy()[red])) if red.any() else None,
        "tie_frequency_at_min_complexity": float((D["n_min_complexity_tied"].to_numpy()[h] > 1).mean()),
        "median_n_tied_at_min_complexity": float(np.median(D["n_min_complexity_tied"].to_numpy()[h])),
    }
    pd.DataFrame([S6]).to_csv(PC.T8 / "simplest_vs_best_adequate.csv", index=False)
    report["simplest_vs_best"] = S6

    # ================================================================== §8 Pareto
    par_sizes, front_has = [], []
    for i in range(nb):
        m = adeq[i]
        if not m.any():
            par_sizes.append(0); continue
        idx = np.flatnonzero(m)
        k = G_oob[i, idx]; c = G_npar[i, idx]
        keep = np.ones(idx.size, bool)
        for a in range(idx.size):
            for b in range(idx.size):
                if a == b:
                    continue
                if (k[b] >= k[a]) and (c[b] <= c[a]) and ((k[b] > k[a]) or (c[b] < c[a])):
                    keep[a] = False
                    break
        par_sizes.append(int(keep.sum()))
    par_sizes = np.array(par_sizes)
    PH = par_sizes[h]
    PARE = {
        "n_basins_with_adequate": int(h.sum()),
        "median_pareto_set_size": float(np.median(PH)),
        "mean_pareto_set_size": float(np.mean(PH)),
        "fraction_single_pareto_candidate": float((PH == 1).mean()),
        "fraction_two_or_more": float((PH >= 2).mean()),
        "fraction_equal_to_n_adequate": float((PH == D["n_adequate_in_top5"].to_numpy()[h]).mean()),
        "max_pareto_set_size": int(PH.max()),
    }
    pd.DataFrame([PARE]).to_csv(PC.T8 / "pareto_summary.csv", index=False)
    D["pareto_set_size"] = par_sizes
    D.to_csv(PC.T8 / "basin_parsimony_selection.csv", index=False)
    report["pareto"] = PARE

    # ================================================================== §9 fallbacks
    zero = ~has
    i_fa, ok_fa = pick_simplest(top5)                     # simplest in Top-5 (tie: seen KGE)
    def pick_best_adequacy_then_simple(mask, score):
        """max `score`, tie -> min n_parameters, tie -> max SEEN KGE (all rule-available)."""
        s_ = np.where(mask, score, -BIG)
        np_m = np.where(mask, G_npar, BIG)
        hi_ = np.where(mask, G_seen, -BIG)
        idx = np.lexsort((-hi_, np_m, -s_), axis=1)[:, 0]
        return idx, mask[rr[:, 0], idx]

    i_fb, ok_fb = pick_best_adequacy_then_simple(top5, SFS)   # Rule 3 as specified
    rowsF = []
    for nm_, idx_, ok_ in (("Fallback_A_simplest_in_top5", i_fa, ok_fa),
                           ("Fallback_B_best_seen_adequacy_in_top5", i_fb, ok_fb)):
        rowsF.append({"fallback": nm_, "n_basins": int(zero.sum()),
                      **regret_stats((oracle - kge_at(idx_))[zero])})
        rowsF[-1]["median_n_parameters_selected"] = float(
            np.median(G_npar[rr[:, 0], idx_][zero]))
        rowsF[-1]["fraction_complexity_reduced_vs_top5median"] = float(np.mean(
            G_npar[rr[:, 0], idx_][zero] < np.median(np.where(top5, G_npar, np.nan), axis=1)[zero]))
    FB = pd.DataFrame(rowsF)
    FB.to_csv(PC.T8 / "fallback_comparison.csv", index=False)
    report["fallback_comparison"] = FB.to_dict("records")

    # ================================================================== §10 full rules
    i_top1 = order[:, 0]
    rules = [
        ("Rule 0: Seen Top-1", i_top1, np.ones(nb, bool)),
        ("Rule 2: simplest in Seen Top-5", i_fa, np.ones(nb, bool)),
        ("Rule 3: best SEEN-adequacy in Seen Top-5 (tie -> simplest -> seen KGE)", i_fb, np.ones(nb, bool)),
    ]
    rowsU = []
    for label, idx_, valid in rules:
        rowsU.append({"rule": label, "uses_oob_truth_in_selection": False,
                      "regret_reference": "OOB oracle (all 36)",
                      "median_n_parameters_selected": float(np.median(G_npar[rr[:, 0], idx_][valid])),
                      "fallback_frequency": 0.0,
                      **regret_stats((oracle - kge_at(idx_))[valid])})
    rowsU.append({"rule": "Rule 3b: highest SEEN-KGE among adequate (no parsimony)",
                  "uses_oob_truth_in_selection": False,
                  "regret_reference": "OOB oracle (all 36)",
                  "median_n_parameters_selected": float(np.median(G_npar[rr[:, 0], i_seenkge][has])),
                  "fallback_frequency": float(zero.mean()),
                  **regret_stats((oracle - kge_at(i_seenkge))[has])})
    # Rule 1 = the Top-5 set itself (a set, not a single pick): report its best-in-set regret
    rowsU.append({"rule": "Rule 1: Seen Top-5 candidate set (set coverage)", "uses_oob_truth_in_selection": False,
                  "regret_reference": "OOB oracle (all 36)",
                  "median_n_parameters_selected": float(np.median(np.where(top5, G_npar, np.nan))),
                  "fallback_frequency": 0.0,
                  **regret_stats(D["regret_top5best_vs_oracle"].to_numpy())})
    # Rule 3c — same coverage and same fallback as Rule 4, but NO parsimony inside the
    # adequate set (it takes the highest SEEN-KGE adequate candidate).  Comparing Rule 4 with
    # Rule 3c isolates the pure parsimony cost on an identical basin set.
    idx3c = np.where(has, i_seenkge, i_fa)
    rowsU.append({"rule": "Rule 3c: [adequate ? best SEEN-KGE : Fallback A]  (no parsimony)",
                  "uses_oob_truth_in_selection": False,
                  "regret_reference": "OOB oracle (all 36)",
                  "median_n_parameters_selected": float(np.median(G_npar[rr[:, 0], idx3c])),
                  "fallback_frequency": float(zero.mean()),
                  **regret_stats((oracle - kge_at(idx3c)))})
    # Rule 4 — OOB-adequacy filter (the frozen stage-2 configuration), then simplest, then fallback
    idx4 = np.where(has, i_simple, i_fa)
    rowsU.append({"rule": "Rule 4: RELAXED_70(OOB) filter -> simplest -> Fallback A",
                  "uses_oob_truth_in_selection": False,
                  "regret_reference": "OOB oracle (all 36)",
                  "median_n_parameters_selected": float(np.median(G_npar[rr[:, 0], idx4])),
                  "fallback_frequency": float(zero.mean()),
                  **regret_stats((oracle - kge_at(idx4)))})
    # Rule 4d — deployable variant: the adequacy filter itself uses the SEEN flags
    adeq_s = top5 & A70s
    has_s = adeq_s.sum(axis=1) >= 1
    i_s4, _ = pick_simplest(adeq_s)
    i_fa_s, _ = pick_simplest(top5)
    idx4s = np.where(has_s, i_s4, i_fa_s)
    rowsU.append({"rule": "Rule 4d (deployable): RELAXED_70(SEEN) filter -> simplest -> Fallback A",
                  "uses_oob_truth_in_selection": False,
                  "regret_reference": "OOB oracle (all 36)",
                  "median_n_parameters_selected": float(np.median(G_npar[rr[:, 0], idx4s])),
                  "fallback_frequency": float((~has_s).mean()),
                  **regret_stats((oracle - kge_at(idx4s)))})
    rowsU.append({"rule": "Reference: OOB oracle (unreachable)", "uses_oob_truth_in_selection": True,
                  "regret_reference": "OOB oracle (all 36)",
                  "median_n_parameters_selected": float(np.median(np.where(np.isfinite(G_oob), G_npar, np.nan))),
                  "fallback_frequency": 0.0, **regret_stats(np.zeros(nb))})
    U = pd.DataFrame(rowsU)
    U.to_csv(PC.T8 / "parsimony_rules_comparison.csv", index=False)
    report["rules"] = U.to_dict("records")
    # paired cost of parsimony on an IDENTICAL basin set and fallback: Rule 3c - Rule 4
    paired = kge_at(idx3c) - kge_at(idx4)            # >= 0 when parsimony loses performance
    D["paired_parsimony_cost"] = paired
    report["paired_parsimony_cost"] = {
        "definition": "OOB KGE(Rule 3c) - OOB KGE(Rule 4): identical pool, coverage and fallback; "
                      "the only difference is 'highest SEEN-KGE adequate' vs 'simplest adequate'",
        "n": int(nb),
        "fraction_zero_cost": float((np.abs(paired) < 1e-12).mean()),
        "median_paired_cost": float(np.median(paired)),
        "mean_paired_cost": float(np.mean(paired)),
        "q75_paired_cost": float(np.percentile(paired, 75)),
        "q90_paired_cost": float(np.percentile(paired, 90)),
        "fraction_cost_gt_0p01": float((paired > 0.01).mean()),
        "fraction_cost_gt_0p02": float((paired > 0.02).mean()),
        "fraction_cost_gt_0p05": float((paired > 0.05).mean()),
        "median_n_parameters_saved": float(
            np.median(G_npar[rr[:, 0], idx3c] - G_npar[rr[:, 0], idx4])),
    }
    report["pure_parsimony_cost"] = {
        "definition": "Rule 4 minus Rule 3c: identical coverage (Top-5 + RELAXED_70) and identical "
                      "Fallback A; the ONLY difference is 'simplest adequate' vs "
                      "'highest SEEN-KGE adequate'.",
        "median_regret_rule4": float(np.median((oracle - kge_at(idx4)))),
        "median_regret_rule3c": float(np.median((oracle - kge_at(idx3c)))),
        "median_cost": float(np.median((oracle - kge_at(idx4))) - np.median((oracle - kge_at(idx3c)))),
        "mean_cost": float(np.mean((oracle - kge_at(idx4))) - np.mean((oracle - kge_at(idx3c)))),
        "frac_le_0.01_rule4": float(((oracle - kge_at(idx4)) <= 0.01).mean()),
        "frac_le_0.01_rule3c": float(((oracle - kge_at(idx3c)) <= 0.01).mean()),
    }

    # ================================================================== §11 sensitivity
    rowsS = []
    base_pick = idx4.copy()
    for k in (3, 5):
        for ad_name, ad_o, ad_s in (("RELAXED_60", A60o, A60s), ("RELAXED_70", A70o, A70s),
                                    ("RELAXED_80", A80o, A80s)):
            for metric, GP in (("n_parameters", G_npar), ("n_states", G_nst)):
                p = pool(k)
                a = p & ad_o
                hs = a.sum(axis=1) >= 1
                np_m = np.where(a, GP, BIG)
                hi_m = np.where(a, G_seen, -BIG)
                ii = np.lexsort((-hi_m, np_m), axis=1)[:, 0]
                ifa, _ = pick_simplest(p)
                ic = np.where(hs, ii, ifa)
                rowsS.append({
                    "candidate_pool": f"Top-{k}", "adequacy": ad_name, "complexity_metric": metric,
                    "coverage_fraction": float(hs.mean()),
                    "agreement_with_baseline": float((ic == base_pick).mean()),
                    "median_n_parameters_selected": float(np.median(G_npar[rr[:, 0], ic])),
                    **{f"regret_{kk}": vv for kk, vv in regret_stats((oracle - kge_at(ic))).items()
                       if kk in ("median", "mean", "q90")},
                    "frac_regret_le_0.01": float(((oracle - kge_at(ic)) <= 0.01).mean()),
                    "frac_regret_le_0.05": float(((oracle - kge_at(ic)) <= 0.05).mean()),
                })
    SEN = pd.DataFrame(rowsS)
    SEN.to_csv(PC.T8 / "parsimony_sensitivity.csv", index=False)
    report["sensitivity"] = {
        "table": SEN.to_dict("records"),
        "agreement_with_baseline_range": [float(SEN.agreement_with_baseline.min()),
                                          float(SEN.agreement_with_baseline.max())],
        "baseline": "Top-5 + RELAXED_70(OOB) + n_parameters",
    }

    # final write so every derived column (incl. the paired parsimony cost) is persisted
    D.to_csv(PC.T8 / "basin_parsimony_selection.csv", index=False)

    with open(PC.T8 / "parsimony.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    # ================================================================== console
    log("== §1 complexity inventory (COMPARABLE dimensions only) ==")
    log(f"   n_parameters: {inv.n_parameters.min()} .. {inv.n_parameters.max()} "
        f"(median {inv.n_parameters.median():.1f}, {inv.n_parameters.nunique()} distinct) | "
        f"n_states: {inv.n_states.min()} .. {inv.n_states.max()} "
        f"(median {inv.n_states.median():.1f}, {inv.n_states.nunique()} distinct)")
    log(f"   NOT_COMPARABLE: n_fluxes, n_process_modules "
        f"(NUMBER_INFO = MARRMoT model number, not a count)")
    log("\n== §3 complexity vs performance (n=36 models) ==")
    log(R3.round(4).to_string(index=False))
    log(f"\n== §4 selection coverage: Top-5 & RELAXED_70(OOB) ==")
    log(f"   basins with >=1 adequate candidate: {h.sum()}/{nb} = {h.mean():.3f}")
    log(f"   mean adequate per basin {n_adeq.mean():.2f}, median {np.median(n_adeq):.0f}")
    log("\n== §5 regret ==")
    log(REG.round(4).to_string(index=False))
    log("\n== §6/§7 complexity saved ==")
    for kk, vv in S6.items():
        log(f"   {kk:48s} {vv}")
    log("\n== §8 Pareto ==")
    for kk, vv in PARE.items():
        log(f"   {kk:48s} {vv}")
    log("\n== §9 fallback (zero-adequate basins) ==")
    log(FB.round(4).to_string(index=False))
    log("\n== §10 full rule comparison ==")
    log(U[["rule", "median_n_parameters_selected", "fallback_frequency", "median", "q90",
           "frac_le_0.01", "frac_le_0.05"]].round(4).to_string(index=False))
    log("\n== §10b paired cost of parsimony (identical pool, coverage and fallback) ==")
    for kk, vv in report["paired_parsimony_cost"].items():
        log(f"   {kk:36s} {vv}")
    log("\n== §11 sensitivity (agreement with the Top-5/RELAXED_70/n_parameters baseline) ==")
    log(SEN[["candidate_pool", "adequacy", "complexity_metric", "coverage_fraction",
             "agreement_with_baseline", "regret_median", "frac_regret_le_0.01"]].round(4).to_string(index=False))
    log(f"\nwrote -> {PC.T8}")


if __name__ == "__main__":
    main()
