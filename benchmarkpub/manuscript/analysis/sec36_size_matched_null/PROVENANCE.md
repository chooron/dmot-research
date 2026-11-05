# Unit Provenance: sec36_size_matched_null  (POST-FREEZE analysis family, H1-H2)

## 0. POST-FREEZE registration
- Created **after the freeze** as a new analysis family. It reads only frozen artifacts (the frozen
  equivalence-set table and the external type assignment) and adds new tables; no frozen value, decision,
  protocol or existing numeric table is modified.
- Manifest rows carry the status **`POST-FREEZE`**; in the exported lists they are citable but always carry
  `freeze_phase = post_freeze` (the marker is never erased or merged).
- The external classification is used strictly as a grouping factor; no hydrological meaning is attached to it.
- No equivalence judgement is recomputed: `t1_equivalence_sets.csv` is used as frozen.

## 1. H1 — size-matched null distribution
- **Question form**: is a type's minimal cover size lower than the cover size obtained from a random draw of the
  **same number of basins** from the same pool?
- **Sampling pool**: the **515 type-matched basins** used by G1-G4 (same source as the types);
  the 16 unmatched basins are excluded (they carry no type).
- **Main tier**: 95% cover, three conditions x 10 types, **R=1000** draws without replacement per cell.
- **Secondary tier**: 90% and 100% cover, same structure, **R=200**.
- **Solver**: `scipy.optimize.milp (HiGHS), direct minimal-cover formulation, single MILP per draw`; every draw is solved exactly (no approximation, no reduced R).
- **Random seeds**: main tier `20261006`, secondary tier `20261007`
  (one generator per tier, cells visited in the order condition = IC, seen-dPL, OOB-dPL x type ascending).
- **Direction definition**: one-sided empirical percentile p_null_le_observed = (#null draws with cover <= observed + 1) / (R + 1); small values mean the type's observed cover is lower than the size-matched random draws
- **Non-optimal solves**: 0 (none were replaced by approximate values).
- **Solve counts**: 42090 exact MILP solves (42000 null draws + 90 observed solves); wall-clock time is reported in the run log only and is not persisted, keeping every output deterministic.

### Main-tier results (95%, R=1000)
- **IC**: t0 obs=3 null_med=4.0 [P5=3.0, P95=4.0] p_le=0.112 (n=200) | t1 obs=4 null_med=4.0 [P5=3.0, P95=5.0] p_le=0.907 (n=79) | t2 obs=2 null_med=2.0 [P5=1.0, P95=3.0] p_le=0.888 (n=6) | t3 obs=2 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.081 (n=42) | t4 obs=3 null_med=2.0 [P5=1.0, P95=3.0] p_le=0.987 (n=7) | t5 obs=3 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.735 (n=17) | t6 obs=2 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.391 (n=21) | t7 obs=5 null_med=4.0 [P5=3.0, P95=5.0] p_le=0.999 (n=53) | t8 obs=3 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.658 (n=44) | t9 obs=3 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.640 (n=46)
- **seen-dPL**: t0 obs=4 null_med=5.0 [P5=4.0, P95=6.0] p_le=0.162 (n=200) | t1 obs=5 null_med=5.0 [P5=3.0, P95=6.0] p_le=0.820 (n=79) | t2 obs=1 null_med=2.0 [P5=1.0, P95=3.0] p_le=0.157 (n=6) | t3 obs=3 null_med=4.0 [P5=2.0, P95=5.0] p_le=0.441 (n=42) | t4 obs=2 null_med=2.0 [P5=1.0, P95=4.0] p_le=0.634 (n=7) | t5 obs=2 null_med=3.0 [P5=2.0, P95=5.0] p_le=0.129 (n=17) | t6 obs=2 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.269 (n=21) | t7 obs=6 null_med=4.0 [P5=3.0, P95=6.0] p_le=0.993 (n=53) | t8 obs=5 null_med=4.0 [P5=3.0, P95=5.0] p_le=0.964 (n=44) | t9 obs=3 null_med=4.0 [P5=3.0, P95=5.0] p_le=0.356 (n=46)
- **OOB-dPL**: t0 obs=8 null_med=10.0 [P5=9.0, P95=11.0] p_le=0.017 (n=200) | t1 obs=8 null_med=8.0 [P5=7.0, P95=10.0] p_le=0.520 (n=79) | t2 obs=2 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.339 (n=6) | t3 obs=4 null_med=6.0 [P5=5.0, P95=8.0] p_le=0.023 (n=42) | t4 obs=2 null_med=3.0 [P5=2.0, P95=4.0] p_le=0.223 (n=7) | t5 obs=4 null_med=5.0 [P5=4.0, P95=7.0] p_le=0.269 (n=17) | t6 obs=4 null_med=5.0 [P5=3.0, P95=7.0] p_le=0.397 (n=21) | t7 obs=8 null_med=7.0 [P5=6.0, P95=9.0] p_le=0.840 (n=53) | t8 obs=8 null_med=7.0 [P5=5.0, P95=9.0] p_le=0.930 (n=44) | t9 obs=7 null_med=7.0 [P5=5.0, P95=9.0] p_le=0.713 (n=46)

### Secondary-tier results (90% and 100%, R=200)
- **IC 90%**: t0 obs=3 null_med=3.0 p_le=0.985 | t1 obs=3 null_med=3.0 p_le=0.950 | t2 obs=2 null_med=2.0 p_le=0.910 | t3 obs=2 null_med=3.0 p_le=0.408 | t4 obs=3 null_med=2.0 p_le=0.985 | t5 obs=2 null_med=2.0 p_le=0.557 | t6 obs=1 null_med=2.0 p_le=0.020 | t7 obs=4 null_med=3.0 p_le=1.000 | t8 obs=3 null_med=3.0 p_le=0.955 | t9 obs=3 null_med=3.0 p_le=0.965
- **seen-dPL 90%**: t0 obs=2 null_med=3.0 p_le=0.045 | t1 obs=3 null_med=3.0 p_le=0.677 | t2 obs=1 null_med=2.0 p_le=0.204 | t3 obs=2 null_med=3.0 p_le=0.403 | t4 obs=2 null_med=2.0 p_le=0.682 | t5 obs=2 null_med=3.0 p_le=0.398 | t6 obs=2 null_med=2.0 p_le=0.587 | t7 obs=5 null_med=3.0 p_le=0.995 | t8 obs=4 null_med=3.0 p_le=0.985 | t9 obs=2 null_med=3.0 p_le=0.264
- **OOB-dPL 90%**: t0 obs=6 null_med=7.0 p_le=0.060 | t1 obs=6 null_med=6.0 p_le=0.562 | t2 obs=2 null_med=3.0 p_le=0.348 | t3 obs=3 null_med=5.0 p_le=0.020 | t4 obs=2 null_med=3.0 p_le=0.194 | t5 obs=3 null_med=4.0 p_le=0.249 | t6 obs=4 null_med=4.0 p_le=0.706 | t7 obs=6 null_med=6.0 p_le=0.846 | t8 obs=7 null_med=5.0 p_le=0.980 | t9 obs=6 null_med=5.0 p_le=0.866
- **IC 100%**: t0 obs=5 null_med=7.0 p_le=0.010 | t1 obs=6 null_med=5.0 p_le=0.930 | t2 obs=2 null_med=2.0 p_le=0.861 | t3 obs=2 null_med=4.0 p_le=0.005 | t4 obs=3 null_med=2.0 p_le=0.990 | t5 obs=3 null_med=3.0 p_le=0.796 | t6 obs=2 null_med=3.0 p_le=0.114 | t7 obs=6 null_med=5.0 p_le=0.980 | t8 obs=4 null_med=4.0 p_le=0.597 | t9 obs=4 null_med=4.0 p_le=0.552
- **seen-dPL 100%**: t0 obs=8 null_med=10.0 p_le=0.080 | t1 obs=7 null_med=7.0 p_le=0.731 | t2 obs=1 null_med=2.0 p_le=0.124 | t3 obs=4 null_med=5.0 p_le=0.244 | t4 obs=2 null_med=2.0 p_le=0.706 | t5 obs=2 null_med=4.0 p_le=0.114 | t6 obs=3 null_med=4.0 p_le=0.423 | t7 obs=7 null_med=5.0 p_le=0.950 | t8 obs=6 null_med=5.0 p_le=0.846 | t9 obs=4 null_med=5.0 p_le=0.169
- **OOB-dPL 100%**: t0 obs=13 null_med=16.0 p_le=0.025 | t1 obs=11 null_med=11.0 p_le=0.627 | t2 obs=2 null_med=3.0 p_le=0.294 | t3 obs=5 null_med=8.0 p_le=0.010 | t4 obs=2 null_med=3.0 p_le=0.204 | t5 obs=4 null_med=5.0 p_le=0.303 | t6 obs=5 null_med=6.0 p_le=0.418 | t7 obs=10 null_med=9.0 p_le=0.831 | t8 obs=10 null_med=9.0 p_le=0.915 | t9 obs=8 null_med=9.0 p_le=0.473

## 2. H2 — absolute per-type cover counts
- Re-derived from the frozen equivalence sets with the same exact MILP: 90 cells
  (3 conditions x 10 types x 90/95/100%), each with the type's N and the budget-minus-one certificate.
- Optimality: `proves_k_minus_1_insufficient` true in 90/90 cells;
  solver status 0 in 90/90.
- Absolute agreement with the G3 table: 90 rows compared, max abs diff 0.000e+00 on the cover counts and 0.000e+00 on the certificates, 0 rows with any difference.

### Absolute minimal cover counts (type -> models / type N)
- **IC 90%**: t0:3/200 | t1:3/79 | t2:2/6 | t3:2/42 | t4:3/7 | t5:2/17 | t6:1/21 | t7:4/53 | t8:3/44 | t9:3/46
- **IC 95%**: t0:3/200 | t1:4/79 | t2:2/6 | t3:2/42 | t4:3/7 | t5:3/17 | t6:2/21 | t7:5/53 | t8:3/44 | t9:3/46
- **IC 100%**: t0:5/200 | t1:6/79 | t2:2/6 | t3:2/42 | t4:3/7 | t5:3/17 | t6:2/21 | t7:6/53 | t8:4/44 | t9:4/46
- **seen-dPL 90%**: t0:2/200 | t1:3/79 | t2:1/6 | t3:2/42 | t4:2/7 | t5:2/17 | t6:2/21 | t7:5/53 | t8:4/44 | t9:2/46
- **seen-dPL 95%**: t0:4/200 | t1:5/79 | t2:1/6 | t3:3/42 | t4:2/7 | t5:2/17 | t6:2/21 | t7:6/53 | t8:5/44 | t9:3/46
- **seen-dPL 100%**: t0:8/200 | t1:7/79 | t2:1/6 | t3:4/42 | t4:2/7 | t5:2/17 | t6:3/21 | t7:7/53 | t8:6/44 | t9:4/46
- **OOB-dPL 90%**: t0:6/200 | t1:6/79 | t2:2/6 | t3:3/42 | t4:2/7 | t5:3/17 | t6:4/21 | t7:6/53 | t8:7/44 | t9:6/46
- **OOB-dPL 95%**: t0:8/200 | t1:8/79 | t2:2/6 | t3:4/42 | t4:2/7 | t5:4/17 | t6:4/21 | t7:8/53 | t8:8/44 | t9:7/46
- **OOB-dPL 100%**: t0:13/200 | t1:11/79 | t2:2/6 | t3:5/42 | t4:2/7 | t5:4/17 | t6:5/21 | t7:10/53 | t8:10/44 | t9:8/46

## 3. H3 — manifest list handling (see `_manifest/PROVENANCE.md`)
- The sec35 POST-FREEZE rows were moved from the excluded list into the citable list; the H1/H2 rows added
  here are likewise citable. Both lists gained a `freeze_phase` column (`pre_freeze` / `post_freeze`).
- The remaining exclusion reasons (`LOADED`, `SUPERSEDED`, `SENSITIVITY`) and their wording are unchanged.

## 4. Output tables
- `tables/sec36_h1_run_meta.csv` (700 bytes)
- `tables/sec36_h1_size_matched_null_distribution.csv` (1509711 bytes)
- `tables/sec36_h1_size_matched_null_summary.csv` (34424 bytes)
- `tables/sec36_h2_type_cover_absolute.csv` (4206 bytes)
- `tables/sec36_h2_vs_g3_comparison.csv` (109 bytes)

## 5. Compliance notes
- No hydrological model, CMA-ES run, dPL training or recommender fit was re-run; no equivalence rule was recomputed.
- No type was merged, dropped or re-clustered; small types are reported with their own N.
- All conditions, all types and all three coverage levels are reported, without selecting by significance.
