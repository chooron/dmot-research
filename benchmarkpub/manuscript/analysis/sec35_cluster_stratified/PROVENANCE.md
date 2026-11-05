# Unit Provenance: sec35_cluster_stratified  (POST-FREEZE analysis family)

## 0. POST-FREEZE registration
- This unit was created **after the freeze** as a new analysis family (G1-G4). It reads only frozen artifacts
  and adds new tables; no frozen value, decision, protocol or existing numeric table is modified.
- Manifest rows for this family carry the status **`POST-FREEZE`** so they are not mixed with the pre-freeze
  `RECOMPUTED` / `RE-VERIFIED` rows; they are also excluded from the freeze-ready citable list.
- **The external classification is used strictly as a grouping factor.** No hydrological meaning, naming or
  interpretation is attached to any type label; no type is merged, re-clustered, dropped or re-weighted.

## 1. Estimands covered
- **G1**: PUB-penalty (`penalty_median_pub`, median over the 36 models of `delta_kge_pub_median4`) across the
  10 external types; matched 515 basins, unmatched 16 reported and excluded;
  Kruskal-Wallis H=29.6911, df=9, p=4.950897e-04, rank epsilon-squared (epsilon^2)=0.0410;
  all 45 pairwise Dunn comparisons with Holm correction reported (1 with Holm p < 0.05).
- **G1b** (descriptive only, no test, no correction): per-type Spearman rho between `arid`
  (global BH-FDR q = 0.075367, the smallest-q attribute of the A4 recomputed screening)
  and the same penalty; N per type reported.
- **G2**: type distribution of the 26 worst-penalty basins (`int(0.05 x 531) = 26`, unchanged rule):
  25 of 26 are type-matched (1 unmatched);
  2 x 10 contingency table with observed and expected counts; exact test p=0.000480
  (Monte-Carlo Fisher-type exact test, 100000 draws, seed 20261005, se=6.90e-05);
  asymptotic chi-square p=0.000101 reported alongside.
- **G3**: equivalence-set size per condition x type with Kruskal-Wallis; exact minimal cover per condition x type
  for 90 / 95 / 100% of that type's basins with optimality certificates (max basins coverable at budget K*-1);
  95%-cover counts compared with the global values (IC 4 / seen-dPL 6 / OOB-dPL 12).
- **G4**: parsimony cost (ranking member minus most-parsimonious member) per condition x type with Kruskal-Wallis,
  using only the ID-joined A3 recomputation.

## 2. Approach declarations
- **G1** `scripts/sec35_01_penalty_by_type_g1.py`: response recomputed by basin-ID join from
  `pub_same_pipeline_model_basin.csv`; grouping from `data/gauge_cluster.csv` (ID-joined); Kruskal-Wallis via
  `scipy.stats.kruskal`; effect size **rank epsilon-squared**, defined as epsilon^2 = (H - k + 1) / (n - k)
  (H = Kruskal-Wallis statistic, k = number of types, n = number of basins); pairwise **Dunn z-test** on average
  ranks with tie correction, two-sided, **Holm** correction over the full family of 45 pairs.
  The epsilon-squared estimator can be negative when H < k - 1; negative values are reported as computed.
- **G1b**: Spearman rho computed within each type on the same ID-joined arrays; **no p-value, no correction**.
- **G2** `scripts/sec35_02_worst_basins_by_type_g2.py`: basin sums of `kge_seendpl_canonical - kge_heldout`
  (negative values retained), descending, top 26; contingency table restricted to matched basins;
  exact test implemented as a **Monte-Carlo enumeration of the multivariate hypergeometric null**
  (Fisher probability-ordering statistic, 100,000 draws, seed 20261005) because the 2 x 10 table has no
  closed-form exact test in the available stack; the asymptotic chi-square result is reported next to it.
- **G3** `scripts/sec35_03_equivalence_library_by_type_g3.py`: equivalence sizes read from
  `t1_equivalence_sets.csv` by basin-ID join; minimal covers solved exactly with `scipy.optimize.milp` (HiGHS),
  status 0 = proven optimal, plus the budget-minus-one certificate; targets are fractions of each type's own basin count
  (`ceil(target x n_type)`), so for small types the 95% target can equal the type's full basin count (reported as-is).
- **G4** `scripts/sec35_04_parsimony_by_type_g4.py`: response read from
  `sec34_a3_parsimony_penalty_idjoin_per_basin.csv` (ID-joined A3 recomputation; the SUPERSEDED table is never used);
  the subtraction order is asserted to be `ranking_member_minus_parsimonious_member`.
- All basin-level lookups and joins go through `_shared/basin_accessor.py` (`canonical_basin_ids`, `basin_position_map`,
  `pad_basin_id`, `align_series_by_id`).

## 3. Key results (as written to the tables)
- G1 per-type penalty median range: 0.022442 - 0.096943; type sizes {0: 200, 1: 79, 2: 6, 3: 42, 4: 7, 5: 17, 6: 21, 7: 53, 8: 44, 9: 46}
- G1 unmatched basins (16): 02196000|04015330|04213000|06221400|06224000|07261000|10336645|10336660|11143000|11482500|12040500|12041200|12082500|12092000|12175500|14137000
- G3 equivalence-size Kruskal-Wallis:
  * IC: H=51.3669, df=9, p=5.953702e-08, epsilon^2=0.0839
  * seen-dPL: H=31.4166, df=9, p=2.510463e-04, epsilon^2=0.0444
  * OOB-dPL: H=36.1603, df=9, p=3.713150e-05, epsilon^2=0.0538
- G3 95% minimal cover per type (type -> models; global value in brackets):
  * IC (global all-531 value 4): 0:3, 1:4, 2:2, 3:2, 4:3, 5:3, 6:2, 7:5, 8:3, 9:3
  * seen-dPL (global all-531 value 6): 0:4, 1:5, 2:1, 3:3, 4:2, 5:2, 6:2, 7:6, 8:5, 9:3
  * OOB-dPL (global all-531 value 12): 0:8, 1:8, 2:2, 3:4, 4:2, 5:4, 6:4, 7:8, 8:8, 9:7
- G3 optimality certificates: 90/90 type x condition x target cells prove budget K*-1 insufficient (solver status 0 in 90/90).
- G4 parsimony-cost Kruskal-Wallis:
  * IC: H=17.4608, df=9, p=4.196985e-02, epsilon^2=0.0168
  * seen-dPL: H=6.0797, df=9, p=7.319118e-01, epsilon^2=-0.0058
  * OOB-dPL: H=8.6301, df=9, p=4.720923e-01, epsilon^2=-0.0007

## 4. Output tables
- `tables/sec35_g1_kruskal_wallis.csv` (326 bytes)
- `tables/sec35_g1_matched_summary.csv` (393 bytes)
- `tables/sec35_g1_pairwise_dunn_holm.csv` (3897 bytes)
- `tables/sec35_g1_penalty_by_type.csv` (754 bytes)
- `tables/sec35_g1b_attribute_penalty_spearman_by_type.csv` (1049 bytes)
- `tables/sec35_g2_fisher_exact.csv` (559 bytes)
- `tables/sec35_g2_worst26_contingency.csv` (363 bytes)
- `tables/sec35_g3_equivalence_size_by_type.csv` (1634 bytes)
- `tables/sec35_g3_equivalence_size_kruskal.csv` (770 bytes)
- `tables/sec35_g3_min_cover_95_vs_global.csv` (965 bytes)
- `tables/sec35_g3_min_cover_by_type.csv` (4193 bytes)
- `tables/sec35_g4_parsimony_by_type.csv` (2071 bytes)
- `tables/sec35_g4_parsimony_kruskal.csv` (1039 bytes)

## 5. Compliance notes
- No hydrological model, CMA-ES run, dPL training or recommender fit was re-run.
- No frozen value (fold split, snow filter, equivalence threshold rule, bootstrap protocol, seed list) was changed.
- SUPERSEDED tables were not used: G4 reads only the ID-joined A3 table; G1/G1b use the A4 recomputed screening for the attribute choice.
- All 10 types, all 45 pairwise comparisons and all three conditions are reported; nothing is selected by significance.
- No 10 x 15 stratified significance screen was performed (G1b is descriptive only).
