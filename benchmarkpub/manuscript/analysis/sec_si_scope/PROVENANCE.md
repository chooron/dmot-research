# Unit Provenance: sec_si_scope

## 1. Estimands Covered
- **Cross-metric candidate set overlap**: Top-3 overlap of logKGE (0.2147 dPL, 0.2310 IC) and FHV (0.4325 dPL, 0.2994 IC) with standard KGE (annotated EVALUATION_ONLY_NO_TARGET_MATCHED_RETRAINING)
- **Supplementary metric recommendation**: G_set under logKGE (+0.1210) and FHV (-0.0411)
- **Equivalence label definitions & bootstrap protocol parameters**: 15 water-year blocks (1996..2010), 1000 draws, seed 20260925, q05 rule across IC, seen-dPL, and OOB-dPL

## 2. Approach Declarations & Parity Verifications

### Script 01: `scripts/si_01_cross_metric_overlap.py`
- **Approach**: (b) Loaded frozen verified artifacts from chapter4_robustness_supplement_20260925 and chapter4_final_closure_20260925
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmark/analysis/chapter4_robustness_supplement_20260925/tables/rb2_cross_metric_overlap.csv` (bytes: 187, mtime: 2026-09-24 08:20:52 UTC, sha256: `8cf3d2f763847ec2658be5b4b01a6426e4e28143111252e737d1bbd145f90f19`)
  * `/home/jingxin/code/dmg-research/project/benchmark/analysis/chapter4_final_closure_20260925/tables/s2_supp_metrics_summary.json` (bytes: 1743, mtime: 2026-09-24 10:17:03 UTC, sha256: `7c3f0346b371d7ea615e0d982b4750c95f9b7d990927e2e0e6a9d93b43e3e70a`)
- dPL logKGE Top-3 overlap with KGE: 0.2147
- dPL FHV Top-3 overlap with KGE: 0.4325
- IC logKGE Top-3 overlap with KGE: 0.2310
- IC FHV Top-3 overlap with KGE: 0.2994

### Script 02: `scripts/si_02_equivalence_protocol_params.py`
- **Approach**: (a) Compiled from s2_summary.json and frozen protocol specifications
- Upstream artifacts verified & loaded:
  * `/home/jingxin/code/dmg-research/project/benchmark/analysis/chapter4_equivalence_recommendability_20260925/tables/s2_summary.json` (bytes: 2464, mtime: 2026-09-23 13:24:05 UTC, sha256: `26527c188f8102db5c2f297b9fe9a1a402f852f33d3da2b136965a6ed2047f13`)
- Conditions documented: 3
- Replicate draws: 1000
- Random seed: 20260925

## 3. Output Tables Generated
- `tables/sec_si_cross_metric_overlap.csv` (362 bytes)
- `tables/sec_si_equivalence_protocol_params.csv` (1079 bytes)
- `tables/sec_si_supplementary_metric_recommendation.csv` (284 bytes)
