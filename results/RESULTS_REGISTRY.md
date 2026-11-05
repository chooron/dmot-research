# Scientific Results Registry & Provenance Ledger

This registry documents the formal training, calibration, and out-of-basin (OOB) evaluation results managed in `/home/jingxin/code/dmot-research`.

---

## 1. Summary of Scientific Results

| Evaluation Regime | Scope | Run ID / Source | Checkpoints / Storage | Metrics Tables | Status |
|---|---|---|---|---|:---:|
| **IC (Independent Calibration)** | 36 models × 531 basins | `ic_dpl_aligned_full300_20260819` | Linked from `dmg-research/releases/benchmark/training_results/ic/` | `results/ic/metrics/` | **PASS (Fully Managed)** |
| **dPL (Differentiable Parameter Learning)** | 36 models × 531 basins (seen) | `dpl_canonical_v2_20260831` | Linked from `dmg-research/releases/benchmark/training_results/dpl/` | `results/dpl/metrics/` | **PASS (Fully Managed)** |
| **OOB (Spatial Cross-Validation)** | 30 models (baseline) + 6 models (repaired) | `oob36_20260930` + `oob_six_models_v1_repair_20261028` | Linked from `dmg-research/project/benchmarkpub/data/` | `results/oob/` (`seen_ic_oob36_v1_repaired.csv`) | **PASS (Fully Managed)** |

---

## 2. The 6 Updated Models Audit

In October 2026, six models underwent verified boundary and numerical repairs:

1. **`plateau`**: Integrated water availability constraint in saturation excess (`available_s1 = torch.clamp(S1_tmp2 - nearzero, min=0.0); flux_r = torch.minimum(flux_r_potential, available_s1)`).
2. **`gsfb`**: Lower-zone actual evapotranspiration (ET) boundary threshold protection.
3. **`penman`**: S1/S2 lower-zone ET deficit gate smoothing.
4. **`smar`**: Soil moisture layer ET exhaustion gate smoothing.
5. **`tcm`**: Mean precipitation scaling and ET deficit gate boundary protection.
6. **`vic`**: Calendar day-of-year (DOY) interception and ARNO curve continuity.

### Source Hash & Training Status Alignment

| Model | Source SHA-256 in `dmot-research` | Frozen Matrix Match | IC Source | dPL Source | OOB Repair Status (2026-10-28) |
|---|---|:---:|---|---|---|
| `gsfb` | `80fba6066d18c1b1934d4f82bc1a6b80d91aaec303859498814b0fd5edf5f900` | Yes | 10 starts / 300 gen | Canonical v2 | 5/5 folds DONE (`runs/gsfb/fold_0..4/best.pt`) |
| `penman` | `59f44d7e7f373b28d57dd3d0734997cf0e4abae6f197e00b073b7f9d4c18b61b` | Yes | 10 starts / 300 gen | Canonical v2 | 5/5 folds DONE (`runs/penman/fold_0..4/best.pt`) |
| `plateau` | `6311c225db08d55ef8e38ce034be6647a0dd63383b51436f72c0016fc11f0db5` | Yes | 10 starts / 300 gen | Canonical v2 | 5/5 folds DONE (`runs/plateau/fold_0..4/best.pt`) |
| `smar` | `40b6d36e00fa0c2e0a9c3a615362b8524118f53d7cc0138affcbcbc4b39fb9cf` | Yes | 10 starts / 300 gen | Canonical v2 | 5/5 folds DONE (`runs/smar/fold_0..4/best.pt`) |
| `tcm` | `508b691a3c8f570fa7db7364151920a97c1fc76cc4734f6b348c7d2ff4987511` | Yes | 10 starts / 300 gen | Canonical v2 | 5/5 folds DONE (`runs/tcm/fold_0..4/best.pt`) |
| `vic` | `9b90bea408a57d32e968ec1fb3d2bea6f5ac3b807535c050e70306d9dd3eb334` | Yes | 10 starts / 300 gen | Canonical v2 | 5/5 folds DONE (`runs/vic/fold_0..4/best.pt`) |

**Verification**:
The repaired OOB results for these 6 models are recorded in `results/oob/RUN_LOG_OOB_SIX_V1_REPAIR.csv` and unified with the 30 unchanged models in `results/oob/seen_ic_oob36_v1_repaired.csv` (19,116 rows).
When joined by `(model, gauge_id)`, the values in `results/oob/kge_oob_dpl_531x36_long.csv` match `seen_ic_oob36_v1_repaired.csv` with **maximum absolute difference = 0.00e+00**.

---

## 3. Storage and File Inventory

- **Local Git-managed Artifacts**:
  - `results/ic/`: 36 per-basin CSVs, summary tables, pairing gate, and execution manifests.
  - `results/dpl/`: `CANONICAL_V2_SUMMARY.csv`, audit replay comparison tables, and `canonical_manifest.yaml`.
  - `results/oob/`: `seen_ic_oob36_v1_repaired.csv`, `kge_all_conditions_long.csv`, `kge_oob_dpl_531x36_long.csv`, `kge_ic_531x36_long.csv`, `kge_seen_dpl_531x36_long.csv`, `parameters_*_531x36_long.csv`, and `basin_model_transfer_penalty.csv`.
- **Large Raw Simulation Data (Linked Read-Only)**:
  - `benchmarkpub/data/oob36_20260930`: 2.07 GB raw OOB daily simulated streamflow (`qsim/`) and observations (`obs/`).
  - `benchmarkpub/data/oob_six_models_v1_repair_20261028`: 30 fold checkpoint files (`runs/`) for the 6 repaired models.
