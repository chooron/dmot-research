# Formal Release Archives & Artifacts Ledger

This document registers the verified release packages for the dMoT research ecosystem, their code lineages, checksums, and storage locations.

---

## 1. Formal Publication Releases

| Package Identifier | Target Publication / Role | Version & Date | Formal Archive Location | Size | SHA-256 |
|---|---|---|---|---|---|
| **`releases/benchmarkpub`** | *How many conceptual models do we need for ungauged basins? Potential coverage versus attained performance* (GMD Technical Note) | v1.0.0 (2026-10-10) | `dmg-research/releases/benchmarkpub.zip` (Zenodo upload package) | 422,499,326 B (402.9 MB) | `5e631a506c7f4904e725cd1f9235b57111c3b3bc55202e35bbcd4f6e6ff6681b` |
| **`releases/benchmark`** | *Catchment-parameter organization across basin-wise calibration and differentiable parameter learning in conceptual hydrological models* | Production (2026-09-18) | `dmg-research/releases/benchmark/training_results.zip` | 234,590,278 B (223.7 MB) | `7ad6a5f70bb0080bc650e640adff04ba3a0279c13b307ec33571d7c35f991f86` |
| **`releases/benchmarkdev`** | *dMoTpy Software Release Candidate & GMD Verification* | 1.0.0rc1 (2026-11-03) | `dmg-research/releases/benchmarkdev/` | 265,379,927 B (253.1 MB) | `A_dmotpy: 978912e75e92976e107567ecb575e94be7eb294e7514a60f9e160f64be87be09` |

---

## 2. Release File Inventory & Integrity

### A. `benchmarkpub` v1.0.0 Package Composition (622 Tracked Files)
- **Checkpoints (`checkpoints/`)**: 219 files (180 OOB-dPL fold parameterizers + 36 seen-dPL parameterizers).
- **Code (`code/`)**: 163 files (complete standalone replication code for Section 3 analyses and figures).
- **Data (`data/`)**: 137 files (basin attributes, performance metrics including repaired 6-model OOB evaluations).
- **Figures (`figures/`)**: 60 files (Figures 1–7 and A1–A2 in 300 dpi PNG + PDF + minimal input CSVs).
- **Supplementary Tables (`supplementary_tables/`)**: 21 files (Tables S1–S5).
- **Protocol & Metadata**: `README.md`, `RELEASE_REPORT.md`, `CITATION.cff`, `LICENSE`, `.zenodo.json`, `MANIFEST.sha256`.

### B. Correspondence with Data Freeze & 6 Repaired Models
- **Freeze Record (`freeze_record_20261005.json`)**: 311 / 311 files mapped. 292 files are byte-identical to the initial Oct 5 freeze; 19 files contain the documented post-freeze bootstrap re-runs from Oct 6–7.
- **6 Repaired Models**: The OOB predictions and parameter matrices in `releases/benchmarkpub/data/performance/` and `parameters/` incorporate the verified Oct 28 repair runs for `gsfb`, `penman`, `plateau`, `smar`, `tcm`, `vic`.

---

## 3. Storage & Replication Strategy

- **Lightweight Release Metadata & Tables**: Directly tracked in Git under `releases/` and `results/`.
- **Large Standalone Release Archives**:
  - Maintained in `dmg-research/releases/` and staged for cold storage replication to external storage via `results/F_DRIVE_ARCHIVE_MANIFEST.json`.
