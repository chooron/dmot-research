#!/usr/bin/env python3
"""verify_reproducibility.py — Re-run all scripts and verify bit-for-bit output reproducibility."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "dmotpy").exists()), Path("/home/jingxin/code/dmg-research"))
BASE = ROOT / "project" / "benchmarkpub" / "manuscript" / "analysis"
PYTHON = sys.executable

SCRIPTS = [
    # sec31
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_01_delta_kge.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_02_variance_decomposition.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_03_rank_preservation.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_04_equivalence_and_coverage.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_06_fold_composition.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_07_jackknife_equivalence_e1.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_08_multiyear_block_bootstrap_c1.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_09_uncertainty_bandwidth_comparison_c2.py",
    BASE / "sec31_transfer_and_library" / "scripts" / "sec31_05_provenance.py",

    # sec32
    BASE / "sec32_library_members" / "scripts" / "sec32_01_library_and_covers.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_02_coverage_ladder_regret.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_03_model_structural_attributes.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_05_process_crosstabs.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_06_bootstrap_cover_ci_e2.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_07_optimal_solutions_enumeration_e3.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_08_singleton_and_bottleneck_e4.py",
    BASE / "sec32_library_members" / "scripts" / "sec32_04_provenance.py",

    # sec33
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_01_usage_rules_t2.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_02_rule_c_vs_b_t3_1.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_03_fixed_ensemble_scan_t3_2.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_04_gset_and_regret.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_05_cluster_typology.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_07_diversity_sensitivity_v5.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_08_recommender_k2_k5_e8.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_09_rules_ladder_c3.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_10_rule_c_k2_k5_per_basin_f1.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_11_constrained_fixed_ensemble_f2.py",
    BASE / "sec33_usage_and_conditioning" / "scripts" / "sec33_06_provenance.py",

    # sec34
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_01_complexity_vs_instability.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_02_multiseed_dispersion.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_03_penalty_concentration.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_04_attribute_screening.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_06_parsimony_penalty_e5.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_07_partial_correlation_e6.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_08_displacement_correlation_recompute_e7.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_09_equivalence_dispersion_c4.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_10_equivalence_bandwidth_diagnostic_k1_k5.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_11_audit_basin_alignment_a1_a2.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_12_basin_audit_supplement.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_13_attribute_screening_idjoin_a4.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_14_bottleneck_singleton_idjoin.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_15_a3_recompute_parsimony_dispersion_idjoin.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_16_a5_remaining_idjoin.py",
    BASE / "sec34_determinacy_and_bounds" / "scripts" / "sec34_05_provenance.py",

    # sec35 (POST-FREEZE)
    BASE / "sec35_cluster_stratified" / "scripts" / "sec35_01_penalty_by_type_g1.py",
    BASE / "sec35_cluster_stratified" / "scripts" / "sec35_02_worst_basins_by_type_g2.py",
    BASE / "sec35_cluster_stratified" / "scripts" / "sec35_03_equivalence_library_by_type_g3.py",
    BASE / "sec35_cluster_stratified" / "scripts" / "sec35_04_parsimony_by_type_g4.py",
    BASE / "sec35_cluster_stratified" / "scripts" / "sec35_05_provenance.py",

    # sec36 (POST-FREEZE, H-family)
    BASE / "sec36_size_matched_null" / "scripts" / "sec36_01_size_matched_null_h1.py",
    BASE / "sec36_size_matched_null" / "scripts" / "sec36_02_type_cover_absolute_h2.py",
    BASE / "sec36_size_matched_null" / "scripts" / "sec36_03_provenance.py",

    # sec_si
    BASE / "sec_si_scope" / "scripts" / "si_01_cross_metric_overlap.py",
    BASE / "sec_si_scope" / "scripts" / "si_02_equivalence_protocol_params.py",
    BASE / "sec_si_scope" / "scripts" / "si_03_provenance.py",

    # _verification
    BASE / "_verification" / "scripts" / "v1_recompute_abstract_numbers.py",
    BASE / "_verification" / "scripts" / "v1_provenance.py",

    # manifest
    BASE / "_manifest" / "scripts" / "compile_master_manifest.py",
    BASE / "_manifest" / "scripts" / "export_citable_numbers_f3.py",
    BASE / "_manifest" / "scripts" / "compile_manifest_provenance.py",
]


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def snapshot_files() -> dict[Path, str]:
    all_f = sorted(BASE.rglob("*"))
    return {f: hash_file(f) for f in all_f if f.is_file() and not f.name.endswith(".py") and not f.name.endswith(".pyc") and "__pycache__" not in str(f)}


def main() -> None:
    print(f"Step 1: Snapshotting {len(snapshot_files())} files before re-run...")
    before = snapshot_files()

    print(f"Step 2: Re-running all {len(SCRIPTS)} scripts in order...")
    for s in SCRIPTS:
        cmd = [PYTHON, str(s)]
        res = subprocess.run(cmd, cwd=s.parent, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED on script {s.name}:\n{res.stderr}")
            sys.exit(1)
        print(f"  [OK] {s.parent.parent.name}/{s.name}")

    print(f"Step 3: Snapshotting files after re-run...")
    after = snapshot_files()

    print("Step 4: Comparing before vs after...")
    mismatches = []
    max_numeric_diff = 0.0

    for f, h_before in before.items():
        h_after = after.get(f)
        if h_before != h_after:
            # Check numerical differences if CSV
            diff_val = 0.0
            if f.suffix == ".csv":
                try:
                    df1 = pd.read_csv(f)
                    df2 = pd.read_csv(f)
                    num_cols = df1.select_dtypes(include=[np.number]).columns
                    diff_val = float(np.nanmax(np.abs(df1[num_cols].to_numpy() - df2[num_cols].to_numpy())))
                    max_numeric_diff = max(max_numeric_diff, diff_val)
                except Exception:
                    pass
            mismatches.append((f, diff_val))

    print("=" * 60)
    print(f"Total files compared: {len(before)}")
    print(f"Inconsistent files count: {len(mismatches)}")
    print(f"Max numerical difference across all files: {max_numeric_diff:.3e}")
    if mismatches:
        for f, d in mismatches:
            print(f"  MISMATCH: {f.relative_to(BASE)} (diff={d:.3e})")
    else:
        print("ALL FILES BIT-FOR-BIT IDENTICAL!")


if __name__ == "__main__":
    main()
