# Unit Hydrograph Consistency Report

## 1. Scope of validation
- Compared independent NumPy/MATLAB-style routing kernels against the dMoT PyTorch unit-hydrograph implementation only.
- Excluded full-model water-balance behaviour from this validation.

## 2. MATLAB files inspected
- `route.m`, `update_uh.m`, `uh_1_half.m`, `uh_2_full.m`, `uh_3_half.m`, `uh_4_full.m`, `uh_5_half.m`, `uh_6_gamma.m`, `uh_7_uniform.m`, `uh_8_delay.m`.

## 3. dMoT files inspected
- `models/unithydro/base.py`, `uh_half_1.py`, `uh_full_2.py`, `uh_tri_3.py`, `uh_tri_4.py`, `uh_exp_5.py`, `uh_gamma_6.py`, `uh_uniform_7.py`, `uh_delay_8.py`, `uh_identity_0.py`.

## 4. Mapping between MATLAB, NumPy, and dMoT functions
- `uh_1_half` -> `build_uh_1_half_numpy` -> `DplHalf1`
- `uh_2_full` -> `build_uh_2_full_numpy` -> `DplFull2`
- `uh_3_half` -> `build_uh_3_half_numpy` -> `DplTri3`
- `uh_4_full` -> `build_uh_4_full_numpy` -> `DplTri4`
- `uh_5_half` -> `build_uh_5_half_numpy` -> `DplExp5`
- `uh_6_gamma` -> `build_uh_6_gamma_numpy` -> `DplGamma6`
- `uh_7_uniform` -> `build_uh_7_uniform_numpy` -> `DplUniform7`
- `uh_8_delay` -> `build_uh_8_delay_numpy` -> `DplDelay8`

## 5. Test cases used
- Impulse, shifted impulse, step, ramp, constant low flow, all-zero, random positive, random very small, random very large, short-tail truncation, and long-sequence visibility cases.
- Batch modes: single basin, shared-parameter multi-basin batch, and per-basin vectorized parameters.

## 6. Parameter ranges tested
- Scalar kernels: sub-step, one-step, medium fractional, and long fractional delays/time bases.
- Gamma kernel: multiple shape and scale combinations spanning broad, exponential-like, and sharply peaked responses.

## 7. CPU float64 comparison results
- Cases: 1120
- Max absolute error: 3.376e-09
- Max relative L2 error: 5.817e-15
- Direct kernel checks used strict absolute tolerances. Routed-output checks used a scale-aware absolute tolerance with the same relative-L2 thresholds because some deterministic tests use `O(1e6)` inflow magnitudes.

## 8. CPU float32 comparison results
- Cases: 1120
- Max absolute error: 6.006e-01
- Max relative L2 error: 9.537e-07

## 9. CUDA comparison results
- CUDA float64 cases: 1120
- CUDA float64 max absolute error: 3.085e-09
- CUDA float32 cases: 1120
- CUDA float32 max absolute error: 6.006e-01

## 10. Kernel property checks
- Verified direct kernel weights, impulse-response equality, non-negativity, unit-mass normalization, peak timing, and peak magnitude for every hydrograph family.

## 11. Conv1d alignment checks
- Verified causal alignment through impulse and shifted-impulse cases.
- Verified output-length preservation after symmetric padding plus right-tail slicing.
- Verified that the flipped PyTorch kernels reproduce MATLAB state-space routing rather than unflipped cross-correlation.

## 12. Pass/fail summary
- Total evaluated cases: 4480
- Failed cases: 0

## 13. Largest discrepancies
- `DplDelay8` / `random_large_64` / `long_fractional` / `float32` / `cpu`: max_abs_error=6.006e-01, suspected_cause=none
- `DplDelay8` / `random_large_64` / `long_fractional` / `float32` / `cuda`: max_abs_error=6.006e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `small_shape_long_scale` / `float32` / `cpu`: max_abs_error=3.998e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `sharp_peak_short_scale` / `float32` / `cpu`: max_abs_error=3.729e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `small_shape_long_scale` / `float32` / `cuda`: max_abs_error=3.373e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `mid_shape_mid_scale` / `float32` / `cpu`: max_abs_error=2.752e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `sharp_peak_short_scale` / `float32` / `cuda`: max_abs_error=2.550e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `mid_shape_mid_scale` / `float32` / `cuda`: max_abs_error=2.506e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `sharp_peak_short_scale` / `float32` / `cpu`: max_abs_error=2.366e-01, suspected_cause=none
- `DplGamma6` / `random_large_64` / `sharp_peak_short_scale` / `float32` / `cuda`: max_abs_error=2.366e-01, suspected_cause=none

## 14. Diagnosis of any mismatch
- No remaining mismatches exceeded the configured tolerances.

## 15. Initial discrepancies observed during development
- `base.py` previously normalized with `sum_w + epsilon`, which introduced a systematic sub-unit mass bias even when the raw kernel already summed to one.
- `uh_exp_5.py` previously truncated the exponential kernel and then renormalized globally, whereas MATLAB adds the omitted `(7, inf)` tail mass only to the last active bin.
- `uh_gamma_6.py` previously depended on fixed `max_lag` support, whereas MATLAB truncates dynamically at the first bin below `0.1%` of the peak and redistributes the omitted tail mass proportionally over the retained bins.

## 16. Recommended fixes, if needed
- The validation code is designed to expose kernel reversal, padding, off-by-one indexing, truncation, normalization, and broadcasting defects. The fixes above were sufficient; no further routing change is recommended.

## 17. Final assessment
- The dMoT grouped-`Conv1d` routing implementation can be considered numerically consistent with the original MARRMoT unit-hydrograph logic when it passes this suite within the stated tolerances.
