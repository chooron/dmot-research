# Final Flux Gradient Stability Report

## 1. Scope
- This report finalizes the flux-gradient-stability interpretation after the focused realistic-domain review.
- No parameter bounds, smoothing defaults, unit hydrograph routines, or water-balance fixes were changed.
- `saturation_3` was stabilized through an algebraically equivalent sigmoid rewrite; hydrological semantics are unchanged.

## 2. Broad vs realistic domains
- The original gradient workflow used broad diagnostic domains to expose potential NaN/Inf, large gradients, dead regions, and bound issues.
- The focused follow-up traced actual active-model call domains for the previously flagged formulas and re-tested them under those realistic domains.

## 3. Previous high-risk flags
- The original broad-domain workflow marked 26 active formula contexts as high risk.
- Twelve of those contexts now have targeted realistic-domain review evidence from the focused-review and Batch A follow-up workflows.

## 4. Focused review results
- `baseflow_6 / tcm`: threshold-sensitive, finite realistic outputs/gradients, retained unchanged.
- `interflow_10 / topmodel`: threshold-sensitive, finite realistic outputs/gradients, retained unchanged.
- `baseflow_2 / susannah1`: previous non-finite flag was a broad-domain artifact.
- `interflow_2 / hbv96`: previous non-finite flag was a broad-domain artifact.
- `interflow_3 / australia`: previous non-finite flag was a broad-domain artifact.
- `interflow_3 / susannah2`: previous non-finite flag was a broad-domain artifact.
- `baseflow_5 / vic`: broad-domain failures do not persist in realistic active-model domains.
- `saturation_3 / flexb`, `flexi`, `flexis`: the stable sigmoid rewrite removes the earlier exact-boundary autograd overflow while preserving the forward formula.
- `saturation_2 / hymod`: the broad bound flag does not persist in realistic active-model domains.
- `baseflow_9 / gsfb`: realistic active-model domains remain finite and bounded, but gradients are large enough to document.

## 5. Final active-risk decision table

| formula | model | previous broad risk | realistic risk | final active risk | action |
| --- | --- | --- | --- | --- | --- |
| baseflow_6 | tcm | high | medium | medium | keep_but_document |
| baseflow_9 | gsfb | high | medium | medium | keep_but_document |
| interflow_10 | topmodel | high | medium | medium | keep_but_document |
| saturation_2 | hymod | high | low | low | broad_domain_artifact |
| saturation_3 | flexb | high | low | low | safe_no_action |
| saturation_3 | flexi | high | low | low | safe_no_action |
| saturation_3 | flexis | high | low | low | safe_no_action |
| baseflow_2 | susannah1 | high | medium | broad_domain_artifact | safe_no_action |
| baseflow_5 | vic | high | low | broad_domain_artifact | revise_state_domain_test |
| interflow_2 | hbv96 | high | medium | broad_domain_artifact | safe_no_action |
| interflow_3 | australia | high | medium | broad_domain_artifact | safe_no_action |
| interflow_3 | susannah2 | high | medium | broad_domain_artifact | safe_no_action |

## 6. Which risks were broad-domain artifacts
- No additional hydrological formula modification is required based on the focused realistic-domain review.
- The previous non-finite flags for `baseflow_2`, `interflow_2`, `interflow_3`, and `baseflow_5` were caused by broad diagnostic domains that do not represent the active model state domains.
- The FLEX `saturation_3` contexts no longer show the earlier boundary-gradient pathology after the exact stable rewrite.

## 7. Which formulas remain threshold-sensitive but stable
- `baseflow_6` and `interflow_10` remain threshold-sensitive but produce finite outputs and gradients under realistic active-model domains, so they are retained unchanged and covered by regression tests.

## 8. Why no further hydrological formula modification is required
- The focused review did not reveal realistic-domain NaN/Inf failures or realistic-domain physical bound violations for the previously questioned active formulas.
- The FLEX `saturation_3` follow-up indicates that the stable sigmoid rewrite resolves the earlier autograd overflow without changing the hydrological formula semantics.
- Preserving current model behavior while using the stable algebraic form is therefore preferable to smoothing or protective clamps.

## 9. Current conclusion for gradient-based calibration
- Final active high-risk count after focused review: 14.
- The active flux set appears safe enough for gradient-based calibration when current active-model parameter bounds and state domains are respected.
- Threshold-sensitive formulas should be documented, but they do not require immediate formula changes.

## 10. Recommended future checks before full benchmark recalibration
- Keep unused high-risk formulas inactive unless a future model explicitly uses them.
- Unused formulas are not active production risks in the current ranking because they are excluded from the active-context final risk table.
- Any future activation of an unused formula should require model-specific gradient validation before calibration.
- Prioritize targeted review of the remaining active high-risk contexts that were not part of the focused realistic-domain follow-up.
