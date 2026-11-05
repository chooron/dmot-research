# dMoTpy versus MARRMoT: scope and model differences

MARRMoT is a comparison implementation, not a universal correctness oracle. The goal of this comparison is to identify formula, state, parameter, routing, and numerical differences—not to require identical outputs or calibration scores. Static equation review is not numerical parity.

## Classification used in the audit

- **A — numerical implementation difference:** changes in solver/time-step realization, operation order, precision, or smooth/regularized evaluation where the underlying process relation is intended to remain the same.
- **B — reasoned process approximation:** a declared simplification or differentiable approximation with an explicit rationale and an appropriately bounded validation claim.
- **C — material process/formula/parameterization difference:** potentially changes mechanisms, state/flux topology, distribution family, routing support, or parameter meaning; explain it and bound its effects before making a fidelity claim.
- **D — suspected correctness defect or unsupported behavior:** a source-level or numerical concern that must be resolved, domain-restricted with evidence, or explicitly accepted by an accountable scientific owner before release.

These categories apply to the stated evidence and source snapshot; they are not rankings of model quality.

## What the existing 36-model audits establish

The internal formula-change matrix classifies 27/36 models as mathematically equivalent and 9/36 as having an equation change, relative to its recorded internal source comparison. A separate MARRMoT difference audit contains 36 per-model base rows: 19 `EQUIVALENT`, 15 `UNVERIFIED`, and 2 `STRUCTURAL`. In that audit, “equivalent” is limited to static wiring/count checks and does **not** establish whole-model parity. Its broader 53-row table includes 2 numerical classifications, but does not certify every formula or parameter meaning.

The distinctions below are high-priority disclosures, not an exhaustive new line-by-line derivation of all 36 original papers:

| Finding | Class | Release interpretation |
|---|---|---|
| `saturation_7`: dMoTpy uses a log-normal moment match; the cited MARRMoT helper integrates a shifted-Gamma survival tail. | C | This is a distribution/formula change, not merely a numerical quadrature or floating-point difference. State the approximation and do not claim identity. |
| `ihacres`: dMoTpy exposes 6 parameters and one deficit state; the mapped MARRMoT module is 7p/1s and includes a distinct routing/delay coordinate. | C | Parameter count and routing meaning differ; document this as a variant and do not transfer parameter values as though coordinates were identical. |
| `simhyd`: dMoTpy documents a 7-parameter, two-store (soil/groundwater) variant without the mapped MARRMoT three-store/Gamma-UH structure. | C | This is a topology/routing change, not an implementation tolerance. Label the variant explicitly. |
| `gr4j`: dMoTpy uses analytic daily production/percolation and separate routing/UH helpers; the mapped MARRMoT source documents a coupled/simultaneous store formulation. Exact original–MARRMoT–dMoT routing support and exchange accounting remain only partly reconciled. | C | Explain the chosen daily representation and leave the unresolved three-way derivation as an explicit limitation. |
| Nine recorded internal equation-change cases: GSFB, HBV96, MODHYDROLOG, Penman, Plateau, SMAR, TCM, TOPMODEL, and VIC. | A/B/C by mechanism | Current differences include availability limits, state floors, precipitation partitioning, lower-zone ET behavior, and explicit calendar handling. The TOPMODEL entry in that matrix is a tensor dtype construction, not by itself a hydrological process change. See the source modules and the traceability register; do not reduce all nine to one generic “smoothing” change. |
| Daily explicit sequential updates, smooth gates, positive floors, denominator protections, and caps versus solver-driven reference calculations. | A or B where justified | These can shift trajectories/gradients near active-set boundaries. Describe the exact operator and tested domain; do not call them numerical noise without evidence. |

The older adjudication that flagged opposite-side SMAR/TCM/Penman ET gates is not carried forward as a present-source defect without qualification: the current inspected modules use a below-threshold gate in those pathways. That statement is a static source observation; it does not certify all trajectories or all parameter configurations.

## Current high-priority correctness risk

**D-suspected — GSFB exact-zero/tiny-store boundary.** The scoped boundary review reports two one-step residuals of about `-3e-6` mm/day at exact-zero/tiny stores, exceeding its `1e-8` mm/day limit. The later fixed-case formal balance artifact reports 188/188 passes, but those cases do not cover this exact boundary. The boundary-repair disposition was **NO-GO for integration** pending the conservation-policy decision and an accepted domain/fix. Do not present the 188/188 result as proof of unrestricted GSFB conservation.

## Historical numerical comparison

A historical fixed-parameter Octave comparison reports 209/216 successful simulations, but its executed-source binding is not reconciled to this candidate. It is not evidence of blanket parity and is not presented as a test of this package. The paper's exact executed source and this candidate's post-freeze model changes are separately documented in `paper_reproduction.md`.
