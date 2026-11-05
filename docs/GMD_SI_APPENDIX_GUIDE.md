# GMD main text, SI, optional Appendix, and documentation plan

This is a placement guide, not a manuscript edit. The Methods v2 draft was not present in the repository, and the SI v3 cited by an older patch note was absent from the checked `~/Downloads` location. No line-by-line duplicate-content check or patch was possible.

## Main text

Keep only what is needed to understand and evaluate the paper's scientific claims: the model-ensemble scope; conceptual state/flux topology; daily differentiable simulation and parameterization at a high level; the exact experiments and estimands; the key measured results; and limitations that materially qualify interpretation. State that model variants are not claimed to be MARRMoT-output-identical. Any benchmark or performance statement must name its exact source/data/runner and basin protocol.

## Supplementary Information (preferred home for detailed methods)

1. **S1 — Model equations and shared fluxes:** all 36 source-bound equations, variable/state definitions, and shared flux definitions; distinguish transcription from independent validation.
2. **S2 — Model inventory and parameter bounds:** state/store counts and meanings, parameter names/ranges, transformations, and models with changed parameter counts or routing.
3. **S3 — Numerical realization:** daily update order, analytical versus sequential steps, smooth gates, floors, caps, unit hydrographs, safe denominators, and edge-domain limitations.
4. **S4 — Numerical and gradient checks:** protocols, acceptance rules, sample counts, PASS/FAIL/NOT_EVALUATED/N/A status, source hashes, and the exact bounds on claims. Separate the 188 balance cases from timestep-refinement and gradient evidence.
5. **S5 — Training and selection:** archived optimizer/epoch/checkpoint rules, training-window objective, and the fact that the temporal hold-out was not used for checkpoint selection. Use only values tied to the archived experiment bundle.
6. **S6 — Data and forcing:** catchment/attribute provenance, precomputed PET caveat, input schema, preprocessing, and licensing/provenance of third-party datasets.
7. **S7 — Source and reproduction provenance:** identify the archived executed-source hash and the current package snapshot separately; state the four post-freeze model changes and their measured frozen-IC KGE impacts. Do not use a Git label as the executed-source identity.
8. **S8 — Model-by-model differences and limits:** the 36-row traceability table, with A/B/C/D issue class, source evidence, unresolved scope, and no blanket equivalence claim.

Avoid repeating complete tables already in the main text. The actual SI draft must be restored before converting this outline into insertions.

## Optional Appendix

Not required by itself. Use only if a derivation is too long for the SI—for example a worked derivation of the GR4J daily production/routing representation or a detailed comparison of a selected threshold/distribution change. Put the concise result and its limitations in the main text/SI, then cross-reference the Appendix. Do not duplicate full model equations between Appendix and SI.

## Online documentation

Use for install/API instructions, forcing tensor shape and units, registry/parameter lookup, environment specification, examples, test command, and reproduction-bundle retrieval. It is not a substitute for the paper's method description, SI evidence, a license, or a public frozen archive.
