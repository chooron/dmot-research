# Numerical and differentiable implementation notes

## Time step and process order

The model-step implementations operate on daily forcings and update stores/processes in an explicit sequential order. They are differentiable PyTorch computations, not a general-purpose continuous-time ODE solver. For model families with routing, the Python implementation may use analytic daily expressions or separate unit-hydrograph helpers where the reference implementation uses solver-driven within-step updates. Such changes are implementation choices only when the state/flux equations and parameter meanings remain equivalent; otherwise they are model-variant differences and must be disclosed.

## Smooth gates, bounds, and numerical protections

The source uses smooth threshold functions, positive floors, safe denominators, availability caps, and other protections in selected pathways. These support finite arithmetic and gradients, but can change flux values or gradients near thresholds, empty stores, and parameter-domain boundaries. A smooth approximation is not automatically equivalent to a hard threshold, and exact balance on selected test cases does not establish correctness over the full state/parameter domain.

The current GSFB source uses local physical-threshold handling and smooth availability caps. A separate boundary review still reports strict daily residuals of approximately `-3e-6` mm/day for exact-zero/tiny-store one-step probes, compared with the `1e-8` mm/day threshold used in that scoped review. The fixed-case 188/188 balance result does not cover that edge case; the issue remains a release gate, not a blanket claim that all GSFB runs fail.

## Parameters and mappings

Per-model bounds are defined by `dmotpy.models.core.PARAM_INFO` and the corresponding `*_PARAMS_BOUNDS` constants. `docs/model_reference.csv` records the bounds and initial-state/store mapping from this candidate's source. The `auto` mapping policy and any experiment-specific log transforms are part of the configuration; parameter values and ranges should not be inferred from a model name alone.

## Losses and gradient evidence

`dmotpy.losses.KgeLoss` defaults to `eps=1e-5`; the inverse/log loss classes also have their own stability and floor parameters. The GMD experiment audit records a different configured KGE stabilization value (`0.1`) for its reported protocol. Use the archived experiment configuration—not this package default—when describing or reproducing that analysis.

Existing evidence is bounded: the source-bound closeout records 188/188 fixed balance cases, while the paper handoff records 30/36 synthetic time-step-refinement assessments and no real-forcing substep error bound or 365-day AD/FD validation. These are historical artifacts scoped to their recorded snapshots and protocols; this candidate adds no new scientific experiment.
