# Model reference and parameter bounds

`model_reference.csv` contains all 36 current model identifiers, state counts and store descriptions, parameter counts, the bounds declared by this source snapshot, comparison-module names, and original-literature citation leads. Bounds are inclusive source constants, not calibrated ranges or recommendations. They are copied from `dmotpy.models.core.PARAM_INFO` / the corresponding model `*_PARAMS_BOUNDS`; state labels follow the model initializer and module equations.

## Citation status

The literature entries are leads transcribed from the mapped MARRMoT model-file headers. They are not a completed primary-source bibliography and have not all been independently checked for author spelling, pages, DOI, or which cited paper defines the exact variant. Some legacy headers have encoding/DOI-format anomalies. Before public release, verify each lead against the original publication and replace the provisional wording with the full citation. No MARRMoT source code is included in this package.

## Reading the table

- `n_parameters` and `n_states` are the values in the source registry.
- `process_stores` gives the documented state meanings; `S1..Sn` entries refer to the source initializer when a more specific interpretation is not established in the table.
- `parameter_bounds` uses `name[lower..upper]` pairs separated by semicolons. These are physical/model coordinates before any configured normalized-to-physical parameter mapping.
- `original_literature_leads` is a citation lead, not a blanket claim of conformance to the cited original equations.
- `marrmot_comparison_module` names the comparison module recorded by the static audit; parameter coordinates and route/topology may differ.
