# Paper results and candidate source identity

The GMD v1.0 canonical experiment audit identifies the executed source tree by the archived snapshot SHA-256:

`57211c13310c29d8271b39369d85eeb6f5f2960add9eeb71a9259930cd7c5963`

The executed snapshot, not a Git label, is the reproducibility identity. A provenance closure compared 401 files (388 identical, 13 different) and found all 37 model sources and shared evaluation files byte-identical to the then-local checkout. Later edits changed four model files. The archived experiment snapshot is not included in this package.

## Measured frozen-IC impact of later source edits

The recorded replay compares the archived executed snapshot and the later working tree under the same frozen IC table across 2,124 basin–model rows. The archived snapshot reproduces the frozen table to about `5e-11`; the later source has the following maximum absolute KGE differences:

| Model | Maximum absolute KGE difference |
|---|---:|
| GSFB | 0.278 |
| Plateau | 0.062 |
| HBV96 | 4.1e-5 |
| MODHYDROLOG | 4.0e-6 |

Accordingly, the archived snapshot is the source identity for those paper numbers. This candidate is the later model source and must not be described as reproducing every published value. No paper values were changed and no replay was run while preparing this candidate.

The archived dPL replay reports 36/36 model medians passing its recorded gate with maximum absolute median-KGE difference `4.9070747e-12`, when run with the required archived Caravan attribute file and the official protocol. Those are archived, source- and data-bound results, not a new validation of this package.

## Boundaries of this package

- No trained IC/dPL checkpoint, optimizer state, forcing series, or historical prediction series is included.
- The parameter/checkpoint Git revision labels are not used as executed-code identities.
- Exact reproduction requires the original archived source, data, runner, and environment. This package does not claim to replace that bundle.
- The historical MARRMoT/Octave comparison is not bound to this candidate and cannot support a full-parity claim.
- The package's `KgeLoss` class defaults to `eps=1e-5`; the archived GMD training protocol records an explicitly configured `epsilon=0.1`. Do not use the class default to infer the paper's metric configuration.
