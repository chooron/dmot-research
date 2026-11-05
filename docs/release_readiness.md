# Release readiness: local candidate only

**Disposition: NO-GO for public Zenodo deposit.** This package is prepared only for local inspection. MARRMoT full numerical equivalence and a clean Git tree/tag are not release gates; the unresolved items below are independent scientific, legal, package, and public-record requirements.

## Blocking gates

1. **Rights and license:** the dMoTpy source has no verified project license or rights-holder statement. MARRMoT source headers state GPLv3, and compatibility/attribution for translated or derivative formulas has not been legally reviewed. No license has been invented or assigned here.
2. **Authors/citation:** no verified creator names, affiliations, ORCIDs, copyright owners, or final paper DOI were supplied. `CITATION_DRAFT.md` is intentionally incomplete and not a valid Zenodo deposit record.
3. **Scientific boundary:** the scoped GSFB exact-zero/tiny-store residual finding remains unresolved under the cited strict threshold; the candidate integration disposition was NO-GO. A public release must either resolve it, establish and document a valid domain restriction, or record an authorized scientific waiver.
4. **Paper-source correspondence:** the paper's executed source is the archived SHA-256 in `paper_reproduction.md`; this candidate differs in four model files, including materially measured GSFB and Plateau impacts. Public wording must make the separation unambiguous and must not claim that this tree generated every paper result.
5. **Reference completeness:** `docs/model_reference.csv` is an inventory of citation leads from the comparison module headers, not a fully verified bibliography. Rows with source-encoding or bibliographic issues require primary-source verification before a public release.

## Candidate package scope

The candidate carries 36 model modules, shared runtime modules, neural-network and trainer source, 30 test modules, and 12 pre-existing validation artifacts required by the complete artifact-integrity test. It excludes all bulk `validation_results` files other than those 12, research scripts, old module-level test material, all caches, checkpoints, optimizer state, forcing or observation series, MARRMoT code, and Git metadata. The staged Python model-source files match the recorded final-closeout hashes for all 36 models; this confirms file identity only, not a scientifically correct model family or parity.
The Euler status CSV is copied verbatim from `dmotpy/validation_results/euler_convergence_final/` only to satisfy two legacy tests that assert its presence and shape. It is a historical result fixture, not regenerated and not evidence of scientific behavior for this candidate; its interpretations remain source-bound.

The independent installation and complete test results are recorded in the accompanying assessment. Passing packaging/tests does not clear the scientific or legal gates above.
