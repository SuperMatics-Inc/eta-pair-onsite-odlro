# Thermal eta-pair order: replay supplement

Davis Rens, SuperMatics Inc. ([ORCID](https://orcid.org/0000-0001-5839-5197)).

This repository accompanies the preprint [Thermal eta-pair order at exact half filling with on-site repulsion](https://doi.org/10.5281/zenodo.24364883). The archived software release is [10.5281/zenodo.24364885](https://doi.org/10.5281/zenodo.24364885). The DOI links resolve after the records are published.

The paper proves order under the stated lattice, interaction, temperature, and ensemble hypotheses for a pair-hopping model. The finite-system numerical checks here are not, by themselves, a proof of thermodynamic long-range order.

## Replay

From `replay/`, run `python check.py` with NumPy installed and `python certify_integral.py`. The first tests finite CAR matrices numerically; the second uses exact integer and rational arithmetic for the integral bound and its mutant checks.

With the toolchain in `replay/lean-toolchain`, `lake exe cache get` followed by `lake build` checks `KomaCore.lean` using the pinned manifest. This formalizes selected algebraic inequalities, not the reflection-positivity or infrared-bound proof. See the paper for the full hypotheses and arguments.

The replay code and Lean source are licensed under Apache-2.0. The paper is distributed separately. The source snapshot is from manuscript commit `2ee80e803f`.

AI agents contributed proofs, certificates, code, Lean sources and manuscript drafts and edits under the author's direction and review.
