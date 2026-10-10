# Manuscript and reproducible build

The `develop/lyx1` manuscript and [PDF](main.pdf) report the completed,
audited 405-run study: 117 validation calibrations and 288 confirmations.
Eight no-clipping FashionMNIST runs met the predefined failure rule and remain
in all aggregates. There are no process errors or pending experiment results.

The frozen DAGC rule loses to tuned fixed clipping in all three conditions.
Replay meets the stated per-comparison equivalence criterion, and full CIFAR
traces show that lower-bound projection masks the alignment term. The paper
reports these negative results explicitly rather than retaining the earlier
broad method-superiority claim.

The revised title is **Dynamics-Aware Gradient Clipping: Separating Feedback
from Threshold Scheduling**. The paper includes a local mean-field analysis,
new full-data comparisons, and corrected reporting of legacy experiments.
Legacy tests are explicitly exploratory because official test scores were
available during development. All current numerical tables and figures are generated from the exported
seed-level evidence. Narrative result fragments are checked against that evidence.

The final evidence is included at `results/lyx1/evidence/`. Rebuild from the
repository root:

```sh
bash scripts/build_paper.sh
```

This first regenerates tables and figures from seed-level evidence and then
compiles with either `pdflatex` plus `bibtex`, or `tectonic`. To compile only
already-generated paper inputs:

```sh
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`main.tex`, `references.bib`, the local style files, `generated/`, and the
referenced `figures/` files are required. The source uses T1 font encoding so
that the Times-based template also renders correctly through Tectonic/XeTeX.

The intended venue is ICML 2027, with the ICML 2026 style retained provisionally.
The new template and submission rules still need to be checked when available.
This is a research draft, not an accepted paper. The preprint option avoids claiming that the draft is under review. The anonymous author block
is retained; final affiliation and correspondence details require the authors.

## Integrated local revision (2026-10-10)

The completed lyx1 confirmation manuscript is integrated with the earlier audited literature, screening/V1 appendices and seven mathematical propositions. All formal proofs are in the appendix. The publication copy is maintained on `develop/yzs`; main is unchanged. See `integration_notes_20261010.md`.
