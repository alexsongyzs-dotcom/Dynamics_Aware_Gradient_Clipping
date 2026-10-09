# Manuscript and reproducible build

The `develop/lyx1` manuscript is being revised around validation-isolated
confirmation and explicit mechanism controls. The final PDF will be rebuilt
when all 405 frozen-protocol outcomes pass the evidence audit. Until then,
`main.tex` contains visible pending-result text and the committed `main.pdf`
remains the previous main-branch manuscript; it should not be cited as the new
study's result.

The revised title is **Dynamics-Aware Gradient Clipping: Separating Feedback
from Threshold Scheduling**. The paper includes a local mean-field analysis,
new full-data comparisons, and corrected reporting of legacy experiments.
Legacy tests are explicitly exploratory because official test scores were
available during development. The direction and significance of the new
results are not assumed in advance.

After final evidence is exported to `results/lyx1/evidence/`, run from the
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
