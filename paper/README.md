# Current manuscript and reproducible figures

Current title: **Dynamics-Aware Gradient Clipping: Separating Feedback from Threshold Scheduling**.

- [PDF](main.pdf)
- [LaTeX source](main.tex)
- [References](references.bib)
- [Experiment audit](audit_20261010/audit_report.md)
- [Literature update and source links](literature_update_20261010.md)

The 2026-10-10 revision integrates the exploratory V1/V2 experiments and the completed 156-run screening. It corrects uncertainty calculations, method labels, test-data reuse reporting and the zero-gain control. The manuscript actually cites 43 works. The confirmation campaign in develop/lyx1 has no complete audited export in the fetched repository; pending-result text is explicit. This is a research draft, not an accepted paper.

The main figures are compact and placed alongside their corresponding results; full original benchmarks, screening and supplementary diagnostics are in the appendix.

Compile from this directory:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

To regenerate the audited historical figures from repository CSV/JSON records, run from the repository root:

```sh
python paper/audit_and_plot.py
python paper/compact_figures.py
```

Then compile the paper again. The figure scripts require NumPy, pandas, SciPy and Matplotlib. Local generated tables, figures and style files are included. ICML 2026 style is retained provisionally for the intended ICML 2027 venue; switch to the official template when available.

Future manuscript updates replace these stable TeX/PDF paths through ordinary commits. Previous versions remain in Git history.
