# Paper manuscript

- [Paper PDF](main.pdf)
- [LaTeX source](main.tex)
- [References](references.bib)

This directory contains the current manuscript and all local figures and style files required to compile it. The PDF is copied from the latest compiled manuscript; only figure paths in the uploaded TeX have been changed to make this directory self-contained.

Compile from this directory with:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript targets ICML 2027 and currently uses the ICML 2026 style as a placeholder. Update to the official 2027 template when available. This is a research draft, not an accepted paper.