#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m analysis.confirmatory_report
cd paper
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs main.tex
else
  printf '%s\n' 'Install pdflatex with bibtex, or tectonic, then rerun this script.' >&2
  exit 1
fi
