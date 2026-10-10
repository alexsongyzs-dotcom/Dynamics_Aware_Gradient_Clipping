# Completed develop/lyx1 evidence

This directory contains the compact export of the frozen 405-outcome study.
There are 117 calibration and 288 confirmation outcomes: 397 full-budget
completions and eight predefined scientific failures, retained in all analyses.
No process error is counted as a scientific result.

## Inventory

- `evidence/manifest.json`: frozen protocol and training-source digest.
- `evidence/seed_results.jsonl`: all 405 configurations, final results, diagnostics,
  and epoch training metrics. Failed results retain the reason and executed steps.
- `evidence/selection.json`: complete validation candidate scores and selected values.
- `evidence/trace_bins.jsonl`: 100-bin confirmation diagnostics for figures.
- `evidence/replay/`: full-length predetermined threshold arrays for each condition.
- `evidence/audit.json`: completeness checks and SHA-256 hashes for every original
  result and full-resolution trace, plus replay construction checks.
- `evidence/calibration_carryover.json`: pre-confirmation sample-size amendment
  and validation of unchanged calibration results carried forward.
- `evidence/training_environment.json`: actual Python, packages, CUDA, GPU,
  implementation commit `233bb79`, and protocol amendment `57f5565`.
- `all_runs.csv`, `summary.csv`, `primary_comparisons.csv`,
  `secondary_comparisons.csv`, `analysis.json`: derived numerical analyses.
- `mechanism_audit.json`: post-confirmation descriptive full-trace checks of
  direction masking and inactive fixed clipping at CIFAR-10 LR 0.1.

Original absolute replay paths are replaced by relative evidence paths in the
export. Each affected record carries an export note; the original run identity
and original-result hash remain available. File hashes in `audit.json` refer to
the original raw artifacts, not the sanitized JSONL rows.

The persistent experiment archive retains original JSON records, full-resolution
`trace.npz` files, checkpoints, logs, the initial partial calibration execution,
and the sample-size amendment. These large raw files are not copied into Git.
The compact export is sufficient to regenerate all aggregate tables and figures.
To reproduce the full-trace audit, obtain the original campaign archive or run
a new campaign; binned curves cannot establish exact trajectory equality.

## Reproduction

From the repository root:

```bash
python -m analysis.confirmatory_report
bash scripts/build_paper.sh
```

With a complete raw campaign:

```bash
python -m analysis.audit_confirmatory_mechanism --root PATH_TO_CAMPAIGN \
  --output PATH_TO_NEW_AUDIT.json
```

The export is immutable: choose a new destination for a new training campaign.
No reported outcome was selected, discarded, or retuned after confirmation.
Calibration uses validation data only, but earlier FashionMNIST method development
had accessed the official test set; this historical limitation is disclosed.
