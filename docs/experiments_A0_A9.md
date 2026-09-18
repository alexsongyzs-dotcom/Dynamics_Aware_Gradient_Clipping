# DAGC Algorithm Evaluation Program

This program evaluates DAGC as an adaptive clipping algorithm. Curvature and
Hessian claims are intentionally out of scope.

## A0 — Instrumentation and reproducibility

Validate gradient norm, clipping coefficient, clipping exposure, switching
count, update norm, and gradient alignment. Record configurations, seeds,
hardware, wall-clock time, and per-run metrics.

## A1 — Learning-rate--threshold phase diagram

On a diagnostic model and dataset, sweep learning rate and fixed clipping
threshold. Measure convergence rate, final validation accuracy, divergence
rate, clipping exposure, and switching count. This identifies the regimes in
which fixed clipping is too weak, too aggressive, or useful.

## A2 — Temporal diagnostics

Compare no clipping, tuned fixed clipping, AGC, and DAGC. Report clipping
frequency and intensity, burst length, first exposure time, switching count,
gradient alignment, and loss/gradient-norm oscillation summaries. Use these
diagnostics to explain how DAGC differs operationally from fixed clipping.

## A3 — DAGC ablations

Remove one controller component at a time: exposure feedback, alignment
feedback, adaptive bounds, and threshold adaptation. Sweep `gamma`, `beta`,
and the initial threshold. The outcome is a concise ablation table showing
which components improve the accuracy--stability trade-off.

## A4 — Robustness benchmark

Evaluate on at least two datasets, two architectures, and SGD plus AdamW.
Include no clipping, tuned fixed global-norm clipping, AGC, and DAGC. Main
outcomes are test accuracy, training stability, optimization speed, clipping
exposure, hyperparameter robustness, and runtime overhead. Use at least five
independent seeds and report mean with 95% confidence intervals.

## A5 — Moderate-scale validation

After A1--A4 identify stable settings, run the strongest benchmark on a larger
vision model or dataset within the available compute budget. This validates
that the algorithm's benefit is not confined to the diagnostic setup.

## Publication criterion

The paper should claim an algorithmic improvement only if DAGC consistently
matches or exceeds the best baseline across the main benchmark while requiring
less threshold tuning or providing a clearer stability benefit. A single-seed
probe is useful for debugging but is not publication evidence.
