# Next Experiments: Establishing DAGC's Useful Regime

The completed diagnostic benchmark shows that DAGC is not a universal
replacement for unclipped training in already stable, low-learning-rate runs.
The next experiments should test the intended use case: training regimes where
large updates create instability or require clipping.

## Stage 1 — Find the stability boundary

Use FashionMNIST with the small CNN and SGD. For each learning rate
`[0.05, 0.10, 0.20, 0.40, 0.80]`, run five seeds for no clipping, tuned fixed
global-norm clipping, AGC, and DAGC. Train for 20 epochs on the full training
set. Mark a run as unstable when its loss becomes non-finite or it fails to
reach a pre-registered training-loss criterion.

For fixed clipping, tune the threshold only on a separate calibration seed from
`[0.25, 0.5, 1.0, 2.0, 4.0]`; freeze the selected threshold before evaluation.
For DAGC, tune `init_c` and `gamma` on the same calibration protocol, then
freeze them. Do not tune on the five evaluation seeds.

Primary outcomes are validation accuracy, divergence rate, time to reach a
fixed accuracy target, clipping frequency, and clipping intensity. The main
figure is accuracy and divergence rate against learning rate.

## Stage 2 — Strengthen the ablation

At one or two pre-registered high-learning-rate settings where clipping is
needed, rerun A3 with 10 seeds and 20 epochs. Compare full DAGC with no
exposure feedback, no alignment feedback, frozen threshold, and tuned fixed
clipping. Report mean, 95% confidence interval, and paired seed-wise
differences.

## Stage 3 — Generalize only after Stage 1 succeeds

Extend the selected settings to CIFAR-10 and a compatible 3-channel model.
The current diagnostic model code supports MNIST-family data only, so this
stage requires adding the CIFAR model path before execution. Evaluate SGD and
AdamW separately; preserve the tuning/evaluation split.

## Decision rule

Proceed toward a paper only if DAGC matches or exceeds the tuned fixed baseline
in the unstable band, while providing either a lower divergence rate, better
accuracy, or lower threshold sensitivity. The completed low-learning-rate A4
benchmark remains a negative-control result and should be reported honestly.
