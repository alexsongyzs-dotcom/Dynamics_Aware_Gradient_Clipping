# Small-scale validation: FashionMNIST, seed 0

## Run definition

- Dataset: FashionMNIST (60,000 training / 10,000 test images)
- Model: small CNN
- Optimizer: SGD, learning rate 0.1, momentum 0.9, weight decay 1e-4
- Training: 10 epochs, batch size 256, random seed 0
- Fixed-clipping threshold: 0.6276, the median gradient norm from a 3-epoch no-clipping probe
- Policies: no clipping, fixed global-norm clipping, DAGC

## Observed result

DAGC reached 91.59% test accuracy, compared with 91.32% for fixed clipping and
91.00% without clipping.  Its clipping frequency was 8.1%, compared with 60.3%
for the fixed policy, and it recorded 47 threshold-crossing switches compared
with 924 for fixed clipping.

This is an exploratory single-seed result.  It demonstrates feasibility and a
useful signal, but it is not statistical evidence for the paper's central claim.
The planned multi-seed paired A6/A7 experiments remain necessary.

## Files

- `figures/training_diagnostics.png`: step-wise loss, gradient norm, alignment,
  and clipping threshold.
- `data/summary.csv`: values printed by the completed run.

## Measurement limitation

The Hessian top-eigenvalue output was 0.0 at every logged checkpoint for every
policy.  That is not credible for this network and dataset, so curvature is
excluded from interpretation until the Hessian implementation is corrected and
validated independently.

