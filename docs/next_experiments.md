# Next Experiments: Scaling the Frozen DAGC-V2 Protocol

The completed diagnostic benchmark shows that clipping is not a universal
replacement for unclipped training in already stable, low-learning-rate runs.
The completed held-out V2 benchmark establishes a positive signal in an
unstable band: V2 exceeds calibrated fixed clipping and AGC on FashionMNIST-CNN
at learning rates 0.4 and 0.8, FashionMNIST-MLP at 0.8, and MNIST-CNN at 0.8.
All four evaluations use eight held-out paired seeds.

## Completed stage — Identify and validate the unstable band

FashionMNIST with the small CNN and SGD was used to locate the unstable band.
V2 was selected on development seeds 200--202 at LR 0.8 and then frozen at
`init_c=0.5`, `gamma=0.05`, `beta=0.9`, `relax=0.3`, `osc_weight=3.0`, and
`exposure_target=0.1`. Held-out CNN evaluations use LR 0.8 (seeds 300--307)
and LR 0.4 (308--315). The original V1 controller is retained as an ablation.

Fixed global-norm thresholds are calibrated on separate seeds and frozen before
evaluation: 0.25 for FashionMNIST-CNN, and 0.10 for FashionMNIST-MLP and
MNIST-CNN. The reported primary outcomes are accuracy, divergence, clipping
frequency, clipping intensity, and seed-wise paired differences.

## Stage 1 — Add a three-channel benchmark

Add a CIFAR-10-compatible three-channel model path. Use a fixed training budget
and a learning-rate pilot to locate an instability band separately for SGD and
AdamW. Calibrate every method with equal search budget on disjoint seeds:
fixed global clipping, AGC, ZClip, AdaGC, and frozen DAGC-V2. Do not change the
V2 controller parameters on CIFAR-10.

Run at least eight held-out paired seeds in the selected unstable setting and
report accuracy, divergence rate, and paired differences. If the high learning
rate gives universal divergence, lower it until fixed clipping can train while
no clipping is unstable; this is the intended comparison regime.

## Stage 2 — Strengthen the V2 ablation

At one or two pre-registered high-learning-rate settings where clipping is
needed, rerun the ablation with 10 seeds and 20 epochs. Compare V2 with V1,
an exposure-only V2 variant, an alignment-free V2 variant, frozen threshold,
and tuned fixed clipping. Report mean, 95% confidence interval, divergence,
and paired seed-wise differences.

## Stage 3 — Increase schedule and subset scale

Repeat the strongest CNN condition with a larger training subset and a longer
schedule. Keep calibration and held-out seeds disjoint. This checks whether V2
remains useful when the short 15-epoch diagnostic schedule is removed.

## Decision rule

The current manuscript may state the MNIST-family diagnostic claim: frozen V2
consistently exceeds the calibrated fixed baseline and AGC in four held-out
unstable-training evaluations. A broader claim requires Stage 1 and matched
baseline tuning. The completed low-learning-rate A4 benchmark remains a
negative-control result and should be reported honestly.
