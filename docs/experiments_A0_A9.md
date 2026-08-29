# Experimental Program A0-A9

Load-bearing experiments (determine the paper's central claims):
**A2, A4, A5, A6, A7**. A9 is contingent.

## A0 — Instrumentation and reproducibility validation
Validate all dynamical measurements and the experimental infrastructure.
Not a main scientific result.

## A1 — Local stability threshold in neural networks
Hypothesis: training dynamics change systematically as eta_t * lambda_max(H_t)
approaches/exceeds the classical stability scale (S_t ~ 2).
Questions: does clipping (a) alter the frequency of unstable excursions,
(b) reduce overshoot after high-curvature encounters, (c) change time spent
above the boundary, (d) redirect the trajectory?

## A2 — Learning-rate--clipping phase diagram
Dense (eta, c) sweep; per configuration measure convergence, test
performance, clipping exposure, switching count, oscillation score,
curvature, update magnitude. Main visualization: neural training dynamical
phase diagram.

## A3 — Clipping exposure
Compare F_clip, I_clip, burst length, first exposure time, early exposure.
Which statistics best predict optimization and generalization?

## A4 — Switching geometry along training trajectories
Windows around s_t s_{t+1} < 0; measure changes in L_t, ||g_t||,
||dtheta_t||, S_t, C_1(t). Is repeated switching associated with a
reproducible local signature?

## A5 — Oscillatory and alternating dynamics
C_1(t), C_2(t), R_2(t), autocorrelation (loss, gradient norm, update
direction), spectral concentration, switching-conditioned oscillation.
Does clipping suppress, amplify, or reorganize oscillatory regimes?

## A6 — Controlled trajectory selection
Paired runs, identical everything except clipping policy. Small delta-c
perturbations; early-only vs late-only vs full vs none. Track divergence in
parameter space, prediction space, representation space, final curvature,
final generalization. **At least 10 paired seeds (20 where feasible).**
Final solutions judged functionally different via:
linear mode connectivity barrier, CKA, calibration/subgroup differences,
held-out disagreement rates.

## A7 — Matched-update causal controls
Rule out "clipping just reduces step size". Two concrete controls:
1. **norm-matched scaling**: rescale the raw update to the clipped norm at
   every step (destroys state dependence, matches step sizes);
2. **random gating**: Bernoulli gating at the empirical clipping frequency
   (preserves frequency, destroys timing).
Compare optimization performance, switching, oscillation, curvature,
trajectory selection.

## A8 — Dynamics-aware gradient clipping (DAGC)
Controller selected after A1-A7. Baselines: no clipping, fixed global norm,
tuned fixed, AGC, noise-scale adaptive clipping, SAM, normalized-update
controls, matched-update controls. Primary outcomes: final test performance,
optimization speed, training stability, clipping exposure, learning-rate
robustness, clipping-hyperparameter robustness, computational overhead.

## A9 — Large-scale validation (contingent)
Run only if A1-A8 support the central claims. ImageNet-scale is optional;
may be downgraded to a moderately larger vision model if compute is limited.
