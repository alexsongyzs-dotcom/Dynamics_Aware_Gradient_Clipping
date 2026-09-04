# Gated Experimental Program (supersedes A0-A9)

The filename is retained for compatibility, but the old linear A0-A9 program
is replaced by gates G0-G3 and phases P0-P5. A later phase cannot be treated as
confirmatory until the earlier gate has passed.

## P0 — Measurement and replay validation (G0)

Validate on a tiny deterministic problem before any scientific sweep:

1. raw gradient norm equals a direct tensor calculation;
2. applied gradient norm equals the declared pre-moment coefficient;
3. proposed and applied update norms are distinct for post-update controls;
4. vanilla-SGD norm matching is numerically identical to global clipping;
5. checkpoints reproduce the next minibatches, RNG draws, optimizer state,
   policy state, event state, and logged rows;
6. event labels contain no future-derived input features.

Failing G0 invalidates all downstream evidence.

## P1 — Screening and predictive increment (G1)

Use two seeds over a coarse learning-rate and threshold grid. Record failures,
loss, accuracy, raw/applied norms, exposure, episode count, burst length,
directional diagnostics, and checkpoint curvature. Recompute events over the
preregistered hysteresis grid and compare to block/circular surrogates.

Episode structure must first be reproducible over seeds and robust to detector
settings and surrogates. Then construct a future adverse-event target
separately within each run. Compare:

- Base: instantaneous loss, raw gradient norm, learning rate, and optimizer
  state variables;
- Base+History: Base plus exposure EMA, duty cycle, switch rate, burst age,
  time since transition, and mean clipping intensity.

Use whole-run grouped cross-validation. Primary increment is held-out AUPRC;
also report AUROC, Brier score, and log loss. Training steps are never treated
as independent statistical replicates. G1 passes only if at least one history
feature has stable incremental value across held-out runs/configurations.

## P2 — Causal timing and placement branches (G2)

Fork from complete checkpoints with identical model, optimizer, scheduler,
sampler, augmentation RNG, policy state, and minibatch order. Compare:

1. reference gain replay before optimizer state;
2. globally time-shuffled gains;
3. block-shuffled gains;
4. random locations with the same active-gain multiset;
5. reference applied-update norm imposed after optimizer state.

Before training any branch, verify sequence length, sorted gain multiset, sum,
and active-step count. Primary outcomes must be preregistered: failure,
short-horizon loss change, trajectory divergence, and optimizer-state change.
Use paired seed/checkpoint estimates with paired confidence intervals.

G2 passes only if a timing or placement intervention changes a preregistered
functional/stability outcome, not parameter distance alone.

## P3 — Predictor freeze and contingent controller (G3)

Freeze the 1-2 signals that survive grouped validation, then evaluate the
single-aggressiveness `event_timed` policy against no clipping, fixed global
clipping, AutoClip, AGC, AdaGC, and StableAdamW. G3 requires a stability benefit
with non-inferiority on the remaining primary outcomes; a single accuracy gain
is insufficient.

## P4 — Trajectory, confirmation, and scale

For supported P3 effects, compare prediction disagreement, symmetric KL,
calibration, CKA on a fixed probe set, linear mode-connectivity barrier, final
curvature, and held-out accuracy. Parameter distance alone is insufficient to
claim a different solution.

Only after G3 passes, expand selected cells to confirmatory seeds and at most
one larger-scale task. Report wall-clock, memory, communication, and failed-run
costs.

## P5 — Freeze and writing

Freeze the event definition, primary outcomes, tables, configs, and anonymous
reproduction snapshot. Preserve all failed and negative runs.

## Seed policy and multiplicity

- screening: two seeds;
- confirmatory comparisons: at least five independent seeds;
- trajectory-selection and paired causal tests: at least ten pairs when
  feasible;
- family-wise exploration is labeled exploratory; confirmatory families use
  Benjamini-Hochberg adjustment;
- report run-level failure rates as first-class outcomes.
