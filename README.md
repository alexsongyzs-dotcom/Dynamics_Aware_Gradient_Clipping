# Dynamics-Aware Gradient Clipping

Research code for testing whether the **temporal organization and optimizer
placement** of clipping interventions causally change training trajectories.
The repository does not assume that a new controller is useful: the proposed
event-timed policy remains contingent on passing preregistered measurement,
prediction, and causal gates.

The revised Chinese review and research plan is in
[`docs/paper_idea_review_and_revised_plan_cn.md`](docs/paper_idea_review_and_revised_plan_cn.md).

## Main claim under test

For vanilla SGD, multiplying an update by the global clipping coefficient is
exactly global norm clipping; it is not an independent matched-norm control.
The informative interventions in this project therefore preserve marginal
gain statistics while changing timing, or hold the applied update norm fixed
while changing whether the intervention enters before or after optimizer-state
updates.

## Implemented research stack

- per-step raw/applied gradient and proposed/applied update measurements;
- hysteretic clipping episodes and backward-looking history features;
- SGD-momentum and Adam second-moment diagnostics;
- checkpoint-only Hessian-vector power iteration on a fixed probe batch;
- frozen-gain replay, time shuffle, block shuffle, random gating, and
  pre-moment versus post-update branches;
- grouped predictive-increment analysis and paired branch statistics;
- parameter, prediction, representation, calibration, and mode-connectivity
  trajectory comparisons;
- full model/optimizer/scheduler/sampler/policy/RNG checkpoints and immutable
  JSONL run records.

## Layout

```text
configs/experiments/   gated P0-P3 experiment configurations
configs/sweeps/        screening grids
src/                   training, policies, diagnostics, checkpoints
analysis/              offline event, prediction, causal, and trajectory analysis
scripts/               plan-first entrypoints and branch preparation
tests/                 invariants and measurement-contract tests
docs/                  research plan, instrumentation, and Linux handoff
```

## Safe workflow

The planning commands do not train by default:

```bash
python scripts/quick_experiment.py
python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml
```

After migration to a prepared Linux environment, follow
[`docs/reproducibility.md`](docs/reproducibility.md). Actual execution requires
an explicit `--execute` flag or the `DAGC_EXECUTE=1` safety switch.

## Scientific gates

| Gate | Required evidence | Decision |
|---|---|---|
| G0 | Instrumentation identities and resume determinism pass | otherwise fix infrastructure |
| G1 | Robust event history improves held-out-run prediction beyond instantaneous variables | otherwise stop event thesis |
| G2 | Timing/placement branches change preregistered outcomes with matched marginals | otherwise do not claim causality |
| G3 | Event-timed policy beats tuned fixed and strong adaptive baselines on a stability outcome without material regressions | otherwise retain mechanism paper only |

Only after G0-G2 pass should `configs/experiments/p3_event_timed.yaml` be used
as an algorithm-development starting point; its confirmatory comparison is G3.

## License

MIT; see [`LICENSE`](LICENSE).
