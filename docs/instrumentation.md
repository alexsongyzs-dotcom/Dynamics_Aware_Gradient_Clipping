# Instrumentation Contract

Every result row is append-only JSONL and carries a schema version, run ID,
and configuration hash. The contract separates quantities observed before the
optimizer from quantities applied to the model.

| Field | Meaning |
|---|---|
| `raw_grad_norm` | global norm after backward and before intervention |
| `coefficient` | scalar actually applied to the gradient/update |
| `applied_grad_norm` | norm entering the optimizer for pre-moment placement |
| `proposed_update_norm` | optimizer's raw parameter delta |
| `applied_update_norm` | parameter delta after any post-update intervention |
| `threshold` | declared gradient threshold, when meaningful |
| `clipping_coordinate` | raw norm divided by threshold, or inverse gain proxy |
| `placement` | `pre_moment` or `post_update` |
| `gradient_alignment` | cosine of fixed-coordinate gradient sketches |
| `momentum_buffer_alignment` | raw-gradient/momentum-buffer cosine |
| `adam_second_moment_mismatch` | gradient RMS relative to stored Adam second moment |
| StableAdamW RMS/scale fields | previous-step tensor update mismatch and clipping scale |
| episode/history fields | persistent state, transition, duty cycle, burst age, exposure |

AGC is tensor/unit-relative, so its single logged coefficient is the minimum
per-tensor coefficient and must not be misread as an exact global threshold.
The same caveat applies to AdaGC. StableAdamW records its per-tensor update RMS
inside optimizer state; causal comparisons use the proposed/applied parameter
delta rather than pretending it has a global gradient threshold.

## Episode definition

The primary event detector operates on `q_t` with separate entry and exit
margins and a minimum dwell time. Direct sign crossings are retained only as a
legacy sensitivity measure. Detector margins and dwell time must be varied on
a preregistered grid; block/circular surrogate sequences provide an artifact
check.

## Curvature scope

Top Hessian eigenvalues are estimated by Hessian-vector products and power
iteration only at declared checkpoints on a fixed probe batch. The classical
`eta * lambda_max = 2` boundary is exact only for deterministic gradient
descent on a local quadratic; for stochastic, momentum, or adaptive training
it is labeled a diagnostic proxy.

## Storage and identity

- fixed-coordinate sketches are cheap directional diagnostics, not a substitute
  for confirmatory full-gradient probes;
- training-batch sketches are logged each step, while a separate fixed-probe
  gradient norm and alignment are recorded only at curvature checkpoints;
- full parameter trajectories are not stored;
- checkpoints contain model, optimizer, scheduler, sampler, policy, event and
  history state, CPU/CUDA RNG state, and the prior gradient sketch;
- branch manifests store sequence hashes and reference paths;
- JSON serialization maps non-finite diagnostics to `null` rather than invalid
  JSON values.
