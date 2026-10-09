# Engineering Status

## Implemented locally

- standalone inheritable YAML configuration loader;
- CIFAR/MNIST-family loaders with stateful sampler and fixed probe subset;
- CNN, CIFAR ResNet, MLP, and small ViT models;
- no clipping, fixed global clipping, AutoClip, AGC, AdaGC, StableAdamW,
  frozen replay controls, and contingent event-timed clipping;
- explicit pre-moment/post-update intervention placement;
- per-step records, complete checkpoints, and branch-manifest generation;
- event, oscillation, phase-diagram, exposure, predictive, optimizer-state,
  causal, stability, and trajectory analysis utilities;
- unit tests for core mathematical and experimental invariants.

## Validation on develop/lyx

On 2026-10-09 the local CPU suite passed all 35 tests. These include exact
checkpoint recovery with augmentation/dropout, frozen-gain replay and common
continuation, dataset split separation, interrupted-log reconciliation, and
retaining a negative G2 report when all reference runs fail, and deterministic
adaptive pooling with output/gradient agreement against PyTorch.

The first server G0 attempt passed the previous 31-test suite with CUDA replay enabled but exposed
an unsupported deterministic CUDA backward in the real SmallCNN's adaptive
pooling. Its logs are retained. The explicit pooling-window implementation
fixes that operation without disabling deterministic algorithms; the revised
source (`488032c`) passed the 35-test server suite, including strict CUDA replay
and backward tests, and all six real-model G0 checks.
The largest relative norm error was `1.17343e-7` against the `2e-6` tolerance.
P1's 156 screening runs ended at 2026-10-09 16:28:02 UTC: 154 completed all
20 epochs on the declared full training split; two ended under the frozen loss
explosion rule. All 156 configuration identities and completed-run step budgets
were verified. The eight runs resumed after maintenance have continuous,
duplicate-free logs. There were no infrastructure errors.

G1 did not pass. Only two runs supported within-run two-class predictive
comparisons, below the minimum of ten; predictive increments did not meet the
threshold, and family-held-out validation was not informative. Event robustness
was 27/63 (42.86%), below 75%. The campaign stopped at its scientific gate;
P2/P3/P4 were not run. This is insufficient evidence under this protocol, not a
general refutation of the mechanism. See the [full result report](server30_results.md).

The full server campaign is specified in `docs/server30_protocol.md` and
`configs/campaigns/lyx_full.yaml`. CPU tests do not establish GPU reproducibility
or scientific efficacy. GPU checks and each scientific gate have separate
persisted reports; a stage is complete only when its actual report exists.

The current server output is
`/mnt/data/users/yuxin/workspace/DAGC-develop-lyx/experiments/lyx_full_20261009_r2`.
Its `status.json` is the live progress record. Completed G0 evidence was also
retrieved to `results/server30/` in the local checkout; the frozen scientific
source digest matches the local source. Data are cached at
`/tmp/dagc-yuxin-runtime/data`; original archives and checksums remain in the
persistent task directory. G1 evidence and analysis are archived locally in
`results/server30/final/`; selected numerical evidence is in `docs/evidence/`.
The background process exited successfully and recurring supervision is paused.
