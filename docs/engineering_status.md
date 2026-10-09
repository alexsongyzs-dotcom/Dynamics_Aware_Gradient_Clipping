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

On 2026-10-09 the local CPU suite passed all 31 tests. These include exact
checkpoint recovery with augmentation/dropout, frozen-gain replay and common
continuation, dataset split separation, interrupted-log reconciliation, and
retaining a negative G2 report when all reference runs fail.

The full server campaign is specified in `docs/server30_protocol.md` and
`configs/campaigns/lyx_full.yaml`. CPU tests do not establish GPU reproducibility
or scientific efficacy. GPU checks and each scientific gate have separate
persisted reports; a stage is complete only when its actual report exists.
