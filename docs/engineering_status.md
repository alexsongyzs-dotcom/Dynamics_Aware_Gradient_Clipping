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

## Deliberately not executed on this machine

Training, dataset download, dependency installation, notebooks, and unit tests
are deferred to the Linux environment. The current checkout therefore contains
no new empirical result and makes no claim that the implementation has passed
runtime validation. Follow `docs/reproducibility.md` after migration.
