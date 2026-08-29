# Instrumentation and Dynamical Diagnostics

Per-step quantities logged cheaply (experiment A0):

| Quantity | Definition |
|---|---|
| gradient norm | ||g_t|| |
| clipping coefficient | alpha_t = min(1, c_t / ||g_t||) |
| update norm | ||theta_{t+1} - theta_t|| |
| learning rate | eta_t |
| train/test loss | L_t |
| gradient cosine similarity | a_t = <g_t, g_{t-1}> / (||g_t|| ||g_{t-1}||) |
| switching events | s_t s_{t+1} < 0, s_t = ||g_t||/c_t - 1 |
| Hessian top eigenvalue | at selected checkpoints only (power iteration, HVP) |

## Hessian measurements

- Use Hessian-vector products + power iteration; never full eigendecomposition.
- Measure only on diagnostic models and at selected checkpoints
  (default: epochs [0, 10, 30, 60, 89]).
- Validate cheap curvature proxies (kappa_hat_t, a_t) against Hessian-based
  quantities before using them in larger experiments.

## Trajectory storage

Full parameter trajectories are not stored. Instead:

- periodic checkpoints
- random projections (default dimension 512)
- selected-layer representations
- PCA coordinates
- prediction-space probes

See the research outline (Section "Computational Infrastructure") for details.
