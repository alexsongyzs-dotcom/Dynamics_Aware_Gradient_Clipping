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

## Trajectory storage

Full parameter trajectories are not stored. Instead:

- periodic checkpoints
- random projections (default dimension 512)
- selected-layer representations
- PCA coordinates
- prediction-space probes

See the research outline (Section "Computational Infrastructure") for details.
