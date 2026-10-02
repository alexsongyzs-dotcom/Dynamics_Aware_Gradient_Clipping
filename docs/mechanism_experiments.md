# Mechanism experiments

Entry point: `python scripts/run_v2_mechanism.py`.

FashionMNIST CNN, 5,000 train / 2,000 test images per seed; SGD LR 0.8,
momentum 0.9, weight decay 1e-4; 15 epochs, batch size 256 (300 steps).

Calibration: seeds 800--802, full V2 and fixed thresholds 0.05/0.1/0.2/0.3/0.5.
Select threshold by mean intensity discrepancy, not accuracy: selected 0.2.
Replay schedule averages the three calibration V2 threshold traces step by step.
Evaluation: seeds 900--907, eight policies, paired initialization/subsets/batch order.

Policies: full V2; no alignment (osc_weight=0); no exposure (relax=0);
frozen multiplicative adaptation (gamma=0, median-based bounds remain active);
matched fixed 0.2; calibration-trace replay; AutoClip 10th percentile of all
observed norms including current; rate tracking with exposure EMA 0.9 and
c_next=c*exp(0.05*tanh(0.3*(E-0.1))). The tracking analogue has no median bounds.
AutoClip and rate tracking are specified configurations, not fully tuned baselines.

V2: init_c=0.5, gamma=0.05, beta=beta_a=0.9, relax=0.3,
osc_weight=3, exposure_target=0.1, median window 200, scale bounds 0.1/10.

Results: `results/v2_mechanism/raw_runs.csv` and `summary.csv`.
Intervals use mean +/- 1.96*sample_sd/sqrt(8); paired differences are V2 minus
alternative within seed. Exact ties are excluded from wins. JSON files include
complete step trajectories. Existing JSONs are reused: use a fresh output directory
when changing the protocol.

The exposure signal triggers protection rather than tracking a target rate.
Removing exposure costs 5.47 points; removing direction costs 0.16 +/- 0.41.
Replay is only 0.21 +/- 0.53 points below V2. These controls narrow the claim.
