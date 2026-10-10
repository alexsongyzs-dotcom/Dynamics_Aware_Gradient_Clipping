# Experiment audit, 2026-10-10

Scope: all located legacy raw CSV tables, cross-checks of mechanism and perturbation CSV against per-run JSON, and archived 156-run screening. Remote develop/lyx1@050beca still contains no complete confirmatory evidence export; server execution has not been independently inspected.

Corrections: A3/A4 are V1, not V2. Historical normal-approximation intervals replaced with Student-t intervals in regenerated figures. V2 plot includes all four conditions and no-clipping. Failed V2 runs are zero in the regenerated benchmark, with observed-score and failure-zero means both retained in statistics. Legacy tuning reused official test data. Zero gain retains moving bounds. Gain/reference sweep has three seeds per cell, never six pooled replicates. Unverified ZClip/AdaGC reproductions are excluded from manuscript efficacy claims.

Limitations: historical A3/A4 raw tables lack failure flags and complete per-run identity/configuration records. CSV consistency does not certify scientific completeness. Existing summary statistics were not overwritten. Mechanism trajectories show between-seed SD, not confidence intervals or causal evidence. Full server traces/checkpoints and ongoing confirmation cannot be audited from the committed files.

Files: inventory.csv (source hashes, duplicates, bounds and failures), recomputed_statistics.csv (seed-level recomputation), json_crosschecks.csv (mechanism/perturbation source consistency).

Warnings:
- results\a1_high_lr_pilot\raw_runs.csv: failure flag missing; no-failure claim is not verified
- results\a3_ablation_small\raw_runs.csv: failure flag missing; no-failure claim is not verified
- results\a4_multiseed_small\raw_runs.csv: failure flag missing; no-failure claim is not verified

Screening recheck: 156 unique run IDs, 2 recorded failures, 14 adverse events. All available raw-table duplicate and accuracy-bound checks passed. Audited raw-table rows total 733 (not necessarily distinct runs across campaigns). No evidence of numeric mean mismatch in A3/A4; the concerns are uncertainty calculation, labels, missing failure metadata and test-data reuse.
