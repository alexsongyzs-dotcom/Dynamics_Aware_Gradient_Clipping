# Experiment evidence guide

This guide maps the published artifacts to their research roles. It does not move or rename results. File counts are artifact counts, not independent successful runs. Inventory date: 2026-10-10.

## Recommended reading order

1. Read the [manuscript](../../paper/main.pdf) and its experimental and mathematical appendices.
2. Inspect the [audit report](../../paper/audit_20261010/audit_report.md) and [recomputed statistics](../../paper/audit_20261010/recomputed_statistics.csv).
3. Use the evidence map below to locate raw tables and per-run trajectories.
4. Consult the [published-file inventory](published_files.csv) for sizes, SHA-256 hashes, CSV row counts, and available completion/failure fields. Missing fields do not imply successful runs.

## Evidence map

| Research stage | Artifact directory | Role / limitation | Published files in inventory |
|---|---|---|---:|
| Original exploration / V1 | [a1_high_lr_pilot](../../results/a1_high_lr_pilot/) | High-learning-rate development screening; not confirmatory V2 evidence. | 2 |
| Original exploration / V1 | [a3_ablation_small](../../results/a3_ablation_small/) | V1 component ablations, five seeds; failure flags are absent. | 3 |
| Original exploration / V1 | [a4_multiseed_small](../../results/a4_multiseed_small/) | V1 benchmark: eight conditions, four policies, five seeds; not V2 evidence. | 3 |
| V2 development / calibration | [dagc_v2_development](../../results/dagc_v2_development/) | Reference-level and gain sweep; keep different gain settings separate. | 1 |
| V2 development / calibration | [mlp_calibration_lr0.8](../../results/mlp_calibration_lr0.8/) | FashionMNIST MLP calibration. | 1 |
| V2 development / calibration | [mnist_calibration_lr0.8](../../results/mnist_calibration_lr0.8/) | MNIST CNN calibration. | 1 |
| V2 exploratory evaluation | [dagc_v2_evaluation_lr0.4](../../results/dagc_v2_evaluation_lr0.4/) | FashionMNIST CNN, learning rate 0.4. | 1 |
| V2 exploratory evaluation | [dagc_v2_evaluation](../../results/dagc_v2_evaluation/) | FashionMNIST CNN, learning rate 0.8; archived output name. | 1 |
| V2 exploratory evaluation | [mlp_evaluation_lr0.8](../../results/mlp_evaluation_lr0.8/) | FashionMNIST MLP, learning rate 0.8. | 1 |
| V2 exploratory evaluation | [mnist_evaluation_lr0.8](../../results/mnist_evaluation_lr0.8/) | MNIST CNN, learning rate 0.8. | 1 |
| Mechanism controls | [v2_mechanism](../../results/v2_mechanism/) | 64 paired evaluation runs; calibration is separate. Exposure, alignment, zero gain, fixed intensity, and replay controls. | 85 |
| Learning-rate perturbations | [v2_lr_perturbation](../../results/v2_lr_perturbation/) | 48 paired runs: pulse and drop interventions; full V2, no alignment, and calibration-trace replay. | 51 |
| Additional screening / smoke | [unstable_band](../../results/unstable_band/) | Exploratory screening; unverified comparator implementations do not support superiority claims. | 4 |
| Additional screening / smoke | [small_scale_fashionmnist_seed0](../../results/small_scale_fashionmnist_seed0/) | Single-seed smoke run; not multi-seed evidence. | 4 |
| Historical summaries / figures | [figures](../../results/figures/) | Historical plots; prefer audited manuscript figures for interpretation. | 7 |
| Historical summaries / figures | [tables](../../results/tables/) | Historical summary tables. | 1 |

## How to interpret the results

The earlier V2 evaluations use fresh evaluation seeds, but the test distribution was also consulted during development. Treat them as exploratory evaluations, rather than a clean held-out confirmation study. The audited manuscript uses Student-t intervals with seeds as the unit of replication; older plots and summaries may use normal approximations.

The mechanism and learning-rate perturbation experiments do not establish an incremental accuracy benefit from alignment or online feedback over calibration-trace replay. The zero-gain control retains moving bounds, so it does not have a constant clipping threshold. Appendix E supplies conditional mathematical results, not a universal proof that DAGC outperforms replay or no-alignment variants.

The separate [156-run screening archive](../../paper/evidence/) failed its G1 gate. Its [results](../../paper/evidence/server30_results.md) and [verification](../../paper/evidence/server30_verification.json) are screening evidence, not completed V2 confirmation results. Downstream gated stages were not executed.

The confirmation implementation is on `develop/lyx1`, not in the main-branch training engine. At the latest inspected committed snapshot, confirmation was still in progress; no completed audited confirmation export is included here. See the [continuation instructions](../next_experiments_20261010.md). Do not substitute main-branch scripts for that protocol.

## Reproduction and file handling

Run the main-branch scripts from the repository root. Installation and commands are in the [repository README](../../README.md); the [mechanism protocol](../mechanism_experiments.md) describes the controls. Keep a fresh output directory for every changed protocol to prevent cached outcomes from being reused.

Downloaded data, checkpoints, Python/test caches, and unpublished local smoke outputs are excluded from this inventory. No training data or experimental results were relocated during organization. The local inventory originally covered 181 result files; this public inventory includes only already tracked result files with verified matching hashes. It is an organization inventory, not a claim that every listed artifact received a scientific audit.

## Figures and manuscript source

Use the [audited benchmark figure](../../paper/figures/audited_v2_benchmarks.png), [mechanism figure](../../paper/figures/audited_mechanism.png), and [trajectory figure](../../paper/figures/audited_trajectories.png) alongside the corresponding manuscript sections. Earlier figures are retained for provenance. The published [LaTeX source](../../paper/main.tex) and [compilation instructions](../../paper/README.md) accompany the PDF.
