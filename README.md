# Dynamics-Aware Gradient Clipping (DAGC)

**Research project:** Dynamics-aware gradient clipping for stable and efficient
neural-network training.

Target venue: ICML 2027.

This repository contains the experimental code base for the machine-learning paper
planned in the research outlines:

- English outline: `../ai_paper_research_outline/ai_paper_research_outline.pdf`
- Chinese outline: `../人工智能论文研究大纲/`

## Scientific Goal

Gradient clipping is treated as a state-dependent controller. The project:

1. measures dynamical diagnostics during real neural-network training;
2. establishes their relationship to optimization instability and performance;
3. builds a computationally cheap **dynamics-aware adaptive gradient clipping (DAGC)** method;
4. validates it across architectures, datasets, optimizers, and learning rates.

**Central falsifiable thesis:** target-exposure feedback, together with
gradient-direction dynamics, improves the accuracy--stability trade-off over
calibrated fixed clipping and standard adaptive-clipping baselines in unstable
training regimes.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/          # Hydra-style YAML configurations
│   ├── datasets/
│   ├── models/
│   ├── optimizers/
│   ├── clipping/
│   └── sweeps/
├── src/              # Core implementation (train, clipping, dynamics, ...)
├── analysis/         # Post-hoc analysis scripts (phase diagrams, statistics, ...)
├── scripts/          # Entry points and reproducibility scripts
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks (quick_experiment.ipynb: train + visualize)
├── docs/             # Instrumentation, experiment protocol, reproducibility
├── data/             # Datasets (gitignored)
└── results/          # Raw results, checkpoints, figures, tables (gitignored)
```

## Quick Start

```bash
pip install -r requirements.txt

# Option 1: interactive notebook (training + visualization)
jupyter notebook notebooks/quick_experiment.ipynb

# Option 2: command line
python scripts/quick_experiment.py --epochs 10
```

## Reproduce the diagnostic benchmark

The benchmark uses a deterministic 5,000-example training subset and a
2,000-example test subset for fast, diagnostic-scale comparisons. It evaluates
five seeds per condition.

```bash
# DAGC component ablation (A3)
python scripts/run_algorithm_benchmarks.py --suite a3

# Multi-seed robustness benchmark (A4)
python scripts/run_algorithm_benchmarks.py --suite a4

# Create accuracy figures from the completed CSV files
python analysis/plot_algorithm_benchmarks.py
```

Results are saved in `results/a3_ablation_small/` and
`results/a4_multiseed_small/`. The runner checkpoints completed rows in
`raw_runs.csv` and resumes safely after an interruption.

## DAGC-V2 held-out benchmark

DAGC-V2 corrects the original controller's one-sided exposure rule. The
original version merely slows threshold relaxation when clipping exposure is
high. V2 regulates exposure around a target: excess exposure directly tightens
the threshold. V2 parameters were selected once on separate development seeds
and then frozen.

The following held-out results use eight paired evaluation seeds per policy;
the fixed threshold is calibrated on separate seeds in every condition.

| Condition | Fixed | AGC | DAGC-V1 | DAGC-V2 | V2 vs. fixed |
|---|---:|---:|---:|---:|---:|
| FashionMNIST / CNN, SGD LR 0.4 | 86.76 ± 0.40 | 87.02 ± 0.70 | 83.18 ± 0.81 | **87.58 ± 0.63** | +0.82 pp (6/8) |
| FashionMNIST / CNN, SGD LR 0.8 | 84.38 ± 1.19 | 84.95 ± 0.63 | 78.35 ± 4.67 | **86.56 ± 0.61** | +2.18 pp (7/8) |
| FashionMNIST / MLP, SGD LR 0.8 | 82.59 ± 0.83 | 82.29 ± 1.12 | 77.51 ± 1.99 | **84.12 ± 0.52** | +1.53 pp (8/8) |
| MNIST / CNN, SGD LR 0.8 | 97.23 ± 0.43 | 96.84 ± 0.36 | 85.19 ± 7.41 | **98.39 ± 0.15** | +1.16 pp (8/8) |

Values are test accuracy (%) with 95% confidence intervals. No clipping
diverges in 75--100% of these high-learning-rate runs. These are
diagnostic-scale MNIST-family experiments, not a large-scale generalization
claim.

```bash
# Reproduce the frozen V2 CNN evaluation at LR 0.8 (seeds 300--307)
python scripts/run_dagc_v2_evaluation.py --lr 0.8 --seeds 300 301 302 303 304 305 306 307

# Cross-learning-rate check, FashionMNIST CNN
python scripts/run_dagc_v2_evaluation.py --lr 0.4 --seeds 308 309 310 311 312 313 314 315

# Independent architecture and dataset evaluations
python scripts/run_mlp_evaluation.py
python scripts/run_mnist_evaluation.py

# Produce the held-out results figure and CSV summary
python analysis/plot_dagc_v2_evaluations.py
```

The raw CSVs are stored under `results/*_evaluation*/`; the figure and concise
summary are in `results/figures/dagc_v2/`. Every runner checkpoints completed
seed-policy pairs and resumes safely after interruption.

## Earlier diagnostic result

In the completed 5-seed, low-learning-rate diagnostic benchmark, DAGC's full
controller outperformed its own ablations on FashionMNIST with a small CNN and
SGD. It did **not** consistently outperform unclipped training across the
otherwise stable A4 conditions. This is expected for a clipping method: its
central claim must be evaluated in high-learning-rate or otherwise unstable
regimes where stabilizing updates is necessary.

The updated evaluation protocol and remaining scale-up work are in
`docs/next_experiments.md`.

## Experimental Program

| ID | Purpose |
|----|---------|
| A0 | Instrumentation and reproducibility validation |
| A1 | Learning-rate--clipping phase diagram |
| A2 | Clipping-exposure and switching diagnostics |
| A3 | DAGC ablations and controller-signal analysis |
| A4 | Robustness across architectures, optimizers, and datasets |
| A5 | Moderate-scale validation |

See `docs/experiments_A0_A9.md` for details.

## Related Documents

- Research outline (English): `../ai_paper_research_outline/`
- Research outline (Chinese): `../人工智能论文研究大纲/`
- Companion mathematics paper: `../../Paper_04/`

## License

MIT (see LICENSE).
