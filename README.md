# Dynamics-Aware Gradient Clipping (DAGC)

**Research project:** Dynamics-aware gradient clipping for stable and efficient
neural-network training.

Target venues: ICML / NeurIPS / ICLR.

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

**Central falsifiable thesis:** using clipping exposure and gradient-direction
dynamics to adapt the threshold improves the accuracy--stability trade-off over
fixed clipping and standard adaptive-clipping baselines.

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

## Current diagnostic result

In the completed 5-seed, low-learning-rate diagnostic benchmark, DAGC's full
controller outperformed its own ablations on FashionMNIST with a small CNN and
SGD. It did **not** consistently outperform unclipped training across the
otherwise stable A4 conditions. This is expected for a clipping method: its
central claim must be evaluated in high-learning-rate or otherwise unstable
regimes where stabilizing updates is necessary.

The next evaluation protocol is in `docs/next_experiments.md`.

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
