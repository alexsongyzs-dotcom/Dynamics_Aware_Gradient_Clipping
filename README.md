# Dynamics-Aware Gradient Clipping (DAGC)

**Research project:** Dynamics-aware gradient clipping for neural network training:
stability, switching dynamics, and trajectory control.

Target venues: ICML / NeurIPS / ICLR.

This repository contains the experimental code base for the machine-learning paper
planned in the research outlines:

- English outline: `../ai_paper_research_outline/ai_paper_research_outline.pdf`
- Chinese outline: `../人工智能论文研究大纲/`

## Scientific Goal

Gradient clipping is treated not as a mere magnitude safeguard but as a
**state-dependent switching controller** that interacts with local stability,
oscillatory training dynamics, and trajectory selection. The project:

1. measures dynamical diagnostics during real neural-network training;
2. establishes their relationship to optimization instability and trajectory selection;
3. builds a computationally cheap **dynamics-aware adaptive gradient clipping (DAGC)** method;
4. validates it across architectures, datasets, optimizers, and learning rates.

**Central falsifiable thesis:** the temporal organization of clipping exposure
(when clipping activates, how frequently, for how long) causally affects
trajectory selection and the final solution, through mechanisms distinct from
the mere reduction of update magnitude.

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
├── src/              # Core implementation (train, clipping, dynamics, hessian, ...)
├── analysis/         # Post-hoc analysis scripts (phase diagrams, statistics, ...)
├── scripts/          # Entry points and reproducibility scripts
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks (quick_experiment.ipynb: train + visualize)
├── docs/             # Instrumentation, experiment protocol (A0-A9), reproducibility
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

## Experimental Program (A0-A9)

| ID | Purpose |
|----|---------|
| A0 | Instrumentation and reproducibility validation |
| A1 | Local stability threshold in neural networks |
| A2 | Learning-rate--clipping phase diagram |
| A3 | Clipping exposure statistics |
| A4 | Switching-event dynamics |
| A5 | Oscillatory / period-2-like dynamics |
| A6 | Controlled trajectory selection |
| A7 | Matched-update causal controls |
| A8 | DAGC algorithm evaluation |
| A9 | Large-scale validation |

See `docs/experiments_A0_A9.md` for details.

## Related Documents

- Research outline (English): `../ai_paper_research_outline/`
- Research outline (Chinese): `../人工智能论文研究大纲/`
- Companion mathematics paper: `../../Paper_04/`

## License

MIT (see LICENSE).
