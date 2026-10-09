# Dynamics-Aware Gradient Clipping

Research code for **Dynamics-Aware Gradient Clipping: Separating Feedback from
Threshold Scheduling**. The `develop/lyx1` study starts from `main` and tests
whether exposure and gradient-alignment feedback adds value beyond tuned
clipping and predetermined threshold schedules.

DAGC-V2 lowers its clipping threshold when clipping is already frequent. This
is positive exposure feedback. A stationary mean-field calculation, including
the exposure EMA, shows that an interior target-exposure equilibrium is
unstable. This statement concerns the controller model; it is not a theorem
about neural-network training convergence.

## Current study

The [frozen protocol](docs/lyx1_protocol.md) specifies 405 runs: 117 validation
calibrations and 288 confirmations, comparing eight policies over twelve paired
seeds in each of three conditions. **The study is complete:** 397 runs finished
their full budget and eight met the predefined failure rule. All 405 outcomes
passed the evidence audit; there were no process errors.

The frozen DAGC-V2 rule **does not outperform validation-tuned fixed clipping**.
Its paired deficits are significant after the prespecified nine-test Holm
correction. The evidence supports a limited mechanism study, not a broad claim
of a better optimizer.

| Condition | DAGC accuracy (%) | Tuned fixed accuracy (%) | DAGC minus fixed (pp), paired 95% interval | Holm p |
|---|---:|---:|---:|---:|
| FashionMNIST, LR 0.8 | 92.41 | 92.60 | -0.18 [-0.29, -0.08] | 0.0176 |
| CIFAR-10, LR 0.1 | 89.50 | 90.67 | -1.17 [-1.32, -1.02] | 0.0044 |
| CIFAR-10, LR 0.4 | 90.87 | 91.16 | -0.29 [-0.41, -0.17] | 0.0103 |

Calibration-trace replay meets the prespecified per-comparison equivalence
criterion (paired 90% interval within +/-0.5 pp) in all three conditions; this
is not a multiplicity-adjusted equivalence claim. DAGC occupies its moving
lower bound on 97.2--99.1% of updates. On CIFAR-10, removing alignment leaves
all twelve paired trajectories unchanged in each condition: negative alignment
signals occur, but lower-bound projection masks their effect.

Read the [English manuscript](paper/main.pdf), [Chinese result report](docs/lyx1_results.md),
[complete numerical analysis](results/lyx1/analysis.json), and
[evidence audit](results/lyx1/evidence/audit.json).

| Condition | Training / validation / test | Model | Epochs | Initial LR |
|---|---|---|---:|---:|
| FashionMNIST | 55,000 / 5,000 / 10,000 | SmallCNN | 30 | 0.8 |
| CIFAR-10 | 45,000 / 5,000 / 10,000 | ResNet-20 | 60 | 0.1 |
| CIFAR-10 | 45,000 / 5,000 / 10,000 | ResNet-20 | 60 | 0.4 |

Policies are no clipping, tuned fixed clipping, tuned unit-wise AGC, tuned
AutoClip, tuned lagged-median scaling, frozen DAGC-V2, DAGC without alignment,
and calibration-trace replay. Four baselines each receive three candidates by
three calibration seeds. Test data is excluded from this campaign's selection;
FashionMNIST test data had been used in earlier method development, which
remains a limitation.

The first execution plan used eight confirmation seeds. Before any confirmation
run, the plan was amended to twelve because an eight-pair exact sign-flip test
cannot pass the first Holm threshold for nine primary comparisons. This
amendment and the unchanged calibration runs carried forward are audited.

## Installation and reproduction

Use Python 3.10 or later and a CUDA-capable PyTorch installation. The reported
training environment uses Python 3.12 and PyTorch 2.8. Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download the official FashionMNIST and CIFAR-10 training and test data into a
local directory first. The confirmation engine uses `download=False` so that
an unavailable dataset fails explicitly instead of changing a running job.
For example:

```python
from torchvision.datasets import FashionMNIST, CIFAR10
for dataset in (FashionMNIST, CIFAR10):
    for train in (True, False):
        dataset('data', train=train, download=True)
```

Launch the fixed study on eight GPUs (or specify fewer GPU IDs):

```bash
python -m scripts.run_confirmatory --root results/lyx1/raw/campaign \
  --data-dir data --gpus 0,1,2,3,4,5,6,7
python -m scripts.confirmatory_status --root results/lyx1/raw/campaign
```

An unchanged invocation reuses matching completed runs and resumes epoch
checkpoints. Changes to source or configuration require a new output root.
Do not launch two schedulers against the same root concurrently. Private random
generators keep initialization, batches, and augmentation paired across
policies. Nonfinite runs remain in accuracy aggregates at zero; finite
chance-level outcomes remain at their observed accuracy.

The committed evidence already regenerates the reported tables and PDF:

```bash
bash scripts/build_paper.sh
```

For a new full training campaign, export to a fresh directory; the exporter
refuses to replace existing evidence:

```bash
python -m scripts.export_confirmatory --root results/lyx1/raw/campaign \
  --output results/lyx1/raw/reproduced-evidence
python -m analysis.confirmatory_report --evidence results/lyx1/raw/reproduced-evidence
python -m analysis.audit_confirmatory_mechanism --root results/lyx1/raw/campaign \
  --output results/lyx1/raw/reproduced-mechanism-audit.json
```

The mechanism audit requires full-resolution raw traces. See the
[evidence inventory](results/lyx1/README.md) for what is included in the compact
archive and what remains with the raw experiment.

The export checks run identities, condition settings, evaluation splits, seed
coverage, epoch continuity, and failure accounting. The analysis produces
seed-level tables, paired Student-t intervals, exact sign-flip tests with Holm
correction, and figures. The paper build requires `pdflatex`/`bibtex` or
`tectonic`. See [paper instructions](paper/README.md).

## Method

```text
alpha = min(1, threshold / max(raw_gradient_norm, epsilon))
clipped_gradient = alpha * raw_gradient
E = beta * E + (1 - beta) * indicator(raw_gradient_norm > threshold)
A = beta_a * A + (1 - beta_a) * consecutive_raw_gradient_alignment
signal = relax * (reference - E) - direction_weight * max(0, -A)
next_threshold = project(threshold * exp(gain * tanh(signal)), bounds)
```

Clipping is applied before the next threshold is computed. The bounds use
0.1 and 10 times a rolling median of the last 200 raw gradient norms, with
numerical floors. The frozen V2 parameters are `init_c=0.5`, `gamma=0.05`,
`beta=beta_a=0.9`, `relax=0.3`, `osc_weight=3.0`, and `exposure_target=0.1`.
The previous full gradient costs memory proportional to the parameter count.
The V1 rule remains available as `exposure_target=None`.

## Earlier exploratory evidence

The earlier 5,000-training/2,000-test, 15-epoch runs used distinct training
seeds for development and evaluation but reused the official test distribution
for selection. They are exploratory evidence, not clean data-held-out
confirmation. Their raw artifacts remain under `results/`.

Recomputed paired 95% t intervals from the eight-seed mechanism study give:

| DAGC minus comparator | Difference (percentage points) | 95% interval |
|---|---:|---:|
| Intensity-matched fixed | +1.18 | [0.62, 1.74] |
| No alignment | +0.16 | [-0.33, 0.66] |
| Calibration-trace replay | +0.21 | [-0.43, 0.86] |
| No exposure | +5.47 | [4.04, 6.90] |

These observations do not establish an incremental benefit from online
alignment or threshold feedback. The learning-rate perturbations also leave
replay competitive. The development reference sweep is now stratified by gain,
with three seeds per setting; gains are not pooled as six independent
replicates. Regenerate these corrections with:

```bash
python -m analysis.confirmatory_report --legacy-only
python -m analysis.plot_tau_sweep
```

Older ZClip and AdaGC variants are excluded from the new comparison because
their local implementations and changed warmup have not been verified as
faithful reproductions. No claim about their published-default performance is
made. Historical protocol documents and plots should be read in that context.

## Tests and layout

```bash
python -m pytest -q
```

The tests include data-split disjointness, independent augmentation randomness,
causal clipping, exact epoch-resume agreement on synthetic data, stale-result
rejection, paired inference, and multiplicity correction.

- `src/confirmatory.py`: independent full-data training engine.
- `src/cifar_resnet.py`: CIFAR ResNet-20.
- `configs/lyx1/protocol.json`: fixed experiment specification.
- `scripts/run_confirmatory.py`: calibration and confirmation orchestration.
- `analysis/confirmatory_report.py`: regenerated paper tables and figures.
- `paper/`: English manuscript, styles, generated tables, and PDF.

## Authors and license

Yuxin Li and Zisong Yuan. Released under the [MIT License](LICENSE).
