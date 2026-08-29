# Reproducibility

## Environment

- Python >= 3.10, PyTorch >= 2.1 (see requirements.txt)
- One GPU is sufficient for diagnostic-scale experiments (A0-A5);
  A6-A8 benefit from 2-4 GPUs; A9 requires a larger cluster.

## Seeds and statistics

- Main experiments: at least 5 independent random seeds (trajectory-selection
  experiments: at least 10 paired seeds).
- Report mean +/- 95% CI; bootstrap CIs and paired comparisons where
  appropriate; correct for multiple comparisons when many hypotheses are
  tested.
- Training iterations from a single run are NOT independent samples.

## Paired-run protocol (A6/A7)

Two runs share: initialization, minibatch sequence, data augmentation
randomness, optimizer, learning-rate schedule, and all other stochastic
seeds. Only the clipping policy differs.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml   # A2
python -m src.train clipping=fixed  logging.log_every=1                 # A4/A5
python -m src.train clipping=dagc   logging.log_every=1                 # A4/A5
```

See `scripts/reproduce.sh` (Linux) or `scripts/reproduce.ps1` (Windows).
