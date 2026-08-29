# Reproduce the load-bearing experiments (A2, A4, A5, A6, A7) on Windows.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml
python -m src.train clipping=fixed logging.log_every=1
python -m src.train clipping=dagc logging.log_every=1
Write-Host "See docs/experiments_A0_A9.md for A6/A7 paired-run instructions."
