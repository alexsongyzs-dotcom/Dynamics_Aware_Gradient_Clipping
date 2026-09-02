# Planning helper. Full training is intended for the Linux host.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml
Write-Host "Plan generated only. Run scripts/reproduce.sh on Linux after reviewing the gates."
