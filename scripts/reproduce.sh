#!/usr/bin/env bash
# Linux entrypoint for the gated P0-P3 workflow.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${DAGC_EXECUTE:-0}" != "1" ]]; then
  echo "Safety stop: set DAGC_EXECUTE=1 on the Linux training host."
  exit 2
fi

python -m pytest tests
python -m src.train --config configs/experiments/p0_measurement.yaml --verbose
python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml --execute
echo "Inspect G0-G2 before preparing P2 causal branches; see docs/experiments_A0_A9.md."
