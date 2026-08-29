#!/usr/bin/env bash
# Reproduce the load-bearing experiments (A2, A4, A5, A6, A7).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml
# A4/A5: single runs with full dynamical logging
python -m src.train clipping=fixed logging.log_every=1
python -m src.train clipping=dagc logging.log_every=1
# A6/A7: paired runs (see docs/experiments_A0_A9.md for the pairing protocol)
echo "See docs/experiments_A0_A9.md for A6/A7 paired-run instructions."
