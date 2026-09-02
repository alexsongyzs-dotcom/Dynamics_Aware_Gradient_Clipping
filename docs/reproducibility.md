# Linux Handoff and Reproducibility

The Windows checkout is a code-only handoff. No training result is claimed
until the following sequence is completed on Linux.

## 1. Environment and data

Use Python 3.10 or newer and a PyTorch build appropriate for the target CUDA
driver. Create an isolated environment, install `requirements.txt`, place the
datasets under `data/`, and leave `dataset.download: false` for controlled
offline runs. Record GPU model, driver, CUDA, PyTorch, torchvision, and package
lock output with the experiment artifacts.

## 2. Validate before training

```bash
python -m pytest tests
python scripts/quick_experiment.py
```

The second command only writes three small P0 configs. Review them, then use
`--execute` for the smoke runs. Do not continue if the unit tests, checkpoint
resume comparison, or P0 measurement identities fail.

## 3. Screening

```bash
python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml
python scripts/run_sweep.py --config configs/sweeps/phase_diagram.yaml --execute
```

The first command materializes and records the full plan; it does not train.
The second is the explicit execution step. Screening uses `num_workers: 0` so
the sampler and augmentation RNG can be reproduced exactly in paired work.

## 4. Causal branches

After a reference run completes, generate branch inputs without training:

```bash
python scripts/prepare_branches.py \
  --reference-steps results/p2_causal_reference/<run>.steps.jsonl \
  --shuffle-seed 1000 \
  --output results/branches/seed-1000
```

Inspect `branch_manifest.json` and verify the sequence hashes and invariants.
Run each generated YAML through `python -m src.train --config <file>` only
after G1 and G2 have passed.

## 5. Safety-gated convenience entrypoint

```bash
DAGC_EXECUTE=1 bash scripts/reproduce.sh
```

Without the environment switch, the script exits before tests or training.
It intentionally stops after P1 and asks for gate review rather than launching
causal or event-timed experiments automatically.

## Statistical identity

Independent seeds are the unit of inference. Paired branches share checkpoint,
seed, minibatch order, and RNG state. Report configuration hashes, checkpoint
paths, sequence hashes, all failures, paired effects with uncertainty, and the
preregistered multiple-comparison correction. Never report step count as the
sample size.
