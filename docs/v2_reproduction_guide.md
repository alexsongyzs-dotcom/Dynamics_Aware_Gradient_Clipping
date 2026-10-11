# DAGC-V2: implementation identity and reproduction guide

## What the completed study actually ran

The archived lyx1 engine instantiated `DynamicsAwareClipping(init_c=0.5, gamma=0.05, exposure_target=0.1, osc_weight=3.0)`. Remaining settings match the original V2 defaults: exposure/alignment EMA 0.9, exposure gain 0.3, rolling norm window 200, lower/upper scale 0.1/10. `exposure_target=None` activates V1; 0.1 activates V2. `policy=dagc` is a shared class selector. `ENGINE_VERSION=lyx1-v1` names the training engine, not the controller variant.

Do not attribute the completed negative results to a V1/V2 mix-up without evidence of a different intended implementation. The verification command below checks the reported training-source digest against commit 233bb79 and compares 320 V2 state transitions with the confirmation policy. This is an implementation check, not a retraining experiment. Windows CRLF checkout bytes can differ from the Linux archive digest; the check reports both byte identity and source identity after line-ending normalization. The archived digest is independently verified from Git blobs, and normalized current sources must match.

## Complete source map

- `src/clipping.py`: canonical V1/V2 controller and clipping operators.
- `src/train.py`, `src/data.py`, `src/models.py`, `src/dynamics.py`: original small-subset training and raw-gradient alignment.
- `configs/v2/original_fashion_cnn.json`: every original V2 controller parameter explicit.
- `scripts/run_v2_reproduction.py`: V2-only original-protocol reproduction with per-run diagnostics and output identity guards.
- `scripts/verify_v2_identity.py`: archived provenance and controller identity check.
- `scripts/run_dagc_v2_evaluation.py`, `scripts/run_mlp_evaluation.py`, `scripts/run_mnist_evaluation.py`: historical multi-policy benchmark runners; retain their archived protocols/output paths.
- `src/confirmatory.py`, `src/cifar_resnet.py`, `scripts/run_confirmatory.py`, `configs/lyx1/protocol.json`: completed full-data calibration/confirmation engine.
- `scripts/export_confirmatory.py`, `analysis/confirmatory_report.py`: evidence export and paired statistics.

Clipping uses the current threshold on raw data gradients. Consecutive-gradient alignment uses raw full gradients, not clipped gradients or optimizer updates. The controller then computes the next threshold; SGD momentum and coupled weight decay are applied by the optimizer. The rolling bounds include the current norm when updating the next threshold. The original engine divides by norm plus epsilon; confirmation uses max(norm, epsilon). This numerical convention differs at epsilon scale and must be disclosed, but is not a switch to V1.

## Obtain and verify

Use a fresh checkout; do not overwrite a colleague's active campaign:

```bash
git clone --branch develop/yzs https://github.com/alexsongyzs-dotcom/Dynamics_Aware_Gradient_Clipping.git DAGC_V2_reproduction
cd DAGC_V2_reproduction
python -m pip install -r requirements.txt
python -m pip install -e .
python -m scripts.verify_v2_identity
python -m pytest -q
```

Install a compatible PyTorch/CUDA build on the training machine. No GPU training is triggered by the verification or dry-run commands.

## First reproduce the original setting

```bash
python -m scripts.run_v2_reproduction --output results/v2_original_recheck_lr0p8 --seeds 300 301 302 303 304 305 306 307 --dry-run
python -m scripts.run_v2_reproduction --output results/v2_original_recheck_lr0p8 --seeds 300 301 302 303 304 305 306 307
```

This uses FashionMNIST SmallCNN, 5,000 training/2,000 test images, 15 epochs, batch 256, constant LR 0.8, momentum 0.9 and weight decay 1e-4. Data downloads into `data/` unless `--data-dir` is supplied. Output includes a source/configuration manifest and one JSON per seed with full diagnostics. A changed configuration requires a new output directory. Saved completed runs are reused only within a matching manifest. A CPU smoke run can use a separate copied configuration with fewer images/epochs and `--device cpu`; it is not benchmark evidence.

Historical V2 accuracy was approximately 86.56% on Fashion CNN LR 0.8 (eight seeds); the other original conditions were Fashion CNN LR 0.4: 87.58%, Fashion MLP LR 0.8: 84.12%, MNIST CNN LR 0.8: 98.39%. A reproduction need not be bit-identical across software/hardware; report seed-level differences, failures and environment. These are exploratory test-distribution results, not clean validation-isolated confirmation.

To reproduce another condition, copy the JSON, edit dataset/model/LR and any condition-specific settings to match its archived runner, and choose a new output directory. The V2-only runner does not tune baselines or replace the paired benchmark suite. Do not call its outcome a new superiority comparison.

## Why the completed study can differ

The new study uses 55,000 or 45,000 training images, 30 or 60 epochs, cosine LR decay to 1% of base LR, validation-only accuracy calibration, and a ResNet-20 for CIFAR-10. The old mechanism fixed comparator matched clipping intensity; the new one maximizes validation accuracy. Those protocol changes can alter ranking while keeping the controller V2.

If the original rerun differs materially, compare source hashes, controller arguments, preprocessing, subsets, initialization/batch streams, clipping order and software before interpreting accuracy. Do not retune on confirmation test results or overwrite the completed evidence.

Once original reproduction is understood, write a separate follow-up protocol that changes one factor at a time (schedule, duration, data size, architecture, or baseline selection). Use calibration/validation to select settings and fresh paired seeds for the final comparison. This is a post-confirmation follow-up, not a correction of the completed study. The V2 lower-bound-only ablation can test the observed projection masking; lower-bound occupancy alone cannot establish a causal improvement from changing the bound.

## Optional exact completed-study rerun

The 405-run study is already complete; this command is optional and expensive, not the first diagnostic step. Dataset files must already exist because the confirmation loader uses `download=False`. Set `--gpus` to available GPU IDs, even a single ID:

```bash
python -m scripts.run_confirmatory --root results/lyx1_independent_reproduction --data-dir data --gpus 0
python -m scripts.confirmatory_status --help
python -m scripts.export_confirmatory --help
```

Use a new campaign root. Existing results/lyx1 evidence must remain unchanged. Consult `docs/lyx1_protocol.md` and the export CLI before analysis.
