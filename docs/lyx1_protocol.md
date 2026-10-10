# develop/lyx1: full-data confirmation and mechanism study

Frozen protocol: `configs/lyx1/protocol.json`. This study starts from `main`
commit `db933c8`. It is independent of the earlier `develop/lyx` campaign.
The engine's four source files and every configuration are hashed; an existing
run is accepted only if its complete identity matches. The frozen manifest is
written before training. Pilot seed 1100 checks implementation and throughput
only; it cannot select a learning rate, method, or hyperparameter.

## Questions and decision rules

1. Does the frozen DAGC-V2 configuration transfer from short, small-subset
   FashionMNIST experiments to longer training on all available training data?
2. Does DAGC outperform a tuned fixed threshold, a lagged historical-median
   scale rule, or a threshold schedule constructed on calibration seeds?
3. Is consecutive-gradient alignment necessary for the observed effect?

A method-wide superiority claim requires positive paired accuracy differences
against all three primary comparators in all three conditions, with Holm
correction over the nine two-sided sign-flip comparisons. Report effect sizes
and paired Student-t 95% intervals regardless of significance. A null result
is not equivalence. Descriptive equivalence uses a prespecified +/-0.5 percentage
point margin and requires the paired 90% interval to lie wholly inside it;
this is not a familywise equivalence claim. Twelve training seeds are the
sampling units, not minibatches, epochs, or hyperparameter configurations.

## Data and training

- FashionMNIST: fixed class-stratified split of 55,000 training and 5,000
  validation images; official 10,000-image test set for confirmation only.
- CIFAR-10: fixed class-stratified split of 45,000 training and 5,000 validation
  images; official 10,000-image test set for confirmation only.
- Split seed 1729, independent of training seeds. No validation examples enter
  training; no official test scores enter tuning. Confirmation retains the
  same training partition to preserve calibration/replay timing.
- FashionMNIST uses the existing SmallCNN for 30 epochs at base LR 0.8.
  CIFAR-10 uses a 20-layer residual network, widths 16/32/64, option-A
  shortcuts, batch normalization, for 60 epochs at base LR 0.1 or 0.4.
  These are fixed lower/higher rates; instability is measured, not presumed.
- SGD, momentum 0.9, batch 256, no dropped final batch. Weight decay is 1e-4
  on FashionMNIST and 5e-4 on CIFAR-10. Cosine learning-rate decay reaches
  1% of the initial rate at the last step. No warmup or mixed precision.
- CIFAR augmentation: zero padding by four, uniformly sampled 32x32 crop,
  horizontal flip with probability 0.5. Fixed normalization constants are
  in the engine. FashionMNIST has normalization only.
- All policies use identical initial weights, minibatches, and augmentations
  for a given seed. Private random generators prevent logging or policy state
  from changing the input stream. CUDA deterministic algorithms are enabled;
  TF32 is disabled. Evaluation occurs only at the final epoch.
- Nonfinite loss/gradient or loss above 1e6 stops the run. A nonfinite final
  evaluation also fails. Failed runs receive zero accuracy and remain in every
  aggregate. Finite chance-level runs remain with their observed accuracy.
  There is no best-checkpoint selection or rerunning unfavorable seeds.

## Policies and calibration

Each of fixed, AGC, AutoClip, and median receives three candidates by three
calibration seeds (2100--2102), nine runs per condition and policy. Select the
highest mean final validation accuracy; grid order breaks exact ties. Search
ranges are explicit, finite, and do not imply exhaustive tuning. DAGC retains
its historical setting and thus has a different development history; equal
new baseline budgets do not erase that asymmetry.

| Policy | Candidate values / definition |
|---|---|
| Fixed global clipping | 0.05, 0.5, 5.0 |
| Unit-wise AGC | coefficient 0.005, 0.03, 0.18; epsilon 1e-3; all parameters |
| AutoClip | history percentile 1, 10, 50; includes current norm |
| Lagged median scale | coefficient 0.025, 0.1, 0.4; last 200 raw norms; initial threshold 0.5 |
| DAGC-V2 | initial 0.5, gain 0.05, exposure/alignment EMA 0.9, relaxation 0.3, direction weight 3, reference exposure 0.1, median bounds 0.1/10 |
| No alignment | same as DAGC with direction weight zero; no retuning |
| Replay | arithmetic mean of three full DAGC calibration threshold traces, indexed by step; no feedback on confirmation trajectories |
| None | unmodified gradients |

The lagged median uses the lower median for even-length histories, matching
DAGC's PyTorch median convention. It clips using past information only. AGC
uses output-unit norms for convolution and linear weights and a single norm
for vectors. AutoClip uses the authors' all-history percentile convention.
These are clipping rules applied to SGD in this study, not replications of
NFNet or audio-source-separation end-task results.

References checked for implementation conventions:
[official AGC](https://github.com/google-deepmind/deepmind-research/blob/master/nfnets/agc_optax.py),
[official AutoClip](https://github.com/pseeth/autoclip/blob/master/autoclip.py).
The legacy ZClip and AdaGC implementations are not included in this new
comparison; their provenance and exact correspondence require a separate audit.

## Budget and outputs

117 calibration runs (39 per condition), followed by 288 confirmation runs
(8 policies x 12 seeds x 3 conditions), total 405. Full-data pilots are additional
and reported separately. No adaptive stopping based on the primary result.
The campaign can resume completed runs and epoch checkpoints without changing
its protocol. A failed infrastructure process blocks completion; a scientific
failure is recorded as an outcome.

Retain the protocol, source digest, calibration selection, every seed-level
result, training curves, threshold/gradient traces, environment details, and
final tables. Raw checkpoints stay on the server; compact evidence is exported
into `results/lyx1/` for the paper. Completion means all 405 outcomes are
accounted for and the paper tables can be regenerated from them.

## Audit of earlier evidence

The existing manuscript used separate *training seeds*, but calibration and
method development accessed subsets of the same official test set. This is
not a clean train/validation/test separation. Those results must be labeled
exploratory. Moreover, pooling two controller gains and three seeds into six
independent replicates confounds settings with replication. Retain seed-level
results, stratify by gain, and do not use that pooled interval as confirmatory
uncertainty. Earlier evidence that replay nearly matches DAGC must remain
visible, including if the new experiment yields an unfavorable result.

## Sample-size amendment before confirmation

Version 1 specified eight confirmation seeds. Before any confirmation run,
version 2 increases this to twelve seeds (3100--3111). With eight pairs, the
smallest two-sided exact sign-flip p-value is 2/256, above 0.05/9; hence the
original nine-comparison Holm criterion could never reject. Twelve pairs
allow a minimum p-value of 2/4096. The server audit records zero confirmation
results and calibration-only running processes at amendment. No method,
hyperparameter, condition, outcome definition or calibration seed changed.
Identical completed calibration runs are carried forward after source/config
identity verification; no unfavorable result is rerun or discarded.
