# develop/lyx1 execution status

State: confirmation running; all 117 calibration runs have completed. The final numerical report and revised PDF are pending the
complete experiment; this file does not claim the paper or training is finished.

- Branch starts from `main` commit `db933c8`.
- Training implementation and initial protocol: `233bb79`.
- Twelve-seed amendment made before any confirmation: `57f5565`.
- Fixed protocol: `configs/lyx1/protocol.json` and `docs/lyx1_protocol.md`.
- Planned totals: 117 calibration runs and 288 confirmation runs, 405 outcomes.
- Calibration seeds: 2100--2102. Confirmation seeds: 3100--3111.
- Three full-data, two-epoch implementation pilots on seed 1100 completed.
- Thirteen implementation/statistics tests and three evidence-export checks
  passed. Resume tests use synthetic data.

The first confirmation snapshot accounts for 24 completed outcomes, including
eight no-clipping runs that met the predefined failure rule; process errors
are zero. These failures remain in all final aggregates.

Eight GPUs run independent jobs. Calibration evaluates only the fixed
validation partition. The scheduler freezes all selected thresholds and replay
schedules before starting confirmation. The existing 20-minute follow-up
checks progress and reports meaningful changes; it should be paused after the
paper, evidence, and branch are finalized.

The original eight-seed execution was stopped during calibration to correct
the minimum attainable p-value under nine-test Holm correction. Its manifest,
outputs, and amendment audit are retained. Unchanged calibration outcomes were
verified and copied to a new execution directory. No confirmation outcome
existed at the time of this amendment.

The result export will verify every source/config identity, dataset split,
condition setting, seed, completed epoch, selected validation winner, replay
trace, and recorded failure. It creates an immutable compact evidence directory
for regeneration of all paper results. Full checkpoints and diagnostic traces
remain in the experiment archive.

Current writing work corrects three issues in the original manuscript: test
scores used during development were described as independent held-out evidence;
different controller gains were pooled as independent seeds; and the
zero-gain ablation was labeled as a fully frozen threshold despite moving
scale bounds. Prior replay and learning-rate-perturbation null results remain
visible. New conclusions will follow the complete audited data.
