# develop/lyx1 execution status

State: **complete**. All 405 planned outcomes have been audited, analyzed, and
incorporated into the English manuscript and PDF. Training finished on
2026-10-10 (Asia/Shanghai). The experiment monitor is paused at delivery.

- Branch starts from `main` commit `db933c8`.
- Training implementation and initial protocol: `233bb79`.
- Twelve-seed amendment before any confirmation: `57f5565`.
- Analysis/export preparation: `050beca`.
- Fixed protocol: `configs/lyx1/protocol.json` and `docs/lyx1_protocol.md`.
- Completed accounting: 117 calibration + 288 confirmation = 405 outcomes.
- Full-budget completions: 397; predefined scientific failures: 8.
- Process errors: 0; unfinished experiment jobs: 0.
- Calibration seeds: 2100--2102; confirmation seeds: 3100--3111.
- Three implementation pilots on seed 1100 were additional and did not tune settings.

Every source/configuration identity, dataset split, condition setting, seed,
completed epoch, selected validation winner, replay trace, and recorded failure
passed the export audit. The full-trace mechanism audit was independently
regenerated from the archived raw data and matched exactly. The training source
digest is retained in `results/lyx1/evidence/training_environment.json`.

All eight scientific failures occurred without clipping on FashionMNIST. They
are retained at zero accuracy; the other four no-clipping runs finished at
10%. All clipped confirmation runs completed. No unfavorable result was rerun
or removed.

The original eight-seed execution was stopped during calibration because its
minimum exact p-value could not pass the nine-test Holm threshold. There were
zero confirmation results at amendment. Verified identical calibration outcomes
were carried forward; both the amendment and original execution remain archived.

The final evidence does not support broad DAGC superiority. DAGC loses to tuned
fixed clipping in all three conditions, while replay satisfies the prescribed
per-comparison equivalence criterion. See [the result report](lyx1_results.md)
and [the manuscript](../paper/main.pdf). Full raw traces and checkpoints remain
in the persistent experiment archive; compact evidence is committed here.
