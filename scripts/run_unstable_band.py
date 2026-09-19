"""Calibrated high-learning-rate comparison: DAGC vs clipped baselines.

This implements the Stage 1 protocol in ``docs/next_experiments.md``.

Design decisions that matter for interpretation:

1. Per-method tuning on held-out calibration seeds, frozen before evaluation.
   Fixed clipping is tuned over a threshold grid and DAGC over
   ``(init_c, gamma)``; AGC, ZClip, and AdaGC keep their published
   hyperparameters and are labelled ``@default``, so a weak showing from them
   is a statement about their out-of-the-box behaviour and not the method.

2. Seed-wise pairing.  ``train_run`` reseeds everything from the run seed, so
   two policies run with the same seed see the same initialization, the same
   batch order, and therefore the same gradient sequence until their first
   parameter update diverges.  Every reported difference is therefore paired
   across seeds, which is far more sensitive than comparing marginal intervals.

3. Divergence is a primary outcome, not a nuisance.  A run that leaves a
   plausible loss range, or jumps by more than the configured factor in one
   step, stops early and is recorded with the step at which it failed.
   Accuracy for diverged runs is kept as measured (it is genuinely near chance)
   but is always reported alongside the divergence rate.

4. Warm-up lengths are explicit.  ZClip (25 steps) and AdaGC (100 steps) spend
   their entire warm-up inside a short diagnostic run, which leaves them no
   chance to act at all.  Both instead take a warm-up length from the command
   line -- the same treatment for both, so neither is favoured -- and the
   values used are written into every row of the output.

Usage:
    python scripts/run_unstable_band.py --stage all --epochs 15 --train-size 5000
    python scripts/run_unstable_band.py --stage evaluate   # resumes; skips done runs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.train import train_run

METRICS = (
    "test_acc", "mean_loss", "f_clip", "i_clip", "n_switch",
    "mean_c1", "mean_c2", "diverged", "divergence_step", "steps_completed",
)

FAMILIES = ("none", "fixed", "agc", "dagc", "zclip", "adagc")


def default_policies(warmup_zclip: int, warmup_adagc: int) -> dict[str, dict]:
    """Polices held at published defaults, except the two warm-up lengths.

    ``alpha``/``z_thres`` (ZClip) and ``lambda_rel``/``lambda_abs`` (AdaGC) are
    the values published by their respective papers and are not tuned.  The
    warm-up lengths come from the caller because both methods would otherwise
    spend a short run entirely inside warm-up and never act.
    """
    return {
        "none": {"name": "none"},
        "agc": {"name": "agc", "clip_value": 0.01, "eps": 1e-3},
        "zclip": {"name": "zclip", "alpha": 0.97, "z_thres": 2.5, "warmup": warmup_zclip},
        "adagc": {"name": "adagc", "lambda_rel": 0.5, "beta_adagc": 0.999,
                  "t_start": warmup_adagc, "lambda_abs": 1.0},
    }

FIXED_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
DAGC_GRID = ((1.0, 0.05), (1.0, 0.15), (0.5, 0.05), (2.0, 0.15))


def candidates() -> list[tuple[str, str, dict]]:
    """(family, label, clipping config) candidates for the calibration stage."""
    out: list[tuple[str, str, dict]] = []
    for value in FIXED_GRID:
        out.append(("fixed", f"fixed@{value:g}", {"name": "fixed", "threshold": value}))
    for init_c, gamma in DAGC_GRID:
        out.append((
            "dagc",
            f"dagc@{init_c:g},{gamma:g}",
            {"name": "dagc", "gamma": gamma, "beta": 0.9, "relax": 0.3,
             "osc_weight": 3.0, "init_c": init_c},
        ))
    return out


def base_config(args, lr: float, seed: int, clipping: dict) -> dict:
    return {
        "dataset": "fashion_mnist",
        "model": "small_cnn",
        "data_dir": str(ROOT / "data"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "optimizer": "sgd",
        "seed": seed,
        "device": args.device,
        "num_workers": 0,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "projection_dim": 8,
        "clipping": clipping,
        "divergence_loss": args.divergence_loss,
        "divergence_jump": args.divergence_jump,
    }


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key_of(row: dict) -> tuple:
    """Unique run key, including the hyperparameter candidate label.

    Calibration evaluates several labels within the same method family; omitting
    the label incorrectly marks all but one candidate as already complete when
    resuming a partially finished calibration.
    """
    return (row["family"], row["label"], float(row["lr"]), int(row["seed"]))


def deduplicate(rows: list[dict]) -> list[dict]:
    """Keep the most recent row for an interrupted/restarted run key."""
    latest: dict[tuple, dict] = {}
    for row in rows:
        latest[key_of(row)] = row
    return list(latest.values())


def run_one(args, family: str, label: str, lr: float, seed: int, clipping: dict) -> dict:
    result = train_run(base_config(args, lr, seed, clipping))
    row = {
        "family": family,
        "label": label,
        "lr": lr,
        "seed": seed,
        "dataset": "fashion_mnist",
        "model": "small_cnn",
        "optimizer": "sgd",
        "epochs": args.epochs,
        "train_size": args.train_size or 60000,
        "test_size": args.test_size or 10000,
        "warmup_zclip": args.warmup_zclip,
        "warmup_adagc": args.warmup_adagc,
        "diverged": int(bool(result["diverged"])),
        "divergence_step": result["divergence_step"] if result["divergence_step"] is not None else "",
        "divergence_reason": result["divergence_reason"] or "",
    }
    for metric in ("test_acc", "mean_loss", "f_clip", "i_clip", "n_switch",
                   "mean_c1", "mean_c2", "steps_completed"):
        row[metric] = result[metric]
    return row


def stage_calibrate(args) -> list[dict]:
    raw_path = args.out_dir / "calibration_raw.csv"
    rows = deduplicate(read_rows(raw_path))
    done = {key_of(r) for r in rows}
    todo = [
        (family, label, clipping, lr, seed)
        for lr in args.lrs
        for seed in args.calib_seeds
        for family, label, clipping in candidates()
        if (family, label, lr, seed) not in done
    ]
    print(f"[calibrate] {len(todo)} runs to go ({len(done)} already complete)", flush=True)
    for index, (family, label, clipping, lr, seed) in enumerate(todo, start=1):
        row = run_one(args, family, label, lr, seed, clipping)
        rows.append(row)
        write_rows(raw_path, rows)
        print(f"[calibrate {index}/{len(todo)}] lr={lr:g} {label} seed={seed}: "
              f"acc={row['test_acc']:.4f} diverged={row['diverged']}", flush=True)
    return select_best(rows, args)


def select_best(rows: list[dict], args) -> list[dict]:
    """Pick one config per family and learning rate: fewest divergences, then accuracy."""
    selection: list[dict] = []
    for lr in args.lrs:
        for family in ("fixed", "dagc"):
            subset = [r for r in rows if r["family"] == family and float(r["lr"]) == float(lr)]
            if not subset:
                continue
            by_label: dict[str, list[dict]] = {}
            for row in subset:
                by_label.setdefault(row["label"], []).append(row)
            scored = []
            for label, group in by_label.items():
                accs = np.array([float(g["test_acc"]) for g in group])
                div = np.array([float(g["diverged"]) for g in group])
                scored.append((float(div.mean()), -float(accs.mean()), label, group))
            scored.sort()
            div_rate, neg_acc, label, group = scored[0]
            detail = " ".join(
                f"{lb}: acc={-na:.4f} div={dr:.2f}" for dr, na, lb, _ in scored
            )
            selection.append({
                "family": family, "lr": float(lr), "label": label,
                "div_rate": div_rate, "acc": -neg_acc, "n": len(group),
                "candidates": detail,
            })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "calibration_selection.json").open("w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print("\n[calibrate] selection (fewest divergences, then accuracy):", flush=True)
    for item in selection:
        print(f"  lr={item['lr']:g} {item['family']:5} -> {item['label']:16} "
              f"acc={item['acc']:.4f} div={item['div_rate']:.2f}", flush=True)
    return rows


def load_selection(args) -> list[dict]:
    path = args.out_dir / "calibration_selection.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --stage calibrate first")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve(label: str, family: str) -> dict:
    if family == "fixed":
        return {"name": "fixed", "threshold": float(label.split("@")[1])}
    if family == "dagc":
        init_c, gamma = (float(x) for x in label.split("@")[1].split(","))
        return {"name": "dagc", "gamma": gamma, "beta": 0.9, "relax": 0.3,
                "osc_weight": 3.0, "init_c": init_c}
    raise ValueError(f"unknown tuned family {family}")


def stage_evaluate(args) -> None:
    selection = load_selection(args)
    tuned = {(item["family"], float(item["lr"])): item["label"] for item in selection}
    policies = default_policies(args.warmup_zclip, args.warmup_adagc)
    raw_path = args.out_dir / "eval_raw.csv"
    rows = deduplicate(read_rows(raw_path))
    done = {key_of(r) for r in rows}

    todo = []
    for lr in args.lrs:
        for family in ("fixed", "dagc"):
            label = tuned.get((family, float(lr)))
            if label is None:
                raise SystemExit(f"no calibrated {family} for lr={lr}")
            todo.append((family, label, resolve(label, family), lr))
        for family, clipping in policies.items():
            todo.append((family, f"{family}@default", dict(clipping), lr))

    pending = [
        (family, label, clip, lr, seed)
        for family, label, clip, lr in todo
        for seed in args.eval_seeds
        if (family, label, lr, seed) not in done
    ]
    print(f"[evaluate] {len(pending)} runs to go ({len(done)} already complete)", flush=True)
    for index, (family, label, clipping, lr, seed) in enumerate(pending, start=1):
        row = run_one(args, family, label, lr, seed, clipping)
        rows.append(row)
        write_rows(raw_path, rows)
        print(f"[evaluate {index}/{len(pending)}] lr={lr:g} {family:6} seed={seed}: "
              f"acc={row['test_acc']:.4f} div={row['diverged']}", flush=True)


def summarize(args) -> None:
    """Per-policy means plus the paired contrasts the paper actually needs."""
    rows = read_rows(args.out_dir / "eval_raw.csv")
    if not rows:
        return
    by_cell: dict[tuple[float, int], dict[str, dict]] = {}
    for row in rows:
        by_cell.setdefault((float(row["lr"]), int(row["seed"])), {})[row["family"]] = row

    summary: list[dict] = []
    for lr in args.lrs:
        seeds = sorted(s for (l, s) in by_cell if l == float(lr))
        acc = {f: [] for f in FAMILIES}
        div = {f: [] for f in FAMILIES}
        for seed in seeds:
            cell = by_cell[(float(lr), seed)]
            for family in FAMILIES:
                if family in cell:
                    acc[family].append(float(cell[family]["test_acc"]))
                    div[family].append(float(cell[family]["diverged"]))
        for family in FAMILIES:
            if not acc[family]:
                continue
            values = np.array(acc[family])
            entry = {
                "lr": lr, "family": family, "n": len(values),
                "acc_mean": float(values.mean()),
                "acc_ci95": float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else float("nan"),
                "div_rate": float(np.mean(div[family])),
            }
            for other in FAMILIES:
                if other == family or other not in by_cell.get((float(lr), seeds[0]), {}):
                    continue
                if len(acc[other]) != len(values):
                    continue
                diff = values - np.array(acc[other])
                entry[f"paired_vs_{other}_mean"] = float(diff.mean())
                entry[f"paired_vs_{other}_ci95"] = (
                    float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))
                    if len(diff) > 1 else float("nan")
                )
                entry[f"paired_vs_{other}_wins"] = int((diff > 0).sum())
            summary.append(entry)
    write_rows(args.out_dir / "eval_summary.csv", summary)

    print("\n=== per-policy results ===", flush=True)
    print(f"{'lr':>5} {'policy':7} {'n':>3} {'acc %':>18} {'div rate':>9}", flush=True)
    for lr in args.lrs:
        for family in FAMILIES:
            row = next((r for r in summary
                        if r["family"] == family and float(r["lr"]) == float(lr)), None)
            if row is None:
                continue
            print(f"{lr:5g} {family:7} {row['n']:3d} "
                  f"{row['acc_mean']*100:9.2f} +/- {row['acc_ci95']*100:5.2f} "
                  f"{row['div_rate']:9.2f}", flush=True)

    print("\n=== paired differences (same seeds, so paired) ===", flush=True)
    for lr in args.lrs:
        dagc = next((r for r in summary
                     if r["family"] == "dagc" and float(r["lr"]) == float(lr)), None)
        if dagc is None:
            continue
        for other in ("fixed", "none", "agc", "zclip", "adagc"):
            key = f"paired_vs_{other}_mean"
            if key in dagc:
                print(f"  lr={lr:g}: DAGC - {other:6} = {dagc[key]*100:+6.2f} pts "
                      f"(95% CI {dagc[f'paired_vs_{other}_ci95']*100:5.2f}, "
                      f"DAGC wins {dagc[f'paired_vs_{other}_wins']}/{dagc['n']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibrate", "evaluate", "all"), default="all")
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.4, 0.8])
    parser.add_argument("--calib-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--warmup-zclip", type=int, default=25)
    parser.add_argument("--warmup-adagc", type=int, default=50)
    parser.add_argument("--divergence-loss", type=float, default=100.0)
    parser.add_argument("--divergence-jump", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "unstable_band")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("calibrate", "all"):
        stage_calibrate(args)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(args)
        summarize(args)
    print(f"\nsaved under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
