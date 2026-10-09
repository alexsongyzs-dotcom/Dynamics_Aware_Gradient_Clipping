"""Execute the original lyx research program on independent GPU workers.

The immutable protocol is materialized before training. G0/G1/G2/G3 gate
decisions are persisted, and a failed scientific gate ends with a report,
never with silently relabeled exploratory results or a forced controller run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configuration import load_config
from src.records import config_hash
from analysis.research_campaign import write_json


def source_digest() -> str:
    digest = hashlib.sha256()
    for folder in ("src", "analysis", "scripts", "configs", "tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


class Campaign:
    def __init__(self, protocol_path: Path, output: Path, data_dir: Path, gpus: list[int]):
        self.protocol_path = protocol_path.resolve()
        self.protocol = yaml.safe_load(self.protocol_path.read_text())
        self.output = output.resolve()
        self.data_dir = data_dir.resolve()
        self.gpus = gpus
        self.output.mkdir(parents=True, exist_ok=True)
        self.base = load_config(self.protocol_path.parent / self.protocol["base_config"])
        self.lock = threading.Lock()
        self.events = self.output / "campaign_events.jsonl"
        self.state = {"stage": "initializing", "running": {}, "completed": 0, "failed_processes": 0}
        identity = dict(protocol=self.protocol, source_digest=source_digest(), data_dir=str(self.data_dir))
        identity_path = self.output / "frozen_protocol.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text())
            if previous != identity:
                raise ValueError("campaign source/protocol/data changed; use a new output directory")
        else:
            write_json(identity_path, identity)
        (self.output / "protocol.yaml").write_text(yaml.safe_dump(self.protocol, sort_keys=False))

    def event(self, kind: str, **values):
        with self.lock:
            item = dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), event=kind, **values)
            with self.events.open("a") as stream:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
            write_json(self.output / "status.json", {**self.state, "last_event": item})
        print(json.dumps(item), flush=True)

    def configuration(self, family: dict, stage: str, label: str, seed: int, epochs: int,
                      lr: float, clipping: dict, *, evaluate_test: bool = False) -> dict:
        cfg = copy.deepcopy(self.base)
        cfg.update(seed=seed, epochs=epochs, device="cuda", batch_size=self.protocol["training"]["batch_size"],
                   cpu_threads=self.protocol["training"]["cpu_threads"], num_workers=0,
                   deterministic=True, pin_memory=True, failure_loss=self.protocol["training"]["failure_loss"])
        run_id = f"{family['name']}__lr-{lr:g}__{label}__seed-{seed}"
        stage_dir = self.output / stage
        cfg.update(run_id=run_id, output_dir=str(stage_dir / "runs"))
        cfg["dataset"] = dict(name=family["dataset"], data_dir=str(self.data_dir), download=False,
                              validation_size=0 if evaluate_test else self.protocol["training"]["validation_size"],
                              split_seed=self.protocol["training"]["split_seed"],
                              evaluation_split="test" if evaluate_test else "validation")
        cfg["model"] = dict(name=family["model"])
        cfg["optimizer"] = dict(name=family["optimizer"], lr=float(lr), momentum=0.9, nesterov=False,
                                weight_decay=0.0005 if family["dataset"].startswith("cifar") else 0.0001,
                                schedule="none" if stage in {"p0", "p1"} else "cosine")
        cfg["clipping"] = {"placement": "pre_moment", **clipping}
        cfg["logging"] = dict(probe_size=self.protocol["training"]["probe_size"],
                              gradient_sketch_size=self.protocol["training"]["gradient_sketch_size"],
                              full_gradient_alignment=True, save_gradient_sketch=True,
                              history_window=100, exposure_beta=0.95,
                              episode=dict(on_margin=0.05, off_margin=0.05, min_dwell_steps=3),
                              hessian_checkpoints=[] if stage in {"p0", "p1", "p3_calibration", "p4_calibration"} else [0, epochs - 1],
                              hessian_power_iterations=12, evaluate_every_epochs=0)
        cfg["checkpointing"] = dict(directory=str(stage_dir / "checkpoints"), every_epochs=0,
                                     steps=[], resume_from=None, save_initial=True, save_latest=True, save_final=True)
        cfg["campaign"] = dict(family=family["name"], stage=stage, label=label)
        return cfg

    def materialize(self, config: dict, path: Path | None = None) -> dict:
        cfg = copy.deepcopy(config)
        path = path or self.output / cfg["campaign"]["stage"] / "configs" / f"{cfg['run_id']}.yaml"
        cfg["campaign"]["config_path"] = str(path.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            prior = yaml.safe_load(path.read_text())
            if prior != cfg:
                raise ValueError(f"immutable run configuration changed: {path}")
        else:
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return cfg

    def run_jobs(self, configs: list[dict], stage: str):
        self.state["stage"] = stage
        self.state["stage_total"] = len(configs)
        self.state["stage_completed"] = 0
        self.event("stage_started", stage=stage, jobs=len(configs))
        jobs = queue.Queue()
        for config in configs:
            jobs.put(config)
        stop = threading.Event()
        errors = []

        def worker(gpu: int):
            lock_path = Path(f"/tmp/dagc-research-gpu-{gpu}.lock")
            with lock_path.open("w") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                while not stop.is_set():
                    try:
                        cfg = jobs.get_nowait()
                    except queue.Empty:
                        return
                    run_id = cfg["run_id"]
                    summary_path = Path(cfg["output_dir"]) / f"{run_id}.summary.json"
                    try:
                        if summary_path.exists():
                            prior = json.loads(summary_path.read_text())
                            if prior["config_hash"] != config_hash(cfg):
                                raise ValueError(f"cached result identity mismatch: {run_id}")
                            if not prior.get("paused", False):
                                with self.lock:
                                    self.state["stage_completed"] += 1
                                self.event("cached_run", run_id=run_id, stage=stage)
                                continue
                        manifest = Path(cfg["output_dir"]) / f"{run_id}.manifest.json"
                        command = [sys.executable, "-m", "src.train", "--config", cfg["campaign"]["config_path"], "--verbose"]
                        if manifest.exists():
                            checkpoint_dir = Path(cfg["checkpointing"]["directory"])
                            latest = checkpoint_dir / f"{run_id}.latest.pt"
                            initial = checkpoint_dir / f"{run_id}.initial.pt"
                            checkpoint = latest if latest.exists() else initial
                            if not checkpoint.exists():
                                raise RuntimeError(f"interrupted run has no checkpoint: {run_id}")
                            command += ["--set", f"checkpointing.resume_from={checkpoint}"]
                        log_path = self.output / stage / "logs" / f"{run_id}-{config_hash(cfg)}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        environment = os.environ.copy()
                        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                                           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
                                           PYTHONUNBUFFERED="1")
                        with self.lock:
                            self.state["running"][str(gpu)] = run_id
                        self.event("run_started", gpu=gpu, stage=stage, run_id=run_id)
                        started = time.monotonic()
                        with log_path.open("a") as log:
                            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                            code = process.wait()
                        if code or not summary_path.exists():
                            raise RuntimeError(f"training process exited {code}; see {log_path}")
                        result = json.loads(summary_path.read_text())
                        if result.get("paused"):
                            raise RuntimeError("unexpected paused result")
                        with self.lock:
                            self.state["completed"] += 1
                            self.state["stage_completed"] += 1
                        self.event("run_completed", gpu=gpu, stage=stage, run_id=run_id,
                                   scientific_failure=result["failed"], accuracy=result["evaluation_accuracy"],
                                   seconds=round(time.monotonic() - started, 2))
                    except Exception as exc:
                        with self.lock:
                            errors.append(str(exc))
                            self.state["failed_processes"] += 1
                        stop.set()
                        self.event("process_error", gpu=gpu, stage=stage, run_id=run_id, error=str(exc))
                    finally:
                        with self.lock:
                            self.state["running"].pop(str(gpu), None)
                        jobs.task_done()
        with ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in self.gpus]
            for future in futures:
                future.result()
        if errors:
            raise RuntimeError("; ".join(errors))
        self.event("stage_completed", stage=stage, jobs=len(configs))

    def p0(self) -> dict:
        destination = self.output / "p0"
        destination.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(self.gpus[0]), DAGC_TEST_DEVICE="cuda", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with (destination / "tests.log").open("w") as stream:
            subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        configs = []
        for family in self.protocol["screening"]["families"]:
            for label, clipping in (("fixed", dict(name="fixed", threshold=0.3)),
                                    ("autoclip", dict(name="autoclip", percentile=10, warmup=4))):
                cfg = self.configuration(family, "p0", label, 777, 1, family["learning_rates"][0], clipping)
                cfg.update(max_steps=16)
                configs.append(self.materialize(cfg))
        self.run_jobs(configs, "p0")
        import numpy as np
        from src.branching import read_step_rows
        checks = []
        for cfg in configs:
            rows = read_step_rows(Path(cfg["output_dir"]) / f"{cfg['run_id']}.steps.jsonl")
            error = max(abs(row["applied_grad_norm"] - row["coefficient"] * row["raw_grad_norm"]) /
                        max(1.0, row["raw_grad_norm"]) for row in rows)
            valid = len(rows) == 16 and error < 2e-6 and all(np.isfinite(row["applied_update_norm"]) for row in rows)
            checks.append(dict(run_id=cfg["run_id"], passed=bool(valid), relative_norm_error=error))
        report = dict(gate="G0", passed=all(row["passed"] for row in checks),
                      cuda_tests_passed=True, measurement_checks=checks)
        write_json(destination / "gate_g0.json", report)
        return report

    def p1(self) -> tuple[dict, list[dict]]:
        from analysis.research_campaign import screen_report
        configs = []
        setting = self.protocol["screening"]
        for family in setting["families"]:
            for lr in family["learning_rates"]:
                policies = [(f"fixed-{threshold:g}", dict(name="fixed", threshold=threshold)) for threshold in setting["thresholds"]]
                if setting["include_unclipped"]:
                    policies.insert(0, ("none", dict(name="none")))
                for label, policy in policies:
                    for seed in setting["seeds"]:
                        configs.append(self.materialize(self.configuration(family, "p1", label, seed, setting["epochs"], lr, policy)))
        write_json(self.output / "p1/plan.json", {"count": len(configs), "runs": [cfg["campaign"]["config_path"] for cfg in configs]})
        self.run_jobs(configs, "p1")
        self.event("analysis_started", stage="p1")
        return screen_report(configs, self.protocol["prediction"], self.output / "p1/analysis"), configs

    def p2(self, cells: list[dict]) -> dict:
        from analysis.research_campaign import causal_report
        from src.branching import prepare_causal_branches
        settings = self.protocol["causal"]
        references = []
        for cell in cells:
            source = yaml.safe_load(Path(cell["source_config"]).read_text())
            family = dict(name=cell["family"], dataset=source["dataset"]["name"], model=source["model"]["name"], optimizer=source["optimizer"]["name"])
            for seed in settings["seeds"]:
                cfg = self.configuration(family, "p2_reference", "fixed", seed, settings["epochs"],
                                         cell["learning_rate"], dict(name="fixed", threshold=cell["threshold"]))
                cfg["checkpointing"]["steps"] = [settings["start_step"], settings["start_step"] + settings["intervention_steps"]]
                references.append(self.materialize(cfg))
        self.run_jobs(references, "p2_reference")
        branches = []
        pairs = []
        for reference in references:
            run_id = reference["run_id"]
            summary = json.loads((Path(reference["output_dir"]) / f"{run_id}.summary.json").read_text())
            if summary["failed"]:
                pairs.append(dict(reference=reference, skipped="reference_failed"))
                continue
            branch_root = self.output / "p2_branches" / run_id
            manifest_path = prepare_causal_branches(
                reference_steps=Path(reference["output_dir"]) / f"{run_id}.steps.jsonl",
                base_config=reference["campaign"]["config_path"], output_dir=branch_root,
                shuffle_seed=10000 + reference["seed"], block_size=settings["block_size"],
                checkpoint_path=Path(reference["checkpointing"]["directory"]) / f"{run_id}.step-{settings['start_step']}.pt",
                start_step=settings["start_step"], intervention_steps=settings["intervention_steps"],
            )
            manifest = json.loads(manifest_path.read_text())
            pair_configs = []
            for label, path in manifest["branch_configs"].items():
                cfg = yaml.safe_load(Path(path).read_text())
                cfg["campaign"] = dict(family=reference["campaign"]["family"], stage="p2_branches", label=label,
                                       config_path=str(Path(path).resolve()))
                # These configs are freshly generated from a frozen reference;
                # all later resumption uses the same serialized form.
                Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
                pair_configs.append(cfg)
                branches.append(cfg)
            pairs.append(dict(reference=reference, branches=pair_configs))
        write_json(self.output / "p2_branches/pairs.json", pairs)
        self.run_jobs(branches, "p2_branches")
        records = self.causal_endpoints(pairs)
        report = causal_report(records, settings, self.output / "p2_branches/analysis")
        report["failed_references"] = sum("skipped" in pair for pair in pairs)
        if report["failed_references"]:
            report["passed"] = False
            report["reason"] = "reference_failures_prevent_complete_ten_seed_pairs"
        write_json(self.output / "p2_branches/analysis/gate_g2.json", report)
        return report

    def causal_endpoints(self, pairs: list[dict]) -> list[dict]:
        import numpy as np
        import torch
        from src.branching import read_step_rows
        from src.data import build_data_bundle
        from src.models import build_model
        from src.trajectory import function_distance, parameter_distance, expected_calibration_error, cka_similarity, linear_mode_connectivity_barrier
        records = []
        torch.set_num_threads(self.protocol["training"]["cpu_threads"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpus[0])
        for pair in pairs:
            if "skipped" in pair:
                continue
            reference_cfg = pair["reference"]
            family = reference_cfg["campaign"]["family"]
            branches = pair["branches"]
            replay_cfg = next(cfg for cfg in branches if cfg["campaign"]["label"] == "reference_gain_replay_pre_moment")
            replay_summary = json.loads((Path(replay_cfg["output_dir"]) / f"{replay_cfg['run_id']}.summary.json").read_text())
            original_summary = json.loads((Path(reference_cfg["output_dir"]) / f"{reference_cfg['run_id']}.summary.json").read_text())
            if replay_summary["failed"]:
                raise RuntimeError("reference gain replay failed; G0 replay invariant invalid")
            replay_payload = torch.load(replay_summary["final_checkpoint"], map_location="cpu", weights_only=False)
            original_payload = torch.load(original_summary["final_checkpoint"], map_location="cpu", weights_only=False)
            if any(not torch.equal(original_payload["model"][key], replay_payload["model"][key]) for key in original_payload["model"]):
                raise RuntimeError("reference gain replay does not exactly match original endpoint")
            dataset = reference_cfg["dataset"]
            bundle = build_data_bundle(dataset=dataset["name"], data_dir=str(self.data_dir), batch_size=128,
                seed=reference_cfg["seed"], probe_size=256, validation_size=dataset["validation_size"],
                evaluation_split="validation", split_seed=dataset["split_seed"], pin_memory=False)
            kwargs = dict(num_classes=bundle.num_classes, input_channels=bundle.input_channels,
                          image_size=28 if dataset["name"] in {"mnist", "fashion_mnist"} else 32)
            reference_model = build_model(reference_cfg["model"]["name"], **kwargs).cuda()
            reference_model.load_state_dict(replay_payload["model"])
            for cfg in branches:
                result = json.loads((Path(cfg["output_dir"]) / f"{cfg['run_id']}.summary.json").read_text())
                rows = read_step_rows(Path(cfg["output_dir"]) / f"{cfg['run_id']}.steps.jsonl")
                end = cfg["causal_branch"]["intervention_end_step"]
                short_losses = [row["loss"] for row in rows if end - 50 <= row["step"] < end and row.get("loss") is not None]
                record = dict(family=family, seed=cfg["seed"], intervention=cfg["campaign"]["label"],
                              failed=int(result["failed"]), accuracy_with_failures=0.0 if result["failed"] else result["evaluation_accuracy"],
                              short_horizon_loss=float(np.mean(short_losses)) if len(short_losses) == 50 else self.protocol["training"]["failure_loss"],
                              cumulative_update_norm=sum(row.get("applied_update_norm") or 0.0 for row in rows),
                              adverse_event_count=result["adverse_event_count"])
                if cfg["campaign"]["label"] == "reference_norm_post_update":
                    matched = [row for row in rows if row["step"] < end and row.get("target_update_norm") is not None]
                    relative_error = max((abs(row["applied_update_norm"] - row["target_update_norm"]) /
                                          max(1e-6, row["target_update_norm"]) for row in matched), default=0.0)
                    record["maximum_relative_norm_matching_error"] = relative_error
                    record["enlarged_steps"] = sum(row.get("coefficient", 1.0) > 1.0 for row in matched)
                    if relative_error > 5e-4:
                        raise RuntimeError("post-update branch failed its declared norm-matching invariant")
                end_checkpoint = Path(cfg["checkpointing"]["directory"]) / f"{cfg['run_id']}.step-{end}.pt"
                reference_end = Path(reference_cfg["checkpointing"]["directory"]) / f"{reference_cfg['run_id']}.step-{end}.pt"
                if end_checkpoint.exists():
                    first = torch.load(reference_end, map_location="cpu", weights_only=False)["optimizer"]["state"]
                    second = torch.load(end_checkpoint, map_location="cpu", weights_only=False)["optimizer"]["state"]
                    differences, scales = [], []
                    for key in first:
                        for field in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                            if field in first[key] and field in second.get(key, {}):
                                differences.append((first[key][field] - second[key][field]).square().sum())
                                scales.append(first[key][field].square().sum())
                    record["optimizer_state_relative_distance_at_intervention_end"] = (
                        float(torch.stack(differences).sum().sqrt() / (1 + torch.stack(scales).sum().sqrt())) if differences else 0.0)
                if not result["failed"]:
                    model = build_model(reference_cfg["model"]["name"], **kwargs).cuda()
                    payload = torch.load(result["final_checkpoint"], map_location="cpu", weights_only=False)
                    model.load_state_dict(payload["model"])
                    layer = "avgpool" if reference_cfg["model"]["name"].startswith("resnet") else "classifier.1"
                    record.update(parameter_distance=parameter_distance(reference_model, model),
                        validation_disagreement=function_distance(reference_model, model, bundle.test_loader, "disagreement"),
                        symmetric_kl=function_distance(reference_model, model, bundle.probe_loader, "symmetric_kl"),
                        ece=expected_calibration_error(model, bundle.test_loader),
                        cka=cka_similarity(reference_model, model, bundle.probe_loader, [layer]),
                        mode_connectivity_barrier=linear_mode_connectivity_barrier(reference_model, model, bundle.probe_loader, n_steps=5))
                    del model, payload
                records.append(record)
            del reference_model, replay_payload, original_payload
        return records

    def finish(self, gate: dict, stage: str):
        self.state["stage"] = ("complete" if stage == "p4" else
                               "requested_stage_complete" if gate.get("passed") else "stopped_at_scientific_gate")
        report = dict(last_stage=stage, gate=gate, status=self.state["stage"],
                      protocol=str(self.output / "frozen_protocol.json"),
                      note="A failed gate is a retained research result; later algorithm/scale claims are not made.")
        write_json(self.output / "campaign_report.json", report)
        self.event("campaign_finished", stage=stage, gate_passed=gate.get("passed", False))

    def execute(self, stop_after: str):
        g0 = self.p0()
        if not g0["passed"]:
            self.finish(g0, "p0")
            return
        if stop_after == "p0":
            self.event("requested_stage_complete", stage="p0")
            return
        g1, _ = self.p1()
        if not g1["passed"] or stop_after == "p1":
            self.finish(g1, "p1")
            return
        g2 = self.p2(g1["selected_cells"])
        if not g2["passed"] or stop_after == "p2":
            self.finish(g2, "p2")
            return
        # P3 is defined in a separate module to keep the measurement/causal
        # route usable without importing controller evaluation code.
        from scripts.run_research_confirmation import run_confirmation
        g3 = run_confirmation(self, g1["selected_cells"])
        if not g3["passed"] or stop_after == "p3":
            self.finish(g3, "p3")
            return
        from scripts.run_research_confirmation import run_scale
        g4 = run_scale(self, g3)
        self.finish(g4, "p4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/campaigns/lyx_full.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-after", choices=["p0", "p1", "p2", "p3", "p4"], default="p4")
    args = parser.parse_args()
    campaign = Campaign(args.protocol, args.output, args.data_dir, args.gpus)
    if not args.execute:
        print(f"Frozen campaign at {args.output}; pass --execute to train", flush=True)
        return
    with (campaign.output / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            campaign.execute(args.stop_after)
        except Exception as exc:
            campaign.state["stage"] = "infrastructure_error"
            campaign.event("campaign_error", error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
