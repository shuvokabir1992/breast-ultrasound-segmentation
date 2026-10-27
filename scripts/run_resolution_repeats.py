"""Run four frozen resolution/seed repeats sequentially and summarize validation.

Run from the project root in the existing venv. Existing seed-42 runs are reused.
The runner uses only the standard library; child scripts use baseline packages.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


SEEDS = (42, 43, 44)
SIZES = (256, 384)
METRICS = ("dice", "iou", "sensitivity", "precision")
CONTROL_EXCEPTIONS = {"image_size", "seed", "run_dir", "manifest"}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control_settings(config):
    return {k: v for k, v in config.items() if k not in CONTROL_EXCEPTIONS}


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def inspect_training(run_dir, expected, manifest_hash, require_complete=True):
    required = ("config.json", "manifest.csv", "environment.json", "history.csv", "best.pt", "last.pt")
    if not all((run_dir / name).is_file() for name in required):
        raise ValueError(f"Incomplete run: {run_dir}. Preserve it under another name before restarting; checkpoint resume is unavailable.")
    config = read_json(run_dir / "config.json")
    for key in ("image_size", "seed"):
        if int(config[key]) != int(expected[key]):
            raise ValueError(f"{run_dir}: unexpected {key}")
    if config.get("smoke", False) or control_settings(config) != control_settings(expected):
        raise ValueError(f"{run_dir}: training settings differ from the frozen protocol")
    if sha256(run_dir / "manifest.csv") != manifest_hash:
        raise ValueError(f"{run_dir}: manifest differs from the frozen protocol")
    environment = read_json(run_dir / "environment.json")
    if environment["manifest_sha256"] != manifest_hash or environment.get("smoke", False):
        raise ValueError(f"{run_dir}: invalid environment record")
    history = csv_rows(run_dir / "history.csv")
    if not history or [int(r["epoch"]) for r in history] != list(range(1, len(history) + 1)):
        raise ValueError(f"{run_dir}: incomplete or nonsequential history")
    if any(r["smoke"].lower() != "false" or not math.isfinite(float(r["val_case_dice"])) for r in history):
        raise ValueError(f"{run_dir}: smoke/nonfinite history")
    best = max(history, key=lambda r: float(r["val_case_dice"]))
    complete = (len(history) == int(config["epochs"]) or
                len(history) - int(best["epoch"]) >= int(config["patience"]))
    if require_complete and not complete:
        raise ValueError(f"Incomplete run: {run_dir}. Preserve it under another name before restarting; checkpoint resume is unavailable.")
    return config, best, history, environment


def validated_evaluation(run_dir, eval_dir, expected, manifest_hash, validation_rows):
    config, best, history, environment = inspect_training(run_dir, expected, manifest_hash)
    metrics = read_json(eval_dir / "metrics.json")
    if metrics.get("split") != "val":
        raise ValueError(f"{eval_dir}: only validation evaluations are accepted")
    if metrics["manifest_sha256"] != manifest_hash or metrics["checkpoint_sha256"] != sha256(run_dir / "best.pt"):
        raise ValueError(f"{eval_dir}: evaluation identity does not match the run")
    if int(metrics["checkpoint_epoch"]) != int(best["epoch"]) or float(metrics["threshold"]) != float(config["threshold"]):
        raise ValueError(f"{eval_dir}: checkpoint selection or threshold mismatch")
    rows = csv_rows(eval_dir / "per_image.csv")
    if len(rows) != len({r["id"] for r in rows}) or {r["id"] for r in rows} != set(validation_rows):
        raise ValueError(f"{eval_dir}: must contain every validation image exactly once")
    grouped = defaultdict(list)
    for row in rows:
        reference = validation_rows[row["id"]]
        for key in ("case_id", "pathology", "device"):
            if row[key] != reference[key]:
                raise ValueError(f"{eval_dir}: inconsistent image metadata")
        for key in METRICS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"{eval_dir}: invalid overlap metric")
        value = row["hd95_px"]
        row["hd95_px"] = None if value in ("", "None", "null") else float(value)
        if row["hd95_px"] is not None and (not math.isfinite(row["hd95_px"]) or row["hd95_px"] < 0):
            raise ValueError(f"{eval_dir}: invalid HD95")
        grouped[row["case_id"]].append(row)
    cases = []
    for case_id, views in sorted(grouped.items()):
        case = {"case_id": case_id, "images": len(views),
                "pathology": views[0]["pathology"], "device": views[0]["device"],
                **{key: statistics.mean(r[key] for r in views) for key in METRICS}}
        case["hd95_px"] = (statistics.mean(r["hd95_px"] for r in views)
                           if all(r["hd95_px"] is not None for r in views) else None)
        cases.append(case)
    if metrics["images"] != len(rows) or metrics["cases"] != len(cases):
        raise ValueError(f"{eval_dir}: evaluation counts do not match CSV")
    for key in METRICS:
        if not math.isclose(statistics.mean(r[key] for r in cases), metrics["case_macro"][key], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"{eval_dir}: recomputed Case metric does not match JSON")
    defined = [r["hd95_px"] for r in cases if r["hd95_px"] is not None]
    hd_report = metrics["hd95"]
    if hd_report["case_fully_defined_count"] != len(defined) or hd_report["case_undefined_count"] != len(cases) - len(defined):
        raise ValueError(f"{eval_dir}: HD95 defined/undefined counts differ")
    actual_hd = statistics.mean(defined) if defined else None
    expected_hd = hd_report["case_mean_fully_defined_px"]
    if (actual_hd is None) != (expected_hd is None) or (actual_hd is not None and not math.isclose(actual_hd, expected_hd, rel_tol=0, abs_tol=1e-10)):
        raise ValueError(f"{eval_dir}: recomputed Case HD95 differs")
    return {"image_size": int(config["image_size"]), "seed": int(config["seed"]),
            "checkpoint_epoch": int(best["epoch"]), "epochs_completed": len(history),
            "training_minutes": float(history[-1]["seconds"]) / 60,
            "run_dir": str(run_dir), "evaluation_dir": str(eval_dir),
            "metrics": metrics, "per_case": cases, "config": config,
            "environment": environment}


def make_plan(root, baseline_run, candidate_run, baseline_val, candidate_val):
    sources = {256: (baseline_run, baseline_val), 384: (candidate_run, candidate_val)}
    configs = {size: read_json(paths[0] / "config.json") for size, paths in sources.items()}
    for size, config in configs.items():
        if int(config["image_size"]) != size or int(config["seed"]) != 42 or config.get("smoke", False):
            raise ValueError("Source runs must be full seed-42 runs at 256 and 384 pixels")
    if control_settings(configs[256]) != control_settings(configs[384]):
        keys = sorted(k for k in set(configs[256]) | set(configs[384])
                      if k not in CONTROL_EXCEPTIONS and configs[256].get(k) != configs[384].get(k))
        raise ValueError(f"Source configurations differ beyond resolution/run paths: {keys}")
    manifest = baseline_run / "manifest.csv"
    manifest_hash = sha256(manifest)
    if sha256(candidate_run / "manifest.csv") != manifest_hash:
        raise ValueError("Source manifests differ")
    all_rows = csv_rows(manifest)
    if len(all_rows) != len({r["id"] for r in all_rows}):
        raise ValueError("Duplicate image IDs in frozen manifest")
    for case_id in {r["case_id"] for r in all_rows}:
        if len({r["split"] for r in all_rows if r["case_id"] == case_id}) != 1:
            raise ValueError("A Case crosses splits")
    validation = {r["id"]: r for r in all_rows if r["split"] == "val"}
    existing = [validated_evaluation(run, evaluation, configs[size], manifest_hash, validation)
                for size, (run, evaluation) in sources.items()]
    jobs = []
    for seed in (43, 44):
        for size in SIZES:
            name = f"unet_baseline_{size}_seed{seed}"
            run = root / "runs" / name
            evaluation = root / "reports" / f"{name}_val"
            config_path = root / "configs" / "resolution_repeats" / f"{name}.json"
            config = copy.deepcopy(configs[256])
            config.update({"image_size": size, "seed": seed, "run_dir": str(run),
                           "manifest": str(manifest), "smoke": False})
            train_command = [sys.executable, "-u", "scripts/train_baseline.py", "--config", str(config_path),
                             "--num-workers", str(config["num_workers"]), "--run-dir", str(run)]
            eval_command = [sys.executable, "-u", "scripts/evaluate_baseline.py", "--checkpoint", str(run / "best.pt"),
                            "--split", "val", "--num-workers", str(config["num_workers"]),
                            "--save-masks", "--output-dir", str(evaluation)]
            jobs.append({"name": name, "config": config, "config_path": config_path,
                         "run_dir": run, "eval_dir": evaluation,
                         "train_command": train_command, "eval_command": eval_command})
    return existing, jobs, manifest_hash, validation


def execute_jobs(root, jobs, manifest_hash, validation, env, runner=subprocess.run):
    results = []
    for index, job in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {job['name']}", flush=True)
        run, evaluation = job["run_dir"], job["eval_dir"]
        if run.exists() and any(run.iterdir()):
            inspect_training(run, job["config"], manifest_hash)
            print("Reusing verified completed training.", flush=True)
        else:
            path = job["config_path"]
            if path.exists() and read_json(path) != job["config"]:
                raise ValueError(f"Existing generated config differs: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("x") as handle:
                    json.dump(job["config"], handle, indent=2)
                    handle.write("\n")
            print(shlex.join(job["train_command"]), flush=True)
            runner(job["train_command"], cwd=root, env=env, check=True)
            inspect_training(run, job["config"], manifest_hash)
        if evaluation.exists() and any(evaluation.iterdir()):
            if not (evaluation / "metrics.json").is_file() or not (evaluation / "per_image.csv").is_file():
                raise ValueError(f"Incomplete evaluation: {evaluation}. Preserve it under another name, then rerun.")
        else:
            print(shlex.join(job["eval_command"]), flush=True)
            runner(job["eval_command"], cwd=root, env=env, check=True)
        results.append(validated_evaluation(run, evaluation, job["config"], manifest_hash, validation))
        print(f"Validation Case Dice: {results[-1]['metrics']['case_macro']['dice']:.6f}", flush=True)
    return results


def summarize_runs(runs, manifest_hash):
    identities = [(r["image_size"], r["seed"]) for r in runs]
    if len(identities) != 6 or set(identities) != {(size, seed) for size in SIZES for seed in SEEDS}:
        raise ValueError("Summary requires exactly seeds 42, 43 and 44 at each resolution")
    grouped = {}
    for size in SIZES:
        rows = sorted((r for r in runs if r["image_size"] == size), key=lambda r: r["seed"])
        stats = {}
        for metric in METRICS:
            values = [r["metrics"]["case_macro"][metric] for r in rows]
            stats[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "n": len(values)}
        for pathology in ("benign", "malignant"):
            values = [r["metrics"]["by_pathology"][pathology]["case_macro"]["dice"] for r in rows]
            stats[f"{pathology}_case_dice"] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "n": len(values)}
        hd = [r["metrics"]["hd95"]["case_mean_fully_defined_px"] for r in rows]
        defined = [x for x in hd if x is not None]
        stats["hd95_defined_case_mean_px"] = {"mean": statistics.mean(defined) if defined else None,
                                               "sample_std": statistics.stdev(defined) if len(defined) > 1 else None,
                                               "n": len(defined),
                                               "note": "Each run averages its fully defined Cases; check undefined counts in per-run metrics."}
        grouped[str(size)] = stats
    by_identity = {(r["image_size"], r["seed"]): r for r in runs}
    deltas = [{"seed": seed, "case_dice_384_minus_256":
               by_identity[(384, seed)]["metrics"]["case_macro"]["dice"] -
               by_identity[(256, seed)]["metrics"]["case_macro"]["dice"]} for seed in SEEDS]
    first = max(SIZES, key=lambda size: (grouped[str(size)]["dice"]["mean"], -size))
    return {"split": "val", "seeds": list(SEEDS), "image_sizes": list(SIZES),
            "manifest_sha256": manifest_hash, "postprocessing": "none",
            "experiment_scope": "Three training seeds per resolution on one fixed Case-disjoint holdout; not ten-fold cross-validation.",
            "selection_rule": "Rank by mean validation Case Dice across seeds 42/43/44; exact ties favour 256. Report subgroup and secondary metrics. Ranking is descriptive, not proof of superiority.",
            "uncertainty": "sample_std is across training seeds (n=3), not a Case-bootstrap confidence interval or standard error.",
            "by_resolution": grouped, "paired_seed_dice_differences": deltas,
            "leading_resolution_by_mean_validation_case_dice": first,
            "runs": sorted(runs, key=lambda r: (r["image_size"], r["seed"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--plan", action="store_true", help="Validate sources and print commands without training or writing files")
    parser.add_argument("--baseline-run-dir", default="runs/unet_baseline_seed42")
    parser.add_argument("--candidate-run-dir", default="runs/unet_baseline_384_seed42")
    parser.add_argument("--baseline-val-dir", default="reports/unet_baseline_seed42_val_masks")
    parser.add_argument("--candidate-val-dir", default="reports/unet_baseline_384_seed42_val")
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        raise ValueError("--gpu must be one physical GPU index, for example 3")
    root = args.project_root.resolve()
    for name in ("train_baseline.py", "evaluate_baseline.py"):
        if not (root / "scripts" / name).is_file():
            raise FileNotFoundError(f"Missing baseline script: {root / 'scripts' / name}")
    existing, jobs, manifest_hash, validation = make_plan(
        root, resolve(root, args.baseline_run_dir), resolve(root, args.candidate_run_dir),
        resolve(root, args.baseline_val_dir), resolve(root, args.candidate_val_dir))
    print(f"Python: {sys.executable}\nPhysical GPU: {args.gpu}; PCI_BUS_ID ordering\nManifest: {manifest_hash}", flush=True)
    print("Reusing the two completed seed-42 validation evaluations.", flush=True)
    if args.plan:
        for job in jobs:
            print(f"\n{job['name']}\n{shlex.join(job['train_command'])}\n{shlex.join(job['eval_command'])}")
        print("Plan checked. No files written and no GPU job started.")
        return
    output = root / "reports" / "resolution_repeats"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another resolution-repeat runner is already active") from error
        env = os.environ.copy()
        env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu,
                    "PYTHONUNBUFFERED": "1"})
        protocol = {"seeds": list(SEEDS), "image_sizes": list(SIZES), "manifest_sha256": manifest_hash,
                    "selection_rule": "Mean validation Case Dice across three seeds; exact ties favour 256.",
                    "control_settings": control_settings(existing[0]["config"]),
                    "postprocessing": "none", "gpu": args.gpu}
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Existing repeat protocol differs; preserve the experiment before changing settings")
        if not protocol_path.exists():
            with protocol_path.open("x") as handle:
                json.dump(protocol, handle, indent=2)
                handle.write("\n")
        completed = execute_jobs(root, jobs, manifest_hash, validation, env)
        summary = summarize_runs(existing + completed, manifest_hash)
        target = output / "summary.json"
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
        print("\nCompleted all four repeats and validation evaluations.", flush=True)
        for size in SIZES:
            s = summary["by_resolution"][str(size)]["dice"]
            print(f"{size} px: validation Case Dice {s['mean']:.6f} +/- {s['sample_std']:.6f} (sample SD; n=3)", flush=True)
        print(f"Combined results: {target}\nTest images were not evaluated.", flush=True)


if __name__ == "__main__":
    main()
