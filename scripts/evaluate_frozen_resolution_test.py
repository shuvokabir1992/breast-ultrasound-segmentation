"""Evaluate six frozen checkpoints on test; select nothing from test scores.

Requires the existing baseline scripts and run_resolution_repeats.py. The
384-pixel primary setting was selected using mean validation Case Dice.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from run_resolution_repeats import (
    METRICS, SEEDS, SIZES, control_settings, csv_rows, read_json, resolve,
    sha256, validated_evaluation,
)


SELECTED_RESOLUTION = 384
EXPECTED_MANIFEST = "e37a37ea057ff2e93728150bb36a67bcc7f47dcbb3228a78fb3bfe8f3d68820c"
EXPECTED_CHECKPOINTS = {
    (256, 42): "4754f8bbc1ae4ed57fffbab4256612b1c6642461fc05fea5fb2e54e15fe360f8",
    (256, 43): "793b98c517e273f9c796dbab50f341672c94cf615e816989ce5bd2b8eb02fb0f",
    (256, 44): "9b97730263c19634412b2d2b3cb1c1b3814ed93a27c3fb5e81c3254a41d2841e",
    (384, 42): "a0d078f02c53ac201058bfbc9d0549b092dc8305d2914e9cf4d52d33410111e6",
    (384, 43): "ff09abace0c861f53a2b1598ac69073be0a319093c13dd820b222fde7bf9d287",
    (384, 44): "bd0f4d95382bb55976a82de3586be30b933f2b1844d703c6eedf78a3149359c4",
}


def read_test_result(job, test_rows, manifest_hash):
    folder = job["output_dir"]
    metrics = read_json(folder / "metrics.json")
    if metrics.get("split") != "test":
        raise ValueError(f"{folder}: expected a test evaluation")
    if (metrics["checkpoint_sha256"] != job["checkpoint_sha256"] or
            metrics["manifest_sha256"] != manifest_hash or float(metrics["threshold"]) != 0.5 or
            int(metrics["checkpoint_epoch"]) != job["checkpoint_epoch"]):
        raise ValueError(f"{folder}: frozen checkpoint/manifest/threshold/epoch mismatch")
    rows = csv_rows(folder / "per_image.csv")
    if len(rows) != len({r["id"] for r in rows}) or {r["id"] for r in rows} != set(test_rows):
        raise ValueError(f"{folder}: test image IDs differ from the frozen manifest")
    grouped = defaultdict(list)
    for row in rows:
        for key in ("case_id", "pathology", "device"):
            if row[key] != test_rows[row["id"]][key]:
                raise ValueError(f"{folder}: inconsistent test metadata")
        for key in METRICS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("Invalid test overlap metric")
        row["hd95_px"] = None if row["hd95_px"] in ("", "None", "null") else float(row["hd95_px"])
        if row["hd95_px"] is not None and (not math.isfinite(row["hd95_px"]) or row["hd95_px"] < 0):
            raise ValueError("Invalid test distance metric")
        grouped[row["case_id"]].append(row)
    cases = []
    for case_id, views in sorted(grouped.items()):
        cases.append({"case_id": case_id, "images": len(views),
                      "pathology": views[0]["pathology"], "device": views[0]["device"],
                      **{key: statistics.mean(v[key] for v in views) for key in METRICS},
                      "hd95_px": statistics.mean(v["hd95_px"] for v in views)
                      if all(v["hd95_px"] is not None for v in views) else None})
    if metrics["images"] != len(rows) or metrics["cases"] != len(cases):
        raise ValueError("Test evaluation counts differ from the CSV")
    for key in METRICS:
        if not math.isclose(statistics.mean(r[key] for r in cases), metrics["case_macro"][key], rel_tol=0, abs_tol=1e-10):
            raise ValueError("Recomputed test Case metrics differ from JSON")
        for column, output in (("pathology", "by_pathology"), ("device", "by_device")):
            for value, group in metrics.get(output, {}).items():
                selected = [r for r in cases if r[column] == value]
                if group["cases"] != len(selected) or not selected or not math.isclose(
                        statistics.mean(r[key] for r in selected), group["case_macro"][key], rel_tol=0, abs_tol=1e-10):
                    raise ValueError("Recomputed test subgroup metric differs from JSON")
    defined = [r["hd95_px"] for r in cases if r["hd95_px"] is not None]
    distance = metrics["hd95"]
    if distance["case_fully_defined_count"] != len(defined) or distance["case_undefined_count"] != len(cases) - len(defined):
        raise ValueError("Test HD95 defined/undefined counts differ")
    actual = statistics.mean(defined) if defined else None
    expected = distance["case_mean_fully_defined_px"]
    if (actual is None) != (expected is None) or (actual is not None and not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)):
        raise ValueError("Recomputed test HD95 differs from JSON")
    return {"image_size": job["image_size"], "seed": job["seed"],
            "role": "primary" if job["image_size"] == SELECTED_RESOLUTION else "reference",
            "checkpoint_sha256": job["checkpoint_sha256"], "checkpoint_epoch": job["checkpoint_epoch"],
            "evaluation_dir": str(folder), "metrics": metrics, "per_case": cases}


def make_plan(source, root, expected_checkpoints=EXPECTED_CHECKPOINTS, expected_manifest=EXPECTED_MANIFEST):
    rows = source["runs"]
    identities = [(r["image_size"], r["seed"]) for r in rows]
    if len(rows) != 6 or set(identities) != set(expected_checkpoints):
        raise ValueError("Expected six frozen resolution/seed identities")
    if source["split"] != "val" or source["postprocessing"] != "none" or source["manifest_sha256"] != expected_manifest:
        raise ValueError("Validation summary differs from the frozen experiment")
    controls = control_settings(rows[0]["config"])
    jobs, verified = [], []
    all_manifest_rows = None
    for row in sorted(rows, key=lambda r: (r["image_size"], r["seed"])):
        size, seed = row["image_size"], row["seed"]
        config = row["config"]
        if config["seed"] != seed or config["image_size"] != size or control_settings(config) != controls:
            raise ValueError("Frozen configuration mismatch")
        if row["metrics"]["checkpoint_sha256"] != expected_checkpoints[size, seed]:
            raise ValueError("Validation summary names an unexpected checkpoint")
        run = resolve(root, row["run_dir"])
        manifest_rows = csv_rows(run / "manifest.csv")
        if all_manifest_rows is None:
            all_manifest_rows = manifest_rows
        val_rows = {r["id"]: r for r in manifest_rows if r["split"] == "val"}
        checked = validated_evaluation(run, resolve(root, row["evaluation_dir"]), config,
                                       expected_manifest, val_rows)
        if checked["metrics"]["checkpoint_sha256"] != expected_checkpoints[size, seed]:
            raise ValueError("Saved checkpoint differs from the frozen identity")
        verified.append(checked)
        output = root / "reports" / f"unet_baseline_{size}_seed{seed}_test"
        original_test = root / "reports" / "unet_baseline_seed42_test"
        if size == 256 and seed == 42 and original_test.exists() and any(original_test.iterdir()):
            output = original_test
        command = [sys.executable, "-u", "scripts/evaluate_baseline.py", "--checkpoint", str(run / "best.pt"),
                   "--split", "test", "--num-workers", str(config["num_workers"]),
                   "--save-masks", "--output-dir", str(output)]
        jobs.append({"image_size": size, "seed": seed, "checkpoint_sha256": expected_checkpoints[size, seed],
                     "checkpoint_epoch": checked["checkpoint_epoch"], "output_dir": output, "command": command})
    means = {size: statistics.mean(r["metrics"]["case_macro"]["dice"] for r in verified if r["image_size"] == size)
             for size in SIZES}
    ranked = max(SIZES, key=lambda size: (means[size], -size))
    if ranked != SELECTED_RESOLUTION:
        raise ValueError("Validation ranking no longer matches the frozen 384-pixel selection")
    test_rows = {r["id"]: r for r in all_manifest_rows if r["split"] == "test"}
    if not test_rows:
        raise ValueError("Frozen manifest has no test data")
    for job in jobs:
        folder = job["output_dir"]
        if folder.exists() and any(folder.iterdir()):
            if not all((folder / name).is_file() for name in ("metrics.json", "per_image.csv")):
                raise ValueError(f"Incomplete test evaluation: {folder}. Preserve it under another name before rerunning.")
            read_test_result(job, test_rows, expected_manifest)
    protocol = {"selected_resolution": SELECTED_RESOLUTION, "reference_resolution": 256,
                "seeds": list(SEEDS), "selection_rule": "Higher mean validation Case Dice over seeds 42/43/44; exact ties favour 256.",
                "validation_case_dice_means": {str(k): v for k, v in means.items()},
                "manifest_sha256": expected_manifest, "threshold": 0.5, "postprocessing": "none",
                "metric_resolution": "native image pixels", "test_images": len(test_rows),
                "test_cases": len({r["case_id"] for r in test_rows.values()}),
                "checkpoint_sha256": {f"{size}_seed{seed}": value for (size, seed), value in expected_checkpoints.items()},
                "test_selection": "No choice of resolution, checkpoint, seed, threshold or postprocessing will be made from test scores.",
                "scope": "Three training seeds per resolution on one fixed holdout. The original 256/seed42 test result was inspected earlier; this is a reused holdout, not independent external validation."}
    return jobs, test_rows, protocol


def execute(jobs, test_rows, protocol, root, env, runner=subprocess.run):
    results = []
    for index, job in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] Test: {job['image_size']} px, seed {job['seed']}", flush=True)
        folder = job["output_dir"]
        if folder.exists() and any(folder.iterdir()):
            print("Reusing a verified test evaluation.", flush=True)
        else:
            print(shlex.join(job["command"]), flush=True)
            runner(job["command"], cwd=root, env=env, check=True)
        result = read_test_result(job, test_rows, protocol["manifest_sha256"])
        results.append(result)
        print(f"Case Dice: {result['metrics']['case_macro']['dice']:.6f}", flush=True)
    return results


def aggregate(results, protocol):
    if len(results) != 6 or {(r["image_size"], r["seed"]) for r in results} != {(size, seed) for size in SIZES for seed in SEEDS}:
        raise ValueError("Six test evaluations are required")
    grouped = {}
    for size in SIZES:
        rows = [r for r in results if r["image_size"] == size]
        stats = {}
        for metric in METRICS:
            values = [r["metrics"]["case_macro"][metric] for r in rows]
            stats[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "n": 3}
        for pathology in ("benign", "malignant"):
            values = [r["metrics"]["by_pathology"][pathology]["case_macro"]["dice"] for r in rows]
            stats[f"{pathology}_case_dice"] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "n": 3}
        values = [r["metrics"]["hd95"]["case_mean_fully_defined_px"] for r in rows]
        values = [x for x in values if x is not None]
        stats["hd95_defined_case_mean_px"] = {"mean": statistics.mean(values) if values else None,
                                               "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                                               "n": len(values), "note": "Check per-run undefined Case counts before comparing distances."}
        grouped[str(size)] = stats
    return {"split": "test", "protocol": protocol,
            "selected_resolution": SELECTED_RESOLUTION, "by_resolution": grouped,
            "uncertainty": "sample_std describes three training seeds; this is not a Case-bootstrap interval or standard error.",
            "runs": sorted(results, key=lambda r: (r["image_size"], r["seed"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validation-summary", default="reports/resolution_repeats/summary.json")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        raise ValueError("Use one physical GPU index")
    root = args.project_root.resolve()
    source_path = resolve(root, args.validation_summary)
    jobs, test_rows, protocol = make_plan(read_json(source_path), root)
    protocol["validation_summary_sha256"] = sha256(source_path)
    print("Frozen primary resolution: 384. Reference: 256. Seeds: 42, 43, 44.", flush=True)
    print(f"Physical GPU: {args.gpu}. Test cohort: {protocol['test_images']} images, {protocol['test_cases']} Cases.", flush=True)
    if args.plan:
        for job in jobs:
            existing = job["output_dir"].exists() and any(job["output_dir"].iterdir())
            print(f"{'REUSE' if existing else 'EVALUATE'} {job['image_size']} / seed {job['seed']}: {job['output_dir']}")
        print("Plan checked. No test inference started and no files written.")
        return
    output = root / "reports" / "frozen_resolution_test"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A frozen test runner is already active") from error
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Existing frozen test protocol differs")
        if not protocol_path.exists():
            with protocol_path.open("x") as handle:
                json.dump(protocol, handle, indent=2)
                handle.write("\n")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
        results = execute(jobs, test_rows, protocol, root, env)
        report = aggregate(results, protocol)
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "summary.json")
        print(f"\nCompleted frozen test reporting: {output / 'summary.json'}", flush=True)
        print("The primary setting remains 384; no selection was made from test scores.", flush=True)


if __name__ == "__main__":
    main()
