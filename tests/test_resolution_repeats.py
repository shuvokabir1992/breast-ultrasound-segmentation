"""CPU tests of orchestration/integrity; GPU training is not simulated as evidence."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_resolution_repeats as rr


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ResolutionRepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        for name in ("train_baseline.py", "evaluate_baseline.py"):
            (self.root / "scripts" / name).write_text("# Synthetic fixture: never executed\n")
        self.records = []
        for case, view, split, pathology in ((1, "s", "train", "benign"),
                                            (2, "l", "val", "benign"),
                                            (2, "r", "val", "benign"),
                                            (3, "s", "val", "malignant"),
                                            (4, "s", "test", "benign")):
            image_id = f"bus_{case:04d}-{view}"
            self.records.append({"id": image_id, "case_id": str(case), "split": split,
                                 "fold": str({"train": 3, "val": 2, "test": 1}[split]),
                                 "pathology": pathology, "image_path": f"Images/{image_id}.png",
                                 "mask_path": f"Masks/{image_id}.png", "width": "64", "height": "48",
                                 "device": "synthetic"})
        self.manifest = self.root / "source_manifest.csv"
        write_csv(self.manifest, self.records)
        self.hash = rr.sha256(self.manifest)
        self.sources = {}
        for size in (256, 384):
            name = "unet_baseline_seed42" if size == 256 else "unet_baseline_384_seed42"
            run = self.root / "runs" / name
            evaluation = self.root / "reports" / f"{name}_val"
            config = {"data_root": "data/raw/BUSBRA", "manifest": str(self.manifest),
                      "run_dir": str(run), "image_size": size, "seed": 42,
                      "batch_size": 16, "num_workers": 0, "epochs": 3, "patience": 2,
                      "learning_rate": 0.001, "weight_decay": 0.0001,
                      "threshold": 0.5, "amp": True, "horizontal_flip_probability": 0.5,
                      "smoke": False, "model": {"channels": [32, 64, 128, 256, 512]}}
            self.make_training(run, config)
            self.make_evaluation(run, evaluation)
            self.sources[size] = (run, evaluation)

    def make_training(self, run, config):
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps(config))
        (run / "manifest.csv").write_bytes(self.manifest.read_bytes())
        (run / "environment.json").write_text(json.dumps({"manifest_sha256": self.hash, "smoke": False}))
        (run / "best.pt").write_bytes(b"Synthetic checkpoint identity; no model is stored")
        (run / "last.pt").write_bytes(b"Synthetic final identity; no model is stored")
        write_csv(run / "history.csv", [{"epoch": epoch, "val_case_dice": dice,
                                          "seconds": epoch * 60, "smoke": False}
                                         for epoch, dice in ((1, 0.9), (2, 0.8), (3, 0.7))])

    def make_evaluation(self, run, evaluation, undefined=False):
        evaluation.mkdir(parents=True)
        config = rr.read_json(run / "config.json")
        shift = (0.01 if config["image_size"] == 384 else 0) + (config["seed"] - 42) * 0.001
        rows = []
        for reference, dice in zip((r for r in self.records if r["split"] == "val"), (0.8, 0.6, 0.7)):
            dice += shift
            row = {key: reference[key] for key in ("id", "case_id", "pathology", "device")}
            row.update({"dice": dice, "iou": dice / (2 - dice), "sensitivity": dice,
                        "precision": dice, "empty_prediction": False, "empty_target": False,
                        "hd95_px": 5.0})
            rows.append(row)
        if undefined:
            rows[0].update({"dice": 0, "iou": 0, "sensitivity": 0, "precision": 0,
                            "empty_prediction": True, "hd95_px": None})
        cases = {}
        for row in rows:
            cases.setdefault(row["case_id"], []).append(row)
        case_metrics = [{key: statistics.mean(r[key] for r in views) for key in rr.METRICS}
                        for views in cases.values()]
        defined = [statistics.mean(r["hd95_px"] for r in views) for views in cases.values()
                   if all(r["hd95_px"] is not None for r in views)]
        metrics = {"split": "val", "images": len(rows), "cases": len(cases),
                   "checkpoint_epoch": 1, "checkpoint_sha256": rr.sha256(run / "best.pt"),
                   "manifest_sha256": self.hash, "threshold": config["threshold"],
                   "case_macro": {key: statistics.mean(r[key] for r in case_metrics) for key in rr.METRICS},
                   "hd95": {"case_fully_defined_count": len(defined), "case_undefined_count": len(cases) - len(defined),
                             "case_mean_fully_defined_px": statistics.mean(defined) if defined else None},
                   "by_pathology": {pathology: {"case_macro": {"dice": statistics.mean(
                       statistics.mean(r["dice"] for r in views) for views in cases.values()
                       if views[0]["pathology"] == pathology)}} for pathology in ("benign", "malignant")}}
        (evaluation / "metrics.json").write_text(json.dumps(metrics))
        write_csv(evaluation / "per_image.csv", rows)

    def plan(self):
        return rr.make_plan(self.root, self.sources[256][0], self.sources[384][0],
                            self.sources[256][1], self.sources[384][1])

    def test_plan_preserves_controls_and_uses_frozen_manifest(self):
        existing, jobs, manifest_hash, validation = self.plan()
        self.assertEqual([(j["config"]["image_size"], j["config"]["seed"]) for j in jobs],
                         [(256, 43), (384, 43), (256, 44), (384, 44)])
        self.assertEqual(len(existing), 2)
        self.assertEqual(manifest_hash, self.hash)
        self.assertEqual(len(validation), 3)
        for job in jobs:
            self.assertEqual(rr.control_settings(job["config"]), rr.control_settings(existing[0]["config"]))
            self.assertEqual(rr.sha256(Path(job["config"]["manifest"])), self.hash)
            self.assertIn("val", job["eval_command"])
            self.assertNotIn("test", job["eval_command"])

    def test_changed_source_hyperparameter_is_rejected(self):
        path = self.sources[384][0] / "config.json"
        config = rr.read_json(path)
        config["learning_rate"] = 0.002
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "differ beyond resolution"):
            self.plan()

    def test_test_evaluation_is_rejected(self):
        path = self.sources[256][1] / "metrics.json"
        metrics = rr.read_json(path)
        metrics["split"] = "test"
        path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "only validation"):
            self.plan()

    def test_incomplete_training_is_rejected(self):
        run = self.sources[256][0]
        write_csv(run / "history.csv", [{"epoch": 1, "val_case_dice": 0.7, "seconds": 60, "smoke": False},
                                          {"epoch": 2, "val_case_dice": 0.8, "seconds": 120, "smoke": False}])
        with self.assertRaisesRegex(ValueError, "Incomplete run"):
            self.plan()

    def test_mixed_checkpoint_and_tampered_csv_are_rejected(self):
        path = self.sources[256][1] / "metrics.json"
        original = rr.read_json(path)
        changed = dict(original, checkpoint_sha256="wrong")
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.plan()
        path.write_text(json.dumps(original))
        csv_path = self.sources[256][1] / "per_image.csv"
        rows = rr.csv_rows(csv_path)
        rows[0]["dice"] = 0.01
        write_csv(csv_path, rows)
        with self.assertRaisesRegex(ValueError, "recomputed Case metric"):
            self.plan()

    def test_undefined_hd95_is_retained(self):
        run, evaluation = self.sources[256]
        for child in evaluation.iterdir():
            child.unlink()
        evaluation.rmdir()
        self.make_evaluation(run, evaluation, undefined=True)
        existing, _, _, _ = self.plan()
        original = next(r for r in existing if r["image_size"] == 256)
        self.assertEqual(original["metrics"]["hd95"]["case_undefined_count"], 1)
        self.assertIsNone(next(r for r in original["per_case"] if r["case_id"] == "2")["hd95_px"])

    def test_sequential_children_complete_summary_and_safe_reuse(self):
        existing, jobs, manifest_hash, validation = self.plan()
        before = {str(p): rr.sha256(p) for size in self.sources for base in self.sources[size]
                  for p in base.rglob("*") if p.is_file()}
        calls = []
        def child(command, cwd, env, check):
            self.assertEqual(Path(cwd), self.root)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertTrue(check)
            self.assertIsNone(os.environ.get("RESOLUTION_TEST_SHOULD_NOT_EXIST"))
            if "scripts/train_baseline.py" in command:
                config = rr.read_json(Path(command[command.index("--config") + 1]))
                self.make_training(Path(command[command.index("--run-dir") + 1]), config)
                calls.append("train")
            else:
                self.assertEqual(command[command.index("--split") + 1], "val")
                run = Path(command[command.index("--checkpoint") + 1]).parent
                self.make_evaluation(run, Path(command[command.index("--output-dir") + 1]))
                calls.append("val")
            return subprocess.CompletedProcess(command, 0)
        env = {"CUDA_VISIBLE_DEVICES": "3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        with contextlib.redirect_stdout(io.StringIO()):
            completed = rr.execute_jobs(self.root, jobs, manifest_hash, validation, env, runner=child)
        self.assertEqual(calls, ["train", "val"] * 4)
        summary = rr.summarize_runs(existing + completed, manifest_hash)
        self.assertAlmostEqual(summary["by_resolution"]["256"]["dice"]["mean"], 0.701)
        self.assertAlmostEqual(summary["by_resolution"]["256"]["dice"]["sample_std"], 0.001)
        self.assertAlmostEqual(summary["by_resolution"]["384"]["dice"]["mean"], 0.711)
        self.assertEqual(summary["leading_resolution_by_mean_validation_case_dice"], 384)
        for pair in summary["paired_seed_dice_differences"]:
            self.assertAlmostEqual(pair["case_dice_384_minus_256"], 0.01)
        self.assertEqual(before, {p: rr.sha256(Path(p)) for p in before})
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            rr.summarize_runs(existing + completed[:-1], manifest_hash)
        def forbidden(*args, **kwargs):
            self.fail("Completed training/evaluation must be reused")
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = rr.execute_jobs(self.root, jobs, manifest_hash, validation, env, runner=forbidden)
        self.assertEqual(len(repeated), 4)

    def test_plan_cli_is_read_only_and_starts_no_children(self):
        before = {str(p): rr.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        command = [sys.executable, str(ROOT / "scripts/run_resolution_repeats.py"),
                   "--project-root", str(self.root), "--plan",
                   "--baseline-val-dir", str(self.sources[256][1]),
                   "--candidate-val-dir", str(self.sources[384][1])]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("No files written and no GPU job started", result.stdout)
        self.assertIn("--split val", result.stdout)
        self.assertNotIn("--split test", result.stdout)
        self.assertEqual(before, {str(p): rr.sha256(p) for p in self.root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
