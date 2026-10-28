"""CPU checks of frozen test orchestration; no real training or CUDA claims."""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_resolution_repeats as fixtures
import run_resolution_repeats as rr
import evaluate_frozen_resolution_test as frozen


class FrozenTestChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ResolutionRepeatTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        # Add a second synthetic test Case so both pathology groups are represented.
        extra = dict(next(r for r in self.fixture.records if r["split"] == "test"))
        extra.update({"id": "bus_0005-s", "case_id": "5", "pathology": "malignant",
                      "image_path": "Images/bus_0005-s.png", "mask_path": "Masks/bus_0005-s.png"})
        self.fixture.records.append(extra)
        fixtures.write_csv(self.fixture.manifest, self.fixture.records)
        self.fixture.hash = rr.sha256(self.fixture.manifest)
        for run, evaluation in self.fixture.sources.values():
            (run / "manifest.csv").write_bytes(self.fixture.manifest.read_bytes())
            environment = rr.read_json(run / "environment.json")
            environment["manifest_sha256"] = self.fixture.hash
            (run / "environment.json").write_text(json.dumps(environment))
            metrics = rr.read_json(evaluation / "metrics.json")
            metrics["manifest_sha256"] = self.fixture.hash
            (evaluation / "metrics.json").write_text(json.dumps(metrics))
        existing, jobs, manifest_hash, validation = self.fixture.plan()
        def child(command, cwd, env, check):
            if "scripts/train_baseline.py" in command:
                config = rr.read_json(Path(command[command.index("--config") + 1]))
                self.fixture.make_training(Path(command[command.index("--run-dir") + 1]), config)
            else:
                run = Path(command[command.index("--checkpoint") + 1]).parent
                self.fixture.make_evaluation(run, Path(command[command.index("--output-dir") + 1]))
            return subprocess.CompletedProcess(command, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            completed = rr.execute_jobs(self.root, jobs, manifest_hash, validation, {}, runner=child)
        self.source = rr.summarize_runs(existing + completed, manifest_hash)
        self.expected = {(r["image_size"], r["seed"]): r["metrics"]["checkpoint_sha256"] for r in self.source["runs"]}

    def plan(self):
        return frozen.make_plan(self.source, self.root, self.expected, self.fixture.hash)

    def save_test(self, job, undefined=False):
        folder = job["output_dir"]
        folder.mkdir(parents=True)
        rows = []
        for reference in self.fixture.records:
            if reference["split"] != "test":
                continue
            # Deliberately make 256 better on synthetic test: selection must stay 384.
            dice = 0.80 if job["image_size"] == 256 else 0.70
            row = {k: reference[k] for k in ("id", "case_id", "pathology", "device")}
            row.update({"dice": dice, "iou": dice / (2 - dice), "sensitivity": dice,
                        "precision": dice, "empty_prediction": False, "empty_target": False,
                        "hd95_px": 5.0})
            rows.append(row)
        if undefined:
            rows[0].update({"dice": 0, "iou": 0, "sensitivity": 0, "precision": 0,
                            "empty_prediction": True, "hd95_px": None})
        case_macro = {key: statistics.mean(r[key] for r in rows) for key in rr.METRICS}
        groups = {}
        for column, name in (("pathology", "by_pathology"), ("device", "by_device")):
            groups[name] = {}
            for value in {r[column] for r in rows}:
                selected = [r for r in rows if r[column] == value]
                groups[name][value] = {"cases": len(selected), "case_macro": {
                    key: statistics.mean(r[key] for r in selected) for key in rr.METRICS}}
        defined = [r["hd95_px"] for r in rows if r["hd95_px"] is not None]
        metrics = {"split": "test", "images": len(rows), "cases": len(rows),
                   "checkpoint_sha256": job["checkpoint_sha256"], "checkpoint_epoch": job["checkpoint_epoch"],
                   "manifest_sha256": self.fixture.hash, "threshold": 0.5, "case_macro": case_macro,
                   "hd95": {"case_fully_defined_count": len(defined), "case_undefined_count": len(rows) - len(defined),
                             "case_mean_fully_defined_px": statistics.mean(defined) if defined else None}, **groups}
        (folder / "metrics.json").write_text(json.dumps(metrics))
        fixtures.write_csv(folder / "per_image.csv", rows)

    def test_plan_freezes_validation_winner_and_six_checkpoints(self):
        jobs, test_rows, protocol = self.plan()
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len(test_rows), 2)
        self.assertEqual(protocol["selected_resolution"], 384)
        for job in jobs:
            self.assertEqual(job["command"][job["command"].index("--split") + 1], "test")
            self.assertNotIn("scripts/train_baseline.py", job["command"])

    def test_changed_controls_and_checkpoint_are_rejected(self):
        self.source["runs"][1]["config"]["learning_rate"] = 0.5
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            self.plan()
        self.source["runs"][1]["config"]["learning_rate"] = 0.001
        self.source["runs"][0]["metrics"]["checkpoint_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unexpected checkpoint"):
            self.plan()

    def test_reuse_original_and_no_selection_from_test(self):
        jobs, test_rows, protocol = self.plan()
        original = next(j for j in jobs if j["image_size"] == 256 and j["seed"] == 42)
        original["output_dir"] = self.root / "reports" / "unet_baseline_seed42_test"
        self.save_test(original)
        jobs, test_rows, protocol = self.plan()
        self.assertEqual(jobs[0]["output_dir"], original["output_dir"])
        before = {str(path): rr.sha256(path) for r in self.source["runs"]
                  for path in Path(r["run_dir"]).rglob("*") if path.is_file()}
        calls = []
        def child(command, cwd, env, check):
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            output = Path(command[command.index("--output-dir") + 1])
            job = next(j for j in jobs if j["output_dir"] == output)
            self.save_test(job)
            calls.append(job)
            return subprocess.CompletedProcess(command, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            results = frozen.execute(jobs, test_rows, protocol, self.root,
                                     {"CUDA_VISIBLE_DEVICES": "3"}, runner=child)
        self.assertEqual(len(calls), 5)
        report = frozen.aggregate(results, protocol)
        self.assertGreater(report["by_resolution"]["256"]["dice"]["mean"], report["by_resolution"]["384"]["dice"]["mean"])
        self.assertEqual(report["selected_resolution"], 384)
        self.assertEqual(before, {p: rr.sha256(Path(p)) for p in before})
        def forbidden(*args, **kwargs):
            self.fail("All six verified test evaluations should be reused")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(frozen.execute(jobs, test_rows, protocol, self.root, {}, runner=forbidden)), 6)

    def test_wrong_split_and_checkpoint_identity_are_rejected(self):
        jobs, test_rows, protocol = self.plan()
        job = jobs[0]
        self.save_test(job)
        path = job["output_dir"] / "metrics.json"
        metrics = rr.read_json(path)
        metrics["split"] = "val"
        path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "expected a test"):
            frozen.read_test_result(job, test_rows, self.fixture.hash)
        metrics["split"] = "test"
        metrics["checkpoint_sha256"] = "wrong"
        path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            frozen.read_test_result(job, test_rows, self.fixture.hash)

    def test_test_csv_ids_are_checked(self):
        jobs, test_rows, protocol = self.plan()
        job = jobs[0]
        self.save_test(job)
        path = job["output_dir"] / "per_image.csv"
        rows = rr.csv_rows(path)
        rows[0]["id"] = "not-a-test-image"
        fixtures.write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "image IDs differ"):
            frozen.read_test_result(job, test_rows, self.fixture.hash)

    def test_undefined_distance_is_retained(self):
        jobs, test_rows, protocol = self.plan()
        job = jobs[0]
        self.save_test(job, undefined=True)
        result = frozen.read_test_result(job, test_rows, self.fixture.hash)
        self.assertEqual(result["metrics"]["hd95"]["case_undefined_count"], 1)
        self.assertEqual(sum(r["hd95_px"] is None for r in result["per_case"]), 1)


if __name__ == "__main__":
    unittest.main()
