"""Meaningful checks for the validation-only review and postprocessing comparison."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from review_validation_failures import largest_component, paired_comparison, verify_evaluation
from breastseg.metrics import binary_metrics, summarize
from breastseg.splits import MANIFEST_FIELDS, sha256


class ValidationReviewTests(unittest.TestCase):
    def test_largest_component_removes_detached_false_positive(self):
        truth = np.zeros((48, 48), dtype=bool)
        truth[20:32, 20:32] = True
        prediction = truth.copy()
        prediction[1:5, 1:5] = True
        before = binary_metrics(prediction, truth, True)
        after = binary_metrics(largest_component(prediction), truth, True)
        self.assertLess(before["dice"], 1)
        self.assertGreater(before["hd95_px"], 0)
        self.assertEqual(after["dice"], 1)
        self.assertEqual(after["hd95_px"], 0)

    def test_largest_component_can_remove_real_reference_foreground(self):
        truth = np.zeros((48, 48), dtype=bool)
        truth[20:32, 20:32] = True
        truth[1:6, 1:6] = True
        before = binary_metrics(truth, truth, True)
        after = binary_metrics(largest_component(truth), truth, True)
        self.assertEqual(before["dice"], 1)
        self.assertLess(after["dice"], before["dice"])
        self.assertLess(after["sensitivity"], 1)

    def test_empty_prediction_and_diagonal_connectivity(self):
        empty = np.zeros((8, 8), dtype=bool)
        self.assertFalse(largest_component(empty).any())
        diagonal = np.eye(8, dtype=bool)
        np.testing.assert_array_equal(largest_component(diagonal), diagonal)

    def test_paired_comparison_weights_cases_equally(self):
        before = [{"id": "a-l", "case_id": "a", "dice": 0.5},
                  {"id": "a-r", "case_id": "a", "dice": 0.5},
                  {"id": "b-s", "case_id": "b", "dice": 0.5}]
        after = [{**r, "dice": 0.7 if r["case_id"] == "a" else 0.3} for r in before]
        result = paired_comparison(before, after)
        self.assertAlmostEqual(result["case_dice_delta"], 0)
        self.assertEqual(result["cases_improved"], 1)
        self.assertEqual(result["cases_worsened"], 1)

    def test_test_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "test split rejected"):
            verify_evaluation({"split": "test"}, [], [], "unused")

    def test_complete_cli_native_masks_and_no_test_data_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, run, evaluation, output = (root / n for n in ("data", "run", "eval", "review"))
            for directory in (data / "Images", data / "Masks", run, evaluation / "predicted_masks"):
                directory.mkdir(parents=True)
            records, exported = [], []
            for number, suffix, split in ((1, "s", "train"), (2, "l", "val"),
                                           (2, "r", "val"), (3, "s", "test")):
                image_id = f"bus_{number:04d}-{suffix}"
                row = {"id": image_id, "case_id": str(number), "split": split,
                       "fold": str({"train": 3, "val": 2, "test": 1}[split]),
                       "pathology": "benign", "image_path": f"Images/{image_id}.png",
                       "mask_path": f"Masks/{image_id}.png", "width": 64, "height": 48,
                       "device": "synthetic"}
                records.append(row)
                if split == "test":
                    continue  # No test image or mask exists; reading either would fail.
                truth = np.zeros((48, 64), dtype=bool)
                truth[20:32, 20:32] = True
                if split == "train":
                    truth[1:5, 1:5] = True
                Image.fromarray(truth.astype(np.uint8) * 255).save(data / row["mask_path"])
                Image.fromarray(np.full(truth.shape, 110, dtype=np.uint8)).save(data / row["image_path"])
                if split == "val":
                    prediction = truth.copy()
                    if suffix == "l":
                        prediction[1:5, 1:5] = True
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(evaluation / "predicted_masks" / f"{image_id}.png")
                    exported.append({"id": image_id, "case_id": "2", "pathology": "benign",
                                     "device": "synthetic", **binary_metrics(prediction, truth, True)})
            with (run / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(records)
            (run / "best.pt").write_bytes(b"Synthetic identity only: the review does not load model weights")
            (run / "config.json").write_text(json.dumps({"data_root": str(data), "seed": 42,
                                                        "threshold": 0.5, "image_size": 256}))
            metrics = {**summarize(exported), "split": "val", "checkpoint_epoch": 28,
                       "checkpoint_sha256": sha256(run / "best.pt"),
                       "manifest_sha256": sha256(run / "manifest.csv"), "threshold": 0.5}
            (evaluation / "metrics.json").write_text(json.dumps(metrics))
            with (evaluation / "per_image.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
                writer.writeheader()
                writer.writerows(exported)
            source_files = list(data.rglob("*.png")) + list(evaluation.rglob("*")) + list(run.iterdir())
            before = {str(p): sha256(p) for p in source_files if p.is_file()}
            command = [sys.executable, str(ROOT / "scripts/review_validation_failures.py"),
                       "--run-dir", str(run), "--eval-dir", str(evaluation),
                       "--output-dir", str(output), "--ids", "bus_0002-l", "bus_0002-r",
                       "--compare-largest-component"]
            result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "review_summary.json").read_text())
            self.assertEqual(report["baseline"]["images"], 2)
            self.assertEqual(report["baseline"]["cases"], 1)
            self.assertEqual(report["largest_component"]["case_macro"]["dice"], 1)
            self.assertGreater(report["paired_comparison"]["case_dice_delta"], 0)
            self.assertEqual(report["reference_audit"]["train"]["images_with_multiple_components"], 1)
            self.assertFalse(report["test_images_or_masks_read"])
            with Image.open(output / "failure_overview.png") as image:
                image.load()
                self.assertGreater(image.width, 1000)
            self.assertTrue((output / "review_per_image.csv").exists())
            self.assertTrue((output / "reference_components.csv").exists())
            self.assertEqual(before, {p: sha256(p) for p in before})
            # Tampered exported scores must be rejected, rather than silently rescored.
            exported[0]["dice"] = 0.123
            with (evaluation / "per_image.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
                writer.writeheader()
                writer.writerows(exported)
            command[command.index("--output-dir") + 1] = str(root / "tampered_review")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("recomputed dice differs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
