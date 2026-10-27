"""Research-critical checks; optional torch tests run when torch/MONAI exist."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from breastseg.metrics import binary_metrics, hd95, summarize
from breastseg.preprocess import letterbox
from breastseg.splits import MANIFEST_FIELDS, prepare, sha256, validate_manifest


METADATA_DIR = Path(os.environ.get("BUSBRA_METADATA_DIR", "data/raw/BUSBRA"))
HAS_TORCH = (importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("monai") is not None)


class DimensionAuditTests(unittest.TestCase):
    """CSV dimensions must not override a successfully verified PNG pair."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "BUSBRA"
        (self.root / "Images").mkdir(parents=True)
        (self.root / "Masks").mkdir()
        self.output = Path(self.temp.name) / "processed"
        records = []
        for case, fold in ((1, 1), (2, 2), (856, 3)):
            name = f"bus_{case:04d}-l"
            actual_width, actual_height = 40, 32
            image = Image.new("L", (actual_width, actual_height), color=90)
            mask = np.zeros((actual_height, actual_width), dtype=np.uint8)
            mask[8:20, 12:28] = 255
            image.save(self.root / "Images" / f"{name}.png")
            Image.fromarray(mask).save(self.root / "Masks" / f"mask_{case:04d}-l.png")
            records.append({
                "ID": name, "Case": str(case), "Pathology": "benign",
                "Width": str(actual_width),
                "Height": str(actual_height + (case == 856)),
                "Device": "synthetic", "_fold": fold,
            })
        fields = ["ID", "Case", "Pathology", "Width", "Height", "Device"]
        with (self.root / "bus_data.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: r[key] for key in fields} for r in records)
        for k in (5, 10):
            fields = ["ID", "Pathology", "kFold"] + [f"valid_{f}" for f in range(1, k + 1)]
            with (self.root / f"{k}-fold-cv.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in records:
                    writer.writerow({
                        "ID": row["ID"], "Pathology": row["Pathology"], "kFold": row["_fold"],
                        **{f"valid_{f}": "NaN" if f == row["_fold"] else "1"
                           for f in range(1, k + 1)},
                    })

    def test_matching_png_sizes_replace_csv_dimensions_and_preserve_original(self):
        before = sha256(self.root / "bus_data.csv")
        rows, report = prepare(self.root, self.output)
        selected = next(r for r in rows if r["id"] == "bus_0856-l")
        self.assertEqual((selected["width"], selected["height"]), (40, 32))
        self.assertEqual(sha256(self.root / "bus_data.csv"), before)
        self.assertEqual(report["dimension_check"], {
            "status": "checked",
            "manifest_dimension_source": "decoded_matching_png_pair",
            "discrepancy_count": 1,
            "discrepancies": [{
                "id": "bus_0856-l", "metadata_width": 40, "metadata_height": 33,
                "png_width": 40, "png_height": 32,
            }],
            "policy": report["dimension_check"]["policy"],
        })
        self.assertTrue(report["baseline_protocol"]["image_files_fully_checked"])
        self.assertEqual(set(report["baseline_protocol"]["case_overlap_counts"].values()), {0})
        saved = json.loads((self.output / "metadata_audit.json").read_text())
        self.assertEqual(saved["dimension_check"], report["dimension_check"])

    def test_image_mask_size_disagreement_still_fails(self):
        Image.new("L", (40, 31), color=255).save(self.root / "Masks/mask_0856-l.png")
        with self.assertRaisesRegex(ValueError, "image/mask shapes differ"):
            prepare(self.root, self.output)

    def test_corrupt_png_still_fails(self):
        (self.root / "Images/bus_0856-l.png").write_bytes(b"corrupt synthetic PNG")
        with self.assertRaises(OSError):
            prepare(self.root, self.output)


class PreprocessingTests(unittest.TestCase):
    def test_letterbox_preserves_geometry_binary_labels_and_valid_region(self):
        image = Image.fromarray(np.full((10, 20), 128, dtype=np.uint8))
        native_mask = np.zeros((10, 20), dtype=np.uint8)
        native_mask[2:8, 5:15] = 255
        item = letterbox(image, Image.fromarray(native_mask), size=32)
        self.assertEqual(item["layout"], {
            "native_h": 10, "native_w": 20, "resized_h": 16,
            "resized_w": 32, "top": 8, "left": 0,
        })
        self.assertEqual(item["image"].shape, (32, 32))
        self.assertEqual(item["valid"].sum(), 16 * 32)
        self.assertTrue(np.all(item["image"][:8] == 0))
        self.assertTrue(np.all(item["mask"][item["valid"] == 0] == 0))
        self.assertTrue(set(np.unique(item["mask"])).issubset({0.0, 1.0}))
        np.testing.assert_array_equal(item["native_mask"], native_mask > 0)
        self.assertAlmostEqual(float(item["image"][10, 10]), 128 / 255, places=6)

    def test_rejects_nonsquare_network_stride_and_nonbinary_targets(self):
        image = Image.new("L", (20, 10))
        with self.assertRaises(ValueError):
            letterbox(image, size=250)
        with self.assertRaises(ValueError):
            letterbox(image, Image.new("L", (20, 10), color=127), size=32)
        with self.assertRaises(ValueError):
            letterbox(image, Image.new("L", (21, 10)), size=32)

    def test_a_small_lesion_cannot_silently_disappear(self):
        image = Image.new("L", (400, 200))
        mask = np.zeros((200, 400), dtype=np.uint8)
        mask[0, 0] = 255
        with self.assertRaisesRegex(ValueError, "Lesion disappears"):
            letterbox(image, Image.fromarray(mask), size=32)


class MetricTests(unittest.TestCase):
    def test_analytic_overlap(self):
        target = np.zeros((5, 5), dtype=bool)
        target[1:3, 1:3] = True
        prediction = np.zeros_like(target)
        prediction[1, 1:4] = True
        result = binary_metrics(prediction, target)
        self.assertAlmostEqual(result["dice"], 4 / 7)
        self.assertAlmostEqual(result["iou"], 2 / 5)
        self.assertAlmostEqual(result["sensitivity"], 1 / 2)
        self.assertAlmostEqual(result["precision"], 2 / 3)

    def test_empty_prediction_is_a_dice_failure_and_undefined_distance(self):
        empty = np.zeros((8, 8), dtype=bool)
        target = empty.copy()
        target[3:5, 3:5] = True
        result = binary_metrics(empty, target, include_hd95=True)
        self.assertEqual(result["dice"], 0.0)
        self.assertEqual(result["iou"], 0.0)
        self.assertEqual(result["sensitivity"], 0.0)
        self.assertIsNone(result["hd95_px"])
        both_empty = binary_metrics(empty, empty, include_hd95=True)
        self.assertEqual(both_empty["dice"], 1.0)
        self.assertEqual(both_empty["hd95_px"], 0.0)

    def test_hd95_uses_native_euclidean_pixels(self):
        target = np.zeros((16, 16), dtype=bool)
        prediction = target.copy()
        target[5, 5] = True
        prediction[8, 9] = True
        self.assertEqual(hd95(target, target), 0.0)
        self.assertEqual(hd95(prediction, target), 5.0)

    def test_cases_have_equal_weight_despite_different_image_counts(self):
        def row(case, value, distance):
            return {
                "case_id": case, "dice": value, "iou": value,
                "sensitivity": value, "precision": value,
                "empty_prediction": False, "empty_target": False,
                "hd95_px": distance,
            }
        rows = [row("A", 1.0, 0.0), row("A", 0.0, None), row("B", 0.0, 3.0)]
        result = summarize(rows, bootstrap=100, seed=42)
        self.assertAlmostEqual(result["image_macro"]["dice"], 1 / 3)
        self.assertEqual(result["case_macro"]["dice"], 0.25)
        self.assertEqual(result["hd95"]["image_undefined_count"], 1)
        self.assertEqual(result["hd95"]["case_undefined_count"], 1)
        self.assertEqual(result["hd95"]["case_mean_fully_defined_px"], 3.0)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result, summarize(rows, bootstrap=100, seed=42))


@unittest.skipUnless((METADATA_DIR / "bus_data.csv").is_file(), "BUS-BRA metadata not available")
class UploadedMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rows, self.report = prepare(
            Path(self.temp.name) / "not_downloaded", Path(self.temp.name) / "prepared",
            metadata_dir=METADATA_DIR, metadata_only=True)

    def test_expected_case_counts_and_strict_separation(self):
        self.assertEqual(self.report["images"], 1875)
        self.assertEqual(self.report["cases"], 1064)
        actual = self.report["baseline_protocol"]
        self.assertEqual(actual["splits"]["train"]["images"], 1489)
        self.assertEqual(actual["splits"]["train"]["cases"], 851)
        self.assertEqual(actual["splits"]["val"]["images"], 193)
        self.assertEqual(actual["splits"]["val"]["cases"], 107)
        self.assertEqual(actual["splits"]["test"]["images"], 193)
        self.assertEqual(actual["splits"]["test"]["cases"], 106)
        self.assertEqual(set(actual["case_overlap_counts"].values()), {0})
        self.assertFalse(actual["image_files_fully_checked"])
        self.assertFalse(actual["uses_valid_flags"])
        self.assertEqual(self.report["official_partitions"]["10"]["folds"][0]
                         ["cases_with_both_validation_flags"], 244)
        self.assertEqual(self.report["official_partitions"]["5"]["folds"][0]
                         ["cases_with_both_validation_flags"], 210)

    def test_a_single_related_image_cannot_cross_splits(self):
        rows = copy.deepcopy(self.rows)
        first = next(r for r in rows if r["case_id"] == "1")
        other = next(r for r in rows if r["case_id"] == "1" and r["id"] != first["id"])
        other["split"] = "test" if first["split"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "spans inconsistent split"):
            validate_manifest(rows)

    def test_ambiguous_ids_and_absolute_paths_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(self.rows + [self.rows[0]])
        rows = copy.deepcopy(self.rows)
        rows[0]["image_path"] = "../other_dataset/lesion.png"
        with self.assertRaisesRegex(ValueError, "relative"):
            validate_manifest(rows)

    def test_input_row_order_does_not_change_assignments(self):
        metadata_dir = Path(self.temp.name) / "shuffled"
        metadata_dir.mkdir()
        for name in ("bus_data.csv", "5-fold-cv.csv", "10-fold-cv.csv"):
            with (METADATA_DIR / name).open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = reader.fieldnames
            with (metadata_dir / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(reversed(rows))
        shuffled, _ = prepare(Path(self.temp.name) / "none",
                              Path(self.temp.name) / "shuffled_result",
                              metadata_dir=metadata_dir, metadata_only=True)
        self.assertEqual(self.rows, shuffled)


@unittest.skipUnless(HAS_TORCH, "PyTorch/MONAI unavailable in this runtime")
class TorchIntegrationTests(unittest.TestCase):
    def test_cpu_training_evaluation_prediction_and_smoke_checkpoint_rejection(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "Images").mkdir(parents=True)
            (data / "Masks").mkdir()
            rows = []
            for case, split, fold in ((1, "train", 3), (2, "val", 2), (3, "test", 1)):
                for view, (height, width) in zip(("l", "r"), ((32, 40), (36, 48))):
                    name = f"bus_{case:04d}-{view}"
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[height // 3:2 * height // 3, width // 3:2 * width // 3] = 255
                    image = np.where(mask > 0, 170, 60).astype(np.uint8)
                    Image.fromarray(image).save(data / "Images" / f"{name}.png")
                    Image.fromarray(mask).save(data / "Masks" / f"mask_{case:04d}-{view}.png")
                    rows.append({
                        "id": name, "case_id": str(case), "split": split, "fold": fold,
                        "pathology": "benign", "device": "synthetic",
                        "width": width, "height": height,
                        "image_path": f"Images/{name}.png",
                        "mask_path": f"Masks/mask_{case:04d}-{view}.png",
                    })
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            config = json.loads((project_root / "configs/unet_baseline.json").read_text())
            config.update(data_root=str(data), manifest=str(manifest),
                          run_dir=str(root / "training"), image_size=32,
                          batch_size=2, num_workers=0, epochs=2, patience=2)
            config["model"]["channels"] = [4, 8, 16, 32, 64]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1")

            def run_script(name, *args, success=True):
                result = subprocess.run(
                    [sys.executable, str(project_root / "scripts" / name), *map(str, args)],
                    cwd=project_root, env=environment, capture_output=True, text=True,
                    timeout=90)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            run_script("train_baseline.py", "--config", config_path, "--device", "cpu")
            checkpoint = root / "training/best.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((root / "training/last.pt").is_file())
            with (root / "training/history.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            run_script("evaluate_baseline.py", "--checkpoint", checkpoint,
                       "--split", "val", "--device", "cpu", "--output-dir", root / "evaluation")
            result = json.loads((root / "evaluation/metrics.json").read_text())
            self.assertEqual(result["images"], 2)
            self.assertEqual(result["cases"], 1)
            self.assertTrue(0 <= result["case_macro"]["dice"] <= 1)
            self.assertTrue((root / "evaluation/predictions.png").is_file())
            output = root / "predicted_mask.png"
            run_script("predict_baseline.py", "--checkpoint", checkpoint,
                       "--image", data / "Images/bus_0003-l.png",
                       "--output", output, "--device", "cpu")
            with Image.open(output) as prediction:
                self.assertEqual(prediction.size, (40, 32))
                self.assertTrue(set(np.unique(np.asarray(prediction))).issubset({0, 255}))
            run_script("train_baseline.py", "--config", config_path, "--device", "cpu",
                       "--smoke", "--run-dir", root / "smoke")
            rejected = run_script(
                "evaluate_baseline.py", "--checkpoint", root / "smoke/best.pt",
                "--device", "cpu", "--output-dir", root / "invalid", success=False)
            self.assertIn("Smoke checkpoints", rejected.stderr)

    def test_padding_has_zero_effect_on_loss_and_gradients(self):
        import torch
        from breastseg.torch_pipeline import masked_dice_bce
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 6:10, 6:10] = 1
        valid = torch.zeros_like(target)
        valid[:, :, 4:12, 4:12] = 1
        a = torch.zeros_like(target, requires_grad=True)
        b = torch.where(valid.bool(), torch.zeros_like(target), torch.full_like(target, 80))
        self.assertAlmostEqual(masked_dice_bce(a, target, valid).item(),
                               masked_dice_bce(b, target, valid).item(), places=7)
        masked_dice_bce(a, target, valid).backward()
        self.assertTrue(torch.all(a.grad[valid == 0] == 0).item())

    def test_real_monai_forward_backward_and_weights_only_checkpoint(self):
        import torch
        from breastseg.torch_pipeline import build_model, masked_dice_bce
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = {"model": {"channels": [4, 8, 16, 32, 64],
                            "strides": [2, 2, 2, 2], "num_res_units": 2,
                            "norm": "INSTANCE"}}
        model = build_model(config)
        image = torch.rand(2, 1, 32, 32)
        target = torch.zeros_like(image)
        target[:, :, 12:20, 12:20] = 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = next(model.parameters()).detach().clone()
        loss = masked_dice_bce(model(image), target, torch.ones_like(target))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.parameters())))
        model.eval()
        with torch.inference_mode():
            expected = model(image)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"config": config, "model_state_dict": model.state_dict()}, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored = build_model(payload["config"]).eval()
            restored.load_state_dict(payload["model_state_dict"])
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)

    def test_variable_native_sizes_collate_and_restore(self):
        import torch
        from breastseg.torch_pipeline import SegmentationDataset, collate_batch, score_batch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Images").mkdir()
            (root / "Masks").mkdir()
            rows = []
            for index, (height, width) in enumerate(((10, 20), (25, 15))):
                name = f"example_{index}"
                image = np.full((height, width), 90, dtype=np.uint8)
                mask = np.zeros_like(image)
                mask[2:5, 3:6] = 255
                Image.fromarray(image).save(root / "Images" / f"{name}.png")
                Image.fromarray(mask).save(root / "Masks" / f"{name}.png")
                rows.append({"id": name, "case_id": str(index), "pathology": "benign",
                             "device": "synthetic", "width": width, "height": height,
                             "image_path": f"Images/{name}.png",
                             "mask_path": f"Masks/{name}.png"})
            dataset = SegmentationDataset(rows, root, image_size=32)
            batch = collate_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["image"].shape), (2, 1, 32, 32))
            logits = torch.full_like(batch["image"], -20)
            scores, predictions = score_batch(logits, batch, include_hd95=True)
            self.assertEqual(predictions[0].shape, (10, 20))
            self.assertEqual(predictions[1].shape, (25, 15))
            self.assertTrue(all(r["dice"] == 0 and r["hd95_px"] is None for r in scores))


if __name__ == "__main__":
    unittest.main()
