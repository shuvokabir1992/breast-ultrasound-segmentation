"""CPU checks for RGB masks, external cohort aggregation and frozen inference."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image

from breastseg import busuclm_external as ext
import evaluate_busuclm_external as runner


class ExternalChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "nested" / "BUS-UCLM"
        (self.root / "images").mkdir(parents=True)
        (self.root / "masks").mkdir()
        self.source = [
            ["ABCD_000.png", "32x24", "Benign", "No", "No", "No"],
            ["ABCD_001.png", "36x24", "Malignant", "Y", "Yes", "No"],
            ["EFGH_000.png", "32x30", "Benign", "No", "No", "No"],
            ["IJKL_000.png", "40x24", "Normal", "No", "No", "Yes"],
        ]
        for index, row in enumerate(self.source):
            width, height = map(int, row[1].split("x"))
            image = np.random.default_rng(index).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            mask = np.zeros_like(image)
            if row[2] != "Normal":
                mask[5:10, 8:15] = (0, 255, 0) if row[2] == "Benign" else (255, 0, 0)
            Image.fromarray(image).save(self.root / "images" / row[0])
            Image.fromarray(mask).save(self.root / "masks" / row[0])
        with (self.root / "INFO.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(ext.FIELDS)
            writer.writerows(self.source)

    def audit(self):
        root, manifest, report = ext.audit_dataset(self.base / "nested", require_full=False)
        self.assertTrue(report["valid"], report["issues"])
        return root, manifest, report

    def scores(self):
        root, manifest, _ = self.audit()
        scored = []
        for row in manifest["records"]:
            _, truth, _ = ext.decode_pair(root, row, verify_hashes=True)
            prediction = truth.copy()
            if row["id"] == "ABCD_001":
                prediction[:] = False
            scored.append(ext.score_prediction(row, prediction, truth))
        return manifest["records"], scored

    def reviewed_fixture(self, image_id, policy):
        row = next(r for r in ext.metadata(self.root / "INFO.csv") if r["id"] == image_id)
        path = self.root / row["mask_path"]
        with Image.open(path) as source:
            enlarged = Image.new("RGB", (source.width + 9, source.height + 7))
            enlarged.paste(source, (3, 2))
            enlarged.save(path)
        with Image.open(self.root / row["image_path"]) as image, Image.open(path) as mask:
            rule = {"image_size_wh": list(image.size), "mask_size_wh": list(mask.size),
                    "image_sha256": ext.sha256(self.root / row["image_path"]),
                    "mask_sha256": ext.sha256(path), "label": row["label"], "policy": policy}
        return row, rule

    def test_only_reviewed_empty_masks_can_use_native_image_grid(self):
        row, rule = self.reviewed_fixture("IJKL_000", "empty_reference_on_native_image_grid")
        original = (self.root / row["mask_path"]).read_bytes()
        with mock.patch.dict(ext.GEOMETRY_CASES, {row["id"]: rule}):
            image, truth, details = ext.decode_pair(self.root, row)
            self.assertEqual(truth.shape, (image.height, image.width))
            self.assertFalse(truth.any())
            self.assertEqual(details["reference_policy"], "verified_empty_on_native_image_grid")
            self.assertNotEqual(details["source_mask_width"], details["width"])
            _, manifest, report = self.audit()
            self.assertEqual(report["empty_reference_native_grid_ids"], [row["id"]])
            audited = next(r for r in manifest["records"] if r["id"] == row["id"])
            item = runner.ExternalDataset([audited], self.root, 32)[0]
            self.assertEqual(item["native_mask"].shape, truth.shape)
            self.assertEqual(tuple(item["image"].shape), (1, 32, 32))
        self.assertEqual(original, (self.root / row["mask_path"]).read_bytes())

    def test_unknown_empty_mismatch_is_still_an_error(self):
        row, _ = self.reviewed_fixture("IJKL_000", "empty_reference_on_native_image_grid")
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            ext.decode_pair(self.root, row)
        _, _, report = ext.audit_dataset(self.root, require_full=False)
        self.assertFalse(report["valid"])

    def test_reviewed_positive_geometry_is_quarantined_and_stays_verified(self):
        row, rule = self.reviewed_fixture("ABCD_000", "quarantine_unresolved_positive_geometry")
        with mock.patch.dict(ext.GEOMETRY_CASES, {row["id"]: rule}):
            with self.assertRaises(ext.UnresolvedReferenceGeometry) as caught:
                ext.decode_pair(self.root, row)
            self.assertIsNone(caught.exception.details["target_pixels"])
            self.assertEqual(caught.exception.details["source_target_pixels"], 35)
            root, manifest, report = self.audit()
            self.assertEqual(report["decoded_pairs"], 4)
            self.assertEqual(report["eligible_pairs"], 3)
            self.assertEqual(report["quarantined_pairs"], 1)
            self.assertNotIn(row["id"], {r["id"] for r in manifest["records"]})
            folder = self.base / "audit_with_quarantine"
            folder.mkdir()
            (folder / "audit.json").write_bytes(ext.json_bytes(report))
            (folder / "manifest.json").write_bytes(ext.json_bytes(manifest))
            _, records, _ = ext.load_audit(root, folder, require_full=False)
            self.assertEqual(len(records), 3)
            (root / row["image_path"]).write_bytes(b"changed quarantined source")
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                ext.load_audit(root, folder, require_full=False)

    def test_reviewed_geometry_requires_exact_source_bytes(self):
        row, rule = self.reviewed_fixture("IJKL_000", "empty_reference_on_native_image_grid")
        with mock.patch.dict(ext.GEOMETRY_CASES, {row["id"]: rule}):
            path = self.root / row["image_path"]
            with Image.open(path) as image:
                changed = np.asarray(image).copy()
            changed[0, 0] = (0, 0, 0)
            Image.fromarray(changed).save(path)
            with self.assertRaisesRegex(ValueError, "reviewed geometry source bytes"):
                ext.decode_pair(self.root, row)

    def test_regrid_never_accepts_nonempty_normal_reference(self):
        row, rule = self.reviewed_fixture("IJKL_000", "empty_reference_on_native_image_grid")
        path = self.root / row["mask_path"]
        with Image.open(path) as image:
            changed = np.asarray(image).copy()
        changed[0, 0] = (0, 255, 0)
        Image.fromarray(changed).save(path)
        rule["mask_sha256"] = ext.sha256(path)
        with mock.patch.dict(ext.GEOMETRY_CASES, {row["id"]: rule}):
            with self.assertRaisesRegex(ValueError, "disagree with metadata"):
                ext.decode_pair(self.root, row)

    def test_v2_protocol_records_geometry_policy_and_exclusions(self):
        row, rule = self.reviewed_fixture("ABCD_000", "quarantine_unresolved_positive_geometry")
        with mock.patch.dict(ext.GEOMETRY_CASES, {row["id"]: rule}):
            _, _, report = self.audit()
            protocol = runner.make_protocol(ROOT, report, [])
        self.assertEqual(protocol["protocol_version"], 2)
        self.assertEqual(protocol["source_images"], 4)
        self.assertEqual(protocol["eligible_images"], 3)
        self.assertEqual(protocol["quarantined_records"][0]["id"], row["id"])
        self.assertEqual(protocol["geometry_policy_sha256"], ext.sha256(ext.GEOMETRY_POLICY_PATH))
        self.assertEqual(protocol["primary_resolution"], 384)

    def test_rgb_union_mixed_patient_labels_and_unresolved_flag(self):
        root, manifest, report = self.audit()
        self.assertEqual(report["mixed_lesion_pathology_patients"], ["ABCD"])
        self.assertEqual(report["acquisition_flags"]["doppler_raw"]["Y"], 1)
        self.assertEqual(manifest["records"][1]["doppler_raw"], "Y")
        for row in manifest["records"]:
            _, truth, _ = ext.decode_pair(root, row, True)
            self.assertEqual(int(truth.sum()), 0 if row["label"] == "normal" else 35)

    def test_complete_metadata_guard_rejects_small_fixture(self):
        with self.assertRaisesRegex(ValueError, "differs from the supplied V3"):
            ext.audit_dataset(self.root)

    def test_corrupt_png_and_missing_files_fail_audit(self):
        (self.root / "images" / "ABCD_000.png").write_bytes(b"not a PNG")
        (self.root / "masks" / "EFGH_000.png").unlink()
        _, _, report = ext.audit_dataset(self.root, require_full=False)
        self.assertFalse(report["valid"])
        self.assertTrue(any(r.get("error") == "missing file" for r in report["issues"]))
        self.assertGreaterEqual(len(report["issues"]), 2)

    def test_wrong_mask_colours_and_label_disagreement_fail(self):
        path = self.root / "masks" / "ABCD_000.png"
        mask = np.asarray(Image.open(path)).copy()
        mask[5:10, 8:15] = (255, 255, 255)
        Image.fromarray(mask).save(path)
        _, _, report = ext.audit_dataset(self.root, require_full=False)
        self.assertFalse(report["valid"])
        self.assertIn("unexpected mask colours", report["issues"][0]["error"])
        mask[5:10, 8:15] = (255, 0, 0)
        Image.fromarray(mask).save(path)
        _, _, report = ext.audit_dataset(self.root, require_full=False)
        self.assertIn("disagree with metadata", report["issues"][0]["error"])

    def test_shape_mismatch_and_transparency_fail(self):
        path = self.root / "masks" / "ABCD_000.png"
        Image.new("RGB", (12, 12)).save(path)
        _, _, report = ext.audit_dataset(self.root, require_full=False)
        self.assertIn("dimensions differ", report["issues"][0]["error"])
        image = Image.new("RGBA", (32, 24), (10, 10, 10, 0))
        image.save(self.root / "images" / "ABCD_000.png")
        _, _, report = ext.audit_dataset(self.root, require_full=False)
        self.assertIn("Nonopaque PNG", report["issues"][0]["error"])

    def test_csv_dimension_disagreement_is_audited_without_source_edit(self):
        original = (self.root / "INFO.csv").read_bytes()
        with Image.open(self.root / "images" / "EFGH_000.png") as source:
            source.resize((64, 60)).save(self.root / "images" / "EFGH_000.png")
        with Image.open(self.root / "masks" / "EFGH_000.png") as source:
            source.resize((64, 60), Image.Resampling.NEAREST).save(self.root / "masks" / "EFGH_000.png")
        _, manifest, report = self.audit()
        self.assertTrue(any("CSV dimensions" in r["warning"] for r in report["warnings"]))
        self.assertEqual(next(r for r in manifest["records"] if r["id"] == "EFGH_000")["width"], 64)
        self.assertEqual(original, (self.root / "INFO.csv").read_bytes())

    def test_patient_weighting_and_normals_do_not_inflate_dice(self):
        records, rows = self.scores()
        report = ext.summarize_external(rows)
        positive = report["cohorts"]["all_images"]["lesion_positive"]
        # ABCD: mean(1,0)=0.5. EFGH: 1. Equal-patient result is 0.75.
        self.assertEqual(positive["images"], 3)
        self.assertEqual(positive["patients"], 2)
        self.assertAlmostEqual(positive["patient_macro"]["dice"], 0.75)
        self.assertAlmostEqual(positive["image_macro"]["dice"], 2 / 3)
        self.assertIsNone(rows[-1]["dice"])
        self.assertEqual(report["cohorts"]["all_images"]["normal"]["patient_macro_foreground_fraction"], 0)
        self.assertEqual(report["cohorts"]["plain_bmode"]["images"], 2)
        self.assertEqual(report["cohorts"]["doppler_unresolved"]["images"], 1)
        self.assertEqual(report["by_lesion_label"]["benign"]["patients"], 2)
        self.assertEqual(report["by_lesion_label"]["malignant"]["patients"], 1)
        ext.validate_scores(rows, records)

    def test_empty_lesion_prediction_retains_undefined_hd95(self):
        _, rows = self.scores()
        report = ext.summarize_external(rows)["cohorts"]["all_images"]["lesion_positive"]
        self.assertEqual(report["empty_prediction_images"], 1)
        self.assertEqual(report["hd95"]["image_undefined_count"], 1)
        self.assertEqual(report["hd95"]["patient_undefined_count"], 1)
        self.assertEqual(report["hd95"]["patient_fully_defined_count"], 1)
        self.assertIsNone(next(r for r in rows if r["id"] == "ABCD_001")["hd95_px"])

    def test_normal_foreground_burden_and_any_pixel_rule(self):
        root, manifest, _ = self.audit()
        row = manifest["records"][-1]
        _, truth, _ = ext.decode_pair(root, row, True)
        prediction = np.zeros_like(truth)
        prediction[0, 0] = True
        result = ext.score_prediction(row, prediction, truth)
        summary = ext.normal_summary([result])
        self.assertAlmostEqual(summary["patient_macro_foreground_fraction"], 1 / truth.size)
        self.assertEqual(summary["patient_macro_fraction_images_with_any_foreground"], 1)
        self.assertEqual(summary["patients_with_any_foreground_on_normal_images"], 1)
        self.assertTrue(all(result[k] is None for k in (*ext.OVERLAP, "hd95_px")))

    def test_audit_fingerprints_reject_file_changes(self):
        root, manifest, report = self.audit()
        folder = self.base / "audit"
        folder.mkdir()
        (folder / "audit.json").write_bytes(ext.json_bytes(report))
        (folder / "manifest.json").write_bytes(ext.json_bytes(manifest))
        ext.load_audit(root, folder, require_full=False)
        with self.assertRaisesRegex(ValueError, "complete audited"):
            ext.load_audit(root, folder)
        (root / "images" / "ABCD_000.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since audit"):
            ext.load_audit(root, folder, require_full=False)

    def test_invalid_export_ids_and_metadata_are_rejected(self):
        records, rows = self.scores()
        invalid = copy.deepcopy(rows)
        invalid[0]["id"] = invalid[1]["id"]
        with self.assertRaisesRegex(ValueError, "result IDs"):
            ext.validate_scores(invalid, records)
        invalid = copy.deepcopy(rows)
        invalid[0]["patient_id"] = "WRNG"
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            ext.validate_scores(invalid, records)
        invalid = copy.deepcopy(rows)
        invalid[-1]["dice"] = 1.0
        with self.assertRaisesRegex(ValueError, "must remain null"):
            ext.validate_scores(invalid, records)

    def test_plan_is_read_only_and_rejects_unrelated_outputs(self):
        _, manifest, _ = self.audit()
        output = self.base / "new_output"
        value = runner.preflight(output, [], manifest["records"], {"primary_resolution": 384})
        self.assertEqual(len(value), 64)
        self.assertFalse(output.exists())
        output.mkdir()
        (output / ".runner.lock").touch()
        runner.preflight(output, [], manifest["records"], {"primary_resolution": 384})
        (output / "unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "unrelated files"):
            runner.preflight(output, [], manifest["records"], {})

    def test_frozen_jobs_require_checkpoint_and_training_manifest_bytes(self):
        project = self.base / "project"
        frozen = []
        for size, seed, epoch, run_name, _ in ext.FROZEN:
            run = project / "runs" / run_name
            run.mkdir(parents=True)
            (run / "manifest.csv").write_bytes(b"fixture training manifest\n")
            checkpoint = run / "best.pt"
            checkpoint.write_bytes(f"fixture checkpoint {size} {seed}".encode())
            frozen.append((size, seed, epoch, run_name, ext.sha256(checkpoint)))
        first = project / "runs" / frozen[0][3]
        with mock.patch.object(runner, "TRAIN_MANIFEST_SHA256", ext.sha256(first / "manifest.csv")):
            jobs = runner.make_jobs(project, frozen=frozen)
            self.assertEqual(len(jobs), 6)
            self.assertEqual({(r["image_size"], r["seed"]) for r in jobs},
                             {(s, seed) for s in (256, 384) for seed in (42, 43, 44)})
            saved = (first / "best.pt").read_bytes()
            (first / "best.pt").write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                runner.make_jobs(project, frozen=frozen)
            (first / "best.pt").write_bytes(saved)
            (first / "manifest.csv").write_bytes(b"different split")
            with self.assertRaisesRegex(ValueError, "training manifest mismatch"):
                runner.make_jobs(project, frozen=frozen)

    def test_aggregate_never_selects_from_external_scores(self):
        _, rows = self.scores()
        source = ext.summarize_external(rows)
        results = []
        for size in (256, 384):
            for seed in (42, 43, 44):
                metrics = copy.deepcopy(source)
                metrics["cohorts"]["all_images"]["lesion_positive"]["patient_macro"]["dice"] = 0.9 if size == 256 else 0.5
                results.append({"image_size": size, "seed": seed, "metrics": metrics})
        report = ext.aggregate_runs(results)
        self.assertEqual(report["selected_resolution"], 384)
        self.assertEqual(report["by_resolution"]["384"]["role"], "primary")
        self.assertGreater(report["by_resolution"]["256"]["metrics"]["lesion_positive_patient_dice"]["mean"],
                           report["by_resolution"]["384"]["metrics"]["lesion_positive_patient_dice"]["mean"])

    def test_cpu_monai_inference_reuse_and_source_preservation(self):
        import torch
        from breastseg.torch_pipeline import build_model
        torch.set_num_threads(1)
        root, manifest, _ = self.audit()
        records = manifest["records"]
        original = {p: ext.sha256(p) for p in root.rglob("*") if p.is_file()}
        config = {"image_size": 32, "seed": 42, "threshold": 0.5,
                  "model": {"channels": [2, 4], "strides": [2], "num_res_units": 1, "norm": "INSTANCE"}}
        model = build_model(config)
        path = self.base / "fixture.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config, "smoke": False,
                    "epoch": 1, "manifest_sha256": ext.TRAIN_MANIFEST_SHA256}, path)
        job = {"image_size": 32, "seed": 42, "epoch": 1, "checkpoint": path,
               "checkpoint_sha256": ext.sha256(path), "name": "fixture"}
        output = self.base / "inference"
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.evaluate_one(job, records, root, output, "fixture-protocol", "cpu", 2, 0)
        self.assertEqual(result["metrics"]["images"], 4)
        self.assertEqual(result["metrics"]["cohorts"]["all_images"]["normal"]["images"], 1)
        self.assertEqual(result, runner.read_result(output, job, records, "fixture-protocol"))
        for row in records:
            with Image.open(output / "predicted_masks" / f"{row['id']}.png") as prediction:
                self.assertEqual(prediction.size, (row["width"], row["height"]))
                self.assertTrue(set(np.unique(prediction).tolist()).issubset({0, 255}))
        self.assertEqual(original, {p: ext.sha256(p) for p in original})
        self.assertEqual(job["checkpoint_sha256"], ext.sha256(path))
        with self.assertRaisesRegex(ValueError, "identity"):
            runner.read_result(output, job, records, "different-protocol")
        mask_path = output / "predicted_masks" / f"{records[0]['id']}.png"
        original_mask = mask_path.read_bytes()
        mask_path.write_bytes(b"changed prediction")
        with self.assertRaisesRegex(ValueError, "file changed"):
            runner.read_result(output, job, records, "fixture-protocol")
        mask_path.write_bytes(original_mask)
        (output / "per_image.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "file changed"):
            runner.read_result(output, job, records, "fixture-protocol")

    def test_network_input_uses_no_reference_mask(self):
        import torch
        root, manifest, _ = self.audit()
        row = manifest["records"][0]
        first = runner.ExternalDataset([row], root, 32)[0]
        path = root / row["mask_path"]
        mask = np.zeros((row["height"], row["width"], 3), dtype=np.uint8)
        mask[1:3, 1:3] = (0, 255, 0)
        Image.fromarray(mask).save(path)
        _, _, details = ext.decode_pair(root, row)
        replacement = {**row, **details}
        second = runner.ExternalDataset([replacement], root, 32)[0]
        self.assertTrue(torch.equal(first["image"], second["image"]))
        self.assertFalse(np.array_equal(first["native_mask"], second["native_mask"]))


if __name__ == "__main__":
    unittest.main()
