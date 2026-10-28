"""BUS-UCLM V3 audit and patient-grouped external metrics; no model training."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from breastseg.metrics import binary_metrics


VERSION = 2
GEOMETRY_POLICY_PATH = Path(__file__).with_name("busuclm_geometry_policy.json")
GEOMETRY_CASES = json.loads(GEOMETRY_POLICY_PATH.read_text(encoding="utf-8"))["cases"]
QUARANTINE_IDS = frozenset(k for k, v in GEOMETRY_CASES.items()
                          if v["policy"] == "quarantine_unresolved_positive_geometry")
EMPTY_GRID_IDS = frozenset(k for k, v in GEOMETRY_CASES.items()
                         if v["policy"] == "empty_reference_on_native_image_grid")
INFO_SHA256 = "a9d5356f0639caa1bafa5d46de58c70e81894c39ddb4feff10a7858d87d9e91c"
TRAIN_MANIFEST_SHA256 = "e37a37ea057ff2e93728150bb36a67bcc7f47dcbb3228a78fb3bfe8f3d68820c"
MODEL = {"channels": [32, 64, 128, 256, 512], "strides": [2, 2, 2, 2],
         "num_res_units": 2, "norm": "INSTANCE"}
FROZEN = (
    (256, 42, 28, "unet_baseline_seed42", "4754f8bbc1ae4ed57fffbab4256612b1c6642461fc05fea5fb2e54e15fe360f8"),
    (256, 43, 23, "unet_baseline_256_seed43", "793b98c517e273f9c796dbab50f341672c94cf615e816989ce5bd2b8eb02fb0f"),
    (256, 44, 25, "unet_baseline_256_seed44", "9b97730263c19634412b2d2b3cb1c1b3814ed93a27c3fb5e81c3254a41d2841e"),
    (384, 42, 20, "unet_baseline_384_seed42", "a0d078f02c53ac201058bfbc9d0549b092dc8305d2914e9cf4d52d33410111e6"),
    (384, 43, 29, "unet_baseline_384_seed43", "ff09abace0c861f53a2b1598ac69073be0a319093c13dd820b222fde7bf9d287"),
    (384, 44, 24, "unet_baseline_384_seed44", "bd0f4d95382bb55976a82de3586be30b933f2b1844d703c6eedf78a3149359c4"),
)
OVERLAP = ("dice", "iou", "pixel_recall", "pixel_precision")
FIELDS = ("Image", "Resolution", "Label", "Doppler", "Marks", "Combined")


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_root(path):
    path = Path(path).resolve()
    candidates = [path / "INFO.csv"] if (path / "INFO.csv").is_file() else list(path.rglob("INFO.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one INFO.csv below {path}; found {len(candidates)}")
    root = candidates[0].parent
    if not (root / "images").is_dir() or not (root / "masks").is_dir():
        raise ValueError("INFO.csv must have sibling images/ and masks/ directories")
    return root


def metadata(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected INFO.csv columns: {reader.fieldnames}")
        source = list(reader)
    rows, seen = [], set()
    for source_row in source:
        if set(source_row) != set(FIELDS) or any(not source_row[k] for k in FIELDS):
            raise ValueError("Missing or additional metadata fields")
        name = source_row["Image"]
        if not re.fullmatch(r"[A-Z]{4}_[0-9]{3}\.png", name) or name in seen:
            raise ValueError(f"Invalid or duplicate image filename: {name}")
        seen.add(name)
        if source_row["Label"] not in ("Benign", "Malignant", "Normal"):
            raise ValueError(f"Unknown Label: {source_row}")
        if source_row["Doppler"] not in ("No", "Yes", "Y"):
            raise ValueError(f"Unknown Doppler value: {source_row}")
        if any(source_row[k] not in ("No", "Yes") for k in ("Marks", "Combined")):
            raise ValueError(f"Unknown acquisition flag: {source_row}")
        if not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", source_row["Resolution"]):
            raise ValueError(f"Invalid Resolution: {source_row}")
        rows.append({"id": Path(name).stem, "filename": name, "patient_id": name[:4],
                     "label": source_row["Label"].lower(), "doppler_raw": source_row["Doppler"],
                     "marks_raw": source_row["Marks"], "combined_raw": source_row["Combined"],
                     "csv_resolution": source_row["Resolution"],
                     "image_path": f"images/{name}", "mask_path": f"masks/{name}"})
    return sorted(rows, key=lambda r: r["id"])


def _png(path, expected_hash=None):
    data = Path(path).read_bytes()
    actual_hash = digest(data)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"File changed since audit: {path}")
    with Image.open(io.BytesIO(data)) as source:
        if source.format != "PNG" or getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"Expected a single-frame PNG: {path}")
        source.verify()
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        if source.mode not in ("L", "RGB", "RGBA", "P"):
            raise ValueError(f"Unsupported PNG mode {source.mode}: {path}")
        if source.mode == "RGBA" or "transparency" in source.info:
            if source.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"Nonopaque PNG requires explicit handling: {path}")
        image = source.copy()
    return image, actual_hash


class UnresolvedReferenceGeometry(ValueError):
    """A reviewed, byte-identified source pair is decoded but not evaluable."""
    def __init__(self, details):
        self.details = details
        super().__init__("Nonempty reference geometry has no established mapping to the image")


def decode_pair(root, row, verify_hashes=False):
    image, image_hash = _png(root / row["image_path"], row.get("image_sha256") if verify_hashes else None)
    mask, mask_hash = _png(root / row["mask_path"], row.get("mask_sha256") if verify_hashes else None)
    known = GEOMETRY_CASES.get(row["id"])
    if known is not None:
        if (image_hash != known["image_sha256"] or mask_hash != known["mask_sha256"] or
                list(image.size) != known["image_size_wh"] or list(mask.size) != known["mask_size_wh"] or
                row["label"] != known["label"]):
            raise ValueError(f"{row['id']}: reviewed geometry source bytes or metadata differ; no exception applied")
    if image.size != mask.size and known is None:
        raise ValueError(f"{row['id']}: image/mask dimensions differ: image {image.size}, mask {mask.size}")
    if mask.mode == "L":
        raise ValueError(f"{row['id']}: expected class-coloured RGB mask, found grayscale")
    colours = mask.convert("RGB").getcolors(maxcolors=4)
    allowed = {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
    if colours is None or any(tuple(colour) not in allowed for _, colour in colours):
        raise ValueError(f"{row['id']}: unexpected mask colours; expected black/green/red")
    colour_counts = {tuple(colour): int(count) for count, colour in colours}
    foreground = {colour for colour in colour_counts if colour != (0, 0, 0)}
    expected = {"normal": set(), "benign": {(0, 255, 0)}, "malignant": {(255, 0, 0)}}[row["label"]]
    if foreground != expected:
        raise ValueError(f"{row['id']}: mask colours/emptiness disagree with metadata label {row['label']}")
    rgb = np.asarray(mask.convert("RGB"))
    truth = (rgb[:, :, 0] == 255) | (rgb[:, :, 1] == 255)
    source_target_pixels = int(truth.sum())
    details = {"width": image.width, "height": image.height, "image_mode": image.mode,
               "mask_mode": mask.mode, "image_sha256": image_hash, "mask_sha256": mask_hash,
               "source_mask_width": mask.width, "source_mask_height": mask.height,
               "source_target_pixels": source_target_pixels, "target_pixels": source_target_pixels,
               "reference_policy": "native_aligned",
               "mask_colour_counts": {",".join(map(str, colour)): count for colour, count in sorted(colour_counts.items())}}
    if image.size != mask.size:
        if known["policy"] == "empty_reference_on_native_image_grid" and row["label"] == "normal" and source_target_pixels == 0:
            # An identically zero reference has no foreground coordinates to map.
            # This is an explicit, source-hash-gated exception, never a resize.
            truth = np.zeros((image.height, image.width), dtype=bool)
            details["reference_policy"] = "verified_empty_on_native_image_grid"
        elif known["policy"] == "quarantine_unresolved_positive_geometry" and source_target_pixels > 0:
            details.update(target_pixels=None, reference_policy="quarantine_unresolved_positive_geometry",
                           exclusion_reason="Nonempty source mask and image have different grids; no validated crop/scale/offset mapping is available")
            raise UnresolvedReferenceGeometry(details)
        else:
            raise ValueError(f"{row['id']}: reviewed geometry rule does not match the decoded reference")
    if verify_hashes and any(details[k] != row[k] for k in details):
        raise ValueError(f"{row['id']}: decoded values differ from audited manifest")
    return image, truth, details


def audit_dataset(path, require_full=True):
    root = find_root(path)
    info_hash = sha256(root / "INFO.csv")
    source = metadata(root / "INFO.csv")
    if require_full and (info_hash != INFO_SHA256 or len(source) != 683):
        raise ValueError("INFO.csv differs from the supplied V3 metadata; stop and review the source")
    names = {r["filename"] for r in source}
    issues, warnings, records, quarantined = [], [], [], []
    for folder in ("images", "masks"):
        actual = {p.name for p in (root / folder).iterdir() if p.is_file()}
        for name in sorted(names - actual):
            issues.append({"file": f"{folder}/{name}", "error": "missing file"})
        for name in sorted(actual - names):
            issues.append({"file": f"{folder}/{name}", "error": "file not listed in metadata"})
    for row in source:
        if not all((root / row[k]).is_file() for k in ("image_path", "mask_path")):
            continue
        try:
            _, _, details = decode_pair(root, row)
            records.append({**row, **details})
            if row["csv_resolution"] != f"{details['width']}x{details['height']}":
                warnings.append({"id": row["id"], "warning": "CSV dimensions differ from native image; original CSV unchanged",
                                 "csv": row["csv_resolution"], "png": f"{details['width']}x{details['height']}"})
            if row["doppler_raw"] == "Y":
                warnings.append({"id": row["id"], "warning": "Doppler=Y preserved as unresolved; included in all-images cohort, excluded from No/No B-mode subgroup"})
        except UnresolvedReferenceGeometry as error:
            quarantined.append({**row, **error.details})
        except (OSError, ValueError, SyntaxError) as error:
            issues.append({"id": row["id"], "error": str(error)})
    patients = defaultdict(set)
    for row in source:
        if row["label"] != "normal":
            patients[row["patient_id"]].add(row["label"])
    policy_hash = sha256(GEOMETRY_POLICY_PATH)
    regridded = [r["id"] for r in records if r["reference_policy"] == "verified_empty_on_native_image_grid"]
    if require_full:
        if {r["id"] for r in quarantined} != QUARANTINE_IDS:
            issues.append({"error": "Expected exactly the four reviewed positive-reference quarantines"})
        if set(regridded) != EMPTY_GRID_IDS:
            issues.append({"error": "Expected exactly the nine reviewed empty native-grid references"})
    manifest = {"schema_version": VERSION, "metadata_sha256": info_hash,
                "geometry_policy_sha256": policy_hash, "records": records, "quarantined_records": quarantined}
    report = {"schema_version": VERSION, "dataset": "BUS-UCLM", "version": 3,
              "doi": "10.17632/7fvgj4jsp7.3", "licence": "CC BY 4.0",
              "metadata_sha256": info_hash, "metadata_images": len(source),
              "decoded_pairs": len(records) + len(quarantined), "eligible_pairs": len(records),
              "quarantined_pairs": len(quarantined), "quarantined_records": quarantined,
              "empty_reference_native_grid_ids": regridded, "geometry_policy_sha256": policy_hash,
              "patient_prefixes": len({r['patient_id'] for r in source}),
              "labels": dict(Counter(r["label"] for r in source)),
              "eligible_labels": dict(Counter(r["label"] for r in records)),
              "acquisition_flags": {k: dict(Counter(r[k] for r in source)) for k in ("doppler_raw", "marks_raw", "combined_raw")},
              "decoded_dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
              "mixed_lesion_pathology_patients": sorted(p for p, labels in patients.items() if len(labels) > 1),
              "issues": issues, "warnings": warnings,
              "valid": not issues and len(records) + len(quarantined) == len(source),
              "full_frozen_v3": require_full, "manifest_sha256": digest(json_bytes(manifest)),
              "dimension_rule": "Native image dimensions. Matching source masks remain unchanged; nine byte-identified all-black normal references are represented on their native image grids without interpolation. Four byte-identified nonempty mismatched references are quarantined. All other mismatches fail. Raw files and CSV unchanged."}
    return root, manifest, report


def load_audit(path, audit_dir, verify_files=True, require_full=True):
    root = find_root(path)
    audit_dir = Path(audit_dir)
    report = read_json(audit_dir / "audit.json")
    manifest_path = audit_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not report.get("valid") or report.get("issues") or report["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Invalid or changed audit bundle")
    if manifest["schema_version"] != VERSION or report["schema_version"] != VERSION:
        raise ValueError("Unsupported audit schema")
    if (manifest["geometry_policy_sha256"] != sha256(GEOMETRY_POLICY_PATH) or
            report["geometry_policy_sha256"] != manifest["geometry_policy_sha256"]):
        raise ValueError("Geometry policy changed since audit")
    rows, quarantined = manifest["records"], manifest["quarantined_records"]
    all_records = rows + quarantined
    source = metadata(root / "INFO.csv")
    if (manifest["metadata_sha256"] != sha256(root / "INFO.csv") or
            manifest["metadata_sha256"] != report["metadata_sha256"] or len(source) != len(all_records)):
        raise ValueError("Metadata changed since audit")
    if (report["eligible_pairs"] != len(rows) or report["quarantined_pairs"] != len(quarantined) or
            report["quarantined_records"] != quarantined):
        raise ValueError("Audit eligibility/quarantine counts differ from manifest")
    if require_full and (not report["full_frozen_v3"] or len(all_records) != 683 or len(rows) != 679 or
                         {r["id"] for r in quarantined} != QUARANTINE_IDS or
                         {r["id"] for r in rows if r["reference_policy"] == "verified_empty_on_native_image_grid"} != EMPTY_GRID_IDS or
                         manifest["metadata_sha256"] != INFO_SHA256):
        raise ValueError("External inference requires the complete audited 683-source-image V3 dataset with 679 eligible images and four reviewed quarantines")
    if len({r["id"] for r in all_records}) != len(all_records):
        raise ValueError("Duplicate audit record")
    lookup = {r["id"]: r for r in all_records}
    for expected in source:
        actual = lookup.get(expected["id"], {})
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError("Manifest metadata differs from source CSV")
    for folder in ("images", "masks"):
        if {p.name for p in (root / folder).iterdir() if p.is_file()} != {r["filename"] for r in all_records}:
            raise ValueError("Dataset file inventory changed since audit")
    if verify_files:
        for row in all_records:
            for kind in ("image", "mask"):
                if sha256(root / row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                    raise ValueError(f"File changed since audit: {row[kind + '_path']}")
    return root, rows, report


def score_prediction(row, prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.dtype != bool or truth.dtype != bool or prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Expected equal-sized boolean 2D prediction and reference masks")
    if truth.shape != (row["height"], row["width"]) or int(truth.sum()) != row["target_pixels"]:
        raise ValueError("Reference mask differs from the audit")
    actual = int(truth.sum())
    if (row["label"] == "normal") != (actual == 0):
        raise ValueError("Reference emptiness disagrees with metadata")
    predicted = int(prediction.sum())
    result = {k: row[k] for k in ("id", "patient_id", "label", "doppler_raw", "marks_raw", "combined_raw", "width", "height")}
    result.update(target_pixels=actual, predicted_pixels=predicted, image_pixels=truth.size,
                  foreground_fraction=predicted / truth.size, any_foreground=predicted > 0,
                  empty_prediction=predicted == 0, empty_target=actual == 0)
    if actual:
        value = binary_metrics(prediction, truth, include_hd95=True)
        result.update(dice=value["dice"], iou=value["iou"], pixel_recall=value["sensitivity"],
                      pixel_precision=value["precision"], hd95_px=value["hd95_px"])
    else:
        # Never count empty normal references as perfect segmentation Dice.
        result.update({k: None for k in (*OVERLAP, "hd95_px")})
    return result


def mean(values):
    return statistics.mean(values) if values else None


def _bootstrap(values, n=2000):
    rng = np.random.default_rng(42)
    data = np.asarray(values, dtype=float)
    samples = rng.choice(data, size=(n, len(data)), replace=True).mean(axis=1)
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def positive_summary(rows, bootstrap=0):
    grouped = defaultdict(list)
    for row in rows:
        if row["label"] == "normal":
            raise ValueError("Normal images must not enter lesion-positive overlap averages")
        grouped[row["patient_id"]].append(row)
    patients = []
    for patient_id, views in sorted(grouped.items()):
        patients.append({"patient_id": patient_id, "images": len(views),
                         "benign_images": sum(r["label"] == "benign" for r in views),
                         "malignant_images": sum(r["label"] == "malignant" for r in views),
                         **{k: mean([r[k] for r in views]) for k in OVERLAP},
                         "hd95_px": mean([r["hd95_px"] for r in views]) if all(r["hd95_px"] is not None for r in views) else None})
    defined_images = [r["hd95_px"] for r in rows if r["hd95_px"] is not None]
    defined_patients = [r["hd95_px"] for r in patients if r["hd95_px"] is not None]
    result = {"images": len(rows), "patients": len(patients),
              "image_macro": {k: mean([r[k] for r in rows]) for k in OVERLAP},
              "patient_macro": {k: mean([r[k] for r in patients]) for k in OVERLAP},
              "empty_prediction_images": sum(r["empty_prediction"] for r in rows),
              "hd95": {"units": "native pixels; physical spacing not established",
                       "image_mean_defined_px": mean(defined_images), "image_defined_count": len(defined_images),
                       "image_undefined_count": len(rows) - len(defined_images),
                       "patient_mean_fully_defined_px": mean(defined_patients),
                       "patient_fully_defined_count": len(defined_patients),
                       "patient_undefined_count": len(patients) - len(defined_patients)},
              "per_patient": patients}
    if bootstrap and patients:
        result.update(patient_dice_bootstrap_95ci=_bootstrap([r["dice"] for r in patients], bootstrap),
                      bootstrap_resamples=bootstrap, bootstrap_seed=42,
                      bootstrap_scope="Patient sampling conditional on this fixed checkpoint; excludes training-seed and selection uncertainty")
    return result


def normal_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["label"] != "normal":
            raise ValueError("Only normal-reference images belong in normal-image false-positive reporting")
        grouped[row["patient_id"]].append(row)
    patients = [{"patient_id": patient_id, "images": len(views),
                 "foreground_fraction": mean([r["foreground_fraction"] for r in views]),
                 "fraction_images_with_any_foreground": mean([int(r["any_foreground"]) for r in views]),
                 "any_foreground_on_at_least_one_normal_image": any(r["any_foreground"] for r in views)}
                for patient_id, views in sorted(grouped.items())]
    return {"images": len(rows), "patients": len(patients),
            "image_mean_foreground_fraction": mean([r["foreground_fraction"] for r in rows]),
            "image_fraction_with_any_foreground": mean([int(r["any_foreground"]) for r in rows]),
            "patient_macro_foreground_fraction": mean([r["foreground_fraction"] for r in patients]),
            "patient_macro_fraction_images_with_any_foreground": mean([r["fraction_images_with_any_foreground"] for r in patients]),
            "patients_with_any_foreground_on_normal_images": sum(r["any_foreground_on_at_least_one_normal_image"] for r in patients),
            "per_patient": patients,
            "interpretation": "Any foreground means at least one predicted pixel at threshold 0.5, without filtering. This is not patient diagnostic specificity; patients can also have lesion-positive images."}


def summarize_external(rows):
    cohorts = {
        "all_images": rows,
        "plain_bmode": [r for r in rows if r["doppler_raw"] == "No" and r["combined_raw"] == "No"],
        "unmarked_plain_bmode": [r for r in rows if all(r[k] == "No" for k in ("doppler_raw", "combined_raw", "marks_raw"))],
        "doppler_yes": [r for r in rows if r["doppler_raw"] == "Yes"],
        "doppler_unresolved": [r for r in rows if r["doppler_raw"] not in ("Yes", "No")],
        "combined_yes": [r for r in rows if r["combined_raw"] == "Yes"],
        "marks_yes": [r for r in rows if r["marks_raw"] == "Yes"],
        "marks_no": [r for r in rows if r["marks_raw"] == "No"],
    }
    output = {}
    for name, selected in cohorts.items():
        output[name] = {"images": len(selected), "patients": len({r["patient_id"] for r in selected}),
                        "lesion_positive": positive_summary([r for r in selected if r["label"] != "normal"], 2000 if name == "all_images" else 0),
                        "normal": normal_summary([r for r in selected if r["label"] == "normal"])}
    return {"images": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "cohorts": output,
            "by_lesion_label": {label: positive_summary([r for r in rows if r["label"] == label]) for label in ("benign", "malignant")},
            "aggregation": "Average images within patient, then patients equally; subgroup eligibility is image-level. Lesion-positive and normal-reference metrics remain separate."}


def validate_scores(rows, records):
    expected = {r["id"]: r for r in records}
    if len(rows) != len(expected) or len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(expected):
        raise ValueError("External result IDs differ from the full audited manifest")
    for row in rows:
        reference = expected[row["id"]]
        for k in ("patient_id", "label", "doppler_raw", "marks_raw", "combined_raw", "width", "height", "target_pixels"):
            if row[k] != reference[k]:
                raise ValueError("External result metadata differs from audit")
        n = reference["width"] * reference["height"]
        count = row["predicted_pixels"]
        if type(count) is not int or not 0 <= count <= n or row["image_pixels"] != n:
            raise ValueError("Invalid predicted pixel counts")
        if row["foreground_fraction"] != count / n or row["any_foreground"] != (count > 0) or row["empty_prediction"] != (count == 0):
            raise ValueError("Inconsistent normal-image foreground metrics")
        normal = reference["label"] == "normal"
        if row["empty_target"] != normal:
            raise ValueError("Inconsistent reference emptiness")
        for k in OVERLAP:
            value = row[k]
            if normal:
                if value is not None:
                    raise ValueError("Normal-reference overlap must remain null")
            elif value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid lesion-positive overlap score")
        hd = row["hd95_px"]
        if (normal or count == 0) and hd is not None:
            raise ValueError("Unexpected distance for empty mask")
        if not normal and count > 0 and (hd is None or not math.isfinite(hd) or hd < 0):
            raise ValueError("Missing/invalid nonempty HD95")
        if not normal and count == 0 and any(row[k] != 0 for k in OVERLAP):
            raise ValueError("Empty lesion prediction must score zero overlap")


def aggregate_runs(results):
    identities = {(r["image_size"], r["seed"]) for r in results}
    if len(results) != 6 or identities != {(s, seed) for s in (256, 384) for seed in (42, 43, 44)}:
        raise ValueError("Expected all six frozen external evaluations")
    grouped = {}
    for size in (256, 384):
        runs = [r for r in results if r["image_size"] == size]
        measures = {}
        for metric in OVERLAP:
            measures[f"lesion_positive_patient_{metric}"] = [r["metrics"]["cohorts"]["all_images"]["lesion_positive"]["patient_macro"][metric] for r in runs]
        measures["lesion_positive_patient_hd95_defined_px"] = [r["metrics"]["cohorts"]["all_images"]["lesion_positive"]["hd95"]["patient_mean_fully_defined_px"] for r in runs]
        for metric in ("patient_macro_foreground_fraction", "patient_macro_fraction_images_with_any_foreground"):
            measures[f"normal_{metric}"] = [r["metrics"]["cohorts"]["all_images"]["normal"][metric] for r in runs]
        stats = {}
        for name, values in measures.items():
            defined = [v for v in values if v is not None]
            stats[name] = {"mean": mean(defined), "sample_std": statistics.stdev(defined) if len(defined) > 1 else None,
                           "defined_seed_count": len(defined), "undefined_seed_count": len(values) - len(defined)}
        grouped[str(size)] = {"role": "primary" if size == 384 else "reference", "metrics": stats}
    return {"selected_resolution": 384, "reference_resolution": 256, "by_resolution": grouped,
            "uncertainty": "Sample SD is across three training seeds; per-run bootstrap intervals concern patient sampling conditional on fixed checkpoints. Neither is evidence of diagnostic validity.",
            "runs": results}
