"""Read the supplied metadata and keep every Case in exactly one split."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MANIFEST_FIELDS = [
    "id", "case_id", "split", "fold", "pathology", "image_path",
    "mask_path", "width", "height", "device",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {required}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def unique_index(rows, key, label):
    result = {}
    for row in rows:
        value = row[key]
        if value is None or not value.strip():
            raise ValueError(f"{label}: missing {key}")
        if value in result:
            raise ValueError(f"{label}: duplicate {key}={value}")
        result[value] = row
    return result


def split_counts(rows):
    cases = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in cases and cases[case_id] != row["pathology"]:
            raise ValueError(f"Case {case_id} has inconsistent pathology")
        cases[case_id] = row["pathology"]
    return {
        "images": len(rows),
        "cases": len(cases),
        "pathology_images": dict(sorted(Counter(r["pathology"] for r in rows).items())),
        "pathology_cases": dict(sorted(Counter(cases.values()).items())),
    }


def validate_manifest(rows):
    """Fail on ambiguous IDs, escaping paths, or related images across splits."""
    unique_index(rows, "id", "manifest")
    by_case = defaultdict(list)
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split for {row['id']}")
        if row["pathology"] not in {"benign", "malignant"}:
            raise ValueError(f"Unknown pathology for {row['id']}")
        if not str(row["case_id"]).strip():
            raise ValueError(f"Missing case_id for {row['id']}")
        if int(row["width"]) < 1 or int(row["height"]) < 1:
            raise ValueError(f"Invalid size for {row['id']}")
        for key in ("image_path", "mask_path"):
            path = Path(row[key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Manifest paths must be relative to data_root: {path}")
        by_case[row["case_id"]].append(row)
    for case_id, views in by_case.items():
        for key in ("split", "fold", "pathology", "device"):
            if len({r[key] for r in views}) != 1:
                raise ValueError(f"Case {case_id} spans inconsistent {key} values")
    for split in ("train", "val", "test"):
        if not any(r["split"] == split for r in rows):
            raise ValueError(f"Empty {split} split")


def inspect_metadata(metadata_dir):
    """Audit both outer folds and the inner 0/1 flags without guessing flag roles."""
    directory = Path(metadata_dir)
    metadata = read_csv(directory / "bus_data.csv",
                        ["ID", "Case", "Pathology", "Width", "Height", "Device"])
    index = unique_index(metadata, "ID", "bus_data.csv")
    cases = defaultdict(list)
    for row in metadata:
        match = re.fullmatch(r"bus_(\d+)-[lrs]", row["ID"])
        if not match or not row["Case"].isdigit():
            raise ValueError(f"Unexpected BUS-BRA ID or Case: {row}")
        if int(match.group(1)) != int(row["Case"]):
            raise ValueError(f"Case and ID disagree: {row['ID']}")
        if row["Pathology"] not in {"benign", "malignant"}:
            raise ValueError(f"Unknown pathology: {row['ID']}")
        cases[row["Case"]].append(row)
    for case_id, views in cases.items():
        for column in ("Pathology", "Device"):
            if len({r[column] for r in views}) != 1:
                raise ValueError(f"Case {case_id} has inconsistent {column}")
    report = {
        "images": len(metadata),
        "cases": len(cases),
        "images_per_case": dict(sorted(Counter(map(len, cases.values())).items())),
        "pathology_images": dict(sorted(Counter(r["Pathology"] for r in metadata).items())),
        "pathology_cases": dict(sorted(Counter(v[0]["Pathology"] for v in cases.values()).items())),
        "devices": dict(sorted(Counter(r["Device"] for r in metadata).items())),
        "metadata_sha256": {},
        "official_partitions": {},
        "group_definition": "Case as supplied in bus_data.csv; related l/r/s IDs stay together",
        "validation_flag_interpretation": (
            "Counts measure Cases containing both 0 and 1 in valid_f outside test fold f. "
            "The audit does not need to assign train/validation meanings to 0 and 1."
        ),
    }
    folds = {}
    for k in (5, 10):
        name = f"{k}-fold-cv.csv"
        rows = read_csv(directory / name, ["ID", "Pathology", "kFold"] +
                        [f"valid_{i}" for i in range(1, k + 1)])
        fold_index = unique_index(rows, "ID", name)
        if set(fold_index) != set(index):
            raise ValueError(f"{name}: IDs do not exactly match bus_data.csv")
        grouped = defaultdict(list)
        for row in rows:
            if row["Pathology"] != index[row["ID"]]["Pathology"]:
                raise ValueError(f"{name}: pathology mismatch for {row['ID']}")
            fold = int(row["kFold"])
            if not 1 <= fold <= k:
                raise ValueError(f"{name}: invalid kFold")
            grouped[index[row["ID"]]["Case"]].append(row)
        crossing = [c for c, vv in grouped.items() if len({r["kFold"] for r in vv}) != 1]
        per_fold = []
        for f in range(1, k + 1):
            key = f"valid_{f}"
            test = [r for r in rows if int(r["kFold"]) == f]
            other = [r for r in rows if int(r["kFold"]) != f]
            if any(r[key].strip().lower() not in {"nan", ""} for r in test):
                raise ValueError(f"{name}: test rows contain nonempty {key}")
            if any(r[key] not in {"0", "1"} for r in other):
                raise ValueError(f"{name}: unexpected {key} outside test fold")
            overlap = [
                c for c, vv in grouped.items()
                if {r[key] for r in vv if int(r["kFold"]) != f} == {"0", "1"}
            ]
            per_fold.append({
                "fold": f,
                "images": len(test),
                "cases": len({index[r["ID"]]["Case"] for r in test}),
                "cases_with_both_validation_flags": len(overlap),
                "example_case_ids": sorted(overlap, key=int)[:5],
            })
        report["official_partitions"][str(k)] = {
            "cases_spanning_outer_folds": len(crossing), "folds": per_fold,
        }
        if crossing:
            raise ValueError(f"{name}: {len(crossing)} Cases cross outer folds")
        folds[k] = fold_index
    for name in ("bus_data.csv", "5-fold-cv.csv", "10-fold-cv.csv"):
        report["metadata_sha256"][name] = sha256(directory / name)
    return index, folds, report


def prepare(data_root, output_dir, metadata_dir=None, metadata_only=False,
            test_fold=1, val_fold=2):
    data_root, output_dir = Path(data_root), Path(output_dir)
    if not 1 <= test_fold <= 10 or not 1 <= val_fold <= 10 or test_fold == val_fold:
        raise ValueError("Choose different test/validation folds from 1 through 10")
    metadata, folds, report = inspect_metadata(metadata_dir or data_root)
    report["dimension_check"] = {
        "status": "not_checked" if metadata_only else "checked",
        "manifest_dimension_source": (
            "csv_unverified" if metadata_only else "decoded_matching_png_pair"
        ),
        "discrepancy_count": None if metadata_only else 0,
        "discrepancies": [],
        "policy": (
            "Preserve the source CSV. When image and mask decode successfully and "
            "have identical dimensions, use their actual PNG dimensions in the "
            "manifest and record any difference from the CSV."
        ),
    }
    rows = []
    mask_directory = "Masks"
    if not metadata_only:
        if not (data_root / mask_directory).is_dir() and (data_root / "Mask").is_dir():
            mask_directory = "Mask"
        if not (data_root / "Images").is_dir() or not (data_root / mask_directory).is_dir():
            raise FileNotFoundError(f"Expected Images and Masks directories inside {data_root}")
    for image_id, record in sorted(metadata.items()):
        fold = int(folds[10][image_id]["kFold"])
        split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
        row = {
            "id": image_id, "case_id": record["Case"], "split": split, "fold": fold,
            "pathology": record["Pathology"], "image_path": f"Images/{image_id}.png",
            "mask_path": f"{mask_directory}/mask_{image_id.removeprefix('bus_')}.png",
            "width": int(record["Width"]), "height": int(record["Height"]),
            "device": record["Device"],
        }
        if not metadata_only:
            from PIL import Image
            import numpy as np
            with Image.open(data_root / row["image_path"]) as image:
                image.load()
                image_size = image.size
            with Image.open(data_root / row["mask_path"]) as mask:
                mask.load()
                if mask.size != image_size:
                    raise ValueError(f"{image_id}: image/mask shapes differ")
                values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
                if not values.issubset({0, 255}) or 255 not in values:
                    raise ValueError(f"{image_id}: expected nonempty binary 0/255 mask")
            if image_size != (row["width"], row["height"]):
                report["dimension_check"]["discrepancies"].append({
                    "id": image_id,
                    "metadata_width": row["width"],
                    "metadata_height": row["height"],
                    "png_width": image_size[0],
                    "png_height": image_size[1],
                })
            row["width"], row["height"] = image_size
        rows.append(row)
    if not metadata_only:
        report["dimension_check"]["discrepancy_count"] = len(
            report["dimension_check"]["discrepancies"])
    validate_manifest(rows)
    report["baseline_protocol"] = {
        "outer_partition_file": "10-fold-cv.csv", "test_fold": test_fold,
        "validation_fold": val_fold,
        "training_folds": [i for i in range(1, 11) if i not in {test_fold, val_fold}],
        "uses_valid_flags": False,
        "description": "Single fixed holdout using whole supplied outer folds; not a completed 10-fold CV experiment",
        "image_files_fully_checked": not metadata_only,
        "splits": {s: split_counts([r for r in rows if r["split"] == s])
                   for s in ("train", "val", "test")},
        "case_overlap_counts": {
            f"{a}_{b}": len({r["case_id"] for r in rows if r["split"] == a} &
                           {r["case_id"] for r in rows if r["split"] == b})
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "busbra_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report["manifest_sha256"] = sha256(manifest)
    (output_dir / "metadata_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return rows, report


def load_manifest(path):
    rows = read_csv(path, MANIFEST_FIELDS)
    validate_manifest(rows)
    return rows
