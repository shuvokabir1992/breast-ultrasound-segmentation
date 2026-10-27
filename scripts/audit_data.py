"""Inspect an *already downloaded* BUS-BRA dataset without changing its contents.

Usage: python scripts/audit_data.py --root data/raw --output reports/audit.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def find_image_mask_directories(root: Path):
    """Find original `Images/` + `Masks/` (or `Mask/`) within extracted archives."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")
    for image_dir in sorted(root.rglob("Images")):
        for mask_name in ("Masks", "Mask"):
            mask_dir = image_dir.parent / mask_name
            if image_dir.is_dir() and mask_dir.is_dir():
                return image_dir, mask_dir
    raise FileNotFoundError(
        f"Could not find matching Images/ and Masks/ directories below {root}. "
        "Inspect archive extraction path first."
    )


def file_key(path: Path):
    """Pair bus_####.png with mask_####.png using the numeric suffix."""
    stem = Path(path).stem.lower()
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return re.sub(r"^(bus|image|img|mask)[_-]?", "", stem)


def index_unique(paths):
    index, duplicates = {}, []
    for p in sorted(paths):
        k = file_key(p)
        if k in index:
            duplicates.append([str(index[k]), str(p)])
        else:
            index[k] = p
    return index, duplicates


def pair_image_masks(image_dir: Path, mask_dir: Path):
    image_index, dupe_images = index_unique(image_dir.glob("*.png"))
    mask_index, dupe_masks = index_unique(mask_dir.glob("*.png"))
    common = sorted(set(image_index) & set(mask_index), key=lambda x: str(x))
    return {
        "pairs": [(image_index[k], mask_index[k]) for k in common],
        "image_count": len(image_index),
        "mask_count": len(mask_index),
        "missing_masks": [str(image_index[k]) for k in image_index.keys() - mask_index.keys()],
        "orphan_masks": [str(mask_index[k]) for k in mask_index.keys() - image_index.keys()],
        "duplicate_image_keys": dupe_images,
        "duplicate_mask_keys": dupe_masks,
    }


def audit(root: Path):
    image_dir, mask_dir = find_image_mask_directories(root)
    matches = pair_image_masks(image_dir, mask_dir)
    dimensions, mask_values, mismatch, empty_masks, failed_files = Counter(), Counter(), [], [], []
    for image_path, mask_path in matches["pairs"]:
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                dimensions[f"{image.width}x{image.height}"] += 1
                if image.size != mask.size:
                    mismatch.append([str(image_path), str(mask_path), image.size, mask.size])
                array = np.asarray(mask.convert("L"))
                vals, counts = np.unique(array, return_counts=True)
                mask_values.update({int(v): int(c) for v, c in zip(vals, counts)})
                if not np.any(array > 0):
                    empty_masks.append(str(mask_path))
        except (OSError, ValueError) as exc:
            failed_files.append({"image": str(image_path), "mask": str(mask_path), "error": str(exc)})
    csv_files = []
    for csv_path in sorted(image_dir.parent.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, nrows=5)
            csv_files.append({"path": str(csv_path), "columns": list(df.columns), "preview_rows": len(df)})
        except Exception as exc:
            csv_files.append({"path": str(csv_path), "read_error": str(exc)})
    result = {
        "dataset_root": str(image_dir.parent.resolve()),
        "image_dir": str(image_dir), "mask_dir": str(mask_dir),
        "image_count": matches["image_count"], "mask_count": matches["mask_count"],
        "paired_count": len(matches["pairs"]),
        "missing_masks": matches["missing_masks"],
        "orphan_masks": matches["orphan_masks"],
        "duplicate_image_keys": matches["duplicate_image_keys"],
        "duplicate_mask_keys": matches["duplicate_mask_keys"],
        "shape_mismatches": mismatch, "read_errors": failed_files,
        "empty_masks": empty_masks,
        "image_dimensions": dict(dimensions),
        "mask_grayscale_pixel_counts": {str(k): v for k, v in sorted(mask_values.items())},
        "csv_metadata": csv_files,
        "warning": "Audit inspects mask intensities only. Manually verify mask polarity/semantics and patient-level splits before training.",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory containing an extracted BUS-BRA archive")
    parser.add_argument("--output", type=Path, default=Path("reports/audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Found {result['paired_count']} pairs. Metadata CSVs: {len(result['csv_metadata'])}")
    print(f"Image dimensions: {result['image_dimensions']}")
    print(f"Missing masks: {len(result['missing_masks'])}, mismatches: {len(result['shape_mismatches'])}")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
