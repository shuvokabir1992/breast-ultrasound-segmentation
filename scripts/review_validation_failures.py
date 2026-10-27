"""Inspect saved native-size validation masks; optionally compare a fixed LCC rule.

Run from the project root after evaluate_baseline.py --split val --save-masks.
Uses the existing breastseg package. Never loads test images or test masks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breastseg.metrics import binary_metrics, summarize
from breastseg.splits import load_manifest, sha256


DEFAULT_IDS = ["bus_0452-r", "bus_0452-l", "bus_1047-r", "bus_1047-l",
               "bus_0026-s", "bus_0321-s"]
STRUCTURE = generate_binary_structure(2, 2)  # Eight-connected foreground.


def read_mask(path, expected_shape):
    with Image.open(path) as source:
        if source.mode not in ("1", "L", "I", "I;16"):
            raise ValueError(f"{path}: expected a single-channel binary mask")
        array = np.array(source)
    if array.shape != expected_shape:
        raise ValueError(f"{path}: expected native shape {expected_shape}, got {array.shape}")
    if not set(np.unique(array)).issubset({0, 1, 255}):
        raise ValueError(f"{path}: expected binary values 0/1 or 0/255")
    return array > 0


def components(mask):
    labels, count = label(mask, structure=STRUCTURE)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return labels, sizes


def largest_component(mask):
    labels, sizes = components(mask)
    if not len(sizes):
        return np.zeros_like(mask, dtype=bool)
    # Equal areas: choose the component encountered first in raster order.
    return labels == int(np.argmax(sizes) + 1)


def component_info(mask):
    _, sizes = components(mask)
    sizes = sorted((int(x) for x in sizes), reverse=True)
    return {"component_count": len(sizes), "foreground_px": int(mask.sum()),
            "component_sizes_px": sizes,
            "outside_largest_px": int(mask.sum()) - (sizes[0] if sizes else 0)}


def verify_evaluation(metrics, rows, records, manifest_hash):
    if metrics.get("split") != "val":
        raise ValueError("This review accepts validation evaluations only; test split rejected")
    if metrics.get("manifest_sha256") != manifest_hash:
        raise ValueError("Evaluation and run manifest hashes differ")
    expected = {r["id"]: r for r in records if r["split"] == "val"}
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("per_image.csv must contain every validation image exactly once")
    if metrics.get("images") != len(rows):
        raise ValueError("Image count differs between metrics.json and per_image.csv")
    if metrics.get("cases") != len({r["case_id"] for r in rows}):
        raise ValueError("Case count differs between metrics.json and per_image.csv")
    for row in rows:
        for key in ("case_id", "pathology", "device"):
            if row[key] != expected[row["id"]][key]:
                raise ValueError(f"{row['id']}: {key} differs from run manifest")
    return expected


def verify_row(exported, computed):
    for key in ("dice", "iou", "sensitivity", "precision", "hd95_px"):
        value = exported[key]
        recorded = None if value in ("", "None", "null") else float(value)
        actual = computed[key]
        if (recorded is None) != (actual is None):
            raise ValueError(f"{exported['id']}: recomputed {key} differs from evaluation")
        if actual is not None and not math.isclose(actual, recorded, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{exported['id']}: recomputed {key} differs from evaluation")
    for key in ("empty_prediction", "empty_target"):
        if exported[key].lower() not in ("true", "false"):
            raise ValueError(f"{exported['id']}: invalid {key}")
        if (exported[key].lower() == "true") != computed[key]:
            raise ValueError(f"{exported['id']}: recomputed {key} differs from evaluation")


def grouped_summary(rows):
    result = summarize(rows)
    for column, name in (("pathology", "by_pathology"), ("device", "by_device")):
        result[name] = {value: summarize([r for r in rows if r[column] == value])
                        for value in sorted({r[column] for r in rows})}
    return result


def paired_comparison(before, after, seed=42):
    if [r["id"] for r in before] != [r["id"] for r in after]:
        raise ValueError("Paired comparison requires matching image IDs in the same order")
    groups = defaultdict(list)
    for a, b in zip(before, after):
        if a["case_id"] != b["case_id"]:
            raise ValueError("Paired comparison requires matching Case IDs")
        groups[a["case_id"]].append(b["dice"] - a["dice"])
    values = np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)]
    changes = sorted(({"id": a["id"], "case_id": a["case_id"],
                       "baseline_dice": a["dice"], "candidate_dice": b["dice"],
                       "dice_delta": b["dice"] - a["dice"]}
                      for a, b in zip(before, after)), key=lambda r: (r["dice_delta"], r["id"]))
    return {"case_dice_delta": float(values.mean()),
            "paired_case_bootstrap_95ci": [float(v) for v in np.percentile(means, [2.5, 97.5])],
            "bootstrap_resamples": 2000, "bootstrap_seed": seed,
            "cases_improved": int((values > 1e-12).sum()),
            "cases_worsened": int((values < -1e-12).sum()),
            "cases_unchanged": int((np.abs(values) <= 1e-12).sum()),
            "largest_image_losses": [r for r in changes if r["dice_delta"] < -1e-12][:10],
            "largest_image_gains": [r for r in reversed(changes) if r["dice_delta"] > 1e-12][:10],
            "interpretation": "Exploratory validation comparison; does not include training-seed variability"}


def draw_examples(examples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(len(examples), 4, figsize=(14, 2.8 * len(examples)), squeeze=False)
    for cells, example in zip(axes, examples):
        image, target, prediction = (example[k] for k in ("image", "target", "prediction"))
        for cell in cells:
            cell.imshow(image, cmap="gray", vmin=0, vmax=255)
            cell.axis("off")
        for cell, mask in ((cells[1], target), (cells[2], prediction)):
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = (1.0, 0.8, 0.0, 0.50)
            cell.imshow(overlay)
        errors = np.zeros((*target.shape, 4))
        errors[target & prediction] = (0.0, 0.62, 0.45, 0.65)
        errors[~target & prediction] = (0.84, 0.37, 0.0, 0.80)
        errors[target & ~prediction] = (0.0, 0.45, 0.70, 0.80)
        cells[3].imshow(errors)
        distance = example["hd95_px"]
        distance_text = "undefined" if distance is None else f"{distance:.1f} px"
        cells[0].set_title(f"{example['id']} | Case {example['case_id']}\n{image.shape[1]} x {image.shape[0]} native pixels", fontsize=10)
        cells[1].set_title(f"Reference | {example['target_components']} component(s)", fontsize=10)
        cells[2].set_title(f"Original prediction | {example['pred_components']} component(s)\nDice {example['dice']:.4f} | HD95 {distance_text}", fontsize=10)
        cells[3].set_title(f"Errors | recall {example['sensitivity']:.3f}\nprecision {example['precision']:.3f}", fontsize=10)
    fig.suptitle("Validation failure review: original predictions", fontsize=15, y=0.997)
    fig.legend(handles=[Patch(color="#009e73", label="Correct foreground (TP)"),
                        Patch(color="#d55e00", label="Extra foreground (FP)"),
                        Patch(color="#0072b2", label="Missed foreground (FN)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/unet_baseline_seed42"))
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--compare-largest-component", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new output directory to preserve existing results")
    config = json.loads((args.run_dir / "config.json").read_text())
    metrics = json.loads((args.eval_dir / "metrics.json").read_text())
    if metrics.get("split") != "val":
        raise ValueError("This review accepts validation evaluations only; test split rejected")
    manifest = args.run_dir / "manifest.csv"
    records = load_manifest(manifest)
    with (args.eval_dir / "per_image.csv").open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    val = verify_evaluation(metrics, exported, records, sha256(manifest))
    if metrics["checkpoint_sha256"] != sha256(args.run_dir / "best.pt"):
        raise ValueError("Evaluation checkpoint hash does not match run_dir/best.pt")
    if float(metrics["threshold"]) != float(config["threshold"]):
        raise ValueError("Evaluation threshold differs from the run configuration")
    if len(args.ids) != len(set(args.ids)) or not set(args.ids).issubset(val):
        raise ValueError("--ids must be distinct validation image IDs from this manifest")
    root = args.data_root or Path(config["data_root"])
    selected, baseline, candidate, details = {}, [], [], []
    for row in sorted(exported, key=lambda r: r["id"]):
        metadata = val[row["id"]]
        shape = (int(metadata["height"]), int(metadata["width"]))
        truth = read_mask(root / metadata["mask_path"], shape)
        prediction = read_mask(args.eval_dir / "predicted_masks" / f"{row['id']}.png", shape)
        if not truth.any():
            raise ValueError(f"{row['id']}: empty reference mask")
        score = {key: row[key] for key in ("id", "case_id", "pathology", "device")}
        score.update(binary_metrics(prediction, truth, include_hd95=True))
        verify_row(row, score)
        baseline.append(score)
        truth_info, pred_info = component_info(truth), component_info(prediction)
        detail = {**score,
                  "reference_components": truth_info["component_count"],
                  "predicted_components": pred_info["component_count"],
                  "reference_component_sizes_px": json.dumps(truth_info["component_sizes_px"]),
                  "predicted_component_sizes_px": json.dumps(pred_info["component_sizes_px"]),
                  "reference_outside_largest_px": truth_info["outside_largest_px"],
                  "predicted_outside_largest_px": pred_info["outside_largest_px"]}
        if args.compare_largest_component:
            cleaned = largest_component(prediction)
            updated = {key: score[key] for key in ("id", "case_id", "pathology", "device")}
            updated.update(binary_metrics(cleaned, truth, include_hd95=True))
            candidate.append(updated)
            detail.update({f"lcc_{key}": updated[key] for key in
                           ("dice", "iou", "sensitivity", "precision", "hd95_px")})
            detail["lcc_dice_delta"] = updated["dice"] - score["dice"]
        details.append(detail)
        if row["id"] in args.ids:
            with Image.open(root / metadata["image_path"]) as source:
                image = np.array(source.convert("L"))
            if image.shape != shape:
                raise ValueError(f"{row['id']}: image dimensions differ from manifest")
            selected[row["id"]] = {**score, "image": image, "target": truth,
                                   "prediction": prediction,
                                   "target_components": truth_info["component_count"],
                                   "pred_components": pred_info["component_count"]}

    # Inspect annotations in training and validation only. Components are not a lesion count.
    reference_audit = []
    for row in records:
        if row["split"] not in ("train", "val"):
            continue
        truth = read_mask(root / row["mask_path"], (int(row["height"]), int(row["width"])))
        if not truth.any():
            raise ValueError(f"{row['id']}: empty reference mask")
        info = component_info(truth)
        reference_audit.append({"id": row["id"], "case_id": row["case_id"],
                                "split": row["split"], **info,
                                "component_sizes_px": json.dumps(info["component_sizes_px"])})
    baseline_summary = grouped_summary(baseline)
    for key in ("dice", "iou", "sensitivity", "precision"):
        if not math.isclose(baseline_summary["case_macro"][key], metrics["case_macro"][key],
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Recomputed aggregate differs from evaluation")
    report = {"split": "val", "seed": int(config["seed"]),
              "checkpoint_epoch": metrics["checkpoint_epoch"],
              "checkpoint_sha256": metrics["checkpoint_sha256"],
              "manifest_sha256": metrics["manifest_sha256"],
              "source_metrics_sha256": sha256(args.eval_dir / "metrics.json"),
              "source_per_image_sha256": sha256(args.eval_dir / "per_image.csv"),
              "threshold": metrics["threshold"], "image_size": config["image_size"],
              "component_connectivity": "8-connected foreground; diagonal neighbours connected",
              "components_are_lesions": False,
              "test_images_or_masks_read": False,
              "baseline": baseline_summary,
              "reference_audit": {split: {
                  "images": sum(r["split"] == split for r in reference_audit),
                  "images_with_multiple_components": sum(r["split"] == split and r["component_count"] > 1
                                                          for r in reference_audit)}
                                  for split in ("train", "val")},
              "selection": "No change selected automatically. Review validation results and annotation topology."}
    if candidate:
        report["candidate_policy"] = "Keep the largest 8-connected predicted foreground component in every validation image; no ground truth used by the rule. Native resolution; unchanged threshold."
        report["largest_component"] = grouped_summary(candidate)
        report["paired_comparison"] = paired_comparison(baseline, candidate, int(config["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "review_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(args.output_dir / "review_per_image.csv", details)
    write_csv(args.output_dir / "reference_components.csv", reference_audit)
    draw_examples([selected[k] for k in args.ids], args.output_dir / "failure_overview.png")
    print(f"Checked all {len(baseline)} validation images in {baseline_summary['cases']} Cases.")
    print(f"Original validation Case Dice: {baseline_summary['case_macro']['dice']:.6f}")
    if candidate:
        print(f"Largest-component validation Case Dice: {report['largest_component']['case_macro']['dice']:.6f}")
        print(f"Paired Case Dice change: {report['paired_comparison']['case_dice_delta']:+.6f}")
    print(f"Saved review to {args.output_dir}")


if __name__ == "__main__":
    main()
