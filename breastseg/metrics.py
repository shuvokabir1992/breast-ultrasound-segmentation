"""Per-image metrics and equal-weight aggregation by supplied Case ID."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def binary_metrics(prediction, target, include_hd95=False):
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("Metrics require matching 2D masks")
    predicted, actual = int(pred.sum()), int(truth.sum())
    intersection = int(np.logical_and(pred, truth).sum())
    union = predicted + actual - intersection
    both_empty = predicted == actual == 0
    result = {
        "dice": 2.0 * intersection / (predicted + actual) if union else 1.0,
        "iou": intersection / union if union else 1.0,
        "sensitivity": intersection / actual if actual else float(both_empty),
        "precision": intersection / predicted if predicted else float(both_empty),
        "empty_prediction": predicted == 0,
        "empty_target": actual == 0,
    }
    if include_hd95:
        result["hd95_px"] = hd95(pred, truth)
    return result


def hd95(prediction, target):
    """Max of directional 95th-percentile boundary distances, in native pixels.

    Four-neighbour boundaries; Euclidean distances. Both empty -> 0. One empty
    -> None, retained as an undefined distance rather than silently omitted.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("HD95 requires matching 2D masks")
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        return None
    structure = generate_binary_structure(2, 1)
    a = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    b = truth & ~binary_erosion(truth, structure=structure, border_value=0)
    distance_to_b = distance_transform_edt(~b)[a]
    distance_to_a = distance_transform_edt(~a)[b]
    return float(max(np.percentile(distance_to_b, 95), np.percentile(distance_to_a, 95)))


def _mean(values):
    return float(np.mean(values)) if values else None


def summarize(rows, bootstrap=0, seed=42):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    metrics = ("dice", "iou", "sensitivity", "precision")
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    cases = []
    for case_id, views in sorted(grouped.items()):
        cases.append({"case_id": case_id, **{
            key: _mean([r[key] for r in views]) for key in metrics}})
    result = {
        "images": len(rows), "cases": len(cases),
        "image_macro": {key: _mean([r[key] for r in rows]) for key in metrics},
        "case_macro": {key: _mean([r[key] for r in cases]) for key in metrics},
        "empty_prediction_images": sum(bool(r["empty_prediction"]) for r in rows),
        "empty_target_images": sum(bool(r["empty_target"]) for r in rows),
        "aggregation": "Mean per image within each Case, then mean across Cases",
    }
    if "hd95_px" in rows[0]:
        distances = [r["hd95_px"] for r in rows if r["hd95_px"] is not None]
        case_distances = [
            float(np.mean([r["hd95_px"] for r in vv]))
            for vv in grouped.values() if all(r["hd95_px"] is not None for r in vv)
        ]
        result["hd95"] = {
            "units": "native image pixels; physical spacing not established",
            "image_mean_defined_px": _mean(distances),
            "image_defined_count": len(distances),
            "image_undefined_count": len(rows) - len(distances),
            "case_mean_fully_defined_px": _mean(case_distances),
            "case_fully_defined_count": len(case_distances),
            "case_undefined_count": len(cases) - len(case_distances),
            "empty_mask_rule": "Both empty=0; exactly one empty=undefined (null)",
        }
    if bootstrap:
        values = np.asarray([r["dice"] for r in cases])
        rng = np.random.default_rng(seed)
        means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                          for _ in range(bootstrap)])
        result["case_dice_bootstrap_95ci"] = [float(x) for x in np.percentile(means, [2.5, 97.5])]
        result["bootstrap_resamples"] = bootstrap
    return result
