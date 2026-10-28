"""Audit the frozen validation/test JSON exports and plot recorded seed results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


SIZES = (256, 384)
SEEDS = (42, 43, 44)
METRICS = ("dice", "iou", "sensitivity", "precision")


def audit(source, split):
    if source["split"] != split:
        raise ValueError(f"Expected {split} exports")
    runs = {(r["image_size"], r["seed"]): r for r in source["runs"]}
    if len(source["runs"]) != 6 or set(runs) != {(size, seed) for size in SIZES for seed in SEEDS}:
        raise ValueError("Expected exactly six resolution/seed identities")
    reference = {c["case_id"]: c for c in runs[256, 42]["per_case"]}
    for row in runs.values():
        metrics = row["metrics"]
        cases = row["per_case"]
        if metrics["split"] != split or len(cases) != len({c["case_id"] for c in cases}):
            raise ValueError("Inconsistent split or duplicate Cases")
        if set(c["case_id"] for c in cases) != set(reference):
            raise ValueError("Case identities differ across models")
        if metrics["cases"] != len(cases) or metrics["images"] != sum(c["images"] for c in cases):
            raise ValueError("Export counts differ")
        for case in cases:
            for key in ("images", "pathology", "device"):
                if case[key] != reference[case["case_id"]][key]:
                    raise ValueError("Case metadata differs across models")
        for metric in METRICS:
            if not math.isclose(statistics.mean(c[metric] for c in cases), metrics["case_macro"][metric], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Recomputed Case average differs from reported value")
            for column, label in (("pathology", "by_pathology"), ("device", "by_device")):
                for value, subgroup in metrics[label].items():
                    selected = [c for c in cases if c[column] == value]
                    if len(selected) != subgroup["cases"] or not math.isclose(
                            statistics.mean(c[metric] for c in selected), subgroup["case_macro"][metric], rel_tol=0, abs_tol=1e-12):
                        raise ValueError("Recomputed subgroup average differs")
        distances = [c["hd95_px"] for c in cases if c["hd95_px"] is not None]
        reported = metrics["hd95"]
        if len(distances) != reported["case_fully_defined_count"] or len(cases) - len(distances) != reported["case_undefined_count"]:
            raise ValueError("HD95 availability counts differ")
        actual = statistics.mean(distances) if distances else None
        expected = reported["case_mean_fully_defined_px"]
        if (actual is None) != (expected is None) or (actual is not None and not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Recomputed HD95 differs")
    for size in SIZES:
        for metric in METRICS:
            values = [runs[size, seed]["metrics"]["case_macro"][metric] for seed in SEEDS]
            reported = source["by_resolution"][str(size)][metric]
            if not math.isclose(statistics.mean(values), reported["mean"], rel_tol=0, abs_tol=1e-12) or not math.isclose(
                    statistics.stdev(values), reported["sample_std"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Recomputed seed mean/sample SD differs")
        for pathology in ("benign", "malignant"):
            values = [runs[size, seed]["metrics"]["by_pathology"][pathology]["case_macro"]["dice"] for seed in SEEDS]
            reported = source["by_resolution"][str(size)][f"{pathology}_case_dice"]
            if not math.isclose(statistics.mean(values), reported["mean"], rel_tol=0, abs_tol=1e-12) or not math.isclose(
                    statistics.stdev(values), reported["sample_std"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Recomputed subgroup seed mean/sample SD differs")
        distances = [runs[size, seed]["metrics"]["hd95"]["case_mean_fully_defined_px"] for seed in SEEDS]
        distances = [value for value in distances if value is not None]
        reported = source["by_resolution"][str(size)]["hd95_defined_case_mean_px"]
        if distances and not math.isclose(statistics.mean(distances), reported["mean"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Recomputed seed HD95 mean differs")
        if len(distances) > 1 and not math.isclose(statistics.stdev(distances), reported["sample_std"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Recomputed seed HD95 sample SD differs")
    return runs, reference


def plot(validation, test, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colours = {256: "#0072b2", 384: "#d55e00"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4))
    for group_x, cohort in enumerate((validation, test)):
        for size, offset in ((256, -0.13), (384, 0.13)):
            values = [cohort[size, seed]["metrics"]["case_macro"]["dice"] for seed in SEEDS]
            for jitter, value in zip((-0.035, 0, 0.035), values):
                axes[0].scatter(group_x + offset + jitter, value, color=colours[size], s=30, alpha=0.65, zorder=3)
            axes[0].errorbar(group_x + offset, statistics.mean(values), yerr=statistics.stdev(values),
                             fmt="s", color=colours[size], capsize=5, markersize=6, linewidth=1.5, zorder=4)
    axes[0].set(xticks=(0, 1), xticklabels=("Validation\n107 Cases", "Test\n106 Cases"),
                ylabel="Case-macro Dice", ylim=(0.87, 0.90), xlim=(-0.45, 1.45),
                title="The validation ranking did not carry over to test")
    for group_x, pathology in enumerate(("benign", "malignant")):
        for size, offset in ((256, -0.13), (384, 0.13)):
            values = [test[size, seed]["metrics"]["by_pathology"][pathology]["case_macro"]["dice"] for seed in SEEDS]
            for jitter, value in zip((-0.035, 0, 0.035), values):
                axes[1].scatter(group_x + offset + jitter, value, color=colours[size], s=30, alpha=0.65, zorder=3)
            axes[1].errorbar(group_x + offset, statistics.mean(values), yerr=statistics.stdev(values),
                             fmt="s", color=colours[size], capsize=5, markersize=6, linewidth=1.5, zorder=4)
    axes[1].set(xticks=(0, 1), xticklabels=("Benign\n72 Cases", "Malignant\n34 Cases"),
                ylabel="Test Case-macro Dice", ylim=(0.83, 0.92), xlim=(-0.45, 1.45),
                title="Test subgroup results across three seeds")
    for ax in axes:
        ax.grid(axis="y", color="#e4e7ed", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(handles=[Line2D([0], [0], marker="s", color=colours[size], label=f"{size} px", markersize=5) for size in SIZES],
                   loc="lower left", frameon=False)
    fig.suptitle("BUS-BRA frozen baseline: validation and test results", fontsize=15, y=0.995)
    fig.text(0.5, 0.93, "384 px selected by validation | 256 px reference | seeds 42, 43, 44 | threshold 0.5 | no postprocessing", ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.02, "Points are individual runs; error bars are sample SD across three training seeds, not confidence intervals. Test is a reused fixed holdout.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 0.90), w_pad=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new output directory")
    v = json.loads(args.validation_summary.read_text())
    t = json.loads(args.test_summary.read_text())
    validation, val_cases = audit(v, "val")
    test, test_cases = audit(t, "test")
    if set(val_cases) & set(test_cases):
        raise ValueError("Validation and test Case IDs overlap")
    protocol = t["protocol"]
    val_hash = hashlib.sha256(args.validation_summary.read_bytes()).hexdigest()
    if protocol["validation_summary_sha256"] != val_hash or t["selected_resolution"] != 384 or protocol["selected_resolution"] != 384:
        raise ValueError("Frozen selection/provenance differs")
    if protocol["postprocessing"] != "none" or protocol["threshold"] != 0.5:
        raise ValueError("Frozen inference settings differ")
    for identity, row in test.items():
        previous = validation[identity]
        if row["checkpoint_sha256"] != previous["metrics"]["checkpoint_sha256"] or row["checkpoint_epoch"] != previous["checkpoint_epoch"]:
            raise ValueError("Checkpoint changed after validation selection")
        expected = protocol["checkpoint_sha256"][f"{identity[0]}_seed{identity[1]}"]
        if row["metrics"]["checkpoint_sha256"] != expected or row["checkpoint_sha256"] != expected:
            raise ValueError("Frozen checkpoint identity differs")
        if row["metrics"]["manifest_sha256"] != v["manifest_sha256"] or row["metrics"]["threshold"] != 0.5:
            raise ValueError("Manifest or threshold mismatch")
    indexes = {identity: {c["case_id"]: c for c in row["per_case"]} for identity, row in test.items()}
    ids = sorted(test_cases)
    values = np.array([statistics.mean(indexes[384, seed][case]["dice"] - indexes[256, seed][case]["dice"] for seed in SEEDS) for case in ids])
    rng = np.random.default_rng(42)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)]
    report = {
        "selected_resolution": 384, "reference_resolution": 256, "seeds": list(SEEDS),
        "scope": "One fixed Case-disjoint holdout, three training seeds per resolution; reused test cohort, not independent external validation.",
        "source_sha256": {"validation": val_hash, "test": hashlib.sha256(args.test_summary.read_bytes()).hexdigest()},
        "validation": v["by_resolution"], "test": t["by_resolution"],
        "test_counts": {"images": test[384, 42]["metrics"]["images"], "cases": len(test_cases)},
        "paired_seed_test_dice_differences_384_minus_256": {
            str(seed): test[384, seed]["metrics"]["case_macro"]["dice"] - test[256, seed]["metrics"]["case_macro"]["dice"] for seed in SEEDS},
        "exploratory_paired_case_bootstrap": {
            "metric": "Mean over the three fixed seed pairs of per-Case Dice differences (384 minus 256)",
            "mean_difference": float(values.mean()), "95ci": [float(x) for x in np.percentile(means, [2.5, 97.5])],
            "resamples": 2000, "bootstrap_seed": 42,
            "limitation": "Case-sampling uncertainty conditional on the six fixed checkpoints. Does not include training-seed variability or selection effects."},
        "runs": [{"image_size": identity[0], "seed": identity[1], "checkpoint_epoch": row["checkpoint_epoch"],
                  "checkpoint_sha256": row["checkpoint_sha256"], "test_case_macro": row["metrics"]["case_macro"],
                  "test_hd95": row["metrics"]["hd95"]} for identity, row in sorted(test.items())],
        "checks": "Matching six frozen checkpoint identities, matching Cases and metadata within cohorts, disjoint validation/test Cases, recomputed Case/subgroup metrics and seed averages; all passed.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_final_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(validation, test, args.output_dir / "figures" / "baseline_final_results.png")
    print(json.dumps({"checks": "passed", "primary_test_case_dice": report["test"]["384"]["dice"],
                      "reference_test_case_dice": report["test"]["256"]["dice"],
                      "paired_case_difference": report["exploratory_paired_case_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
