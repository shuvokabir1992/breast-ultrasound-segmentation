"""Plot the six recorded validation runs; error bars are sample SD over seeds."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    runs = {(r["image_size"], r["seed"]): r for r in source["runs"]}
    if source["split"] != "val" or len(runs) != 6 or set(runs) != {(size, seed) for size in (256, 384) for seed in (42, 43, 44)}:
        raise ValueError("Expected the complete six-run validation summary")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colours = {256: "#0072b2", 384: "#d55e00"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), gridspec_kw={"width_ratios": [1, 1.12]})
    for seed in (42, 43, 44):
        values = [runs[size, seed]["metrics"]["case_macro"]["dice"] for size in (256, 384)]
        axes[0].plot([0, 1], values, color="#b8bdc5", linewidth=1.2, zorder=1)
        for x, size, value in zip((0, 1), (256, 384), values):
            axes[0].scatter(x, value, s=50, color=colours[size], edgecolor="white", linewidth=0.5, zorder=3)
        axes[0].annotate(f"seed {seed}", (1, values[1]), xytext=(32, 0), textcoords="offset points", va="center", fontsize=9)
    for x, size in enumerate((256, 384)):
        values = [runs[size, seed]["metrics"]["case_macro"]["dice"] for seed in (42, 43, 44)]
        axes[0].errorbar(x + 0.13, statistics.mean(values), yerr=statistics.stdev(values),
                         fmt="s", color="#222222", capsize=5, markersize=5, linewidth=1.5, zorder=4)
    axes[0].set(xticks=[0, 1], xticklabels=["256 px", "384 px"], xlim=(-0.30, 1.50),
                ylim=(0.875, 0.900), ylabel="Validation Case Dice", title="Overall: the mean advantage is very small")
    axes[0].legend(handles=[Line2D([0], [0], marker="s", color="#222222", label="Mean +/- sample SD", markersize=5)],
                   loc="upper left", frameon=False, fontsize=9)
    for size, offset in ((256, -0.13), (384, 0.13)):
        for x, pathology in enumerate(("benign", "malignant")):
            values = [runs[size, seed]["metrics"]["by_pathology"][pathology]["case_macro"]["dice"] for seed in (42, 43, 44)]
            for jitter, value in zip((-0.025, 0, 0.025), values):
                axes[1].scatter(x + offset + jitter, value, s=28, color=colours[size], alpha=0.60, zorder=3)
            axes[1].errorbar(x + offset, statistics.mean(values), yerr=statistics.stdev(values),
                             fmt="s", color=colours[size], capsize=5, linewidth=1.5, markersize=6, zorder=4)
    axes[1].set(xticks=[0, 1], xticklabels=["Benign (73 Cases)", "Malignant (34 Cases)"], xlim=(-0.45, 1.45),
                ylim=(0.84, 0.92), ylabel="Validation Case Dice", title="Malignant Dice is lower at 384 for all 3 seeds")
    axes[1].legend(handles=[Line2D([0], [0], marker="s", color=colours[size], label=f"{size} px", markersize=5) for size in (256, 384)],
                   loc="lower left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#e4e7ed", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("BUS-BRA: three training seeds per resolution on one fixed holdout", fontsize=14, y=0.99)
    fig.text(0.5, 0.925, "Same 193 validation images / 107 Cases | seeds 42, 43, 44 | threshold 0.5 | no postprocessing", ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.025, "Points are individual training runs. Error bars are sample SD across 3 seeds, not confidence intervals. No test scores are shown.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.065, 1, 0.91), w_pad=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
