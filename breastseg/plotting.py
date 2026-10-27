"""Save fixed evaluation examples without requiring a graphical desktop."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_examples(examples, output):
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 2.8 * len(examples)),
                             squeeze=False)
    for cells, example in zip(axes, examples):
        image, target, prediction = example["image"], example["target"], example["prediction"]
        for cell in cells:
            cell.imshow(image, cmap="gray", vmin=0, vmax=255)
            cell.axis("off")
        cells[0].set_title(example["id"])
        cells[1].imshow(np.ma.masked_where(~target, target), cmap="summer",
                        alpha=0.5, vmin=0, vmax=1)
        cells[1].set_title("Ground truth")
        cells[2].imshow(np.ma.masked_where(~prediction, prediction), cmap="autumn",
                        alpha=0.5, vmin=0, vmax=1)
        cells[2].set_title(f"Prediction | Dice {example['dice']:.3f}")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
