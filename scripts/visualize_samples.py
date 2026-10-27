"""Preview example BUS-BRA image/mask pairs. For research quality checks only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from audit_data import find_image_mask_directories, pair_image_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/preview.png"))
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    im_dir, mask_dir = find_image_mask_directories(args.root)
    pairs = pair_image_masks(im_dir, mask_dir)["pairs"]
    if not pairs:
        raise RuntimeError("No image-mask pairs found")
    rng = np.random.default_rng(args.seed)
    inds = rng.choice(len(pairs), size=min(max(args.num, 1), len(pairs)), replace=False)
    fig, ax = plt.subplots(len(inds), 3, figsize=(11, max(3, len(inds) * 2.7)), squeeze=False)
    for row, idx in enumerate(inds):
        img_path, mask_path = pairs[idx]
        with Image.open(img_path) as source, Image.open(mask_path) as source_mask:
            img = np.asarray(source.convert("L"))
            mask = np.asarray(source_mask.convert("L")) > 0
        ax[row, 0].imshow(img, cmap="gray")
        ax[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax[row, 2].imshow(img, cmap="gray")
        ax[row, 2].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        ax[row, 0].set_title(img_path.name)
        ax[row, 1].set_title("Mask (any grayscale > 0)")
        ax[row, 2].set_title("Overlay")
        for cell in ax[row]:
            cell.axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview: {args.output}")


if __name__ == "__main__":
    main()
