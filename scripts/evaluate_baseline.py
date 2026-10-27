"""Score a completed checkpoint at native resolution with Dice, IoU and HD95."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch

from breastseg.metrics import summarize
from breastseg.plotting import save_examples
from breastseg.splits import load_manifest, sha256
from breastseg.torch_pipeline import SegmentationDataset, build_model, make_loader, score_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("smoke", False):
        raise ValueError("Smoke checkpoints use partial data and cannot produce research evaluation results.")
    config = dict(checkpoint["config"])
    if args.data_root is not None:
        config["data_root"] = str(args.data_root)
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    manifest = args.manifest or (args.checkpoint.parent / "manifest.csv")
    if sha256(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Manifest differs from the one used to train this checkpoint")
    records = load_manifest(manifest)
    dataset = SegmentationDataset(
        [r for r in records if r["split"] == args.split],
        config["data_root"], int(config["image_size"]))
    device = torch.device(args.device)
    loader = make_loader(dataset, config, device_type=device.type)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    selected = np.random.default_rng(int(config["seed"])).choice(
        len(dataset), size=min(8, len(dataset)), replace=False)
    example_ids = {dataset.rows[int(i)]["id"] for i in selected}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new output directory to preserve existing results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "predicted_masks"
    if args.save_masks:
        mask_dir.mkdir()
    scored, examples = [], []
    # Use full float32 for the final metric export; training uses AMP if requested.
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            rows, predictions = score_batch(logits, batch, float(config["threshold"]), True)
            scored.extend(rows)
            for index, (row, prediction) in enumerate(zip(rows, predictions)):
                if args.save_masks:
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(
                        mask_dir / f"{row['id']}.png")
                if row["id"] in example_ids:
                    with Image.open(Path(config["data_root"]) / batch["image_path"][index]) as source:
                        image = np.array(source.convert("L"))
                    examples.append({
                        "id": row["id"], "image": image,
                        "target": batch["native_mask"][index],
                        "prediction": prediction, "dice": row["dice"],
                    })
    summary = summarize(scored, bootstrap=2000, seed=int(config["seed"]))
    summary.update({
        "split": args.split, "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(manifest), "threshold": config["threshold"],
        "metric_resolution": "Original image dimensions, after removing padding and resizing probabilities",
        "selection_metric": "Validation Case-macro Dice",
        "by_pathology": {
            value: summarize([r for r in scored if r["pathology"] == value])
            for value in sorted({r["pathology"] for r in scored})
        },
        "by_device": {
            value: summarize([r for r in scored if r["device"] == value])
            for value in sorted({r["device"] for r in scored})
        },
    })
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    save_examples(examples, args.output_dir / "predictions.png")
    print(json.dumps({k: summary[k] for k in
                      ("split", "images", "cases", "case_macro", "hd95",
                       "case_dice_bootstrap_95ci")}, indent=2))
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
