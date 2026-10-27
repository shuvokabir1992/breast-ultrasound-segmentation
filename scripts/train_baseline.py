"""Train a 2D residual MONAI U-Net with Case-disjoint validation."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from breastseg.metrics import summarize
from breastseg.splits import load_manifest, sha256
from breastseg.torch_pipeline import (
    SegmentationDataset, build_model, make_loader, masked_dice_bce, score_batch,
)


def validate_config(config):
    if int(config["image_size"]) < 32 or int(config["image_size"]) % 16:
        raise ValueError("image_size must be a multiple of 16, at least 32")
    for key in ("batch_size", "epochs", "patience"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 < float(config["threshold"]) < 1:
        raise ValueError("threshold must be between zero and one")


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet_baseline.json"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="One epoch, two train batches and two validation batches")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("epochs", "num_workers"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.run_dir is not None:
        config["run_dir"] = str(args.run_dir)
    elif args.smoke:
        config["run_dir"] = "runs/unet_smoke"
    if args.smoke:
        config["epochs"] = 1
    config["smoke"] = args.smoke
    validate_config(config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the environment or request --device cpu.")
    device = torch.device(args.device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    amp = bool(config["amp"]) and device.type == "cuda"
    manifest_path = Path(config["manifest"])
    records = load_manifest(manifest_path)
    train = SegmentationDataset(
        [r for r in records if r["split"] == "train"], config["data_root"],
        int(config["image_size"]), float(config["horizontal_flip_probability"]))
    validation = SegmentationDataset(
        [r for r in records if r["split"] == "val"], config["data_root"],
        int(config["image_size"]))
    train_loader = make_loader(train, config, shuffle=True, device_type=device.type)
    val_loader = make_loader(validation, config, device_type=device.type)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = Path(config["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already contains files. Choose a new --run-dir.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(manifest_path, run_dir / "manifest.csv")
    environment = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "visible_gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("monai", "numpy", "pillow", "scipy")},
        "manifest_sha256": sha256(manifest_path),
        "train_images": len(train), "validation_images": len(validation),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "smoke": args.smoke,
        "reproducibility": "Seeds recorded; deterministic operations requested with warnings. Bitwise portability is not guaranteed.",
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment, indent=2), flush=True)
    if args.smoke:
        print("SMOKE TEST: partial data; metrics are not experiment results.", flush=True)
    history_path = run_dir / "history.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_case_dice", "val_image_dice",
              "val_case_iou", "learning_rate", "seconds", "smoke"]
    best, bad_epochs = -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if args.smoke and batch_index >= 2:
                break
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(image)
            loss = masked_dice_bce(logits, target, valid)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(image)
            seen += len(image)
        train_loss = total / seen
        model.eval()
        total, seen, scored = 0.0, 0, []
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                if args.smoke and batch_index >= 2:
                    break
                image = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(image)
                loss = masked_dice_bce(logits, target, valid)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
                total += float(loss.item()) * len(image)
                seen += len(image)
                rows, _ = score_batch(logits, batch, float(config["threshold"]))
                scored.extend(rows)
        metrics = summarize(scored)
        value = metrics["case_macro"]["dice"]
        improved = value > best
        if improved:
            best, bad_epochs = value, 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": total / seen,
            "val_case_dice": value, "val_image_dice": metrics["image_macro"]["dice"],
            "val_case_iou": metrics["case_macro"]["iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started, "smoke": args.smoke,
        }
        scheduler.step(value)
        with history_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        payload = {
            "format_version": 1, "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "best_val_case_dice": best,
            "manifest_sha256": environment["manifest_sha256"], "smoke": args.smoke,
        }
        save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best.pt", payload)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
              f"val loss {row['val_loss']:.4f} | val Case Dice {value:.4f} | "
              f"best {best:.4f}", flush=True)
        if bad_epochs >= int(config["patience"]):
            print(f"Early stopping after {bad_epochs} epochs without improvement.", flush=True)
            break
    print(f"Finished. Checkpoints and logs: {run_dir}", flush=True)
    print("Test images were not evaluated.", flush=True)


if __name__ == "__main__":
    main()
