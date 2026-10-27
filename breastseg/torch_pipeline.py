"""PyTorch/MONAI data loading, model construction, loss and native-size scoring."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

from .metrics import binary_metrics
from .preprocess import letterbox


class SegmentationDataset(Dataset):
    def __init__(self, rows, data_root, image_size=256, flip_probability=0.0):
        self.rows = list(rows)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.flip_probability = flip_probability
        if not self.rows:
            raise ValueError("Dataset is empty")
        if not 0 <= flip_probability <= 1:
            raise ValueError("Invalid flip probability")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_root / row["image_path"]) as source:
            image = source.convert("L")
        with Image.open(self.data_root / row["mask_path"]) as source:
            mask = source.convert("L")
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"{row['id']}: image dimensions differ from manifest")
        if random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        item = letterbox(image, mask, self.image_size)
        return {
            "image": torch.from_numpy(item["image"][None]),
            "mask": torch.from_numpy(item["mask"][None]),
            "valid": torch.from_numpy(item["valid"][None]),
            "layout": item["layout"], "native_mask": item["native_mask"],
            "id": row["id"], "case_id": row["case_id"],
            "pathology": row["pathology"], "device_name": row["device"],
            "image_path": row["image_path"],
        }


def collate_batch(items):
    tensors = {"image", "mask", "valid"}
    return {key: torch.stack([r[key] for r in items]) if key in tensors
            else [r[key] for r in items] for key in items[0]}


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, config, shuffle=False, device_type="cuda"):
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=shuffle,
        num_workers=int(config["num_workers"]), pin_memory=device_type == "cuda",
        collate_fn=collate_batch, worker_init_fn=seed_worker, generator=generator,
        persistent_workers=False, drop_last=False,
    )


def build_model(config):
    model = config["model"]
    return UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=tuple(model["channels"]), strides=tuple(model["strides"]),
        num_res_units=int(model["num_res_units"]), norm=model["norm"],
    )


def masked_dice_bce(logits, target, valid):
    """Equal-weight soft Dice and BCE; exclude letterbox padding from both."""
    logits, target, valid = logits.float(), target.float(), valid.float()
    dims = tuple(range(1, logits.ndim))
    count = valid.sum(dim=dims).clamp_min(1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none")
           * valid).sum(dim=dims) / count
    probability = torch.sigmoid(logits) * valid
    masked_target = target * valid
    intersection = (probability * masked_target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + masked_target.sum(dim=dims)
    dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return (0.5 * bce + 0.5 * dice_loss).mean()


def restore_probability(probability, layout):
    """Remove padding, then resize probabilities to the original H/W."""
    top, left = layout["top"], layout["left"]
    height, width = layout["resized_h"], layout["resized_w"]
    crop = probability[:, top:top + height, left:left + width]
    return F.interpolate(
        crop[None].float(), size=(layout["native_h"], layout["native_w"]),
        mode="bilinear", align_corners=False,
    )[0, 0]


def score_batch(logits, batch, threshold=0.5, include_hd95=False):
    probabilities = torch.sigmoid(logits.detach().float())
    rows, predictions = [], []
    for index, layout in enumerate(batch["layout"]):
        native = restore_probability(probabilities[index], layout).cpu().numpy()
        prediction = native >= threshold
        row = {
            "id": batch["id"][index], "case_id": batch["case_id"][index],
            "pathology": batch["pathology"][index],
            "device": batch["device_name"][index],
            **binary_metrics(prediction, batch["native_mask"][index], include_hd95),
        }
        rows.append(row)
        predictions.append(prediction)
    return rows, predictions
