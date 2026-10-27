"""Image-only preprocessing; ground truth is never used to crop an input."""
from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox(image, mask=None, size=256):
    """Fit a full image inside a square, preserving its pixel aspect ratio."""
    if not isinstance(size, int) or size < 16 or size % 16:
        raise ValueError("size must be a positive multiple of 16")
    image = image.convert("L")
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = min(size, max(1, round(width * scale)))
    new_height = min(size, max(1, round(height * scale)))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    resized = np.asarray(
        image.resize((new_width, new_height), Image.Resampling.BILINEAR),
        dtype=np.float32) / 255.0
    canvas = np.zeros((size, size), dtype=np.float32)
    valid = np.zeros_like(canvas)
    canvas[top:top + new_height, left:left + new_width] = resized
    valid[top:top + new_height, left:left + new_width] = 1.0
    result = {
        "image": canvas, "valid": valid,
        "layout": {"native_h": height, "native_w": width,
                   "resized_h": new_height, "resized_w": new_width,
                   "top": top, "left": left},
    }
    if mask is not None:
        if mask.size != image.size:
            raise ValueError("Image and mask dimensions differ")
        array = np.asarray(mask.convert("L"))
        if not set(np.unique(array).tolist()).issubset({0, 255}):
            raise ValueError("Expected a binary PNG mask with values 0/255")
        resized_mask = np.asarray(mask.convert("L").resize(
            (new_width, new_height), Image.Resampling.NEAREST)) > 0
        if np.any(array > 0) and not resized_mask.any():
            raise ValueError("Lesion disappears during resizing; increase image_size")
        target = np.zeros_like(canvas)
        target[top:top + new_height, left:left + new_width] = resized_mask
        result.update(mask=target, native_mask=np.array(array > 0, copy=True))
    return result
