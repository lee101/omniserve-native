"""Hair isolation for talking-avatar layers.

SAM2 (box on the upper silhouette) when the weights load; otherwise a
skin-aware split of the existing cutout alpha. Front hair is bangs / locks
over the face so the overlay can sway them independently of back hair.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

HAIR_MIN_COVERAGE = float(os.getenv("HAIR_MIN_COVERAGE", "0.07"))
SAM2_MODEL = os.getenv("HAIR_SAM2_MODEL", "facebook/sam2.1-hiera-tiny")
HAIR_BACKEND = os.getenv("HAIR_BACKEND", "auto")

_sam2: dict[str, Any] | None = None
_sam2_failed = False


def pixel_looks_like_skin(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return (
        (r >= 95)
        & (g >= 45)
        & (b >= 25)
        & (r >= g + 8)
        & (r >= b + 12)
        & ((r - b) >= 18)
        & (chroma >= 18)
    )


def opaque_bounds(alpha: np.ndarray, threshold: int = 24) -> tuple[int, int, int, int, int]:
    ys, xs = np.where(alpha > threshold)
    if xs.size == 0:
        return 0, 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(xs.size)


def geometric_hair_masks(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    alpha = rgba[..., 3]
    rgb = rgba[..., :3]
    min_x, min_y, max_x, max_y, opaque = opaque_bounds(alpha)
    height, width = alpha.shape
    front = np.zeros((height, width), dtype=bool)
    back = np.zeros((height, width), dtype=bool)
    if opaque == 0:
        return front, back, 0.0
    box_w = max(1, max_x - min_x + 1)
    box_h = max(1, max_y - min_y + 1)
    cx = 0.5 * (min_x + max_x)
    cy = min_y + 0.40 * box_h
    rx = 0.24 * box_w
    ry = 0.30 * box_h
    yy, xx = np.ogrid[:height, :width]
    opaque_m = alpha > 24
    skin = pixel_looks_like_skin(rgb)
    nx = (xx - cx) / max(rx, 1.0)
    ny = (yy - cy) / max(ry, 1.0)
    dist = nx * nx + ny * ny
    fy = (yy - min_y) / box_h
    side_pad = int(0.22 * box_w)
    side = (xx < min_x + side_pad) | (xx > max_x - side_pad)
    hair_col = np.any(opaque_m & ~skin & (fy <= 0.34) & side, axis=0)
    bangs = opaque_m & ~skin & (dist <= 1.05) & (ny < -0.05) & (fy < 0.48)
    crown = opaque_m & ~skin & (fy < 0.28) & (dist > 0.35)
    hanging = opaque_m & ~skin & (fy < 0.92) & hair_col[np.newaxis, :]
    hair = bangs | crown | hanging
    front = bangs | (hair & (fy < 0.50) & (np.abs(nx) < 1.2))
    back = hair & ~front
    coverage = float(hair.sum()) / float(opaque)
    return front, back, coverage


def apply_mask(rgba: np.ndarray, mask: np.ndarray) -> Image.Image:
    out = rgba.copy()
    out[..., 3] = np.where(mask, out[..., 3], 0)
    return Image.fromarray(out, mode="RGBA")


def load_sam2() -> dict[str, Any] | None:
    global _sam2, _sam2_failed
    if HAIR_BACKEND == "geometric":
        return None
    if _sam2 is not None:
        return _sam2
    if _sam2_failed:
        return None
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except Exception as error:
        print(f"sam2 import failed: {error}")
        _sam2_failed = True
        return None
    model_id = SAM2_MODEL
    device = os.getenv("BIREFNET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        _sam2 = {"processor": processor, "model": model, "device": device, "torch": torch}
        print(f"hair sam2 loaded model={model_id} device={device}")
        return _sam2
    except Exception as error:
        print(f"sam2 load failed: {error}")
        _sam2_failed = True
        return None


def sam2_hair_mask(image: Image.Image, alpha: np.ndarray) -> np.ndarray | None:
    runtime = load_sam2()
    if runtime is None:
        return None
    min_x, min_y, max_x, max_y, opaque = opaque_bounds(alpha)
    if opaque == 0:
        return None
    box_h = max(1, max_y - min_y + 1)
    box = [float(min_x), float(min_y), float(max_x), float(min_y + int(box_h * 0.52))]
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]
    rgb = image.convert("RGB")
    try:
        inputs = processor(images=rgb, input_boxes=[[[box]]], return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs.get("reshaped_input_sizes")
        )
        mask = masks[0]
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask)
        while mask.ndim > 2:
            mask = mask[0]
        if mask.shape[0] != alpha.shape[0] or mask.shape[1] != alpha.shape[1]:
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((alpha.shape[1], alpha.shape[0]), Image.BILINEAR)
            mask = np.asarray(mask_img) > 127
        else:
            mask = mask > 0.5
        mask = mask & (alpha > 24)
        coverage = float(mask.sum()) / float(opaque)
        if coverage < HAIR_MIN_COVERAGE or coverage > 0.72:
            return None
        return mask
    except Exception as error:
        print(f"sam2 hair mask failed: {error}")
        return None


def split_hair_layers(image: Image.Image) -> dict[str, Any]:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[..., 3]
    backend = "silhouette-split"
    sam_mask = None
    if HAIR_BACKEND != "geometric":
        sam_mask = sam2_hair_mask(image, alpha)
    front, back, coverage = geometric_hair_masks(rgba)
    if sam_mask is not None:
        backend = "sam2"
        rgb = rgba[..., :3]
        skin = pixel_looks_like_skin(rgb)
        min_x, min_y, max_x, max_y, opaque = opaque_bounds(alpha)
        box_h = max(1, max_y - min_y + 1)
        yy, xx = np.ogrid[: alpha.shape[0], : alpha.shape[1]]
        cx = 0.5 * (min_x + max_x)
        box_w = max(1, max_x - min_x + 1)
        nx = (xx - cx) / max(0.24 * box_w, 1.0)
        fy = (yy - min_y) / box_h
        hair = sam_mask & ~skin
        front = hair & (fy < 0.50) & (np.abs(nx) < 1.2)
        back = hair & ~front
        coverage = float(hair.sum()) / float(max(opaque, 1))
    skipped = coverage < HAIR_MIN_COVERAGE
    return {
        "skipped": skipped,
        "coverage": coverage,
        "backend": backend,
        "front": None if skipped else apply_mask(rgba, front),
        "back": None if skipped else apply_mask(rgba, back),
    }
