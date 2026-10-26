#!/usr/bin/env python3
"""Anime sketch control workload: canny edges or a scribble drive the drawing.

Applies the adjustable Canny/Scribble control LoRA for Anima: the control image
is VAE-encoded and appended to the target latent as a second, clean frame, the
LoRA is merged once at load, and only the first frame of each prediction is
used. This doubles the token count, so a step costs about 2.4x a plain step.
The LoRA was trained against Anima preview 3; it is run here on v1.0.
"""

from __future__ import annotations

import os

import numpy as np

from workloads import anima_image as base
from workloads import anime_control_core as core

NAME = "sketch-control"
REPO = os.getenv("ANIMA_SKETCH_REPO", "CabalResearch/Anima-Canny-Scribble-Adjustable-Control-LoRA")
REVISION = os.getenv("ANIMA_SKETCH_REVISION", "803df89f3f042a790b4af9d2e0679255726acc4c")
FILE = os.getenv("ANIMA_SKETCH_FILE", "Anima Scribble+Canny Variable Control LoRA.safetensors")
ALLOWED_INPUTS = core.COMMON_INPUTS | {
    "reference_image", "edge_image", "canny_low", "canny_high",
}
_pipe = None
timings = core.timings


def canny_map(rgb: np.ndarray, width: int, height: int, low: int = 25, high: int = 140) -> np.ndarray:
    """White edges on black at the output size (ComfyUI Canny 0.1/0.55 equals 25/140)."""
    import cv2

    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), low, high)
    return np.repeat(edges[..., None], 3, axis=2)


def fit_map(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((width, height), Image.NEAREST))


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = os.getenv("ANIMA_SKETCH_PATH", "").strip() or hf_hub_download(REPO, FILE, revision=REVISION)
    blocks, adapter, extras = core.split_lora(load_file(path, device="cuda"))
    if adapter or extras:
        raise RuntimeError(f"sketch adapter has unexpected tensors: {sorted(extras)[:4]} {sorted(adapter)[:2]}")
    _pipe = core.finalize_pipeline(NAME, core.build_pipeline(NAME, blocks, None))
    return _pipe


def generate(pipe, values: dict, control=None):
    import torch

    common = core.common_values(values)
    marks = core.Marks()
    width, height = common["width"], common["height"]
    if control is None:
        if values.get("edge_image"):
            control = fit_map(core.decode_image(values["edge_image"], "edge_image"), width, height)
        else:
            low, high = int(values.get("canny_low", 25)), int(values.get("canny_high", 140))
            if not 0 <= low < high <= 255:
                raise ValueError("canny_low and canny_high must satisfy 0 <= low < high <= 255")
            control = canny_map(core.decode_image(values.get("reference_image"), "reference_image"), width, height, low, high)
    marks("edges")
    with torch.inference_mode():
        reference = core.encode_latent(pipe, control)
        marks("control")
        image = core.sample(
            pipe, common, marks=marks,
            pack=lambda latents: torch.cat([latents, reference], dim=2),
            unpack=lambda velocity: velocity[:, :, :1],
        )
    marks.publish()
    return image, control, common["seed"]


def handler(job, pipe=None):
    values = base._inputs(job)
    unknown = sorted(set(values) - ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"unknown sketch inputs: {', '.join(unknown)}")
    output_format = str(values.get("output_format", "webp")).lower()
    if output_format not in base.FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    image, control, seed = generate(pipe or load_pipeline(), values)
    return core.encode_outputs(image, {"edges": control}, seed, output_format, "sketch")


def _synthetic_edges(width, height):
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[height // 4: height // 4 + 3, width // 4: 3 * width // 4] = 255
    canvas[height // 4: 3 * height // 4, width // 2: width // 2 + 3] = 255
    return canvas


def warmup(sizes=None, save=None):
    pipe = load_pipeline()
    return core.run_warmup(
        NAME, pipe,
        lambda p, w, h: generate(p, core.warm_values(w, h), control=_synthetic_edges(w, h)),
        sizes, save,
    )


if __name__ == "__main__":
    core.serve(NAME, handler, warmup)
