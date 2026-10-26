#!/usr/bin/env python3
"""Anime pose control workload for the OmniServe manifest runtime.

A reference photo is reduced to a 133-point skeleton, drawn as a thin pose map,
VAE-encoded and fused into the 28-block Anima v1.0 DiT at patch-embed. The
adapter LoRA is merged into the weights once at load, so a pose request costs
the same per step as plain text-to-image. `warmup()` compiles every served
shape before the first request and persists the inductor cache.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from workloads import anima_image as base
from workloads import anime_control_core as core
from workloads.anime_control_core import controlled_patch_embed, merge_lora  # noqa: F401

NAME = "pose-control"
POSE_REPO = os.getenv("ANIMA_POSE_REPO", "Claquasse/Anima-Control-Pose")
POSE_REVISION = os.getenv("ANIMA_POSE_REVISION", "8f559771d5a49a02fa03f7df2a05ccb7eecb3a2a")
POSE_FILE = os.getenv("ANIMA_POSE_FILE", "anima_pose_preview2.safetensors")
DETECTOR_FILES = (
    "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
    "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
)
DETECTOR_DEVICE = os.getenv("ANIMA_POSE_DETECTOR_DEVICE", "cpu")
STYLES = ("R0_thin", "R1_thick", "R2_puppet", "heatmap")
ALLOWED_INPUTS = core.COMMON_INPUTS | {
    "reference_image", "skeleton_image", "control_strength", "skeleton_style", "hands", "face", "feet",
}

BODY = list(range(17))
FEET = list(range(17, 23))
FACE = list(range(23, 91))
HAND = list(range(91, 133))
COCO_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
              (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]
_PALETTE = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0), (0, 255, 128),
            (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128),
            (180, 180, 180), (220, 220, 220)]
LIMB_COLOR = {limb: _PALETTE[i % len(_PALETTE)] for i, limb in enumerate(COCO_LIMBS)}
JOINT_COLOR = [_PALETTE[i % len(_PALETTE)] for i in range(17)]
_WIDTHS = {"R0_thin": (2, 3), "R1_thick": (8, 6), "R2_puppet": (22, 11), "heatmap": (2, 3)}


_pipe = None
_embedder = None
_estimator = None
timings = core.timings
_sizes = core.parse_sizes
control_tokens = core.patchify_latent
decode_reference = core.decode_image


class NoPersonError(ValueError):
    pass


def split_adapter(state: dict):
    blocks, adapter, extras = core.split_lora(state)
    control = {
        key[len("control_embedder.proj."):]: value
        for key, value in extras.items()
        if key.startswith("control_embedder.proj.")
    }
    if set(control) != {"weight", "bias"}:
        raise ValueError("adapter is missing control_embedder weights")
    return blocks, adapter, control


def load_pipeline():
    global _pipe, _embedder
    if _pipe is not None:
        return _pipe

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pose_path = os.getenv("ANIMA_POSE_PATH", "").strip() or hf_hub_download(
        POSE_REPO, POSE_FILE, revision=POSE_REVISION
    )
    blocks, adapter_pairs, control = split_adapter(load_file(pose_path, device="cuda"))
    pipe = core.build_pipeline(NAME, blocks, adapter_pairs)
    embedder = torch.nn.Linear(64, pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim)
    embedder.load_state_dict({k: v.float() for k, v in control.items()})
    _embedder = embedder.to("cuda", torch.float32).eval().requires_grad_(False)
    _pipe = core.finalize_pipeline(NAME, pipe)
    return _pipe


def _detector_dir() -> Path:
    try:
        from rtmlib.tools.file import _get_rtmhub_dir

        return Path(_get_rtmhub_dir()) / "checkpoints"
    except Exception:
        cache = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        return Path(cache) / "rtmlib" / "hub" / "checkpoints"


def _default_estimator():
    global _estimator
    if _estimator is None:
        import shutil

        from huggingface_hub import hf_hub_download

        target = _detector_dir()
        target.mkdir(parents=True, exist_ok=True)
        for name in DETECTOR_FILES:
            destination = target / name
            if not destination.is_file() or destination.stat().st_size < 1_000_000:
                shutil.copyfile(
                    hf_hub_download(POSE_REPO, f"detector/{name}", revision=POSE_REVISION), destination
                )
        from rtmlib import Wholebody

        _estimator = Wholebody(to_openpose=False, mode="balanced", backend="onnxruntime", device=DETECTOR_DEVICE)
    return _estimator


def detect_pose(image_rgb: np.ndarray, resolution: int, estimator=None):
    import cv2

    pose = estimator or _default_estimator()
    canvas = cv2.cvtColor(cv2.resize(image_rgb, (resolution, resolution)), cv2.COLOR_RGB2BGR)
    keypoints, scores = pose(canvas)
    if keypoints is None or len(keypoints) == 0:
        raise NoPersonError("no person detected in the reference image")
    keypoints, scores = np.asarray(keypoints, float), np.asarray(scores, float)
    if keypoints.ndim == 2:
        keypoints, scores = keypoints[None], scores[None]
    best = max(range(len(keypoints)), key=lambda i: _extent(keypoints[i], scores[i]))
    if (scores[best][BODY] >= 0.3).sum() < 5:
        raise NoPersonError("no full person pose found in the reference image")
    return keypoints[best], scores[best]


def _extent(keypoints, scores) -> float:
    visible = keypoints[:17][scores[:17] >= 0.3]
    if len(visible) < 2:
        return 0.0
    span = visible.max(axis=0) - visible.min(axis=0)
    return float(span[0] * span[1]) * float(scores[:17].mean())


def render_skeleton(keypoints, scores, width, height, source_resolution, style="R0_thin",
                    hands=True, face=True, feet=True, threshold=0.3) -> np.ndarray:
    import cv2

    if style not in STYLES:
        raise ValueError(f"skeleton_style must be one of {', '.join(STYLES)}")
    points = np.asarray(keypoints, float) * np.array([width / source_resolution, height / source_resolution])
    scores = np.asarray(scores, float).copy()
    if not hands:
        scores[HAND] = 0.0
    if not face:
        scores[FACE] = 0.0
    if not feet:
        scores[FEET] = 0.0
    stick, joint = _WIDTHS[style]
    canvas = np.zeros((height, width, 3), np.uint8)

    def at(index):
        return int(points[index][0]), int(points[index][1])

    if style == "heatmap":
        acc = np.zeros((height, width, 3), np.float32)
        for index in BODY + (FEET if feet else []):
            if scores[index] < threshold:
                continue
            x, y = at(index)
            if 0 <= x < width and 0 <= y < height:
                blob = np.zeros((height, width), np.float32)
                blob[y, x] = 1.0
                blob = cv2.GaussianBlur(blob, (0, 0), 11)
                blob /= blob.max() + 1e-9
                acc += blob[..., None] * np.array(JOINT_COLOR[index % 17], np.float32)
        return np.clip(acc, 0, 255).astype(np.uint8)
    for a, b in COCO_LIMBS:
        if scores[a] >= threshold and scores[b] >= threshold:
            cv2.line(canvas, at(a), at(b), LIMB_COLOR[(a, b)], stick)
    for index in BODY:
        if scores[index] >= threshold:
            cv2.circle(canvas, at(index), joint, (255, 255, 255), -1)
    for index in FEET:
        if scores[index] >= threshold:
            cv2.circle(canvas, at(index), max(1, joint // 2), (255, 255, 255), -1)
    if style == "R0_thin":
        for index in HAND:
            if scores[index] >= threshold:
                cv2.circle(canvas, at(index), 2, (0, 255, 255), -1)
        for index in FACE:
            if scores[index] >= threshold:
                cv2.circle(canvas, at(index), 2, (255, 255, 255), -1)
    return canvas


def fit_skeleton(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((width, height), Image.BICUBIC))


def generate(pipe, values: dict, skeleton=None):
    import torch

    common = core.common_values(values)
    strength = float(values.get("control_strength", 1.0))
    if not 0.0 <= strength <= 2.0:
        raise ValueError("control_strength must be between 0 and 2")
    marks = core.Marks()
    if skeleton is None and values.get("skeleton_image"):
        skeleton = fit_skeleton(core.decode_image(values["skeleton_image"], "skeleton_image"), common["width"], common["height"])
    if skeleton is None:
        reference = decode_reference(values.get("reference_image"))
        resolution = max(common["width"], common["height"])
        keypoints, scores = detect_pose(reference, resolution)
        skeleton = render_skeleton(
            keypoints, scores, common["width"], common["height"], resolution,
            str(values.get("skeleton_style", "R0_thin")),
            hands=bool(values.get("hands", True)), face=bool(values.get("face", True)),
            feet=bool(values.get("feet", True)),
        )
    marks("detect")
    with torch.inference_mode():
        tokens = control_tokens(_embedder, core.encode_latent(pipe, skeleton)) * strength
        pipe.control_slot.tokens = tokens
        marks("control")
        try:
            image = core.sample(pipe, common, marks=marks)
        finally:
            pipe.control_slot.tokens = None
    marks.publish()
    return image, skeleton, common["seed"]


def handler(job, pipe=None):
    values = base._inputs(job)
    unknown = sorted(set(values) - ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"unknown pose inputs: {', '.join(unknown)}")
    output_format = str(values.get("output_format", "webp")).lower()
    if output_format not in base.FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    image, skeleton, seed = generate(pipe or load_pipeline(), values)
    return core.encode_outputs(image, {"skeleton": skeleton}, seed, output_format, "pose")


def _synthetic_skeleton(width, height):
    keypoints = np.zeros((133, 2))
    scores = np.zeros(133)
    for index, (x, y) in enumerate(
        [(50, 10), (46, 8), (54, 8), (42, 10), (58, 10), (36, 28), (64, 28), (30, 46), (70, 46),
         (26, 62), (74, 62), (42, 58), (58, 58), (41, 78), (59, 78), (40, 96), (60, 96)]
    ):
        keypoints[index], scores[index] = (x, y), 0.9
    return render_skeleton(keypoints, scores, width, height, 100)


def warmup(sizes=None, save=None):
    pipe = load_pipeline()
    try:
        detect_pose(np.zeros((64, 64, 3), np.uint8), 256)
    except NoPersonError:
        pass
    return core.run_warmup(
        NAME, pipe,
        lambda p, w, h: generate(p, core.warm_values(w, h), skeleton=_synthetic_skeleton(w, h)),
        sizes, save,
    )


if __name__ == "__main__":
    core.serve(NAME, handler, warmup)
