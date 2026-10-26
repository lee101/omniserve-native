#!/usr/bin/env python3
"""Anime depth control workload: a depth map fixes the scene geometry.

Implements the VACE-style depth ControlNet for Anima v1.0: four extra Anima
blocks (spaced every seventh base block) run on a patch-embedded depth latent
plus the base tokens, and their zero-conv projections are added to the base
residual stream after blocks 0, 7, 14 and 21. A depth map can be supplied or
estimated from a reference photo.
"""

from __future__ import annotations

import os
import re

import numpy as np

from workloads import anima_image as base
from workloads import anime_control_core as core

NAME = "depth-control"
REPO = os.getenv("ANIMA_DEPTH_REPO", "TaihoC/Anima-ControlNet-VACE-Depth")
REVISION = os.getenv("ANIMA_DEPTH_REVISION", "4ffaa3cf5854d5eac5b13447e043d2a8ff2cc4e7")
FILE = os.getenv("ANIMA_DEPTH_FILE", "anima-vace-depth.safetensors")
ESTIMATOR_REPO = os.getenv("ANIMA_DEPTH_ESTIMATOR", "depth-anything/Depth-Anything-V2-Small-hf")
ALLOWED_INPUTS = core.COMMON_INPUTS | {
    "depth_image", "reference_image", "control_strength", "control_start", "control_end", "invert_depth",
}
EVERY_N = 7
_pipe = None
_control = None
_estimator = None
timings = core.timings

CONTROL_KEY = re.compile(r"control_blocks\.(\d+)\.(before_proj|after_proj)\.(weight|bias)")
BLOCK_KEY = re.compile(r"control_blocks\.(\d+)\.block\.(.+)\.(weight|bias)")


def convert_keys(tensors: dict):
    """Rename VACE checkpoint keys to diffusers block names; returns (embedder, blocks)."""
    embedder, blocks = {}, {}
    for key, value in tensors.items():
        if key == "control_embedder.proj.1.weight":
            embedder["weight"] = value
            continue
        match = CONTROL_KEY.fullmatch(key)
        if match:
            index, name, kind = match.groups()
            blocks.setdefault(int(index), {})[f"{name}.{kind}"] = value
            continue
        match = BLOCK_KEY.fullmatch(key)
        if match:
            index, module, kind = match.groups()
            mapped = base.BLOCK_MAP.get(module)
            if mapped is None:
                raise ValueError(f"unmapped control block module: {key}")
            blocks.setdefault(int(index), {})[f"{mapped}.{kind}"] = value
            continue
        raise ValueError(f"unexpected key in depth adapter: {key}")
    if "weight" not in embedder or not blocks:
        raise ValueError("depth adapter is missing its embedder or control blocks")
    if sorted(blocks) != list(range(len(blocks))):
        raise ValueError("control block indices are not contiguous")
    return embedder, [blocks[i] for i in sorted(blocks)]


class Control:
    def __init__(self, embedder, blocks):
        self.embedder = embedder
        self.blocks = blocks
        self.tokens = None
        self.strength = None
        self.run = None


def build_control(pipe, tensors: dict):
    import torch
    from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock

    embedder_weights, block_weights = convert_keys(tensors)
    config = pipe.transformer.config
    dim = config.num_attention_heads * config.attention_head_dim
    embedder = torch.nn.Linear(embedder_weights["weight"].shape[1], dim, bias=False)
    embedder.load_state_dict({"weight": embedder_weights["weight"].float()})
    blocks = torch.nn.ModuleList()
    for index, weights in enumerate(block_weights):
        block = CosmosTransformerBlock(
            num_attention_heads=config.num_attention_heads, attention_head_dim=config.attention_head_dim,
            cross_attention_dim=config.text_embed_dim, mlp_ratio=config.mlp_ratio,
            adaln_lora_dim=config.adaln_lora_dim, before_proj=index == 0, after_proj=True,
        )
        missing, unexpected = block.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"depth control block {index}: missing={missing[:6]} unexpected={unexpected[:6]}")
        blocks.append(block)
    if len(pipe.transformer.transformer_blocks) // len(blocks) != EVERY_N:
        raise RuntimeError("depth adapter spacing does not match the base block count")
    control = Control(embedder.to("cuda", torch.bfloat16).eval().requires_grad_(False),
                      blocks.to("cuda", torch.bfloat16).eval().requires_grad_(False))
    pipe.transformer.register_to_config(controlnet_block_every_n=EVERY_N)
    return control


def run_control(control, tokens, base_tokens, context, embedded_timestep, temb, rope):
    hints = []
    hidden = tokens.expand(base_tokens.shape[0], -1, -1)
    for block in control.blocks:
        hidden, hint = block(
            hidden, context, embedded_timestep, temb, image_rotary_emb=rope,
            latents=base_tokens if block.before_proj is not None else None,
        )
        hints.append(hint)
    return hints


def load_pipeline():
    global _pipe, _control
    if _pipe is not None:
        return _pipe

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = os.getenv("ANIMA_DEPTH_PATH", "").strip() or hf_hub_download(REPO, FILE, revision=REVISION)
    pipe = core.build_pipeline(NAME)
    control = build_control(pipe, load_file(path, device="cuda"))
    control.run = torch.compile(run_control, dynamic=False) if core.COMPILE_ENABLED else run_control
    _control = control
    _pipe = core.finalize_pipeline(NAME, pipe)
    return _pipe


def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """Relative depth from a photo, near = bright, as a 3-channel uint8 map."""
    global _estimator
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if _estimator is None:
        processor = AutoImageProcessor.from_pretrained(ESTIMATOR_REPO)
        model = AutoModelForDepthEstimation.from_pretrained(ESTIMATOR_REPO).to("cuda", torch.float16).eval()
        _estimator = (processor, model)
    processor, model = _estimator
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.inference_mode():
        depth = model(pixel_values=inputs["pixel_values"].to("cuda", torch.float16)).predicted_depth
        depth = torch.nn.functional.interpolate(depth[None], size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
    depth = depth.float()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return np.repeat((depth.cpu().numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)


def fit_map(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(depth).resize((width, height), Image.BICUBIC))


def generate(pipe, values: dict, depth=None):
    import torch

    common = core.common_values(values)
    strength = float(values.get("control_strength", 1.0))
    if not 0.0 <= strength <= 2.0:
        raise ValueError("control_strength must be between 0 and 2")
    start, end = float(values.get("control_start", 0.0)), float(values.get("control_end", 1.0))
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("control_start and control_end must satisfy 0 <= start < end <= 1")
    marks = core.Marks()
    width, height, steps = common["width"], common["height"], common["steps"]
    if depth is None:
        if values.get("depth_image"):
            depth = fit_map(core.decode_image(values["depth_image"], "depth_image"), width, height)
        else:
            reference = core.decode_image(values.get("reference_image"), "reference_image")
            depth = fit_map(estimate_depth(reference), width, height)
        if bool(values.get("invert_depth", False)):
            depth = 255 - depth
    marks("depth")
    transformer = core.raw_transformer(pipe)
    with torch.inference_mode():
        latent = core.encode_latent(pipe, depth)
        latent = torch.cat([latent, torch.zeros_like(latent[:, :1])], dim=1)
        _control.tokens = core.patchify_latent(_control.embedder, latent).flatten(1, 3).to(torch.bfloat16)
        marks("control")

        def step_hook(index, sigma, model_input, timestep, context):
            if not start * steps <= index < end * steps:
                return None
            padded = torch.cat([model_input, model_input.new_zeros(model_input.shape[0], 1, *model_input.shape[2:])], dim=1)
            rope = transformer.rope(padded, fps=None)
            base_tokens = transformer.patch_embed(padded).flatten(1, 3)
            temb, embedded_timestep = transformer.time_embed(base_tokens, timestep)
            hints = _control.run(_control, _control.tokens, base_tokens, context, embedded_timestep, temb, rope)
            return {"block_controlnet_hidden_states": [hint * strength for hint in hints]}

        image = core.sample(pipe, common, step_hook=step_hook, marks=marks)
    marks.publish()
    return image, depth, common["seed"]


def handler(job, pipe=None):
    values = base._inputs(job)
    unknown = sorted(set(values) - ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"unknown depth inputs: {', '.join(unknown)}")
    output_format = str(values.get("output_format", "webp")).lower()
    if output_format not in base.FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    image, depth, seed = generate(pipe or load_pipeline(), values)
    return core.encode_outputs(image, {"depth": depth}, seed, output_format, "depth")


def _synthetic_depth(width, height):
    ramp = np.linspace(0, 255, height, dtype=np.uint8)[:, None].repeat(width, axis=1)
    return np.repeat(ramp[..., None], 3, axis=2)


def warmup(sizes=None, save=None):
    pipe = load_pipeline()
    return core.run_warmup(
        NAME, pipe,
        lambda p, w, h: generate(p, core.warm_values(w, h), depth=_synthetic_depth(w, h)),
        sizes, save,
    )


if __name__ == "__main__":
    core.serve(NAME, handler, warmup)
