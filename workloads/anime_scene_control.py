"""Anime scene control workload: colour regions and a 3D body pose steer the layout.

Loads the ControlNet-LLLite v2 adapters for Anima v1.0 (regional instance-colour
map and shaded 3D-pose mesh render). Each adapter has its own conditioning
network (one 64-channel embedding per token) and a FiLM bottleneck on the input
of every self-attention q/k/v linear. Both can run together for multi-character
scenes; unused adapters are skipped by a zero strength.
"""

from __future__ import annotations

import os
import re

import numpy as np

from workloads import anima_image as base
from workloads import anime_control_core as core

NAME = "scene-control"
REPO = os.getenv("ANIMA_SCENE_REPO", "toyxyz/Anima_ControlNet-LLLite")
REVISION = os.getenv("ANIMA_SCENE_REVISION", "07199fde970384b7bfc1f50d7290be26499c95c5")
ADAPTER_FILES = {
    "regional": os.getenv("ANIMA_SCENE_REGIONAL_FILE", "anima_regional_4335.safetensors"),
    "pose": os.getenv("ANIMA_SCENE_POSE_FILE", "anima_3dpose_6200.safetensors"),
}
LOADED = [item for item in os.getenv("ANIMA_SCENE_ADAPTERS", "regional,pose").split(",") if item in ADAPTER_FILES]
ALLOWED_INPUTS = core.COMMON_INPUTS | {
    "region_map", "regions", "regional_strength", "pose_map", "pose_strength", "control_start", "control_end",
}
MAX_REGIONS = 12
_pipe = None
_states: dict = {}
timings = core.timings


def region_mask(regions, width: int, height: int) -> np.ndarray:
    """Render rectangles (fractions of the canvas) as flat colours on black."""
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError(f"regions must list 1 to {MAX_REGIONS} rectangles")
    canvas = np.zeros((height, width, 3), np.uint8)
    for item in regions:
        try:
            x, y, w, h = (float(item[key]) for key in ("x", "y", "w", "h"))
            color = str(item.get("color", "#ff0000")).lstrip("#")
            rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError("each region needs x, y, w, h (0-1) and a #rrggbb color") from error
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError("region coordinates must be fractions of the canvas")
        x0, y0 = int(x * width), int(y * height)
        x1, y1 = min(width, int((x + w) * width)), min(height, int((y + h) * height))
        canvas[y0:y1, x0:x1] = rgb
    return canvas


def prepare_mask(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bicubic-resize to the output size and clamp, as the reference node does."""
    from PIL import Image

    resized = Image.fromarray(image).resize((width, height), Image.BICUBIC)
    return np.asarray(resized)


def conditioning_module():
    import torch
    from torch import nn

    def gn(channels):
        return nn.GroupNorm(8, channels)

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm1, self.conv1 = gn(channels), nn.Conv2d(channels, channels, 3, padding=1)
            self.norm2, self.conv2 = gn(channels), nn.Conv2d(channels, channels, 3, padding=1)

        def forward(self, x):
            h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
            return x + self.conv2(torch.nn.functional.silu(self.norm2(h)))

    class ASPP(nn.Module):
        def __init__(self, channels, dilations=(1, 2, 4, 8)):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, 1) if d == 1 else nn.Conv2d(channels, channels, 3, padding=d, dilation=d),
                    gn(channels),
                )
                for d in dilations
            ])
            self.global_conv = nn.Sequential(nn.Conv2d(channels, channels, 1), gn(channels))
            self.proj = nn.Sequential(nn.Conv2d(channels * (len(dilations) + 1), channels, 1), gn(channels))

        def forward(self, x):
            silu = torch.nn.functional.silu
            outs = [silu(branch(x)) for branch in self.branches]
            g = silu(self.global_conv(x.mean((2, 3), keepdim=True)))
            g = torch.nn.functional.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return silu(self.proj(torch.cat(outs + [g], 1)))

    class Conditioning(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.norm1 = nn.Conv2d(3, 64, 4, stride=4), gn(64)
            self.conv2, self.norm2 = nn.Conv2d(64, 64, 3, padding=1), gn(64)
            self.conv3, self.norm3 = nn.Conv2d(64, 128, 4, stride=4), gn(128)
            self.resblocks = nn.ModuleList([ResBlock(128) for _ in range(4)])
            self.aspp = ASPP(128)
            self.proj = nn.Conv2d(128, 64, 1)
            self.out_norm = nn.LayerNorm(64)

        def forward(self, image):
            silu = torch.nn.functional.silu
            h = silu(self.norm1(self.conv1(image)))
            h = silu(self.norm2(self.conv2(h)))
            h = silu(self.norm3(self.conv3(h)))
            for block in self.resblocks:
                h = block(h)
            h = self.proj(self.aspp(h))
            return self.out_norm(h.flatten(2).transpose(1, 2))

    return Conditioning()


class Layer:
    """One q/k/v module: x + up(silu(FiLM(mid([cond + depth, silu(down(x))])))) * strength."""

    def __init__(self, weights: dict, state):
        import torch

        self.torch = torch
        self.state = state
        self.down = (weights["down.weight"], weights["down.bias"])
        self.mid = (weights["mid.weight"], weights["mid.bias"])
        self.film = (weights["cond_to_film.weight"], weights["cond_to_film.bias"])
        self.up = (weights["up.weight"], weights["up.bias"])
        self.depth = weights["depth_embed"]

    def __call__(self, module, args):
        f = self.torch.nn.functional
        x = args[0]
        cond = self.state.cond_emb + self.depth
        h = f.silu(f.linear(x, *self.down))
        gamma, beta = f.linear(cond, *self.film).chunk(2, -1)
        mid = f.linear(self.torch.cat([cond.expand(x.shape[0], -1, -1), h], -1), *self.mid)
        mid = f.silu(mid * (1 + gamma) + beta)
        return (x + f.linear(mid, *self.up) * self.state.strength.to(x.dtype),)


LAYER_KEY = re.compile(r"lllite_dit_blocks_(\d+)_self_attn_([qkv])_proj\.(.+)")


def split_weights(tensors: dict):
    conditioning, layers = {}, {}
    for key, value in tensors.items():
        if key.startswith("lllite_conditioning1."):
            conditioning[key[len("lllite_conditioning1."):]] = value
            continue
        match = LAYER_KEY.fullmatch(key)
        if not match:
            raise ValueError(f"unexpected key in regional adapter: {key}")
        index, which, rest = match.groups()
        layers.setdefault((int(index), f"to_{which}"), {})[rest] = value
    return conditioning, layers


class State:
    cond_emb = None
    strength = None
    net = None


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pipe = core.build_pipeline(NAME)
    blocks = pipe.transformer.transformer_blocks
    for adapter in LOADED:
        path = os.getenv(f"ANIMA_SCENE_{adapter.upper()}_PATH", "").strip() or hf_hub_download(
            REPO, ADAPTER_FILES[adapter], revision=REVISION
        )
        conditioning_weights, layer_weights = split_weights(load_file(path, device="cuda"))
        if len(layer_weights) != 3 * len(blocks):
            raise RuntimeError(f"{adapter} adapter covers {len(layer_weights)} modules, expected {3 * len(blocks)}")
        net = conditioning_module()
        net.load_state_dict({k: v.float() for k, v in conditioning_weights.items()}, strict=True)
        state = State()
        state.net = net.to("cuda", torch.bfloat16).eval().requires_grad_(False)
        state.strength = torch.zeros((), device="cuda", dtype=torch.float32)
        state.cond_emb = torch.zeros(1, 1, 64, device="cuda", dtype=torch.bfloat16)
        for (index, target), weights in sorted(layer_weights.items()):
            layer = Layer({k: v.to("cuda", torch.bfloat16) for k, v in weights.items()}, state)
            getattr(blocks[index].attn1, target).register_forward_pre_hook(layer)
        _states[adapter] = state
    _pipe = core.finalize_pipeline(NAME, pipe)
    return _pipe


def condition(state, image: np.ndarray, width: int, height: int):
    import torch

    picture = torch.from_numpy(np.ascontiguousarray(prepare_mask(image, width, height))).to("cuda", torch.float32)
    picture = (picture / 255.0 * 2.0 - 1.0).permute(2, 0, 1)[None].to(torch.bfloat16)
    state.cond_emb = state.net(picture)


def generate(pipe, values: dict, maps=None):
    import torch

    common = core.common_values(values)
    start, end = float(values.get("control_start", 0.0)), float(values.get("control_end", 1.0))
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("control_start and control_end must satisfy 0 <= start < end <= 1")
    width, height, steps = common["width"], common["height"], common["steps"]
    if width % 16 or height % 16:
        raise ValueError("width and height must be divisible by 16")
    marks = core.Marks()
    if maps is None:
        maps = {}
        if values.get("regions") is not None:
            maps["regional"] = region_mask(values["regions"], width, height)
        elif values.get("region_map"):
            maps["regional"] = core.decode_image(values["region_map"], "region_map")
        if values.get("pose_map"):
            maps["pose"] = core.decode_image(values["pose_map"], "pose_map")
    strengths = {}
    for adapter in ("regional", "pose"):
        value = float(values.get(f"{adapter}_strength", 1.0 if adapter in maps else 0.0))
        if not -2.0 <= value <= 2.0:
            raise ValueError(f"{adapter}_strength must be between -2 and 2")
        if adapter in maps and adapter not in _states:
            raise ValueError(f"the {adapter} adapter is not loaded on this worker")
        strengths[adapter] = value if adapter in maps else 0.0
    if not maps:
        raise ValueError("provide region_map, regions or pose_map")
    marks("maps")
    with torch.inference_mode():
        for adapter, state in _states.items():
            if adapter in maps:
                condition(state, maps[adapter], width, height)
            else:
                state.cond_emb = torch.zeros(1, (width // 16) * (height // 16), 64, device="cuda", dtype=torch.bfloat16)
        marks("control")

        def step_hook(index, sigma, model_input, timestep, context):
            active = start * steps <= index < end * steps
            for adapter, state in _states.items():
                state.strength.fill_(strengths[adapter] if active else 0.0)
            return None

        try:
            image = core.sample(pipe, common, step_hook=step_hook, marks=marks)
        finally:
            for state in _states.values():
                state.strength.fill_(0.0)
    marks.publish()
    return image, maps, common["seed"]


def handler(job, pipe=None):
    values = base._inputs(job)
    unknown = sorted(set(values) - ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"unknown scene inputs: {', '.join(unknown)}")
    output_format = str(values.get("output_format", "webp")).lower()
    if output_format not in base.FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    image, maps, seed = generate(pipe or load_pipeline(), values)
    extras = {f"{name}-map": prepare_mask(array, image.width, image.height) for name, array in maps.items()}
    return core.encode_outputs(image, extras, seed, output_format, "scene")


def _synthetic_pose(width, height):
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[height // 8: 7 * height // 8, 3 * width // 8: 5 * width // 8] = (200, 120, 60)
    return canvas


def warmup(sizes=None, save=None):
    pipe = load_pipeline()

    def request(p, w, h):
        maps = {"regional": region_mask([{"x": 0, "y": 0, "w": 0.5, "h": 1, "color": "#ff0000"}], w, h)}
        if "pose" in _states:
            maps["pose"] = _synthetic_pose(w, h)
        generate(p, core.warm_values(w, h), maps=maps)

    return core.run_warmup(NAME, pipe, request, sizes, save)


if __name__ == "__main__":
    core.serve(NAME, handler, warmup)
