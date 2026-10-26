#!/usr/bin/env python3
"""Shared Anima v1.0 runtime for the anime control workloads.

Loads the 28-block v1.0 DiT and LLM adapter, optionally merges LoRA deltas into
them once, and exposes one sampler with per-step hooks so each control adapter
(pose, regional, depth, sketch) only supplies its conditioning. `run_warmup`
compiles every served shape before the first request and persists the cache.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path

import numpy as np

from workloads import anima_image as base

V1_REPO = os.getenv("ANIMA_V1_REPO", "circlestone-labs/Anima")
V1_REVISION = os.getenv("ANIMA_V1_REVISION", "f973fc41ec7545364ac9776c2440285f43ff2a30")
V1_FILE = os.getenv("ANIMA_V1_FILE", "split_files/diffusion_models/anima-base-v1.0.safetensors")
SNAPSHOT_PATTERNS = [
    "*.json", "*.txt", "*.jinja", "text_encoder/model.safetensors", "vae/*", "llm_adapter/*",
    "t5_tokenizer/*", "tokenizer/*", "scheduler/*", "*.py", "transformer/config.json",
]
COMPILE_ENABLED = os.getenv("ANIMA_COMPILE", "1").lower() not in {"0", "false", "off"}
COMPILE_MODE = os.getenv("ANIMA_COMPILE_MODE", "default")
COMPILE_ARTIFACT_DIR = Path(os.getenv("ANIMA_COMPILE_ARTIFACT_DIR", "/runpod-volume/anima-compile-cache"))
WARMUP_SIZES = os.getenv("ANIME_CONTROL_WARMUP_SIZES", os.getenv("ANIME_POSE_WARMUP_SIZES", "1024x1024,768x768"))
CFG_CUTOFF = float(os.getenv("ANIME_CONTROL_CFG_CUTOFF", os.getenv("ANIME_POSE_CFG_CUTOFF", "0.6")))
MAX_REFERENCE_BYTES = 12 << 20
MAX_PIXELS = 1024 * 1024
COMMON_INPUTS = {
    "workload", "kind", "profile", "execution_profile", "_omniserve_profile", "_omniserve_retry",
    "prompt", "negative_prompt", "width", "height",
    "num_inference_steps", "guidance_scale", "seed", "output_format", "cfg_cutoff",
}

_embed_cache: dict = {}
timings: dict = {}


def split_lora(state: dict, strip: str = "diffusion_model."):
    """Group a LoRA state dict into diffusers-named transformer and llm_adapter pairs.

    Accepts lora_A/lora_B and lora_down/lora_up naming; `alpha` scales by alpha/rank.
    Anything that is not a LoRA tensor is returned untouched in `extras`.
    """
    blocks, adapter, alphas, extras = {}, {}, {}, {}
    for key, value in state.items():
        name = key[len(strip):] if key.startswith(strip) else key
        if name.endswith(".alpha"):
            alphas[name[: -len(".alpha")]] = float(value)
            continue
        match = re.fullmatch(r"(.+)\.lora_(A|B|down|up)\.weight", name)
        if not match:
            extras[name] = value
            continue
        module, side = match.groups()
        side = {"down": "A", "up": "B"}.get(side, side)
        if module.startswith("llm_adapter."):
            target = module[len("llm_adapter."):] + ".weight"
            group = adapter
        else:
            parts = re.fullmatch(r"blocks\.(\d+)\.(.+)", module)
            mapped = base.BLOCK_MAP.get(parts.group(2)) if parts else None
            if mapped is None:
                raise ValueError(f"unmapped adapter module: {key}")
            target = f"transformer_blocks.{parts.group(1)}.{mapped}.weight"
            group = blocks
        group.setdefault(target, {"module": module})[side] = value
    for group in (blocks, adapter):
        for target, pair in group.items():
            if "A" not in pair or "B" not in pair:
                raise ValueError(f"incomplete lora pair for {target}")
            rank = pair["A"].shape[0]
            pair["scale"] = alphas.get(pair.pop("module"), rank) / rank
    return blocks, adapter, extras


def merge_lora(module, pairs: dict, strength: float = 1.0) -> int:
    import torch

    params = dict(module.named_parameters())
    with torch.no_grad():
        for target, pair in pairs.items():
            if target not in params:
                raise KeyError(f"adapter target not in model: {target}")
            weight = params[target]
            delta = pair["B"].to(weight.device, torch.float32) @ pair["A"].to(weight.device, torch.float32)
            weight.copy_((weight.float() + strength * pair.get("scale", 1.0) * delta).to(weight.dtype))
    return len(pairs)


def controlled_patch_embed(inner):
    import torch

    class ControlledPatchEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.tokens = None

        def forward(self, hidden_states):
            out = self.inner(hidden_states)
            if self.tokens is None:
                return out
            return out + self.tokens.to(out.dtype)

    return ControlledPatchEmbed()


def patchify_latent(embedder, latent):
    batch, channels, frames, height, width = latent.shape
    patches = latent.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
    patches = patches.permute(0, 2, 3, 5, 1, 4, 6).reshape(batch, frames, height // 2, width // 2, channels * 4)
    return embedder(patches.to(embedder.weight.dtype))


def compile_artifact(name: str) -> Path:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return COMPILE_ARTIFACT_DIR / f"anime-{name}-sm{major}{minor}-torch{torch.__version__.split('+')[0]}.bin"


def build_pipeline(name: str, transformer_pairs=None, adapter_pairs=None, lora_strength: float = 1.0):
    """Load the v1.0 pipeline, merge LoRA deltas once, compile and return it.

    The pipeline carries `control_slot` (patch-embed additive control tokens).
    """
    import torch
    from diffusers import CosmosTransformer3DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = os.getenv("ANIMA_CUDNN_BENCHMARK", "0").lower() in {"1", "true", "on"}

    pipeline_class = base._pipeline_class()
    config = dict(
        CosmosTransformer3DModel.load_config(
            base.BASE_MODEL_ID, subfolder="transformer", revision=base.BASE_MODEL_REVISION
        )
    )
    with torch.device("meta"):
        transformer = CosmosTransformer3DModel.from_config(config)

    checkpoint = os.getenv("ANIMA_V1_PATH", "").strip() or hf_hub_download(V1_REPO, V1_FILE, revision=V1_REVISION)
    raw = load_file(checkpoint, device="cuda")
    adapter_state = {
        key[len("net.llm_adapter."):]: value for key, value in raw.items() if key.startswith("net.llm_adapter.")
    }
    missing, unexpected = transformer.load_state_dict(base.remap_checkpoint(raw), strict=False, assign=True)
    still_meta = [item for item, value in transformer.state_dict().items() if value.is_meta]
    if missing or unexpected or still_meta:
        raise RuntimeError(
            f"Anima v1.0 remap incomplete: missing={missing[:8]} unexpected={unexpected[:8]} meta={still_meta[:8]}"
        )
    del raw

    transformer = transformer.to(dtype=torch.bfloat16)
    if transformer_pairs:
        merge_lora(transformer, transformer_pairs, lora_strength)

    from huggingface_hub import snapshot_download

    folder = snapshot_download(base.BASE_MODEL_ID, revision=base.BASE_MODEL_REVISION, allow_patterns=SNAPSHOT_PATTERNS)
    pipe = pipeline_class.from_pretrained(
        folder,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        transformer=None,
    )
    pipe.llm_adapter.load_state_dict(adapter_state, strict=True)
    if adapter_pairs:
        merge_lora(pipe.llm_adapter, adapter_pairs, lora_strength)
    pipe.register_modules(transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    mode = os.getenv("ANIMA_VAE_TILING", "auto").lower()
    if mode in {"1", "true", "on"} or (mode == "auto" and torch.cuda.get_device_properties(0).total_memory < 30 * 2**30):
        pipe.vae.enable_tiling()
    return pipe


def finalize_pipeline(name: str, pipe):
    """Install the control slot, fused kernels and compile; call after adapters add hooks."""
    import torch

    slot = controlled_patch_embed(pipe.transformer.patch_embed)
    pipe.transformer.patch_embed = slot
    pipe.control_slot = slot
    pipe.compiled_sizes = set()
    if os.getenv("ANIMA_FUSED", "1").lower() not in {"0", "false", "off"} and base._anima_cog() is not None:
        from anima_cog.patch import apply_fused_blocks

        patched = apply_fused_blocks(pipe.transformer)
        print(f"anime-{name}: fused adaLN kernels on {patched}/{len(pipe.transformer.transformer_blocks)} blocks", flush=True)
    if COMPILE_ENABLED:
        inductor = __import__("torch._inductor.config", fromlist=["config"])
        inductor.post_grad_custom_pre_pass = None
        inductor.post_grad_custom_post_pass = None
        artifact = compile_artifact(name)
        loader = getattr(torch.compiler, "load_cache_artifacts", None)
        if loader is not None and artifact.is_file():
            try:
                loader(artifact.read_bytes())
            except Exception as error:
                print(f"anime-{name}: compile cache replay failed ({error})", flush=True)
        pipe.eager_transformer = pipe.transformer
        pipe.transformer = torch.compile(pipe.transformer, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
    return pipe


def raw_transformer(pipe):
    return getattr(pipe.transformer, "_orig_mod", pipe.transformer)


def decode_image(value, field: str = "reference_image") -> np.ndarray:
    from PIL import Image, ImageOps

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        raw = base64.b64decode(text, validate=True)
    except Exception as error:
        raise ValueError(f"{field} must be base64 image data") from error
    if not raw or len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError(f"{field} must be between 1 byte and 12 MiB")
    try:
        picture = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as error:
        raise ValueError(f"{field} is not a readable image") from error
    return np.asarray(picture)


def cached_embeds(pipe, text, device, dtype):
    key = (id(pipe), text, str(dtype))
    embeds = _embed_cache.get(key)
    if embeds is None:
        embeds = pipe._encode_prompt([text], device, dtype, 512)
        if len(_embed_cache) >= base._EMBED_CACHE_LIMIT:
            _embed_cache.pop(next(iter(_embed_cache)))
        _embed_cache[key] = embeds
    return embeds


def vae_stats(pipe, like):
    import torch

    config = pipe.vae.config
    shape = (1, config.z_dim, 1, 1, 1)
    mean = torch.tensor(config.latents_mean).view(shape).to(like.device, like.dtype)
    inv_std = 1.0 / torch.tensor(config.latents_std).view(shape).to(like.device, like.dtype)
    return mean, inv_std


def encode_latent(pipe, rgb: np.ndarray):
    """uint8 HWC image -> normalized 16-channel latent (1, 16, 1, h/8, w/8)."""
    import torch

    picture = torch.from_numpy(np.ascontiguousarray(rgb)).to(pipe._execution_device, torch.float32)
    picture = (picture / 127.5 - 1.0).permute(2, 0, 1)[None, :, None]
    latent = pipe.vae.encode(picture.to(pipe.vae.dtype)).latent_dist.mode().float()
    mean, inv_std = vae_stats(pipe, latent)
    return (latent - mean) * inv_std


def common_values(values: dict, default_steps: int = 24) -> dict:
    prompt = str(values.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")
    width = base._dimension(values.get("width"), 1024)
    height = base._dimension(values.get("height"), 1024)
    if width * height > MAX_PIXELS:
        raise ValueError("images may contain at most 1048576 pixels")
    steps = int(values.get("num_inference_steps", default_steps))
    if not 10 <= steps <= 50:
        raise ValueError("num_inference_steps must be between 10 and 50")
    guidance = float(values.get("guidance_scale", 4.0))
    if not 1.0 <= guidance <= 8.0:
        raise ValueError("guidance_scale must be between 1.0 and 8.0")
    cutoff = float(values.get("cfg_cutoff", CFG_CUTOFF))
    if not 0.2 <= cutoff <= 1.0:
        raise ValueError("cfg_cutoff must be between 0.2 and 1")
    return {
        "prompt": prompt, "negative_prompt": str(values.get("negative_prompt", "")).strip(),
        "width": width, "height": height, "steps": steps, "guidance": guidance, "cutoff": cutoff,
        "seed": base._seed(values.get("seed")),
    }


class Marks:
    def __init__(self):
        import torch

        self._torch = torch
        self.values: dict = {}
        self._clock = time.perf_counter()

    def __call__(self, name):
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        now = time.perf_counter()
        self.values[name] = round(now - self._clock, 3)
        self._clock = now

    def publish(self):
        timings.clear()
        timings.update(self.values)


def sample(pipe, common: dict, *, step_hook=None, pack=None, unpack=None, marks=None):
    """Denoise with batched CFG (dropped after `cutoff`) and decode to a PIL image.

    step_hook(index, sigma, model_input, timestep, context) -> extra transformer kwargs.
    pack(latents) builds the transformer input (e.g. appends a reference frame);
    unpack(output) returns the target-frame velocity.
    """
    import torch

    mark = marks or Marks()
    device = pipe._execution_device
    dtype = pipe.text_encoder.dtype
    width, height, steps, guidance = common["width"], common["height"], common["steps"], common["guidance"]
    prompt_embeds = cached_embeds(pipe, common["prompt"], device, dtype)
    do_cfg = guidance > 1.0
    negative_embeds = cached_embeds(pipe, common["negative_prompt"], device, dtype) if do_cfg else None
    pair = torch.cat([prompt_embeds, negative_embeds], dim=0) if do_cfg else prompt_embeds
    mark("text")

    pipe.scheduler.set_timesteps(steps, device=device)
    transformer_dtype = pipe.transformer.dtype
    generator = torch.Generator(device=device).manual_seed(common["seed"])
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, height, width, 1, torch.float32, device, generator, None
    )
    padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
    guided_steps = steps if not do_cfg else max(1, round(steps * common["cutoff"]))
    step_sigmas = pipe.scheduler.sigmas[:steps].to(transformer_dtype)
    step_timesteps = {b: step_sigmas[:, None].expand(steps, b).contiguous() for b in ((2, 1) if do_cfg else (1,))}
    eager = getattr(pipe, "eager_transformer", None)
    denoiser = pipe.transformer if eager is None or (width, height) in pipe.compiled_sizes else eager
    for index, timestep_value in enumerate(pipe.scheduler.timesteps):
        sigma = pipe.scheduler.sigmas[index]
        guided = do_cfg and index < guided_steps
        batch = 2 if guided else 1
        context = pair if guided or not do_cfg else prompt_embeds
        packed = pack(latents) if pack else latents
        model_input = packed.to(transformer_dtype).repeat(batch, 1, 1, 1, 1)
        timestep = step_timesteps[batch][index]
        extra = step_hook(index, sigma, model_input, timestep, context) if step_hook else None
        velocity = denoiser(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=context,
            padding_mask=padding_mask,
            return_dict=False,
            **(extra or {}),
        )[0].float()
        if unpack:
            velocity = unpack(velocity)
        if guided:
            velocity = velocity[1:2] + guidance * (velocity[0:1] - velocity[1:2])
        latents = pipe.scheduler.step(velocity, timestep_value, latents, return_dict=False)[0]
    mark("denoise")

    mean, inv_std = vae_stats(pipe, latents)
    video = pipe.vae.decode((latents / inv_std + mean).to(pipe.vae.dtype), return_dict=False)[0]
    image = pipe.video_processor.postprocess_video(video, output_type="pil")[0][0]
    mark("decode")
    return image


def encode_outputs(image, extras: dict, seed: int, output_format: str, stem: str) -> dict:
    from PIL import Image

    fmt = str(output_format or "webp").lower()
    if fmt not in base.FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    pillow_format, content_type = base.FORMATS[fmt]
    buffer = io.BytesIO()
    image.save(buffer, format=pillow_format, **({"quality": 92} if fmt != "png" else {}))
    payload = buffer.getvalue()
    suffix = "jpg" if fmt == "jpeg" else fmt
    outputs = [{
        "filename": f"{stem}-{seed}.{suffix}", "content_type": content_type,
        "data": base64.b64encode(payload).decode("ascii"),
    }]
    for label, array in extras.items():
        side = io.BytesIO()
        Image.fromarray(array).save(side, format="PNG")
        outputs.append({
            "filename": f"{stem}-{seed}-{label}.png", "content_type": "image/png",
            "data": base64.b64encode(side.getvalue()).decode("ascii"),
        })
    return {"outputs": outputs, "seed": seed, "bytes": len(payload), "timings": dict(timings)}


def parse_sizes(text: str):
    return [tuple(int(v) for v in item.split("x")) for item in text.split(",") if item.strip()]


def run_warmup(name: str, pipe, request_for, sizes=None, save=None):
    """Run one synthetic request per served shape, then persist the inductor cache.

    `save=None` writes the cache only when no artifact exists yet.
    """
    import torch

    if save is None:
        save = not compile_artifact(name).is_file()
    started = time.perf_counter()
    for width, height in sizes or parse_sizes(WARMUP_SIZES):
        pipe.compiled_sizes.add((width, height))
        request_for(pipe, width, height)
        print(f"anime-{name}: warmed {width}x{height} ({time.perf_counter() - started:.0f}s)", flush=True)
    if not (save and COMPILE_ENABLED):
        return None
    saved = torch.compiler.save_cache_artifacts()
    if not saved:
        print(f"anime-{name}: torch produced no cache artifacts", flush=True)
        return None
    target = compile_artifact(name)
    for artifact in (target, Path(os.getenv("TMPDIR", "/tmp")) / target.name):
        try:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(saved[0])
            print(f"anime-{name}: wrote {artifact} ({len(saved[0]) >> 20} MiB)", flush=True)
            return artifact
        except OSError:
            continue
    return None


def warm_values(width: int, height: int, **extra) -> dict:
    values = {"prompt": "1girl, solo", "width": width, "height": height, "num_inference_steps": 10, "seed": 1}
    if CFG_CUTOFF < 1.0:
        values["cfg_cutoff"] = CFG_CUTOFF
    values.update(extra)
    return values


def serve(name: str, handler, warmup):
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--precompile":
        warmup(parse_sizes(",".join(sys.argv[2:])) or None)
        raise SystemExit(0)
    if base.runpod is None:
        raise SystemExit("runpod SDK is not installed")
    warmup()
    base.runpod.serverless.start({"handler": handler})
