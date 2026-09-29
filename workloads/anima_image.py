#!/usr/bin/env python3
"""Anima 2.9B text-to-image workload for the OmniServe manifest runtime.

Loads the pinned reference weights the same way the Cog worker does — meta-device
instantiation, `assign=True` checkpoint adoption, and no base-transformer read —
then denoises with one batched classifier-free-guidance forward per step and
replays a prebuilt inductor cache so the first request never compiles.

The weights are non-commercial: the handler fails closed until the operator sets
`APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED=1` after obtaining a CircleStone license.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import os
import random
import re
import sys
from pathlib import Path

try:
    import runpod
except ModuleNotFoundError:  # Native local workers do not need the provider SDK.
    runpod = None

MODEL_ID = os.getenv("ANIMA_MODEL_ID", "Gazingstars123/Anima-2.9B")
MODEL_FILE = os.getenv("ANIMA_MODEL_FILE", "Anima-2.9B-preview-v1.safetensors")
MODEL_REVISION = os.getenv("ANIMA_MODEL_REVISION", "fb00923d6a68424b731048d4c65da61eed1a6cc2")
BASE_MODEL_ID = os.getenv("ANIMA_BASE_MODEL_ID", "CalamitousFelicitousness/Anima-sdnext-diffusers")
BASE_MODEL_REVISION = os.getenv("ANIMA_BASE_MODEL_REVISION", "587e3941c37ace6234f9c0daa5c908408652870a")
SPACE_ID = "akhaliq/Anima-2.9B"
SPACE_REVISION = "88543bfa289482b451631a565f54653b22b1c1cb"
SPACE_SOURCE_HASHES = {
    "pipeline.py": "c2a963876781988ed1343788c9a3e529b076d569dd909bcc12b534271686b83a",
    "modeling_llm_adapter.py": "9fc725376c9373c0db9da3efff123996e5cbfaae7e12abb136b1ef99c31c2aff",
}
COMPILE_ARTIFACT_DIR = Path(os.getenv("ANIMA_COMPILE_ARTIFACT_DIR", "/runpod-volume/anima-compile-cache"))
COMPILE_ENABLED = os.getenv("ANIMA_COMPILE", "1").lower() not in {"0", "false", "off"}
COMPILE_MODE = os.getenv("ANIMA_COMPILE_MODE", "default")
MAX_SEED = 2**31 - 1
FORMATS = {"webp": ("WEBP", "image/webp"), "png": ("PNG", "image/png"), "jpeg": ("JPEG", "image/jpeg")}
ALLOWED_INPUTS = {
    "workload", "kind", "profile", "prompt", "negative_prompt", "width", "height",
    "num_inference_steps", "guidance_scale", "seed", "output_format",
    "accel", "fused", "teleport", "teleport_fraction",
}

BLOCK_MAP = {
    "self_attn.q_proj": "attn1.to_q",
    "self_attn.k_proj": "attn1.to_k",
    "self_attn.v_proj": "attn1.to_v",
    "self_attn.output_proj": "attn1.to_out.0",
    "self_attn.q_norm": "attn1.norm_q",
    "self_attn.k_norm": "attn1.norm_k",
    "cross_attn.q_proj": "attn2.to_q",
    "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.v_proj": "attn2.to_v",
    "cross_attn.output_proj": "attn2.to_out.0",
    "cross_attn.q_norm": "attn2.norm_q",
    "cross_attn.k_norm": "attn2.norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
}
TOP_MAP = {
    "net.x_embedder.proj.1.weight": "patch_embed.proj.weight",
    "net.t_embedding_norm.weight": "time_embed.norm.weight",
    "net.final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
    "net.final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
    "net.final_layer.linear.weight": "proj_out.weight",
}
_pipeline = None
_runner = None
_teleport_pipeline = None
_cache = None
_embed_cache = {}


def _anima_cog():
    """Optional acceleration package; the dense path must work without it."""
    try:
        import anima_cog

        return anima_cog
    except ModuleNotFoundError:
        return None


def commercial_license_accepted() -> bool:
    return os.getenv("APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED", "").strip() == "1"


def _inputs(job: dict) -> dict:
    values = job.get("input") or {}
    return values.get("input", values) if isinstance(values, dict) else {}


def _dimension(value, default: int) -> int:
    if value is None:
        return default
    number = int(value)
    if number % 16 != 0:
        raise ValueError("width and height must be divisible by 16")
    if not 512 <= number <= 1536:
        raise ValueError("width and height must be between 512 and 1536")
    return number


def _seed(value) -> int:
    if value is None or int(value) < 0:
        return random.SystemRandom().randint(0, MAX_SEED)
    return int(value) % (MAX_SEED + 1)


def remap_checkpoint(state_dict: dict) -> dict:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("net.blocks."):
            block_n, module_key = key[len("net.blocks."):].split(".", 1)
            suffix = ""
            for candidate in (".weight", ".bias"):
                if module_key.endswith(candidate):
                    module_key, suffix = module_key[: -len(candidate)], candidate
                    break
            mapped = BLOCK_MAP.get(module_key)
            if mapped:
                remapped[f"transformer_blocks.{block_n}.{mapped}{suffix}"] = value
            continue
        mapped = TOP_MAP.get(key)
        if mapped:
            remapped[mapped] = value
            continue
        match = re.match(r"net\.t_embedder\.\d+\.(linear_[12]\.weight)", key)
        if match:
            remapped[f"time_embed.t_embedder.{match.group(1)}"] = value
    return remapped


def _pipeline_class():
    from huggingface_hub import hf_hub_download

    source_dir = None
    for filename, expected in SPACE_SOURCE_HASHES.items():
        path = Path(hf_hub_download(SPACE_ID, filename, repo_type="space", revision=SPACE_REVISION))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"pinned Anima Space source hash mismatch for {filename}")
        source_dir = path.parent
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    importlib.import_module("modeling_llm_adapter")
    return importlib.import_module("pipeline").AnimaTextToImagePipeline


def _compile_artifact(torch) -> Path:
    major, minor = torch.cuda.get_device_capability()
    return COMPILE_ARTIFACT_DIR / f"anima-sm{major}{minor}-torch{torch.__version__.split('+')[0]}.bin"


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from diffusers import CosmosTransformer3DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipeline_class = _pipeline_class()
    config = dict(
        CosmosTransformer3DModel.load_config(
            BASE_MODEL_ID, subfolder="transformer", revision=BASE_MODEL_REVISION
        )
    )
    config["num_layers"] = 40
    with torch.device("meta"):
        transformer = CosmosTransformer3DModel.from_config(config)

    checkpoint = os.getenv("ANIMA_MODEL_PATH", "").strip() or hf_hub_download(
        MODEL_ID, MODEL_FILE, revision=MODEL_REVISION
    )
    state_dict = remap_checkpoint(load_file(checkpoint, device="cuda"))
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)
    still_meta = [name for name, value in transformer.state_dict().items() if value.is_meta]
    if missing or unexpected or still_meta:
        raise RuntimeError(
            f"Anima checkpoint remap incomplete: missing={missing[:8]} "
            f"unexpected={unexpected[:8]} uninitialized={still_meta[:8]}"
        )

    pipe = pipeline_class.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        transformer=None,
    )
    pipe.register_modules(transformer=transformer.to(dtype=torch.bfloat16))
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    accel = _anima_cog()
    if accel is not None and os.getenv("ANIMA_FUSED", "1").lower() not in {"0", "false", "off"}:
        patched = accel.patch.apply_fused_blocks(pipe.transformer)
        print(f"anima: fused adaLN kernels on {patched}/{len(pipe.transformer.transformer_blocks)} blocks", flush=True)
    if COMPILE_ENABLED:
        artifact = _compile_artifact(torch)
        loader = getattr(torch.compiler, "load_cache_artifacts", None)
        if loader is not None and artifact.is_file():
            try:
                loader(artifact.read_bytes())
            except Exception as error:  # torch build mismatch must not block serving
                print(f"anima: compile cache replay failed ({error})", flush=True)
        pipe.transformer = torch.compile(pipe.transformer, mode=COMPILE_MODE, fullgraph=False, dynamic=False)

    _pipeline = pipe
    return pipe


_EMBED_CACHE_LIMIT = 32


def _cached_embeds(pipe, text: str, device, dtype):
    key = (text, str(dtype))
    embeds = _embed_cache.get(key)
    if embeds is None:
        embeds = pipe._encode_prompt([text], device, dtype, 512)
        if len(_embed_cache) >= _EMBED_CACHE_LIMIT:
            _embed_cache.pop(next(iter(_embed_cache)))
        _embed_cache[key] = embeds
    return embeds


def generate(pipe, values: dict):
    import torch

    prompt = str(values.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")
    negative_prompt = str(values.get("negative_prompt", "")).strip()
    width = _dimension(values.get("width"), 832)
    height = _dimension(values.get("height"), 1216)
    steps = int(values.get("num_inference_steps", 28))
    if not 10 <= steps <= 50:
        raise ValueError("num_inference_steps must be between 10 and 50")
    guidance = float(values.get("guidance_scale", 4.0))
    if not 1.0 <= guidance <= 8.0:
        raise ValueError("guidance_scale must be between 1.0 and 8.0")
    seed = _seed(values.get("seed"))
    teleport = bool(values.get("teleport", False)) or str(values.get("accel", "")).lower() == "teleport"
    teleport_fraction = float(values.get("teleport_fraction", 0.3))
    if not 0.05 <= teleport_fraction <= 0.9:
        raise ValueError("teleport_fraction must be between 0.05 and 0.9")
    meta: dict = {"accel": "dense"}

    if teleport:
        accel_pkg = _anima_cog()
        if accel_pkg is None:
            raise ValueError("teleport requires the anima-cog acceleration package")
        global _runner, _teleport_pipeline, _cache
        from anima_cog.runner import AnimaRunner
        from anima_cog.teleport import AnimaLatentCache, ConfidenceGate, TeleportPipeline

        if _teleport_pipeline is None:
            _cache = AnimaLatentCache(
                os.getenv("ANIMA_TELEPORT_CACHE", "/runpod-volume/anima-teleport"),
                resolution=(width, height),
            )
            _runner = AnimaRunner(pipe)
            _teleport_pipeline = TeleportPipeline(_runner, _cache, confidence=ConfidenceGate())
        result = _teleport_pipeline.generate(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
            teleport_fraction=teleport_fraction,
        )
        meta = {
            "accel": "teleport",
            "method": result["method"],
            "cache_hits": result["cache_hits"],
            "bigram_hits": result["bigram_hits"],
            "total_units": result["total_units"],
            "start_step": result["start_step"],
            "steps_saved": result.get("steps_saved", 0),
            "text_similarity": result.get("text_similarity"),
            "units": result["units"],
            "cache_stats": _cache.stats(),
        }
        return result["image"], seed, meta

    device = pipe._execution_device
    dtype = pipe.text_encoder.dtype
    with torch.inference_mode():
        prompt_embeds = _cached_embeds(pipe, prompt, device, dtype)
        do_cfg = guidance > 1.0
        if do_cfg:
            negative_embeds = _cached_embeds(pipe, negative_prompt, device, dtype)
            encoder_hidden_states = torch.cat([prompt_embeds, negative_embeds], dim=0)
        else:
            encoder_hidden_states = prompt_embeds

        pipe.scheduler.set_timesteps(steps, device=device)
        transformer_dtype = pipe.transformer.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels, height, width, 1, torch.float32, device, generator, None
        )
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        for index, timestep_value in enumerate(pipe.scheduler.timesteps):
            sigma = pipe.scheduler.sigmas[index]
            model_input = latents.to(transformer_dtype)
            batch = 2 if do_cfg else 1
            velocity = pipe.transformer(
                hidden_states=model_input.repeat(batch, 1, 1, 1, 1),
                timestep=sigma.expand(batch).to(transformer_dtype),
                encoder_hidden_states=encoder_hidden_states,
                padding_mask=padding_mask,
                return_dict=False,
            )[0].float()
            if do_cfg:
                velocity = velocity[1:2] + guidance * (velocity[0:1] - velocity[1:2])
            latents = pipe.scheduler.step(velocity, timestep_value, latents, return_dict=False)[0]

        mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / inv_std + mean
        video = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
        image = pipe.video_processor.postprocess_video(video, output_type="pil")[0][0]
    return image, seed, meta

def handler(job, pipe=None):
    if not commercial_license_accepted():
        raise RuntimeError("Anima hosting requires APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED=1")
    values = _inputs(job)
    unknown = sorted(set(values) - ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"unknown Anima inputs: {', '.join(unknown)}")
    output_format = str(values.get("output_format", "webp")).lower()
    if output_format not in FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")

    image, seed, meta = generate(pipe or load_pipeline(), values)
    pillow_format, content_type = FORMATS[output_format]
    buffer = io.BytesIO()
    image.save(buffer, format=pillow_format, **({"quality": 92} if output_format != "png" else {}))
    payload = buffer.getvalue()
    suffix = "jpg" if output_format == "jpeg" else output_format
    return {
        "outputs": [{
            "filename": f"anima-{seed}.{suffix}",
            "content_type": content_type,
            "data": base64.b64encode(payload).decode("ascii"),
        }],
        "seed": seed,
        "bytes": len(payload),
        "accel": meta.get("accel", "dense"),
        "teleport": meta if meta.get("accel") == "teleport" else None,
    }


if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod SDK is not installed")
    runpod.serverless.start({"handler": handler})
