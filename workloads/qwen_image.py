#!/usr/bin/env python3
"""Qwen Image 2.1 ("ra2") text-to-image and reference-edit workload.

Runs the same weights the local `ra2` lane serves — `qwen-image-2.1-Q4_K_M.gguf`
DiT, a Qwen3VL-8B GGUF text encoder with its mmproj vision tower, and the 2.1
VAE — but through `libstable-diffusion.so` loaded once per worker and kept
resident. Per-request cost is therefore sampling only: the alternative,
shelling out to `sd-cli` per job, re-reads ~11 GB of weights on every request
and roughly doubles the bill for a ~13 s job.

The job input is the OmniServe `/v1/images/generations` and
`/v1/images/edits` body verbatim, so the C gateway can relay an overflow
request unchanged, and the response is the same `{"created", "model",
"format", "data":[{"b64_json", "seed", "inference_time_ms"}]}` shape those
routes return — plus the `outputs` array and `timings` object this worker adds
for the app.nz cog surfaces. `image_base64` selects the reference edit; without
it the request is text-to-image.

The ctypes bindings mirror `stable-diffusion.h` at the commit the image pins
(`RA2_SD_COMMIT`). `verify_abi()` reads the defaults the library itself writes
into those structs back out through `sd_ctx_params_to_str` /
`sd_img_gen_params_to_str`, so a drifted header fails at image build time
(`python qwen_image.py --selftest`) and at first load instead of generating
garbage or crashing.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import runpod
except ModuleNotFoundError:  # Native local workers do not need the provider SDK.
    runpod = None

MODEL = "qwen-image-2.1"
MAX_SEED = 2**63 - 1
FORMATS = {"webp": ("WEBP", "image/webp"), "png": ("PNG", "image/png"), "jpeg": ("JPEG", "image/jpeg")}
ALLOWED_INPUTS = {
    "workload", "kind", "profile", "prompt", "negative_prompt", "width", "height",
    "size", "steps", "num_inference_steps", "guidance_scale", "seed", "output_format",
    "image_base64", "strength", "n",
}

DIT_REPO = os.getenv("RA2_DIT_REPO", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF")
TEXT_ENCODER_REPO = os.getenv("RA2_TEXT_ENCODER_REPO", "Qwen/Qwen3-VL-8B-Instruct-GGUF")
# The VAE comes from the same repo as the DiT, which is exactly where
# `deploy/qwen-ra2-prod.sh models` pulls it from: worker and local lane load
# identical bytes.
VAE_REPO = os.getenv("RA2_VAE_REPO", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF")
DIT_FILE = os.getenv("RA2_DIT_FILE", "qwen-image-2.1-Q4_K_M.gguf")
TEXT_ENCODER_FILE = os.getenv("RA2_TEXT_ENCODER_FILE", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
MMPROJ_FILE = os.getenv("RA2_MMPROJ_FILE", "mmproj-Qwen3VL-8B-Instruct-F16.gguf")
VAE_FILE = os.getenv("RA2_VAE_FILE", "vae/qwen_image_2.1_vae_bf16.safetensors")
VAE_URL = os.getenv("RA2_VAE_URL", "")

SD_LIB = os.getenv("RA2_SD_LIB", "/opt/stable-diffusion.cpp/build/bin/libstable-diffusion.so")
STEPS = int(os.getenv("RA2_STEPS", "20"))
GUIDANCE = float(os.getenv("RA2_GUIDANCE", "1.0"))
CACHE_MODE = os.getenv("RA2_CACHE_MODE", "easycache")
CACHE_THRESHOLD = float(os.getenv("RA2_CACHE_THRESHOLD", "0.05"))
CACHE_START = float(os.getenv("RA2_CACHE_START", "0.15"))
CACHE_END = float(os.getenv("RA2_CACHE_END", "0.95"))
SAMPLE_METHOD = os.getenv("RA2_SAMPLE_METHOD", "")
FLOW_SHIFT = float(os.getenv("RA2_FLOW_SHIFT", "3.0"))
REFERENCE_EDIT = os.getenv("RA2_REFERENCE_EDIT", "1").lower() not in {"0", "false", "no", "off"}
VAE_TILING = os.getenv("RA2_VAE_TILING", "1").lower() not in {"0", "false", "no", "off"}
VAE_TILE = int(os.getenv("RA2_VAE_TILE", "32"))
VAE_TILE_OVERLAP = float(os.getenv("RA2_VAE_TILE_OVERLAP", "0.5"))
FLASH_ATTN = os.getenv("RA2_FLASH_ATTN", "1").lower() not in {"0", "false", "no", "off"}
MMAP = os.getenv("RA2_MMAP", "1").lower() not in {"0", "false", "no", "off"}
EAGER_LOAD = os.getenv("RA2_EAGER_LOAD", "0").lower() in {"1", "true", "yes", "on"}
AUTO_FIT = os.getenv("RA2_AUTO_FIT", "0").lower() in {"1", "true", "yes", "on"}
BACKEND = os.getenv("RA2_BACKEND", "")
PARAMS_BACKEND = os.getenv("RA2_PARAMS_BACKEND", "")
MODEL_ARGS = os.getenv("RA2_MODEL_ARGS", "")
THREADS = int(os.getenv("RA2_THREADS", "-1"))
MAX_SIDE = int(os.getenv("RA2_MAX_SIDE", "2048"))
MAX_BATCH = int(os.getenv("RA2_MAX_BATCH", "1"))
WEBP_QUALITY = int(os.getenv("RA2_WEBP_QUALITY", "85"))
JPEG_QUALITY = int(os.getenv("RA2_JPEG_QUALITY", "92"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

_lock = threading.Lock()
_ctx = None
_lib = None
_state: dict[str, Any] = {}


# ---- stable-diffusion.cpp ABI --------------------------------------------

c_void_pp = ctypes.POINTER(ctypes.c_void_p)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)


class SdSlgParams(ctypes.Structure):
    _fields_ = [
        ("layers", ctypes.POINTER(ctypes.c_int)),
        ("layer_count", ctypes.c_size_t),
        ("layer_start", ctypes.c_float),
        ("layer_end", ctypes.c_float),
        ("scale", ctypes.c_float),
    ]


class SdGuidanceParams(ctypes.Structure):
    _fields_ = [
        ("txt_cfg", ctypes.c_float),
        ("img_cfg", ctypes.c_float),
        ("distilled_guidance", ctypes.c_float),
        ("slg", SdSlgParams),
    ]


class SdSampleParams(ctypes.Structure):
    _fields_ = [
        ("guidance", SdGuidanceParams),
        ("scheduler", ctypes.c_int),
        ("sample_method", ctypes.c_int),
        ("sample_steps", ctypes.c_int),
        ("eta", ctypes.c_float),
        ("shifted_timestep", ctypes.c_int),
        ("custom_sigmas", ctypes.POINTER(ctypes.c_float)),
        ("custom_sigmas_count", ctypes.c_int),
        ("flow_shift", ctypes.c_float),
        ("extra_sample_args", ctypes.c_char_p),
    ]


class SdImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channel", ctypes.c_uint32),
        ("data", c_uint8_p),
    ]


class SdPmParams(ctypes.Structure):
    _fields_ = [
        ("id_images", ctypes.POINTER(SdImage)),
        ("id_images_count", ctypes.c_int),
        ("id_embed_path", ctypes.c_char_p),
        ("style_strength", ctypes.c_float),
    ]


class SdPulidParams(ctypes.Structure):
    _fields_ = [("id_embedding_path", ctypes.c_char_p), ("id_weight", ctypes.c_float)]


class SdCacheParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("reuse_threshold", ctypes.c_float),
        ("start_percent", ctypes.c_float),
        ("end_percent", ctypes.c_float),
        ("error_decay_rate", ctypes.c_float),
        ("use_relative_threshold", ctypes.c_bool),
        ("reset_error_on_compute", ctypes.c_bool),
        ("Fn_compute_blocks", ctypes.c_int),
        ("Bn_compute_blocks", ctypes.c_int),
        ("residual_diff_threshold", ctypes.c_float),
        ("max_warmup_steps", ctypes.c_int),
        ("max_cached_steps", ctypes.c_int),
        ("max_continuous_cached_steps", ctypes.c_int),
        ("taylorseer_n_derivatives", ctypes.c_int),
        ("taylorseer_skip_interval", ctypes.c_int),
        ("scm_mask", ctypes.c_char_p),
        ("scm_policy_dynamic", ctypes.c_bool),
        ("spectrum_w", ctypes.c_float),
        ("spectrum_m", ctypes.c_int),
        ("spectrum_lam", ctypes.c_float),
        ("spectrum_window_size", ctypes.c_int),
        ("spectrum_flex_window", ctypes.c_float),
        ("spectrum_warmup_steps", ctypes.c_int),
        ("spectrum_stop_percent", ctypes.c_float),
    ]


class SdTilingParams(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_bool),
        ("temporal_tiling", ctypes.c_bool),
        ("tile_size_x", ctypes.c_int),
        ("tile_size_y", ctypes.c_int),
        ("target_overlap", ctypes.c_float),
        ("rel_size_x", ctypes.c_float),
        ("rel_size_y", ctypes.c_float),
        ("extra_tiling_args", ctypes.c_char_p),
    ]


class SdHiresParams(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_bool),
        ("upscaler", ctypes.c_int),
        ("model_path", ctypes.c_char_p),
        ("scale", ctypes.c_float),
        ("target_width", ctypes.c_int),
        ("target_height", ctypes.c_int),
        ("steps", ctypes.c_int),
        ("denoising_strength", ctypes.c_float),
        ("upscale_tile_size", ctypes.c_int),
        ("custom_sigmas", ctypes.POINTER(ctypes.c_float)),
        ("custom_sigmas_count", ctypes.c_int),
    ]


class SdImgGenParams(ctypes.Structure):
    _fields_ = [
        ("loras", ctypes.c_void_p),
        ("lora_count", ctypes.c_uint32),
        ("prompt", ctypes.c_char_p),
        ("negative_prompt", ctypes.c_char_p),
        ("clip_skip", ctypes.c_int),
        ("init_image", SdImage),
        ("ref_images", ctypes.POINTER(SdImage)),
        ("ref_images_count", ctypes.c_int),
        ("ref_image_args", ctypes.c_char_p),
        ("mask_image", SdImage),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("sample_params", SdSampleParams),
        ("strength", ctypes.c_float),
        ("seed", ctypes.c_int64),
        ("batch_count", ctypes.c_int),
        ("control_image", SdImage),
        ("control_strength", ctypes.c_float),
        ("ip_adapter_image", SdImage),
        ("ip_adapter_strength", ctypes.c_float),
        ("pm_params", SdPmParams),
        ("pulid_params", SdPulidParams),
        ("vae_tiling_params", SdTilingParams),
        ("cache", SdCacheParams),
        ("hires", SdHiresParams),
        ("qwen_image_layers", ctypes.c_int),
        ("circular_x", ctypes.c_bool),
        ("circular_y", ctypes.c_bool),
    ]


class SdEmbedding(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("path", ctypes.c_char_p)]


class SdCtxParams(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("clip_l_path", ctypes.c_char_p),
        ("clip_g_path", ctypes.c_char_p),
        ("clip_vision_path", ctypes.c_char_p),
        ("t5xxl_path", ctypes.c_char_p),
        ("llm_path", ctypes.c_char_p),
        ("llm_vision_path", ctypes.c_char_p),
        ("diffusion_model_path", ctypes.c_char_p),
        ("high_noise_diffusion_model_path", ctypes.c_char_p),
        ("uncond_diffusion_model_path", ctypes.c_char_p),
        ("embeddings_connectors_path", ctypes.c_char_p),
        ("vae_path", ctypes.c_char_p),
        ("audio_vae_path", ctypes.c_char_p),
        ("audio_encoder_path", ctypes.c_char_p),
        ("taesd_path", ctypes.c_char_p),
        ("control_net_path", ctypes.c_char_p),
        ("ip_adapter_path", ctypes.c_char_p),
        ("motion_module_path", ctypes.c_char_p),
        ("embeddings", ctypes.POINTER(SdEmbedding)),
        ("embedding_count", ctypes.c_uint32),
        ("photo_maker_path", ctypes.c_char_p),
        ("pulid_weights_path", ctypes.c_char_p),
        ("tensor_type_rules", ctypes.c_char_p),
        ("n_threads", ctypes.c_int),
        ("wtype", ctypes.c_int),
        ("rng_type", ctypes.c_int),
        ("sampler_rng_type", ctypes.c_int),
        ("prediction", ctypes.c_int),
        ("lora_apply_mode", ctypes.c_int),
        ("enable_mmap", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("diffusion_flash_attn", ctypes.c_bool),
        ("tae_preview_only", ctypes.c_bool),
        ("diffusion_conv_direct", ctypes.c_bool),
        ("vae_conv_direct", ctypes.c_bool),
        ("force_sdxl_vae_conv_scale", ctypes.c_bool),
        ("vae_format", ctypes.c_int),
        ("max_vram", ctypes.c_char_p),
        ("disable_prefetch", ctypes.c_bool),
        ("eager_load", ctypes.c_bool),
        ("backend", ctypes.c_char_p),
        ("params_backend", ctypes.c_char_p),
        ("split_mode", ctypes.c_char_p),
        ("auto_fit", ctypes.c_bool),
        ("rpc_servers", ctypes.c_char_p),
        ("model_args", ctypes.c_char_p),
        ("disable_segmented_compute", ctypes.c_bool),
        ("linear_scale", ctypes.c_float),
        ("attn_scale", ctypes.c_float),
        ("tokenizer", ctypes.c_char_p),
        ("sage_attn", ctypes.c_bool),
    ]


SD_CACHE_MODES = {"easycache": 1, "ucache": 2, "dbcache": 3, "taylorseer": 4, "cache-dit": 5, "spectrum": 6}


class Lib:
    """Thin owner of the dlopen'd library, its function table, and libc free."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise RuntimeError(f"libstable-diffusion not found at {path} (set RA2_SD_LIB)")
        self.lib = ctypes.CDLL(path)
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
        lib.sd_ctx_params_init.argtypes = [ctypes.POINTER(SdCtxParams)]
        lib.sd_ctx_params_init.restype = None
        lib.sd_img_gen_params_init.argtypes = [ctypes.POINTER(SdImgGenParams)]
        lib.sd_img_gen_params_init.restype = None
        lib.sd_ctx_params_to_str.argtypes = [ctypes.POINTER(SdCtxParams)]
        lib.sd_ctx_params_to_str.restype = ctypes.c_void_p
        lib.sd_img_gen_params_to_str.argtypes = [ctypes.POINTER(SdImgGenParams)]
        lib.sd_img_gen_params_to_str.restype = ctypes.c_void_p
        lib.new_sd_ctx.argtypes = [ctypes.POINTER(SdCtxParams)]
        lib.new_sd_ctx.restype = ctypes.c_void_p
        lib.free_sd_ctx.argtypes = [ctypes.c_void_p]
        lib.free_sd_ctx.restype = None
        lib.generate_image.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(SdImgGenParams),
            ctypes.POINTER(ctypes.POINTER(SdImage)), ctypes.POINTER(ctypes.c_int),
        ]
        lib.generate_image.restype = ctypes.c_bool
        lib.free_sd_images.argtypes = [ctypes.POINTER(SdImage), ctypes.c_int]
        lib.free_sd_images.restype = None
        lib.sd_version.argtypes = []
        lib.sd_version.restype = ctypes.c_char_p
        lib.sd_commit.argtypes = []
        lib.sd_commit.restype = ctypes.c_char_p
        lib.sd_get_num_physical_cores.argtypes = []
        lib.sd_get_num_physical_cores.restype = ctypes.c_int

    def _take_str(self, pointer: int) -> str:
        """Copy a malloc'd char* the library returned, then free it."""
        if not pointer:
            return ""
        try:
            return ctypes.string_at(pointer).decode("utf-8", "replace")
        finally:
            self.libc.free(ctypes.c_void_p(pointer))

    def ctx_params_to_str(self, params: SdCtxParams) -> str:
        return self._take_str(self.lib.sd_ctx_params_to_str(ctypes.byref(params)))

    def img_params_to_str(self, params: SdImgGenParams) -> str:
        return self._take_str(self.lib.sd_img_gen_params_to_str(ctypes.byref(params)))


def verify_abi(lib: Lib) -> dict[str, Any]:
    """Fail loudly when the bindings above do not match the linked header.

    Every claim below is a value the library itself wrote into the struct, so a
    field that moved shows up as a mismatch here rather than as a wrong image or
    a segfault under load.
    """
    ctx = SdCtxParams()
    lib.lib.sd_ctx_params_init(ctypes.byref(ctx))
    if ctx.n_threads < 1:
        raise RuntimeError("ABI mismatch: sd_ctx_params_init left n_threads unset")

    img = SdImgGenParams()
    lib.lib.sd_img_gen_params_init(ctypes.byref(img))
    expected = {
        "width": (img.width, 512),
        "height": (img.height, 512),
        "sample_steps": (img.sample_params.sample_steps, 20),
        "clip_skip": (img.clip_skip, -1),
        "seed": (img.seed, -1),
        "batch_count": (img.batch_count, 1),
        "qwen_image_layers": (img.qwen_image_layers, 3),
        "ref_images_count": (img.ref_images_count, 0),
        "cache_mode": (img.cache.mode, 0),
        "hires_enabled": (bool(img.hires.enabled), False),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise RuntimeError(f"ABI mismatch: {name} is {got!r}, expected {want!r}")
    if abs(img.strength - 0.75) > 1e-6 or abs(img.vae_tiling_params.target_overlap - 0.5) > 1e-6:
        raise RuntimeError("ABI mismatch: sd_img_gen_params defaults do not match")

    marker = b"/abi/probe/qwen-image.gguf"
    ctx.diffusion_model_path = marker
    ctx.backend = b"te=cpu"
    ctx.params_backend = b"*=cpu"
    ctx.eager_load = True
    ctx.flash_attn = True
    ctx.diffusion_flash_attn = True
    ctx.enable_mmap = True
    ctx.auto_fit = False
    text = lib.ctx_params_to_str(ctx)
    for fragment in (
        "diffusion_model_path: /abi/probe/qwen-image.gguf",
        "backend: te=cpu",
        "params_backend: *=cpu",
        "eager_load: true",
        "flash_attn: true",
        "diffusion_flash_attn: true",
        "auto_fit: false",
    ):
        if fragment not in text:
            raise RuntimeError(f"ABI mismatch: {fragment!r} missing from sd_ctx_params_to_str")
    probe = SdImgGenParams()
    lib.lib.sd_img_gen_params_init(ctypes.byref(probe))
    probe.cache.mode = SD_CACHE_MODES["easycache"]
    probe.cache.reuse_threshold = CACHE_THRESHOLD
    probe.cache.start_percent = CACHE_START
    probe.cache.end_percent = CACHE_END
    cache_text = lib.img_params_to_str(probe)
    if "cache: easycache" not in cache_text or f"threshold={CACHE_THRESHOLD:.3f}" not in cache_text:
        raise RuntimeError("ABI mismatch: sd_cache_params_t layout does not round-trip")
    if not lib.lib.sd_version():
        raise RuntimeError("ABI mismatch: sd_version() returned nothing")
    return {
        "version": lib.lib.sd_version().decode("utf-8", "replace"),
        "commit": (lib.lib.sd_commit() or b"").decode("utf-8", "replace"),
        "cache_probe": cache_text.splitlines()[-1],
    }


# ---- weights --------------------------------------------------------------

def _writable_root() -> Path:
    """Models root that survives worker restarts and exists on this host.

    A Serverless worker mounts the shared network volume at `/runpod-volume`; an
    app.nz pod mounts its weights volume at `/workspace`. Whichever of those is
    really present wins, so one image serves both tiers without the caller
    having to know which one it is on.
    """
    candidates = []
    override = os.getenv("RA2_MODELS_DIR", "").strip()
    if override:
        candidates.append(Path(override))
    candidates.extend([Path("/runpod-volume/omniserve/qwen-image-2.1"), Path("/workspace/models/qwen-image-2.1")])
    for candidate in candidates:
        if candidate.is_dir() or candidate.parent.is_dir():
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
    fallback = Path(os.getenv("TMPDIR", "/tmp")) / "omniserve/qwen-image-2.1"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _download(repo: str, filename: str, root: Path) -> Path:
    target = root / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    if not repo:
        raise RuntimeError(
            f"{filename} is missing from {root} and no source repo is configured; "
            f"set RA2_VAE_REPO or RA2_VAE_URL (and pre-seed the network volume) to fetch it"
        )
    from huggingface_hub import hf_hub_download

    print(f"[qwen-image] downloading {repo}/{filename} into {root}", flush=True)
    started = time.monotonic()
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(root), token=HF_TOKEN)
    print(f"[qwen-image] {filename} ready in {time.monotonic() - started:.1f}s", flush=True)
    return Path(path)


def _download_url(url: str, filename: str, root: Path) -> Path:
    target = root / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    import urllib.request

    print(f"[qwen-image] downloading {url} into {root}", flush=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
        while True:
            chunk = response.read(1 << 22)
            if not chunk:
                break
            handle.write(chunk)
    partial.rename(target)
    return target


def ensure_models() -> dict[str, Path]:
    """Fetch the pinned weight set once per volume, then reuse it forever."""
    global _state
    if _state.get("models"):
        return _state["models"]
    root = _writable_root()
    # HF_HOME points at the Serverless network volume; on a pod that path does
    # not exist, and letting the hub create it there would fail the download the
    # pod needs. Fall back to the volume that does exist.
    home = Path(os.getenv("HF_HOME", "."))
    if not (home.is_dir() or home.parent.is_dir()):
        os.environ["HF_HOME"] = str(root / "huggingface")
    models = {
        "dit": _download(DIT_REPO, DIT_FILE, root),
        "text_encoder": _download(TEXT_ENCODER_REPO, TEXT_ENCODER_FILE, root),
        "mmproj": _download(TEXT_ENCODER_REPO, MMPROJ_FILE, root),
        "vae": _download_url(VAE_URL, VAE_FILE, root) if VAE_URL else _download(VAE_REPO, VAE_FILE, root),
    }
    _state["models"] = models
    _state["models_dir"] = root
    return models


# ---- engine ---------------------------------------------------------------

def _lib_instance() -> Lib:
    global _lib
    if _lib is None:
        _lib = Lib(SD_LIB)
        _state["abi"] = verify_abi(_lib)
    return _lib


def load() -> ctypes.c_void_p:
    """Create the resident context. Idempotent; the worker keeps it warm."""
    global _ctx
    if _ctx is not None:
        return _ctx
    lib = _lib_instance()
    models = ensure_models()
    params = SdCtxParams()
    lib.lib.sd_ctx_params_init(ctypes.byref(params))
    refs: list[bytes] = []

    def keep(value: str) -> ctypes.c_char_p:
        raw = str(value).encode()
        refs.append(raw)
        return ctypes.c_char_p(raw)

    params.diffusion_model_path = keep(models["dit"])
    params.llm_path = keep(models["text_encoder"])
    params.llm_vision_path = keep(models["mmproj"])
    params.vae_path = keep(models["vae"])
    params.n_threads = THREADS
    params.flash_attn = FLASH_ATTN
    params.diffusion_flash_attn = FLASH_ATTN
    params.enable_mmap = MMAP
    params.eager_load = EAGER_LOAD
    params.auto_fit = AUTO_FIT
    if BACKEND:
        params.backend = keep(BACKEND)
    if PARAMS_BACKEND:
        params.params_backend = keep(PARAMS_BACKEND)
    if MODEL_ARGS:
        params.model_args = keep(MODEL_ARGS)

    started = time.monotonic()
    ctx = lib.lib.new_sd_ctx(ctypes.byref(params))
    if not ctx:
        raise RuntimeError("stable-diffusion.cpp could not load the Qwen Image 2.1 weight set")
    _state["load_ms"] = int((time.monotonic() - started) * 1000)
    _state["refs"] = refs
    _ctx = ctx
    print(
        f"[qwen-image] loaded {MODEL} in {_state['load_ms']} ms "
        f"(sd.cpp {_state.get('abi', {}).get('commit', '?')}, models {_state.get('models_dir')})",
        flush=True,
    )
    return ctx


def release() -> None:
    """Drop the context so a profile switch or shutdown frees VRAM."""
    global _ctx
    lib = _lib
    if _ctx is not None and lib is not None:
        lib.lib.free_sd_ctx(_ctx)
    _ctx = None
    _state.pop("refs", None)


# ---- request mapping ------------------------------------------------------

def _inputs(job: dict) -> dict:
    values = job.get("input") or {}
    return values.get("input", values) if isinstance(values, dict) else {}


def _dimension(value, fallback: int) -> int:
    number = fallback if value is None else int(value)
    if number % 64 or not 64 <= number <= MAX_SIDE:
        raise ValueError(f"width and height must be 64..{MAX_SIDE} in 64-pixel increments")
    return number


def _seed(value) -> int:
    if value is None:
        return int.from_bytes(os.urandom(8), "big") % (MAX_SEED + 1)
    seed = int(value)
    if seed == 0:
        return 0
    return seed % (MAX_SEED + 1)


def _size(values: dict) -> tuple[int | None, int | None]:
    raw = values.get("size")
    if not isinstance(raw, str) or "x" not in raw:
        return None, None
    width, _, height = raw.partition("x")
    try:
        return int(width), int(height)
    except ValueError:
        raise ValueError("size must be WIDTHxHEIGHT in 64-pixel increments") from None


def _decode_image(image_base64: str):
    from PIL import Image

    try:
        raw = base64.b64decode(image_base64, validate=False)
    except Exception as error:  # noqa: BLE001 - surfaced as a 400 by the caller
        raise ValueError(f"image_base64 is not valid base64: {error}") from None
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise ValueError("image_base64 must be a PNG or JPEG") from None
    return image


def _source_image(image) -> tuple[SdImage, Any]:
    """sd_image_t over the decoded RGB bytes; the buffer must outlive the call."""
    payload = image.tobytes()
    buffer = ctypes.create_string_buffer(payload, len(payload))
    struct = SdImage(
        width=image.width, height=image.height, channel=3,
        data=ctypes.cast(buffer, c_uint8_p),
    )
    return struct, buffer


def build_params(values: dict, models: dict[str, Path]) -> tuple[SdImgGenParams, list]:
    """Map an OmniServe image body onto sd_img_gen_params_t.

    Defaults and the reference-edit flag match the local `ra2` lane
    (`src/backend_sd.c`), so an overflow request and a local request for the
    same body produce the same image.
    """
    lib = _lib_instance()
    params = SdImgGenParams()
    lib.lib.sd_img_gen_params_init(ctypes.byref(params))
    keep: list = []

    def own(value) -> ctypes.c_char_p:
        raw = value if isinstance(value, bytes) else str(value).encode()
        keep.append(raw)
        return ctypes.c_char_p(raw)

    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt required")
    params.prompt = own(prompt)
    negative = values.get("negative_prompt")
    if negative is not None and not isinstance(negative, str):
        raise ValueError("negative_prompt must be a string")
    params.negative_prompt = own(negative or "")

    width, height = _size(values)
    width = _dimension(values.get("width", width), 1024)
    height = _dimension(values.get("height", height), 1024)
    params.width = width
    params.height = height

    steps = values.get("steps", values.get("num_inference_steps"))
    params.sample_params.sample_steps = STEPS if steps is None else int(steps)
    if not 1 <= params.sample_params.sample_steps <= 100:
        raise ValueError("steps must be between 1 and 100")

    guidance = values.get("guidance_scale")
    txt_cfg = GUIDANCE if guidance is None else float(guidance)
    if not 0.0 <= txt_cfg <= 30.0:
        raise ValueError("guidance_scale must be between 0 and 30")
    params.sample_params.guidance.txt_cfg = txt_cfg
    if SAMPLE_METHOD:
        from_sd = lib.lib
        # The CLI accepts names; the library only takes the enum. Bind the
        # lookup so a typo fails here instead of silently sampling default.
        from_sd.str_to_sample_method.argtypes = [ctypes.c_char_p]
        from_sd.str_to_sample_method.restype = ctypes.c_int
        method = from_sd.str_to_sample_method(SAMPLE_METHOD.encode())
        if method < 0:
            raise ValueError(f"RA2_SAMPLE_METHOD {SAMPLE_METHOD!r} is not a stable-diffusion.cpp sampler")
        params.sample_params.sample_method = method

    params.seed = _seed(values.get("seed"))
    batch = values.get("n")
    params.batch_count = 1 if batch is None else int(batch)
    if not 1 <= params.batch_count <= MAX_BATCH:
        raise ValueError(f"n must be between 1 and {MAX_BATCH}")

    source = values.get("image_base64")
    if source is not None:
        if not isinstance(source, str) or not source:
            raise ValueError("image_base64 must be a base64 PNG or JPEG")
        image = _decode_image(source)
        struct, buffer = _source_image(image)
        keep.append(buffer)
        keep.append(struct)
        if REFERENCE_EDIT:
            refs = (SdImage * 1)(struct)
            keep.append(refs)
            params.ref_images = ctypes.cast(refs, ctypes.POINTER(SdImage))
            params.ref_images_count = 1
            params.sample_params.flow_shift = FLOW_SHIFT
        else:
            params.init_image = struct
            strength = values.get("strength")
            params.strength = 0.6 if strength is None else float(strength)
            if not 0.0 < params.strength <= 1.0:
                raise ValueError("strength must be in (0, 1]")

    if VAE_TILING:
        params.vae_tiling_params.enabled = True
        params.vae_tiling_params.tile_size_x = VAE_TILE
        params.vae_tiling_params.tile_size_y = VAE_TILE
        params.vae_tiling_params.target_overlap = VAE_TILE_OVERLAP

    mode = CACHE_MODE.strip().lower()
    if mode and mode not in {"off", "none", "disabled"}:
        if mode not in SD_CACHE_MODES:
            raise ValueError(f"RA2_CACHE_MODE {mode!r} is not one of {', '.join(sorted(SD_CACHE_MODES))}")
        params.cache.mode = SD_CACHE_MODES[mode]
        params.cache.start_percent = CACHE_START
        params.cache.end_percent = CACHE_END
        if mode == "easycache":
            params.cache.reuse_threshold = CACHE_THRESHOLD
        elif mode == "taylorseer":
            params.cache.taylorseer_skip_interval = int(os.getenv("RA2_TAYLORSEER_SKIP", "2"))
        elif mode == "spectrum":
            params.cache.spectrum_warmup_steps = int(os.getenv("RA2_SPECTRUM_WARMUP", "4"))
            params.cache.spectrum_stop_percent = float(os.getenv("RA2_SPECTRUM_STOP", "0.9"))
    else:
        params.cache.mode = 0

    return params, keep


def encode(images, count: int, output_format: str) -> list[dict]:
    from PIL import Image

    pillow_format, content_type = FORMATS[output_format]
    suffix = "jpg" if output_format == "jpeg" else output_format
    out: list[dict] = []
    for index in range(count):
        image = images[index]
        raw = ctypes.string_at(image.data, image.width * image.height * image.channel)
        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(image.channel, "RGB")
        if image.channel == 4:
            # Qwen Image 2.1 decodes RGBA; flatten the alpha the way the local
            # lane does before encoding, so the payload stays RGB.
            picture = Image.frombytes("RGBA", (image.width, image.height), raw).convert("RGB")
        else:
            picture = Image.frombytes(mode, (image.width, image.height), raw)
        buffer = io.BytesIO()
        options = {"quality": WEBP_QUALITY} if output_format == "webp" else (
            {"quality": JPEG_QUALITY} if output_format == "jpeg" else {})
        picture.save(buffer, format=pillow_format, **options)
        payload = buffer.getvalue()
        out.append({
            "filename": f"ra2-{_state.get('seed', 0)}-{index}.{suffix}",
            "content_type": content_type,
            "data": base64.b64encode(payload).decode("ascii"),
        })
    return out


def generate(values: dict) -> dict:
    """Run one request. Returns the OmniServe image response plus timings."""
    ctx = load()
    lib = _lib_instance()
    models = _state["models"]
    output_format = str(values.get("output_format", "webp")).lower()
    if output_format == "jpg":
        output_format = "jpeg"
    if output_format not in FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    params, keep = build_params(values, models)
    _state["seed"] = params.seed

    images = ctypes.POINTER(SdImage)()
    count = ctypes.c_int(0)
    started = time.monotonic()
    ok = lib.lib.generate_image(ctx, ctypes.byref(params), ctypes.byref(images), ctypes.byref(count))
    sample_ms = int((time.monotonic() - started) * 1000)
    if not ok or not images or count.value < 1:
        if images:
            lib.lib.free_sd_images(images, count.value)
        raise RuntimeError("stable-diffusion.cpp failed to generate an image")
    try:
        started = time.monotonic()
        encoded = encode(images, count.value, output_format)
        encode_ms = int((time.monotonic() - started) * 1000)
    finally:
        lib.lib.free_sd_images(images, count.value)
        keep.clear()

    created = int(time.time())
    pillow_format = output_format.upper()
    data = [{
        "b64_json": item["data"],
        "seed": params.seed + index,
        "inference_time_ms": sample_ms,
        "format": output_format,
    } for index, item in enumerate(encoded)]
    if params.cache.mode == SD_CACHE_MODES["easycache"]:
        for item in data:
            item["denoiser_cache"] = {"requested": "easycache", "approximate": True, "threshold": CACHE_THRESHOLD}
    return {
        "created": created,
        "model": MODEL,
        "format": output_format,
        "data": data,
        # app.nz's cog surfaces read the first media artifact out of `outputs`
        # and inline it as a data URL; the gateway relays the fields above.
        "outputs": encoded,
        "seed": params.seed,
        "bytes": sum(len(base64.b64decode(item["data"])) for item in encoded),
        "timings": {
            "load_ms": _state.get("load_ms"),
            "sample_ms": sample_ms,
            "encode_ms": encode_ms,
            "format": pillow_format,
            "cache_mode": CACHE_MODE,
            "reference_edit": bool(params.ref_images_count),
        },
    }


def handler(job: dict, _pipe=None) -> dict:
    values = _inputs(job)
    if not isinstance(values, dict):
        raise ValueError("input must be an object")
    # The manifest runtime stamps its own envelope keys (`_omniserve_profile`,
    # `_omniserve_retry`) into the input it hands the plugin; they are plumbing,
    # not caller input, so they are dropped before the contract is checked.
    values = {key: value for key, value in values.items() if not key.startswith("_")}
    unknown = sorted(set(values) - ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"unknown Qwen Image inputs: {', '.join(unknown)}")
    # One worker, one GPU: serialise so two requests never fight for the same
    # context and the billed seconds stay attributable to one job.
    with _lock:
        return generate(values)


def selftest(*, fetch: bool = True) -> int:
    """Prove the bindings match the linked library, and optionally the weights.

    `fetch=False` is the image-build gate: it loads the library and checks the
    struct round-trip without touching the network or a GPU.
    """
    lib = _lib_instance()
    print(json.dumps(verify_abi(lib), indent=2))
    if not fetch:
        print(f"abi ok (models root would be {_writable_root()})")
        return 0
    models = ensure_models()
    for name, path in models.items():
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()[:1 << 20]).hexdigest()[:16]
        print(f"{name}: {path} {size / 1e6:.1f} MB head={digest}")
    print("abi and weights ok")
    return 0


if __name__ == "__main__":
    if "--abi-check" in sys.argv:
        raise SystemExit(selftest(fetch=False))
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if runpod is None:
        raise SystemExit("runpod SDK is not installed")
    runpod.serverless.start({"handler": handler})
