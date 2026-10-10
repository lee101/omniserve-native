#!/usr/bin/env python3
"""Multitenant Qwen worker: Qwen Image 2.1 ("ra2") and Qwen Image Edit 2511 in one process.

Routed by input `task`: `ra2` (default; text-to-image, or reference edit with
`image_base64`) and `edit` (Qwen Image Edit 2511, `image_base64` required). The two
pipelines share no weights (ra2: Qwen3-VL-8B encoder + 2.1 VAE; 2511: Qwen2.5-VL-7B
encoder + Qwen Image VAE), so each task owns one resident stable-diffusion.cpp
context. Both stay resident when VRAM allows; otherwise the least recently used
context is freed before the other loads (a reload from page cache costs seconds,
streaming weights per step costs tens of seconds). Weights are baked into the image.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads import qwen_image as qi  # noqa: E402

try:
    import runpod
except ModuleNotFoundError:
    runpod = None

GIB = 1 << 30
ROOT = Path(os.getenv("MT_MODELS_DIR", "/models"))
RESERVE_GIB = float(os.getenv("MT_RESERVE_GIB", "1.0"))
PRELOAD = os.getenv("MT_PRELOAD", "auto").strip().lower()
ALLOWED = qi.ALLOWED_INPUTS | {"task", "probe_path", "probe_gb"}


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


def _task(name: str, **spec) -> dict:
    up = name.upper()
    spec["steps"] = int(os.getenv(f"MT_{up}_STEPS", spec["steps"]))
    spec["guidance"] = float(os.getenv(f"MT_{up}_GUIDANCE", spec["guidance"]))
    spec["cache_mode"] = os.getenv(f"MT_{up}_CACHE_MODE", spec["cache_mode"]).strip().lower()
    spec["cache_threshold"] = float(os.getenv(f"MT_{up}_CACHE_THRESHOLD", spec["cache_threshold"]))
    spec["vae_tiling"] = _flag(f"MT_{up}_VAE_TILING", spec["vae_tiling"])
    spec["compute_gib"] = float(os.getenv(f"MT_{up}_COMPUTE_GIB", spec["compute_gib"]))
    spec["files"] = {k: ROOT / name / v for k, v in spec["files"].items()}
    spec["name"] = name
    return spec


TASKS = {
    "ra2": _task(
        "ra2", model="qwen-image-2.1",
        files={"dit": "qwen-image-2.1-Q4_K_M.gguf", "llm": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
               "llm_vision": "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "vae": "qwen_image_2.1_vae_bf16.safetensors"},
        model_args="", steps="20", guidance="1.0", cache_mode="easycache", cache_threshold="0.08",
        vae_tiling="0", compute_gib="4.5", image_required=False,
    ),
    "edit": _task(
        "edit", model="qwen-image-edit-2511",
        files={"dit": "qwen-image-edit-2511-Q4_K_M.gguf", "llm": "qwen_2.5_vl_7b.safetensors",
               "vae": "qwen_image_vae.safetensors"},
        model_args="qwen_image_zero_cond_t=true", steps="20", guidance="2.5", cache_mode="off",
        cache_threshold="0", vae_tiling="0", compute_gib="4.0", image_required=True,
    ),
}
ALIASES = {"ra2": "ra2", "generate": "ra2", "t2i": "ra2", "qwen-image-2.1": "ra2", "ra2-edit": "ra2",
           "edit": "edit", "qwen-edit": "edit", "qwen-image-edit": "edit", "qwen-image-edit-2511": "edit"}

_lock = threading.Lock()
_lib: qi.Lib | None = None
_ctxs: dict[str, dict] = {}
_info: dict[str, Any] = {}


LOG_LEVEL = int(os.getenv("MT_LOG_LEVEL", "2"))
_log: list[str] = []
_LOG_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


@_LOG_CB
def _on_log(level, text, _data):
    line = (text or b"").decode("utf-8", "replace").rstrip()
    if level >= 2 or "taking" in line or "completed" in line:
        _log.append(line)
        del _log[:-200]
    if level >= LOG_LEVEL:
        print(f"[sd{level}] {line}", flush=True)


def lib() -> qi.Lib:
    global _lib
    if _lib is None:
        _lib = qi.Lib(qi.SD_LIB)
        _info["abi"] = qi.verify_abi(_lib)
        _lib.lib.sd_set_log_callback.argtypes = [_LOG_CB, ctypes.c_void_p]
        _lib.lib.sd_set_log_callback(_on_log, None)
    return _lib


def stages() -> dict:
    out = {}
    for line in _log:
        m = re.search(r"([A-Za-z_][\w ]*?)(?: completed)?,? taking ([\d.]+) ?(ms|s)\b", line.split(" - ")[-1])
        if m:
            out[m.group(1).strip()[-40:]] = round(float(m.group(2)) * (1 if m.group(3) == "ms" else 1000))
    return out


_cudart = None


def vram() -> tuple[float, float]:
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL(os.getenv("MT_CUDART", "libcudart.so.12"))
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if _cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        raise RuntimeError("cudaMemGetInfo failed")
    return free.value / GIB, total.value / GIB


def gpu_name() -> str:
    if "gpu" not in _info:
        try:
            _info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                          text=True, capture_output=True, timeout=10).stdout.strip() or "unknown"
        except Exception:
            _info["gpu"] = "unknown"
    return _info["gpu"]


def need_gib(spec: dict, placement: str = "") -> float:
    files = spec["files"]
    if placement == "*=cpu":
        return spec["compute_gib"]
    keys = [k for k in files if not (placement == "te=cpu" and k.startswith("llm"))]
    return sum(files[k].stat().st_size for k in keys) / GIB + spec["compute_gib"]


def free_ctx(name: str) -> None:
    entry = _ctxs.pop(name, None)
    if entry:
        lib().lib.free_sd_ctx(entry["ctx"])


def plan(total: float) -> dict[str, str]:
    """Placement that keeps both tasks resident, preferring GPU text encoders (ra2 first)."""
    usable = total - RESERVE_GIB
    ra2, edit = TASKS["ra2"], TASKS["edit"]
    for pr, pe in (("", ""), ("", "te=cpu"), ("te=cpu", ""), ("te=cpu", "te=cpu")):
        if need_gib(ra2, pr) + need_gib(edit, pe) <= usable:
            return {"ra2": pr, "edit": pe}
    return {}


def load_ctx(name: str, placement: str, budget: float | None) -> int:
    spec = TASKS[name]
    params = qi.SdCtxParams()
    lib().lib.sd_ctx_params_init(ctypes.byref(params))
    refs: list[bytes] = []

    def keep(value) -> ctypes.c_char_p:
        raw = str(value).encode()
        refs.append(raw)
        return ctypes.c_char_p(raw)

    files = spec["files"]
    params.diffusion_model_path = keep(files["dit"])
    params.llm_path = keep(files["llm"])
    if "llm_vision" in files:
        params.llm_vision_path = keep(files["llm_vision"])
    params.vae_path = keep(files["vae"])
    params.n_threads = qi.THREADS
    params.flash_attn = qi.FLASH_ATTN
    params.diffusion_flash_attn = qi.FLASH_ATTN
    params.enable_mmap = True
    params.eager_load = _flag("MT_EAGER_LOAD", "1")
    if placement:
        params.params_backend = keep(placement)
    if budget is not None:
        params.max_vram = keep(f"{budget:.1f}")
    if spec["model_args"]:
        params.model_args = keep(spec["model_args"])
    started = time.monotonic()
    ctx = lib().lib.new_sd_ctx(ctypes.byref(params))
    if not ctx:
        raise RuntimeError(f"stable-diffusion.cpp could not load the {spec['model']} weight set")
    load_ms = int((time.monotonic() - started) * 1000)
    _ctxs[name] = {"ctx": ctx, "refs": refs, "used": time.monotonic(), "placement": placement or "gpu",
                   "streamed": budget is not None, "load_ms": load_ms}
    return load_ms


def ensure(name: str) -> dict:
    """Resident context for a task: fit beside others, else evict LRU, else stream under a budget."""
    spec = TASKS[name]
    if name in _ctxs:
        _ctxs[name]["used"] = time.monotonic()
        return {"load_ms": 0, "evicted": []}
    forced = os.getenv(f"MT_{name.upper()}_PLACEMENT")
    preferred = _info.get("plan", {}).get(name, "")
    options = [forced] if forced is not None else [preferred] + [p for p in ("", "te=cpu") if p != preferred]
    evicted: list[str] = []
    while True:
        free, total = vram()
        fit = next((p for p in options if need_gib(spec, p) + RESERVE_GIB <= free), None)
        if fit is not None or not _ctxs:
            break
        victim = min(_ctxs, key=lambda k: _ctxs[k]["used"])
        free_ctx(victim)
        evicted.append(victim)
    budget = None
    if fit is None:
        fit = forced if forced is not None else "*=cpu"
    load_ms = load_ctx(name, fit, budget)
    print(f"[qwen-mt] loaded {name} in {load_ms} ms placement={fit or 'gpu'} need={need_gib(spec, fit):.1f}GiB "
          f"free={free:.1f}/{total:.1f} budget={budget} evicted={evicted}", flush=True)
    return {"load_ms": load_ms, "evicted": evicted}


def build(spec: dict, values: dict) -> tuple[qi.SdImgGenParams, list]:
    params = qi.SdImgGenParams()
    lib().lib.sd_img_gen_params_init(ctypes.byref(params))
    keep: list = []

    def own(value) -> ctypes.c_char_p:
        raw = str(value).encode()
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
    w, h = qi._size(values)
    fallback = 512 if spec["name"] == "edit" else 1024
    params.width = qi._dimension(values.get("width", w), fallback)
    params.height = qi._dimension(values.get("height", h), fallback)
    steps = values.get("steps", values.get("num_inference_steps"))
    params.sample_params.sample_steps = spec["steps"] if steps is None else int(steps)
    if not 1 <= params.sample_params.sample_steps <= 100:
        raise ValueError("steps must be between 1 and 100")
    guidance = values.get("guidance_scale")
    cfg = spec["guidance"] if guidance in (None, 0, 0.0) else float(guidance)
    if not 0.0 < cfg <= 30.0:
        raise ValueError("guidance_scale must be between 0 and 30")
    params.sample_params.guidance.txt_cfg = cfg
    params.seed = qi._seed(values.get("seed"))
    params.batch_count = 1
    if values.get("n") not in (None, 1):
        raise ValueError("n must be 1")
    source = values.get("image_base64")
    if source is None and spec["image_required"]:
        raise ValueError("image_base64 is required for task edit")
    if source is not None:
        if not isinstance(source, str) or not source:
            raise ValueError("image_base64 must be a base64 PNG, JPEG or WebP")
        struct, buffer = qi._source_image(qi._decode_image(source))
        refs = (qi.SdImage * 1)(struct)
        keep += [buffer, struct, refs]
        params.ref_images = ctypes.cast(refs, ctypes.POINTER(qi.SdImage))
        params.ref_images_count = 1
        params.sample_params.flow_shift = qi.FLOW_SHIFT
    if spec["vae_tiling"]:
        params.vae_tiling_params.enabled = True
        params.vae_tiling_params.tile_size_x = qi.VAE_TILE
        params.vae_tiling_params.tile_size_y = qi.VAE_TILE
        params.vae_tiling_params.target_overlap = qi.VAE_TILE_OVERLAP
    mode = spec["cache_mode"]
    if mode in qi.SD_CACHE_MODES:
        params.cache.mode = qi.SD_CACHE_MODES[mode]
        params.cache.start_percent = qi.CACHE_START
        params.cache.end_percent = qi.CACHE_END
        params.cache.reuse_threshold = spec["cache_threshold"]
    return params, keep


def generate(name: str, values: dict) -> dict:
    spec = TASKS[name]
    fmt = str(values.get("output_format", "webp")).lower().replace("jpg", "jpeg")
    if fmt not in qi.FORMATS:
        raise ValueError("output_format must be webp, png, or jpeg")
    params, keep = build(spec, values)
    load = ensure(name)
    entry = _ctxs[name]
    images = ctypes.POINTER(qi.SdImage)()
    count = ctypes.c_int(0)
    _log.clear()
    started = time.monotonic()
    ok = lib().lib.generate_image(entry["ctx"], ctypes.byref(params), ctypes.byref(images), ctypes.byref(count))
    sample_ms = int((time.monotonic() - started) * 1000)
    if not ok or not images or count.value < 1:
        if images:
            lib().lib.free_sd_images(images, count.value)
        raise RuntimeError("stable-diffusion.cpp failed to generate an image: " + " | ".join(
            line for line in _log if "ERROR" in line.upper() or "fail" in line)[-400:])
    try:
        qi._state["seed"] = params.seed
        started = time.monotonic()
        encoded = qi.encode(images, count.value, fmt)
        encode_ms = int((time.monotonic() - started) * 1000)
    finally:
        lib().lib.free_sd_images(images, count.value)
        keep.clear()
    item = {"b64_json": encoded[0]["data"], "seed": params.seed, "inference_time_ms": sample_ms, "format": fmt}
    if params.cache.mode:
        item["denoiser_cache"] = {"requested": spec["cache_mode"], "approximate": True,
                                  "threshold": spec["cache_threshold"]}
    return {
        "created": int(time.time()), "model": spec["model"], "format": fmt, "data": [item],
        "outputs": encoded, "seed": params.seed,
        "timings": {"task": name, "gpu": gpu_name(), "load_ms": load["load_ms"], "evicted": load["evicted"],
                    "placement": entry["placement"], "streamed": entry["streamed"], "sample_ms": sample_ms, "encode_ms": encode_ms,
                    "stages_ms": stages(), "resident": sorted(_ctxs), "boot_ms": _info.get("boot_ms"),
                    "since_boot_s": round(time.monotonic() - _info.get("t0", time.monotonic()), 1),
                    "requests": _info.get("requests", 0)},
    }


def probe(values: dict) -> dict:
    free, total = vram()
    out = {"gpu": gpu_name(), "free_gib": round(free, 1), "total_gib": round(total, 1), "resident": sorted(_ctxs),
           "boot_ms": _info.get("boot_ms"), "preload": _info.get("preload")}
    path = values.get("probe_path")
    if path:
        limit = min(float(values.get("probe_gb", 4)), 16.0) * GIB
        started, read = time.monotonic(), 0
        with open(path, "rb", buffering=0) as handle:
            while read < limit:
                chunk = handle.read(64 << 20)
                if not chunk:
                    break
                read += len(chunk)
        seconds = time.monotonic() - started
        out["read"] = {"path": path, "gib": round(read / GIB, 2), "s": round(seconds, 2),
                       "gib_s": round(read / GIB / max(seconds, 1e-6), 2)}
    return out


def handler(job: dict, _pipe=None) -> dict:
    values = qi._inputs(job)
    if not isinstance(values, dict):
        raise ValueError("input must be an object")
    values = {k: v for k, v in values.items() if not k.startswith("_")}
    unknown = sorted(set(values) - ALLOWED)
    if unknown:
        return {"error": f"unknown inputs: {', '.join(unknown)}"}
    raw = str(values.pop("task", "") or values.get("workload") or "ra2").strip().lower()
    for key in ("workload", "kind", "profile"):
        values.pop(key, None)
    with _lock:
        _info["requests"] = _info.get("requests", 0) + 1
        if raw == "probe":
            return probe(values)
        name = ALIASES.get(raw)
        if name is None:
            return {"error": f"unknown task {raw!r}; use ra2 or edit"}
        try:
            return generate(name, values)
        except ValueError as error:
            return {"error": str(error)}


def boot() -> None:
    _info["t0"] = time.monotonic()
    started = time.monotonic()
    lib()
    _, total = vram()
    _info["plan"] = plan(total)
    order = [] if PRELOAD in {"", "none", "0"} else (["ra2", "edit"] if _info["plan"] else ["ra2"]) \
        if PRELOAD == "auto" else PRELOAD.split(",")
    for name in order:
        ensure(name)
    _info["preload"] = sorted(_ctxs)
    _info["boot_ms"] = int((time.monotonic() - started) * 1000)
    print(f"[qwen-mt] boot {_info['boot_ms']} ms gpu={gpu_name()} plan={_info['plan']} resident={_info['preload']}",
          flush=True)


if __name__ == "__main__":
    boot()
    if runpod is None:
        raise SystemExit("runpod SDK is not installed")
    runpod.serverless.start({"handler": handler})
