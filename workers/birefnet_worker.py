#!/usr/bin/env python3
"""Persistent CUDA BiRefNet worker behind the native C gateway."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.nn.functional as functional
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

try:
    import omatte
except ImportError:  # pragma: no cover - worker still runs without the library
    omatte = None

try:
    import object_store
except ImportError:  # pragma: no cover
    object_store = None

try:
    import video_matting
except ImportError:  # pragma: no cover - image cutouts remain available
    video_matting = None


MODEL_ID = os.getenv("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
DEVICE = os.getenv("BIREFNET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = int(os.getenv("BIREFNET_INPUT_SIZE", "1024"))
MAX_DOWNLOAD_BYTES = int(os.getenv("BIREFNET_MAX_DOWNLOAD_BYTES", str(64 << 20)))
# Colour decontamination: recover the true foreground colour so the backdrop
# (green screens especially) stops bleeding into semi-transparent edges.
DECONTAMINATE = os.getenv("BIREFNET_DECONTAMINATE", "1") == "1"
DECONTAMINATE_MAX_PIXELS = int(os.getenv("BIREFNET_DECONTAMINATE_MAX_PIXELS", str(16 << 20)))
# WebP keeps the alpha channel at a fraction of PNG's size; 85 is the quality
# used across the stack. exact=True keeps the RGB under transparent pixels, so a
# client-side brush can paint parts of the background back in.
WEBP_QUALITY = int(os.getenv("BIREFNET_WEBP_QUALITY", "85"))
DEFAULT_FORMAT = os.getenv("BIREFNET_OUTPUT_FORMAT", "webp").lower()
CACHE_ENABLED = os.getenv("BIREFNET_CACHE", "1") == "1"
JOB_TTL_SECONDS = int(os.getenv("BIREFNET_JOB_TTL", "3600"))
JOB_WORKERS = max(1, int(os.getenv("BIREFNET_JOB_WORKERS", "1")))
VIDEO_JOB_MAX_PENDING = max(1, int(os.getenv("VIDEO_MATTING_MAX_PENDING", "2")))
VIDEO_IDLE_RELEASE_SECONDS = max(0, int(os.getenv("VIDEO_MATTING_IDLE_RELEASE_SECONDS", "300")))
EVENTS_PATH = os.getenv("OMNISERVE_EVENTS_PATH", "").strip()
EVENTS_MAX_BYTES = max(1 << 20, int(os.getenv("OMNISERVE_EVENTS_MAX_BYTES", str(8 << 20))))

# Backdrop generation goes back out through the native gateway, not straight to
# the diffusion backend. The gateway pins /v1/images/backgrounds to the
# background tier, so a batch of replaced backdrops can never preempt an
# interactive request; calling the backend directly would bypass admission
# entirely and oversubscribe the device this worker is already sharing.
GATEWAY_BASE = os.getenv("BIREFNET_GATEWAY_BASE", "http://127.0.0.1:8791").rstrip("/")
GATEWAY_SECRET = os.getenv("BIREFNET_GATEWAY_SECRET", os.getenv("OMNISERVE_NATIVE_SECRET", ""))
BACKGROUND_ART_PATH = os.getenv("BIREFNET_ART_PATH", "/v1/images/backgrounds")
BACKGROUND_STYLE_PATH = os.getenv("BIREFNET_ART_STYLE_PATH", "/v1/images/backgrounds/style")
BACKGROUND_TIMEOUT = int(os.getenv("BIREFNET_ART_TIMEOUT", "300"))
# Backdrops are the workload latent teleportation was built for: nobody is
# waiting on them, the prompts repeat across a batch, and a backdrop that lands
# a little off the exact sampler trajectory is invisible behind a subject. The
# subject itself is never generated here, so the quality risk is bounded.
BACKGROUND_TELEPORT = os.getenv("BIREFNET_ART_TELEPORT", "1") == "1"
BACKGROUND_STEPS = int(os.getenv("BIREFNET_ART_STEPS", "9"))
# The diffusion backend authenticates on a query parameter, and the gateway
# relays the query string through untouched, so this rides along rather than
# being a second thing the gateway has to know about.
BACKGROUND_ART_SECRET = os.getenv("BIREFNET_ART_SECRET", "")


class RemoveBackgroundRequest(BaseModel):
    image_url: str = Field(min_length=1)
    output_format: str = DEFAULT_FORMAT
    foreground_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    decontaminate: bool | None = None
    cache: bool = True
    # The estimator solves for the backdrop jointly with the foreground at every
    # pixel of every level, so returning it costs nothing extra - it is already
    # computed. Useful for relighting, and for showing what was removed.
    return_background: bool = False
    # Replacement backdrop: "#rrggbb", a colour name the estimator understands
    # ("estimated" reuses the solved backdrop), or an http(s)/data image URL.
    background: str | None = None
    # Generate the backdrop instead, with the diffusion lane.
    background_prompt: str | None = None
    # Above zero, the generated backdrop is style-transferred from the estimated
    # one rather than made from scratch, so its lighting matches the subject's.
    background_strength: float = Field(default=0.0, ge=0.0, le=1.0)

    def cache_params(self) -> dict[str, Any]:
        """Everything that changes the pixels, and nothing that does not."""
        return {
            "format": self.output_format.lower(),
            "threshold": round(self.foreground_threshold, 4),
            "decontaminate": DECONTAMINATE if self.decontaminate is None else self.decontaminate,
            "model": MODEL_ID,
            "input_size": INPUT_SIZE,
            "quality": WEBP_QUALITY if self.output_format.lower() == "webp" else 0,
            "return_background": self.return_background,
            "background": self.background or "",
            "background_prompt": self.background_prompt or "",
            "background_strength": round(self.background_strength, 3),
        }

    def wants_extras(self) -> bool:
        """True when the answer is more than one image, so it has to be JSON."""
        return bool(self.return_background or self.background or self.background_prompt)


class ForegroundGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int = Field(default=0, ge=0)
    output_format: str = DEFAULT_FORMAT
    foreground_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    decontaminate: bool | None = None
    cache: bool = True


class VideoBackgroundRequest(BaseModel):
    video_url: str = Field(min_length=1)
    preserve_audio: bool = True
    output_upload_url: str = Field(min_length=1)
    output_public_url: str = Field(min_length=1)
    route_override: str = "auto"


class Runtime:
    model: Any = None
    transform: Any = None
    dtype: torch.dtype = torch.float16


runtime = Runtime()

# Jobs let a browser fire once and poll, instead of holding a long request open.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# The GPU is serialised anyway; one worker keeps queueing honest.
_job_pool = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="birefnet-job")
_video_job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-matting-job")
_event_lock = threading.Lock()
_idle_stop = threading.Event()
_video_stats: dict[str, Any] = {
    "submitted": 0,
    "completed": 0,
    "errors": 0,
    "fallbacks": 0,
    "rejected": 0,
    "total_seconds": 0.0,
    "last_duration_seconds": 0.0,
    "last_route": "",
    "last_error": "",
    "last_error_at": 0.0,
    "last_activity_at": time.time(),
    "last_release_at": 0.0,
}


def _prune_jobs(now: float) -> None:
    stale = [key for key, job in _jobs.items() if now - job.get("updated", now) > JOB_TTL_SECONDS]
    for key in stale:
        _jobs.pop(key, None)


def _set_job(job_id: str, **fields: Any) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id, "created": time.time()})
        job.update(fields)
        job["updated"] = time.time()
        _prune_jobs(job["updated"])
        return dict(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _emit_event(level: str, event: str, **fields: Any) -> None:
    """Emit a secret-free event to journald and the monitoring JSONL feed."""
    payload = {"timestamp": time.time(), "level": level, "event": event, **fields}
    encoded = json.dumps(payload, separators=(",", ":"), default=str)
    print(f"OMNISERVE_EVENT {encoded}", flush=True)
    if not EVENTS_PATH:
        return
    try:
        with _event_lock:
            path = Path(EVENTS_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size >= EVENTS_MAX_BYTES:
                os.replace(path, path.with_suffix(path.suffix + ".1"))
            with path.open("a", encoding="utf-8") as output:
                output.write(encoded + "\n")
    except OSError as error:  # journald remains the durable fallback
        print(f"OMNISERVE_EVENT_WRITE_ERROR {error}", flush=True)


def _safe_error(error: Any) -> str:
    message = str(error)[:1000]
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", message)


def _video_counts_locked() -> dict[str, int]:
    statuses = [job.get("status") for job in _jobs.values() if job.get("kind") == "video"]
    return {
        "pending": sum(status in {"queued", "routing", "matting"} for status in statuses),
        "queued": statuses.count("queued"),
        "running": sum(status in {"routing", "matting"} for status in statuses),
    }


def _gpu_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "available": True,
            "free_mib": round(free_bytes / (1 << 20)),
            "total_mib": round(total_bytes / (1 << 20)),
            "allocated_mib": round(torch.cuda.memory_allocated() / (1 << 20)),
            "reserved_mib": round(torch.cuda.memory_reserved() / (1 << 20)),
        }
    except Exception as error:  # pragma: no cover - driver failure
        return {"available": True, "error": str(error)[:300]}


def _video_capacity() -> dict[str, Any]:
    with _jobs_lock:
        counts = _video_counts_locked()
        stats = dict(_video_stats)
    finished = stats["completed"] + stats["errors"]
    stats["average_seconds"] = round(stats["total_seconds"] / finished, 3) if finished else 0.0
    for key in ("total_seconds", "last_duration_seconds"):
        stats[key] = round(stats[key], 3)
    runtime_health = video_matting.health() if video_matting is not None else {"status": "unavailable"}
    runtime_ready = (
        video_matting is not None
        and torch.cuda.is_available()
        and runtime_health.get("person_detector") != "error"
    )
    return {
        **counts,
        "max_pending": VIDEO_JOB_MAX_PENDING,
        "accepting": runtime_ready and counts["pending"] < VIDEO_JOB_MAX_PENDING,
        "runtime_ready": runtime_ready,
        "stats": stats,
        "gpu": _gpu_metrics(),
    }


def _idle_release_loop() -> None:
    if VIDEO_IDLE_RELEASE_SECONDS <= 0:
        return
    interval = min(30, max(5, VIDEO_IDLE_RELEASE_SECONDS // 4))
    while not _idle_stop.wait(interval):
        with _jobs_lock:
            pending = _video_counts_locked()["pending"]
            idle_for = time.time() - _video_stats["last_activity_at"]
        if pending or idle_for < VIDEO_IDLE_RELEASE_SECONDS or video_matting is None:
            continue
        runtime_health = video_matting.health()
        if not runtime_health.get("rvm_loaded") and runtime_health.get("person_detector") == "loading":
            continue
        video_matting.release()
        released_at = time.time()
        with _jobs_lock:
            _video_stats["last_release_at"] = released_at
        _emit_event("info", "video_models_released", idle_seconds=round(idle_for, 1))


def load_model() -> None:
    if DEVICE.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    runtime.dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=runtime.dtype,
    ).to(DEVICE)
    model.eval()
    if DEVICE.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    if os.getenv("BIREFNET_TORCH_COMPILE", "0") == "1":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    runtime.model = model
    runtime.transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def read_image(value: str) -> Image.Image:
    if value.startswith("data:"):
        try:
            payload = value.split(",", 1)[1]
            data = base64.b64decode(payload, validate=True)
        except (IndexError, ValueError) as error:
            raise HTTPException(400, "invalid data URL") from error
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise HTTPException(413, "source image is too large")
    else:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise HTTPException(400, "image_url must use http, https, or data")
        try:
            with requests.get(value, timeout=(10, 60), stream=True) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(1 << 20):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(413, "source image is too large")
                    chunks.append(chunk)
                data = b"".join(chunks)
        except requests.RequestException as error:
            raise HTTPException(502, f"source image download failed: {error}") from error
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as error:
        raise HTTPException(400, "source is not a supported image") from error


CSS_COLOURS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "grey": (0.5, 0.5, 0.5),
    "gray": (0.5, 0.5, 0.5),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def parse_colour(value: str) -> tuple[float, float, float] | None:
    """"#rgb", "#rrggbb" or one of a few names, as floats in [0, 1]."""
    text = value.strip().lower()
    if text in CSS_COLOURS:
        return CSS_COLOURS[text]
    if not text.startswith("#"):
        return None
    digits = text[1:]
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    if len(digits) != 6:
        return None
    try:
        return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def image_to_device(image: Image.Image, device: str):
    """(h, w, 3) float32 in [0, 1] on the GPU, uploaded as uint8.

    Converting on the device means 3 bytes per pixel cross the bus instead of
    12, which on a 1024x1024 cutout is 3 MB rather than 12 MB.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device, non_blocking=True)
    return tensor.float().div_(255.0)


def device_to_image(tensor) -> Image.Image:
    array = tensor.clamp(0.0, 1.0).mul_(255.0).add_(0.5).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def estimate_foreground_background(image: Image.Image, alpha_device, alpha_array: np.ndarray,
                                   want_background: bool):
    """The decontamination pass: recovers true foreground (and backdrop) colour.

    Without it the RGB under a semi-transparent edge is still the composite -
    foreground blended with the old backdrop - so a cutout keeps a green or blue
    fringe when it is placed on a new background.

    Returns (foreground, background) as either CUDA tensors or numpy arrays; the
    caller does not need to care which, but `on_device` in the result says so.
    Falls back to the host path when the device API is missing, and returns
    (None, None) when foreground estimation is unavailable or the image is
    larger than the pass is configured to handle.
    """
    if omatte is None or image.width * image.height > DECONTAMINATE_MAX_PIXELS:
        return None, None, False

    if alpha_device is not None and omatte.device_api_available():
        source = image_to_device(image, str(alpha_device.device))
        alpha = alpha_device.contiguous()
        result = omatte.estimate_foreground_torch(source, alpha, return_background=want_background)
        foreground, background = result if want_background else (result, None)
        return foreground, background, True

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    result = omatte.estimate_foreground(rgb, alpha_array.astype(np.float32),
                                        return_background=want_background)
    foreground, background = result if want_background else (result, None)
    return foreground, background, False


def to_pil_rgb(colours, on_device) -> Image.Image:
    if on_device:
        return device_to_image(colours)
    return Image.fromarray(np.clip(colours * 255.0 + 0.5, 0, 255).astype("uint8"), mode="RGB")


def composite_over(foreground, alpha_device, alpha_array, on_device, backdrop=None,
                   backdrop_rgb=None) -> Image.Image:
    """alpha * foreground + (1 - alpha) * backdrop, wherever the pixels live."""
    if on_device:
        backdrop_device = None
        if backdrop is not None:
            backdrop_device = image_to_device(backdrop, str(alpha_device.device))
        out = omatte.composite_torch(foreground, alpha_device, background=backdrop_device,
                                     background_rgb=backdrop_rgb)
        return device_to_image(out)

    alpha = alpha_array.astype(np.float32)[..., None]
    if backdrop is not None:
        back = np.asarray(backdrop.convert("RGB"), dtype=np.float32) / 255.0
    else:
        back = np.asarray(backdrop_rgb or (0.0, 0.0, 0.0), dtype=np.float32)
    blended = alpha * foreground + (1.0 - alpha) * back
    return Image.fromarray(np.clip(blended * 255.0 + 0.5, 0, 255).astype("uint8"), mode="RGB")


def encode_image(rgba: Image.Image, output_format: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if output_format == "webp":
        # exact=True: do not discard RGB under transparent pixels, so a brush
        # can paint the background back in without smearing.
        rgba.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6, exact=True)
        return buffer.getvalue(), "image/webp"
    rgba.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), "image/png"


def artifact_key(request: RemoveBackgroundRequest, name: str) -> str | None:
    """Cache key for one output of one request, or None when caching is off.

    Every artifact of a request gets its own key with the artifact name folded
    in, so asking only for the cutout and later also for the backdrop does not
    have one overwrite the other.
    """
    if not (CACHE_ENABLED and request.cache and object_store is not None):
        return None
    params = dict(request.cache_params())
    params["artifact"] = name
    return object_store.cache_key(request.image_url, params,
                                  suffix=request.output_format.lower())


def produce_cutout(request: RemoveBackgroundRequest) -> dict[str, Any]:
    """Cutout (and any extra artifacts) with a content-addressed cache in front.

    The cache key is the (normalised) source URL plus every parameter that
    changes the pixels, so asking twice for the same picture is one GPU run -
    and with a replacement backdrop that is one segmentation, one matte solve
    and one diffusion call saved, not just the cheap part.
    """
    output_format = request.output_format.lower()
    if output_format not in {"webp", "png"}:
        raise HTTPException(400, "output_format must be webp or png")

    media_type = "image/webp" if output_format == "webp" else "image/png"
    wanted = ["cutout"]
    if request.return_background:
        wanted.append("background")
    if request.background_prompt or (request.background or "").strip() not in {"", "transparent"}:
        wanted.append("composite")

    keys = {name: artifact_key(request, name) for name in wanted}
    if all(key is not None and object_store.exists(key) for key in keys.values()):
        artifacts = {}
        for name, key in keys.items():
            url = object_store.public_url(key)
            artifacts[name] = {"key": key, "url": url,
                               "content": None if url else object_store.get(key)}
        return {"cached": True, "media_type": media_type, "artifacts": artifacts}

    image = read_image(request.image_url)
    produced = remove_background(image, request)

    artifacts: dict[str, Any] = {}
    for name, content in produced.items():
        key = keys.get(name) or artifact_key(request, name)
        url = None
        if key is not None:
            try:
                url = object_store.put(key, content, media_type)
            except Exception as error:  # storage must never break the response
                print(f"cutout upload failed: {error}")
        artifacts[name] = {"key": key, "url": url, "content": content}

    return {
        "cached": False,
        "media_type": media_type,
        "artifacts": artifacts,
        "width": image.width,
        "height": image.height,
    }


def _gateway_url(path: str) -> str:
    return f"{GATEWAY_BASE}{path}"


def _gateway_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {GATEWAY_SECRET}"} if GATEWAY_SECRET else {}


def generate_backdrop(prompt: str, width: int, height: int, init_url: str | None,
                      strength: float, seed: int = 0) -> Image.Image:
    """Asks the diffusion lane for a replacement backdrop.

    With an init_url and a strength above zero this is a style transfer over the
    *estimated* backdrop, so the new scene inherits the original's lighting and
    the subject does not look pasted on. Without one it is plain text-to-image.
    """
    query = {"secret": BACKGROUND_ART_SECRET} if BACKGROUND_ART_SECRET else {}
    if strength > 0.0 and init_url:
        response = requests.get(
            _gateway_url(BACKGROUND_STYLE_PATH),
            params={**query, "prompt": prompt, "image_url": init_url, "strength": strength},
            headers=_gateway_headers(), timeout=(10, BACKGROUND_TIMEOUT),
        )
    else:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "num_inference_steps": BACKGROUND_STEPS,
            # Nobody is waiting on a backdrop, so it should never be the reason
            # an interactive request queues.
            "low_priority": True,
            # Backdrop prompts repeat hard across a batch, which is exactly the
            # case latent teleportation replays from cache instead of sampling.
            "teleport": BACKGROUND_TELEPORT,
        }
        response = requests.post(_gateway_url(BACKGROUND_ART_PATH), json=payload, params=query,
                                 headers=_gateway_headers(), timeout=(10, BACKGROUND_TIMEOUT))
    if response.status_code >= 400:
        raise HTTPException(502, f"backdrop generation failed: {response.status_code}")

    content_type = response.headers.get("Content-Type", "")
    data = response.content
    if content_type.startswith("application/json"):
        body = response.json()
        url = body.get("url") or body.get("image_url")
        entries = body.get("data") or []
        if not url and entries:
            first = entries[0]
            url = first.get("url")
            if not url and first.get("b64_json"):
                data = base64.b64decode(first["b64_json"])
                url = None
        if url:
            return read_image(url)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def publish_intermediate(image: Image.Image, key_hint: str) -> str | None:
    """Uploads a working image so the diffusion lane can fetch it by URL."""
    if object_store is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=90, method=4)
    payload = buffer.getvalue()
    try:
        key = object_store.cache_key(key_hint, {"stage": "estimated-backdrop"}, suffix="webp")
        return object_store.put(key, payload, "image/webp")
    except Exception as error:  # a missing bucket downgrades to text-to-image
        print(f"backdrop upload failed: {error}")
        return None


def resolve_backdrop(request: RemoveBackgroundRequest, estimated: Image.Image | None,
                     width: int, height: int) -> tuple[Image.Image | None, tuple[float, ...] | None]:
    """Turns the request's backdrop spec into either an image or a solid colour."""
    if request.background_prompt:
        init_url = None
        if request.background_strength > 0.0 and estimated is not None:
            init_url = publish_intermediate(estimated, request.image_url)
        backdrop = generate_backdrop(request.background_prompt, width, height, init_url,
                                     request.background_strength)
        if backdrop.size != (width, height):
            backdrop = ImageOps.fit(backdrop, (width, height), method=Image.LANCZOS)
        return backdrop, None

    spec = (request.background or "").strip()
    if not spec or spec == "transparent":
        return None, None
    if spec == "estimated":
        return estimated, None
    colour = parse_colour(spec)
    if colour is not None:
        return None, colour
    backdrop = read_image(spec)
    if backdrop.size != (width, height):
        backdrop = ImageOps.fit(backdrop, (width, height), method=Image.LANCZOS)
    return backdrop, None


@torch.inference_mode()
def segment(image: Image.Image, threshold: float):
    """BiRefNet alpha, left on the device it was produced on.

    The matte pass that follows reads it there, so bringing it to the host only
    to send it straight back was ~4 MB of pure round trip per cutout.
    """
    if runtime.model is None:
        raise HTTPException(503, "BiRefNet is not loaded")
    tensor = runtime.transform(image).unsqueeze(0).to(DEVICE, dtype=runtime.dtype)
    if DEVICE.startswith("cuda"):
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=runtime.dtype, enabled=DEVICE.startswith("cuda")):
        prediction = runtime.model(tensor)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[-1]
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[-1]
    mask = torch.sigmoid(prediction)
    mask = functional.interpolate(mask, size=(image.height, image.width), mode="bilinear",
                                  align_corners=False)
    mask = mask[0, 0].float().clamp(0, 1)
    if threshold > 0:
        mask = torch.where(mask >= threshold, mask, torch.zeros_like(mask))
    mask = mask.contiguous()
    return (mask, mask.cpu().numpy()) if mask.is_cuda else (None, mask.numpy())


@torch.inference_mode()
def remove_background(image: Image.Image, request: RemoveBackgroundRequest) -> dict[str, bytes]:
    """Cutout, and whatever else the request asked for.

    Returns a dict of artifact name -> encoded bytes. "cutout" is always
    present; "background" is the estimated backdrop, "composite" is the subject
    over a replacement one.
    """
    output_format = request.output_format.lower()
    mask_device, mask_array = segment(image, request.foreground_threshold)
    alpha = Image.fromarray((mask_array * 255).astype("uint8"), mode="L")

    want_decontaminate = DECONTAMINATE if request.decontaminate is None else request.decontaminate
    needs_backdrop = bool(request.return_background or
                          (request.background_prompt and request.background_strength > 0.0) or
                          (request.background or "").strip() == "estimated")
    # The subject has to be decontaminated whenever it will be placed on
    # something new, or the old backdrop's colour comes with it.
    replacing = bool(request.background_prompt or
                     ((request.background or "").strip() not in {"", "transparent"}))
    if replacing:
        want_decontaminate = True

    foreground = None
    estimated_backdrop = None
    on_device = False
    if want_decontaminate or needs_backdrop:
        try:
            foreground, estimated_backdrop, on_device = estimate_foreground_background(
                image, mask_device, mask_array, needs_backdrop)
        except Exception as error:  # never fail the cutout over colour cleanup
            print(f"colour decontamination skipped: {error}")
            foreground, estimated_backdrop, on_device = None, None, False

    base = to_pil_rgb(foreground, on_device) if foreground is not None else image
    rgba = base.convert("RGBA")
    rgba.putalpha(alpha)

    artifacts: dict[str, bytes] = {"cutout": encode_image(rgba, output_format)[0]}

    backdrop_image = None
    if estimated_backdrop is not None:
        backdrop_image = to_pil_rgb(estimated_backdrop, on_device)
        if request.return_background:
            artifacts["background"] = encode_image(backdrop_image, output_format)[0]

    if replacing:
        backdrop, backdrop_rgb = resolve_backdrop(request, backdrop_image, image.width,
                                                  image.height)
        if backdrop is not None or backdrop_rgb is not None:
            if foreground is None:
                # No matte pass ran, so composite the observed pixels. The edge
                # keeps some of the old backdrop; say so rather than pretending.
                composite = Image.composite(image.convert("RGB"),
                                            backdrop or Image.new("RGB", image.size, tuple(
                                                int(c * 255 + 0.5) for c in (backdrop_rgb or (0, 0, 0)))),
                                            alpha)
            else:
                composite = composite_over(foreground, mask_device, mask_array, on_device,
                                           backdrop=backdrop, backdrop_rgb=backdrop_rgb)
            artifacts["composite"] = encode_image(composite, output_format)[0]

    return artifacts


@asynccontextmanager
async def lifespan(_: FastAPI):
    _idle_stop.clear()
    load_model()
    if video_matting is not None:
        video_matting.preload_person_detector_async()
    idle_thread = threading.Thread(target=_idle_release_loop, name="video-model-reaper", daemon=True)
    idle_thread.start()
    yield
    _idle_stop.set()
    idle_thread.join(timeout=2)
    runtime.model = None
    if video_matting is not None:
        video_matting.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="BiRefNet worker", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "input_size": INPUT_SIZE,
        "decontaminate": DECONTAMINATE and omatte is not None,
        "matte_library": omatte.library_path() if omatte else None,
        "matte_cuda": bool(omatte and omatte.cuda_available()),
        "output_format": DEFAULT_FORMAT,
        "webp_quality": WEBP_QUALITY,
        "cache": object_store.describe() if (CACHE_ENABLED and object_store) else {"backend": "off"},
        "video_matting": video_matting.health() if video_matting is not None else {"status": "unavailable"},
        "video_capacity": _video_capacity(),
    }


@app.get("/ready/video")
def video_readiness() -> Response:
    capacity = _video_capacity()
    payload = {
        "status": "ready" if capacity["accepting"] else "unavailable",
        "accepting": capacity["accepting"],
        "pending": capacity["pending"],
        "max_pending": capacity["max_pending"],
        "runtime_ready": capacity["runtime_ready"],
        "gpu": capacity["gpu"],
    }
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
        status_code=200 if capacity["accepting"] else 503,
        headers={"Retry-After": "5"} if not capacity["accepting"] else None,
    )


def artifact_payload(result: dict[str, Any]) -> dict[str, Any]:
    """JSON view of a multi-artifact result: URLs, or inline data when no bucket."""
    payload: dict[str, Any] = {
        "cached": result["cached"],
        "media_type": result["media_type"],
    }
    for name, artifact in result["artifacts"].items():
        entry = {"key": artifact["key"], "url": artifact["url"]}
        if not artifact["url"] and artifact["content"]:
            entry["data_url"] = (f"data:{result['media_type']};base64,"
                                 + base64.b64encode(artifact["content"]).decode())
        payload[name] = entry
    return payload


@app.post("/v1/images/background-removals")
def background_removal(request: RemoveBackgroundRequest) -> Response:
    """Synchronous cutout: returns the image bytes, cache-hit or not.

    A request that asks for more than one image gets JSON instead, because a
    single body cannot carry a cutout, its backdrop and a composite.
    """
    if request.background_prompt:
        # Diffusion takes seconds to minutes, and this handler holds a gateway
        # permit for its whole duration. Making the caller use the job lane is
        # what keeps a backdrop replacement from occupying an interactive slot.
        raise HTTPException(400, "background_prompt requires the /jobs endpoint")

    result = produce_cutout(request)
    if request.wants_extras():
        return Response(content=json.dumps(artifact_payload(result)),
                        media_type="application/json")

    cutout = result["artifacts"]["cutout"]
    content = cutout["content"]
    if content is None and cutout.get("url"):
        # Stored remotely and we were not asked to proxy the bytes back.
        return Response(
            status_code=302,
            headers={"Location": cutout["url"], "X-Cutout-Cached": "1",
                     "X-Cutout-Url": cutout["url"]},
        )

    headers = {
        "X-BiRefNet-Model": MODEL_ID,
        "X-Cutout-Cached": "1" if result["cached"] else "0",
    }
    if cutout.get("url"):
        headers["X-Cutout-Url"] = cutout["url"]
    if result.get("width"):
        headers["X-Source-Width"] = str(result["width"])
        headers["X-Source-Height"] = str(result["height"])
    return Response(content=content, media_type=result["media_type"], headers=headers)


def _run_job(job_id: str, request: RemoveBackgroundRequest) -> None:
    _set_job(job_id, status="running")
    try:
        result = produce_cutout(request)
    except HTTPException as error:
        _set_job(job_id, status="error", error=str(error.detail), http_status=error.status_code)
        return
    except Exception as error:  # pragma: no cover - defensive
        _set_job(job_id, status="error", error=str(error), http_status=500)
        return

    # No bucket configured: artifact_payload hands the bytes back inline so the
    # browser can still show something without another endpoint.
    payload = artifact_payload(result)
    payload["status"] = "done"
    cutout = payload.get("cutout", {})
    # Kept flat for callers written against the single-artifact response.
    payload["url"] = cutout.get("url")
    payload["key"] = cutout.get("key")
    if cutout.get("data_url"):
        payload["data_url"] = cutout["data_url"]
    _set_job(job_id, **payload)


def _run_foreground_generation(job_id: str, request: ForegroundGenerationRequest) -> None:
    _set_job(job_id, status="generating")
    try:
        generated = generate_backdrop(request.prompt, request.width, request.height,
                                      None, 0.0, request.seed)
        source = io.BytesIO()
        generated.save(source, format="WEBP", quality=95, method=4)
        image_url = "data:image/webp;base64," + base64.b64encode(source.getvalue()).decode()
        _set_job(job_id, status="matting")
        result = produce_cutout(RemoveBackgroundRequest(
            image_url=image_url,
            output_format=request.output_format,
            foreground_threshold=request.foreground_threshold,
            decontaminate=request.decontaminate,
            cache=request.cache,
        ))
    except HTTPException as error:
        _set_job(job_id, status="error", error=str(error.detail), http_status=error.status_code)
        return
    except Exception as error:  # pragma: no cover - defensive
        _set_job(job_id, status="error", error=str(error), http_status=500)
        return

    payload = artifact_payload(result)
    payload.update({"status": "done", "seed": request.seed,
                    "source_width": generated.width, "source_height": generated.height})
    cutout = payload.get("cutout", {})
    payload["url"] = cutout.get("url")
    payload["key"] = cutout.get("key")
    if cutout.get("data_url"):
        payload["data_url"] = cutout["data_url"]
    _set_job(job_id, **payload)


def _run_video_background_removal(job_id: str, request: VideoBackgroundRequest) -> None:
    started_at = time.time()
    queued = get_job(job_id) or {}
    queue_seconds = max(0.0, started_at - queued.get("created", started_at))
    _set_job(job_id, status="routing", started_at=started_at,
             queue_seconds=round(queue_seconds, 3))
    with _jobs_lock:
        _video_stats["last_activity_at"] = started_at
    _emit_event("info", "video_job_started", job_id=job_id,
                queue_seconds=round(queue_seconds, 3))
    if video_matting is None:
        message = "video matting runtime is unavailable"
        _set_job(job_id, status="error", error=message, http_status=503,
                 finished_at=time.time())
        with _jobs_lock:
            _video_stats["errors"] += 1
            _video_stats["last_error"] = message
            _video_stats["last_error_at"] = time.time()
            _video_stats["last_activity_at"] = time.time()
        _emit_event("error", "video_job_error", job_id=job_id, error=message,
                    http_status=503)
        return
    try:
        result = video_matting.process(
            request.model_dump(),
            progress=lambda frames: _set_job(job_id, status="matting", frames=frames),
        )
    except Exception as error:  # provider fallback is owned by the caller
        finished_at = time.time()
        elapsed = finished_at - started_at
        message = _safe_error(error)
        _set_job(job_id, status="error", error=message, http_status=500,
                 finished_at=finished_at, elapsed_seconds=round(elapsed, 3))
        with _jobs_lock:
            _video_stats["errors"] += 1
            _video_stats["total_seconds"] += elapsed
            _video_stats["last_duration_seconds"] = elapsed
            _video_stats["last_error"] = message
            _video_stats["last_error_at"] = finished_at
            _video_stats["last_activity_at"] = finished_at
        _emit_event("error", "video_job_error", job_id=job_id, error=message,
                    http_status=500, elapsed_seconds=round(elapsed, 3))
        return
    finished_at = time.time()
    elapsed = finished_at - started_at
    result["status"] = "done"
    result.update({"finished_at": finished_at, "elapsed_seconds": round(elapsed, 3)})
    _set_job(job_id, **result)
    fallback = bool(result.get("fallback_required"))
    route = str(result.get("route") or "")
    with _jobs_lock:
        _video_stats["completed"] += 1
        _video_stats["fallbacks"] += int(fallback)
        _video_stats["total_seconds"] += elapsed
        _video_stats["last_duration_seconds"] = elapsed
        _video_stats["last_route"] = route
        _video_stats["last_activity_at"] = finished_at
    _emit_event("warning" if fallback else "info", "video_job_fallback" if fallback else "video_job_completed",
                job_id=job_id, route=route, fallback_reason=result.get("fallback_reason", ""),
                elapsed_seconds=round(elapsed, 3), frames=result.get("metrics", {}).get("frames", 0))


@app.post("/v1/images/background-removals/jobs")
def enqueue_background_removal(request: RemoveBackgroundRequest) -> dict[str, Any]:
    """Starts a cutout and returns immediately so the client can poll.

    A cache hit is answered inline - no job, no queue, no GPU.
    """
    if not request.wants_extras():
        key = artifact_key(request, "cutout")
        if key is not None and object_store.exists(key):
            url = object_store.public_url(key)
            if url:
                return {"job_id": None, "status": "done", "cached": True, "url": url, "key": key}

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="queued", cached=False)
    _job_pool.submit(_run_job, job_id, request)
    return {"job_id": job_id, "status": "queued", "cached": False, "poll_after_ms": 700}


@app.post("/v1/images/foreground-generations/jobs")
def enqueue_foreground_generation(request: ForegroundGenerationRequest) -> dict[str, Any]:
    """Runs text-to-image and BiRefNet as one queued foreground-art stage."""
    if request.output_format.lower() not in {"webp", "png"}:
        raise HTTPException(400, "output_format must be webp or png")
    job_id = uuid.uuid4().hex
    _set_job(job_id, status="queued", cached=False)
    _job_pool.submit(_run_foreground_generation, job_id, request)
    return {"job_id": job_id, "status": "queued", "cached": False, "poll_after_ms": 700}


@app.post("/v1/videos/background-removals/jobs")
def enqueue_video_background_removal(request: VideoBackgroundRequest) -> dict[str, Any]:
    if request.route_override not in {"auto", "rvm", "standby"}:
        raise HTTPException(400, "route_override must be auto, rvm, or standby")
    job_id = uuid.uuid4().hex
    now = time.time()
    with _jobs_lock:
        _prune_jobs(now)
        pending = _video_counts_locked()["pending"]
        if pending >= VIDEO_JOB_MAX_PENDING:
            _video_stats["rejected"] += 1
            rejected = True
        else:
            _jobs[job_id] = {"job_id": job_id, "kind": "video", "status": "queued",
                             "cached": False, "created": now, "updated": now}
            _video_stats["submitted"] += 1
            _video_stats["last_activity_at"] = now
            rejected = False
    if rejected:
        _emit_event("warning", "video_queue_rejected", pending=pending,
                    max_pending=VIDEO_JOB_MAX_PENDING)
        raise HTTPException(
            429,
            f"local video queue is full ({pending}/{VIDEO_JOB_MAX_PENDING}); use standby provider",
            headers={"Retry-After": "5"},
        )
    _emit_event("info", "video_job_queued", job_id=job_id, pending=pending + 1,
                max_pending=VIDEO_JOB_MAX_PENDING)
    _video_job_pool.submit(_run_video_background_removal, job_id, request)
    return {"job_id": job_id, "status": "queued", "cached": False, "poll_after_ms": 1000}


@app.get("/v1/images/background-removals/jobs/{job_id}")
def background_removal_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.pop("created", None)
    job.pop("updated", None)
    return job


@app.get("/v1/images/foreground-generations/jobs/{job_id}")
def foreground_generation_job(job_id: str) -> dict[str, Any]:
    return background_removal_job(job_id)


@app.get("/v1/videos/background-removals/jobs/{job_id}")
def video_background_removal_job(job_id: str) -> dict[str, Any]:
    return background_removal_job(job_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("BIREFNET_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BIREFNET_PORT", "9094")))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
