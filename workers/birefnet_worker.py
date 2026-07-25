#!/usr/bin/env python3
"""Persistent CUDA BiRefNet worker behind the native C gateway."""

from __future__ import annotations

import argparse
import base64
import io
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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


class RemoveBackgroundRequest(BaseModel):
    image_url: str = Field(min_length=1)
    output_format: str = DEFAULT_FORMAT
    foreground_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    decontaminate: bool | None = None
    cache: bool = True

    def cache_params(self) -> dict[str, Any]:
        """Everything that changes the pixels, and nothing that does not."""
        return {
            "format": self.output_format.lower(),
            "threshold": round(self.foreground_threshold, 4),
            "decontaminate": DECONTAMINATE if self.decontaminate is None else self.decontaminate,
            "model": MODEL_ID,
            "input_size": INPUT_SIZE,
            "quality": WEBP_QUALITY if self.output_format.lower() == "webp" else 0,
        }


class Runtime:
    model: Any = None
    transform: Any = None
    dtype: torch.dtype = torch.float16


runtime = Runtime()

# Jobs let a browser fire once and poll, instead of holding a long request open.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# The GPU is serialised anyway; one worker keeps queueing honest.
_job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="birefnet-job")


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


def decontaminate_colours(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Replaces observed colours with the estimated foreground colours.

    Without this the RGB under a semi-transparent edge is still the composite
    (foreground blended with the old backdrop), so a cutout keeps a green or
    blue fringe when it is placed on a new background.
    """
    if omatte is None:
        return image
    if image.width * image.height > DECONTAMINATE_MAX_PIXELS:
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    foreground = omatte.estimate_foreground(rgb, alpha.astype(np.float32))
    return Image.fromarray(np.clip(foreground * 255.0 + 0.5, 0, 255).astype("uint8"), mode="RGB")


def encode_image(rgba: Image.Image, output_format: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if output_format == "webp":
        # exact=True: do not discard RGB under transparent pixels, so a brush
        # can paint the background back in without smearing.
        rgba.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6, exact=True)
        return buffer.getvalue(), "image/webp"
    rgba.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), "image/png"


def produce_cutout(request: RemoveBackgroundRequest) -> dict[str, Any]:
    """Cutout with a content-addressed cache in front of the model.

    The cache key is the (normalised) source URL plus every parameter that
    changes the pixels, so asking twice for the same picture is one GPU run.
    """
    output_format = request.output_format.lower()
    if output_format not in {"webp", "png"}:
        raise HTTPException(400, "output_format must be webp or png")

    media_type = "image/webp" if output_format == "webp" else "image/png"
    key = None
    if CACHE_ENABLED and request.cache and object_store is not None:
        key = object_store.cache_key(request.image_url, request.cache_params(), suffix=output_format)
        if object_store.exists(key):
            url = object_store.public_url(key)
            payload = None if url else object_store.get(key)
            return {"cached": True, "key": key, "url": url, "content": payload, "media_type": media_type}

    image = read_image(request.image_url)
    content = remove_background(image, request.foreground_threshold, request.decontaminate,
                                output_format=output_format)

    url = None
    if key is not None:
        try:
            url = object_store.put(key, content, media_type)
        except Exception as error:  # storage must never break the response
            print(f"cutout upload failed: {error}")

    return {
        "cached": False,
        "key": key,
        "url": url,
        "content": content,
        "media_type": media_type,
        "width": image.width,
        "height": image.height,
    }


@torch.inference_mode()
def remove_background(image: Image.Image, threshold: float, decontaminate: bool | None = None,
                      output_format: str = DEFAULT_FORMAT) -> bytes:
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
    mask = functional.interpolate(mask, size=(image.height, image.width), mode="bilinear", align_corners=False)
    mask = mask[0, 0].float().cpu().clamp(0, 1)
    if threshold > 0:
        mask = torch.where(mask >= threshold, mask, torch.zeros_like(mask))
    mask_array = mask.numpy()
    alpha = Image.fromarray((mask_array * 255).astype("uint8"), mode="L")

    want_decontaminate = DECONTAMINATE if decontaminate is None else decontaminate
    base = image
    if want_decontaminate:
        try:
            base = decontaminate_colours(image, mask_array)
        except Exception as error:  # never fail the cutout over colour cleanup
            print(f"colour decontamination skipped: {error}")

    rgba = base.convert("RGBA")
    rgba.putalpha(alpha)
    content, _ = encode_image(rgba, output_format)
    return content


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield
    runtime.model = None
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
    }


@app.post("/v1/images/background-removals")
def background_removal(request: RemoveBackgroundRequest) -> Response:
    """Synchronous cutout: returns the image bytes, cache-hit or not."""
    result = produce_cutout(request)
    content = result.get("content")
    if content is None and result.get("url"):
        # Stored remotely and we were not asked to proxy the bytes back.
        return Response(
            status_code=302,
            headers={"Location": result["url"], "X-Cutout-Cached": "1", "X-Cutout-Url": result["url"]},
        )

    headers = {
        "X-BiRefNet-Model": MODEL_ID,
        "X-Cutout-Cached": "1" if result["cached"] else "0",
    }
    if result.get("url"):
        headers["X-Cutout-Url"] = result["url"]
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

    payload = {
        "status": "done",
        "cached": result["cached"],
        "url": result.get("url"),
        "key": result.get("key"),
        "media_type": result["media_type"],
    }
    if not result.get("url") and result.get("content"):
        # No bucket configured: hand the bytes back inline so the browser can
        # still show something without another endpoint.
        payload["data_url"] = (
            f"data:{result['media_type']};base64," + base64.b64encode(result["content"]).decode()
        )
    _set_job(job_id, **payload)


@app.post("/v1/images/background-removals/jobs")
def enqueue_background_removal(request: RemoveBackgroundRequest) -> dict[str, Any]:
    """Starts a cutout and returns immediately so the client can poll.

    A cache hit is answered inline - no job, no queue, no GPU.
    """
    if CACHE_ENABLED and request.cache and object_store is not None:
        key = object_store.cache_key(
            request.image_url, request.cache_params(), suffix=request.output_format.lower()
        )
        if object_store.exists(key):
            url = object_store.public_url(key)
            if url:
                return {"job_id": None, "status": "done", "cached": True, "url": url, "key": key}

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="queued", cached=False)
    _job_pool.submit(_run_job, job_id, request)
    return {"job_id": job_id, "status": "queued", "cached": False, "poll_after_ms": 700}


@app.get("/v1/images/background-removals/jobs/{job_id}")
def background_removal_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.pop("created", None)
    job.pop("updated", None)
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("BIREFNET_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BIREFNET_PORT", "9094")))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
