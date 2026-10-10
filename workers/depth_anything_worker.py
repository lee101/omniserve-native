#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gc
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torch.nn.functional as functional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageOps


MODEL_ID = os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
DEVICE = os.getenv("DEPTH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_DOWNLOAD_BYTES = int(os.getenv("DEPTH_MAX_DOWNLOAD_BYTES", str(64 << 20)))
MAX_PIXELS = int(os.getenv("DEPTH_MAX_PIXELS", str(24 << 20)))
WEBP_QUALITY = int(os.getenv("DEPTH_WEBP_QUALITY", "92"))
PNG_COMPRESS_LEVEL = int(os.getenv("DEPTH_PNG_COMPRESS_LEVEL", "1"))
LOCAL_CONCURRENCY = int(os.getenv("DEPTH_LOCAL_CONCURRENCY", "1"))
OVERFLOW_URL = os.getenv("DEPTH_RUNPOD_URL", "").rstrip("/")
OVERFLOW_KEY = os.getenv("DEPTH_RUNPOD_API_KEY", "")
OVERFLOW_TIMEOUT = float(os.getenv("DEPTH_RUNPOD_TIMEOUT_SECONDS", "120"))
PRICE_CREDITS = int(os.getenv("DEPTH_PRICE_CREDITS", "1"))
IDLE_TIMEOUT_SECONDS = float(os.getenv("DEPTH_IDLE_TIMEOUT", "24"))
log = logging.getLogger("depth_anything_worker")


class DepthRequest(BaseModel):
    image_url: str = Field(min_length=1, max_length=96 << 20)
    output_format: str = "png16"
    invert: bool = True
    preview: bool = True


class Runtime:
    processor: Any = None
    model: Any = None
    dtype: torch.dtype = torch.float32


runtime = Runtime()
local_slots = threading.BoundedSemaphore(max(1, LOCAL_CONCURRENCY))
session = requests.Session()
_last_used = time.monotonic()
_idle_timer: threading.Timer | None = None
_idle_lock = threading.Lock()


def _schedule_unload() -> None:
    global _idle_timer
    if IDLE_TIMEOUT_SECONDS <= 0:
        return
    with _idle_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        timer = threading.Timer(IDLE_TIMEOUT_SECONDS, _maybe_unload)
        timer.daemon = True
        _idle_timer = timer
        timer.start()


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()
    _schedule_unload()


def _maybe_unload() -> None:
    if IDLE_TIMEOUT_SECONDS <= 0:
        return
    if time.monotonic() - _last_used < IDLE_TIMEOUT_SECONDS:
        return
    unload_model()


def unload_model() -> bool:
    if runtime.model is None and runtime.processor is None:
        return False
    runtime.model = None
    runtime.processor = None
    gc.collect()
    try:
        if DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as error:
        log.warning("depth VRAM release failed: %s", error)
        return False
    return True


def ensure_model() -> None:
    if runtime.model is not None and runtime.processor is not None:
        _touch()
        return
    load_model()
    _touch()


def load_model() -> None:
    import transformers
    transformers.utils.import_utils._sklearn_available = False
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    runtime.dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    if DEVICE.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    runtime.processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    runtime.model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID,
        dtype=runtime.dtype,
    ).to(DEVICE).eval()
    if DEVICE.startswith("cuda"):
        runtime.model = runtime.model.to(memory_format=torch.channels_last)
    if os.getenv("DEPTH_TORCH_COMPILE", "0") == "1":
        runtime.model = torch.compile(runtime.model, mode="reduce-overhead", fullgraph=False)


def read_image(value: str) -> Image.Image:
    if value.startswith("data:"):
        try:
            data = base64.b64decode(value.split(",", 1)[1], validate=True)
        except (IndexError, ValueError) as error:
            raise HTTPException(400, "invalid data URL") from error
    else:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise HTTPException(400, "image_url must use http, https, or data")
        try:
            with session.get(value, timeout=(10, 60), stream=True) as response:
                response.raise_for_status()
                chunks = []
                total = 0
                for chunk in response.iter_content(1 << 20):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(413, "source image is too large")
                    chunks.append(chunk)
                data = b"".join(chunks)
        except requests.RequestException as error:
            raise HTTPException(502, f"source image download failed: {error}") from error
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(413, "source image is too large")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as error:
        raise HTTPException(400, "source is not a supported image") from error
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(413, "source image has too many pixels")
    return image


def normalize_depth(depth: np.ndarray, invert: bool) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise HTTPException(500, "model returned no finite depth values")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = np.clip((depth - low) / (high - low), 0, 1).astype(np.float32)
    return normalized if invert else 1.0 - normalized


def data_url(content: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(content).decode()}"


def encode_depth(normalized: np.ndarray, output_format: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if output_format == "png16":
        Image.fromarray(np.rint(normalized * 65535).astype(np.uint16)).save(
            buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL
        )
        return buffer.getvalue(), "image/png"
    if output_format == "png8":
        Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L").save(
            buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL
        )
        return buffer.getvalue(), "image/png"
    if output_format == "webp":
        Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L").save(buffer, "WEBP", quality=WEBP_QUALITY, method=6)
        return buffer.getvalue(), "image/webp"
    raise HTTPException(400, "output_format must be png16, png8, or webp")


def encode_preview(normalized: np.ndarray) -> bytes:
    value = np.rint(normalized * 255).astype(np.uint8)
    rgb = np.stack((value, value, value), axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, "WEBP", quality=WEBP_QUALITY, method=6)
    return buffer.getvalue()


@torch.inference_mode()
def infer(request: DepthRequest) -> dict[str, Any]:
    ensure_model()
    assert runtime.model is not None and runtime.processor is not None
    image = read_image(request.image_url)
    inputs = runtime.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE, dtype=runtime.dtype)
    if DEVICE.startswith("cuda"):
        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=runtime.dtype, enabled=DEVICE.startswith("cuda")):
        prediction = runtime.model(pixel_values=pixel_values).predicted_depth
    prediction = functional.interpolate(
        prediction.unsqueeze(1), size=(image.height, image.width), mode="bicubic", align_corners=False
    )[0, 0].float().cpu().numpy()
    normalized = normalize_depth(prediction, request.invert)
    content, media_type = encode_depth(normalized, request.output_format.lower())
    result = {
        "model": MODEL_ID,
        "license": "apache-2.0",
        "width": image.width,
        "height": image.height,
        "format": request.output_format.lower(),
        "depth_map": data_url(content, media_type),
        "inference_ms": round((time.perf_counter() - started) * 1000, 2),
        "credits": PRICE_CREDITS,
        "worker": "local",
    }
    if request.preview:
        result["preview"] = data_url(encode_preview(normalized), "image/webp")
    return result


def run_overflow(request: DepthRequest) -> dict[str, Any]:
    if not OVERFLOW_URL:
        raise HTTPException(503, "local depth capacity is busy")
    headers = {"content-type": "application/json"}
    if OVERFLOW_KEY:
        headers["authorization"] = f"Bearer {OVERFLOW_KEY}"
    runpod = "api.runpod.ai/v2/" in OVERFLOW_URL
    payload = {"input": request.model_dump()} if runpod else request.model_dump()
    try:
        response = session.post(OVERFLOW_URL, json=payload, headers=headers, timeout=(10, OVERFLOW_TIMEOUT))
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise HTTPException(502, f"RunPod depth overflow failed: {error}") from error
    if runpod and isinstance(result, dict):
        result = result.get("output", result)
    if not isinstance(result, dict) or not result.get("depth_map"):
        raise HTTPException(502, "RunPod depth overflow returned no depth map")
    result["worker"] = "runpod"
    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if _idle_timer is not None:
        _idle_timer.cancel()
    session.close()


app = FastAPI(title="OmniServe Depth Anything V2", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ready": runtime.model is not None, "model": MODEL_ID, "device": DEVICE, "overflow": bool(OVERFLOW_URL), "credits": PRICE_CREDITS, "idle_timeout": IDLE_TIMEOUT_SECONDS}


@app.post("/v1/depth-estimations")
def depth_estimation(request: DepthRequest) -> dict[str, Any]:
    if not local_slots.acquire(blocking=False):
        return run_overflow(request)
    try:
        return infer(request)
    finally:
        local_slots.release()
        _touch()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("DEPTH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DEPTH_PORT", "9099")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
