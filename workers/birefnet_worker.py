#!/usr/bin/env python3
"""Persistent CUDA BiRefNet worker behind the native C gateway."""

from __future__ import annotations

import argparse
import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import requests
import torch
import torch.nn.functional as functional
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


MODEL_ID = os.getenv("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
DEVICE = os.getenv("BIREFNET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = int(os.getenv("BIREFNET_INPUT_SIZE", "1024"))
MAX_DOWNLOAD_BYTES = int(os.getenv("BIREFNET_MAX_DOWNLOAD_BYTES", str(64 << 20)))


class RemoveBackgroundRequest(BaseModel):
    image_url: str = Field(min_length=1)
    output_format: str = "png"
    foreground_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class Runtime:
    model: Any = None
    transform: Any = None
    dtype: torch.dtype = torch.float16


runtime = Runtime()


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


@torch.inference_mode()
def remove_background(image: Image.Image, threshold: float) -> bytes:
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
    alpha = Image.fromarray((mask.numpy() * 255).astype("uint8"), mode="L")
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    output = io.BytesIO()
    rgba.save(output, format="PNG", optimize=True)
    return output.getvalue()


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
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "input_size": INPUT_SIZE}


@app.post("/v1/images/background-removals")
def background_removal(request: RemoveBackgroundRequest) -> Response:
    if request.output_format.lower() != "png":
        raise HTTPException(400, "output_format must be png")
    image = read_image(request.image_url)
    output = remove_background(image, request.foreground_threshold)
    return Response(
        content=output,
        media_type="image/png",
        headers={
            "X-BiRefNet-Model": MODEL_ID,
            "X-Source-Width": str(image.width),
            "X-Source-Height": str(image.height),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("BIREFNET_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BIREFNET_PORT", "9094")))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
