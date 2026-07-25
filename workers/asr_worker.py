#!/usr/bin/env python3
"""Lazy local ASR worker for CUDA/ROCm or CPU.

This process deliberately owns model residency outside the C gateway. An
administrator can hold it before a background training window, which unloads
the model and makes new inference return 503 until release.
"""

from __future__ import annotations

import gc
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline


MODEL_ID = os.getenv("OMNISERVE_ASR_MODEL", "nvidia/parakeet-ctc-0.6b")
DEVICE_REQUEST = os.getenv("OMNISERVE_ASR_DEVICE", "auto").lower()
VRAM_REQUIRED_GIB = float(os.getenv("OMNISERVE_ASR_VRAM_REQUIRED_GIB", "3"))
LOCK = threading.RLock()
PIPE: Any = None
HELD = False

app = FastAPI(title="OmniServe local ASR worker")


def selected_device() -> str:
    if DEVICE_REQUEST == "cpu":
        return "cpu"
    # PyTorch exposes ROCm devices through the CUDA API as well.
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pipe() -> Any:
    global PIPE
    with LOCK:
        if HELD:
            raise HTTPException(503, "ASR worker held for background training")
        if PIPE is None:
            device = 0 if selected_device() == "cuda" else -1
            dtype = torch.float16 if device == 0 else torch.float32
            PIPE = pipeline(
                "automatic-speech-recognition",
                model=MODEL_ID,
                device=device,
                dtype=dtype,
            )
        return PIPE


def unload() -> None:
    global PIPE
    with LOCK:
        PIPE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transcribe_bytes(data: bytes, suffix: str) -> str:
    if not data:
        raise HTTPException(400, "empty audio")
    # A named file lets soundfile/ffmpeg infer compressed input formats.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        result = get_pipe()(str(path))
    finally:
        path.unlink(missing_ok=True)
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    text = text.strip()
    if not text:
        raise HTTPException(502, "model returned an empty transcript")
    return text


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ready": not HELD,
        "held": HELD,
        "loaded": PIPE is not None,
        "model": MODEL_ID,
        "device": selected_device(),
        "runtime": "rocm" if torch.version.hip else ("cuda" if torch.version.cuda else "cpu"),
        "vram_required_gib": VRAM_REQUIRED_GIB,
    }


@app.post("/admin/hold")
def hold() -> dict[str, Any]:
    global HELD
    with LOCK:
        HELD = True
        unload()
    return {"held": True}


@app.post("/admin/release")
def release() -> dict[str, Any]:
    global HELD
    with LOCK:
        HELD = False
    return {"held": False}


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    return {"text": transcribe_bytes(await file.read(), suffix)}


@app.post("/api/v1/audio/transcribe")
async def raw_transcribe(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")
    suffix = ".wav"
    if "ogg" in content_type:
        suffix = ".ogg"
    elif "webm" in content_type:
        suffix = ".webm"
    elif "mpeg" in content_type or "mp3" in content_type:
        suffix = ".mp3"
    return JSONResponse({"text": transcribe_bytes(await request.body(), suffix)})
