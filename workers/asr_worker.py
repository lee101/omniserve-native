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
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from transformers import pipeline


# Default chosen by measurement, not by reputation: on the nine-clip DictatorFlow
# corpus whisper-small.en scores 1.74% WER at RTF 0.18, while the previous
# default (nvidia/parakeet-ctc-0.6b) scored 100% -- it returns "<unk>" for every
# clip through the transformers ASR pipeline and needs NeMo, not this code path.
# See performance/asr-models-final.json and tools/asr_model_bench.py.
MODEL_ID = os.getenv("OMNISERVE_ASR_MODEL", "openai/whisper-small.en")
# The sample rate the ASR frontend expects. A clip already at this rate, mono
# and 16-bit, can skip the disk round-trip entirely (see decode_pcm16_wav).
TARGET_SAMPLE_RATE = 16000
WARMUP = os.getenv("OMNISERVE_ASR_WARMUP", "0") == "1"
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


def decode_pcm16_wav(data: bytes) -> np.ndarray | None:
    """Decode a mono 16-bit WAV already at TARGET_SAMPLE_RATE to float32 samples.

    Returns None for anything else -- multichannel, another sample rate, another
    bit depth, or a compressed container -- so those keep the tempfile path and
    let soundfile/ffmpeg do the conversion. The point is to skip a disk write and
    re-read for the format the desktop client already sends, without changing a
    single sample: int16 / 32768.0 is exactly soundfile's float conversion, so
    the model sees identical input and WER cannot move.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if (
                w.getnchannels() != 1
                or w.getsampwidth() != 2
                or w.getframerate() != TARGET_SAMPLE_RATE
                or w.getcomptype() != "NONE"
            ):
                return None
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    if not frames:
        return None
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


# Tokens a broken model/pipeline pairing emits instead of words.
_DEGENERATE_TOKENS = {"<unk>", "<pad>", "<s>", "</s>", "\u2047"}


def is_degenerate(text: str) -> bool:
    """True when a transcript is only unknown/special tokens."""
    parts = text.split()
    return bool(parts) and all(p.lower() in _DEGENERATE_TOKENS for p in parts)


def transcribe_bytes(data: bytes, suffix: str) -> str:
    if not data:
        raise HTTPException(400, "empty audio")

    pipe = get_pipe()
    samples = decode_pcm16_wav(data)
    if samples is not None:
        result = pipe({"raw": samples, "sampling_rate": TARGET_SAMPLE_RATE})
    else:
        # A named file lets soundfile/ffmpeg infer compressed input formats.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        try:
            result = pipe(str(path))
        finally:
            path.unlink(missing_ok=True)

    text = result.get("text", "") if isinstance(result, dict) else str(result)
    text = text.strip()
    if not text:
        raise HTTPException(502, "model returned an empty transcript")
    if is_degenerate(text):
        # A model wired to the wrong code path can decode every frame to <unk>
        # and still return 200. That is worse than a crash: it looks like a
        # transcript and scores 100% WER. Surface it instead of serving it.
        raise HTTPException(502, f"model {MODEL_ID} returned a degenerate transcript ({text[:40]!r})")
    return text


async def transcribe_async(data: bytes, suffix: str) -> str:
    """Run inference off the event loop.

    The endpoints are async, so calling transcribe_bytes directly blocked the
    whole ASGI loop for the duration of a transcription: concurrent requests
    queued behind it and health checks stalled. A worker thread keeps the loop
    responsive; the GIL is released inside torch during the actual compute.
    """
    return await run_in_threadpool(transcribe_bytes, data, suffix)


@app.on_event("startup")
def warmup() -> None:
    """Optionally pay the model-load cost at boot instead of on a user's request.

    Off by default so a held/training host does not pull weights into VRAM just
    by starting the process.
    """
    if not WARMUP:
        return
    try:
        pipe = get_pipe()
        silence = np.zeros(TARGET_SAMPLE_RATE // 10, dtype=np.float32)
        pipe({"raw": silence, "sampling_rate": TARGET_SAMPLE_RATE})
    except Exception as exc:  # a failed warmup must not stop the worker booting
        print(f"asr warmup skipped: {exc}", flush=True)


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
    return {"text": await transcribe_async(await file.read(), suffix)}


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
    return JSONResponse({"text": await transcribe_async(await request.body(), suffix)})
