#!/usr/bin/env python3
"""RunPod adapter for accelerated MiniMax-Music3 generation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import wave

import numpy as np

try:
    import runpod
except ModuleNotFoundError:  # Unit tests exercise validation and metrics locally.
    runpod = None


MODEL_ID = os.getenv("MUSIC3_MODEL_ID", "MiniMaxAI/MiniMax-Music3")
MODEL_DIR = Path(os.getenv("MUSIC3_MODEL_DIR", "/runpod-volume/models/minimax-music3"))
FLOWMATCHING_FILENAME = "flowmatching_vae.pth"
FLOWMATCHING_BYTES = 9_828_468_476
FLOWMATCHING_SHA256 = "941f3ed9591684679e733d184308be89949abeb1b069a6e17e69a013ecec08fe"
PORT = int(os.getenv("MUSIC3_PORT", "8000"))
STARTUP_TIMEOUT = int(os.getenv("MUSIC3_STARTUP_TIMEOUT_SECONDS", "1800"))
REQUEST_TIMEOUT = int(os.getenv("MUSIC3_REQUEST_TIMEOUT_SECONDS", "1800"))
MAX_INLINE_BYTES = int(os.getenv("MUSIC3_MAX_INLINE_BYTES", str(8 << 20)))
MAX_DURATION_SECONDS = int(os.getenv("MUSIC3_MAX_DURATION_SECONDS", "360"))
_server_lock = threading.Lock()
_server: subprocess.Popen | None = None
_server_started_at = 0.0
_server_log_path = Path(os.getenv("MUSIC3_SERVER_LOG", "/runpod-volume/omniserve/music3/server.log"))


def _input(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("input") or {}
    if isinstance(value, dict) and isinstance(value.get("input"), dict):
        value = value["input"]
    if not isinstance(value, dict):
        raise ValueError("input must be an object")
    return value


def _positive_int(value: Any, *, field: str, default: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return parsed


def normalize_request(job: dict[str, Any]) -> dict[str, Any]:
    value = _input(job)
    caption = str(value.get("instructions") or value.get("caption") or value.get("prompt") or "").strip()
    if not caption:
        raise ValueError("instructions or prompt is required")
    lyrics = str(value.get("lyrics") or value.get("input") or "").strip()
    if not lyrics:
        lyrics = "[Intro]\n(instrumental)\n[Outro]\n(instrumental)"
        if "instrumental" not in caption.lower():
            caption += ", instrumental, no vocals"
    duration = _positive_int(
        value.get("duration_seconds", value.get("duration")),
        field="duration_seconds",
        default=30,
        maximum=MAX_DURATION_SECONDS,
    )
    frames = _positive_int(
        value.get("max_new_tokens", duration * 25),
        field="max_new_tokens",
        default=duration * 25,
        maximum=9000,
    )
    seed = value.get("seed", 0)
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("seed must be a non-negative integer") from error
    if seed < 0 or seed >= 1 << 63:
        raise ValueError("seed must be a non-negative 63-bit integer")
    upload_url = str(value.get("output_upload_url") or "").strip()
    public_url = str(value.get("output_public_url") or "").strip()
    if upload_url and not upload_url.startswith("https://"):
        raise ValueError("output_upload_url must use https")
    if public_url and not public_url.startswith("https://"):
        raise ValueError("output_public_url must use https")
    return {
        "lyrics": lyrics,
        "instructions": caption,
        "duration_seconds": duration,
        "max_new_tokens": frames,
        "seed": seed,
        "output_upload_url": upload_url,
        "output_public_url": public_url,
    }


def _download_model() -> float:
    started = time.monotonic()
    marker = MODEL_DIR / ".omniserve-ready-v3"
    if marker.exists():
        return 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download, snapshot_download

    flowmatching = MODEL_DIR / FLOWMATCHING_FILENAME
    valid_flowmatching = flowmatching.is_file() and flowmatching.stat().st_size == FLOWMATCHING_BYTES
    if valid_flowmatching:
        digest = hashlib.sha256()
        with flowmatching.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
        valid_flowmatching = digest.hexdigest() == FLOWMATCHING_SHA256
    if not valid_flowmatching:
        flowmatching.unlink(missing_ok=True)
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=FLOWMATCHING_FILENAME,
            local_dir=str(MODEL_DIR),
            force_download=True,
        )
        digest = hashlib.sha256()
        with flowmatching.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
        if (
            flowmatching.stat().st_size != FLOWMATCHING_BYTES
            or digest.hexdigest() != FLOWMATCHING_SHA256
        ):
            flowmatching.unlink(missing_ok=True)
            raise RuntimeError("MiniMax-Music3 flow-matching checkpoint failed integrity verification")

    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    marker.write_text(
        json.dumps({
            "model": MODEL_ID,
            "completed_at": time.time(),
            "flowmatching_sha256": FLOWMATCHING_SHA256,
        }),
        encoding="utf-8",
    )
    return time.monotonic() - started


def _health() -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
            return response.status < 300
    except (HTTPError, URLError, TimeoutError):
        return False


def _start_server() -> dict[str, float]:
    global _server, _server_started_at
    if _server is not None and _server.poll() is None and _health():
        return {"model_download_seconds": 0.0, "server_start_seconds": 0.0}
    with _server_lock:
        if _server is not None and _server.poll() is None and _health():
            return {"model_download_seconds": 0.0, "server_start_seconds": 0.0}
        model_download = _download_model()
        started = time.monotonic()
        Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/music3-torchinductor")).mkdir(parents=True, exist_ok=True)
        command = [
            "sgl-omni", "serve", "--model-path", str(MODEL_DIR),
            "--host", "127.0.0.1", "--port", str(PORT),
            "--max-running-requests", os.getenv("MUSIC3_MAX_RUNNING_REQUESTS", "1"),
            "--stages.dit_dav.factory-args.dtype", os.getenv("MUSIC3_ACOUSTIC_DTYPE", "bfloat16"),
        ]
        _server_log_path.parent.mkdir(parents=True, exist_ok=True)
        server_log = _server_log_path.open("ab", buffering=0)
        server_log.write(f"\n--- Music3 server start {time.time()} ---\n".encode())
        _server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        deadline = started + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            return_code = _server.poll()
            if return_code is not None:
                server_log.close()
                try:
                    detail = _server_log_path.read_bytes()[-32768:].decode("utf-8", "replace")
                except OSError:
                    detail = "server log unavailable"
                raise RuntimeError(
                    f"MiniMax-Music3 server exited with status {return_code}:\n{detail}"
                )
            if _health():
                _server_started_at = time.time()
                server_log.close()
                return {
                    "model_download_seconds": round(model_download, 3),
                    "server_start_seconds": round(time.monotonic() - started, 3),
                }
            time.sleep(2)
        _server.terminate()
        server_log.close()
        raise TimeoutError(f"MiniMax-Music3 server did not become ready in {STARTUP_TIMEOUT}s")


def wav_statistics(audio: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(audio), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frame_count = source.getnframes()
        samples = source.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"expected 16-bit WAV, got {sample_width * 8}-bit")
    values = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0
    if not values.size or channels < 1:
        raise ValueError("generated WAV contains no samples")
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    peak_dbfs = 20 * math.log10(max(peak, 1e-12))
    clipped = int(np.count_nonzero(np.abs(values) >= 32767 / 32768))
    silence = int(np.count_nonzero(np.abs(values) < 10 ** (-60 / 20)))
    reshaped = values.reshape(-1, channels)
    stereo_correlation = None
    if channels == 2 and reshaped.shape[0] > 1:
        left, right = reshaped[:, 0], reshaped[:, 1]
        if float(left.std()) > 0 and float(right.std()) > 0:
            stereo_correlation = float(np.corrcoef(left, right)[0, 1])
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bit_depth": sample_width * 8,
        "frames": frame_count,
        "duration_seconds": round(frame_count / sample_rate, 3),
        "bytes": len(audio),
        "sha256": hashlib.sha256(audio).hexdigest(),
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_dbfs, 3),
        "crest_factor_db": round(peak_dbfs - rms_dbfs, 3),
        "dc_offset": round(float(np.mean(values)), 7),
        "clipped_samples": clipped,
        "clipped_percent": round(100 * clipped / values.size, 6),
        "digital_silence_percent": round(100 * silence / values.size, 4),
        "stereo_correlation": None if stereo_correlation is None else round(stereo_correlation, 4),
    }


def _generate(request: dict[str, Any]) -> tuple[bytes, float]:
    payload = json.dumps({
        "model": MODEL_ID,
        "input": request["lyrics"],
        "instructions": request["instructions"],
        "response_format": "wav",
        "seed": request["seed"],
        "max_new_tokens": request["max_new_tokens"],
        "stream": False,
    }).encode()
    started = time.monotonic()
    http_request = Request(
        f"http://127.0.0.1:{PORT}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=REQUEST_TIMEOUT) as response:
            audio = response.read(MAX_INLINE_BYTES * 16 + 1)
    except HTTPError as error:
        detail = error.read(8192).decode("utf-8", "replace")
        raise RuntimeError(f"MiniMax-Music3 returned {error.code}: {detail}") from error
    if not audio:
        raise RuntimeError("MiniMax-Music3 returned empty audio")
    return audio, time.monotonic() - started


def _upload(url: str, audio: bytes) -> float:
    started = time.monotonic()
    request = Request(url, data=audio, headers={"Content-Type": "audio/wav"}, method="PUT")
    with urlopen(request, timeout=300) as response:
        if response.status >= 300:
            raise RuntimeError(f"audio upload returned {response.status}")
    return time.monotonic() - started


def handler(job: dict[str, Any]) -> dict[str, Any]:
    request = normalize_request(job)
    total_started = time.monotonic()
    startup = _start_server()
    audio, generation_seconds = _generate(request)
    stats = wav_statistics(audio)
    upload_seconds = 0.0
    result: dict[str, Any] = {
        "route": "minimax-music3-local",
        "model": MODEL_ID,
        "seed": request["seed"],
        "requested_frames": request["max_new_tokens"],
        "metrics": {
            **stats,
            **startup,
            "generation_seconds": round(generation_seconds, 3),
            "realtime_factor": round(generation_seconds / max(float(stats["duration_seconds"]), 0.001), 3),
            "audio_seconds_per_compute_second": round(float(stats["duration_seconds"]) / max(generation_seconds, 0.001), 3),
            "server_started_at": _server_started_at,
            "optimizations": [
                "backbone-cuda-graph", "rvq-depth-cuda-graph",
                "compiled-dit-blocks", "compiled-dav-decoder", "batched-seeded-sampling",
            ],
        },
    }
    if request["output_upload_url"]:
        upload_seconds = _upload(request["output_upload_url"], audio)
        result["audio_url"] = request["output_public_url"]
    else:
        if len(audio) > MAX_INLINE_BYTES:
            raise ValueError("output_upload_url is required for audio larger than inline limit")
        result["outputs"] = [{
            "filename": "minimax-music3.wav",
            "content_type": "audio/wav",
            "data": base64.b64encode(audio).decode("ascii"),
        }]
    result["metrics"]["upload_seconds"] = round(upload_seconds, 3)
    result["metrics"]["total_seconds"] = round(time.monotonic() - total_started, 3)
    return result


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("runpod is required for the serverless worker")
    runpod.serverless.start({"handler": handler})
