#!/usr/bin/env python3
"""Wan-Animate-2 character animation on the shared OmniServe GPU runtime."""

from __future__ import annotations

import gc
import ipaddress
import os
from pathlib import Path
import random
import socket
import subprocess
import tempfile
import time
from urllib.parse import urljoin, urlparse

import requests


DISTILLED_MODEL = os.getenv(
    "WAN_ANIMATE_DISTILLED_MODEL",
    "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
).strip()
MAX_VIDEO_BYTES = int(os.getenv("WAN_ANIMATE_MAX_VIDEO_BYTES", str(512 << 20)))
MAX_IMAGE_BYTES = int(os.getenv("WAN_ANIMATE_MAX_IMAGE_BYTES", str(32 << 20)))
MAX_SECONDS = float(os.getenv("WAN_ANIMATE_MAX_SECONDS", "8"))
REQUEST_TIMEOUT = (15, 300)
FFMPEG = os.getenv("WAN_ANIMATE_FFMPEG", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("WAN_ANIMATE_FFPROBE", "/usr/bin/ffprobe")

_pipeline = None
_pipeline_profile = ""
_compiled = False


def _inputs(job: dict) -> dict:
    values = job.get("input") or {}
    return values.get("input", values) if isinstance(values, dict) else {}


def _public_url(value: object, field: str) -> str:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be a public HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} credentials are not allowed")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as error:
        raise ValueError(f"{field} host could not be resolved") from error
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError(f"{field} must resolve only to public addresses")
    return raw


def _download(url: str, destination: Path, maximum: int, field: str) -> None:
    current = url
    response = None
    for _ in range(6):
        current = _public_url(current, field)
        response = requests.get(current, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise ValueError(f"{field} redirect has no destination")
            current = urljoin(current, location)
            continue
        break
    else:
        raise ValueError(f"{field} has too many redirects")
    if response is None:
        raise ValueError(f"{field} could not be downloaded")
    total = 0
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with response, partial.open("wb") as output:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > maximum:
                raise ValueError(f"{field} is too large")
            for chunk in response.iter_content(4 << 20):
                if not chunk:
                    continue
                total += len(chunk)
                if total > maximum:
                    raise ValueError(f"{field} is too large")
                output.write(chunk)
        if total == 0:
            raise ValueError(f"{field} is empty")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _number(values: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(values.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a number") from error
    if result < minimum or result > maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")
    return result


def _integer(values: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = _number(values, key, default, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _dimensions(values: dict) -> tuple[int, int]:
    width = _integer(values, "width", 640, 320, 1280)
    height = _integer(values, "height", 800, 320, 1280)
    if width * height > 921_600:
        raise ValueError("width times height must not exceed 921600 pixels")
    if width % 16 or height % 16:
        raise ValueError("width and height must be divisible by 16")
    return width, height


def _normalize_video(source: Path, destination: Path, seconds: float) -> None:
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-t", f"{seconds:.3f}", "-an", "-vf", "fps=24", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        diagnostic = completed.stderr.strip().splitlines()[-1:] or ["unknown decode error"]
        raise ValueError("driving_video_url could not be decoded: " + diagnostic[0][:240])


def _duration(path: Path) -> float:
    completed = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return 0.0


def _load_pipeline(profile: str):
    global _pipeline, _pipeline_profile, _compiled
    if _pipeline is not None and _pipeline_profile == profile:
        return _pipeline
    if _pipeline is not None:
        release()

    import torch
    from diffusers import WanAnimate2Pipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # All automatic lanes use the same distilled weights so changing GPU size
    # changes latency, not the generation objective. The 40-step base model is
    # deliberately not an implicit "faster GPU" switch.
    model_id = DISTILLED_MODEL
    arguments = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "use_safetensors": True}
    if profile != "throughput":
        from diffusers.quantizers import PipelineQuantizationConfig

        arguments["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["transformer"],
        )
    pipeline = WanAnimate2Pipeline.from_pretrained(model_id, **arguments)
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    if profile == "small":
        pipeline.enable_sequential_cpu_offload()
    elif profile == "balanced":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
        compile_enabled = os.getenv("WAN_ANIMATE_TORCH_COMPILE", "1").lower() not in {"0", "false", "no"}
        if compile_enabled:
            pipeline.transformer = torch.compile(
                pipeline.transformer,
                mode=os.getenv("WAN_ANIMATE_COMPILE_MODE", "max-autotune-no-cudagraphs"),
                fullgraph=False,
            )
            _compiled = True
    _pipeline = pipeline
    _pipeline_profile = profile
    return pipeline


def _upload(path: Path, upload_url: str) -> None:
    parsed = urlparse(upload_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("output_upload_url must be an HTTPS presigned URL")
    with path.open("rb") as source:
        response = requests.put(
            upload_url,
            data=source,
            headers={"Content-Type": "video/mp4", "Content-Length": str(path.stat().st_size)},
            timeout=REQUEST_TIMEOUT,
        )
    if response.status_code >= 300:
        raise RuntimeError(f"output upload returned {response.status_code}")


def handler(job: dict) -> dict:
    values = _inputs(job)
    image_url = _public_url(values.get("character_image_url") or values.get("image_url"), "character_image_url")
    video_url = _public_url(values.get("driving_video_url") or values.get("video_url"), "driving_video_url")
    prompt = str(values.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required and should describe the character appearance and background")
    if len(prompt) > 1600:
        raise ValueError("prompt must be 1600 characters or fewer")
    profile = str(values.get("_omniserve_profile") or "small").strip().lower()
    if profile not in {"small", "balanced", "throughput"}:
        raise ValueError("unsupported OmniServe execution profile")
    width, height = _dimensions(values)
    requested_seconds = _number(values, "duration", min(5.0, MAX_SECONDS), 1.0, MAX_SECONDS)
    steps_default = 10
    steps = _integer(values, "num_inference_steps", steps_default, 4, 50)
    guidance = _number(values, "guidance_scale", 1.0, 1.0, 1.0)
    seed = _integer(values, "seed", random.randint(0, 2**31 - 1), 0, 2**31 - 1)
    upload_url = str(values.get("output_upload_url") or "").strip()
    public_url = str(values.get("output_public_url") or "").strip()
    if not upload_url or not public_url:
        raise ValueError("output_upload_url and output_public_url are required")
    public_url = _public_url(public_url, "output_public_url")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wan-animate-2-") as directory:
        work = Path(directory)
        image_path = work / "character.png"
        source_path = work / "driving-input"
        driving_path = work / "driving.mp4"
        output_path = work / "output.mp4"
        _download(image_url, image_path, MAX_IMAGE_BYTES, "character_image_url")
        _download(video_url, source_path, MAX_VIDEO_BYTES, "driving_video_url")
        source_seconds = _duration(source_path)
        if source_seconds <= 0:
            raise ValueError("driving_video_url has no readable duration")
        actual_seconds = min(requested_seconds, source_seconds, MAX_SECONDS)
        _normalize_video(source_path, driving_path, actual_seconds)

        from diffusers.utils import export_to_video, load_image

        pipeline = _load_pipeline(profile)
        inference_started = time.monotonic()
        output = pipeline(
            image=load_image(str(image_path)),
            driving_video=str(driving_path),
            prompt=prompt,
            height=height,
            width=width,
            fps=24,
            num_inference_steps=steps,
            guidance_scale=guidance,
            flow_solver="euler",
            seed=seed,
        )
        inference_seconds = time.monotonic() - inference_started
        export_to_video(output.frames[0], str(output_path), fps=24)
        _upload(output_path, upload_url)
        return {
            "video_url": public_url,
            "duration_seconds": actual_seconds,
            "content_type": "video/mp4",
            "seed": seed,
            "metrics": {
                "model": DISTILLED_MODEL,
                "execution_profile": profile,
                "quantization": "bf16" if profile == "throughput" else "nf4-transformer",
                "cpu_offload": profile != "throughput",
                "torch_compile": _compiled and profile == "throughput",
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "inference_seconds": round(inference_seconds, 3),
                "total_seconds": round(time.monotonic() - started, 3),
            },
        }


def release() -> None:
    global _pipeline, _pipeline_profile, _compiled
    _pipeline = None
    _pipeline_profile = ""
    _compiled = False
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
    except (ImportError, RuntimeError):
        pass
