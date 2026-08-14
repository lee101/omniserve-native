#!/usr/bin/env python3
"""RunPod Serverless worker for detail-preserving transparent video matting."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import ipaddress
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import torch

try:
    import runpod
except ModuleNotFoundError:  # Unit tests and local native workers do not need it.
    runpod = None

from workloads.person_detection import VideoPersonDetector


MODEL_BACKBONE = os.getenv("RVM_BACKBONE", "resnet50").strip().lower()
if MODEL_BACKBONE not in {"mobilenetv3", "resnet50"}:
    raise ValueError("RVM_BACKBONE must be mobilenetv3 or resnet50")
MODEL_PATH = Path(os.getenv("RVM_MODEL_PATH", f"/opt/models/rvm_{MODEL_BACKBONE}_fp32.torchscript"))
MAX_INPUT_BYTES = int(os.getenv("MAX_INPUT_BYTES", str(2 << 30)))
MAX_INLINE_BYTES = int(os.getenv("MAX_INLINE_BYTES", str(240 << 20)))
REQUEST_TIMEOUT = (15, 300)
FFMPEG = os.getenv("VIDEO_MATTING_FFMPEG", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("VIDEO_MATTING_FFPROBE", "/usr/bin/ffprobe")
_model = None
_eager_model = None
_model_engine = ""
_compile_error = ""
_person_detector = VideoPersonDetector()
_person_detector.preload_async()


def _cache_root() -> Path:
    requested = Path(os.getenv("VIDEO_BACKGROUND_CACHE", "/runpod-volume/video-background-remover"))
    try:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    except OSError:
        fallback = Path("/tmp/video-background-remover-cache")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("video_url must be a public HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("video_url credentials are not allowed")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as error:
        raise ValueError("video_url host could not be resolved") from error
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("video_url must resolve only to public addresses")


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(".partial")
    size = 0
    try:
        current = url
        response = None
        for _ in range(6):
            _validate_public_url(current)
            response = requests.get(current, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=False)
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise ValueError("video_url redirect has no destination")
                current = urljoin(current, location)
                continue
            break
        else:
            raise ValueError("video_url has too many redirects")
        if response is None:
            raise ValueError("video_url could not be downloaded")
        with response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > MAX_INPUT_BYTES:
                raise ValueError("input video exceeds the 2 GiB limit")
            with partial.open("wb") as output:
                for chunk in response.iter_content(4 << 20):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_INPUT_BYTES:
                        raise ValueError("input video exceeds the 2 GiB limit")
                    output.write(chunk)
        if size == 0:
            raise ValueError("input video is empty")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), None)
    if not stream:
        raise ValueError("input contains no decodable video stream")
    numerator, denominator = (int(value) for value in stream.get("avg_frame_rate", "0/1").split("/"))
    fps = numerator / max(1, denominator)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    if fps <= 0 or duration <= 0:
        raise ValueError("input video has invalid timing metadata")
    if duration > 30.25:
        raise ValueError("input video must be 30 seconds or shorter")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "has_audio": any(item.get("codec_type") == "audio" for item in payload.get("streams", [])),
    }


def _model_instance():
    global _model, _eager_model, _model_engine, _compile_error
    if _model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        source_root = Path(os.getenv("RVM_SOURCE_PATH", "/opt/robust-video-matting"))
        state_path = Path(os.getenv("RVM_STATE_PATH", f"/opt/models/rvm_{MODEL_BACKBONE}.pth"))
        # Eager is the measured production default. Inductor remains opt-in:
        # it adds a substantial cold start and did not improve resident speed
        # for this recurrent, decode-and-encode pipeline on the RTX 5090.
        compile_requested = os.getenv("RVM_TORCH_COMPILE", "0").lower() not in {"0", "false", "no"}
        if source_root.is_dir() and state_path.is_file():
            import sys

            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            from model import MattingNetwork

            eager = MattingNetwork(MODEL_BACKBONE).eval().cuda().half()
            eager.load_state_dict(torch.load(str(state_path), map_location="cpu", weights_only=True))
            _eager_model = eager
            if compile_requested:
                try:
                    _model = torch.compile(
                        eager,
                        mode=os.getenv("RVM_TORCH_COMPILE_MODE", "default"),
                        dynamic=False,
                        fullgraph=False,
                    )
                    _model_engine = f"rvm-{MODEL_BACKBONE}-fp16-torch-compile"
                except Exception as error:
                    _compile_error = str(error)[:1000]
                    _model = eager
                    _model_engine = f"rvm-{MODEL_BACKBONE}-fp16-eager-compile-fallback"
            else:
                _model = eager
                _model_engine = f"rvm-{MODEL_BACKBONE}-fp16-eager"
        else:
            _model = torch.jit.load(str(MODEL_PATH), map_location="cuda").eval()
            _model_engine = f"rvm-{MODEL_BACKBONE}-fp16-torchscript"
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return _model


def _infer(tensor, recurrent, ratio):
    global _model, _compile_error, _model_engine
    model = _model_instance()
    try:
        outputs = model(tensor, *recurrent, ratio)
        if model is not _eager_model:
            outputs = (*outputs[:2], *(state.clone() for state in outputs[2:]))
        return outputs
    except Exception as error:
        # Inductor failures can occur lazily on the first real input. Preserve
        # service by falling back to the already-loaded eager module.
        if _eager_model is None or model is _eager_model:
            raise
        _compile_error = str(error)[:1000]
        _model = _eager_model
        _model_engine = f"rvm-{MODEL_BACKBONE}-fp16-eager-compile-fallback"
        return _model(tensor, *recurrent, ratio)


def release() -> None:
    """Release RVM before OmniServe switches this GPU to another workload."""
    global _model, _eager_model, _model_engine, _compile_error
    _model = None
    _eager_model = None
    _model_engine = ""
    _compile_error = ""
    _person_detector.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _downsample_ratio(width: int, height: int) -> float:
    configured = os.getenv("RVM_DOWNSAMPLE_RATIO", "").strip()
    if configured:
        return max(0.125, min(1.0, float(configured)))
    # Full scale preserves hair and motion edges through 1080p. Larger sources
    # keep their original RGB pixels but infer the smooth alpha field smaller.
    # Match RVM's official auto_downsample_ratio: keep the internal feature
    # extractor's longest side at 512px while foreground/alpha stay at source
    # resolution. Full-resolution features are dramatically slower and are not
    # the model's intended inference setting.
    return min(1.0, 512 / max(width, height))


def _read_exact(pipe, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = pipe.read(size - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def _matte(source: Path, transparent: Path, info: dict, job: dict) -> dict:
    width, height = info["width"], info["height"]
    fps = info["fps"]
    ratio = _downsample_ratio(width, height)
    decoder = subprocess.Popen(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    vp9_deadline = os.getenv("VIDEO_ALPHA_VP9_DEADLINE", "realtime").strip().lower()
    if vp9_deadline not in {"realtime", "good", "best"}:
        raise ValueError("VIDEO_ALPHA_VP9_DEADLINE must be realtime, good, or best")
    vp9_cpu_used = max(0, min(8, int(os.getenv("VIDEO_ALPHA_VP9_CPU_USED", "6"))))
    vp9_threads = max(1, min(32, int(os.getenv("VIDEO_ALPHA_VP9_THREADS", "16"))))
    encoder = subprocess.Popen(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgba", "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}",
            "-i", "pipe:0", "-an", "-c:v", "libvpx-vp9", "-deadline", vp9_deadline,
            "-cpu-used", str(vp9_cpu_used), "-threads", str(vp9_threads), "-row-mt", "1",
            "-tile-columns", "2", "-frame-parallel", "1", "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0", "-crf", "18", "-b:v", "0",
            "-metadata:s:v:0", "alpha_mode=1", "-y", str(transparent),
        ],
        stdin=subprocess.PIPE,
    )
    if decoder.stdout is None or encoder.stdin is None:
        raise RuntimeError("could not open video pipes")
    model = _model_instance()
    recurrent = [None, None, None, None]
    frame_bytes = width * height * 3
    chunk_frames = max(1, min(24, int(os.getenv("RVM_CHUNK_FRAMES", "8"))))
    frames = 0
    model_seconds = 0.0
    started = time.perf_counter()
    try:
        while True:
            raw = _read_exact(decoder.stdout, frame_bytes * chunk_frames)
            if not raw:
                break
            if len(raw) % frame_bytes:
                raise RuntimeError("decoder returned a partial frame")
            count = len(raw) // frame_bytes
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3).copy()
            infer_rgb = rgb
            if count < chunk_frames:
                infer_rgb = np.concatenate((rgb, np.repeat(rgb[-1:], chunk_frames - count, axis=0)))
            tensor = (
                torch.from_numpy(infer_rgb).permute(0, 3, 1, 2)[None]
                .cuda(non_blocking=True).half().div_(255)
            )
            torch.cuda.synchronize()
            model_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _, alpha, *recurrent = _infer(tensor, recurrent, ratio)
            torch.cuda.synchronize()
            model_seconds += time.perf_counter() - model_started
            if alpha.ndim == 4:
                alpha = alpha[:, None]
            alpha_bytes = (alpha[0, :count, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
            rgba = np.concatenate((rgb, alpha_bytes[..., None]), axis=-1)
            encoder.stdin.write(rgba.tobytes())
            frames += count
            if runpod is not None and (frames == count or frames % 30 < count):
                runpod.serverless.progress_update(job, f"Matted {frames} frames")
    finally:
        decoder.stdout.close()
        encoder.stdin.close()
    decoder_status = decoder.wait()
    encoder_status = encoder.wait()
    if decoder_status != 0 or encoder_status != 0 or not transparent.is_file():
        raise RuntimeError(f"video pipeline failed (decode={decoder_status}, encode={encoder_status})")
    torch.cuda.synchronize()
    pipeline_seconds = time.perf_counter() - started
    return {
        "frames": frames,
        "inference_seconds": model_seconds,
        "inference_fps": frames / model_seconds if model_seconds else 0,
        "pipeline_seconds": pipeline_seconds,
        "pipeline_fps": frames / pipeline_seconds if pipeline_seconds else 0,
        "encode_and_io_seconds": max(0.0, pipeline_seconds - model_seconds),
        "vp9_deadline": vp9_deadline,
        "vp9_cpu_used": vp9_cpu_used,
        "downsample_ratio": ratio,
        "source_width": width,
        "source_height": height,
        "chunk_frames": chunk_frames,
        "engine": _model_engine,
        "torch_compile_error": _compile_error,
    }


def _mux_audio(transparent: Path, source: Path, output: Path, preserve_audio: bool, has_audio: bool) -> None:
    if not preserve_audio or not has_audio:
        shutil.copyfile(transparent, output)
        return
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(transparent), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "libopus",
            "-b:a", "128k", "-shortest", "-y", str(output),
        ],
        check=True,
    )


def _validate_vp9_alpha(path: Path) -> dict[str, object]:
    """Fail closed unless libvpx can decode the WebM's separate alpha plane."""
    completed = subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-c:v", "libvpx-vp9",
            "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-vf", "alphaextract",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        diagnostic = (completed.stderr or completed.stdout or "alpha plane is not decodable")[-1000:].strip()
        raise RuntimeError(f"VP9 alpha validation failed: {diagnostic}")
    return {"alpha_validated": True, "alpha_decoder": "libvpx-vp9"}


def _upload(path: Path, upload_url: str) -> None:
    size = path.stat().st_size
    with path.open("rb") as payload:
        response = requests.put(
            upload_url,
            data=payload,
            headers={"Content-Type": "video/webm", "Content-Length": str(size)},
            timeout=(15, 900),
        )
    response.raise_for_status()


def handler(job: dict) -> dict:
    inputs = job.get("input") or {}
    inputs = inputs.get("input", inputs)
    video_url = str(inputs.get("video_url") or "").strip()
    preserve_audio = bool(inputs.get("preserve_audio", True))
    upload_url = str(inputs.get("output_upload_url") or "").strip()
    public_url = str(inputs.get("output_public_url") or "").strip()
    if not video_url:
        raise ValueError("video_url is required")

    with tempfile.TemporaryDirectory(prefix="matte-") as temporary:
        work = Path(temporary)
        source = work / "source.video"
        _download(video_url, source)
        info = _probe(source)
        route_mode = os.getenv("VIDEO_MATTING_ROUTE", "auto").strip().lower()
        if route_mode not in {"auto", "rvm", "standby"}:
            raise ValueError("VIDEO_MATTING_ROUTE must be auto, rvm, or standby")
        if route_mode == "rvm":
            route = {"detected": True, "forced": True, "error": ""}
        elif route_mode == "standby":
            route = {"detected": False, "forced": True, "error": ""}
        else:
            route = _person_detector.detect(source, duration=info["duration"]).public_dict()
            route["forced"] = False
        if not route["detected"]:
            reason = "person detector uncertainty" if route.get("error") else "no person detected"
            return {
                "fallback_required": True,
                "fallback_reason": reason,
                "route": "standby-general-matting",
                "duration_seconds": info["duration"],
                "metrics": {"person_detection": route},
            }
        content_hash = _sha256(source)
        identity = json.dumps(
            {
                "content": content_hash,
                "preserve_audio": preserve_audio,
                "engine": f"rvm-{MODEL_BACKBONE}-chunked-v3",
                "downsample_ratio": _downsample_ratio(info["width"], info["height"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = hashlib.sha256(identity.encode()).hexdigest()
        cache = _cache_root()
        cached = cache / "outputs" / f"{key}.webm"
        lock_path = cache / "locks" / f"{key}.lock"
        cached.parent.mkdir(parents=True, exist_ok=True)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {"person_detection": route}
        cache_hit = False
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cached.is_file() and cached.stat().st_size:
                cache_hit = True
            else:
                transparent = work / "transparent-no-audio.webm"
                metrics.update(_matte(source, transparent, info, job))
                produced = work / "foreground-vp9-alpha.webm"
                _mux_audio(transparent, source, produced, preserve_audio, info["has_audio"])
                partial = cached.with_suffix(".partial")
                shutil.copyfile(produced, partial)
                partial.replace(cached)

        metrics.update(_validate_vp9_alpha(cached))

        if upload_url and public_url:
            _upload(cached, upload_url)
            video_result = public_url
            inline = None
        else:
            if cached.stat().st_size > MAX_INLINE_BYTES:
                raise ValueError("output is too large to return inline; output_upload_url is required")
            video_result = ""
            inline = base64.b64encode(cached.read_bytes()).decode("ascii")
        result = {
            "video_url": video_result,
            "content_type": "video/webm",
            "duration_seconds": info["duration"],
            "bytes": cached.stat().st_size,
            "sha256": _sha256(cached),
            "cache_hit": cache_hit,
            "metrics": metrics,
            "route": "local-rvm-person",
            "fallback_required": False,
        }
        if inline is not None:
            result["outputs"] = [{"filename": cached.name, "content_type": "video/webm", "data": inline}]
        return result
