"""Person-routed, compiled RVM video matting for the OmniServe worker."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import queue
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
import torch

from gpu_admission import AdaptiveCudaGuard


MODEL_SOURCE = Path(os.getenv("RVM_SOURCE_PATH", "/nvme0n1-disk/models/robust-video-matting"))
MODEL_BACKBONE = os.getenv("RVM_BACKBONE", "resnet50").strip().lower()
if MODEL_BACKBONE not in {"mobilenetv3", "resnet50"}:
    raise ValueError("RVM_BACKBONE must be mobilenetv3 or resnet50")
MODEL_STATE = Path(os.getenv("RVM_STATE_PATH", str(MODEL_SOURCE / f"rvm_{MODEL_BACKBONE}.pth")))
MODEL_TORCHSCRIPT = Path(os.getenv(
    "RVM_TORCHSCRIPT_PATH", str(MODEL_SOURCE / f"rvm_{MODEL_BACKBONE}_fp32.torchscript"),
))
TORCH_HOME = Path(os.getenv("TORCH_HOME", "/nvme0n1-disk/models/torch"))
MAX_INPUT_BYTES = int(os.getenv("VIDEO_MATTING_MAX_INPUT_BYTES", str(2 << 30)))
MAX_DURATION = float(os.getenv("VIDEO_MATTING_MAX_DURATION", "30.25"))
RVM_MIN_FREE_MIB = max(256, int(os.getenv("RVM_MIN_FREE_MIB", "1536")))
RVM_OOM_MARGIN_MIB = max(64, int(os.getenv("RVM_OOM_MARGIN_MIB", "512")))
RVM_OOM_BACKOFF_SECONDS = max(0.1, float(os.getenv("RVM_OOM_BACKOFF_SECONDS", "5")))
RVM_OOM_BACKOFF_MAX_SECONDS = max(
    RVM_OOM_BACKOFF_SECONDS,
    float(os.getenv("RVM_OOM_BACKOFF_MAX_SECONDS", "300")),
)
RVM_OOM_RECOVERY_SUCCESSES = max(1, int(os.getenv("RVM_OOM_RECOVERY_SUCCESSES", "8")))
RVM_VRAM_BROKER_URL = os.getenv("RVM_VRAM_BROKER_URL", "http://127.0.0.1:8791").strip().rstrip("/")
RVM_VRAM_BROKER_REQUIRED = os.getenv("RVM_VRAM_BROKER_REQUIRED", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
RVM_VRAM_LEASE_TTL_SECONDS = max(30, int(os.getenv("RVM_VRAM_LEASE_TTL_SECONDS", "1800")))
REQUEST_TIMEOUT = (15, 300)
FFMPEG = os.getenv("VIDEO_MATTING_FFMPEG", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("VIDEO_MATTING_FFPROBE", "/usr/bin/ffprobe")

_load_lock = threading.Lock()
_person_model = None
_person_error = ""
_person_thread = None
_rvm_model = None
_rvm_eager = None
_rvm_engine = ""
_compile_error = ""


_rvm_guard = AdaptiveCudaGuard(
    RVM_MIN_FREE_MIB,
    oom_margin_mib=RVM_OOM_MARGIN_MIB,
    backoff_seconds=RVM_OOM_BACKOFF_SECONDS,
    backoff_max_seconds=RVM_OOM_BACKOFF_MAX_SECONDS,
    recovery_successes=RVM_OOM_RECOVERY_SUCCESSES,
)


def _acquire_vram_lease(required_mib: int) -> dict:
    """Reserve cross-process headroom from the native broker when configured."""
    if not RVM_VRAM_BROKER_URL:
        return {"brokered": False, "granted": True, "mb": 0, "lease_id": ""}
    try:
        response = requests.post(
            f"{RVM_VRAM_BROKER_URL}/v1/gpu/lease",
            json={"owner": "rvm", "mb": required_mib, "min_mb": required_mib,
                  "tier": "background", "ttl_s": RVM_VRAM_LEASE_TTL_SECONDS},
            timeout=(2, 5),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("granted") and payload.get("lease_id"):
            return {"brokered": True, "granted": True, "mb": int(payload.get("mb") or 0),
                    "lease_id": str(payload["lease_id"])}
        return {"brokered": True, "granted": False, "mb": 0, "lease_id": "",
                "reason": str(payload.get("reason") or "no_headroom")}
    except (requests.RequestException, ValueError, TypeError) as error:
        if RVM_VRAM_BROKER_REQUIRED:
            return {"brokered": True, "granted": False, "mb": 0, "lease_id": "",
                    "reason": f"broker_unavailable: {str(error)[:300]}"}
        return {"brokered": False, "granted": True, "mb": 0, "lease_id": "",
                "warning": f"broker_unavailable: {str(error)[:300]}"}


def _release_vram_lease(lease: dict) -> None:
    lease_id = str(lease.get("lease_id") or "")
    if not lease_id or not RVM_VRAM_BROKER_URL:
        return
    try:
        requests.post(
            f"{RVM_VRAM_BROKER_URL}/v1/gpu/release",
            json={"lease_id": lease_id},
            timeout=(2, 5),
        ).raise_for_status()
    except requests.RequestException:
        # The broker TTL is the crash-safe release path.
        pass


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
    current = url
    response = None
    try:
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
        if not size:
            raise ValueError("input video is empty")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, text=True, stdout=subprocess.PIPE,
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
    if duration > MAX_DURATION:
        raise ValueError("input video must be 30 seconds or shorter")
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": fps, "duration": duration,
        "has_audio": any(item.get("codec_type") == "audio" for item in payload.get("streams", [])),
    }


def _load_person_model() -> None:
    global _person_model, _person_error
    if _person_model is not None or _person_error:
        return
    with _load_lock:
        if _person_model is not None or _person_error:
            return
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable to the person detector")
            from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = ssdlite320_mobilenet_v3_large(weights=weights).eval().cuda()
            with torch.inference_mode():
                model([torch.zeros((3, 320, 320), device="cuda")])
            _person_model = model
        except Exception as error:
            _person_error = str(error)[:1000]


def preload_person_detector_async() -> None:
    global _person_thread
    if _person_thread is None and _person_model is None and not _person_error:
        _person_thread = threading.Thread(target=_load_person_model, name="video-person-detector", daemon=True)
        _person_thread.start()


def _sample_frames(video: Path, duration: float, count: int) -> list[np.ndarray]:
    frames = []
    with tempfile.TemporaryDirectory(prefix="person-route-") as folder:
        destination = Path(folder)
        sample_fps = count / max(duration, 0.001)
        completed = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-vf", f"fps={sample_fps:.8f}", "-frames:v", str(count),
             str(destination / "%03d.png")],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError((completed.stderr or "ffmpeg person sample decode failed")[-1000:])
        for path in sorted(destination.glob("*.png")):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is not None:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError("person router decoded no sample frames")
    return frames


def detect_person(video: Path, duration: float) -> dict:
    started = time.perf_counter()
    threshold = min(0.95, max(0.05, float(os.getenv("PERSON_DETECTOR_CONFIDENCE", "0.35"))))
    samples = max(2, min(24, int(os.getenv("PERSON_DETECTOR_SAMPLES", "8"))))
    _load_person_model()
    if _person_model is None:
        return {"detected": False, "error": _person_error or "person detector unavailable",
                "sampled_frames": 0, "frames_with_person": 0, "max_confidence": 0.0,
                "threshold": threshold, "elapsed_seconds": time.perf_counter() - started}
    try:
        frames = _sample_frames(video, duration, samples)
        tensors = [torch.from_numpy(frame.copy()).permute(2, 0, 1).cuda().float().div_(255) for frame in frames]
        with torch.inference_mode():
            outputs = _person_model(tensors)
        confidences = []
        for output in outputs:
            person = output["scores"][output["labels"] == 1]
            confidences.append(float(person.max().item()) if person.numel() else 0.0)
        return {"detected": any(value >= threshold for value in confidences), "error": "",
                "sampled_frames": len(frames),
                "frames_with_person": sum(value >= threshold for value in confidences),
                "max_confidence": max(confidences, default=0.0), "threshold": threshold,
                "elapsed_seconds": time.perf_counter() - started}
    except Exception as error:
        return {"detected": False, "error": str(error)[:1000], "sampled_frames": 0,
                "frames_with_person": 0, "max_confidence": 0.0, "threshold": threshold,
                "elapsed_seconds": time.perf_counter() - started}


def _load_rvm():
    global _rvm_model, _rvm_eager, _rvm_engine, _compile_error
    if _rvm_model is not None:
        return _rvm_model
    with _load_lock:
        if _rvm_model is not None:
            return _rvm_model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to RVM")
        # Eager is the measured production default. Inductor remains opt-in:
        # it adds a substantial cold start and did not improve resident speed
        # for this recurrent, decode-and-encode pipeline on the RTX 5090.
        compile_enabled = os.getenv("RVM_TORCH_COMPILE", "0") == "1"
        if MODEL_SOURCE.is_dir() and MODEL_STATE.is_file():
            if str(MODEL_SOURCE) not in sys.path:
                sys.path.insert(0, str(MODEL_SOURCE))
            from model import MattingNetwork

            eager = MattingNetwork(MODEL_BACKBONE).eval()
            eager.load_state_dict(torch.load(MODEL_STATE, map_location="cpu", weights_only=True))
            eager = eager.cuda().half()
            _rvm_eager = eager
            if compile_enabled:
                _rvm_model = torch.compile(
                    eager,
                    mode=os.getenv("RVM_TORCH_COMPILE_MODE", "default"),
                    dynamic=False,
                    fullgraph=False,
                )
                _rvm_engine = f"rvm-{MODEL_BACKBONE}-fp16-torch-compile"
            else:
                _rvm_model = eager
                _rvm_engine = f"rvm-{MODEL_BACKBONE}-fp16-eager"
        elif MODEL_TORCHSCRIPT.is_file():
            _rvm_model = torch.jit.load(str(MODEL_TORCHSCRIPT), map_location="cuda").eval()
            _rvm_engine = f"rvm-{MODEL_BACKBONE}-torchscript-compat"
        else:
            raise RuntimeError("RVM source/state and TorchScript compatibility model are unavailable")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return _rvm_model


def rvm_capacity() -> dict:
    if not torch.cuda.is_available():
        return {"ready": False, "reason": "CUDA is unavailable", "minimum_free_mib": RVM_MIN_FREE_MIB}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mib = round(free_bytes / (1 << 20))
        total_mib = round(total_bytes / (1 << 20))
        adaptive = _rvm_guard.capacity(free_mib, total_mib)
        capacity = {
            "ready": adaptive["ready"],
            "free_mib": free_mib,
            "total_mib": total_mib,
            "minimum_free_mib": RVM_MIN_FREE_MIB,
            "rvm_loaded": _rvm_model is not None,
            "adaptive": adaptive,
        }
        if not capacity["ready"]:
            capacity["reason"] = (
                "RVM is cooling down after a CUDA OOM"
                if adaptive["cooldown_seconds"] > 0
                else "insufficient adaptive GPU headroom for RVM"
            )
        return capacity
    except Exception as error:  # pragma: no cover - driver failure
        return {"ready": False, "reason": str(error)[:500], "minimum_free_mib": RVM_MIN_FREE_MIB}


def _infer(tensor, recurrent, ratio):
    global _rvm_model, _rvm_engine, _compile_error
    model = _load_rvm()
    try:
        outputs = model(tensor, *recurrent, ratio)
        if model is not _rvm_eager:
            outputs = (*outputs[:2], *(state.clone() for state in outputs[2:]))
        return outputs
    except Exception as error:
        if _rvm_eager is None or model is _rvm_eager:
            raise
        _compile_error = str(error)[:1000]
        _rvm_model = _rvm_eager
        _rvm_engine = f"rvm-{MODEL_BACKBONE}-fp16-eager-compile-fallback"
        return _rvm_model(tensor, *recurrent, ratio)


def _read_exact(pipe, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = pipe.read(size - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def _encoder_writer(encoder, payloads: queue.Queue, errors: list[str]) -> None:
    """Drain RGBA chunks so CUDA inference can overlap VP9 encoding."""
    try:
        while True:
            payload = payloads.get()
            if payload is None:
                return
            if encoder.stdin is None:
                raise RuntimeError("video encoder stdin is unavailable")
            encoder.stdin.write(payload)
    except (BrokenPipeError, OSError, RuntimeError, ValueError) as error:
        errors.append(str(error)[:1000])
    finally:
        if encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass


def _queue_encoder_payload(payloads: queue.Queue, payload: bytes, errors: list[str]) -> None:
    """Put a bounded chunk without hanging forever after an encoder failure."""
    while True:
        if errors:
            raise RuntimeError(f"video encoder pipe failed: {errors[0]}")
        try:
            payloads.put(payload, timeout=0.25)
            return
        except queue.Full:
            continue


def _black_transparent_rgb(rgb: np.ndarray, alpha_bytes: np.ndarray) -> int:
    """Black invisible video RGB and return the number of changed pixels."""
    invisible = alpha_bytes == 0
    if not invisible.any():
        return 0
    rgb[invisible] = 0
    return int(invisible.sum())


def _downsample_ratio(width: int, height: int) -> float:
    configured = os.getenv("RVM_DOWNSAMPLE_RATIO", "").strip()
    if configured:
        return max(0.125, min(1.0, float(configured)))
    # Match RVM's official auto_downsample_ratio: keep the internal feature
    # extractor's longest side at 512px while foreground/alpha stay at source
    # resolution. Full-resolution features are dramatically slower and are not
    # the model's intended inference setting.
    return min(1.0, 512 / max(width, height))


def _matte(source: Path, transparent: Path, info: dict, progress) -> dict:
    width, height, fps = info["width"], info["height"], info["fps"]
    ratio = _downsample_ratio(width, height)
    chunk_frames = max(1, min(24, int(os.getenv("RVM_CHUNK_FRAMES", "8"))))
    decoder = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    vp9_deadline = os.getenv("VIDEO_ALPHA_VP9_DEADLINE", "realtime").strip().lower()
    if vp9_deadline not in {"realtime", "good", "best"}:
        raise ValueError("VIDEO_ALPHA_VP9_DEADLINE must be realtime, good, or best")
    vp9_cpu_used = max(0, min(8, int(os.getenv("VIDEO_ALPHA_VP9_CPU_USED", "8"))))
    vp9_threads = max(1, min(32, int(os.getenv("VIDEO_ALPHA_VP9_THREADS", "8"))))
    vp9_tile_columns = max(0, min(3, int(os.getenv("VIDEO_ALPHA_VP9_TILE_COLUMNS", "1"))))
    encoder = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "pipe:0", "-an",
         "-c:v", "libvpx-vp9", "-deadline", vp9_deadline, "-cpu-used", str(vp9_cpu_used),
         "-threads", str(vp9_threads), "-row-mt", "1", "-tile-columns", str(vp9_tile_columns),
         "-frame-parallel", "1",
         "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-crf", "18", "-b:v", "0",
         "-metadata:s:v:0", "alpha_mode=1", "-y", str(transparent)],
        stdin=subprocess.PIPE,
    )
    if decoder.stdout is None or encoder.stdin is None:
        raise RuntimeError("could not open video pipes")
    encoder_queue_chunks = max(1, min(4, int(os.getenv("VIDEO_ENCODER_QUEUE_CHUNKS", "2"))))
    encoder_payloads: queue.Queue = queue.Queue(maxsize=encoder_queue_chunks)
    encoder_errors: list[str] = []
    encoder_thread = threading.Thread(
        target=_encoder_writer,
        args=(encoder, encoder_payloads, encoder_errors),
        name="video-alpha-encoder",
        daemon=True,
    )
    encoder_thread.start()
    _load_rvm()
    recurrent = [None, None, None, None]
    frame_bytes = width * height * 3
    frames = 0
    transparent_pixels = 0
    model_seconds = 0.0
    synchronization = "cuda-events+d2h"
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
            tensor = torch.from_numpy(infer_rgb).permute(0, 3, 1, 2)[None].cuda().half().div_(255)
            model_started = torch.cuda.Event(enable_timing=True)
            model_finished = torch.cuda.Event(enable_timing=True)
            model_started.record()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _, alpha, *recurrent = _infer(tensor, recurrent, ratio)
            model_finished.record()
            if alpha.ndim == 4:
                alpha = alpha[:, None]
            # This copy is the real completion boundary: the encoder cannot use
            # alpha before it reaches the host.  Once it returns, both events on
            # the same stream are complete, so elapsed_time is valid without a
            # device-wide synchronize that stalls unrelated model streams.
            alpha_bytes = (alpha[0, :count, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
            model_seconds += model_started.elapsed_time(model_finished) / 1000.0
            # RGB under alpha=0 cannot be displayed. Making it uniform reduces
            # entropy in both the VP9 colour plane and its tiles.
            transparent_pixels += _black_transparent_rgb(rgb, alpha_bytes)
            _queue_encoder_payload(
                encoder_payloads,
                np.concatenate((rgb, alpha_bytes[..., None]), axis=-1).tobytes(),
                encoder_errors,
            )
            frames += count
            progress(frames)
    finally:
        decoder.stdout.close()
        if not encoder_errors:
            while True:
                try:
                    encoder_payloads.put(None, timeout=0.25)
                    break
                except queue.Full:
                    if encoder_errors:
                        break
        encoder_thread.join(timeout=30)
        if encoder_thread.is_alive() and encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            encoder_thread.join(timeout=5)
    decode_status, encode_status = decoder.wait(), encoder.wait()
    if decode_status or encode_status or encoder_errors or not transparent.is_file():
        detail = f", encoder_pipe={encoder_errors[0]}" if encoder_errors else ""
        raise RuntimeError(f"video pipeline failed (decode={decode_status}, encode={encode_status}{detail})")
    pipeline_seconds = time.perf_counter() - started
    return {"frames": frames, "inference_seconds": model_seconds,
            "inference_fps": frames / model_seconds if model_seconds else 0.0,
            "pipeline_seconds": pipeline_seconds,
            "pipeline_fps": frames / pipeline_seconds if pipeline_seconds else 0.0,
            "encode_and_io_seconds": max(0.0, pipeline_seconds - model_seconds),
            "vp9_deadline": vp9_deadline, "vp9_cpu_used": vp9_cpu_used,
            "vp9_threads": vp9_threads, "vp9_tile_columns": vp9_tile_columns,
            "encoder_queue_chunks": encoder_queue_chunks,
            "transparent_pixels": transparent_pixels,
            "synchronization": synchronization,
            "downsample_ratio": ratio, "chunk_frames": chunk_frames,
            "source_width": width, "source_height": height, "engine": _rvm_engine,
            "torch_compile_error": _compile_error}


def _mux_audio(transparent: Path, source: Path, output: Path, preserve: bool, has_audio: bool) -> None:
    if not preserve or not has_audio:
        shutil.copyfile(transparent, output)
        return
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(transparent), "-i", str(source),
         "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "libopus", "-b:a", "128k",
         "-shortest", "-y", str(output)], check=True,
    )


def _validate_vp9_alpha(path: Path) -> dict[str, object]:
    """Fail closed unless libvpx can decode the WebM's separate alpha plane."""
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-c:v", "libvpx-vp9",
         "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-vf", "alphaextract",
         "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        diagnostic = (completed.stderr or completed.stdout or "alpha plane is not decodable")[-1000:].strip()
        raise RuntimeError(f"VP9 alpha validation failed: {diagnostic}")
    return {"alpha_validated": True, "alpha_decoder": "libvpx-vp9"}


def _upload(path: Path, upload_url: str) -> None:
    with path.open("rb") as payload:
        response = requests.put(upload_url, data=payload,
                                headers={"Content-Type": "video/webm", "Content-Length": str(path.stat().st_size)},
                                timeout=(15, 900))
    response.raise_for_status()


def process(request: dict, progress=lambda _frames: None) -> dict:
    video_url = str(request.get("video_url") or "").strip()
    if not video_url:
        raise ValueError("video_url is required")
    route_override = str(request.get("route_override") or "auto").strip().lower()
    if route_override not in {"auto", "rvm", "standby"}:
        raise ValueError("route_override must be auto, rvm, or standby")
    with tempfile.TemporaryDirectory(prefix="omniserve-video-matte-") as folder:
        work = Path(folder)
        source = work / "source.video"
        _download(video_url, source)
        info = _probe(source)
        if route_override == "rvm":
            decision = {"detected": True, "forced": True, "error": "", "sampled_frames": 0,
                        "frames_with_person": 0, "max_confidence": 0.0, "threshold": 0.0,
                        "elapsed_seconds": 0.0}
        elif route_override == "standby":
            decision = {"detected": False, "forced": True, "error": "", "sampled_frames": 0,
                        "frames_with_person": 0, "max_confidence": 0.0, "threshold": 0.0,
                        "elapsed_seconds": 0.0}
        else:
            decision = detect_person(source, info["duration"])
            decision["forced"] = False
        if not decision["detected"]:
            return {"fallback_required": True,
                    "fallback_reason": "person detector uncertainty" if decision.get("error") else "no person detected",
                    "route": "standby-general-matting", "duration_seconds": info["duration"],
                    "metrics": {"person_detection": decision}}
        capacity = rvm_capacity()
        if not capacity["ready"]:
            return {"fallback_required": True,
                    "fallback_reason": capacity.get("reason", "local GPU is at capacity"),
                    "route": "standby-gpu-pressure", "duration_seconds": info["duration"],
                    "metrics": {"person_detection": decision, "rvm_capacity": capacity}}
        lease = _acquire_vram_lease(capacity["adaptive"]["required_free_mib"])
        if not lease["granted"]:
            return {"fallback_required": True,
                    "fallback_reason": f"VRAM broker denied RVM: {lease.get('reason', 'no headroom')}",
                    "route": "standby-gpu-pressure", "duration_seconds": info["duration"],
                    "metrics": {"person_detection": decision, "rvm_capacity": capacity,
                                "vram_lease": lease}}
        transparent = work / "transparent-no-audio.webm"
        metrics = {"person_detection": decision, "rvm_capacity": capacity, "vram_lease": lease}
        try:
            try:
                metrics.update(_matte(source, transparent, info, progress))
            except torch.cuda.OutOfMemoryError:
                _rvm_guard.note_oom(capacity["free_mib"], capacity["total_mib"])
                _release_rvm()
                metrics["rvm_capacity_after_oom"] = rvm_capacity()
                return {"fallback_required": True,
                        "fallback_reason": "local GPU ran out of memory; adaptive RVM backoff engaged",
                        "route": "standby-gpu-pressure", "duration_seconds": info["duration"],
                        "metrics": metrics}
        finally:
            _release_vram_lease(lease)
        _rvm_guard.note_success()
        output = work / "foreground-vp9-alpha.webm"
        _mux_audio(transparent, source, output, bool(request.get("preserve_audio", True)), info["has_audio"])
        metrics.update(_validate_vp9_alpha(output))
        upload_url = str(request.get("output_upload_url") or "").strip()
        public_url = str(request.get("output_public_url") or "").strip()
        if not upload_url or not public_url:
            raise ValueError("output_upload_url and output_public_url are required")
        _upload(output, upload_url)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return {"fallback_required": False, "route": "local-rvm-person", "video_url": public_url,
                "content_type": "video/webm", "duration_seconds": info["duration"],
                "bytes": output.stat().st_size, "sha256": digest, "metrics": metrics}


def health() -> dict:
    return {"person_detector": "ready" if _person_model is not None else ("error" if _person_error else "loading"),
            "person_detector_error": _person_error, "rvm_loaded": _rvm_model is not None,
            "rvm_engine": _rvm_engine, "torch_compile_error": _compile_error,
            "rvm_capacity": rvm_capacity()}


def _release_rvm() -> None:
    global _rvm_model, _rvm_eager, _rvm_engine
    with _load_lock:
        _rvm_model = None
        _rvm_eager = None
        _rvm_engine = ""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release() -> None:
    global _person_model, _person_error, _person_thread
    _release_rvm()
    with _load_lock:
        _person_model = None
        _person_error = ""
        _person_thread = None
