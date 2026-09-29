from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from workloads import video_matting as vm

_matanyone_net = None
ENGINE = "matanyone-v1"


def release() -> None:
    global _matanyone_net
    _matanyone_net = None


def matanyone_source() -> Path:
    configured = os.getenv("MATANYONE_SOURCE_PATH", "").strip()
    for item in (configured, "/opt/matanyone", "/nvme0n1-disk/code/MatAnyone"):
        if item and (Path(item) / "matanyone").is_dir():
            return Path(item)
    raise RuntimeError("max_quality requires MatAnyone; set MATANYONE_SOURCE_PATH")


def coarse_subject_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    border = np.concatenate(
        (
            rgb[0].reshape(-1, 3),
            rgb[-1].reshape(-1, 3),
            rgb[:, 0].reshape(-1, 3),
            rgb[:, -1].reshape(-1, 3),
        )
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    mask = (distance > 18).astype(np.uint8) * 255
    coverage = float(mask.mean()) / 255.0
    if coverage < 0.05 or coverage > 0.97:
        rows, cols = np.ogrid[:height, :width]
        mask = (
            ((cols - width / 2) / max(1.0, 0.38 * width)) ** 2
            + ((rows - height / 2) / max(1.0, 0.46 * height)) ** 2
            <= 1
        ).astype(np.uint8) * 255
    return mask


def _morph_mask(mask: np.ndarray, radius: int, dilate: bool) -> np.ndarray:
    if radius <= 0:
        return mask
    try:
        import cv2
    except Exception:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    if dilate:
        flags = np.not_equal(mask, 0).astype(np.uint8)
        return cv2.dilate(flags, kernel, iterations=1) * 255
    flags = np.equal(mask, 255).astype(np.uint8)
    return cv2.erode(flags, kernel, iterations=1) * 255


def _extract_first_frame(source: Path, dest: Path) -> None:
    subprocess.run(
        [vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(source), "-frames:v", "1", "-update", "1", "-y", str(dest)],
        check=True,
    )


def _png_to_rgb(path: Path) -> np.ndarray:
    completed = subprocess.run(
        [vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True,
        stdout=subprocess.PIPE,
    )
    probe = subprocess.run(
        [vm.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    width, height = (int(value) for value in probe.stdout.strip().split("x"))
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _write_gray_png(mask: np.ndarray, dest: Path) -> None:
    height, width = mask.shape[:2]
    subprocess.run(
        [
            vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s:v", f"{width}x{height}", "-i", "pipe:0", "-frames:v", "1", "-update", "1", "-y", str(dest),
        ],
        input=np.ascontiguousarray(mask).tobytes(),
        check=True,
    )


def prepare_mask(source: Path, mask_url: str, work: Path, info: dict) -> Path:
    width, height = info["width"], info["height"]
    dest = work / "first-mask.png"
    first = work / "first.png"
    _extract_first_frame(source, first)
    if mask_url:
        downloaded = work / "mask-source"
        vm._download(mask_url, downloaded)
        extracted = work / "mask-alpha.png"
        alpha = subprocess.run(
            [
                vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(downloaded),
                "-vf", f"alphaextract,scale={width}:{height}:flags=neighbor",
                "-frames:v", "1", "-update", "1", "-y", str(extracted),
            ],
            capture_output=True,
        )
        if alpha.returncode == 0 and extracted.is_file() and extracted.stat().st_size:
            mask_image = extracted
        else:
            subprocess.run(
                [
                    vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(downloaded),
                    "-vf", f"format=gray,scale={width}:{height}:flags=neighbor",
                    "-frames:v", "1", "-update", "1", "-y", str(extracted),
                ],
                check=True,
            )
            mask_image = extracted
        raw = subprocess.run(
            [vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(mask_image), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
            check=True,
            stdout=subprocess.PIPE,
        )
        mask = np.frombuffer(raw.stdout, dtype=np.uint8).reshape(height, width).copy()
        if float(mask.mean()) < 2:
            mask = coarse_subject_mask(_png_to_rgb(first))
    else:
        mask = coarse_subject_mask(_png_to_rgb(first))
    dilate = max(0, int(os.getenv("MATANYONE_DILATE", "10")))
    erode = max(0, int(os.getenv("MATANYONE_ERODE", "10")))
    mask = _morph_mask(mask, dilate, True)
    mask = _morph_mask(mask, erode, False)
    if float(mask.mean()) < 2:
        raise RuntimeError("MatAnyone first-frame mask is empty")
    _write_gray_png(mask, dest)
    return dest


def _processor():
    global _matanyone_net
    source = matanyone_source()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from matanyone.inference.inference_core import InferenceCore
    from matanyone.model.matanyone import MatAnyone

    if _matanyone_net is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        model_id = os.getenv("MATANYONE_MODEL", "PeiqingYang/MatAnyone")
        network = MatAnyone.from_pretrained(model_id)
        network.to("cuda").eval()
        _matanyone_net = network
    return InferenceCore(_matanyone_net)


def _image_tensor(rgb: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255).to(device)


def _alpha_bytes(processor, output_prob) -> np.ndarray:
    mask = processor.output_prob_to_mask(output_prob)
    alpha = (mask.clamp(0, 1) * 255).round().byte().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha.squeeze()
    return alpha


def matte(source: Path, transparent: Path, info: dict, job: dict, mask_url: str, work: Path) -> dict:
    width, height = info["width"], info["height"]
    fps = info["fps"]
    n_warmup = max(1, int(os.getenv("MATANYONE_WARMUP", "10")))
    mask_path = prepare_mask(source, mask_url, work, info)
    raw_mask = subprocess.run(
        [vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(mask_path), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        check=True,
        stdout=subprocess.PIPE,
    )
    mask_np = np.frombuffer(raw_mask.stdout, dtype=np.uint8).reshape(height, width).copy()
    processor = _processor()
    mask_t = torch.from_numpy(mask_np).float().to(processor.device)
    decoder = subprocess.Popen(
        [
            vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(source),
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
            vm.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
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
    frame_bytes = width * height * 3
    frames = 0
    model_seconds = 0.0
    started = time.perf_counter()
    quality = vm.MatteQualityMonitor()
    try:
        first_raw = vm._read_exact(decoder.stdout, frame_bytes)
        if len(first_raw) != frame_bytes:
            raise RuntimeError("decoder returned no first frame")
        first = np.frombuffer(first_raw, dtype=np.uint8).reshape(height, width, 3).copy()
        image = _image_tensor(first, processor.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            torch.cuda.synchronize()
            model_started = time.perf_counter()
            processor.step(image, mask_t, objects=[1])
            output_prob = processor.step(image, first_frame_pred=True)
            for _ in range(1, n_warmup):
                output_prob = processor.step(image, first_frame_pred=True)
            output_prob = processor.step(image, first_frame_pred=True)
            torch.cuda.synchronize()
            model_seconds += time.perf_counter() - model_started
            alpha = _alpha_bytes(processor, output_prob)
            quality_ok, quality_reason = quality.check(first, alpha)
            if not quality_ok:
                raise RuntimeError(f"MatAnyone matte rejected: {quality_reason}")
            encoder.stdin.write(np.concatenate((first, alpha[..., None]), axis=-1).tobytes())
            frames = 1
            while True:
                raw = vm._read_exact(decoder.stdout, frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError("decoder returned a partial frame")
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                image = _image_tensor(rgb, processor.device)
                torch.cuda.synchronize()
                model_started = time.perf_counter()
                output_prob = processor.step(image)
                torch.cuda.synchronize()
                model_seconds += time.perf_counter() - model_started
                alpha = _alpha_bytes(processor, output_prob)
                quality_ok, quality_reason = quality.check(rgb, alpha)
                if not quality_ok:
                    raise RuntimeError(f"MatAnyone matte rejected: {quality_reason}")
                encoder.stdin.write(np.concatenate((rgb, alpha[..., None]), axis=-1).tobytes())
                frames += 1
                if vm.runpod is not None and frames % 30 == 0:
                    vm.runpod.serverless.progress_update(job, f"Matted {frames} frames")
    finally:
        decoder.stdout.close()
        encoder.stdin.close()
    decoder_status = decoder.wait()
    encoder_status = encoder.wait()
    if decoder_status != 0 or encoder_status != 0 or not transparent.is_file():
        raise RuntimeError(f"video pipeline failed (decode={decoder_status}, encode={encoder_status})")
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
        "source_width": width,
        "source_height": height,
        "warmup_frames": n_warmup,
        "engine": ENGINE,
        "mask_url": bool(mask_url),
        "dtype": "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
    }
