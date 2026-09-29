"""Sampled CUDA person gate for video-matting workload routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time


PERSON_CLASS_ID = 0  # COCO
MODEL_PATH = Path(os.getenv("PERSON_DETECTOR_PATH", "/opt/models/yolo11n.pt"))
FFMPEG = os.getenv("VIDEO_MATTING_FFMPEG", "/usr/bin/ffmpeg")


@dataclass(frozen=True)
class PersonDecision:
    detected: bool
    sampled_frames: int = 0
    frames_with_person: int = 0
    max_confidence: float = 0.0
    threshold: float = 0.35
    device: str = "none"
    elapsed_seconds: float = 0.0
    error: str = ""

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


class VideoPersonDetector:
    """Persistent YOLO11n detector; video decoding is deliberately sampled."""

    def __init__(self) -> None:
        self._model = None
        self._device = "none"
        self._error = ""
        self._load_lock = threading.Lock()
        self._thread = None

    def preload_async(self) -> None:
        if self._model is not None or self._error or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._load, name="video-person-detector-preload", daemon=True)
        self._thread.start()

    def _load(self) -> None:
        if self._model is not None or self._error:
            return
        with self._load_lock:
            if self._model is not None or self._error:
                return
            try:
                import torch
                from ultralytics import YOLO

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable to the person detector")
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError(f"person detector not found: {MODEL_PATH}")
                model = YOLO(str(MODEL_PATH))
                model.to("cuda:0")
                # Force weight upload and kernel initialization outside measured jobs.
                model.predict(
                    source=torch.zeros((1, 3, 640, 640), device="cuda:0"),
                    classes=[PERSON_CLASS_ID],
                    conf=0.35,
                    device=0,
                    half=True,
                    verbose=False,
                )
                self._model = model
                self._device = "cuda:0"
            except Exception as error:  # The caller routes uncertainty to standby.
                self._error = str(error)[:1000]

    @staticmethod
    def _sample_frames(video: Path, duration: float, count: int) -> list:
        import cv2

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

    def detect(self, video: Path, *, duration: float) -> PersonDecision:
        started = time.perf_counter()
        threshold = min(0.95, max(0.05, float(os.getenv("PERSON_DETECTOR_CONFIDENCE", "0.35"))))
        samples = max(2, min(24, int(os.getenv("PERSON_DETECTOR_SAMPLES", "8"))))
        self._load()
        if self._model is None:
            return PersonDecision(
                detected=False,
                threshold=threshold,
                device=self._device,
                elapsed_seconds=time.perf_counter() - started,
                error=self._error or "person detector unavailable",
            )
        try:
            frames = self._sample_frames(video, duration, samples)
            results = self._model.predict(
                source=frames,
                classes=[PERSON_CLASS_ID],
                conf=threshold,
                imgsz=640,
                device=0,
                half=True,
                verbose=False,
            )
            frame_confidences = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                confidences = getattr(boxes, "conf", None) if boxes is not None else None
                frame_confidences.append(
                    max((float(value) for value in confidences.tolist()), default=0.0)
                    if confidences is not None else 0.0
                )
            return PersonDecision(
                detected=any(value >= threshold for value in frame_confidences),
                sampled_frames=len(frames),
                frames_with_person=sum(value >= threshold for value in frame_confidences),
                max_confidence=max(frame_confidences, default=0.0),
                threshold=threshold,
                device=self._device,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as error:
            return PersonDecision(
                detected=False,
                threshold=threshold,
                device=self._device,
                elapsed_seconds=time.perf_counter() - started,
                error=str(error)[:1000],
            )

    def release(self) -> None:
        self._model = None
        self._device = "none"
        self._error = ""
        self._thread = None
