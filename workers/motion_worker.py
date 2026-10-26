#!/usr/bin/env python3
"""Resident video-to-motion worker (GEM-X, SOMA skeleton) behind the native C gateway."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import ipaddress
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import motion_format

GEMX_ROOT = Path(os.getenv("MOTION_GEMX_ROOT", "/vfast/data/code/gvhmr-pipe/gemx")).resolve()
MAX_FRAMES = int(os.getenv("MOTION_MAX_FRAMES", "600"))
MAX_DOWNLOAD_BYTES = int(os.getenv("MOTION_MAX_DOWNLOAD_BYTES", str(96 << 20)))
DEFAULT_FPS_OUT = float(os.getenv("MOTION_FPS_OUT", "20"))
BACKENDS = ("gem-x",)
ALLOW_PRIVATE_URLS = os.getenv("MOTION_ALLOW_PRIVATE_URLS", "0") == "1"
MAX_REDIRECTS = 3
WARMUP = os.getenv("MOTION_WARMUP", "1") == "1"
WARMUP_VIDEO = Path(os.getenv("MOTION_WARMUP_VIDEO", str(Path(__file__).resolve().parent / "assets" / "motion_warmup.mp4")))


class MotionError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class GemxEngine:
    name = "gem-x"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loaded = False
        self.ready = False
        self.load_seconds = 0.0
        self.warm_seconds = 0.0
        self.jobs = 0

    def load(self) -> None:
        import torch

        started = time.perf_counter()
        os.chdir(GEMX_ROOT)
        if str(GEMX_ROOT) not in sys.path:
            sys.path.insert(0, str(GEMX_ROOT))
        spec = importlib.util.spec_from_file_location("gemx_demo_soma", GEMX_ROOT / "scripts/demo/demo_soma.py")
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        self.demo, self.torch = demo, torch

        from gem.utils.sam3db_extractor import SAM3DBExtractor
        from gem.utils.vitpose_extractor import VitPoseExtractor
        from gem.utils.yolox_detector import YOLOXDetector

        with tempfile.TemporaryDirectory() as tmp:
            placeholder = Path(tmp) / "placeholder.mp4"
            placeholder.touch()
            cfg = demo._build_cfg(self._args(placeholder, Path(tmp) / "out", True))
        self.model = demo.hydra.utils.instantiate(cfg.model, _recursive_=False)
        self.model.load_pretrained_model(demo.resolve_ckpt_path(cfg))
        self.model = self.model.eval().cuda()
        self.yolox = YOLOXDetector(device="cuda")
        self.vitpose = VitPoseExtractor(device="cuda:0", pose_type="soma")
        self.sam3db = SAM3DBExtractor(checkpoint_path=None, mhr_path=None, device="cuda:0")
        self.loaded = True
        self.load_seconds = time.perf_counter() - started
        if WARMUP and WARMUP_VIDEO.exists():
            warm_started = time.perf_counter()
            self.run(WARMUP_VIDEO, True, warmup=True)
            self.warm_seconds = time.perf_counter() - warm_started
        self.ready = True

    @staticmethod
    def _args(video: Path, out: Path, static_cam: bool):
        return SimpleNamespace(
            video=str(video), output_root=str(out), static_cam=static_cam, verbose=False, render_mhr=False,
            sam3d_ckpt_path=None, sam3d_mhr_path=None, ckpt=None, exp="gem_soma_regression", retarget=False,
        )

    def _sync(self) -> float:
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def run(self, video: Path, static_cam: bool = True, max_seconds: float = 0.0, warmup: bool = False) -> dict[str, Any]:
        if not (self.ready or (warmup and self.loaded)):
            raise MotionError(503, "motion model is still loading")
        torch, demo = self.torch, self.demo
        with self.lock, torch.no_grad():
            t0 = self._sync()
            length, width, height = demo.get_video_lwh(str(video))
            if length < 2:
                raise MotionError(422, "video has fewer than 2 frames")
            if length > MAX_FRAMES:
                raise MotionError(413, f"video has {length} frames; limit is {MAX_FRAMES}")
            fps = float(demo.cv2.VideoCapture(str(video)).get(demo.cv2.CAP_PROP_FPS)) or 30.0
            if max_seconds and length / fps > max_seconds + 0.05:
                raise MotionError(413, f"video is {length / fps:.1f}s; max_seconds is {max_seconds:g}")
            work = Path(tempfile.mkdtemp(prefix="motion-job-"))
            try:
                cfg = demo._build_cfg(self._args(video, work, static_cam))
                paths = cfg.paths
                frames = demo.read_video_np(str(video))
                t1 = self._sync()

                bbx_np, _ = self._detect(frames)
                bbx_xyxy = demo.smooth_bbx_xyxy(torch.from_numpy(bbx_np).float(), window=5)
                bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, width - 1)
                bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, height - 1)
                bbx_xys = demo.get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
                torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
                t2 = self._sync()

                vitpose = self.vitpose.extract(frames, bbx_xys, img_ds=1.0, path_type="np")
                torch.save(vitpose, paths.vitpose)
                t3 = self._sync()

                eye = torch.eye(4).unsqueeze(0).repeat(length, 1, 1).numpy()
                torch.save(eye, paths.slam)
                sam = self.sam3db.extract_video_features(str(video), bbx_xys, render_mhr=False)
                K = demo.estimate_K(width, height).repeat(sam["transls"].shape[0], 1, 1)
                torch.save({"pose_tokens": sam["pose_tokens"],
                            "pred_cam": demo.get_a_pred_cam(sam["transls"], bbx_xys, K)}, paths.vit_features)
                t4 = self._sync()

                data = demo.load_data_dict(cfg)
                pred = self.model.predict(data, static_cam=True, postproc=True)
                t5 = self._sync()

                params = pred["body_params_global"]
                contact_probs = torch.sigmoid(pred["net_outputs"]["static_conf_logits"][0].float()).cpu().numpy()
                batch = {k: params[k][None].float().cuda() for k in
                         ("body_pose", "global_orient", "transl", "identity_coeffs", "scale_params")}
                joints = self.model.endecoder.fk_v2(**batch)[0].float().cpu().numpy()
                t6 = self._sync()
                params_np = {k: v.float().cpu().numpy() for k, v in params.items()}
            finally:
                for f in work.rglob("*"):
                    if f.is_file():
                        f.unlink()
                for d in sorted((p for p in work.rglob("*") if p.is_dir()), reverse=True):
                    d.rmdir()
                work.rmdir()
            if not warmup:
                self.jobs += 1
        stages = {"decode": t1 - t0, "detect": t2 - t1, "pose2d": t3 - t2, "features": t4 - t3,
                  "motion": t5 - t4, "fk": t6 - t5}
        return {"fps_in": fps, "frames": int(length), "joints77": joints, "params": params_np, "contact_probs": contact_probs,
                "timings": {k: round(v, 4) for k, v in stages.items()}, "gpu_seconds": round(t6 - t0, 4)}

    def _detect(self, frames):
        from gem.utils.yolox_detector import detect_and_track
        return detect_and_track(frames, self.yolox)


def build_response(result: dict[str, Any], *, name: str, prompt: str, fps_out: float, include_soma: bool,
                   backend: str) -> dict[str, Any]:
    motion = motion_format.build_ardy_motion(
        result["joints77"], result["fps_in"], fps_out, contact_probs=result["contact_probs"], id=name, name=name, prompt=prompt, source="video",
        backend=backend,
    )
    out = {
        "backend": backend,
        "frames_in": result["frames"],
        "fps_in": round(result["fps_in"], 3),
        "motion": motion,
        "timings": result["timings"],
        "gpu_seconds": result["gpu_seconds"],
    }
    if include_soma:
        out["soma"] = motion_format.soma_payload(result["params"], result["fps_in"])
    return out


def parse_request(raw: bytes) -> VideoRequest:
    try:
        if hasattr(VideoRequest, "model_validate_json"):
            return VideoRequest.model_validate_json(raw)
        return VideoRequest.parse_raw(raw)
    except ValueError as exc:
        errors = getattr(exc, "errors", None)
        detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errors()) if errors else str(exc)
        raise ValueError(detail or "body must be a JSON object") from exc


def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise MotionError(422, "video_url must be a public http or https URL")
    if ALLOW_PRIVATE_URLS:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise MotionError(422, "video_url host could not be resolved") from exc
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise MotionError(422, "video_url must be a public URL")


def fetch_video(url: str | None, b64: str | None, dest: Path) -> None:
    if b64:
        try:
            dest.write_bytes(base64.b64decode(b64, validate=True))
        except Exception as exc:
            raise MotionError(422, "video_base64 is not valid base64") from exc
        if dest.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise MotionError(413, "video is too large")
        return
    if not url:
        raise MotionError(422, "video_url or video_base64 is required")
    try:
        for _ in range(MAX_REDIRECTS + 1):
            assert_public_url(url)
            resp = requests.get(url, stream=True, timeout=(5, 60), allow_redirects=False)
            if resp.is_redirect and resp.headers.get("location"):
                url = requests.compat.urljoin(url, resp.headers["location"])
                resp.close()
                continue
            break
        else:
            raise MotionError(422, "video_url redirected too many times")
        with resp:
            resp.raise_for_status()
            size = 0
            with dest.open("wb") as handle:
                for chunk in resp.iter_content(1 << 20):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise MotionError(413, "video is too large")
                    handle.write(chunk)
    except requests.RequestException as exc:
        raise MotionError(502, f"could not download video: {exc}") from exc


engine = GemxEngine()


class VideoRequest(BaseModel):
    video_url: str | None = None
    video_base64: str | None = None
    backend: str = "gem-x"
    static_cam: bool = True
    fps_out: float = Field(DEFAULT_FPS_OUT, ge=5, le=60)
    name: str = Field("motion", max_length=120)
    prompt: str = Field("", max_length=2000)
    include_soma: bool = False
    max_seconds: float = Field(0.0, ge=0, le=60)


def create_app():
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=engine.load, daemon=True, name="motion-load").start()
        yield

    app = FastAPI(title="motion-worker", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok" if engine.ready else "loading", "backends": list(BACKENDS), "jobs": engine.jobs,
                "load_seconds": round(engine.load_seconds, 2), "warmup_seconds": round(engine.warm_seconds, 2),
                "max_frames": MAX_FRAMES}

    def process(req: VideoRequest):
        if req.backend not in BACKENDS:
            return JSONResponse({"error": f"backend must be one of {list(BACKENDS)}"}, status_code=422)
        with tempfile.TemporaryDirectory(prefix="motion-in-") as tmp:
            video = Path(tmp) / "input.mp4"
            try:
                fetch_video(req.video_url, req.video_base64, video)
                result = engine.run(video, req.static_cam, req.max_seconds)
            except MotionError as exc:
                return JSONResponse({"error": exc.message}, status_code=exc.status)
        return build_response(result, name=req.name, prompt=req.prompt, fps_out=req.fps_out,
                              include_soma=req.include_soma, backend=req.backend)

    @app.post("/v1/motion/video")
    async def motion_video(request: Request):
        try:
            req = parse_request(await request.body())
        except ValueError as exc:
            return JSONResponse({"error": "invalid request", "detail": str(exc)}, status_code=422)
        return await run_in_threadpool(process, req)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("MOTION_PORT", "8793")))
    parser.add_argument("--host", default=os.getenv("MOTION_HOST", "127.0.0.1"))
    parser.add_argument("--once", help="process one video file and write the response JSON next to it")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.once:
        engine.load()
        print(json.dumps({"load_seconds": round(engine.load_seconds, 2)}))
        video = Path(args.once)
        for i in range(args.repeat):
            result = engine.run(video)
            resp = build_response(result, name=video.stem, prompt="", fps_out=DEFAULT_FPS_OUT,
                                  include_soma=True, backend="gem-x")
            print(json.dumps({"run": i, "frames": resp["frames_in"], "gpu_seconds": resp["gpu_seconds"],
                              "timings": resp["timings"]}))
        video.with_suffix(".motion-response.json").write_text(json.dumps(resp))
        return
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
