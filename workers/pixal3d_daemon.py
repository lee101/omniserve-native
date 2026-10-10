#!/usr/bin/env python3
"""Resident Pixal3D runtime for the shared 5090 (run with the Pixal3D venv).

Weights stay staged in host RAM between jobs, so a job pays only its own GPU time.
Each job holds an omniserve VRAM-broker lease for its duration, so qwen/zimage see
the memory as reserved and back off; no lease means a 503 and the caller falls back
to RunPod. Resolution is capped at 1024: 1536 does not fit the shared budget.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

COG_DIR = os.getenv("PIXAL3D_COG_DIR", "/nvme0n1-disk/code/pixal3dcog")
os.environ.setdefault("PIXAL3D_ROOT", "/nvme0n1-disk/code/Pixal3D")
os.environ.setdefault("PIXAL3D_HF_HOME", "/nvme0n1-disk/models/huggingface")
os.environ.setdefault("PIXAL3D_LOW_VRAM", "1")
os.environ.setdefault("ATTN_BACKEND", "xformers")
sys.path.insert(0, COG_DIR)

BROKER = os.getenv("PIXAL3D_BROKER_URL", "http://127.0.0.1:8791").rstrip("/")
LEASE_MB = int(os.getenv("PIXAL3D_LEASE_MB", "16500"))
LEASE_MIN_MB = int(os.getenv("PIXAL3D_LEASE_MIN_MB", "14500"))
LEASE_TIER = os.getenv("PIXAL3D_LEASE_TIER", "sub")
LEASE_WAIT_MS = int(os.getenv("PIXAL3D_LEASE_WAIT_MS", "45000"))
MAX_RESOLUTION = 1024
MAX_IMAGE_BYTES = 25 << 20

state = {"predictor": None, "error": None, "loading": False}
JOB_LOCK = threading.Lock()


def broker(path: str, body: dict, timeout: float = 90) -> dict:
    request = urllib.request.Request(
        BROKER + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "pixal3d-daemon/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def acquire_lease() -> dict | None:
    try:
        grant = broker("/v1/gpu/lease", {
            "owner": "pixal3d", "mb": LEASE_MB, "min_mb": LEASE_MIN_MB, "ttl_s": 900,
            "pid": os.getpid(), "tier": LEASE_TIER, "wait_ms": LEASE_WAIT_MS}, timeout=LEASE_WAIT_MS / 1000 + 15)
    except (OSError, ValueError):
        return None
    return grant if grant.get("granted") else None


def release_lease(grant: dict) -> None:
    try:
        broker("/v1/gpu/release", {"lease_id": grant["lease_id"]}, timeout=10)
    except (OSError, ValueError, KeyError):
        pass


def load_predictor() -> None:
    if state["predictor"] or state["loading"]:
        return
    state["loading"] = True
    try:
        import predict  # noqa: PLC0415  (pixal3dcog; imports torch and the pipeline)
        import torch  # noqa: PLC0415
        # cudnn autotuning allocates multi-GB trial workspaces; they do not fit beside the other tenants
        torch.backends.cudnn.benchmark = os.getenv("PIXAL3D_CUDNN_BENCHMARK", "0") == "1"
        predictor = predict.Predictor()
        predictor.setup()
        state["predictor"] = predictor
        state["error"] = None
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(f"[pixal3d] load failed: {state['error']}", file=sys.stderr, flush=True)
    finally:
        state["loading"] = False


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "pixal3d-daemon/1"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        size = 0
        while chunk := response.read(1 << 20):
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise ValueError("image too large")
            out.write(chunk)


def generate(payload: dict) -> tuple[int, dict]:
    if state["predictor"] is None:
        return HTTPStatus.SERVICE_UNAVAILABLE, {"error": "pixal_loading", "message": state["error"] or "model loading",
                                                "retry_after_seconds": 30}
    if not JOB_LOCK.acquire(blocking=False):
        return HTTPStatus.SERVICE_UNAVAILABLE, {"error": "worker_busy", "retry_after_seconds": 30}
    started = time.monotonic()
    grant = None
    try:
        grant = acquire_lease()
        if grant is None:
            return HTTPStatus.SERVICE_UNAVAILABLE, {"error": "gpu_busy", "message": "no VRAM lease", "retry_after_seconds": 60}
        import predict  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from cog import Path as CogPath  # noqa: PLC0415
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, grant["mb"] * (1 << 20) / total))
        torch.cuda.reset_peak_memory_stats()
        out_path = Path(payload["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        source = out_path.parent / "source"
        download(payload["image_url"], source)
        params = {**predict.predict_defaults(), "resolution": min(int(payload.get("resolution", 1024)), MAX_RESOLUTION),
                  "texture_size": int(payload.get("texture_size", 1024)),
                  "decimation_target": int(payload.get("decimation_target", 200000)),
                  "seed": int(payload.get("seed", 42)), "upload": False}
        result = state["predictor"].predict(image=CogPath(str(source)), **params)
        shutil.move(str(result.glb), out_path)
        return HTTPStatus.OK, {"seed": result.seed, "resolution": result.resolution, "timings": result.timings,
                               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                               "lease_mb": grant["mb"], "lease_wait_ms": grant.get("waited_ms"),
                               "elapsed_ms": round((time.monotonic() - started) * 1000)}
    except Exception as exc:  # noqa: BLE001
        oom = "out of memory" in str(exc).lower()
        return (HTTPStatus.SERVICE_UNAVAILABLE if oom else HTTPStatus.BAD_GATEWAY,
                {"error": "gpu_busy" if oom else "generation_failed", "message": str(exc)[:500], "retry_after_seconds": 60})
    finally:
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if grant:
            release_lease(grant)
        JOB_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "Pixal3DDaemon/1"

    def send_json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_json(HTTPStatus.OK, {"ready": state["predictor"] is not None, "loading": state["loading"],
                                       "busy": JOB_LOCK.locked(), "error": state["error"]})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            payload["image_url"], payload["out_path"]
        except (ValueError, KeyError, TypeError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "image_url and out_path are required"})
            return
        status, body = generate(payload)
        self.send_json(status, body)

    def log_message(self, message: str, *args: object) -> None:
        sys.stderr.write(f"[pixal3d] {message % args}\n")


def main() -> None:
    bind, port = os.getenv("PIXAL3D_BIND", "127.0.0.1"), int(os.getenv("PIXAL3D_PORT", "9097"))
    threading.Thread(target=load_predictor, daemon=True).start()
    ThreadingHTTPServer((bind, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
