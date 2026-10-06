"""RunPod serverless route for Pixal3D: the same TencentARC/Pixal3D weights on a
dedicated 4090 endpoint, used when the shared 5090 cannot run it (runtime absent
or not enough free VRAM). Stdlib only; one job per request, no local GPU lock."""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

sys.path.insert(0, os.getenv("FRONTIER_LIB", str(Path(__file__).resolve().parents[1])))
try:
    import frontier
except ImportError:
    frontier = None

UA = "omniserve-3d-remote/1"
SLOTS = threading.BoundedSemaphore(int(os.getenv("OMNISERVE_3D_REMOTE_MAX", "2")))


def endpoint() -> str:
    return os.getenv("OMNISERVE_3D_PIXAL_REMOTE_ENDPOINT", "").strip()


def configured() -> bool:
    return bool(endpoint() and os.getenv("RUNPOD_API_KEY"))


def should_route(installed: bool, free_mib: int | None = None, required_mib: int | None = None) -> bool:
    if not configured():
        return False
    mode = os.getenv("OMNISERVE_3D_PIXAL_REMOTE_MODE", "fallback")
    if mode == "off":
        return False
    if mode == "always" or not installed:
        return True
    return free_mib is not None and required_mib is not None and free_mib < required_mib


def call(method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"https://api.runpod.ai/v2/{endpoint()}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.getenv("RUNPOD_API_KEY", ""), "Content-Type": "application/json",
                 "User-Agent": UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def ledger(**row) -> None:
    if frontier is not None:
        frontier.record(workload="pixal3d", source="trellis2_worker", backend="runpod", endpoint=endpoint(),
                        gpu="RTX4090", quality_tier="equal", **row)


def run(model: str, image_url: str, resolution: int, texture_size: int, decimation_target: int, seed: int,
        output_root: Path, call=call, sleep=time.sleep) -> tuple[int, dict]:
    if not SLOTS.acquire(blocking=False):
        return HTTPStatus.SERVICE_UNAVAILABLE, {"error": "remote_busy", "retry_after_seconds": 60}
    try:
        return _run(model, image_url, resolution, texture_size, decimation_target, seed, output_root, call, sleep)
    finally:
        SLOTS.release()


def _run(model, image_url, resolution, texture_size, decimation_target, seed, output_root, call, sleep):
    started = time.monotonic()
    deadline = started + float(os.getenv("OMNISERVE_3D_REMOTE_TIMEOUT_S", "900"))
    job_input = {"image_url": image_url, "resolution": max(1024, resolution), "seed": seed,
                 "texture_size": texture_size, "decimation_target": decimation_target, "upload": True}
    try:
        job = call("POST", "/run", {"input": job_input, "policy": {"executionTimeout": 900000, "ttl": 1200000}})
    except (OSError, ValueError) as exc:
        return HTTPStatus.SERVICE_UNAVAILABLE, {"error": "remote_unavailable", "message": str(exc)[:300],
                                                "retry_after_seconds": 60}
    job_id = str(job.get("id", ""))
    state: dict = {}
    status = "SUBMITTED"
    try:
        while time.monotonic() < deadline:
            try:
                state = call("GET", "/status/" + job_id)
            except (OSError, ValueError):
                sleep(3)
                continue
            status = state.get("status", "")
            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            sleep(3)
        else:
            status = "TIMED_OUT"
            try:
                call("POST", "/cancel/" + job_id, {})
            except (OSError, ValueError):
                pass
    finally:
        ledger(tier="free", status=status, job_id=job_id, queue_ms=state.get("delayTime"),
               exec_ms=state.get("executionTime"), wall_ms=(time.monotonic() - started) * 1000)
    output = state.get("output") if isinstance(state.get("output"), dict) else {}
    if status != "COMPLETED" or output.get("error"):
        return HTTPStatus.BAD_GATEWAY, {"error": "generation_failed", "message": str(output.get("error") or status)[:1000],
                                        "remote_job_id": job_id}
    job_name = f"3d-remote-{time.time_ns()}-{job_id[:12]}"
    url = output.get("glb_url")
    size = None
    if not url and output.get("glb_base64"):
        target = output_root / job_name / "model.glb"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(output["glb_base64"]))
        size = target.stat().st_size
        url = f"/api/3d-assets/{job_name}/model.glb"
    if not url:
        return HTTPStatus.BAD_GATEWAY, {"error": "generation_failed", "message": "remote returned no GLB",
                                        "remote_job_id": job_id}
    return HTTPStatus.OK, {
        "id": job_name, "object": "3d.generation", "model": model,
        "model_glb": {"url": url, "content_type": "model/gltf-binary", "file_name": f"{job_name}.glb",
                      **({"file_size": size} if size else {})},
        "seed": output.get("seed", seed), "resolution": output.get("resolution", job_input["resolution"]),
        "texture_size": texture_size, "backend": "runpod", "remote_job_id": job_id,
        "timings": {"elapsed_ms": round((time.monotonic() - started) * 1000), "remote": output.get("timings"),
                    "queue_ms": state.get("delayTime"), "execution_ms": state.get("executionTime")},
    }
