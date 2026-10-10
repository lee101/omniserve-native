"""Local Pixal3D route: the resident daemon (pixal3d_daemon.py) on the shared 5090.
Stdlib only. The daemon holds a VRAM-broker lease per job; a 503 here means the
caller should fall back to RunPod."""

from __future__ import annotations

import json
import os
import sys
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


def base() -> str:
    return os.getenv("OMNISERVE_3D_PIXAL_DAEMON", "http://127.0.0.1:9097").rstrip("/")


def enabled() -> bool:
    return os.getenv("OMNISERVE_3D_PIXAL_LOCAL", "1") != "0"


def ready(open_url=urllib.request.urlopen) -> bool:
    if not enabled():
        return False
    try:
        with open_url(base() + "/health", timeout=2) as response:
            return bool(json.load(response).get("ready"))
    except (OSError, ValueError):
        return False


def ledger(**row) -> None:
    if frontier is not None:
        frontier.record(workload="pixal3d", source="trellis2_worker", backend="local", gpu="RTX5090",
                        quality_tier="equal", **row)


def run(model: str, image_url: str, resolution: int, texture_size: int, decimation_target: int, seed: int,
        output_root: Path, open_url=urllib.request.urlopen) -> tuple[int, dict]:
    started = time.monotonic()
    job_id = f"3d-local-{time.time_ns()}-{seed & 0xffff:04x}"
    body = {"image_url": image_url, "out_path": str(output_root / job_id / "model.glb"), "resolution": resolution,
            "texture_size": texture_size, "decimation_target": decimation_target, "seed": seed}
    request = urllib.request.Request(base() + "/generate", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    status, result = HTTPStatus.SERVICE_UNAVAILABLE, {"error": "pixal_unavailable", "retry_after_seconds": 60}
    try:
        with open_url(request, timeout=float(os.getenv("OMNISERVE_3D_JOB_TIMEOUT_S", "900"))) as response:
            status, result = response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            result = json.load(exc)
        except ValueError:
            result = {"error": "generation_failed", "message": f"daemon returned {exc.code}"}
    except (OSError, ValueError) as exc:
        result = {"error": "pixal_unavailable", "message": str(exc)[:300], "retry_after_seconds": 60}
    wall_ms = (time.monotonic() - started) * 1000
    ok = status == HTTPStatus.OK
    ledger(tier="free", status=200 if ok else status, job_id=job_id, queue_ms=result.get("lease_wait_ms") if ok else None,
           exec_ms=(result.get("timings") or {}).get("total", 0) * 1000 if ok else None, wall_ms=wall_ms)
    if not ok:
        return status, result
    return HTTPStatus.OK, {
        "id": job_id, "object": "3d.generation", "model": model,
        "model_glb": {"url": f"/api/3d-assets/{job_id}/model.glb", "content_type": "model/gltf-binary",
                      "file_name": f"{job_id}.glb", "file_size": Path(body["out_path"]).stat().st_size},
        "seed": result.get("seed", seed), "resolution": result.get("resolution", resolution),
        "texture_size": texture_size, "backend": "local",
        "timings": {"elapsed_ms": round(wall_ms), "local": result.get("timings"), "peak_gb": result.get("peak_gb")},
    }
