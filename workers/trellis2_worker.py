#!/usr/bin/env python3
"""Cold-start TRELLIS.2/Pixal3D adapter for omniserve-native.

The C gateway owns authentication and weighted admission. This process owns
model-specific Python/CUDA dependencies and launches one inference subprocess
per request so all VRAM is returned when a background job finishes.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


MODEL_TRELLIS = "microsoft/TRELLIS.2-4B"
MODEL_PIXAL = "TencentARC/Pixal3D"
DINO_TRELLIS = "facebook/dinov3-vitl16-pretrain-lvd1689m"
ALLOWED_MODELS = {MODEL_TRELLIS, MODEL_PIXAL}
MAX_BODY = 1 << 20
MAX_IMAGE = 24 << 20
JOB_LOCK = threading.Lock()


def integer_value(payload: dict, key: str, default: int) -> int:
    raw = payload.get(key, default)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{key} must be an integer")
    if isinstance(raw, str) and str(value) != raw.strip():
        raise ValueError(f"{key} must be an integer")
    return value


def gpu_memory_mib() -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        free_text, total_text = result.stdout.splitlines()[0].split(",", 1)
        return int(free_text.strip()), int(total_text.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0, 0


def validate_public_url(raw: str) -> str:
    parsed = urlparse(raw.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("image_url must be a public HTTP(S) URL")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("image_url hostname could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("image_url cannot resolve to a private or local address")
    return raw.strip()


def download_image(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "OmniServe3D/1.0", "Accept": "image/*"})
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError("image_url did not return an image")
        with destination.open("wb") as output:
            remaining = MAX_IMAGE + 1
            while remaining > 0:
                chunk = response.read(min(1 << 20, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        if destination.stat().st_size > MAX_IMAGE:
            destination.unlink(missing_ok=True)
            raise ValueError("source image exceeds 24 MiB")


def append_manifest(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as manifest:
        fcntl.flock(manifest.fileno(), fcntl.LOCK_EX)
        manifest.write(json.dumps(entry, separators=(",", ":")) + "\n")
        manifest.flush()
        os.fsync(manifest.fileno())
        fcntl.flock(manifest.fileno(), fcntl.LOCK_UN)


def publish_file(local_path: Path, object_key: str) -> str:
    remote = os.getenv("OMNISERVE_3D_R2_REMOTE", "").strip()
    public_base = os.getenv("OMNISERVE_3D_R2_PUBLIC_BASE", "").rstrip("/")
    if not remote:
        return ""
    subprocess.run(
        ["rclone", "copyto", str(local_path), f"{remote.rstrip('/')}/{object_key}"],
        check=True,
        timeout=300,
    )
    return f"{public_base}/{object_key}" if public_base else ""


def model_runtime(model: str) -> tuple[Path, str, Path]:
    if model == MODEL_PIXAL:
        repo = Path(os.getenv("OMNISERVE_3D_PIXAL_REPO", "/nvme0n1-disk/code/Pixal3D"))
        python = os.getenv("OMNISERVE_3D_PIXAL_PYTHON", str(repo / ".venv/bin/python"))
        runner = Path(__file__).with_name("run_pixal3d.py")
        return repo, python, runner
    repo = Path(os.getenv("OMNISERVE_3D_TRELLIS_REPO", "/nvme0n1-disk/code/TRELLIS.2"))
    python = os.getenv("OMNISERVE_3D_PYTHON", str(repo / ".venv/bin/python"))
    runner = Path(__file__).with_name("run_trellis2.py")
    return repo, python, runner


def runtime_installed(model: str) -> bool:
    repo, python, runner = model_runtime(model)
    return repo.is_dir() and Path(python).is_file() and runner.is_file()


def cached_hf_model(repo_id: str) -> bool:
    hf_home = Path(
        os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    )
    repo_dir = hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))
    for snapshot in (repo_dir / "snapshots").glob("*"):
        if (
            (snapshot / "config.json").is_file()
            and (
                (snapshot / "model.safetensors").is_file()
                or (snapshot / "pytorch_model.bin").is_file()
                or (snapshot / "model.safetensors.index.json").is_file()
                or (snapshot / "pytorch_model.bin.index.json").is_file()
            )
        ):
            return True
    return False


def model_dependency_issue(model: str) -> dict | None:
    if model == MODEL_TRELLIS and not cached_hf_model(DINO_TRELLIS):
        return {
            "dependency": DINO_TRELLIS,
            "approval_url": f"https://huggingface.co/{DINO_TRELLIS}",
            "message": (
                "TRELLIS.2 requires the gated DINOv3 image encoder; accept its "
                "terms and prefetch it into HF_HOME before serving jobs"
            ),
        }
    return None


def gpu_coordinator_bases() -> list[str]:
    raw = os.getenv("OMNISERVE_3D_GPU_COORDINATORS", "")
    return [
        value.strip().rstrip("/")
        for value in raw.replace("\n", ",").split(",")
        if value.strip()
    ]


def post_gpu_coordinator(base: str, action: str, timeout: int | None = None) -> None:
    if timeout is None:
        timeout = max(
            15,
            int(os.getenv("OMNISERVE_3D_GPU_COORDINATOR_TIMEOUT_S", "180")),
        )
    suffix = f"/admin/{action}"
    if action == "hold":
        hold_seconds = max(
            60,
            int(os.getenv("OMNISERVE_3D_GPU_HOLD_SECONDS", "1200")),
        )
        suffix += f"?seconds={hold_seconds}"
    request = Request(
        base + suffix,
        data=b"",
        headers={"User-Agent": "OmniServe3D/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"{base} returned HTTP {response.status} for GPU {action}")
        response.read(1 << 20)


def run_with_gpu_holds(job) -> tuple[int, dict]:
    acquired: list[str] = []
    try:
        for base in gpu_coordinator_bases():
            post_gpu_coordinator(base, "hold")
            acquired.append(base)
    except (OSError, RuntimeError) as exc:
        for base in reversed(acquired):
            try:
                post_gpu_coordinator(base, "release")
            except (OSError, RuntimeError):
                pass
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "error": "gpu_coordination_failed",
            "message": str(exc),
            "retry_after_seconds": 30,
        }

    try:
        return job()
    finally:
        for base in reversed(acquired):
            try:
                post_gpu_coordinator(base, "release")
            except (OSError, RuntimeError) as exc:
                print(f"[omniserve-3d] failed to release GPU hold at {base}: {exc}", file=sys.stderr)


def run_job(payload: dict, server_base: str) -> tuple[int, dict]:
    model = str(payload.get("model") or MODEL_TRELLIS)
    if model not in ALLOWED_MODELS:
        return HTTPStatus.BAD_REQUEST, {"error": "unsupported model"}
    image_url = payload.get("image_url") or payload.get("input_image_url")
    if not isinstance(image_url, str):
        return HTTPStatus.BAD_REQUEST, {"error": "image_url is required"}
    try:
        image_url = validate_public_url(image_url)
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {"error": str(exc)}

    try:
        resolution = integer_value(payload, "resolution", 512)
        texture_size = integer_value(payload, "texture_size", 1024)
        decimation_target = integer_value(payload, "decimation_target", 200_000)
        seed = integer_value(payload, "seed", 42)
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {"error": str(exc)}

    allowed_resolutions = {512, 1024, 1536}
    if resolution not in allowed_resolutions:
        return HTTPStatus.BAD_REQUEST, {"error": "resolution must be 512, 1024, or 1536"}
    if texture_size not in {1024, 2048, 4096}:
        return HTTPStatus.BAD_REQUEST, {"error": "texture_size must be 1024, 2048, or 4096"}
    if not 10_000 <= decimation_target <= 1_000_000:
        return HTTPStatus.BAD_REQUEST, {
            "error": "decimation_target must be between 10000 and 1000000"
        }
    if not -(2**31) <= seed < 2**31:
        return HTTPStatus.BAD_REQUEST, {"error": "seed must be a signed 32-bit integer"}

    repo, python, runner = model_runtime(model)
    if not runtime_installed(model):
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "error": "model_not_installed",
            "message": f"{model} runtime is not installed",
        }
    dependency_issue = model_dependency_issue(model)
    if dependency_issue:
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "error": "model_dependency_missing",
            **dependency_issue,
        }

    return run_with_gpu_holds(
        lambda: run_validated_job(
            payload,
            server_base,
            model,
            image_url,
            resolution,
            texture_size,
            decimation_target,
            seed,
            repo,
            python,
            runner,
        )
    )


# TRELLIS.2 segments its input itself when handed an opaque image, with a
# general-purpose salient-object model. Feeding it a BiRefNet cutout that has
# already been colour-decontaminated is strictly better in two ways: the matte is
# sharper on hair and thin structures, and the RGB under the soft edge is the
# subject's own colour rather than the subject blended with whatever it was
# photographed against. That second one matters more than it sounds - the
# backdrop colour in the edge band gets baked into the generated texture and
# then lit, so a green-screened figure comes back with a green rim that no
# amount of retexturing removes.
CUTOUT_BASE = os.getenv("OMNISERVE_3D_CUTOUT_BASE", "http://127.0.0.1:8791").rstrip("/")
CUTOUT_PATH = os.getenv("OMNISERVE_3D_CUTOUT_PATH", "/v1/images/background-removals")
CUTOUT_SECRET = os.getenv("OMNISERVE_3D_CUTOUT_SECRET", os.getenv("OMNISERVE_NATIVE_SECRET", ""))
CUTOUT_TIMEOUT = int(os.getenv("OMNISERVE_3D_CUTOUT_TIMEOUT", "180"))
CUTOUT_ENABLED = os.getenv("OMNISERVE_3D_CUTOUT", "1") == "1"


def prepare_cutout(source_path: Path, job_dir: Path) -> Path:
    """Replaces the source with a background-removed RGBA version.

    Best effort by design: a cutout service that is down, slow or unhelpful must
    not fail a 3D job that would have worked without it, so every failure path
    returns the original image and logs why.
    """
    if not CUTOUT_ENABLED:
        return source_path
    try:
        from PIL import Image
    except ImportError:
        return source_path

    try:
        with source_path.open("rb") as handle:
            data = handle.read()
        if Image.open(io.BytesIO(data)).mode in {"RGBA", "LA"}:
            # Already cut out by the caller; segmenting it again would only
            # erode the matte it already has.
            return source_path
    except Exception as error:
        print(f"3d cutout skipped (unreadable source): {error}", flush=True)
        return source_path

    payload = json.dumps({
        "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
        "output_format": "png",
        "decontaminate": True,
    }).encode()
    headers = {"Content-Type": "application/json", "Accept": "image/png"}
    if CUTOUT_SECRET:
        headers["Authorization"] = f"Bearer {CUTOUT_SECRET}"

    try:
        request = Request(f"{CUTOUT_BASE}{CUTOUT_PATH}", data=payload, headers=headers)
        with urlopen(request, timeout=CUTOUT_TIMEOUT) as response:
            cutout = response.read()
        image = Image.open(io.BytesIO(cutout))
        if image.mode != "RGBA":
            raise ValueError(f"cutout came back as {image.mode}, not RGBA")
    except Exception as error:
        print(f"3d cutout skipped: {error}", flush=True)
        return source_path

    cutout_path = job_dir / "source-cutout.png"
    image.save(cutout_path, format="PNG")
    return cutout_path


def run_validated_job(
    payload: dict,
    server_base: str,
    model: str,
    image_url: str,
    resolution: int,
    texture_size: int,
    decimation_target: int,
    seed: int,
    repo: Path,
    python: str,
    runner: Path,
) -> tuple[int, dict]:
    minimum_free_mib = int(os.getenv("OMNISERVE_3D_MIN_FREE_MIB", "24576"))
    free_mib, total_mib = gpu_memory_mib()
    if free_mib < minimum_free_mib:
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "error": "gpu_busy",
            "message": "3D generation waits for an idle GPU",
            "vram_free_mib": free_mib,
            "vram_required_mib": minimum_free_mib,
            "retry_after_seconds": 60,
        }

    output_root = Path(os.getenv("OMNISERVE_3D_OUTPUT_DIR", "/nvme0n1-disk/models/omniserve-3d/outputs"))
    job_id = (
        f"3d-{time.time_ns()}-"
        f"{hashlib.sha256(image_url.encode()).hexdigest()[:10]}-"
        f"{secrets.token_hex(3)}"
    )
    job_dir = output_root / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    source_path = job_dir / "source"
    output_path = job_dir / "model.glb"
    started = time.monotonic()

    try:
        download_image(image_url, source_path)
        source_path = prepare_cutout(source_path, job_dir)
        command = [
            python,
            str(runner),
            "--repo",
            str(repo),
            "--image",
            str(source_path),
            "--output",
            str(output_path),
            "--resolution",
            str(resolution),
            "--texture-size",
            str(texture_size),
            "--decimation-target",
            str(decimation_target),
            "--seed",
            str(seed),
        ]
        environment = os.environ.copy()
        environment.setdefault("ATTN_BACKEND", "xformers")
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("OMNISERVE_3D_JOB_TIMEOUT_S", "900")),
        )
        (job_dir / "worker.log").write_text(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not output_path.is_file():
            return HTTPStatus.BAD_GATEWAY, {
                "error": "generation_failed",
                "message": completed.stderr[-1000:] or "3D worker returned no GLB",
                "job_id": job_id,
            }

        public_url = f"/api/3d-assets/{job_id}/model.glb"
        publish = payload.get("publish") if isinstance(payload.get("publish"), dict) else {}
        if publish.get("r2", True):
            key = f"generated-3d/{model.split('/')[-1].lower()}/{job_id}/model.glb"
            published_url = publish_file(output_path, key)
            if published_url:
                public_url = published_url

        elapsed_ms = round((time.monotonic() - started) * 1000)
        response = {
            "id": job_id,
            "object": "3d.generation",
            "model": model,
            "model_glb": {
                "url": public_url,
                "content_type": "model/gltf-binary",
                "file_name": f"{job_id}.glb",
                "file_size": output_path.stat().st_size,
            },
            "seed": seed,
            "resolution": resolution,
            "texture_size": texture_size,
            "timings": {"elapsed_ms": elapsed_ms},
        }
        if publish.get("searchable", True):
            manifest = Path(
                os.getenv(
                    "OMNISERVE_3D_MANIFEST",
                    "/nvme0n1-disk/code/hiresnz2/simplexgen/data/object-catalog/generated_3d_objects.jsonl",
                )
            )
            append_manifest(
                manifest,
                {
                    "id": job_id,
                    "name": str(payload.get("name") or f"{model.split('/')[-1]} generation"),
                    "prompt": str(payload.get("prompt") or ""),
                    "url": public_url,
                    "model_url": public_url,
                    "thumbnail": image_url,
                    "thumbnail_url": image_url,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "tags": ["generated", "image-to-3d", model.lower(), "category:ai-generated"],
                    "source": f"omniserve-native:{model}",
                },
            )
        return HTTPStatus.OK, response
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return HTTPStatus.BAD_GATEWAY, {"error": "worker_error", "message": str(exc), "job_id": job_id}


class Handler(BaseHTTPRequestHandler):
    server_version = "OmniServe3D/0.1"

    def send_json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        if status == HTTPStatus.SERVICE_UNAVAILABLE and "retry_after_seconds" in body:
            self.send_header("Retry-After", str(body["retry_after_seconds"]))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/status":
            free_mib, total_mib = gpu_memory_mib()
            model_status = {}
            for model in sorted(ALLOWED_MODELS):
                installed = runtime_installed(model)
                dependency_issue = model_dependency_issue(model) if installed else None
                model_status[model] = {
                    "installed": installed,
                    "ready": installed and dependency_issue is None,
                    **({"dependency_issue": dependency_issue} if dependency_issue else {}),
                }
            installed = [
                model for model, status in model_status.items() if status["ready"]
            ]
            self.send_json(
                HTTPStatus.OK,
                {
                    "ready": bool(installed),
                    "busy": JOB_LOCK.locked(),
                    "models": installed,
                    "model_status": model_status,
                    "vram_free_mib": free_mib,
                    "vram_total_mib": total_mib,
                },
            )
            return
        if self.path.startswith("/assets/"):
            relative = Path(unquote(self.path.removeprefix("/assets/")))
            root = Path(os.getenv("OMNISERVE_3D_OUTPUT_DIR", "/nvme0n1-disk/models/omniserve-3d/outputs")).resolve()
            target = (root / relative).resolve()
            if root not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "model/gltf-binary")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with target.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/v1/3d/generations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid request size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON object required"})
            return
        if not JOB_LOCK.acquire(blocking=False):
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "worker_busy", "retry_after_seconds": 30},
            )
            return
        try:
            public_base = os.getenv(
                "OMNISERVE_3D_PUBLIC_BASE",
                f"http://127.0.0.1:{self.server.server_port}",
            ).rstrip("/")
            status, response = run_job(payload, public_base)
            self.send_json(status, response)
        finally:
            JOB_LOCK.release()

    def log_message(self, message: str, *args: object) -> None:
        sys.stderr.write(f"[omniserve-3d] {self.address_string()} {message % args}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default=os.getenv("OMNISERVE_3D_BIND", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OMNISERVE_3D_PORT", "9093")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"omniserve 3D worker listening on {args.bind}:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
