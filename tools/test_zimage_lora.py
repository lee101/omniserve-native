#!/usr/bin/env python3
"""Generate a reviewable Z-Image-Turbo LoRA test locally or via a running server.

The running-server mode is intended for the loopback native gateway, where a
direct ``loras`` path is allowed.  The local mode starts a native gateway from
the current checkout and uses the same request contract.  Outputs are written
as an image plus a JSON sidecar containing timing, server response, and GPU
memory observations.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA = Path("/vfast/data/code/loras/new/nsm_ZIT_000017532.safetensors")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_memory() -> dict[str, Any]:
    """Return a best-effort nvidia-smi snapshot without making it required."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        rows.append({"name": values[0], "total_mib": int(values[1]),
                     "used_mib": int(values[2]), "free_mib": int(values[3])})
    return {"available": bool(rows), "gpus": rows}


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def request_json(base: str, payload: dict[str, Any], timeout: float) -> tuple[bytes, str, dict[str, str]]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/images/generations",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Accept": "application/json, image/*",
            "Content-Type": "application/json",
            "X-Omniserve-Tier": "background",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", ""), dict(response.headers)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace").strip()
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def decode_image(body: bytes, content_type: str) -> tuple[bytes, dict[str, Any]]:
    if "json" not in content_type.lower() and body.lstrip()[:1] not in (b"{", b"["):
        return body, {"transport": "raw"}
    document = json.loads(body)
    row = (document.get("data") or [{}])[0] if isinstance(document, dict) else (document or [{}])[0]
    encoded = row.get("b64_json") or row.get("image_base64")
    if not encoded:
        raise RuntimeError("image response did not contain b64_json/image_base64")
    return base64.b64decode(encoded), {
        "transport": "b64_json",
        "model": document.get("model") if isinstance(document, dict) else None,
        "seed": row.get("seed"),
        "inference_time_ms": row.get("inference_time_ms"),
        "format": row.get("format"),
        "teleport": row.get("teleport"),
    }


def wait_for_server(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base.rstrip('/')}/health", timeout=3) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(1)
    raise RuntimeError(f"server did not become healthy: {base}")


def start_local_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    binary = Path(args.server_binary).expanduser()
    if not binary.is_absolute():
        binary = (Path(args.server_root).expanduser() / binary).resolve()
    if not binary.is_file():
        raise RuntimeError(f"local server binary not found: {binary}")
    environment = os.environ.copy()
    if args.env_file:
        environment.update(load_env_file(Path(args.env_file).expanduser()))
    environment["OMNISERVE_NATIVE_BIND"] = "127.0.0.1"
    environment["OMNISERVE_NATIVE_PORT"] = str(args.port)
    command = [str(binary), "--port", str(args.port), *args.server_arg]
    return subprocess.Popen(command, cwd=args.server_root, env=environment,
                            start_new_session=True, text=True)


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("running", "local", "ssh", "auto"), default="running")
    parser.add_argument("--base", default="http://127.0.0.1:8791")
    parser.add_argument("--lora", default=str(DEFAULT_LORA),
                        help="LoRA path; pass an empty string for the base model")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--prompt", default=(
        "a cinematic portrait of an astronaut botanist in a glass greenhouse on Mars, "
        "warm rim light, intricate practical details, natural expression, editorial photography"
    ))
    parser.add_argument("--seed", type=int, default=17532)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--teleport", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teleport-start-step", type=int, default=7)
    parser.add_argument(
        "--allow-raw", action="store_true",
        help="accept a raw image response without native metadata (for Daisy)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--server-root", default=str(ROOT))
    parser.add_argument("--server-binary", default="build-full/omniserve-native")
    parser.add_argument("--env-file")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--ssh-host", help="SSH host for --mode ssh")
    parser.add_argument("--ssh-remote-port", type=int, default=8791)
    parser.add_argument("--ssh-local-port", type=int, default=18792)
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()

    lora = Path(args.lora).expanduser() if args.lora else None
    if lora is not None and not lora.is_file():
        raise SystemExit(f"LoRA file not found: {lora}")
    if lora is not None and not -4.0 <= args.lora_scale <= 4.0:
        raise SystemExit("--lora-scale must be between -4 and 4")
    if args.width % 64 or args.height % 64:
        raise SystemExit("--width and --height must be multiples of 64")
    if args.teleport and not 1 <= args.teleport_start_step < args.steps:
        raise SystemExit("--teleport-start-step must be between 1 and steps-1")

    process: subprocess.Popen[str] | None = None
    if args.mode == "auto":
        try:
            wait_for_server(args.base, 3)
            args.mode = "running"
        except RuntimeError:
            args.mode = "local"
    if args.mode == "ssh":
        if not args.ssh_host:
            raise SystemExit("--ssh-host is required with --mode ssh")
        args.base = f"http://127.0.0.1:{args.ssh_local_port}"
        process = subprocess.Popen([
            "ssh", "-N", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=no",
            "-L", f"{args.ssh_local_port}:127.0.0.1:{args.ssh_remote_port}",
            args.ssh_host,
        ], start_new_session=True, text=True)
    elif args.mode == "local":
        process = start_local_server(args)

    try:
        wait_for_server(args.base, args.timeout)
        payload = {
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "teleport": args.teleport,
        }
        if args.teleport:
            payload["teleport_start_step"] = args.teleport_start_step
        if lora is not None:
            payload["loras"] = [{"path": str(lora.resolve()), "scale": args.lora_scale}]
        before = gpu_memory()
        started = time.perf_counter()
        body, content_type, headers = request_json(args.base, payload, args.timeout)
        wall_ms = round((time.perf_counter() - started) * 1000, 3)
        image, response_meta = decode_image(body, content_type)
        if lora is not None and not args.allow_raw and (
            response_meta.get("transport") == "raw" or
            response_meta.get("teleport") is None
        ):
            raise RuntimeError(
                "the target server accepted the request but returned no native "
                "LoRA/teleport metadata; it is likely an older proxy or Z-Image "
                "build that ignores loras. Use --mode ssh with the native gateway "
                "or upgrade the local server."
            )
        after = gpu_memory()

        output_dir = args.output_dir.expanduser() / f"zimage_lora_{stamp()}"
        output_dir.mkdir(parents=True, exist_ok=False)
        image_path = output_dir / "image.webp"
        image_path.write_bytes(image)
        report = {
            "mode": args.mode,
            "base": args.base,
            "request": payload,
            "lora": ({"path": str(lora.resolve()), "bytes": lora.stat().st_size}
                     if lora is not None else None),
            "response": response_meta,
            "content_type": content_type,
            "response_bytes": len(body),
            "wall_ms": wall_ms,
            "gpu_before": before,
            "gpu_after": after,
            "server_headers": {key: value for key, value in headers.items()
                                if key.lower() in {"x-request-id", "x-omniserve-image-ms"}},
            "image": str(image_path),
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"image": str(image_path), "report": str(output_dir / "report.json"),
                          "wall_ms": wall_ms, "teleport": response_meta.get("teleport")},
                         indent=2))
        return 0
    finally:
        if process is not None:
            stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
