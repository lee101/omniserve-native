#!/usr/bin/env python3
"""Persistent, cooperative controller for interruptible ASR fine-tuning."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


STATE_DIR = Path(os.getenv("OMNISERVE_TRAINING_STATE_DIR", "./training-state")).resolve()
GATEWAY_STATUS = os.getenv("OMNISERVE_TRAINING_GATEWAY_STATUS", "http://127.0.0.1:8791/status")
ASR_WORKER = os.getenv("OMNISERVE_TRAINING_ASR_WORKER", "http://127.0.0.1:9097").rstrip("/")
MIN_FREE_VRAM_GIB = float(os.getenv("OMNISERVE_TRAINING_MIN_FREE_VRAM_GIB", "18"))
CHECK_INTERVAL_S = float(os.getenv("OMNISERVE_TRAINING_CHECK_INTERVAL_S", "1"))
PREEMPT_GRACE_S = float(os.getenv("OMNISERVE_TRAINING_PREEMPT_GRACE_S", "90"))
SCRIPT = Path(__file__).with_name("train_asr.py")
LOCK = threading.RLock()
RUNNING = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_jobs() -> list[dict[str, Any]]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(STATE_DIR.glob("*.json")):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return jobs


def save_job(job: dict[str, Any]) -> None:
    atomic_json(STATE_DIR / f"{job['id']}.json", job)


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)


def hold_asr_worker(hold: bool) -> None:
    action = "hold" if hold else "release"
    try:
        request_json(f"{ASR_WORKER}/admin/{action}", "POST", {})
    except (OSError, ValueError, urllib.error.URLError):
        # A missing local worker is acceptable; the VRAM/status gates still
        # protect the training process.
        pass


def command_for(job: dict[str, Any]) -> list[str]:
    config = job["config"]
    command = [
        sys.executable,
        str(SCRIPT),
        "--manifest",
        config["manifest"],
        "--audio-root",
        config["audio_root"],
        "--output-dir",
        config["output_dir"],
        "--base-model",
        config.get("base_model", "nvidia/parakeet-ctc-0.6b"),
        "--max-steps",
        str(int(config.get("max_steps", 1000))),
        "--save-steps",
        str(int(config.get("save_steps", 25))),
        "--eval-steps",
        str(int(config.get("eval_steps", 25))),
        "--gradient-accumulation-steps",
        str(int(config.get("gradient_accumulation_steps", 16))),
    ]
    if config.get("bf16", True):
        command.append("--bf16")
    elif config.get("fp16", False):
        command.append("--fp16")
    return command


def serving_waiting(status: dict[str, Any]) -> bool:
    waiting = status.get("admission", {}).get("waiting", {})
    return any(int(waiting.get(tier, 0)) > 0 for tier in ("paid", "sub", "free"))


def run_next_segment() -> tuple[int, dict[str, Any]]:
    global RUNNING
    with LOCK:
        if RUNNING:
            return HTTPStatus.CONFLICT, {"error": "a training segment is already running"}
        pending = [job for job in load_jobs() if job.get("status") in {"queued", "preempted"}]
        if not pending:
            return HTTPStatus.NOT_FOUND, {"error": "no queued training job"}
        job = pending[0]
        RUNNING = True

    process: subprocess.Popen[bytes] | None = None
    preempted = False
    try:
        hold_asr_worker(True)
        status = request_json(GATEWAY_STATUS)
        free_vram = float(status.get("vram_free_gib", -1))
        if not status.get("vram_available") or free_vram < MIN_FREE_VRAM_GIB:
            job["status"] = "queued"
            job["last_reason"] = f"waiting for {MIN_FREE_VRAM_GIB:.1f} GiB free VRAM"
            job["updated_at"] = now()
            save_job(job)
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "status": "waiting_for_vram",
                "free_vram_gib": free_vram,
                "required_vram_gib": MIN_FREE_VRAM_GIB,
            }

        job["status"] = "running"
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["updated_at"] = now()
        save_job(job)
        process = subprocess.Popen(
            command_for(job),
            cwd=str(SCRIPT.parent.parent),
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(CHECK_INTERVAL_S)
            try:
                current = request_json(GATEWAY_STATUS)
            except (OSError, ValueError, urllib.error.URLError):
                current = {}
            if current and serving_waiting(current):
                preempted = True
                process.send_signal(signal.SIGUSR1)
                try:
                    process.wait(timeout=PREEMPT_GRACE_S)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                break
        code = process.wait()
        if preempted:
            job["status"] = "preempted"
            job["last_reason"] = "interactive serving demand"
        elif code == 0:
            job["status"] = "complete"
            job["completed_at"] = now()
        else:
            job["status"] = "failed"
            job["last_reason"] = f"trainer exited {code}"
        job["updated_at"] = now()
        save_job(job)
        return HTTPStatus.OK if code == 0 or preempted else HTTPStatus.INTERNAL_SERVER_ERROR, job
    except Exception as error:  # keep job state useful after controller failures
        job["status"] = "failed"
        job["last_reason"] = f"{type(error).__name__}: {error}"
        job["updated_at"] = now()
        save_job(job)
        return HTTPStatus.INTERNAL_SERVER_ERROR, job
    finally:
        if process and process.poll() is None:
            process.terminate()
        hold_asr_worker(False)
        with LOCK:
            RUNNING = False


def validate_job_config(config: dict[str, Any]) -> None:
    for field in ("manifest", "audio_root", "output_dir"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise ValueError(f"{field} is required")
    manifest = Path(config["manifest"]).resolve()
    audio_root = Path(config["audio_root"]).resolve()
    if not manifest.is_file():
        raise ValueError("manifest does not exist")
    if not audio_root.is_dir():
        raise ValueError("audio_root does not exist")
    # Output can be new, but must not be the source tree or root.
    output = Path(config["output_dir"]).resolve()
    if output in {Path("/"), SCRIPT.parent.parent.resolve()}:
        raise ValueError("unsafe output_dir")


class Handler(BaseHTTPRequestHandler):
    server_version = "omniserve-training-manager/1"
    protocol_version = "HTTP/1.1"

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        elif self.path in {"/status", "/v1/training/status"}:
            self.send_json(200, {"running": RUNNING, "jobs": load_jobs()})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path in {"/v1/training/jobs", "/jobs"}:
            try:
                config = self.body_json()
                validate_job_config(config)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            job = {
                "id": str(uuid.uuid4()),
                "status": "queued",
                "created_at": now(),
                "updated_at": now(),
                "attempts": 0,
                "config": config,
            }
            save_job(job)
            self.send_json(202, job)
        elif self.path in {"/v1/training/jobs/run", "/run"}:
            status, payload = run_next_segment()
            self.send_json(status, payload)
        else:
            self.send_json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"training-manager {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    host = os.getenv("OMNISERVE_TRAINING_BIND", "127.0.0.1")
    port = int(os.getenv("OMNISERVE_TRAINING_PORT", "9098"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"training-manager listening on {host}:{port} state={STATE_DIR}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
