#!/usr/bin/env python3
"""Pod-mode HTTP front end for the manifest workloads.

`runtime/handler.py` is the RunPod Serverless worker: it polls the provider API.
An app.nz Cog pod is driven over HTTP instead — the control plane boots the
image, waits for readiness on `/healthz`, then POSTs `{"input": {...}}` to
`/predictions` and reads `{"output": ...}`. Serving that contract here means the
same image and the same weights serve both tiers, and the serverless-vs-pod
choice stays entirely app.nz's decision.

Start it with the Cog pod command the template declares
(`cogSchema.DockerArgs`, e.g. `python -u /opt/omniserve/runtime/cog_pod.py`).
Requests are serialised because one pod owns one GPU context.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.handler import dispatch  # noqa: E402  (needs the path above)

PORT = int(os.getenv("COG_PORT", "5000"))
MAX_BODY = int(os.getenv("COG_MAX_BODY_BYTES", str(8 << 20)))
# The OmniServe image body, so a gateway relaying /v1/images/generations or
# /v1/images/edits can reach a pod without reshaping the request.
SCHEMA = {
    "outputKind": "image",
    "inputs": [
        {"name": "prompt", "type": "string", "required": True, "order": 0},
        {"name": "negative_prompt", "type": "string", "default": "", "order": 1},
        {"name": "width", "type": "integer", "default": 1024, "order": 2},
        {"name": "height", "type": "integer", "default": 1024, "order": 3},
        {"name": "steps", "type": "integer", "default": 20, "order": 4},
        {"name": "guidance_scale", "type": "number", "default": 1, "order": 5},
        {"name": "seed", "type": "integer", "order": 6},
        {"name": "image_base64", "type": "string", "order": 7},
        {"name": "strength", "type": "number", "default": 0.6, "order": 8},
        {"name": "output_format", "type": "string", "default": "webp", "order": 9},
    ],
}

_run_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "omniserve-cog-pod/1.0"

    def log_message(self, fmt, *args):  # keep the pod log to our own lines
        sys.stderr.write("[cog-pod] " + fmt % args + "\n")

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/healthz", "/health-check", "/health"):
            self._respond(200, {"ready": True, "status": "ready"})
        elif path == "/openapi.json":
            # app.nz introspects this when a deployment was registered without a
            # schema; returning the image contract keeps that path working too.
            self._respond(200, SCHEMA)
        elif path == "/":
            self._respond(200, {"service": "omniserve-cog-pod", "workloads": _workloads()})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/predictions":
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._respond(400, {"status": "failed", "error": "invalid content-length"})
            return
        if length <= 0 or length > MAX_BODY:
            self._respond(400, {"status": "failed", "error": f"body must be 1..{MAX_BODY} bytes"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except ValueError:
            self._respond(400, {"status": "failed", "error": "body must be JSON"})
            return
        values = payload.get("input", payload) if isinstance(payload, dict) else payload
        if not isinstance(values, dict):
            self._respond(400, {"status": "failed", "error": "input must be an object"})
            return
        try:
            with _run_lock:
                output = dispatch({"input": values})
        except BaseException as error:  # noqa: BLE001 - the pod reports, never dies
            self.log_message("prediction failed: %s", error)
            self._respond(200, {"status": "failed", "error": str(error)[:500]})
            return
        self._respond(200, {"status": "succeeded", "output": output})

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _workloads() -> list[str]:
    manifest = os.getenv("OMNISERVE_WORKLOAD_MANIFEST", "/opt/omniserve/workloads/workloads.json")
    try:
        with open(manifest, encoding="utf-8") as handle:
            return sorted(name for name in json.load(handle) if not name.startswith("_"))
    except OSError:
        return []


def main() -> int:
    host = os.getenv("COG_HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, PORT), Handler)
    server.daemon_threads = True
    print(f"[cog-pod] listening on {host}:{PORT}; workloads={_workloads()}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
