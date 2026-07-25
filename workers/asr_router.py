#!/usr/bin/env python3
"""Capacity-aware ASR relay.

The router prefers a local worker, then replays the original request to a
configured fallback when the local worker is unhealthy, lacks VRAM, is held
for training, or returns a transient capacity response. It never retries
caller errors because doing so can hide malformed audio.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LOCAL_UPSTREAM = os.getenv("OMNISERVE_ASR_LOCAL_UPSTREAM", "http://127.0.0.1:9097").rstrip("/")
FALLBACK_UPSTREAM = os.getenv("OMNISERVE_ASR_FALLBACK_UPSTREAM", "").rstrip("/")
GATEWAY_STATUS = os.getenv("OMNISERVE_ASR_GATEWAY_STATUS", "http://127.0.0.1:8791/status")
MIN_FREE_VRAM_GIB = float(os.getenv("OMNISERVE_ASR_MIN_FREE_VRAM_GIB", "3"))
TIMEOUT_S = float(os.getenv("OMNISERVE_ASR_TIMEOUT_S", "600"))
HEALTH_TIMEOUT_S = float(os.getenv("OMNISERVE_ASR_HEALTH_TIMEOUT_S", "1.5"))
TRANSIENT_STATUSES = {
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def local_capacity() -> tuple[bool, str]:
    try:
        health = fetch_json(f"{LOCAL_UPSTREAM}/health", HEALTH_TIMEOUT_S)
    except (OSError, ValueError, urllib.error.URLError):
        return False, "local_unhealthy"
    if not health.get("ready", False) or health.get("held", False):
        return False, "local_not_ready"
    if health.get("device") != "cuda":
        return True, "local_cpu"
    try:
        status = fetch_json(GATEWAY_STATUS, HEALTH_TIMEOUT_S)
        if not status.get("vram_available", False):
            return False, "vram_unknown"
        required = max(float(health.get("vram_required_gib", 0)), MIN_FREE_VRAM_GIB)
        if float(status.get("vram_free_gib", 0)) < required:
            return False, "vram_busy"
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return False, "gateway_unhealthy"
    return True, "local_gpu"


def relay(
    upstream: str,
    method: str,
    path: str,
    body: bytes,
    headers: dict[str, str],
) -> tuple[int, list[tuple[str, str]], bytes]:
    url = f"{upstream}{path}"
    request = urllib.request.Request(url, data=body, method=method)
    for name, value in headers.items():
        if name.lower() not in HOP_HEADERS:
            request.add_header(name, value)
    request.add_header("X-Omniserve-Internal", "local")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, list(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, list(error.headers.items()), error.read()


class Handler(BaseHTTPRequestHandler):
    server_version = "omniserve-asr-router/1"
    protocol_version = "HTTP/1.1"

    def json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if urllib.parse.urlsplit(self.path).path not in {"/health", "/status"}:
            self.json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        usable, reason = local_capacity()
        self.json_response(
            HTTPStatus.OK,
            {
                "status": "ok",
                "local_usable": usable,
                "reason": reason,
                "fallback_configured": bool(FALLBACK_UPSTREAM),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.json_response(HTTPStatus.BAD_REQUEST, {"error": "empty request"})
            return
        body = self.rfile.read(length)
        headers = {name: value for name, value in self.headers.items()}
        usable, reason = local_capacity()
        attempts: list[tuple[str, str]] = []

        if usable:
            try:
                status, response_headers, response_body = relay(
                    LOCAL_UPSTREAM, "POST", self.path, body, headers
                )
                attempts.append(("local", str(status)))
                if status not in TRANSIENT_STATUSES:
                    self.forward(status, response_headers, response_body, "local", attempts)
                    return
            except (OSError, socket.timeout, urllib.error.URLError) as error:
                attempts.append(("local", type(error).__name__))
        else:
            attempts.append(("local", reason))

        if not FALLBACK_UPSTREAM:
            self.json_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "local ASR unavailable and no fallback configured", "attempts": attempts},
            )
            return
        try:
            status, response_headers, response_body = relay(
                FALLBACK_UPSTREAM, "POST", self.path, body, headers
            )
            attempts.append(("fallback", str(status)))
            self.forward(status, response_headers, response_body, "fallback", attempts)
        except (OSError, socket.timeout, urllib.error.URLError) as error:
            attempts.append(("fallback", type(error).__name__))
            self.json_response(
                HTTPStatus.BAD_GATEWAY,
                {"error": "all ASR backends unavailable", "attempts": attempts},
            )

    def forward(
        self,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
        backend: str,
        attempts: list[tuple[str, str]],
    ) -> None:
        self.send_response(status)
        for name, value in headers:
            if name.lower() not in HOP_HEADERS:
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Omniserve-ASR-Backend", backend)
        self.send_header("X-Omniserve-ASR-Attempts", ",".join(f"{a}:{b}" for a, b in attempts))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"asr-router {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    host = os.getenv("OMNISERVE_ASR_BIND", "127.0.0.1")
    port = int(os.getenv("OMNISERVE_ASR_PORT", "9096"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(
        f"asr-router listening on {host}:{port} local={LOCAL_UPSTREAM} "
        f"fallback={'configured' if FALLBACK_UPSTREAM else 'disabled'}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
