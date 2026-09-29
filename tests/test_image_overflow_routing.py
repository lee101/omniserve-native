#!/usr/bin/env python3
"""The embedded image lane overflows to a standing remote when saturated.

Covers the ra2 wiring: a saturated local lane, a lane whose model never loaded,
both routes the lane serves (`/v1/images/generations` and `/v1/images/edits`),
and the two properties that make the relay safe — the body crosses unchanged and
the caller's credential never reaches the metered remote.
"""

import base64
import concurrent.futures
import http.server
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

SERVICE_KEY = "appnz-service-key"
CALLER_KEY = "caller-secret"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OverflowStub(http.server.BaseHTTPRequestHandler):
    """Records what the gateway actually sent, then answers like the cog seam."""

    seen: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server contract
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        OverflowStub.seen.append({
            "path": self.path,
            "authorization": self.headers.get("authorization"),
            "x_api_key": self.headers.get("x-api-key"),
            "secret": self.headers.get("secret"),
            "rapid": self.headers.get("x-rapid-api-key"),
            "tier": self.headers.get("x-omniserve-tier"),
            "body": body.decode("utf-8", "replace"),
        })
        payload = json.dumps({
            "overflow": True,
            "path": self.path,
            "received": json.loads(body),
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def tiny_png() -> str:
    """A real PNG: the gateway decodes image_base64 before it can overflow."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def post(port, path, payload, headers=None, timeout=30):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def status(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as response:
        return json.loads(response.read())


def wait_ready(port, process, key="diffusion", timeout=15):
    """Wait for /status; `key=None` only requires that the gateway answers.

    The bare gateway has no diffusion block at all, and an overflow relay is the
    first thing it can serve.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("gateway exited during startup")
        try:
            snapshot = status(port)
            if key is None or snapshot[key]["ready"]:
                return snapshot
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"gateway never became ready on {port}")


def wait_busy(port, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if status(port)["admission"]["active"] >= 1:
            return
        time.sleep(0.01)
    raise AssertionError("the local lane never occupied its only permit")


def gateway_env(binary_port: int, stub_lib: str, upstream_port: int, *, model: str):
    env = {key: value for key, value in os.environ.items() if not key.startswith("OMNISERVE_NATIVE_")}
    env.update({
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_SLOTS": "1",
        "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
        "OMNISERVE_NATIVE_SECRET": CALLER_KEY,
        "OMNISERVE_NATIVE_SD_MIN_FREE_MB": "0",
        "OMNISERVE_NATIVE_SD_IMAGE_FORMAT": "png",
        "OMNISERVE_NATIVE_SD_LIB": stub_lib,
        "OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM": f"http://127.0.0.1:{upstream_port}/api/cogs/ra2-cog",
        "OMNISERVE_NATIVE_IMAGE_OVERFLOW_API_KEY": SERVICE_KEY,
        "OMNISERVE_NATIVE_OVERFLOW_TIERS": "paid",
        "OMNISERVE_NATIVE_IMAGE_OVERFLOW_TIMEOUT_MS": "20000",
    })
    if model:
        env["OMNISERVE_NATIVE_SD_MODEL"] = model
        env["OMNISERVE_NATIVE_SD_REFERENCE_EDIT"] = "1"
    return env


def main() -> int:
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    stub_lib = os.environ.get("OMNISERVE_SD_STUB")
    if not binary or not os.path.exists(binary) or not stub_lib or not os.path.exists(stub_lib):
        print("skip: OMNISERVE_NATIVE_BIN / OMNISERVE_SD_STUB not set")
        return 0

    upstream_port, gateway_port, bare_port = free_port(), free_port(), free_port()
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", upstream_port), OverflowStub)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    bodies = {
        "generations": {"prompt": "a cost-aware test", "width": 64, "height": 64,
                        "steps": 2, "seed": 41, "cache": True},
        "edits": {"prompt": "repaint in ink", "width": 64, "height": 64, "steps": 2,
                  "seed": 42, "image_base64": tiny_png()},
    }
    paid = {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "paid",
            "X-API-Key": CALLER_KEY}

    processes = []
    with tempfile.TemporaryFile() as log:
        gateway = subprocess.Popen([binary, "--port", str(gateway_port)],
                                   env=gateway_env(gateway_port, stub_lib, upstream_port,
                                                   model="cache-test.gguf"),
                                   stdout=log, stderr=log)
        processes.append(gateway)
        bare = subprocess.Popen([binary, "--port", str(bare_port)],
                                env=gateway_env(bare_port, stub_lib, upstream_port, model=""),
                                stdout=log, stderr=log)
        processes.append(bare)
        try:
            snapshot = wait_ready(gateway_port, gateway)
            if snapshot["overflow"]["image"] is not True:
                print("FAIL: image overflow missing from status", snapshot.get("overflow"))
                return 1
            if snapshot["upstreams"].get("image") is not False:
                print("FAIL: embedded lane reported a proxied image upstream", snapshot["upstreams"])
                return 1

            # 1. Saturated lane: a paid request takes the remote instead of waiting.
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                for route, path, body in (
                    ("generations", "/v1/images/generations", bodies["generations"]),
                    ("edits", "/v1/images/edits", bodies["edits"]),
                ):
                    occupying = pool.submit(post, gateway_port, path,
                                            {**body, "seed": body["seed"] + 100}, paid)
                    wait_busy(gateway_port)
                    code, relayed = post(gateway_port, path, body, paid)
                    if code != 200 or not relayed.get("overflow"):
                        print(f"FAIL: {route} was not relayed while the lane was busy", code, relayed)
                        return 1
                    local_code, _ = occupying.result()
                    if local_code != 200:
                        print(f"FAIL: the occupying {route} request failed", local_code)
                        return 1
                snapshot = status(gateway_port)
                if snapshot["overflow"]["saturated"] != 2:
                    print("FAIL: saturation was not counted", snapshot["overflow"])
                    return 1

            # 2. Free traffic queues locally and never reaches a metered remote.
            before = len(OverflowStub.seen)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                occupying = pool.submit(post, gateway_port, "/v1/images/generations",
                                        {**bodies["generations"], "seed": 7},
                                        {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "paid"})
                wait_busy(gateway_port)
                code, local = post(gateway_port, "/v1/images/generations",
                                   {**bodies["generations"], "seed": 8},
                                   {"Authorization": f"Bearer {CALLER_KEY}"})
                occupying.result()
            if code != 200 or local.get("overflow"):
                print("FAIL: free traffic did not stay local", code, local)
                return 1
            if len(OverflowStub.seen) != before:
                print("FAIL: free traffic reached the metered remote")
                return 1

            # 3. No local model at all: the same relay, counted as a local failure.
            wait_ready(bare_port, bare, key=None)
            code, relayed = post(bare_port, "/v1/images/generations", bodies["generations"],
                                 {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "paid"})
            if code != 200 or not relayed.get("overflow"):
                print("FAIL: a lane without a model did not relay", code, relayed)
                return 1
            if status(bare_port)["overflow"]["local_failed"] != 1:
                print("FAIL: local failure was not counted", status(bare_port)["overflow"])
                return 1

            # 4. What the remote saw: unchanged body, our credential, no caller key.
            if len(OverflowStub.seen) != 3:
                print("FAIL: unexpected number of remote calls", OverflowStub.seen)
                return 1
            for seen, (label, body) in zip(OverflowStub.seen, (
                ("generations", bodies["generations"]),
                ("edits", bodies["edits"]),
                ("no-model", bodies["generations"]),
            )):
                if seen["path"] != "/api/cogs/ra2-cog/predict-sync":
                    print(f"FAIL: {label} path is wrong", seen["path"])
                    return 1
                if seen["authorization"] != f"Bearer {SERVICE_KEY}":
                    print(f"FAIL: {label} did not present the service credential", seen["authorization"])
                    return 1
                if seen["x_api_key"] or seen["secret"] or seen["rapid"]:
                    print(f"FAIL: {label} leaked a caller credential", seen)
                    return 1
                if json.loads(seen["body"]) != body:
                    print(f"FAIL: {label} body changed in flight", seen["body"])
                    return 1
            # The tier header is this gateway's own decision and is not relayed.
            if any(seen["tier"] for seen in OverflowStub.seen):
                print("FAIL: X-Omniserve-Tier reached the remote")
                return 1
            print("image overflow: saturation, edits, local failure, credential stripping ok")
            return 0
        finally:
            for process in processes:
                process.terminate()
                process.wait(timeout=10)
            upstream.shutdown()


if __name__ == "__main__":
    sys.exit(main())
