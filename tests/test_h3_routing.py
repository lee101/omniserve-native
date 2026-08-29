#!/usr/bin/env python3
"""H3 video requests use the authenticated cost-aware app.nz seam."""

import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        payload = json.dumps(
            {
                "path": self.path,
                "request": json.loads(body),
                "authorization": self.headers.get("authorization"),
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def post(port, headers):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/video/generations",
        data=json.dumps({"prompt": "a cost-aware test"}).encode(),
        headers={"content-type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0

    upstream_port, gateway_port = free_port(), free_port()
    stub = http.server.ThreadingHTTPServer(("127.0.0.1", upstream_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_H3_UPSTREAM": f"http://127.0.0.1:{upstream_port}/api/cogs/h3",
        "OMNISERVE_NATIVE_H3_API_KEY": "appnz-service-key",
        "OMNISERVE_NATIVE_H3_PERMITS": "0",
    }
    for name in (
        "OMNISERVE_NATIVE_LLM_GGUF",
        "OMNISERVE_NATIVE_UPSTREAM",
        "OMNISERVE_NATIVE_IMAGE_UPSTREAM",
        "OMNISERVE_NATIVE_SECRET",
    ):
        env.pop(name, None)
    process = subprocess.Popen(
        [binary, "--port", str(gateway_port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{gateway_port}/status", timeout=2
                ) as response:
                    status = json.loads(response.read())
                break
            except Exception:
                if process.poll() is not None:
                    print("FAIL: gateway exited during startup")
                    return 1
                time.sleep(0.1)
        else:
            print("FAIL: gateway never became ready")
            return 1

        if status["upstreams"].get("h3") is not True:
            print("FAIL: H3 upstream missing from status", status)
            return 1
        if status["permits"].get("h3") != 0:
            print("FAIL: remote H3 relay consumed local permits", status)
            return 1

        code, _ = post(gateway_port, {})
        if code != 402:
            print("FAIL: free request reached a metered H3 tier", code)
            return 1

        code, response = post(
            gateway_port,
            {"X-Omniserve-Tier": "paid", "Authorization": "Bearer caller-key"},
        )
        expected = {
            "path": "/api/cogs/h3/predict-sync",
            "request": {"prompt": "a cost-aware test"},
            "authorization": "Bearer appnz-service-key",
        }
        if code != 200 or response != expected:
            print("FAIL: H3 relay mismatch", code, response)
            return 1

        print("H3 cost-aware routing tests passed")
        return 0
    finally:
        process.terminate()
        process.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
