#!/usr/bin/env python3
"""Chronos requests use the native forecast lane without changing payloads."""

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
        payload = json.dumps({
            "path": self.path,
            "request": json.loads(body),
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def request(port, path, method="POST"):
    body = {
        "values": [1.0, 2.0, 3.0],
        "prediction_length": 2,
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if method == "POST" else None,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read()), body
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), body


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
        "OMNISERVE_NATIVE_FORECAST_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        "OMNISERVE_NATIVE_FORECAST_PERMITS": "2",
        "OMNISERVE_NATIVE_SLOTS": "4",
    }
    for name in (
        "OMNISERVE_NATIVE_LLM_GGUF",
        "OMNISERVE_NATIVE_UPSTREAM",
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

        if status["upstreams"].get("forecast") is not True:
            print("FAIL: forecast upstream missing from status", status)
            return 1
        if status["permits"].get("forecast") != 2:
            print("FAIL: forecast permits missing from status", status)
            return 1

        for public_path, worker_path in (
            ("/forecast", "/forecast"),
            ("/forecast_batch", "/forecast_batch"),
            ("/v1/forecasts", "/forecast"),
        ):
            code, response, body = request(gateway_port, public_path)
            if code != 200 or response != {"path": worker_path, "request": body}:
                print("FAIL: forecast relay mismatch", public_path, code, response)
                return 1

        code, response, _ = request(gateway_port, "/forecast", method="GET")
        if code != 405 or response.get("error", {}).get("message") != "POST required":
            print("FAIL: forecast accepted a non-POST request", code, response)
            return 1

        with urllib.request.urlopen(
            f"http://127.0.0.1:{gateway_port}/v1/models", timeout=10
        ) as response:
            models = json.loads(response.read())["data"]
        if not any(model["id"] == "amazon/chronos-2" for model in models):
            print("FAIL: Chronos-2 missing from model inventory", models)
            return 1

        print("forecast routing tests passed")
        return 0
    finally:
        process.terminate()
        process.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
