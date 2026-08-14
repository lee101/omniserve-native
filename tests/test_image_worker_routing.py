#!/usr/bin/env python3
"""Legacy CuteDSL image contracts use their dedicated native proxy target."""

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
            "content_type": self.headers.get("content-type"),
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def post(port, path, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
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

    worker_port, gateway_port = free_port(), free_port()
    stub = http.server.ThreadingHTTPServer(("127.0.0.1", worker_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_IMAGE_WORKER_UPSTREAM": f"http://127.0.0.1:{worker_port}",
        "OMNISERVE_NATIVE_SLOTS": "4",
        "OMNISERVE_NATIVE_IMAGE_PERMITS": "2",
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

        if status["upstreams"].get("image_worker") is not True:
            print("FAIL: image worker missing from status", status)
            return 1

        cases = (
            ("/generate_image", {"prompt": "cat", "num_inference_steps": 8}),
            ("/caption", {"image_url": "https://example.invalid/image.webp"}),
            ("/nsfw_detect", {"image_base64": "AA=="}),
        )
        for path, body in cases:
            code, response = post(gateway_port, path, body)
            expected = {
                "path": path,
                "request": body,
                "content_type": "application/json",
            }
            if code != 200 or response != expected:
                print("FAIL: image worker relay mismatch", path, code, response)
                return 1

        print("image worker routing tests passed")
        return 0
    finally:
        process.terminate()
        process.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
