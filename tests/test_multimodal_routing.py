#!/usr/bin/env python3
"""Structured OpenAI chat content must reach the multimodal upstream."""

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


MARKER = "GEMMA_MULTIMODAL_UPSTREAM"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Stub(http.server.BaseHTTPRequestHandler):
    posts_started = 0

    def do_GET(self):
        if self.path != "/api/v1/speech/catalog":
            self.send_response(404)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        payload = json.dumps({"providers": {"test": {"voices": ["a"]}}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        type(self).posts_started += 1
        if self.path == "/v1/images/generations":
            time.sleep(1.0)
            payload = json.dumps({"data": [{"b64_json": "AA=="}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        request = json.loads(body)
        payload = json.dumps({
            "marker": MARKER,
            "path": self.path,
            "content": request["messages"][-1]["content"],
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def post(port, content):
    payload = json.dumps({
        "model": "google/gemma-4-E4B-it",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get_json(port, path, timeout=2):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return response.status, json.loads(response.read())


def post_image(port, tier=None):
    payload = json.dumps({
        "prompt": "hold scheduler slot",
        "size": "512x512",
    }).encode()
    headers = {"content-type": "application/json"}
    if tier:
        headers["X-Omniserve-Tier"] = tier
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/images/generations",
        data=payload,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0

    upstream_port, gateway_port = free_port(), free_port()
    Stub.posts_started = 0
    stub = http.server.ThreadingHTTPServer(("127.0.0.1", upstream_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_MULTIMODAL_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        "OMNISERVE_NATIVE_IMAGE_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        "OMNISERVE_NATIVE_TTS_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        "OMNISERVE_NATIVE_SLOTS": "1",
        "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
        "OMNISERVE_NATIVE_MULTIMODAL_PERMITS": "1",
        "OMNISERVE_NATIVE_TTS_PERMITS": "1",
        "OMNISERVE_NATIVE_ADMISSION_TIMEOUT_S": "0.2",
    }
    for name in (
        "OMNISERVE_NATIVE_LLM_GGUF",
        "OMNISERVE_NATIVE_LLM_UPSTREAM",
        "OMNISERVE_NATIVE_AUX_UPSTREAM",
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
                urllib.request.urlopen(
                    f"http://127.0.0.1:{gateway_port}/v1/models", timeout=2
                ).read()
                break
            except Exception:
                if process.poll() is not None:
                    print("FAIL: gateway exited during startup")
                    return 1
                time.sleep(0.1)
        else:
            print("FAIL: gateway never became ready")
            return 1

        cases = [
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
             {"type": "text", "text": "describe"}],
            [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
             {"type": "text", "text": "transcribe"}],
            [{"type": "input_text", "text": "typed text"}],
        ]
        for content in cases:
            status, response = post(gateway_port, content)
            if status != 200 or response.get("marker") != MARKER or response.get("content") != content:
                print("FAIL: structured content was not preserved:", status, response)
                return 1

        status, response = post(gateway_port, "plain text stays local")
        error = response.get("error")
        error_message = error.get("message") if isinstance(error, dict) else error
        if status != 503 or error_message != "no LLM model loaded; start with OMNISERVE_NATIVE_LLM_GGUF":
            print("FAIL: plain text unexpectedly used multimodal upstream:", status, response)
            return 1

        posts_before_holder = Stub.posts_started
        holder = threading.Thread(target=post_image, args=(gateway_port,), daemon=True)
        holder.start()
        deadline = time.monotonic() + 2
        while Stub.posts_started <= posts_before_holder and time.monotonic() < deadline:
            time.sleep(0.01)
        status, response = get_json(gateway_port, "/api/v1/speech/catalog")
        if status != 200 or "providers" not in response:
            print("FAIL: speech catalog waited behind scheduler saturation:", status, response)
            return 1
        holder.join(timeout=3)
        if holder.is_alive():
            print("FAIL: scheduler slot holder did not complete")
            return 1

        process.terminate()
        process.wait(timeout=10)

        gateway_port = free_port()
        env["OMNISERVE_NATIVE_SLOTS"] = "2"
        process = subprocess.Popen(
            [binary, "--port", str(gateway_port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{gateway_port}/v1/models", timeout=2
                ).read()
                break
            except Exception:
                if process.poll() is not None:
                    print("FAIL: gateway exited during restart")
                    return 1
                time.sleep(0.1)
        else:
            print("FAIL: restarted gateway never became ready")
            return 1

        posts_before_holder = Stub.posts_started
        holder = threading.Thread(
            target=post_image, args=(gateway_port, "background"), daemon=True
        )
        holder.start()
        deadline = time.monotonic() + 2
        while Stub.posts_started <= posts_before_holder and time.monotonic() < deadline:
            time.sleep(0.01)
        status, response = post(
            gateway_port,
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
        )
        if status != 200 or response.get("marker") != MARKER:
            print("FAIL: background image request starved multimodal admission:", status, response)
            return 1
        holder.join(timeout=3)
        if holder.is_alive():
            print("FAIL: background scheduler slot holder did not complete")
            return 1
        print("multimodal routing tests passed")
        return 0
    finally:
        process.terminate()
        process.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
