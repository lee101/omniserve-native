#!/usr/bin/env python3
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
    request_path = ""
    request_body = b""

    def do_POST(self):
        Stub.request_path = self.path
        Stub.request_body = self.rfile.read(int(self.headers.get("content-length") or 0))
        body = json.dumps({"depth_map": "data:image/png;base64,eA==", "model": "depth-anything-v2-small"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0
    upstream_port, gateway_port = free_port(), free_port()
    stub = http.server.HTTPServer(("127.0.0.1", upstream_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = {**os.environ, "OMNISERVE_NATIVE_BIND": "127.0.0.1", "OMNISERVE_NATIVE_DEPTH_UPSTREAM": f"http://127.0.0.1:{upstream_port}", "OMNISERVE_NATIVE_SLOTS": "2"}
    for key in ("OMNISERVE_NATIVE_LLM_GGUF", "OMNISERVE_NATIVE_LLM_UPSTREAM", "OMNISERVE_NATIVE_SECRET"):
        env.pop(key, None)
    process = subprocess.Popen([binary, "--port", str(gateway_port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{gateway_port}/v1/models", timeout=1).read()
                break
            except Exception:
                if process.poll() is not None:
                    return 1
                time.sleep(0.05)
        payload = json.dumps({"image_url": "https://cdn.test/image.webp"}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{gateway_port}/v1/depth-estimations", data=payload, headers={"content-type": "application/json", "x-omniserve-tier": "paid"})
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read())
        if not result.get("depth_map") or Stub.request_path != "/v1/depth-estimations" or Stub.request_body != payload:
            return 1
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{gateway_port}/v1/depth-estimations", timeout=5)
            return 1
        except urllib.error.HTTPError as error:
            if error.code != 405:
                return 1
        print("depth route tests passed")
        return 0
    finally:
        process.terminate()
        process.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
