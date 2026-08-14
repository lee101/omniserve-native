import json
import os
import socket
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("OMNISERVE_ASR_ROUTER_BIN", ROOT / "build-ci" / "omniserve-asr-router"))


class FakeBackend(BaseHTTPRequestHandler):
    post_status = 200
    text = "fake transcript"
    posts = 0
    last_internal = ""
    last_headers = {}
    last_header_list = []
    received_port = 0
    health = {
        "ready": True,
        "held": False,
        "device": "cuda",
        "vram_required_gib": 1,
        "vram_available": True,
        "vram_free_gib": 24,
    }

    def do_GET(self):
        body = json.dumps(type(self).health).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).posts += 1
        type(self).last_internal = self.headers.get("X-Omniserve-Internal", "")
        type(self).last_headers = dict(self.headers.items())
        type(self).last_header_list = list(self.headers.raw_items())
        type(self).received_port = self.server.server_port
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps({"text": type(self).text}).encode()
        self.send_response(type(self).post_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def start(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class NativeRouterTests(unittest.TestCase):
    def setUp(self):
        if not BINARY.is_file():
            self.skipTest(f"native ASR router not built: {BINARY}")

        class Local(FakeBackend):
            pass

        class Fallback(FakeBackend):
            pass

        class Gateway(FakeBackend):
            pass

        self.Local, self.Fallback, self.Gateway = Local, Fallback, Gateway
        for backend in (Local, Fallback, Gateway):
            backend.posts = 0
            backend.post_status = 200
            backend.text = "fake transcript"
            backend.health = dict(FakeBackend.health)

        self.local = start(Local)
        self.fallback = start(Fallback)
        self.gateway = start(Gateway)
        self.port = free_port()
        env = {
            **os.environ,
            "OMNISERVE_ASR_BIND": "127.0.0.1",
            "OMNISERVE_ASR_PORT": str(self.port),
            "OMNISERVE_ASR_LOCAL_UPSTREAM": f"http://127.0.0.1:{self.local.server_port}",
            "OMNISERVE_ASR_FALLBACK_UPSTREAM": f"http://127.0.0.1:{self.fallback.server_port}",
            "OMNISERVE_ASR_GATEWAY_STATUS": f"http://127.0.0.1:{self.gateway.server_port}/status",
            "OMNISERVE_ASR_HEALTH_TIMEOUT_S": "0.5",
            "OMNISERVE_ASR_TIMEOUT_S": "3",
        }
        self.router = subprocess.Popen(
            [str(BINARY)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=0.2):
                    break
            except (OSError, urllib.error.URLError):
                if self.router.poll() is not None:
                    error = self.router.stderr.read().decode(errors="replace")
                    self.fail(f"native router exited during startup: {error}")
                time.sleep(0.03)
        else:
            self.fail("native router did not become healthy")

    def tearDown(self):
        if hasattr(self, "router"):
            self.router.terminate()
            try:
                self.router.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.router.kill()
                self.router.wait(timeout=2)
            if self.router.stderr:
                self.router.stderr.close()
        for name in ("local", "fallback", "gateway"):
            server = getattr(self, name, None)
            if server:
                server.shutdown()
                server.server_close()

    def post(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/audio/transcriptions?language=en",
            data=b"fake audio",
            method="POST",
            headers={"Content-Type": "audio/wav"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, json.load(response)

    def test_prefers_local_and_marks_internal_request(self):
        status, headers, body = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Omniserve-ASR-Backend"], "local")
        self.assertEqual(headers["X-Omniserve-ASR-Attempts"], "local:200")
        self.assertEqual(body["text"], "fake transcript")
        self.assertEqual(
            self.Local.last_internal, "local",
            {"headers": self.Local.last_header_list, "received_port": self.Local.received_port,
             "local_port": self.local.server_port, "router_port": self.port},
        )
        self.assertEqual(self.Local.posts, 1)
        self.assertEqual(self.Fallback.posts, 0)

    def test_transient_local_failure_replays_to_fallback(self):
        self.Local.post_status = 503
        self.Fallback.text = "managed fallback"
        status, headers, body = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Omniserve-ASR-Backend"], "fallback")
        self.assertEqual(headers["X-Omniserve-ASR-Attempts"], "local:503,fallback:200")
        self.assertEqual(body["text"], "managed fallback")
        self.assertEqual(self.Local.posts, 1)
        self.assertEqual(self.Fallback.posts, 1)

    def test_capacity_gate_skips_local_worker(self):
        self.Gateway.health["vram_free_gib"] = 0.5
        self.Fallback.text = "capacity fallback"
        status, headers, body = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Omniserve-ASR-Backend"], "fallback")
        self.assertIn("local:vram_busy", headers["X-Omniserve-ASR-Attempts"])
        self.assertEqual(body["text"], "capacity fallback")
        self.assertEqual(self.Local.posts, 0)
        self.assertEqual(self.Fallback.posts, 1)

    def test_caller_error_is_not_retried(self):
        self.Local.post_status = 400
        status, headers, _ = self.post()
        self.assertEqual(status, 400)
        self.assertEqual(headers["X-Omniserve-ASR-Backend"], "local")
        self.assertEqual(self.Local.posts, 1)
        self.assertEqual(self.Fallback.posts, 0)


if __name__ == "__main__":
    unittest.main()
