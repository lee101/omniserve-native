import importlib.util
import json
import sys
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("asr_router", ROOT / "workers" / "asr_router.py")
router = importlib.util.module_from_spec(spec)
sys.modules["asr_router"] = router
assert spec.loader
spec.loader.exec_module(router)


class FakeBackend(BaseHTTPRequestHandler):
    post_status = 200
    text = "fake transcript"
    posts = 0

    def do_GET(self):
        body = json.dumps(
            {
                "ready": True,
                "held": False,
                "device": "cuda",
                "vram_required_gib": 1,
                "vram_available": True,
                "vram_free_gib": 24,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).posts += 1
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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class RouterTests(unittest.TestCase):
    def setUp(self):
        class Local(FakeBackend):
            pass

        class Fallback(FakeBackend):
            pass

        class Gateway(FakeBackend):
            pass

        self.Local, self.Fallback = Local, Fallback
        self.local = start(Local)
        self.fallback = start(Fallback)
        self.gateway = start(Gateway)
        router.LOCAL_UPSTREAM = f"http://127.0.0.1:{self.local.server_port}"
        router.FALLBACK_UPSTREAM = f"http://127.0.0.1:{self.fallback.server_port}"
        router.GATEWAY_STATUS = f"http://127.0.0.1:{self.gateway.server_port}/status"
        self.front = start(router.Handler)

    def tearDown(self):
        for server in (self.front, self.local, self.fallback, self.gateway):
            server.shutdown()
            server.server_close()

    def post(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.front.server_port}/v1/audio/transcriptions",
            data=b"fake audio",
            method="POST",
            headers={"Content-Type": "audio/wav"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.headers, json.load(response)

    def test_prefers_local_when_capacity_is_available(self):
        headers, body = self.post()
        self.assertEqual(headers["X-Omniserve-ASR-Backend"], "local")
        self.assertEqual(body["text"], "fake transcript")
        self.assertEqual(self.Local.posts, 1)
        self.assertEqual(self.Fallback.posts, 0)

    def test_transient_local_failure_replays_to_fallback(self):
        self.Local.post_status = 503
        self.Fallback.text = "managed fallback"
        headers, body = self.post()
        self.assertEqual(headers["X-Omniserve-ASR-Backend"], "fallback")
        self.assertEqual(body["text"], "managed fallback")
        self.assertEqual(self.Local.posts, 1)
        self.assertEqual(self.Fallback.posts, 1)


if __name__ == "__main__":
    unittest.main()
