"""mg_image.py: retry/backoff on transient API errors, local-lane fallback, file writing."""
import base64, http.server, json, os, subprocess, sys, tempfile, threading, unittest

TOOL = os.path.join(os.path.dirname(__file__), "..", "tools", "mg_image.py")
START = os.path.join(os.path.dirname(__file__), "fake_lane_start.sh")
WEBP = b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"VP8 " + bytes(100)


def serve(handler_cls):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def handler(statuses, hits, bodies):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            hits.append(1)
            status = statuses[min(len(hits) - 1, len(statuses) - 1)]
            payload = {"error": "nope"} if status != 200 else {
                "credits_used": 4, "result": {"engine": "x", "images": [{"image_base64": base64.b64encode(WEBP).decode()}]}}
            if self.path.endswith("/v1/images/generations") and status == 200:
                payload = {"data": [{"b64_json": base64.b64encode(WEBP).decode()}]}
            raw = json.dumps(payload).encode()
            self.send_response(status); self.end_headers(); self.wfile.write(raw)

        def log_message(self, *args):
            pass
    return H


def run(api, lane, *extra, cwd):
    return subprocess.run([sys.executable, "-S", TOOL, "--api", api, "--key", "k", "--lane", lane,
                           "--out-dir", cwd, "--retries", "2", *extra, "--", "a fox"],
                          capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONHASHSEED": "0", "MG_LANE_START": START})


class MgImage(unittest.TestCase):
    def test_retries_then_succeeds(self):
        hits, bodies = [], []
        api, url = serve(handler([502, 503, 200], hits, bodies))
        with tempfile.TemporaryDirectory() as d:
            r = run(url, "http://127.0.0.1:9", "--quality", "hq", "--steps", "24", cwd=d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(hits), 3)
            self.assertEqual(bodies[-1]["quality"], "hq")
            self.assertEqual(bodies[-1]["num_steps"], 24)
            self.assertTrue(r.stdout.strip().endswith(".webp"))
            self.assertEqual(open(r.stdout.strip(), "rb").read(), WEBP)

    def test_falls_back_to_local_lane(self):
        hits, lane_bodies = [], []
        api, url = serve(handler([502], hits, []))
        lane, lane_url = serve(handler([200], [], lane_bodies))
        with tempfile.TemporaryDirectory() as d:
            r = run(url, lane_url, "--quality", "hq", cwd=d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(hits), 2)
            self.assertEqual(lane_bodies[0]["turbo"], False)
            self.assertIn("local", r.stderr)

    def test_down_lane_is_started_while_retrying(self):
        hits = []
        api, url = serve(handler([502], hits, []))
        with tempfile.TemporaryDirectory() as d:
            r = run(url, "http://127.0.0.1:9", cwd=d)
            self.assertEqual(r.returncode, 1)
            self.assertEqual(len(hits), 3)
            self.assertIn("starting the lane while retrying", r.stderr)
            self.assertIn("fake lane start", r.stderr)

    def test_no_fallback_fails(self):
        api, url = serve(handler([502], [], []))
        lane, lane_url = serve(handler([200], [], []))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run(url, lane_url, "--no-fallback", cwd=d).returncode, 1)

    def test_auth_rejection_is_not_retried(self):
        hits = []
        api, url = serve(handler([401], hits, []))
        lane, lane_url = serve(handler([200], [], []))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run(url, lane_url, cwd=d).returncode, 77)
            self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
