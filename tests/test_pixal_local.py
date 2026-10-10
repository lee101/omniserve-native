import io
import json
import sys
import tempfile
import unittest
import urllib.error
from http import HTTPStatus
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
import pixal_local  # noqa: E402
import trellis2_worker as worker  # noqa: E402


class Resp(io.BytesIO):
    def __init__(self, body: dict, status: int = 200):
        super().__init__(json.dumps(body).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def daemon_ok(root: Path):
    def open_url(request, timeout=0):
        out = Path(json.loads(request.data)["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"glb")
        return Resp({"seed": 7, "resolution": 1024, "timings": {"total": 60.0}, "peak_gb": 14.2, "lease_wait_ms": 10})
    return open_url


def daemon_busy(request, timeout=0):
    raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b'{"error":"gpu_busy","retry_after_seconds":60}'))


class LocalRunTests(unittest.TestCase):
    def test_success_builds_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, body = pixal_local.run("m", "https://x/a.png", 1536, 1024, 200000, 7, Path(tmp), daemon_ok(Path(tmp)))
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(body["backend"], "local")
        self.assertTrue(body["model_glb"]["url"].startswith("/api/3d-assets/3d-local-"))
        self.assertEqual(body["model_glb"]["file_size"], 3)

    def test_busy_passes_status_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, body = pixal_local.run("m", "https://x/a.png", 1024, 1024, 200000, 7, Path(tmp), daemon_busy)
        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "gpu_busy")

    def test_unreachable_daemon_is_unavailable(self):
        def boom(request, timeout=0):
            raise OSError("refused")
        with tempfile.TemporaryDirectory() as tmp:
            status, body = pixal_local.run("m", "https://x/a.png", 1024, 1024, 200000, 7, Path(tmp), boom)
        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "pixal_unavailable")

    def test_ready_false_when_unreachable_or_disabled(self):
        def boom(url, timeout=0):
            raise OSError("refused")
        self.assertFalse(pixal_local.ready(boom))
        with mock.patch.dict(pixal_local.os.environ, {"OMNISERVE_3D_PIXAL_LOCAL": "0"}):
            self.assertFalse(pixal_local.ready(lambda url, timeout=0: Resp({"ready": True})))
        self.assertTrue(pixal_local.ready(lambda url, timeout=0: Resp({"ready": True})))


class RoutingTests(unittest.TestCase):
    ARGS = ("https://x/a.png", 1024, 1024, 200000, 7, None)

    def run_pixal(self, local, ready=True, remote_ok=True, mode="fallback"):
        remote = mock.Mock(return_value=(200, {"backend": "runpod"}))
        with (
            mock.patch.object(worker.pixal_local, "ready", return_value=ready),
            mock.patch.object(worker.pixal_local, "run", return_value=local),
            mock.patch.object(worker.remote_3d, "run", remote),
            mock.patch.object(worker.remote_3d, "configured", return_value=remote_ok),
            mock.patch.object(worker.remote_3d, "should_route", return_value=remote_ok and mode != "off"),
            mock.patch.dict(worker.os.environ, {"OMNISERVE_3D_PIXAL_REMOTE_MODE": mode}),
        ):
            return worker.run_pixal(*self.ARGS), remote

    def test_local_success_never_touches_runpod(self):
        (status, body), remote = self.run_pixal((200, {"backend": "local"}))
        self.assertEqual(body["backend"], "local")
        remote.assert_not_called()

    def test_busy_local_falls_back_to_runpod(self):
        (status, body), remote = self.run_pixal((503, {"error": "gpu_busy"}))
        self.assertEqual(body["backend"], "runpod")
        remote.assert_called_once()

    def test_daemon_down_falls_back_to_runpod(self):
        (status, body), remote = self.run_pixal((503, {}), ready=False)
        self.assertEqual(body["backend"], "runpod")

    def test_no_runpod_returns_local_failure(self):
        (status, body), remote = self.run_pixal((503, {"error": "gpu_busy"}), remote_ok=False)
        self.assertEqual((status, body["error"]), (503, "gpu_busy"))
        remote.assert_not_called()

    def test_off_mode_never_uses_runpod(self):
        (status, body), remote = self.run_pixal((503, {"error": "gpu_busy"}), mode="off")
        self.assertEqual(status, 503)
        remote.assert_not_called()

    def test_always_mode_skips_local(self):
        (status, body), remote = self.run_pixal((200, {"backend": "local"}), mode="always")
        self.assertEqual(body["backend"], "runpod")


if __name__ == "__main__":
    unittest.main()
