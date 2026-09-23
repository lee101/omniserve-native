import os
import sys
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
import remote_3d  # noqa: E402


class Remote3DTests(unittest.TestCase):
    def test_should_route(self):
        with mock.patch.dict(os.environ, {"OMNISERVE_3D_PIXAL_REMOTE_ENDPOINT": "", "RUNPOD_API_KEY": "k"}):
            self.assertFalse(remote_3d.should_route(False))
        env = {"OMNISERVE_3D_PIXAL_REMOTE_ENDPOINT": "e", "RUNPOD_API_KEY": "k", "OMNISERVE_3D_PIXAL_REMOTE_MODE": "fallback"}
        with mock.patch.dict(os.environ, env):
            self.assertTrue(remote_3d.should_route(False))
            self.assertTrue(remote_3d.should_route(True, 1000, 23040))
            self.assertFalse(remote_3d.should_route(True, 30000, 23040))
        with mock.patch.dict(os.environ, {**env, "OMNISERVE_3D_PIXAL_REMOTE_MODE": "off"}):
            self.assertFalse(remote_3d.should_route(False))

    def test_run_maps_output_and_records(self):
        calls = []
        states = iter([{"status": "IN_QUEUE"}, {"status": "COMPLETED", "delayTime": 900, "executionTime": 80000,
                                                "output": {"glb_url": "https://cdn/x.glb", "seed": 5, "resolution": 1024,
                                                           "timings": {"total": 79}}}])

        def fake(method, path, body=None):
            calls.append((method, path, body))
            return {"id": "job-1"} if path == "/run" else next(states)

        recorded = []
        with mock.patch.dict(os.environ, {"OMNISERVE_3D_PIXAL_REMOTE_ENDPOINT": "e"}), \
                mock.patch.object(remote_3d, "ledger", side_effect=lambda **r: recorded.append(r)), \
                tempfile.TemporaryDirectory() as tmp:
            status, body = remote_3d.run("TencentARC/Pixal3D", "https://img/a.png", 512, 1024, 200000, 5, Path(tmp),
                                         call=fake, sleep=lambda s: None)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(body["model_glb"]["url"], "https://cdn/x.glb")
        self.assertEqual(calls[0][2]["input"]["resolution"], 1024)
        self.assertEqual(calls[0][2]["input"]["decimation_target"], 200000)
        self.assertEqual(recorded[0]["status"], "COMPLETED")
        self.assertEqual((recorded[0]["queue_ms"], recorded[0]["exec_ms"]), (900, 80000))

    def test_failure_is_bad_gateway(self):
        def fake(method, path, body=None):
            return {"id": "j"} if path == "/run" else {"status": "FAILED", "output": {"error": "oom"}}

        with mock.patch.object(remote_3d, "ledger"), tempfile.TemporaryDirectory() as tmp:
            status, body = remote_3d.run("m", "https://i", 1024, 1024, 200000, 1, Path(tmp), call=fake, sleep=lambda s: None)
        self.assertEqual(status, HTTPStatus.BAD_GATEWAY)
        self.assertIn("oom", body["message"])


if __name__ == "__main__":
    unittest.main()
