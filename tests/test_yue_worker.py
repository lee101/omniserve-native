import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
spec = importlib.util.spec_from_file_location("yue_worker", ROOT / "workers/yue_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class YueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = patch.object(worker, "CACHE", Path(self.temp.name))
        self.cache.start()
        self.upload = patch.object(worker, "prepare_output_upload", return_value={"audio_url": "https://example.com/music.flac"})
        self.upload.start()
        self.request = worker.normalize_request({"style": "folk", "lyrics": "[Verse]\nMorning light"})
        self.output = {"audio_b64": "aGVsbG8=", "format": "flac", "truncated": {"semantic": True}}

    def tearDown(self):
        self.cache.stop()
        self.upload.stop()
        self.temp.cleanup()

    def test_quality_defaults_and_strict_limits(self):
        self.assertEqual(self.request["ode_steps"], 32)
        self.assertEqual(self.request["cot"], "full")
        for key, value in (("style", ""), ("lyrics", []), ("seed", True), ("seed", -1),
                           ("ode_steps", 0), ("semantic_max_tokens", 10000), ("cfg_scale", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.normalize_request({**self.request, key: value})

    def test_cache_separates_quality_model_and_seed(self):
        key = worker.cache_key(self.request)
        for name, value in (("seed", 4), ("ode_steps", 16), ("format", "mp3")):
            self.assertNotEqual(key, worker.cache_key({**self.request, name: value}))
        with patch.dict(os.environ, {"YUE_MODEL_REVISION": "new"}):
            self.assertNotEqual(key, worker.cache_key(self.request))

    def test_remote_only_when_local_declines(self):
        with patch.object(worker, "local_generate", return_value=self.output), patch.object(worker, "remote_generate") as remote:
            self.assertFalse(worker.generate(self.request)["cached"])
            remote.assert_not_called()
        with patch.object(worker, "local_generate") as local:
            cached = worker.generate(self.request)
            self.assertTrue(cached["cached"])
            self.assertTrue(cached["truncated"]["semantic"])
            local.assert_not_called()

    def test_busy_local_falls_back_without_changing_quality(self):
        with patch.object(worker, "local_generate", return_value=None), patch.object(worker, "remote_generate", return_value=self.output) as remote:
            worker.generate(self.request)
            remote.assert_called_once()
            self.assertEqual(remote.call_args.args, (self.request,))
            self.assertGreater(remote.call_args.kwargs["timeout"], 0)

    def test_expired_result_is_regenerated(self):
        with patch.object(worker, "local_generate", return_value=self.output) as local:
            worker.generate(self.request)
            path = worker.CACHE / (worker.cache_key(self.request) + ".json")
            expired = time.time() - 86401
            os.utime(path, (expired, expired))
            self.assertFalse(worker.generate(self.request)["cached"])
            self.assertEqual(local.call_count, 2)

    def test_unknown_local_failure_does_not_duplicate_on_remote(self):
        with patch.object(worker, "local_generate", side_effect=TimeoutError), patch.object(worker, "remote_generate") as remote:
            with self.assertRaises(TimeoutError):
                worker.generate(self.request)
            remote.assert_not_called()

    def test_singleflight_only_runs_once(self):
        entered = threading.Event()
        release = threading.Event()
        def local(_):
            entered.set()
            release.wait(3)
            return self.output
        with patch.object(worker, "local_generate", side_effect=local) as generate:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                first = pool.submit(worker.generate, self.request)
                self.assertTrue(entered.wait(2))
                second = pool.submit(worker.generate, self.request)
                time.sleep(.05)
                release.set()
                first.result()
                self.assertTrue(second.result()["cached"])
            self.assertEqual(generate.call_count, 1)

    def test_remote_failure_is_not_successful_audio(self):
        states = [{"id": "job-1", "status": "COMPLETED", "output": {"error": "failed"}}]
        with patch.object(worker, "remote_request", side_effect=states):
            with self.assertRaises(RuntimeError):
                worker.remote_generate(self.request)

    def test_remote_bucket_result_retains_metadata(self):
        output = {"audio_url": "https://example.com/music.flac", "format": "flac", "seconds": 60,
                  "truncated": {"semantic": False}}
        with patch.object(worker, "remote_request", return_value={"id": "job-1", "status": "COMPLETED", "output": output}) as request:
            result = worker.remote_generate(self.request)
            self.assertEqual(result["audio_url"], output["audio_url"])
            self.assertEqual(result["backend"], "runpod")
            self.assertFalse(result["truncated"]["semantic"])
            self.assertIn("output_upload", request.call_args.args[1]["input"])
        output["audio_url"] = "https://unexpected.example/music.flac"
        with patch.object(worker, "remote_request", return_value={"id": "job-1", "status": "COMPLETED", "output": output}):
            with self.assertRaises(RuntimeError):
                worker.remote_generate(self.request)

    def test_remote_timeout_cancels_job(self):
        with patch.object(worker, "remote_request", return_value={"id": "job-1", "status": "IN_QUEUE"}) as request, patch.object(worker, "TIMEOUT", -1):
            with self.assertRaises(TimeoutError):
                worker.remote_generate(self.request)
            self.assertEqual(request.call_args.args, ("/cancel/job-1", {}))

    def test_insufficient_memory_never_starts_a_model(self):
        with patch.object(worker, "gpu_memory", return_value=(10000, 32000)), patch.object(worker.multiprocessing, "get_context") as process:
            self.assertIsNone(worker.local_generate(self.request))
            process.assert_not_called()

    def test_http_auth_and_validation_precede_generation(self):
        server = worker.ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with patch.dict(os.environ, {"YUE_WORKER_SECRET": "test-key"}), patch.object(worker, "generate", return_value=self.output) as generate:
                url = f"http://127.0.0.1:{server.server_port}/v1/music/generations"
                for token, payload, expected in (("wrong", self.request, 401),
                                                 ("caf\xe9", self.request, 401),
                                                 ("test-key", {}, 400),
                                                 ("test-key", self.request, 200)):
                    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(request, timeout=3) as response:
                            code = response.status
                    except urllib.error.HTTPError as error:
                        code = error.code
                        error.close()
                    self.assertEqual(code, expected)
                generate.assert_called_once_with(self.request)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
