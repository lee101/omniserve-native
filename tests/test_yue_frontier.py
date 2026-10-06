import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
spec = importlib.util.spec_from_file_location("yue_worker_frontier", ROOT / "workers/yue_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class YueFrontierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.routing = Path(self.tmp.name) / "routing.json"
        self.routing.write_text(json.dumps({"workloads": {"yue": {
            "candidates": [{"id": "local", "kind": "local", "p50_ms": 1000},
                           {"id": "runpod:e", "kind": "runpod", "endpoint": "e", "p50_ms": 3000}],
            "tiers": {"paid": {"policy": "cheapest_within_deadline", "deadline_ms": 60000, "remote_order": ["runpod:e"]},
                      "sub": {"policy": "overflow_on_busy", "remote_order": ["runpod:e"]}}}}}))
        env = {"FRONTIER_LEDGER": "1", "FRONTIER_LEDGER_DB": str(Path(self.tmp.name) / "l.db"),
               "FRONTIER_ROUTING_FILE": str(self.routing), "YUE_FRONTIER_ROUTING": "1", "YUE_RUNPOD_ENDPOINT_ID": "e"}
        self.env = patch.dict(os.environ, env)
        self.env.start()
        worker.frontier.ledger._LEDGER = None
        worker.frontier.router._ROUTER = None
        self.cache = patch.object(worker, "CACHE", Path(self.tmp.name) / "cache")
        self.cache.start()
        self.request = worker.normalize_request({"style": "folk", "lyrics": "[Verse]\nx"})

    def tearDown(self):
        self.cache.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_busy(self, tier, seed):
        worker.LOCAL_LOCK.acquire()
        worker.LOCAL_STARTED = time.monotonic()
        threading.Timer(0.3, worker.LOCAL_LOCK.release).start()

        def fake_local(request, wait_s=0.0):
            if not worker.LOCAL_LOCK.acquire(timeout=wait_s) if wait_s else not worker.LOCAL_LOCK.acquire(blocking=False):
                return None
            worker.LOCAL_LOCK.release()
            return {"audio_b64": "eA==", "backend": "local"}

        def fake_remote(request, timeout=None):
            worker.REMOTE_TIMING.value = {"queue_ms": 50, "exec_ms": 2000, "job_id": "j", "status": "COMPLETED"}
            return {"audio_url": "u", "backend": "runpod"}

        with patch.object(worker, "local_generate", side_effect=fake_local), \
                patch.object(worker, "remote_generate", side_effect=fake_remote):
            return worker.generate({**self.request, "seed": seed}, tier)

    def test_paid_waits_for_short_local_queue_and_sub_overflows(self):
        self.assertEqual(self.run_busy("paid", 1)["backend"], "local")
        time.sleep(0.4)
        self.assertEqual(self.run_busy("sub", 2)["backend"], "runpod")
        time.sleep(0.4)
        self.assertEqual(worker.generate({**self.request, "seed": 2}, "sub")["cached"], True)
        led = worker.frontier.get_ledger()
        led.flush()
        rows = led.conn().execute("SELECT backend, endpoint, tier, queue_ms, exec_ms, cache_hit FROM jobs ORDER BY id").fetchall()
        self.assertEqual([(r[0], r[5]) for r in rows], [("local", 0), ("runpod", 0), ("local", 1)])
        self.assertEqual(rows[1][1:5], ("e", "sub", 50, 2000))

    def test_flag_off_keeps_legacy_overflow(self):
        with patch.dict(os.environ, {"YUE_FRONTIER_ROUTING": "0"}):
            self.assertEqual(self.run_busy("paid", 3)["backend"], "runpod")
        time.sleep(0.4)


if __name__ == "__main__":
    unittest.main()
