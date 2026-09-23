#!/usr/bin/env python3
"""runpod_overflow adapter: concurrent jobs do not serialize, the frontier picks the
endpoint per tier, the primary breaker skips a dead primary, and every job lands
in the ledger with RunPod's delay/execution split."""

import concurrent.futures
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

FRONTIER_LIB = os.environ.get("FRONTIER_LIB", "/nvme0n1-disk/code/omniserve-native")
TOOL = Path(__file__).resolve().parents[1] / "tools" / "runpod_overflow.py"


def load(env):
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("runpod_overflow_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(Path(FRONTIER_LIB, "frontier").is_dir(), "frontier lib not present")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        routing = Path(self.tmp.name) / "routing.json"
        routing.write_text(json.dumps({"workloads": {"ra2": {"candidates": [], "tiers": {
            "paid": {"policy": "cheapest_within_deadline", "remote_order": ["runpod:cheap", "runpod:fast"]},
            "sub": {"policy": "fastest", "remote_order": ["runpod:fast", "runpod:cheap"]}}}}}))
        self.mod = load({"RUNPOD_API_KEY": "k", "RUNPOD_RA2_ENDPOINTS": "cheap,fast", "FRONTIER_ROUTING": "1",
                         "FRONTIER_LEDGER": "1", "FRONTIER_LEDGER_DB": str(Path(self.tmp.name) / "l.db"),
                         "FRONTIER_ROUTING_FILE": str(routing), "FRONTIER_LIB": FRONTIER_LIB,
                         "PRIMARY_UPSTREAM": "http://127.0.0.1:9", "PRIMARY_BREAKER_S": "60"})
        self.mod.frontier.ledger._LEDGER = None
        self.mod.frontier.router._ROUTER = None
        self.calls = []
        self.submitted = []
        jobs = {}

        def fake(method, path, body=None, endpoint=None):
            self.calls.append((method, path, endpoint))
            if path == "/run":
                self.submitted.append((method, path, body))
                job = f"{endpoint}-{len(jobs)}"
                jobs[job] = time.monotonic()
                return {"id": job}
            job = path.rsplit("/", 1)[-1]
            if time.monotonic() - jobs[job] < 1.0:
                return {"status": "IN_PROGRESS"}
            return {"status": "COMPLETED", "delayTime": 12, "executionTime": 1000,
                    "output": {"data": [{"b64_json": "eA=="}], "endpoint": endpoint}}

        self.mod.runpod = fake
        self.sleep = time.sleep
        self.mod.time.sleep = lambda s: self.sleep(0.05)
        self.server = self.mod.http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.mod.Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        time.sleep = self.sleep
        self.tmp.cleanup()

    def post(self, tier):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/images/generations", data=b'{"prompt":"x","model":"ra2","cache":true}',
                                     headers={"Content-Type": "application/json", "X-Omniserve-Tier": tier})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def test_parallel_routing_breaker_and_ledger(self):
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            outs = list(pool.map(self.post, ["paid", "paid", "sub"]))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.8, "overflow jobs serialized")
        self.assertEqual([o["endpoint"] for o in outs], ["cheap", "cheap", "fast"])
        self.assertLessEqual(sum(1 for c in self.calls if c[1] == "/run"), 3)
        self.assertTrue(all(set(b["input"]) == {"prompt"} for m, p, b in [c for c in self.submitted]))
        led = self.mod.frontier.get_ledger()
        led.flush()
        rows = led.conn().execute("SELECT backend, endpoint, tier, queue_ms, exec_ms, status, cold FROM jobs").fetchall()
        self.assertEqual(sorted(r[1] for r in rows), ["cheap", "cheap", "fast"])
        self.assertTrue(all(r[0] == "runpod" and r[3] == 12 and r[4] == 1000 and r[5] == "COMPLETED" and r[6] == 0 for r in rows))
        self.assertGreater(self.mod._primary_down_until, time.monotonic())
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics") as r:
            self.assertIn(b'frontier_jobs{workload="ra2",backend="runpod:cheap"} 2', r.read())


if __name__ == "__main__":
    unittest.main()
