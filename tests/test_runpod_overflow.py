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
import urllib.error
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
        self.fail = {}
        jobs = {}

        def fake(method, path, body=None, endpoint=None):
            self.calls.append((method, path, endpoint))
            if path == "/run":
                if self.fail.get(endpoint) == "submit":
                    raise OSError("connect timeout")
                self.submitted.append((method, path, body))
                job = f"{endpoint}-{len(jobs)}"
                jobs[job] = time.monotonic()
                return {"id": job}
            job = path.rsplit("/", 1)[-1]
            if path.startswith("/cancel"):
                return {}
            if self.fail.get(endpoint) == "failed":
                return {"status": "FAILED", "error": "worker crashed"}
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

    def post_error(self, tier):
        try:
            self.post(tier)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Retry-After"), json.loads(exc.read())
        raise AssertionError("expected an HTTP error")

    def test_failover_to_next_endpoint_and_breaker(self):
        self.fail["cheap"] = "failed"
        for _ in range(3):
            self.assertEqual(self.post("paid")["endpoint"], "fast")
        self.assertTrue(self.mod.breaker("cheap").state()["open"])
        runs = [c[2] for c in self.calls if c[1] == "/run"]
        self.assertEqual(runs, ["cheap", "fast"] * 3)
        self.assertEqual(self.post("paid")["endpoint"], "fast")  # open breaker: cheap skipped
        self.assertEqual([c[2] for c in self.calls if c[1] == "/run"][-1], "fast")
        self.assertEqual(len([c for c in self.calls if c[1] == "/run"]), 7)

    def test_all_open_is_fast_503_with_retry_after(self):
        self.fail.update(cheap="submit", fast="submit")
        for _ in range(3):
            code, _, _ = self.post_error("paid")
            self.assertEqual(code, 502)
        started = time.monotonic()
        code, retry, body = self.post_error("sub")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(code, 503)
        self.assertGreaterEqual(int(retry), 1)
        self.assertIn("breakers open", body["error"]["message"])

    def test_free_and_background_never_reach_runpod(self):
        for tier in ("free", "background", ""):
            code, retry, _ = self.post_error(tier)
            self.assertEqual((code, retry), (503, "5"))
        self.assertFalse([c for c in self.calls if c[1] == "/run"])

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
        self.assertTrue(self.mod.PRIMARY_BREAKER.state()["open"])
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics") as r:
            self.assertIn(b'frontier_jobs{workload="ra2",backend="runpod:cheap"} 2', r.read())


class BreakerTests(unittest.TestCase):
    def test_open_half_open_backoff(self):
        mod = load({"RUNPOD_API_KEY": "k"})
        now = [0.0]
        b = mod.Breaker("x", failures=2, cooldown=10, max_cooldown=25, clock=lambda: now[0])
        b.record(False)
        self.assertTrue(b.allow())
        b.record(False)
        self.assertFalse(b.allow())
        now[0] = 10.5
        self.assertTrue(b.allow())   # probe
        self.assertFalse(b.allow())  # only one
        b.record(False)
        self.assertEqual(b.state()["cooldown_s"], 20)
        now[0] = 31
        self.assertTrue(b.allow())
        b.record(False)
        self.assertEqual(b.state()["cooldown_s"], 25)
        b.record(True)
        self.assertTrue(b.allow())
        self.assertEqual(b.state()["consecutive_failures"], 0)


class PrimaryHealthTests(unittest.TestCase):
    def setUp(self):
        self.mod = load({"RUNPOD_API_KEY": "k", "PRIMARY_UPSTREAM": "http://127.0.0.1:9",
                         "PRIMARY_BREAKER_S": "60"})
        self.replies = []
        self.calls = 0

        def fake_post(path, raw, headers):
            self.calls += 1
            return self.replies.pop(0)

        self.mod.post_primary = fake_post

    def test_client_errors_do_not_lock_the_primary_out(self):
        for code in (403, 404):
            self.replies.append((code, b"{}", None))
            self.assertIsNone(self.mod.try_primary("/v1/images/generations", b"{}", "paid"))
            state = self.mod.PRIMARY_BREAKER.state()
            self.assertFalse(state["open"])
            self.assertEqual(state["consecutive_failures"], 0)
        self.replies.append((200, b"{}", None))
        self.assertEqual(self.mod.try_primary("/v1/images/generations", b"{}", "paid"), (200, b"{}"))

    def test_429_throttles_for_retry_after_without_opening_the_breaker(self):
        self.replies.append((429, b"{}", "7"))
        self.assertIsNone(self.mod.try_primary("/p", b"{}", ""))
        self.assertFalse(self.mod.PRIMARY_BREAKER.state()["open"])
        self.assertEqual(self.mod.PRIMARY_BREAKER.state()["consecutive_failures"], 0)
        before = self.calls
        self.assertIsNone(self.mod.try_primary("/p", b"{}", ""))  # throttled: not even attempted
        self.assertEqual(self.calls, before)
        self.assertGreater(self.mod._primary_throttle_until - time.monotonic(), 5)
        self.mod._primary_throttle_until = 0.0
        self.replies.append((200, b"ok", None))
        self.assertEqual(self.mod.try_primary("/p", b"{}", ""), (200, b"ok"))

    def test_retry_after_parsing_is_bounded(self):
        p = self.mod.parse_retry_after
        self.assertEqual(p("3"), 3.0)
        self.assertEqual(p(None), self.mod.PRIMARY_THROTTLE_DEFAULT_S)
        self.assertEqual(p("Wed, 21 Oct 2026 07:28:00 GMT"), self.mod.PRIMARY_THROTTLE_DEFAULT_S)
        self.assertEqual(p("99999"), self.mod.PRIMARY_THROTTLE_MAX_S)
        self.assertEqual(p("-1"), self.mod.PRIMARY_THROTTLE_DEFAULT_S)

    def test_server_errors_still_open_the_breaker(self):
        self.replies.append((530, b"", None))
        self.assertIsNone(self.mod.try_primary("/p", b"{}", ""))
        self.assertTrue(self.mod.PRIMARY_BREAKER.state()["open"])


class RunJobBudgetTests(unittest.TestCase):
    class Stub:
        path = "/v1/images/generations"

    def run_with_status(self, status):
        mod = load({"RUNPOD_API_KEY": "k", "RUNPOD_RA2_ENDPOINTS": "ep1"})
        calls = []

        def fake(method, path, body=None, endpoint=None):
            calls.append((method, path))
            if path == "/run":
                return {"id": "job1"}
            return {"status": status} if path.startswith("/status") else {}

        mod.runpod = fake
        orig = mod.time.sleep
        mod.time.sleep = lambda s: orig(0.01)
        try:
            result = mod.Handler.run_job(self.Stub(), "ep1", {"prompt": "x"}, mod.WORKLOAD, "paid",
                                         time.monotonic(), 0.2, [])
        finally:
            mod.time.sleep = orig
        return mod, result, calls

    def test_budget_exhaustion_while_running_is_a_timeout_not_a_breaker_failure(self):
        for status in ("IN_PROGRESS", "IN_QUEUE"):
            mod, (code, obj, retry), calls = self.run_with_status(status)
            self.assertEqual((code, retry), (504, False), status)
            self.assertEqual(mod.breaker("ep1").state()["consecutive_failures"], 0, status)
            self.assertIn(("POST", "/cancel/job1"), calls)

    def test_failed_job_still_counts_against_the_breaker(self):
        mod, (code, obj, retry), calls = self.run_with_status("FAILED")
        self.assertEqual((code, retry), (502, True))
        self.assertEqual(mod.breaker("ep1").state()["consecutive_failures"], 1)


if __name__ == "__main__":
    unittest.main()
