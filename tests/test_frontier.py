import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier import Ledger, Router, build_routing, decide, estimate_usd, pareto  # noqa: E402
from frontier.policy import load_seeds, write_routing  # noqa: E402


def seeds():
    return {"tiers": {"free": {"policy": "local_only"}, "paid": {"policy": "cheapest_within_deadline"},
                      "sub": {"policy": "fastest"}},
            "workloads": {"w": {"deadline_ms": 40000, "candidates": [
                {"id": "local", "kind": "local", "p50_ms": 10000, "p95_ms": 15000, "usd_per_job": 0.0003, "quality": 1.0},
                {"id": "runpod:a", "kind": "runpod", "endpoint": "a", "p50_ms": 30000, "usd_per_job": 0.01, "quality": 1.0},
                {"id": "runpod:b", "kind": "runpod", "endpoint": "b", "p50_ms": 20000, "usd_per_job": 0.03, "quality": 1.0},
                {"id": "runpod:lossy", "kind": "runpod", "endpoint": "c", "p50_ms": 5000, "usd_per_job": 0.001, "quality": 0.9},
                {"id": "runpod:dominated", "kind": "runpod", "endpoint": "d", "p50_ms": 35000, "usd_per_job": 0.05, "quality": 1.0}]}}}


class DecideTests(unittest.TestCase):
    def test_policies(self):
        self.assertEqual(decide("local_only", 1e9, 10000, 1000), "local")
        self.assertEqual(decide("overflow_on_busy", 0, 10000, 1e9), "remote")
        self.assertEqual(decide("fastest", 10000, 10000, 30000), "local")
        self.assertEqual(decide("fastest", 25000, 10000, 30000), "remote")
        self.assertEqual(decide("cheapest_within_deadline", 20000, 10000, 30000, 40000), "local")
        self.assertEqual(decide("cheapest_within_deadline", 35000, 10000, 30000, 40000), "remote")
        self.assertEqual(decide("cheapest_within_deadline", 50000, 10000, 70000, 40000), "local")
        self.assertEqual(decide("cheapest_within_deadline", 90000, 10000, 70000, 40000), "remote")


class ParetoTests(unittest.TestCase):
    def test_frontier_excludes_dominated_lossy_and_unavailable(self):
        routing = build_routing(seeds())
        w = routing["workloads"]["w"]
        self.assertEqual(w["frontier"], ["local"])
        cands = {c["id"]: c for c in w["candidates"]}
        self.assertFalse(cands["runpod:lossy"]["quality_ok"])
        front = pareto([{**c, "available": c["id"] != "local"} for c in w["candidates"]])
        self.assertEqual([c["id"] for c in front], ["runpod:b", "runpod:a"])

    def test_tier_orders_and_gateway_block(self):
        w = build_routing(seeds())["workloads"]["w"]
        self.assertEqual(w["tiers"]["paid"]["remote_order"][:2], ["runpod:a", "runpod:b"])
        self.assertEqual(w["tiers"]["paid"]["deadline_ms"], 40000)
        self.assertEqual(w["tiers"]["sub"]["remote_order"][0], "runpod:b")
        self.assertNotIn("runpod:lossy", w["tiers"]["sub"]["remote_order"])
        self.assertEqual(w["gateway"], {"local_p50_ms": 10000, "remote_p50_ms": 20000, "tiers": {
            "paid": {"policy": "cheapest_within_deadline", "deadline_ms": 40000}, "sub": {"policy": "fastest"},
            "free": {"policy": "local_only"}, "background": {"policy": "overflow_on_busy"}}})

    def test_repo_seeds_build(self):
        routing = build_routing(load_seeds())
        self.assertIn("local", routing["workloads"]["ra2"]["frontier"])
        self.assertEqual(routing["workloads"]["pixal3d"]["frontier"], ["runpod:akgefm0nzzr4jo"])
        self.assertNotIn("gateway", routing["workloads"]["pixal3d"])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.tmp.name) / "l.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_costs(self):
        self.assertAlmostEqual(estimate_usd("runpod", exec_ms=3.6e6, endpoint="tmozxvnm9fuuud")[0], 1.10)
        usd, cold = estimate_usd("runpod", exec_ms=1.8e6, queue_ms=1.8e6, endpoint="tmozxvnm9fuuud")
        self.assertTrue(cold)
        self.assertAlmostEqual(usd, 1.10)
        self.assertAlmostEqual(estimate_usd("local", exec_ms=3.6e6)[0], 0.10)
        self.assertAlmostEqual(estimate_usd("local", exec_ms=3.6e6, saturated=True)[0], 0.80)
        self.assertEqual(estimate_usd("local", exec_ms=3.6e6, cache_hit=True)[0], 0.0)

    def test_async_record_measurement_and_reconcile(self):
        now = time.time()
        for i in range(6):
            self.ledger.record(ts=now - i, workload="w", backend="runpod", endpoint="a", queue_ms=100, exec_ms=20000 + i,
                               wall_ms=22000 + i * 1000, status="COMPLETED")
            self.ledger.record(ts=now - i, workload="w", backend="local", exec_ms=8000, status="200")
        self.ledger.record(ts=now, workload="w", backend="local", cache_hit=1, exec_ms=5, status="200")
        self.ledger.flush()
        day = time.strftime("%Y-%m-%d 00:00:00", time.gmtime(now))
        est = self.ledger.conn().execute("SELECT SUM(est_usd) FROM jobs WHERE backend='runpod'").fetchone()[0]
        rows = self.ledger.reconcile("k", endpoints=["a"], fetch=lambda url: [
            {"endpointId": "a", "time": day, "amount": est * 2, "timeBilledMs": 1}, {"endpointId": "zz", "time": day, "amount": 9}])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["factor"], 2.0, places=3)
        self.assertAlmostEqual(self.ledger.billing_factor("a"), 2.0, places=3)
        w = build_routing(seeds(), self.ledger)["workloads"]["w"]
        cands = {c["id"]: c for c in w["candidates"]}
        self.assertEqual(cands["local"]["p50_ms"], 8000)
        self.assertEqual(cands["local"]["source"], "ledger")
        self.assertEqual(cands["runpod:a"]["p50_ms"], 24500)
        self.assertAlmostEqual(cands["runpod:a"]["usd_per_job"], est / 6 * 2, places=6)
        summary = self.ledger.summary()
        self.assertEqual(summary["workloads"]["w"]["local:cache_hit"]["jobs"], 1)
        self.assertEqual(summary["workloads"]["w"]["local"]["jobs"], 6)
        self.assertIn('frontier_jobs{workload="w",backend="runpod:a"} 6', self.ledger.prometheus())

    def test_ingest_gateway_jsonl_is_incremental(self):
        path = Path(self.tmp.name) / "g.jsonl"
        lines = [{"ts": 1, "workload": "ra2", "tier": "paid", "backend": "local", "reason": "queued", "queue_ms": 900,
                  "exec_ms": 11000, "local_wait_ms": 12000, "status": 200},
                 {"ts": 2, "workload": "ra2", "tier": "paid", "backend": "overflow", "reason": "saturated",
                  "queue_ms": 0, "exec_ms": 35000, "local_wait_ms": 48000, "status": 0},
                 {"ts": 3, "workload": "ra2", "tier": "free", "backend": "cache", "reason": "exact", "queue_ms": 0,
                  "exec_ms": 40, "local_wait_ms": 0, "status": 200}]
        lines.append({"ts": 4, "workload": "ra2", "backend": "local", "exec_ms": 1, "status": 200, "port": 8819})
        path.write_text("".join(json.dumps(x) + "\n" for x in lines) + '{"partial":')
        self.assertEqual(self.ledger.ingest_jsonl(path), 3)
        self.assertEqual(self.ledger.ingest_jsonl(path), 0)
        with path.open("a") as handle:
            handle.write('1}\n')
        self.assertEqual(self.ledger.ingest_jsonl(path), 1)
        rows = self.ledger.conn().execute("SELECT backend, cache_hit, est_usd FROM jobs ORDER BY ts").fetchall()
        self.assertEqual([r[0] for r in rows[:3]], ["local", "gateway-overflow", "local"])
        self.assertGreater(rows[0][2], 11000 / 3.6e6 * 0.10)
        self.assertEqual(rows[2][1:], (1, 0.0))


class RouterTests(unittest.TestCase):
    def test_hot_reload_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routing.json"
            router = Router(str(path), min_interval=0)
            self.assertIsNone(router.decide("w", "paid", 0))
            write_routing(build_routing(seeds()), path)
            self.assertEqual(router.decide("w", "paid", 10000), "local")
            self.assertEqual(router.decide("w", "paid", 35000), "remote")
            self.assertEqual(router.decide("w", "free", 1e9), "local")
            self.assertEqual(router.decide("w", "sub", 5000), "local")
            self.assertEqual(router.decide("w", "sub", 15000), "remote")
            self.assertEqual(router.remote_order("w", "sub")[0], "runpod:b")
            s = seeds()
            s["tiers"]["paid"] = {"policy": "overflow_on_busy"}
            write_routing(build_routing(s), path)
            os.utime(path, ns=(time.time_ns(), time.time_ns() + 10**9))
            self.assertEqual(router.decide("w", "paid", 0), "remote")

    def test_missing_local_is_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routing.json"
            write_routing(build_routing(load_seeds()), path)
            router = Router(str(path), min_interval=0)
            self.assertEqual(router.decide("pixal3d", "paid", 0), "remote")
            self.assertEqual(router.remote_order("pixal3d", "free"), ["runpod:akgefm0nzzr4jo"])


if __name__ == "__main__":
    unittest.main()
