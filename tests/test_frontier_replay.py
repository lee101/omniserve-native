import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier.replay import gate, replay  # noqa: E402

GW = {"local_p50_ms": 7000, "remote_p50_ms": 40000,
      "tiers": {"paid": {"policy": "cheapest_within_deadline", "deadline_ms": 45000},
                "free": {"policy": "local_only"}, "sub": {"policy": "fastest"}}}


def rows(tier, waits):
    return [{"tier": tier, "local_wait_ms": w, "status": 200} for w in waits]


class ReplayTests(unittest.TestCase):
    def test_idle_lane_never_spills(self):
        res = replay(rows("paid", [0] * 10) + rows("free", [0] * 10), GW, 0.0025)
        self.assertEqual(res["paid"]["remote"], 0)
        self.assertEqual(gate(res), [])

    def test_paid_spills_only_past_deadline(self):
        res = replay(rows("paid", [10000, 50000]), GW, 0.0025)
        self.assertEqual(res["paid"]["remote"], 1)
        self.assertAlmostEqual(res["paid"]["usd"], 0.0025 + 0.0003)

    def test_free_always_local_and_misses_nothing(self):
        res = replay(rows("free", [1e6]), GW, 0.0025)
        self.assertEqual(res["free"]["remote"], 0)

    def test_legacy_costs_more_than_frontier(self):
        rs = rows("paid", [5000] * 20)
        cur = replay(rs, GW, 0.0025)
        old = replay(rs, GW, 0.0025, policies={"paid": {"policy": "overflow_on_busy"}})
        self.assertGreater(old["paid"]["usd"], cur["paid"]["usd"])
        self.assertTrue(gate(old))

    def test_gate_flags_spilled_free(self):
        res = replay(rows("free", [0] * 4), GW, 0.0025, policies={"free": {"policy": "overflow_on_busy"}})
        self.assertTrue(any("free" in b for b in gate(res)))


if __name__ == "__main__":
    unittest.main()
