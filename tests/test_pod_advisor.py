import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier.pod_advisor import advise, hourly_spend  # noqa: E402

NOW = 1_800_000_000.0 + 1800


def make_db(rows, billing=()):
    path = tempfile.mktemp(suffix=".db")
    con = sqlite3.connect(path)
    con.execute("create table jobs(ts real, backend text, endpoint text, est_usd real)")
    con.execute("create table billing(day text, endpoint text, factor real)")
    con.executemany("insert into jobs values(?,?,?,?)", rows)
    con.executemany("insert into billing values(?,?,?)", billing)
    con.commit()
    con.close()
    return path


class PodAdvisorTests(unittest.TestCase):
    def hours(self, per_hour, endpoint="e", window=3):
        base = int(NOW // 3600) * 3600 - window * 3600
        return [(base + h * 3600 + 10, "runpod", endpoint, per_hour) for h in range(window)]

    def test_sustained_spend_recommends_pod(self):
        db = make_db(self.hours(0.5), [("2026-09-22", "e", 2.0)])
        spend, factors = hourly_spend(db, 3, NOW)
        r = advise(spend, factors, pod_usd_h=0.69)["e"]
        self.assertEqual(r["verdict"], "provision_pod")
        self.assertAlmostEqual(r["min_usd_h"], 1.0)
        self.assertAlmostEqual(r["est_savings_usd_day"], 7.44)

    def test_one_quiet_hour_stays_serverless(self):
        rows = self.hours(2.0)[:-1]
        spend, factors = hourly_spend(make_db(rows), 3, NOW)
        self.assertEqual(advise(spend, factors, 0.69)["e"]["verdict"], "stay_serverless")

    def test_cap_blocks_pod_and_local_rows_ignored(self):
        rows = self.hours(5.0) + [(NOW - 100, "local", None, 9.0), (NOW - 100, "runpod", "e", 99.0)]
        spend, factors = hourly_spend(make_db(rows), 3, NOW)
        r = advise(spend, factors, 2.0, pod_cap_usd_day=24)["e"]
        self.assertEqual(r["verdict"], "pod_over_cap")
        self.assertEqual(r["jobs"], 3)


if __name__ == "__main__":
    unittest.main()
