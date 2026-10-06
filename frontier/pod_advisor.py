"""Serverless-vs-pod break-even advisor over the frontier ledger. Advisory only: never calls RunPod.

A dedicated pod wins once serverless spend per hour on an endpoint stays above pod $/h * margin for
every hour of a window. Serverless spend is ledger est_usd scaled by the latest billing factor
(billed/estimated, covers cold starts). Output feeds a human or a separately-approved actuator."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

DEFAULT_DB = "/nvme0n1-disk/data/omniserve-frontier/ledger.db"


def hourly_spend(db: str, window_h: int, now: float) -> tuple[dict, dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    start = int(now // 3600) * 3600 - window_h * 3600
    spend: dict = {}
    for endpoint, bucket, usd, n in con.execute(
            "select endpoint, cast((ts - ?) / 3600 as int), sum(est_usd), count(*) from jobs "
            "where backend='runpod' and endpoint is not null and ts >= ? and ts < ? group by 1, 2",
            (start, start, start + window_h * 3600)):
        spend.setdefault(endpoint, [[0.0, 0] for _ in range(window_h)])[bucket] = [usd or 0.0, n]
    factors = {e: f for e, f in con.execute(
        "select endpoint, factor from billing b where factor is not null and day = "
        "(select max(day) from billing where endpoint = b.endpoint and factor is not null)")}
    con.close()
    return spend, factors


def advise(spend: dict, factors: dict, pod_usd_h: float, margin: float = 1.2,
           pod_cap_usd_day: float = 24.0) -> dict:
    out = {}
    for endpoint, buckets in sorted(spend.items()):
        factor = factors.get(endpoint) or 1.0
        hourly = [usd * factor for usd, _ in buckets]
        threshold = pod_usd_h * margin
        sustained = bool(hourly) and min(hourly) >= threshold
        capped = pod_usd_h * 24 > pod_cap_usd_day
        verdict = "provision_pod" if sustained and not capped else ("pod_over_cap" if sustained else "stay_serverless")
        out[endpoint] = {"verdict": verdict, "hours": len(hourly), "jobs": sum(n for _, n in buckets),
                         "billing_factor": round(factor, 3), "min_usd_h": round(min(hourly, default=0), 4),
                         "avg_usd_h": round(sum(hourly) / max(len(hourly), 1), 4), "break_even_usd_h": round(threshold, 4),
                         "est_savings_usd_day": round(max(sum(hourly) / max(len(hourly), 1) - pod_usd_h, 0) * 24, 2)
                         if sustained else 0.0}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m frontier.pod_advisor")
    p.add_argument("--db", default=os.getenv("FRONTIER_LEDGER_DB", DEFAULT_DB))
    p.add_argument("--window-h", type=int, default=int(os.getenv("POD_ADVISOR_WINDOW_H", "6")))
    p.add_argument("--pod-usd-h", type=float, default=float(os.getenv("POD_ADVISOR_POD_USD_H", "0.69")))
    p.add_argument("--margin", type=float, default=float(os.getenv("POD_ADVISOR_MARGIN", "1.2")))
    p.add_argument("--pod-cap-usd-day", type=float, default=float(os.getenv("POD_ADVISOR_CAP_USD_DAY", "24")))
    a = p.parse_args(argv)
    spend, factors = hourly_spend(a.db, a.window_h, time.time())
    print(json.dumps(advise(spend, factors, a.pod_usd_h, a.margin, a.pod_cap_usd_day), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
