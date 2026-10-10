from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from .policy import decide

LOCAL_USD = 0.0003


def load_rows(path: str, since: float = 0.0, workload: str | None = None) -> list[dict]:
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ts", 0) < since or r.get("status") != 200 or r.get("backend") == "cache":
            continue
        if workload and r.get("workload") != workload:
            continue
        rows.append(r)
    return rows


def replay(rows: list[dict], gateway: dict, remote_usd: float, deadline_ms: float | None = None,
           policies: dict | None = None) -> dict:
    local_p50, remote_p50 = gateway["local_p50_ms"], gateway["remote_p50_ms"]
    tiers = {**gateway["tiers"], **(policies or {})}
    out: dict = {}
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r.get("tier", "free")].append(r)
    for tier, rs in by_tier.items():
        pol = tiers.get(tier, {"policy": "overflow_on_busy"})
        dl = pol.get("deadline_ms") or deadline_ms
        etas, remote, miss = [], 0, 0
        for r in rs:
            wait = r.get("local_wait_ms") or 0.0
            d = decide(pol["policy"], wait, local_p50, remote_p50, dl, pol.get("allow_overflow", False))
            eta = remote_p50 if d == "remote" else wait + local_p50
            remote += d == "remote"
            miss += bool(dl and eta > dl)
            etas.append(eta)
        etas.sort()
        n = len(rs)
        out[tier] = {"policy": pol["policy"], "n": n, "remote": remote, "remote_frac": remote / n,
                     "usd": remote * remote_usd + (n - remote) * LOCAL_USD,
                     "eta_p50_ms": etas[n // 2], "eta_p95_ms": etas[min(n - 1, int(n * 0.95))],
                     "deadline_miss_frac": miss / n}
    return out


TIER_MAX_REMOTE = {"sub": 0.15, "priority": 0.15}


def gate(result: dict, max_remote_frac: float = 0.05, max_miss_frac: float = 0.02) -> list[str]:
    bad = []
    for tier, s in result.items():
        max_remote_frac = TIER_MAX_REMOTE.get(tier, 0.05) if max_remote_frac == 0.05 else max_remote_frac
        if tier in ("free", "background") and s["remote"]:
            bad.append(f"{tier}: {s['remote']} jobs spilled to paid GPU")
        if s["remote_frac"] > max_remote_frac:
            bad.append(f"{tier}: remote_frac {s['remote_frac']:.3f} > {max_remote_frac}")
        if s["deadline_miss_frac"] > max_miss_frac:
            bad.append(f"{tier}: deadline_miss_frac {s['deadline_miss_frac']:.3f} > {max_miss_frac}")
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m frontier.replay")
    p.add_argument("--log", default="/nvme0n1-disk/data/omniserve-frontier/gateway-ra2.jsonl")
    p.add_argument("--routing", default="/nvme0n1-disk/data/omniserve-frontier/routing.json")
    p.add_argument("--workload", default="ra2")
    p.add_argument("--remote-usd", type=float, default=0.0025)
    p.add_argument("--since", type=float, default=0.0)
    p.add_argument("--compare", default="", help="JSON tier policy overrides to compare against")
    a = p.parse_args(argv)
    gw = json.load(open(a.routing))["workloads"][a.workload]["gateway"]
    rows = load_rows(a.log, a.since, a.workload)
    runs = {"current": replay(rows, gw, a.remote_usd)}
    runs["legacy_overflow_on_busy"] = replay(rows, gw, a.remote_usd, policies={t: {"policy": "overflow_on_busy"} for t in gw["tiers"]})
    if a.compare:
        runs["compare"] = replay(rows, gw, a.remote_usd, policies=json.loads(a.compare))
    for name, res in runs.items():
        print(f"== {name}")
        for tier, s in sorted(res.items()):
            print(f"{tier:<10} {s['policy']:<26} n={s['n']:<6} remote={s['remote']:<5} usd={s['usd']:.3f} "
                  f"eta p50={s['eta_p50_ms']/1000:.1f}s p95={s['eta_p95_ms']/1000:.1f}s miss={s['deadline_miss_frac']:.3f}")
    bad = gate(runs["current"])
    for b in bad:
        print("GATE FAIL", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
