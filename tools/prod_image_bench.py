#!/usr/bin/env python3
"""Latency/throughput probe for /v1/images/generations: distinct seeds (cold), then exact repeats (cache)."""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def call(url, payload, timeout):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return (time.perf_counter() - t) * 1000, len(r.headers.get("content-length", "") and [0] or []), body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8791/v1/images/generations")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    def payload(seed, cache):
        p = {"prompt": "a lighthouse on a cliff at sunset, detailed oil painting", "size": a.size,
             "steps": a.steps, "seed": seed}
        if cache:
            p["cache"] = True
        return p

    def phase(name, seeds, cache):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.conc) as ex:
            res = list(ex.map(lambda s: call(a.url, payload(s, cache), a.timeout), seeds))
        wall = time.perf_counter() - t0
        lat = sorted(r[0] for r in res)
        infer = [r[2]["data"][0].get("inference_time_ms", 0) for r in res]
        hits = sum(1 for r in res if r[2]["data"][0].get("cache", {}).get("hit"))
        print(f"{name:10s} n={len(res)} conc={a.conc} wall={wall:.2f}s img/s={len(res)/wall:.3f} "
              f"p50={statistics.median(lat):.0f}ms max={lat[-1]:.0f}ms infer_p50={statistics.median(infer):.0f}ms hits={hits}")

    seeds = [a.seed0 + i for i in range(a.n)]
    phase("cold", seeds, True)
    phase("repeat", seeds, True)


main()
