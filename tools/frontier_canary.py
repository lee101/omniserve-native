#!/usr/bin/env python3
"""Bounded frontier canary: N concurrent uncached ra2 requests against a gateway,
each with its own tier/deadline, reporting where it ran and how long it took.
Unique seeds keep every render uncached; --repeat adds exact cache hits reported
separately. Usage: OMNISERVE_SECRET=... frontier_canary.py --plan paid,paid,paid:20000"""
import argparse, concurrent.futures, json, os, random, time, urllib.error, urllib.request


def one(base, secret, tier, deadline, seed, size, prompt):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + secret, "X-API-Key": secret,
               "X-Omniserve-Tier": tier}
    if deadline:
        headers["X-Omniserve-Deadline-Ms"] = str(deadline)
    body = json.dumps({"model": "ra2", "prompt": prompt, "size": size, "seed": seed}).encode()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/images/generations", data=body, headers=headers),
                                    timeout=620) as r:
            out = json.load(r)
            code = r.status
    except urllib.error.HTTPError as exc:
        out, code = {"error": exc.read()[:200].decode("utf-8", "replace")}, exc.code
    wall = time.monotonic() - started
    backend = "remote" if "outputs" in out or "omniserve" in out else ("local" if code == 200 else "error")
    return {"tier": tier, "deadline_ms": deadline, "seed": seed, "status": code, "backend": backend,
            "wall_s": round(wall, 2), "cached": bool(out.get("cached") or (out.get("timings") or {}).get("cache_hit"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8792")
    ap.add_argument("--plan", default="paid,paid,paid")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--stagger", type=float, default=0.3)
    ap.add_argument("--repeat", action="store_true")
    args = ap.parse_args()
    secret = os.environ["OMNISERVE_SECRET"]
    base_seed = random.randint(1, 1 << 30)
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for i, item in enumerate(args.plan.split(",")):
            tier, _, deadline = item.partition(":")
            jobs.append(pool.submit(one, args.base, secret, tier, int(deadline or 0), base_seed + i, args.size,
                                    f"frontier canary {base_seed + i}: a lighthouse on a cliff at dusk, oil painting"))
            time.sleep(args.stagger)
        rows = [j.result() for j in jobs]
    if args.repeat:
        rows.append({**one(args.base, secret, "paid", 0, base_seed, args.size,
                           f"frontier canary {base_seed}: a lighthouse on a cliff at dusk, oil painting"), "repeat": True})
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
