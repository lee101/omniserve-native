#!/usr/bin/env python3
"""Frontier policy on the embedded image lane: a busy lane queues locally when the
expected local finish beats the remote, overflows when it does not, honours the
caller deadline, hot-reloads the policy file and writes one ledger line per job."""

import concurrent.futures
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_overflow_routing import (CALLER_KEY, OverflowStub, free_port, gateway_env,  # noqa: E402
                                         post, status, wait_busy, wait_ready)


def policy(paid="cheapest_within_deadline", deadline=5000):
    return {"version": 1, "workloads": {"ra2": {"gateway": {
        "local_p50_ms": 2000, "remote_p50_ms": 3000,
        "tiers": {"paid": {"policy": paid, "deadline_ms": deadline},
                  "free": {"policy": "local_only"}, "background": {"policy": "background"}}}}}}


def write_policy(path, body):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)
    time.sleep(1.2)


def busy_then(port, body, headers, extra_headers):
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        occupying = pool.submit(post, port, "/v1/images/generations", {**body, "seed": body["seed"] + 1000}, headers)
        wait_busy(port)
        started = time.monotonic()
        code, out = post(port, "/v1/images/generations", body, {**headers, **extra_headers})
        elapsed = time.monotonic() - started
        occupying.result()
    return code, out, elapsed


def main() -> int:
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    stub_lib = os.environ.get("OMNISERVE_SD_STUB")
    if not binary or not os.path.exists(binary) or not stub_lib or not os.path.exists(stub_lib):
        print("skip: OMNISERVE_NATIVE_BIN / OMNISERVE_SD_STUB not set")
        return 0
    upstream_port, port = free_port(), free_port()
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", upstream_port), OverflowStub)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    body = {"prompt": "frontier", "width": 64, "height": 64, "steps": 2, "seed": 5}
    paid = {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "paid"}
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile() as log:
        policy_path, ledger = Path(tmp) / "routing.json", Path(tmp) / "gateway.jsonl"
        write_policy(policy_path, policy())
        env = gateway_env(port, stub_lib, upstream_port, model="cache-test.gguf")
        env.update({"OMNISERVE_NATIVE_OVERFLOW_TIERS": "paid,free",
                    "OMNISERVE_NATIVE_FRONTIER_POLICY": str(policy_path),
                    "OMNISERVE_NATIVE_FRONTIER_WORKLOAD": "ra2",
                    "OMNISERVE_NATIVE_FRONTIER_LOG": str(ledger)})
        gateway = subprocess.Popen([binary, "--port", str(port)], env=env, stdout=log, stderr=log)
        try:
            snap = wait_ready(port, gateway)
            if snap["overflow"].get("frontier") is not True:
                print("FAIL: frontier not reported", snap["overflow"])
                return 1
            code, out, _ = busy_then(port, {**body, "seed": 10}, paid, {})
            if code != 200 or out.get("overflow"):
                print("FAIL: paid within deadline should queue locally", code, out)
                return 1
            code, out, _ = busy_then(port, {**body, "seed": 11}, paid, {"X-Omniserve-Deadline-Ms": "3500"})
            if code != 200 or not out.get("overflow"):
                print("FAIL: tight deadline should overflow", code, out)
                return 1
            free = {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "free"}
            seen = len(OverflowStub.seen)
            code, out, _ = busy_then(port, {**body, "seed": 12}, free, {"X-Omniserve-Deadline-Ms": "1"})
            if code != 200 or out.get("overflow") or len(OverflowStub.seen) != seen:
                print("FAIL: local_only tier reached the remote", code, out)
                return 1
            write_policy(policy_path, policy(paid="overflow_on_busy"))
            code, out, _ = busy_then(port, {**body, "seed": 13}, paid, {})
            if code != 200 or not out.get("overflow"):
                print("FAIL: hot reload to overflow_on_busy not applied", code, out)
                return 1
            code, out = post(port, "/v1/images/generations", {**body, "seed": 10, "cache": True}, paid)
            snap = status(port)
            if snap["overflow"]["frontier_kept_local"] != 2 or snap["overflow"]["saturated"] != 2:
                print("FAIL: counters", snap["overflow"])
                return 1
            if [seen["tier"] for seen in OverflowStub.seen] != ["paid", "paid"]:
                print("FAIL: overflow did not carry the caller tier", OverflowStub.seen)
                return 1
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            backends = [r["backend"] for r in rows]
            if backends.count("overflow") != 2 or backends.count("local") < 6:
                print("FAIL: ledger rows", rows)
                return 1
            queued = [r for r in rows if r["backend"] == "local" and r["reason"] == "queued"]
            if len(queued) != 2 or min(r["queue_ms"] for r in queued) < 500 or any(r["workload"] != "ra2" for r in rows):
                print("FAIL: queued rows", queued)
                return 1
            background = {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "background"}
            write_policy(policy_path, policy())
            seen = len(OverflowStub.seen)
            done = {}

            def timed(name, headers, seed):
                code, out = post(port, "/v1/images/generations", {**body, "seed": seed}, headers)
                done[name] = (time.monotonic(), code, bool(out.get("overflow")))

            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                pool.submit(timed, "holder", paid, 20)
                wait_busy(port)
                pool.submit(timed, "background", background, 21)
                deadline = time.monotonic() + 10
                while status(port)["admission"]["waiting"]["background"] < 1:
                    if time.monotonic() > deadline:
                        print("FAIL: background never queued")
                        return 1
                    time.sleep(0.02)
                pool.submit(timed, "paid", paid, 22)
            if any(v[1] != 200 or v[2] for v in done.values()) or len(OverflowStub.seen) != seen:
                print("FAIL: mixed burst left the local lane", done)
                return 1
            if not done["holder"][0] < done["paid"][0] < done["background"][0]:
                print("FAIL: background was admitted before the later paid request", done)
                return 1
        finally:
            gateway.terminate()
            gateway.wait(timeout=10)
            upstream.shutdown()
    print("image frontier routing ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
