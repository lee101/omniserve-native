#!/usr/bin/env python3
"""The pod-mode entry serves the contract app.nz's Cog pod path speaks.

`tryCogHTTP` POSTs `{"input": {...}}` to `{endpoint}/predictions` and reads
`{"output": ...}` back, and readiness is polled on `/healthz`. This runs
`runtime/cog_pod.py` over a stub workload (no GPU, no weights) and checks that
contract, plus the error shape a failing prediction has to report rather than
hanging the pod.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB_WORKLOAD = '''
"""Minimal workload used only by tests/test_cog_pod_contract.py."""


def handler(job, _pipe=None):
    values = job.get("input") or {}
    if values.get("fail"):
        raise ValueError("stub failure")
    return {"created": 1, "model": "stub", "data": [{"b64_json": values.get("prompt", "")}]}
'''


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def post(url, payload, timeout=10):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> int:
    with tempfile.TemporaryDirectory() as work:
        workdir = Path(work)
        (workdir / "stub_workload.py").write_text(STUB_WORKLOAD)
        manifest = workdir / "workloads.json"
        manifest.write_text(json.dumps({"stub": {"module": "stub_workload", "required_mib": 1}}))
        port = free_port()
        env = {key: value for key, value in os.environ.items() if not key.startswith(("OMNISERVE_", "COG_"))}
        env.update({
            "PYTHONPATH": f"{ROOT}:{workdir}",
            "COG_PORT": str(port),
            "COG_HOST": "127.0.0.1",
            "OMNISERVE_WORKLOAD_MANIFEST": str(manifest),
            # The gateway relays the omniserve image body, which carries no
            # workload field, so the default is what selects the lane.
            "OMNISERVE_DEFAULT_WORKLOAD": "stub",
            "OMNISERVE_ENFORCE_VRAM": "0",
            "OMNISERVE_VRAM_BROKER_URL": "",
        })
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "runtime/cog_pod.py")],
                                       env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        print("FAIL: cog_pod exited:", log.read().decode()[-500:])
                        return 1
                    try:
                        status, health = get(f"http://127.0.0.1:{port}/healthz")
                        break
                    except (OSError, ValueError):
                        time.sleep(0.05)
                else:
                    print("FAIL: cog_pod never became ready")
                    return 1
                if status != 200 or health.get("ready") is not True:
                    print("FAIL: /healthz shape", status, health)
                    return 1

                status, schema = get(f"http://127.0.0.1:{port}/openapi.json")
                names = [field["name"] for field in schema.get("inputs", [])]
                if status != 200 or "prompt" not in names or "image_base64" not in names:
                    print("FAIL: /openapi.json does not describe the image contract", schema)
                    return 1

                # The exact request app.nz's tryCogHTTP makes, and the exact
                # field it reads back.
                status, body = post(f"http://127.0.0.1:{port}/predictions",
                                    {"input": {"prompt": "pod contract"}})
                if status != 200 or body.get("status") != "succeeded":
                    print("FAIL: /predictions rejected a valid call", status, body)
                    return 1
                output = body.get("output")
                if not isinstance(output, dict) or output["data"][0]["b64_json"] != "pod contract":
                    print("FAIL: output is not the worker's own result", output)
                    return 1

                # A bare body (the gateway's omniserve shape) works too.
                status, body = post(f"http://127.0.0.1:{port}/predictions", {"prompt": "bare"})
                if status != 200 or body["output"]["data"][0]["b64_json"] != "bare":
                    print("FAIL: bare omniserve body not accepted", status, body)
                    return 1

                # A failure is reported in-band: the pod stays up for the next
                # request instead of dying with the exception.
                status, body = post(f"http://127.0.0.1:{port}/predictions", {"input": {"fail": True}})
                if status != 200 or body.get("status") != "failed" or "stub failure" not in body.get("error", ""):
                    print("FAIL: failure shape", status, body)
                    return 1
                status, body = post(f"http://127.0.0.1:{port}/predictions", {"input": {"prompt": "after failure"}})
                if status != 200 or body.get("status") != "succeeded":
                    print("FAIL: pod did not survive a failed prediction", status, body)
                    return 1

                status, _ = post(f"http://127.0.0.1:{port}/nope", {"input": {}})
                if status != 404:
                    print("FAIL: unknown path answered", status)
                    return 1
                print("cog pod contract ok: healthz, openapi, predictions, bare body, error shape")
                return 0
            finally:
                process.terminate()
                process.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
