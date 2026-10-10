#!/usr/bin/env python3

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.loads(response.read())


def post(port, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def start(binary, stub, extra):
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update({
        "OMNISERVE_ACCESS_LOG": "0",
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_SLOTS": "1",
        "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
        "OMNISERVE_NATIVE_SD_MIN_FREE_MB": "0",
        "OMNISERVE_NATIVE_SD_IMAGE_FORMAT": "png",
        "OMNISERVE_NATIVE_SD_LIB": stub,
        "OMNISERVE_NATIVE_SD_MODEL": "/dev/null",
        "OMNISERVE_NATIVE_VRAM_BROKER": "1",
        "OMNISERVE_NATIVE_SD_RESIDENCY_TICK_MS": "200",
    })
    env.update(extra)
    proc = subprocess.Popen([binary, "--port", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError("gateway exited during startup")
        try:
            if get(port, "/status")["diffusion"]["ready"]:
                return proc, port
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("gateway never became ready")


def residency(port):
    return get(port, "/v1/gpu/status")["residency"]


def main() -> int:
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    stub = os.environ.get("OMNISERVE_SD_STUB")
    if not binary or not stub:
        print("skip: OMNISERVE_NATIVE_BIN / OMNISERVE_SD_STUB not set")
        return 0

    proc, port = start(binary, stub, {})
    try:
        status, _ = post(port, "/v1/images/generations",
                         {"prompt": "a lighthouse at dusk", "size": "64x64", "steps": 1, "seed": 1})
        assert status == 200, status
        time.sleep(1.0)
        r = residency(port)
        assert r["client"]["enabled"] is False, r
        assert r["client"]["reports"] == 0, r
        assert r["broker"]["claims"] == [], r
        assert r["broker"]["promotions"] == 0 and r["broker"]["revocations"] == 0, r
    finally:
        proc.kill()
        proc.wait()

    proc, port = start(binary, stub, {"OMNISERVE_NATIVE_SD_RESIDENCY": "demand"})
    try:
        deadline = time.monotonic() + 5
        while residency(port)["client"]["reports"] == 0:
            assert time.monotonic() < deadline, "demand client never reported"
            time.sleep(0.1)
        assert residency(port)["client"]["enabled"] is True
    finally:
        proc.kill()
        proc.wait()

    print("residency inert without demand mode; demand mode reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
