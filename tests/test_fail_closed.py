#!/usr/bin/env python3
"""Fail-closed auth: a gateway with no credential configured refuses relayed
callers, telemetry endpoints need a service credential when relayed, placeholder
secrets count as unset, dot-segment paths are refused, and the stall sweep
closes a half-sent request. No GPU needed."""
import os, socket, subprocess, sys, time, urllib.error, urllib.request

RELAYED = {"CF-Connecting-IP": "1.2.3.4", "CF-Ray": "x"}
SECRET = "s3cr3t-for-tests-only-0123456789"
OPS = ["/metrics", "/errors", "/status", "/backend_status", "/v1/gpu/vram", "/v1/gpu/status", "/v1/host/memory"]
OPEN = ["/health", "/healthz", "/readyz", "/v1/models", "/docs", "/openapi.json"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(port, path, headers=None, method="GET", body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=method,
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        e.read()
        return e.code


def raw_status(port, request):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
        s.sendall(request)
        data = s.recv(256)
    return int(data.split(b" ", 2)[1]) if data.startswith(b"HTTP/1.1 ") else 0


def start(binary, extra):
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update({"OMNISERVE_NATIVE_BIND": "127.0.0.1", **extra})
    proc = subprocess.Popen([binary, "--port", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
            return proc, port
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("FAIL: gateway exited during startup")
            time.sleep(0.2)
    raise SystemExit("FAIL: gateway never ready")


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0
    failures = 0

    def check(name, got, want):
        nonlocal failures
        if got != want:
            print(f"FAIL: {name}: got {got!r}, want {want!r}")
            failures += 1

    chat = lambda port, h: call(port, "/v1/chat/completions", h, "POST", b'{"messages":[]}')

    proc, port = start(binary, {})
    try:
        check("no secret: relayed chat refused", chat(port, RELAYED), 401)
        check("no secret: loopback chat not refused", chat(port, {}) != 401, True)
        check("no secret: relayed /loras refused", call(port, "/loras", RELAYED), 401)
        for p in OPS:
            check(f"no secret: relayed {p}", call(port, p, RELAYED), 401)
            check(f"no secret: loopback {p}", call(port, p), 200)
        for p in OPEN:
            check(f"no secret: relayed open {p}", call(port, p, RELAYED), 200)
        for p in ["/v1/3d/assets/../status", "/v1/3d/assets/%2e%2e/x", "/v1/images/hair-layers/jobs/..%2fx",
                  "/v1/3d/assets/./x", "/v1/3d/assets/a\\b"]:
            check(f"dot path {p!r}", raw_status(port, f"GET {p} HTTP/1.1\r\nHost: x\r\n\r\n".encode()), 400)
        check("dotted file name allowed", call(port, "/v1/3d/assets/a.glb") != 400, True)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {"OMNISERVE_NATIVE_ALLOW_OPEN": "1"})
    try:
        check("allow open: relayed chat not refused", chat(port, RELAYED) != 401, True)
        check("allow open: relayed /status", call(port, "/status", RELAYED), 200)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {"OMNISERVE_NATIVE_SECRET": SECRET})
    try:
        for p in OPS:
            check(f"secret: relayed {p} anonymous", call(port, p, RELAYED), 401)
            check(f"secret: relayed {p} keyed", call(port, p, {**RELAYED, "X-API-Key": SECRET}), 200)
            check(f"secret: loopback {p}", call(port, p), 200)
        check("secret: wrong-length key", chat(port, {**RELAYED, "Authorization": "Bearer " + SECRET[:-1]}), 401)
        check("secret: longer key", chat(port, {**RELAYED, "Authorization": "Bearer " + SECRET + "x"}), 401)
        check("secret: right key", chat(port, {**RELAYED, "Authorization": "Bearer " + SECRET}) != 401, True)
        check("secret: query key", call(port, "/status?secret=" + SECRET, RELAYED), 200)
        check("secret: empty bearer", chat(port, {**RELAYED, "Authorization": "Bearer "}), 401)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {"OMNISERVE_NATIVE_KEY_TIERS": "", "OMNISERVE_NATIVE_AUTH_MODE": "shadow"})
    try:
        check("shadow gate: relayed training refused without service key",
              call(port, "/v1/training/jobs", RELAYED), 401)
        check("shadow gate: loopback training not refused", call(port, "/v1/training/jobs") != 401, True)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {"OMNISERVE_NATIVE_SECRET": "change-me"})
    try:
        check("placeholder secret: relayed refused even with it",
              chat(port, {**RELAYED, "secret": "change-me"}), 401)
        check("placeholder secret: loopback not refused", chat(port, {}) != 401, True)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {"OMNISERVE_NATIVE_REQUEST_STALL_S": "1"})
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{")
            t0 = time.time()
            data = s.recv(64)
            check("stalled request closed", data, b"")
            check("stalled request closed promptly", time.time() - t0 < 8, True)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    if failures:
        return 1
    print("fail-closed tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
