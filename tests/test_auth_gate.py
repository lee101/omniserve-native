#!/usr/bin/env python3
"""OMNISERVE_NATIVE_AUTH_MODE: relayed callers need a service key or a
credential the subscriber verifier vouches for; unauthenticated gets 401 and
unsubscribed 402 with the subscription_required contract. No GPU needed."""
import http.server, json, os, socket, subprocess, sys, threading, time, urllib.error, urllib.request

VERIFY_SECRET = "verify-secret"
SUBSCRIBERS = {"subuser": 200, "freeuser": 402}
calls = {"n": 0, "fail": False}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub(http.server.BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        calls["n"] += 1
        if calls["fail"]:
            return self._send(503, {})
        if self.headers.get("X-Verify-Secret") != VERIFY_SECRET:
            return self._send(403, {})
        code = SUBSCRIBERS.get(self.headers.get("X-Subscriber-Credential"), 401)
        self._send(code, {"ok": code == 200, "tier": "sub"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        self._send(200, {"tier": self.headers.get("X-Omniserve-Tier")})

    def log_message(self, *args):
        pass


def post(port, headers, path="/v1/images/generations"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b'{"model":"ra2","prompt":"x"}',
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.load(r), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


def start(binary, env_extra):
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update({"OMNISERVE_NATIVE_BIND": "127.0.0.1", **env_extra})
    proc = subprocess.Popen([binary, "--port", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
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
    stub_port = free_port()
    stub = http.server.ThreadingHTTPServer(("127.0.0.1", stub_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    base = {
        "OMNISERVE_NATIVE_KEY_TIERS": "svckey=paid",
        "OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS": f"ra2=http://127.0.0.1:{stub_port}",
        "OMNISERVE_NATIVE_AUTH_VERIFY_URL": f"http://127.0.0.1:{stub_port}/internal/omniserve/verify",
        "OMNISERVE_NATIVE_AUTH_VERIFY_SECRET": VERIFY_SECRET,
        "OMNISERVE_NATIVE_AUTH_OK_TTL_S": "1",
    }
    relayed = {"X-Forwarded-For": "1.2.3.4", "CF-Ray": "x"}
    bearer = lambda k: {"Authorization": "Bearer " + k}
    failures = 0

    def check(name, got, want):
        nonlocal failures
        if got != want:
            print(f"FAIL: {name}: got {got!r}, want {want!r}")
            failures += 1

    proc, port = start(binary, {**base, "OMNISERVE_NATIVE_AUTH_MODE": "enforce"})
    try:
        code, body, hdrs = post(port, relayed)
        check("relayed anonymous", code, 401)
        check("contract code", body.get("error", {}).get("code"), "subscription_required")
        check("contract url", body.get("error", {}).get("subscribe_url"), "https://text-generator.io/subscribe")
        check("contract header", hdrs.get("X-Subscribe-URL"), "https://text-generator.io/subscribe")
        check("X-Omniserve-Internal is no bypass", post(port, {**relayed, "X-Omniserve-Internal": "local"})[0], 401)
        check("service key", post(port, {**relayed, **bearer("svckey")})[1].get("tier"), "paid")
        check("subscriber", post(port, {**relayed, "secret": "subuser"})[1].get("tier"), "sub")
        n = calls["n"]
        post(port, {**relayed, "secret": "subuser"})
        check("verdict cached", calls["n"], n)
        code, body, hdrs = post(port, {**relayed, "X-API-Key": "freeuser"})
        check("unsubscribed", code, 402)
        check("unsubscribed header", hdrs.get("X-Subscribe-URL"), "https://text-generator.io/subscribe")
        check("unknown key", post(port, {**relayed, **bearer("nope")})[0], 401)
        check("internal unkeyed (scope relayed)", post(port, {})[0], 200)
        check("exempt readyz", post(port, relayed, "/readyz")[0] != 401, True)
        time.sleep(1.2)
        calls["fail"] = True
        check("stale subscriber on verifier outage", post(port, {**relayed, "secret": "subuser"})[0], 200)
        check("unknown on verifier outage", post(port, {**relayed, "secret": "other"})[0], 503)
        calls["fail"] = False
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {**base, "OMNISERVE_NATIVE_AUTH_MODE": "enforce", "OMNISERVE_NATIVE_AUTH_SCOPE": "all"})
    try:
        check("scope all: internal unkeyed", post(port, {})[0], 401)
        check("scope all: internal keyed", post(port, bearer("svckey"))[0], 200)
    finally:
        proc.terminate(); proc.wait(timeout=10)

    proc, port = start(binary, {**base, "OMNISERVE_NATIVE_AUTH_MODE": "shadow"})
    try:
        check("shadow serves anonymous", post(port, relayed)[0], 200)
    finally:
        proc.terminate(); proc.wait(timeout=10)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".mode", delete=False) as f:
        f.write("off\n")
    proc, port = start(binary, {**base, "OMNISERVE_NATIVE_AUTH_MODE_FILE": f.name})
    try:
        check("mode file off", post(port, relayed)[0], 200)
        with open(f.name, "w") as fh:
            fh.write("enforce\n")
        time.sleep(3.5)
        check("mode file flipped to enforce", post(port, relayed)[0], 401)
    finally:
        proc.terminate(); proc.wait(timeout=10)
        os.unlink(f.name)
    stub.shutdown()
    if failures:
        return 1
    print("auth-gate tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
