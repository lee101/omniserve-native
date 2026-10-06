#!/usr/bin/env python3
"""OMNISERVE_NATIVE_KEY_TIERS: a mapped caller key authenticates, sets the
caller's default tier and caps what X-Omniserve-Tier may claim; relayed callers
without one stay free. Observed through the tier the gateway forwards to an
image model upstream stub. No GPU needed."""
import http.server, json, os, socket, subprocess, sys, threading, time, urllib.error, urllib.request

SECRET = "shared-secret"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        body = json.dumps({"tier": self.headers.get("X-Omniserve-Tier")}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def post(port, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/generations",
                                 data=b'{"model":"ra2","prompt":"x"}',
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r).get("tier")
    except urllib.error.HTTPError as e:
        return e.code


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0
    stub_port, gw_port = free_port(), free_port()
    stub = http.server.ThreadingHTTPServer(("127.0.0.1", stub_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update({
        "OMNISERVE_NATIVE_BIN": binary,
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_SECRET": SECRET,
        "OMNISERVE_NATIVE_KEY_TIERS": "subkey=sub,paidkey=paid,bogus=gold",
        "OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS": f"ra2=http://127.0.0.1:{stub_port}",
    })
    proc = subprocess.Popen([binary, "--port", str(gw_port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{gw_port}/v1/models", timeout=2).read()
                break
            except Exception:
                if proc.poll() is not None:
                    print("FAIL: gateway exited during startup")
                    return 1
                time.sleep(0.2)
        relayed = {"X-Forwarded-For": "1.2.3.4"}
        bearer = lambda k: {"Authorization": "Bearer " + k}
        cases = [
            ("internal secret, no header", bearer(SECRET), "free"),
            ("internal secret, paid header", {**bearer(SECRET), "X-Omniserve-Tier": "paid"}, "paid"),
            ("relayed secret cannot claim paid", {**relayed, **bearer(SECRET), "X-Omniserve-Tier": "paid"}, "free"),
            ("relayed sub key defaults to sub", {**relayed, **bearer("subkey")}, "sub"),
            ("relayed sub key via X-API-Key", {**relayed, "X-API-Key": "subkey"}, "sub"),
            ("sub key capped at sub", {**relayed, **bearer("subkey"), "X-Omniserve-Tier": "paid"}, "sub"),
            ("sub key may lower to free", {**relayed, **bearer("subkey"), "X-Omniserve-Tier": "free"}, "free"),
            ("paid key", {**relayed, **bearer("paidkey")}, "paid"),
            ("internal header wins over key", {**bearer("subkey"), "X-Omniserve-Tier": "background"}, "background"),
            ("unknown tier entry is not a key", {**relayed, **bearer("bogus")}, 401),
            ("wrong key", {**relayed, **bearer("nope")}, 401),
        ]
        failures = 0
        for name, headers, want in cases:
            got = post(gw_port, headers)
            if got != want:
                print(f"FAIL: {name}: got {got!r}, want {want!r}")
                failures += 1
        if failures:
            return 1
        print("key-tier tests passed")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
