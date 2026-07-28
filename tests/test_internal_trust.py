#!/usr/bin/env python3
"""X-Omniserve-Internal must not be honoured for relayed requests.

:8791 is published to the public internet as api.text-generator.io through a
cloudflared tunnel. The tunnel connects over loopback, so a loopback peer
address alone does not mean "internal" -- the header also has to be absent of
any sign the request was relayed. Getting this wrong hands out free GPU
inference to anyone who can set a header, which is exactly what happened, so
the routing decision is pinned here rather than left to review.

The gateway runs with no GGUF and a stub aux upstream: a request routed to the
credit gate reaches the stub, a request routed locally cannot and says so. The
two answers are unambiguous, so no GPU is needed.
"""
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

STUB_MARKER = "STUB_AUX_UPSTREAM"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        body = json.dumps({"detail": STUB_MARKER}).encode()
        self.send_response(401)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def post(port, headers):
    payload = json.dumps(
        {"model": "any", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"content-type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode()


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0

    stub_port, gw_port = free_port(), free_port()
    stub = http.server.HTTPServer(("127.0.0.1", stub_port), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()

    env = {
        **os.environ,
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_AUX_UPSTREAM": f"http://127.0.0.1:{stub_port}",
        "OMNISERVE_NATIVE_CHAT_COMPAT_PROXY": "1",
    }
    env.pop("OMNISERVE_NATIVE_LLM_GGUF", None)
    env.pop("OMNISERVE_NATIVE_LLM_UPSTREAM", None)
    env.pop("OMNISERVE_NATIVE_SECRET", None)
    proc = subprocess.Popen(
        [binary, "--port", str(gw_port)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
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
        else:
            print("FAIL: gateway never became ready")
            return 1

        internal = {"X-Omniserve-Internal": "local"}
        cases = [
            ("no header goes to the credit gate", {}, True),
            ("clean loopback header serves locally", internal, False),
            ("relayed via CF-Ray", {**internal, "CF-Ray": "9abcdef-AKL"}, True),
            ("relayed via CF-Connecting-IP", {**internal, "CF-Connecting-IP": "1.2.3.4"}, True),
            ("relayed via X-Forwarded-For", {**internal, "X-Forwarded-For": "1.2.3.4"}, True),
            ("relayed via X-Real-IP", {**internal, "X-Real-IP": "1.2.3.4"}, True),
        ]
        failures = 0
        for name, headers, expect_upstream in cases:
            body = post(gw_port, headers)
            hit_upstream = STUB_MARKER in body
            if hit_upstream != expect_upstream:
                where = "credit gate" if hit_upstream else "local model"
                print(f"FAIL: {name}: went to the {where}: {body[:160]}")
                failures += 1
        if failures:
            return 1
        print("internal-trust tests passed")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        stub.shutdown()


if __name__ == "__main__":
    sys.exit(main())
