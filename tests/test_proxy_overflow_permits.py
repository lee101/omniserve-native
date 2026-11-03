#!/usr/bin/env python3
"""A proxied lane with a standing remote must return every permit it takes.

The fast-path try-acquire used to be followed by a second blocking acquire, so
each locally served paid request leaked its permits and the pool eventually
wedged behind a whole-device background waiter.
"""

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(__file__))
from test_image_overflow_routing import CALLER_KEY, free_port, post, status, wait_ready  # noqa: E402


class TtsStub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        payload = json.dumps({"ok": True, "via": self.server.label}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def serve(label):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), TtsStub)
    server.label = label
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0
    local, remote = serve("local"), serve("remote")
    port = free_port()
    env = {key: value for key, value in os.environ.items() if not key.startswith("OMNISERVE_NATIVE_")}
    env.update({
        "OMNISERVE_NATIVE_BIND": "127.0.0.1",
        "OMNISERVE_NATIVE_SLOTS": "4",
        "OMNISERVE_NATIVE_TTS_PERMITS": "2",
        "OMNISERVE_NATIVE_SECRET": CALLER_KEY,
        "OMNISERVE_NATIVE_TTS_UPSTREAM": f"http://127.0.0.1:{local.server_port}",
        "OMNISERVE_NATIVE_TTS_OVERFLOW_UPSTREAM": f"http://127.0.0.1:{remote.server_port}",
        "OMNISERVE_NATIVE_OVERFLOW_TIERS": "paid",
        "OMNISERVE_NATIVE_ADMISSION_TIMEOUT_S": "2",
    })
    paid = {"Authorization": f"Bearer {CALLER_KEY}", "X-Omniserve-Tier": "paid"}
    with tempfile.TemporaryFile() as log:
        gateway = subprocess.Popen([binary, "--port", str(port)], env=env, stdout=log, stderr=log)
        try:
            wait_ready(port, gateway, key=None)
            for i in range(6):
                code, body = post(port, "/v1/audio/speech", {"input": f"hello {i}"}, paid)
                if code != 200 or body.get("via") != "local":
                    print("FAIL: paid speech request was not served locally", code, body)
                    return 1
            admission = status(port)["admission"]
            if admission["used_slots"] != 0 or admission["active"] != 0:
                print("FAIL: permits leaked after locally served overflow-lane requests", admission)
                return 1
            print("proxy overflow permits: all returned")
            return 0
        finally:
            gateway.terminate()
            gateway.wait(timeout=10)
            local.shutdown()
            remote.shutdown()


if __name__ == "__main__":
    sys.exit(main())
