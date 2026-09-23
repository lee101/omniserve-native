#!/usr/bin/env python3
"""Image overflow adapter: omniserve-native relays OpenAI-shaped image requests
here (plain http, localhost). If PRIMARY_UPSTREAM is set (another gateway, e.g.
a LAN GPU box) it is tried first; anything it cannot serve (connection error,
5xx, timeout) runs as a RunPod serverless job on the omniserve-ra2-overflow
endpoint and the worker's OpenAI-shaped output is returned unchanged. Stdlib only."""
import http.server, json, os, time, urllib.request, urllib.error

RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ.get("RUNPOD_RA2_ENDPOINT_ID", "tlofa06vj7iab7")
TOKEN = os.environ.get("OVERFLOW_TOKEN", "")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"
DEADLINE_S = float(os.environ.get("OVERFLOW_DEADLINE_S", "580"))
PRIMARY = os.environ.get("PRIMARY_UPSTREAM", "").rstrip("/")
PRIMARY_TOKEN = os.environ.get("PRIMARY_TOKEN", "")
PRIMARY_TIMEOUT_S = float(os.environ.get("PRIMARY_TIMEOUT_S", "240"))


def try_primary(path, raw, tier):
    """Return (status, body) from the primary gateway, or None to fall back."""
    if not PRIMARY:
        return None
    headers = {"Content-Type": "application/json"}
    if PRIMARY_TOKEN:
        headers["Authorization"] = "Bearer " + PRIMARY_TOKEN
    if tier:
        headers["X-Omniserve-Tier"] = tier
    req = urllib.request.Request(PRIMARY + path, data=raw, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=PRIMARY_TIMEOUT_S) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            return exc.code, exc.read()
        print(f"primary {exc.code}; falling back to runpod", flush=True)
    except Exception as exc:
        print(f"primary unavailable ({type(exc).__name__}); falling back to runpod", flush=True)
    return None


def runpod(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + RUNPOD_KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok", "endpoint": ENDPOINT})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if TOKEN and self.headers.get("Authorization", "") != "Bearer " + TOKEN:
            return self.reply(401, {"error": {"message": "invalid token"}})
        if self.path not in ("/v1/images/generations", "/v1/images/edits", "/v1/images/img2img"):
            return self.reply(404, {"error": {"message": "unsupported path"}})
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            primary = try_primary(self.path, raw, self.headers.get("X-Omniserve-Tier", ""))
            if primary is not None:
                code, payload = primary
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            body = json.loads(raw or b"{}")
            job = runpod("POST", "/run", {"input": body})
            job_id, started = job["id"], time.monotonic()
            while time.monotonic() - started < DEADLINE_S:
                state = runpod("GET", "/status/" + job_id)
                status = state.get("status")
                if status == "COMPLETED":
                    out = state.get("output") or {}
                    if isinstance(out, dict) and out.get("error"):
                        return self.reply(502, {"error": {"message": str(out["error"])[:300]}})
                    return self.reply(200, out)
                if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    return self.reply(502, {"error": {"message": "runpod " + status, "detail": str(state.get("error"))[:300]}})
                time.sleep(1.5)
            try:
                runpod("POST", "/cancel/" + job_id)
            except Exception:
                pass
            self.reply(504, {"error": {"message": "runpod overflow timed out"}})
        except Exception as exc:
            self.reply(502, {"error": {"message": f"runpod overflow failed: {type(exc).__name__}"}})

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer((os.environ.get("BIND", "127.0.0.1"), int(os.environ.get("PORT", "18795"))), Handler).serve_forever()
