#!/usr/bin/env python3
"""Last-resort image overflow: omniserve-native relays OpenAI-shaped image
requests here (plain http, localhost); they run as RunPod serverless jobs on the
omniserve-ra2-overflow endpoint and the worker's OpenAI-shaped output is
returned unchanged. Stdlib only."""
import http.server, json, os, time, urllib.request, urllib.error

RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ.get("RUNPOD_RA2_ENDPOINT_ID", "tlofa06vj7iab7")
TOKEN = os.environ.get("OVERFLOW_TOKEN", "")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"
DEADLINE_S = float(os.environ.get("OVERFLOW_DEADLINE_S", "580"))


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
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
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
