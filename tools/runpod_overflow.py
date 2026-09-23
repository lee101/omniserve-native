#!/usr/bin/env python3
"""Image overflow adapter: omniserve-native relays OpenAI-shaped image requests
here (plain http, localhost). If PRIMARY_UPSTREAM is set (another gateway, e.g.
a LAN GPU box) it is tried first; anything it cannot serve (connection error,
5xx, timeout) runs as a RunPod serverless job on the omniserve-ra2-overflow
endpoint and the worker's OpenAI-shaped output is returned unchanged. Stdlib only."""
import http.server, json, os, sys, threading, time, urllib.request, urllib.error

sys.path.insert(0, os.environ.get("FRONTIER_LIB", "/nvme0n1-disk/code/omniserve-native"))
try:
    import frontier
except ImportError:
    frontier = None

RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ.get("RUNPOD_RA2_ENDPOINT_ID", "tlofa06vj7iab7")
TOKEN = os.environ.get("OVERFLOW_TOKEN", "")
ENDPOINTS = [e.strip() for e in os.environ.get("RUNPOD_RA2_ENDPOINTS", ENDPOINT).split(",") if e.strip()] or [ENDPOINT]
WORKLOAD = os.environ.get("FRONTIER_WORKLOAD", "ra2")
ROUTING = os.environ.get("FRONTIER_ROUTING", "0") == "1"
PRIMARY_BREAKER_S = float(os.environ.get("PRIMARY_BREAKER_S", "0"))
# The RunPod qwen_image worker rejects any input outside workloads/qwen_image.py ALLOWED_INPUTS,
# so routing-only fields a gateway caller sends (model, cache, response_format) must not cross.
INPUT_KEYS = set(filter(None, os.environ.get(
    "RUNPOD_INPUT_KEYS", "workload,kind,profile,prompt,negative_prompt,width,height,size,steps,num_inference_steps,"
    "guidance_scale,seed,output_format,image_base64,strength,n").split(",")))
_primary_down_until = 0.0
_breaker_lock = threading.Lock()
DEADLINE_S = float(os.environ.get("OVERFLOW_DEADLINE_S", "580"))
PRIMARY = os.environ.get("PRIMARY_UPSTREAM", "").rstrip("/")
# Cloudflare's bot rules reject Python-urllib's default user agent (error 1010).
UA = "omniserve-overflow/1.0"
PRIMARY_TOKEN = os.environ.get("PRIMARY_TOKEN", "")
PRIMARY_TIMEOUT_S = float(os.environ.get("PRIMARY_TIMEOUT_S", "240"))


def ledger(**row):
    if frontier is not None:
        frontier.record(workload=WORKLOAD, source="runpod_overflow", **row)


def trip_primary():
    global _primary_down_until
    if PRIMARY_BREAKER_S > 0:
        with _breaker_lock:
            _primary_down_until = time.monotonic() + PRIMARY_BREAKER_S


def pick_endpoint(tier):
    if ROUTING and frontier is not None:
        for candidate in frontier.get_router().remote_order(WORKLOAD, tier or "free"):
            endpoint = candidate.split(":", 1)[-1]
            if endpoint in ENDPOINTS:
                return endpoint
    return ENDPOINTS[0]


def try_primary(path, raw, tier):
    """Return (status, body) from the primary gateway, or None to fall back."""
    if not PRIMARY or time.monotonic() < _primary_down_until:
        return None
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if PRIMARY_TOKEN:
        headers["Authorization"] = "Bearer " + PRIMARY_TOKEN
    if tier:
        headers["X-Omniserve-Tier"] = tier
    req = urllib.request.Request(PRIMARY + path, data=raw, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=PRIMARY_TIMEOUT_S) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        if exc.code < 500 and exc.code not in (403, 404, 429):
            return exc.code, exc.read()
        print(f"primary {exc.code}; falling back to runpod", flush=True)
    except Exception as exc:
        print(f"primary unavailable ({type(exc).__name__}); falling back to runpod", flush=True)
    trip_primary()
    return None


def runpod(method, path, body=None, endpoint=None):
    req = urllib.request.Request(f"https://api.runpod.ai/v2/{endpoint or ENDPOINTS[0]}" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + RUNPOD_KEY, "Content-Type": "application/json", "User-Agent": UA})
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
            return self.reply(200, {"status": "ok", "endpoint": ENDPOINTS[0], "endpoints": ENDPOINTS,
                                    "routing": ROUTING, "ledger": bool(frontier and frontier.get_ledger())})
        led = frontier.get_ledger() if frontier is not None else None
        if led is not None and self.path.startswith("/ledger/summary"):
            return self.reply(200, led.summary())
        if led is not None and self.path == "/metrics":
            raw = led.prometheus().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if TOKEN and self.headers.get("Authorization", "") != "Bearer " + TOKEN:
            return self.reply(401, {"error": {"message": "invalid token"}})
        if self.path not in ("/v1/images/generations", "/v1/images/edits", "/v1/images/img2img"):
            return self.reply(404, {"error": {"message": "unsupported path"}})
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            tier = self.headers.get("X-Omniserve-Tier", "")
            started = time.monotonic()
            primary = try_primary(self.path, raw, tier)
            if primary is not None:
                code, payload = primary
                ledger(backend="gateway", endpoint="primary", tier=tier or "free", status=str(code),
                       wall_ms=(time.monotonic() - started) * 1000, exec_ms=(time.monotonic() - started) * 1000,
                       quality_tier="equal", est_usd=0.0)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            body = json.loads(raw or b"{}")
            dropped = sorted(k for k in body if k not in INPUT_KEYS) if INPUT_KEYS else []
            for key in dropped:
                body.pop(key)
            endpoint = pick_endpoint(tier)
            job = runpod("POST", "/run", {"input": body}, endpoint)
            job_id, submitted = job["id"], time.monotonic()
            state, status = {}, "TIMED_OUT"

            def done(code, obj):
                ledger(backend="runpod", endpoint=endpoint, tier=tier or "free", status=status, job_id=job_id,
                       queue_ms=state.get("delayTime"), exec_ms=state.get("executionTime"),
                       wall_ms=(time.monotonic() - started) * 1000, quality_tier="equal",
                       detail={"http": code, "path": self.path, "dropped": dropped})
                return self.reply(code, obj)

            while time.monotonic() - submitted < DEADLINE_S:
                state = runpod("GET", "/status/" + job_id, None, endpoint)
                status = state.get("status")
                if status == "COMPLETED":
                    out = state.get("output") or {}
                    if isinstance(out, dict) and out.get("error"):
                        status = "OUTPUT_ERROR"
                        return done(502, {"error": {"message": str(out["error"])[:300]}})
                    return done(200, out)
                if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    return done(502, {"error": {"message": "runpod " + status, "detail": str(state.get("error"))[:300]}})
                time.sleep(1.0 if status == "IN_PROGRESS" else 1.5)
            status = "TIMED_OUT"
            try:
                runpod("POST", "/cancel/" + job_id, None, endpoint)
            except Exception:
                pass
            done(504, {"error": {"message": "runpod overflow timed out"}})
        except Exception as exc:
            self.reply(502, {"error": {"message": f"runpod overflow failed: {type(exc).__name__}"}})

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

    Server((os.environ.get("BIND", "127.0.0.1"), int(os.environ.get("PORT", "18795"))), Handler).serve_forever()
