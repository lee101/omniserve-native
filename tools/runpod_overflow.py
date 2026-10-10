#!/usr/bin/env python3
"""Image overflow adapter: omniserve-native relays OpenAI-shaped image requests
here (plain http, localhost). If PRIMARY_UPSTREAM is set (another gateway, e.g.
a LAN GPU box) it is tried first; anything it cannot serve (connection error,
5xx, timeout) runs as a RunPod serverless job on the omniserve-ra2-overflow
endpoint and the worker's OpenAI-shaped output is returned unchanged. Stdlib only."""
import http.client, http.server, json, os, sys, threading, time, urllib.parse, urllib.request, urllib.error

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
# Qwen Image Edit 2511 overflow: requests naming an edit model go to these endpoints as workload
# qwen-edit. Endpoints in RUNPOD_MT_ENDPOINTS run workloads/qwen_mt.py and take an input `task`.
EDIT_MODELS = {"qwen-edit", "qwen-image-edit", "qwen-image-edit-2511", "edit"}
EDIT_ENDPOINTS = [e.strip() for e in os.environ.get("RUNPOD_EDIT_ENDPOINTS", "").split(",") if e.strip()]
MT_ENDPOINTS = {e.strip() for e in os.environ.get("RUNPOD_MT_ENDPOINTS", "").split(",") if e.strip()}
ROUTING = os.environ.get("FRONTIER_ROUTING", "0") == "1"
PRIMARY_BREAKER_S = float(os.environ.get("PRIMARY_BREAKER_S", "0"))
# The RunPod qwen_image worker rejects any input outside workloads/qwen_image.py ALLOWED_INPUTS,
# so routing-only fields a gateway caller sends (model, cache, response_format) must not cross.
INPUT_KEYS = set(filter(None, os.environ.get(
    "RUNPOD_INPUT_KEYS", "workload,kind,profile,prompt,negative_prompt,width,height,size,steps,num_inference_steps,"
    "guidance_scale,seed,output_format,image_base64,strength,n").split(",")))
INPUT_KEYS.add("task")
DEADLINE_S = float(os.environ.get("OVERFLOW_DEADLINE_S", "580"))
PRIMARY = os.environ.get("PRIMARY_UPSTREAM", "").rstrip("/")
# Cloudflare's bot rules reject Python-urllib's default user agent (error 1010).
UA = "omniserve-overflow/1.0"
PRIMARY_TOKEN = os.environ.get("PRIMARY_TOKEN", "")
PRIMARY_TIMEOUT_S = float(os.environ.get("PRIMARY_TIMEOUT_S", "240"))
PRIMARY_CONNECT_S = float(os.environ.get("PRIMARY_CONNECT_S", "3"))
RUNPOD_HTTP_TIMEOUT_S = float(os.environ.get("RUNPOD_HTTP_TIMEOUT_S", "20"))
# An endpoint whose job sits IN_QUEUE this long (no worker picked it up: throttled, no GPUs, bad
# image) is cancelled and the request retried on the next endpoint. 0 disables.
RUNPOD_QUEUE_STALL_S = float(os.environ.get("RUNPOD_QUEUE_STALL_S", "0"))
# Tiers allowed to spend on RunPod (the free primary still serves any tier). Missing header = free.
RUNPOD_TIERS = {t.strip() for t in os.environ.get("RUNPOD_TIERS", "paid,sub,priority").split(",") if t.strip()}
BREAKER_FAILURES = int(os.environ.get("BREAKER_FAILURES", "3"))
BREAKER_COOLDOWN_S = float(os.environ.get("BREAKER_COOLDOWN_S", "30"))
BREAKER_MAX_COOLDOWN_S = float(os.environ.get("BREAKER_MAX_COOLDOWN_S", "600"))


class Breaker:
    """Consecutive-failure circuit breaker. Open for a cooldown that doubles per failed half-open
    probe (capped); after it lapses exactly one caller is let through as the probe."""

    def __init__(self, name, failures=BREAKER_FAILURES, cooldown=BREAKER_COOLDOWN_S,
                 max_cooldown=BREAKER_MAX_COOLDOWN_S, clock=time.monotonic):
        self.name, self.limit, self.base, self.max = name, max(1, failures), cooldown, max(cooldown, max_cooldown)
        self.clock, self.lock = clock, threading.Lock()
        self.fails = self.opens = self.rejects = 0
        self.cooldown, self.open_until = cooldown, 0.0

    def allow(self):
        with self.lock:
            if not self.open_until:
                return True
            now = self.clock()
            if now >= self.open_until:
                self.open_until = now + self.cooldown  # claim the half-open probe
                return True
            self.rejects += 1
            return False

    def record(self, ok):
        with self.lock:
            if ok:
                self.fails, self.open_until, self.cooldown = 0, 0.0, self.base
                return
            self.fails += 1
            if self.fails < self.limit:
                return
            if self.open_until:
                self.cooldown = min(self.cooldown * 2, self.max)
            else:
                self.opens += 1
                print(f"breaker open: {self.name} after {self.fails} failures, {self.cooldown:.0f}s", flush=True)
            self.open_until = self.clock() + self.cooldown

    def retry_after(self):
        with self.lock:
            return max(0.0, self.open_until - self.clock()) if self.open_until else 0.0

    def state(self):
        left = self.retry_after()
        return {"open": left > 0, "open_s": round(left, 1), "consecutive_failures": self.fails,
                "opens": self.opens, "rejects": self.rejects, "cooldown_s": self.cooldown}


# A dead primary (Cloudflare 530/1033 from the tunnel) is decisive after one failure.
PRIMARY_BREAKER = Breaker("primary", failures=1, cooldown=PRIMARY_BREAKER_S or BREAKER_COOLDOWN_S,
                          max_cooldown=max(BREAKER_MAX_COOLDOWN_S, PRIMARY_BREAKER_S or 0))
_BREAKERS = {}
_BREAKERS_LOCK = threading.Lock()


def breaker(endpoint):
    with _BREAKERS_LOCK:
        if endpoint not in _BREAKERS:
            _BREAKERS[endpoint] = Breaker("runpod:" + endpoint)
        return _BREAKERS[endpoint]


def ledger(workload=None, **row):
    if frontier is not None:
        frontier.record(workload=workload or WORKLOAD, source="runpod_overflow", **row)


def workload_of(body):
    names = {str(body.get(k, "")).strip().lower() for k in ("model", "task")}
    return "qwen-edit" if names & EDIT_MODELS else WORKLOAD


def endpoint_order(tier, workload=None):
    """Endpoints for this request, preferred first: the frontier's remote order, then the rest of the pool."""
    workload = workload or WORKLOAD
    pool = EDIT_ENDPOINTS if workload == "qwen-edit" else ENDPOINTS
    order = []
    if ROUTING and frontier is not None and pool:
        for candidate in frontier.get_router().remote_order(workload, tier or "free"):
            endpoint = candidate.split(":", 1)[-1]
            if endpoint in pool and endpoint not in order:
                order.append(endpoint)
    return order + [e for e in pool if e not in order]


def pick_endpoint(tier, workload=None):
    order = endpoint_order(tier, workload)
    return order[0] if order else None


def post_primary(path, raw, headers):
    """(status, body, Retry-After header or None). POST with a short connect timeout and the long render timeout only for the response."""
    url = urllib.parse.urlsplit(PRIMARY + path)
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = cls(url.hostname, url.port, timeout=PRIMARY_CONNECT_S)
    try:
        conn.connect()
        conn.sock.settimeout(PRIMARY_TIMEOUT_S)
        conn.request("POST", url.path + ("?" + url.query if url.query else ""), body=raw, headers=headers)
        r = conn.getresponse()
        return r.status, r.read(), r.getheader("Retry-After")
    finally:
        conn.close()


# A 429 from the primary is "slow down", not "dead": back off for Retry-After instead of
# tripping the (one-failure) dead-primary breaker. Requests meanwhile go to RunPod.
PRIMARY_THROTTLE_DEFAULT_S = float(os.environ.get("PRIMARY_THROTTLE_DEFAULT_S", "5"))
PRIMARY_THROTTLE_MAX_S = float(os.environ.get("PRIMARY_THROTTLE_MAX_S", "120"))
_primary_throttle_until = 0.0


def parse_retry_after(value):
    """Seconds from a Retry-After header (delta-seconds form), bounded; default when absent/bad."""
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return PRIMARY_THROTTLE_DEFAULT_S
    if seconds != seconds or seconds <= 0:
        return PRIMARY_THROTTLE_DEFAULT_S
    return min(seconds, PRIMARY_THROTTLE_MAX_S)


def try_primary(path, raw, tier):
    """Return (status, body) from the primary gateway, or None to fall back.

    Only a primary that is actually unhealthy (5xx, transport error) feeds the breaker. 403/404 are
    about this request (auth, route) and 429 is throttling: the primary answered, so it is alive;
    we still fall back for this request, but a single such reply must not lock the primary out.
    """
    global _primary_throttle_until
    if not PRIMARY or time.monotonic() < _primary_throttle_until or not PRIMARY_BREAKER.allow():
        return None
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if PRIMARY_TOKEN:
        headers["Authorization"] = "Bearer " + PRIMARY_TOKEN
    if tier:
        headers["X-Omniserve-Tier"] = tier
    try:
        code, payload, retry_after = post_primary(path, raw, headers)
        if code < 500 and code not in (403, 404, 429):
            PRIMARY_BREAKER.record(True)
            return code, payload
        if code < 500:
            PRIMARY_BREAKER.record(True)  # reachable: do not count against primary health
            if code == 429:
                _primary_throttle_until = time.monotonic() + parse_retry_after(retry_after)
            print(f"primary {code} (client-side/throttle, not a health failure); falling back to runpod",
                  flush=True)
            return None
        print(f"primary {code}; falling back to runpod", flush=True)
    except Exception as exc:
        print(f"primary unavailable ({type(exc).__name__}); falling back to runpod", flush=True)
    PRIMARY_BREAKER.record(False)
    return None


def runpod(method, path, body=None, endpoint=None):
    req = urllib.request.Request(f"https://api.runpod.ai/v2/{endpoint or ENDPOINTS[0]}" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + RUNPOD_KEY, "Content-Type": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=RUNPOD_HTTP_TIMEOUT_S) as r:
        return json.load(r)


def breaker_states():
    with _BREAKERS_LOCK:
        states = {name: b.state() for name, b in _BREAKERS.items()}
    return {"primary": PRIMARY_BREAKER.state() if PRIMARY else None, "runpod": states}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, code, obj, retry_after=None):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        if retry_after:
            self.send_header("Retry-After", str(retry_after))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok", "endpoint": ENDPOINTS[0], "endpoints": ENDPOINTS,
                                    "edit_endpoints": EDIT_ENDPOINTS, "mt_endpoints": sorted(MT_ENDPOINTS),
                                    "routing": ROUTING, "ledger": bool(frontier and frontier.get_ledger()),
                                    "breakers": breaker_states()})
        led = frontier.get_ledger() if frontier is not None else None
        if led is not None and self.path.startswith("/ledger/summary"):
            return self.reply(200, led.summary())
        if self.path == "/metrics":
            lines = []
            for name, st in [("primary", PRIMARY_BREAKER.state())] * bool(PRIMARY) + sorted(breaker_states()["runpod"].items()):
                lines += [f'overflow_breaker_open{{upstream="{name}"}} {int(st["open"])}',
                          f'overflow_breaker_opens_total{{upstream="{name}"}} {st["opens"]}',
                          f'overflow_breaker_rejects_total{{upstream="{name}"}} {st["rejects"]}']
            raw = ((led.prometheus() if led is not None else "") + "\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.reply(404, {"error": "not found"})

    def run_job(self, endpoint, body, workload, tier, started, budget_s, dropped):
        """One RunPod job. Returns (code, obj, retry_elsewhere); feeds the endpoint's breaker."""
        body = dict(body)
        if endpoint in MT_ENDPOINTS:
            body["task"] = "edit" if workload == "qwen-edit" else "ra2"
        else:
            body.pop("task", None)
        brk = breaker(endpoint)
        state, status, job_id = {}, "SUBMIT_FAILED", None

        def done(code, obj, ok, retry, count=True):
            if count:
                brk.record(ok)
            ledger(workload=workload, backend="runpod", endpoint=endpoint, tier=tier or "free", status=status,
                   job_id=job_id, queue_ms=state.get("delayTime"), exec_ms=state.get("executionTime"),
                   wall_ms=(time.monotonic() - started) * 1000, quality_tier="equal",
                   detail={"http": code, "path": self.path, "dropped": dropped, "retry": retry})
            return code, obj, retry

        try:
            job_id = runpod("POST", "/run", {"input": body}, endpoint)["id"]
        except urllib.error.HTTPError as exc:
            status = f"HTTP_{exc.code}"
            # 4xx other than auth/throttle is this request's fault; anything else is the endpoint's.
            if 400 <= exc.code < 500 and exc.code not in (401, 403, 404, 408, 429):
                return done(502, {"error": {"message": f"runpod rejected job: {exc.code}"}}, True, False)
            return done(502, {"error": {"message": f"runpod submit {exc.code}"}}, False, True)
        except Exception as exc:
            return done(502, {"error": {"message": f"runpod submit failed: {type(exc).__name__}"}}, False, True)
        submitted = time.monotonic()
        status = "TIMED_OUT"
        poll_errors = 0
        while True:
            if time.monotonic() - submitted >= budget_s:
                break
            try:
                state = runpod("GET", "/status/" + job_id, None, endpoint)
                poll_errors = 0
            except Exception:
                poll_errors += 1
                if poll_errors >= 5:
                    status = "POLL_FAILED"
                    break
                time.sleep(2.0)
                continue
            status = state.get("status")
            if status == "COMPLETED":
                out = state.get("output") or {}
                if isinstance(out, dict) and out.get("error"):
                    status = "OUTPUT_ERROR"
                    return done(502, {"error": {"message": str(out["error"])[:300]}}, True, False)
                return done(200, out, True, False)
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                return done(502, {"error": {"message": "runpod " + status, "detail": str(state.get("error"))[:300]}},
                            False, True)
            if (RUNPOD_QUEUE_STALL_S and status == "IN_QUEUE" and
                    time.monotonic() - submitted > RUNPOD_QUEUE_STALL_S):
                status = "QUEUE_STALL"
                break
            time.sleep(1.0 if status == "IN_PROGRESS" else 1.5)
        # Leaving the loop with the job still IN_QUEUE/IN_PROGRESS means our budget ran out, not
        # that the endpoint failed: a timeout (504), not retried and not charged to the breaker.
        budget_exhausted = status in ("TIMED_OUT", "IN_PROGRESS", "IN_QUEUE")
        if budget_exhausted:
            status = "TIMED_OUT"
        try:
            runpod("POST", "/cancel/" + job_id, None, endpoint)
        except Exception:
            pass
        return done(504 if budget_exhausted else 502,
                    {"error": {"message": f"runpod overflow {status.lower()}"}},
                    False, not budget_exhausted, count=not budget_exhausted)

    def do_POST(self):
        if TOKEN and self.headers.get("Authorization", "") != "Bearer " + TOKEN:
            return self.reply(401, {"error": {"message": "invalid token"}})
        if self.path not in ("/v1/images/generations", "/v1/images/edits", "/v1/images/img2img"):
            return self.reply(404, {"error": {"message": "unsupported path"}})
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            tier = self.headers.get("X-Omniserve-Tier", "")
            started = time.monotonic()
            body = json.loads(raw or b"{}")
            workload = workload_of(body)
            primary = try_primary(self.path, raw, tier) if workload == WORKLOAD else None
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
            if (tier or "free") not in RUNPOD_TIERS:
                return self.reply(503, {"error": {"message": f"tier {tier or 'free'} is local-only; retry"}},
                                  retry_after=5)
            dropped = sorted(k for k in body if k not in INPUT_KEYS) if INPUT_KEYS else []
            for key in dropped:
                body.pop(key)
            order = endpoint_order(tier, workload)
            if not order:
                return self.reply(503, {"error": {"message": f"no RunPod endpoint for {workload}"}})
            last = (502, {"error": {"message": "runpod overflow failed"}})
            tried = []
            for endpoint in order:
                remaining = DEADLINE_S - (time.monotonic() - started)
                if remaining < 30:
                    break
                if not breaker(endpoint).allow():
                    continue
                tried.append(endpoint)
                code, obj, retry = self.run_job(endpoint, body, workload, tier, started, remaining, dropped)
                if not retry:
                    return self.reply(code, obj)
                last = (code, obj)
            if not tried:
                wait = min(breaker(e).retry_after() for e in order) or 5
                return self.reply(503, {"error": {"message": "all overflow upstreams unavailable (breakers open); retry"}},
                                  retry_after=max(1, int(wait + 0.999)))
            return self.reply(*last)
        except Exception as exc:
            self.reply(502, {"error": {"message": f"runpod overflow failed: {type(exc).__name__}"}})

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

    Server((os.environ.get("BIND", "127.0.0.1"), int(os.environ.get("PORT", "18795"))), Handler).serve_forever()
