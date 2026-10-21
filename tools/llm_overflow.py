#!/usr/bin/env python3
"""LLM overflow sidecar for omniserve-native.

serve: the gateway relays saturated /v1/chat/completions and /v1/completions here
(plain http). Requests go to a ready RunPod pod with free capacity, else the RunPod
load-balancing serverless endpoint, failing over only before any byte reaches the
caller. A scaler promotes sustained overflow to dedicated pods and retires them on
idle, TTL or the spend cap. setup: create/update the load-balancing endpoint.
pods: list the pods this sidecar owns. See performance/LLM_OVERFLOW.md.

uv run --no-project --with aiohttp tools/llm_overflow.py serve
"""
import argparse, asyncio, hashlib, hmac, json, math, os, secrets, signal, sys, time
from collections import deque
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web

DEFAULTS = dict(
    BIND="127.0.0.1", PORT=18796,
    RUNPOD_API_KEY="", RUNPOD_API_BASE="https://api.runpod.io/v2",
    RUNPOD_LLM_ENDPOINT_ID="", SERVERLESS_URL="", SERVERLESS_SECRET="",
    OVERFLOW_DEADLINE_S=90.0, SERVERLESS_RETRY_S=6.0, CONNECT_TIMEOUT_S=10.0, READ_TIMEOUT_S=330.0,
    POD_IMAGE="", POD_CAPACITY=32, MAX_PODS=1, MAX_USD_HR=2.0,
    POD_PROMOTE_CONCURRENCY=8.0, POD_PROMOTE_UTIL=0.70, POD_DEMOTE_CONCURRENCY=2.0,
    POD_IDLE_S=1200.0, POD_TTL_S=21600.0, POD_BOOT_TIMEOUT_S=900.0, POD_DRAIN_S=120.0, POD_FAILS_UNREADY=3,
    SCALE_INTERVAL_S=15.0, SCALE_COOLDOWN_S=300.0, RECONCILE_S=60.0,
    SHORT_WINDOW_S=120.0, LONG_WINDOW_S=900.0,
    POD_GPU_TYPES="NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 5090,NVIDIA L40S,NVIDIA A40,NVIDIA RTX A6000",
    POD_CLOUD="SECURE", POD_PORT=8000, POD_DISK_GB=40, POD_MIN_CUDA="12.9",
    POD_NAME_PREFIX="omniserve-llm-overflow-", POD_URL_TEMPLATE="https://{id}-{port}.proxy.runpod.net",
    POD_SECRET_KEY="", HF_TOKEN="", REMOTE_ENV_JSON="{}", REGISTRY_AUTH_ID="",
    WARM_CONCURRENCY=2.0, WARM_INTERVAL_S=60.0,
    COLD_REJECT=1, WARM_TTL_S=60.0, COLD_PING_S=15.0,
)
PATHS = ("/v1/chat/completions", "/v1/completions")
STRIP = {"authorization", "x-api-key", "x-rapid-api-key", "secret", "cookie", "host", "content-length",
         "connection", "keep-alive", "transfer-encoding", "te", "upgrade", "proxy-authorization",
         "proxy-connection", "accept-encoding", "expect"}
PASS = {"content-type", "content-encoding", "cache-control", "x-request-id", "x-runpod-worker-id", "retry-after"}
RETRY_STATUS = {401, 403, 404, 408, 429}
LIVE = {"PROVISIONING", "STARTING", "RUNNING"}


def log(msg):
    print(msg, flush=True)


class Cfg:
    def __init__(self, env=None, **over):
        env = os.environ if env is None else env
        for k, d in DEFAULTS.items():
            v = over.get(k, env.get(k, d))
            setattr(self, k, type(d)(v) if v != "" or isinstance(d, str) else d)
        if not self.HF_TOKEN:
            p = Path.home() / ".cache/huggingface/token"
            self.HF_TOKEN = p.read_text().strip() if p.exists() else ""
        if not self.SERVERLESS_URL and self.RUNPOD_LLM_ENDPOINT_ID:
            self.SERVERLESS_URL = f"https://{self.RUNPOD_LLM_ENDPOINT_ID}.api.runpod.ai"
        self.gpus = [g.strip() for g in self.POD_GPU_TYPES.split(",") if g.strip()]
        self.clouds = [c.strip().upper() for c in self.POD_CLOUD.split(",") if c.strip()]
        self.remote_env = json.loads(self.REMOTE_ENV_JSON or "{}")

    def public(self):
        hide = ("KEY", "TOKEN", "SECRET", "ENV_JSON")
        return {k: getattr(self, k) for k in DEFAULTS if not any(h in k for h in hide)}


class Ewma:
    def __init__(self, tau):
        self.tau, self.v = tau, 0.0

    def update(self, x, dt):
        if dt > 0:
            self.v += (1 - math.exp(-dt / self.tau)) * (x - self.v)
        return self.v


class Gauge:
    """In-flight count with a time integral, so ticks see mean concurrency, not a sample."""

    def __init__(self, clock):
        self.clock, self.n, self.area = clock, 0, 0.0
        self.t = self.mark_t = clock()
        self.mark_area = 0.0

    def _acc(self):
        now = self.clock()
        self.area += self.n * (now - self.t)
        self.t = now

    def add(self, d):
        self._acc()
        self.n += d

    def mean(self):
        self._acc()
        dt = self.t - self.mark_t
        v = (self.area - self.mark_area) / dt if dt > 0 else float(self.n)
        self.mark_area, self.mark_t = self.area, self.t
        return v, dt


class Backend:
    def __init__(self, kind, name, url, auth, capacity, clock):
        self.kind, self.name, self.url, self.auth, self.capacity = kind, name, url.rstrip("/"), auth, capacity
        self.gauge = Gauge(clock)
        self.ready, self.draining = kind == "serverless", False
        self.fails = self.requests = self.failovers = self.stream_errors = 0

    def state(self):
        return {"kind": self.kind, "name": self.name, "ready": self.ready, "draining": self.draining,
                "inflight": self.gauge.n, "capacity": self.capacity, "requests": self.requests,
                "failovers": self.failovers, "stream_errors": self.stream_errors, "consecutive_fails": self.fails}


class Pod(Backend):
    def __init__(self, cfg, clock, id, name, secret, gpu="", cost=0.0, created=None, status="PROVISIONING"):
        url = cfg.POD_URL_TEMPLATE.format(id=id, port=cfg.POD_PORT)
        super().__init__("pod", name, url, {"secret": secret}, cfg.POD_CAPACITY, clock)
        self.id, self.gpu, self.cost, self.status = id, gpu, cost, status
        self.created = clock() if created is None else created
        self.ready = False
        self.ready_at = self.low_since = self.drain_since = None
        self.drain_reason = ""
        self.long = Ewma(cfg.LONG_WINDOW_S / 2)

    def state(self):
        return {**super().state(), "id": self.id, "gpu": self.gpu, "usd_hr": self.cost, "status": self.status,
                "age_s": round(self.gauge.clock() - self.created), "long_concurrency": round(self.long.v, 2),
                "drain_reason": self.drain_reason}


def pod_secret(cfg, name):
    master = (cfg.POD_SECRET_KEY or "llm-overflow:" + cfg.RUNPOD_API_KEY).encode()
    return hmac.new(master, name.encode(), hashlib.sha256).hexdigest()


def parse_ts(v):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def retryable(status, raw):
    return status >= 500 or status in RETRY_STATUS or b"no worker" in raw.lower()


class Router:
    def __init__(self, cfg, session, clock=time.time):
        self.cfg, self.session, self.clock = cfg, session, clock
        self.total = Gauge(clock)
        self.pods = {}
        self.requests = self.unavailable = self.client_gone = self.cold_rejects = 0
        self.warm_until = self.last_wake = -math.inf
        self.serverless = None
        if cfg.SERVERLESS_URL:
            auth = {"Authorization": "Bearer " + cfg.RUNPOD_API_KEY} if cfg.RUNPOD_API_KEY else {}
            if cfg.SERVERLESS_SECRET:
                auth["secret"] = cfg.SERVERLESS_SECRET
            self.serverless = Backend("serverless", "serverless", cfg.SERVERLESS_URL, auth, 0, clock)

    def free_pods(self):
        pods = [p for p in self.pods.values() if p.ready and not p.draining and p.gauge.n < p.capacity]
        return sorted(pods, key=lambda p: p.gauge.n / p.capacity)

    def warm(self):
        return self.clock() < self.warm_until

    def mark_warm(self):
        self.warm_until = self.clock() + self.cfg.WARM_TTL_S

    def wake(self):
        if self.serverless is None or self.clock() - self.last_wake < self.cfg.COLD_PING_S:
            return
        self.last_wake = self.clock()
        asyncio.ensure_future(self.ping_serverless(self.cfg.OVERFLOW_DEADLINE_S))

    async def ping_serverless(self, timeout_s):
        sl = self.serverless
        try:
            async with self.session.get(sl.url + "/ping", headers=sl.auth,
                                        timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
                await r.read()
                if r.status == 200:
                    self.mark_warm()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            pass

    async def handle(self, request):
        if request.path not in PATHS:
            return web.json_response({"error": {"message": "unsupported path"}}, status=404)
        # A cold serverless start (2-60 s) is slower than the caller's local queue, which frees
        # a sequence within seconds: answer 429 so the gateway serves locally, and wake the
        # endpoint so overflow that persists finds a warm worker.
        if self.cfg.COLD_REJECT and not self.free_pods() and not self.warm():
            self.cold_rejects += 1
            self.wake()
            return web.json_response({"error": {"message": "overflow capacity is cold", "type": "overflow_cold"}},
                                     status=429, headers={"Retry-After": "2"})
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in STRIP}
        headers["Accept-Encoding"] = "identity"
        started = time.monotonic()
        self.requests += 1
        self.total.add(1)
        tried, last, sl_tries = set(), "no backend", 0
        try:
            while True:
                pods = [p for p in self.free_pods() if p.name not in tried]
                b = pods[0] if pods else self.serverless
                if b is None:
                    break
                if b.kind == "pod":
                    tried.add(b.name)
                else:
                    if sl_tries:
                        left = self.cfg.OVERFLOW_DEADLINE_S - (time.monotonic() - started)
                        if left <= 0:
                            break
                        await asyncio.sleep(min(self.cfg.SERVERLESS_RETRY_S, left))
                        if any(p.name not in tried for p in self.free_pods()):
                            continue
                    sl_tries += 1
                r = await self.attempt(b, request, body, headers)
                if not isinstance(r, str):
                    return r
                b.failovers += 1
                last = f"{b.name}:{r}"
                log(f"failover from={b.kind}:{b.name} reason={r} elapsed={time.monotonic() - started:.1f}s")
            self.unavailable += 1
            return web.json_response({"error": {"message": f"llm overflow unavailable ({last})",
                                                 "type": "overflow_unavailable"}},
                                     status=503, headers={"Retry-After": "5"})
        finally:
            self.total.add(-1)

    async def attempt(self, b, request, body, headers):
        cfg = self.cfg
        b.requests += 1
        b.gauge.add(1)
        resp = None
        try:
            timeout = aiohttp.ClientTimeout(total=None, connect=cfg.CONNECT_TIMEOUT_S, sock_read=cfg.READ_TIMEOUT_S)
            async with self.session.post(b.url + request.path_qs, data=body, headers={**headers, **b.auth},
                                         timeout=timeout) as r:
                if r.status >= 400:
                    raw = await r.content.read(1 << 20)
                    if retryable(r.status, raw):
                        self.fail(b)
                        return "no_workers" if b"no worker" in raw.lower() else str(r.status)
                    b.fails = 0
                    return web.Response(status=r.status, body=raw, headers=self.out_headers(r, b))
                if b.kind == "serverless":
                    self.mark_warm()
                resp = web.StreamResponse(status=r.status, headers=self.out_headers(r, b))
                if r.content_length is not None:
                    resp.content_length = r.content_length
                await resp.prepare(request)
                b.fails = 0
                try:
                    async for chunk in r.content.iter_any():
                        try:
                            await resp.write(chunk)
                        except (ConnectionError, aiohttp.ClientConnectionResetError):
                            self.client_gone += 1
                            return resp
                    await resp.write_eof()
                except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                    b.stream_errors += 1
                    log(f"stream error backend={b.kind}:{b.name} {type(e).__name__} (no retry after bytes)")
                    if request.transport:
                        request.transport.close()
                return resp
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            if resp is not None:
                return resp
            self.fail(b)
            return type(e).__name__
        finally:
            b.gauge.add(-1)

    def fail(self, b):
        b.fails += 1
        if b.kind == "pod" and b.ready and b.fails >= self.cfg.POD_FAILS_UNREADY:
            b.ready = False
            log(f"pod unready pod={b.id} after {b.fails} failures")

    @staticmethod
    def out_headers(r, b):
        h = {k: v for k, v in r.headers.items() if k.lower() in PASS}
        h["X-Overflow-Backend"] = f"{b.kind}:{b.name}"
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            h["X-Accel-Buffering"] = "no"
        return h

    def state(self):
        return {"inflight": self.total.n, "requests": self.requests, "unavailable": self.unavailable,
                "client_gone": self.client_gone, "cold_rejects": self.cold_rejects,
                "serverless_warm_s": max(0, round(self.warm_until - self.clock())) if self.warm() else 0,
                "serverless": self.serverless.state() if self.serverless else None,
                "pods": [p.state() for p in self.pods.values()]}


class RunPodError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"runpod {status}: {detail}")
        self.status, self.detail = status, detail


class RunPodAPI:
    """REST API v2 (https://api.runpod.io/v2)."""

    def __init__(self, key, base, session):
        self.key, self.base, self.session = key, base.rstrip("/"), session

    async def req(self, method, path, body=None, params=None):
        for attempt in range(3):
            async with self.session.request(method, self.base + path, json=body, params=params,
                                            headers={"Authorization": "Bearer " + self.key},
                                            timeout=aiohttp.ClientTimeout(total=30)) as r:
                text = await r.text()
                if r.status == 429 and attempt < 2:
                    await asyncio.sleep(float(r.headers.get("Retry-After") or 5))
                    continue
                if r.status >= 400:
                    raise RunPodError(r.status, text[:300])
                return json.loads(text) if text.strip() else {}

    async def pages(self, path, key, params=None):
        out, params = [], dict(params or {})
        while True:
            d = await self.req("GET", path, params=params)
            out += d.get(key) or []
            pg = d.get("pagination") or {}
            if not pg.get("hasNextPage") or not pg.get("nextCursor"):
                return out
            params["cursor"] = pg["nextCursor"]

    async def list_pods(self):
        return await self.pages("/pods", "pods")

    async def create_pod(self, body):
        return await self.req("POST", "/pods", body)

    async def terminate_pod(self, pod_id):
        try:
            await self.req("DELETE", f"/pods/{pod_id}")
        except RunPodError as e:
            if e.status != 404:
                raise

    async def gpu_types(self):
        return await self.pages("/catalog/gpus", "gpus")

    async def list_endpoints(self):
        return await self.pages("/serverless", "endpoints")

    async def create_endpoint(self, body):
        return await self.req("POST", "/serverless", body)

    async def update_endpoint(self, eid, body):
        return await self.req("PATCH", f"/serverless/{eid}", body)


class Scaler:
    def __init__(self, cfg, router, api, clock=time.time):
        self.cfg, self.router, self.api, self.clock = cfg, router, api, clock
        self.short, self.long = Ewma(cfg.SHORT_WINDOW_S / 2), Ewma(cfg.LONG_WINDOW_S / 2)
        self.last_action = self.last_reconcile = self.last_warm = self.prices_at = -math.inf
        self.prices = {}
        self.decisions = deque(maxlen=50)
        self.blocked = ""

    def decide(self, msg):
        line = f"scale {msg} short={self.short.v:.1f} long={self.long.v:.1f}"
        self.decisions.append({"t": round(self.clock()), "msg": line})
        log(line)

    def live(self):
        return [p for p in self.router.pods.values() if not p.draining]

    def spend(self):
        return sum(p.cost for p in self.router.pods.values())

    async def run(self):
        while True:
            try:
                await self.tick()
            except Exception as e:
                log(f"scaler error {type(e).__name__}: {e}")
            await asyncio.sleep(self.cfg.SCALE_INTERVAL_S)

    async def tick(self):
        now = self.clock()
        mean, dt = self.router.total.mean()
        self.short.update(mean, dt)
        self.long.update(mean, dt)
        for p in self.router.pods.values():
            p.long.update(*p.gauge.mean())
        if self.api and now - self.last_reconcile >= self.cfg.RECONCILE_S:
            self.last_reconcile = now
            await self.reconcile()
        await self.probe(now)
        if self.api:
            await self.retire(now)
            await self.maybe_start(now)
        self.maybe_warm(now)

    async def reconcile(self):
        cfg, now = self.cfg, self.clock()
        seen = set()
        for d in await self.api.list_pods():
            name = d.get("name") or ""
            if not name.startswith(cfg.POD_NAME_PREFIX):
                continue
            pid, status = d["id"], d.get("status", "")
            seen.add(pid)
            p = self.router.pods.get(pid)
            if p is None and status in LIVE | {"EXITED", "ERROR"}:
                secret = (d.get("env") or {}).get("OMNISERVE_NATIVE_SECRET") or pod_secret(cfg, name)
                p = Pod(cfg, self.clock, pid, name, secret, (d.get("gpu") or {}).get("id", ""),
                        float(d.get("cost") or 0), parse_ts(d.get("createdAt")) or now, status)
                p.created = min(p.created, now)
                self.router.pods[pid] = p
                self.decide(f"adopt pod={pid} gpu={p.gpu} status={status} usd_hr={p.cost}")
            if p is None:
                continue
            p.status, p.cost = status, float(d.get("cost") or p.cost)
            if status in ("EXITED", "ERROR") and not p.draining:
                self.drain(p, status.lower())
        for pid, p in list(self.router.pods.items()):
            if pid not in seen and now - p.created > 120:
                del self.router.pods[pid]
                self.decide(f"forget pod={pid} (gone from api)")

    async def probe(self, now):
        pods = [p for p in self.router.pods.values() if not p.draining]
        await asyncio.gather(*(self.ping(p, now) for p in pods))

    async def ping(self, p, now):
        try:
            async with self.router.session.get(p.url + "/ping", headers=p.auth,
                                               timeout=aiohttp.ClientTimeout(total=10)) as r:
                ok = r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            ok = False
        if ok:
            p.fails = 0
            if not p.ready:
                p.ready = True
                if p.ready_at is None:
                    p.ready_at = now
                    p.long.v = min(self.long.v, p.capacity)
                    self.decide(f"ready pod={p.id} boot_s={now - p.created:.0f}")
                else:
                    log(f"pod ready again pod={p.id}")
        elif p.ready:
            self.router.fail(p)
        elif p.ready_at is None and now - p.created > self.cfg.POD_BOOT_TIMEOUT_S:
            self.drain(p, "boot_timeout")

    def drain(self, p, reason):
        p.draining, p.drain_since, p.drain_reason = True, self.clock(), reason
        self.decide(f"drain pod={p.id} reason={reason} inflight={p.gauge.n}")

    async def retire(self, now):
        cfg = self.cfg
        for p in sorted(self.live(), key=lambda p: (p.long.v, -p.cost)):
            if self.spend() - sum(q.cost for q in self.router.pods.values() if q.draining) <= cfg.MAX_USD_HR:
                break
            self.drain(p, "spend_cap")
        for p in self.live():
            if now - p.created > cfg.POD_TTL_S:
                self.drain(p, "ttl")
            elif p.ready_at is not None and p.long.v < cfg.POD_DEMOTE_CONCURRENCY:
                p.low_since = p.low_since or now
                if now - p.low_since >= cfg.POD_IDLE_S and now - self.last_action >= cfg.SCALE_COOLDOWN_S:
                    self.drain(p, "idle")
                    self.last_action = now
            else:
                p.low_since = None
        for p in [p for p in self.router.pods.values() if p.draining]:
            if p.gauge.n == 0 or now - p.drain_since >= cfg.POD_DRAIN_S:
                try:
                    await self.api.terminate_pod(p.id)
                except RunPodError as e:
                    log(f"terminate failed pod={p.id} {e}")
                    continue
                self.router.pods.pop(p.id, None)
                self.decide(f"stop pod={p.id} reason={p.drain_reason} age_s={now - p.created:.0f} usd_hr={p.cost}")

    def want_pod(self):
        cfg, live = self.cfg, self.live()
        threshold = cfg.POD_PROMOTE_CONCURRENCY + sum(p.capacity * cfg.POD_PROMOTE_UTIL for p in live)
        return (len(live) < cfg.MAX_PODS and self.long.v >= threshold
                and self.short.v >= cfg.POD_PROMOTE_CONCURRENCY)

    async def maybe_start(self, now):
        cfg = self.cfg
        if not self.want_pod() or now - self.last_action < cfg.SCALE_COOLDOWN_S:
            return
        if not cfg.POD_IMAGE:
            return self.block("promotion wanted but POD_IMAGE unset")
        if now - self.prices_at > 3600:
            self.prices = {g["id"]: g.get("price") or {} for g in await self.api.gpu_types()}
            self.prices_at = now
        spend = self.spend()
        name = cfg.POD_NAME_PREFIX + secrets.token_hex(4)
        env = {**cfg.remote_env, "HF_TOKEN": cfg.HF_TOKEN, "OMNISERVE_NATIVE_SECRET": pod_secret(cfg, name),
               "PORT": str(cfg.POD_PORT)}
        tried = []
        for gpu in cfg.gpus:
            for cloud in cfg.clouds:
                price = float((self.prices.get(gpu) or {}).get(cloud.lower()) or 0)
                if price <= 0 or spend + price > cfg.MAX_USD_HR:
                    continue
                gcfg = {"id": gpu, "count": 1}
                if cfg.POD_MIN_CUDA:
                    gcfg["minCudaVersion"] = cfg.POD_MIN_CUDA
                body = {"name": name, "image": cfg.POD_IMAGE, "cloud": cloud, "gpu": gcfg,
                        "disk": cfg.POD_DISK_GB, "ports": [f"{cfg.POD_PORT}/http"], "env": env}
                if cfg.REGISTRY_AUTH_ID:
                    body["registry"] = cfg.REGISTRY_AUTH_ID
                try:
                    d = await self.api.create_pod(body)
                except RunPodError as e:
                    tried.append(f"{gpu}/{cloud}:{e.status}")
                    if e.status in (402, 429):
                        break
                    continue
                p = Pod(cfg, self.clock, d["id"], name, env["OMNISERVE_NATIVE_SECRET"], gpu,
                        float(d.get("cost") or price), now, d.get("status", "PROVISIONING"))
                self.router.pods[p.id] = p
                self.last_action = now
                self.blocked = ""
                self.decide(f"start pod={p.id} gpu={gpu} cloud={cloud} usd_hr={p.cost} spend={spend + p.cost:.2f}")
                return
            else:
                continue
            break
        self.last_action = now
        self.block(f"promotion wanted, no pod placed spend={spend:.2f} cap={cfg.MAX_USD_HR} tried={tried}")

    def block(self, why):
        if why != self.blocked:
            self.blocked = why
            self.decide(why)

    def maybe_warm(self, now):
        sl = self.router.serverless
        if (sl is None or self.router.free_pods() or self.short.v < self.cfg.WARM_CONCURRENCY
                or self.short.v < 1.5 * self.long.v or now - self.last_warm < self.cfg.WARM_INTERVAL_S):
            return
        self.last_warm = now
        self.decide("warm serverless /ping")
        asyncio.ensure_future(self._warm(sl))

    async def _warm(self, sl):
        await self.router.ping_serverless(30)

    def state(self):
        return {"short_concurrency": round(self.short.v, 2), "long_concurrency": round(self.long.v, 2),
                "spend_usd_hr": round(self.spend(), 3), "want_pod": self.want_pod(), "blocked": self.blocked,
                "last_action_age_s": None if self.last_action == -math.inf else round(self.clock() - self.last_action),
                "decisions": list(self.decisions)[-10:]}


def build_app(cfg, router, scaler):
    async def health(_):
        return web.json_response({"status": "ok"})

    async def status(_):
        return web.json_response({"router": router.state(), "scaler": scaler.state(), "config": cfg.public()})

    app = web.Application(client_max_size=64 << 20)
    app.router.add_get("/health", health)
    app.router.add_get("/status", status)
    for p in PATHS:
        app.router.add_post(p, router.handle)
    return app


async def serve(cfg, terminate_on_exit):
    session = aiohttp.ClientSession(auto_decompress=False, connector=aiohttp.TCPConnector(limit=0))
    api_session = aiohttp.ClientSession()
    api = RunPodAPI(cfg.RUNPOD_API_KEY, cfg.RUNPOD_API_BASE, api_session) if cfg.RUNPOD_API_KEY else None
    router = Router(cfg, session)
    scaler = Scaler(cfg, router, api)
    runner = web.AppRunner(build_app(cfg, router, scaler), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, cfg.BIND, cfg.PORT).start()
    log(f"llm overflow on {cfg.BIND}:{cfg.PORT} serverless={cfg.SERVERLESS_URL or '-'} "
        f"pods={'on' if api and cfg.POD_IMAGE else 'off'} max_pods={cfg.MAX_PODS} cap=${cfg.MAX_USD_HR}/hr")
    task = asyncio.ensure_future(scaler.run())
    stop = asyncio.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(s, stop.set)
    await stop.wait()
    task.cancel()
    if terminate_on_exit and api:
        for p in list(router.pods.values()):
            await api.terminate_pod(p.id)
            log(f"scale stop pod={p.id} reason=exit")
    await runner.cleanup()
    await session.close()
    await api_session.close()


def endpoint_body(a, cfg):
    env = {**cfg.remote_env, "PORT": str(a.port), "PORT_HEALTH": str(a.port), "HF_TOKEN": cfg.HF_TOKEN}
    if cfg.SERVERLESS_SECRET:
        env["OMNISERVE_NATIVE_SECRET"] = cfg.SERVERLESS_SECRET
    gpu = {"pools": a.pools.split(","), "count": 1}
    if a.min_cuda:
        gpu["minCudaVersion"] = a.min_cuda
    if a.exclude_types:
        gpu["excludedTypes"] = a.exclude_types.split(",")
    body = {"name": a.name, "type": "LOAD_BALANCER", "image": a.image, "gpu": gpu,
            "workers": {"min": 0, "max": a.workers_max, "idleTimeout": a.idle_timeout},
            "scaling": {"type": "REQUEST_COUNT", "requestCount": a.request_count},
            "ports": [f"{a.port}/http"], "env": env, "disk": a.disk, "flashboot": a.flashboot}
    if cfg.REGISTRY_AUTH_ID:
        body["registry"] = cfg.REGISTRY_AUTH_ID
    return body


def redact(body):
    env = {k: ("***" if any(h in k for h in ("TOKEN", "SECRET", "KEY")) else v) for k, v in body.get("env", {}).items()}
    return {**body, "env": env}


async def setup(a, cfg):
    body = endpoint_body(a, cfg)
    if not cfg.HF_TOKEN:
        log("warning: no HF_TOKEN (env or ~/.cache/huggingface/token); private model download will fail")
    async with aiohttp.ClientSession() as session:
        api = RunPodAPI(cfg.RUNPOD_API_KEY, cfg.RUNPOD_API_BASE, session)
        found = [e for e in await api.list_endpoints() if e.get("name") == a.name]
        if found and found[0].get("type") != "LOAD_BALANCER":
            sys.exit(f"endpoint {found[0]['id']} named {a.name} is {found[0].get('type')}, not LOAD_BALANCER")
        if found:
            eid = found[0]["id"]
            patch = {k: v for k, v in body.items() if k not in ("name", "type")}
            log(f"PATCH {cfg.RUNPOD_API_BASE}/serverless/{eid}\n{json.dumps(redact(patch), indent=1)}")
            if a.apply:
                await api.update_endpoint(eid, patch)
        else:
            log(f"POST {cfg.RUNPOD_API_BASE}/serverless\n{json.dumps(redact(body), indent=1)}")
            if not a.apply:
                return log("dry run; pass --apply to create")
            eid = (await api.create_endpoint(body))["id"]
        log(f"{'applied' if a.apply else 'dry run'}; RUNPOD_LLM_ENDPOINT_ID={eid}")


async def list_pods(cfg):
    async with aiohttp.ClientSession() as session:
        for d in await RunPodAPI(cfg.RUNPOD_API_KEY, cfg.RUNPOD_API_BASE, session).list_pods():
            if (d.get("name") or "").startswith(cfg.POD_NAME_PREFIX):
                log(f"{d['id']} {d['name']} {d.get('status')} {(d.get('gpu') or {}).get('id')} "
                    f"${d.get('cost')}/hr created={d.get('createdAt')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--terminate-on-exit", action="store_true")
    st = sub.add_parser("setup", help="create/update the load-balancing endpoint (dry run unless --apply)")
    st.add_argument("--image", required=True)
    st.add_argument("--name", default="omniserve-llm-overflow")
    st.add_argument("--workers-max", type=int, default=2)
    st.add_argument("--idle-timeout", type=int, default=90)
    st.add_argument("--request-count", type=int, default=24, help="in-flight requests per worker before scaling out")
    st.add_argument("--pools", default="ADA_24,ADA_32_PRO,AMPERE_48,ADA_48_PRO")
    st.add_argument("--min-cuda", default="12.9")
    st.add_argument("--exclude-types", default="NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb",
                    help="GPU type ids to drop from the pools (MIG slices are a fraction of the card)")
    st.add_argument("--port", type=int, default=8000)
    st.add_argument("--disk", type=int, default=30)
    st.add_argument("--flashboot", default="FLASHBOOT", choices=["OFF", "FLASHBOOT", "PRIORITY_FLASHBOOT"])
    st.add_argument("--apply", action="store_true")
    sub.add_parser("pods", help="list pods owned by this sidecar")
    a = ap.parse_args()
    cfg = Cfg()
    if a.cmd in ("setup", "pods") and not cfg.RUNPOD_API_KEY:
        sys.exit("RUNPOD_API_KEY required")
    if a.cmd == "setup":
        asyncio.run(setup(a, cfg))
    elif a.cmd == "pods":
        asyncio.run(list_pods(cfg))
    else:
        asyncio.run(serve(cfg, getattr(a, "terminate_on_exit", False)))


if __name__ == "__main__":
    main()
