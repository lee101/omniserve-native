"""tools/llm_overflow.py: router passthrough/failover and scaler policy.

uv run --no-project --with pytest --with aiohttp pytest tests/test_llm_overflow.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import llm_overflow as lo  # noqa: E402

BODY = {"model": "gemma-roleplay-v2", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


class Fake:
    def __init__(self, behavior):
        self.behavior, self.calls = behavior, []

    async def handler(self, req):
        self.calls.append({"headers": dict(req.headers), "body": await req.read(), "path": req.path})
        return await self.behavior(self, req)

    def app(self):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.handler)
        return app


def sse(gate=None, hold=None):
    async def run(fake, req):
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(req)
        await r.write(b'data: {"i":0}\n\n')
        if gate:
            await gate.wait()
        if hold:
            await hold.wait()
        await r.write(b'data: {"i":1}\n\n')
        await r.write(b"data: [DONE]\n\n")
        await r.write_eof()
        return r
    return run


def status(code, body=b'{"error":"x"}', then=None, times=10**9):
    async def run(fake, req):
        if then and len(fake.calls) > times:
            return await then(fake, req)
        return web.Response(status=code, body=body, content_type="application/json")
    return run


async def abort_after_first(fake, req):
    r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await r.prepare(req)
    await r.write(b'data: {"i":0}\n\n')
    await asyncio.sleep(0.05)
    req.transport.close()
    return r


async def start(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def cfg(**kw):
    base = dict(HF_TOKEN="hf_test", RUNPOD_API_KEY="rk_test", SERVERLESS_RETRY_S=0.05, OVERFLOW_DEADLINE_S=1.0,
                POD_SECRET_KEY="k", CONNECT_TIMEOUT_S=2.0)
    return lo.Cfg(env={}, **{**base, **kw})


class Rig:
    """Router on a real port with optional fake pod/serverless backends."""

    def __init__(self, pod=None, serverless=None, capacity=32, **kw):
        self.pod_b, self.sl_b, self.capacity, self.kw = pod, serverless, capacity, kw

    async def __aenter__(self):
        self.runners = []
        sl_url = ""
        if self.sl_b:
            r, sl_url = await start(self.sl_b.app())
            self.runners.append(r)
        self.cfg = cfg(SERVERLESS_URL=sl_url, POD_CAPACITY=self.capacity, **self.kw)
        self.session = aiohttp.ClientSession(auto_decompress=False)
        self.router = lo.Router(self.cfg, self.session)
        if self.pod_b is not None:
            if isinstance(self.pod_b, str):
                url = self.pod_b
            else:
                r, url = await start(self.pod_b.app())
                self.runners.append(r)
            p = lo.Pod(self.cfg, lo.time.time, "p1", "omniserve-llm-overflow-aa", "podsecret")
            p.url, p.ready = url, True
            self.router.pods["p1"] = p
            self.pod = p
        r, self.url = await start(lo.build_app(self.cfg, self.router, lo.Scaler(self.cfg, self.router, None)))
        self.runners.append(r)
        self.client = aiohttp.ClientSession()
        return self

    def post(self, path="/v1/chat/completions"):
        return self.client.post(self.url + path, json=BODY,
                                headers={"Authorization": "Bearer caller-key", "X-API-Key": "caller-key",
                                         "X-Omniserve-Tier": "paid"})

    async def __aexit__(self, *a):
        await self.client.close()
        await self.session.close()
        for r in reversed(self.runners):
            await r.cleanup()


def test_streaming_passthrough_unbuffered_and_auth_swapped():
    async def go():
        gate = asyncio.Event()
        pod, sl = Fake(sse(gate)), Fake(sse())
        async with Rig(pod, sl) as rig:
            async with rig.post() as r:
                assert r.status == 200
                assert r.headers["Content-Type"].startswith("text/event-stream")
                assert r.headers["X-Overflow-Backend"] == "pod:omniserve-llm-overflow-aa"
                first = await asyncio.wait_for(r.content.readany(), 2)
                assert first == b'data: {"i":0}\n\n' and not gate.is_set()
                gate.set()
                rest = await r.read()
            assert rest.endswith(b"data: [DONE]\n\n")
            h = pod.calls[0]["headers"]
            assert h["secret"] == "podsecret"
            assert "Authorization" not in h and "X-API-Key" not in h and h["X-Omniserve-Tier"] == "paid"
            assert json.loads(pod.calls[0]["body"]) == BODY and pod.calls[0]["path"] == "/v1/chat/completions"
            assert sl.calls == [] and rig.pod.gauge.n == 0 and rig.router.total.n == 0
    asyncio.run(go())


def test_failover_pod_5xx_and_refused_to_serverless():
    async def go():
        for pod in (Fake(status(503)), "http://127.0.0.1:1"):
            sl = Fake(sse())
            async with Rig(pod, sl) as rig:
                async with rig.post("/v1/completions") as r:
                    assert r.status == 200 and r.headers["X-Overflow-Backend"] == "serverless:serverless"
                    assert (await r.read()).endswith(b"[DONE]\n\n")
                assert sl.calls[0]["headers"]["Authorization"] == "Bearer rk_test"
                assert sl.calls[0]["path"] == "/v1/completions"
                assert rig.pod.failovers == 1
    asyncio.run(go())


def test_serverless_no_workers_retried_until_worker_up():
    async def go():
        sl = Fake(status(400, b'{"error":"no workers available"}', then=sse(), times=2))
        async with Rig(None, sl) as rig:
            async with rig.post() as r:
                assert r.status == 200
                await r.read()
            assert len(sl.calls) == 3 and rig.router.serverless.failovers == 2
    asyncio.run(go())


def test_503_when_nothing_works():
    async def go():
        sl = Fake(status(502))
        async with Rig(Fake(status(500)), sl, OVERFLOW_DEADLINE_S=0.3) as rig:
            async with rig.post() as r:
                assert r.status == 503 and r.headers["Retry-After"] == "5"
                assert (await r.json())["error"]["type"] == "overflow_unavailable"
            assert len(sl.calls) >= 2 and rig.router.unavailable == 1
        async with Rig() as rig:
            async with rig.post() as r:
                assert r.status == 503
    asyncio.run(go())


def test_no_retry_after_bytes():
    async def go():
        sl = Fake(sse())
        async with Rig(Fake(abort_after_first), sl) as rig:
            got = b""
            try:
                async with rig.post() as r:
                    assert r.status == 200
                    async for chunk in r.content.iter_any():
                        got += chunk
            except aiohttp.ClientError:
                pass
            assert got == b'data: {"i":0}\n\n'
            assert sl.calls == [] and rig.pod.stream_errors == 1 and rig.pod.gauge.n == 0
    asyncio.run(go())


def test_non_retryable_4xx_passes_through():
    async def go():
        sl = Fake(sse())
        async with Rig(Fake(status(400, b'{"error":"bad max_tokens"}')), sl) as rig:
            async with rig.post() as r:
                assert r.status == 400 and await r.read() == b'{"error":"bad max_tokens"}'
            assert sl.calls == []
    asyncio.run(go())


def test_pod_capacity_accounting():
    async def go():
        hold = asyncio.Event()
        pod, sl = Fake(sse(hold=hold)), Fake(sse())
        async with Rig(pod, sl, capacity=2) as rig:
            async def one():
                async with rig.post() as r:
                    return r.headers["X-Overflow-Backend"], await r.read()
            tasks = [asyncio.ensure_future(one()) for _ in range(2)]
            for _ in range(100):
                if rig.pod.gauge.n == 2:
                    break
                await asyncio.sleep(0.01)
            assert rig.pod.gauge.n == 2 and rig.router.free_pods() == []
            third = await one()
            assert third[0] == "serverless:serverless"
            hold.set()
            res = await asyncio.gather(*tasks)
            assert [b for b, _ in res] == ["pod:omniserve-llm-overflow-aa"] * 2
            assert rig.pod.gauge.n == 0 and rig.router.total.n == 0 and len(pod.calls) == 2
    asyncio.run(go())


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


PRICES = [{"id": "NVIDIA GeForce RTX 4090", "price": {"secure": 0.89, "community": 0.34}},
          {"id": "NVIDIA GeForce RTX 5090", "price": {"secure": 1.19}},
          {"id": "NVIDIA L40S", "price": {"secure": 1.09}}]


class FakeAPI:
    def __init__(self, clock, pods=(), fail_gpus=()):
        self.clock, self.pods, self.fail_gpus = clock, list(pods), set(fail_gpus)
        self.created, self.terminated, self.n = [], [], 0

    async def list_pods(self):
        return [dict(p) for p in self.pods]

    async def gpu_types(self):
        return PRICES

    async def create_pod(self, body):
        if body["gpu"]["id"] in self.fail_gpus:
            raise lo.RunPodError(400, "no capacity")
        self.n += 1
        price = next(g["price"]["secure"] for g in PRICES if g["id"] == body["gpu"]["id"])
        d = {"id": f"pod{self.n}", "name": body["name"], "status": "PROVISIONING", "cost": price,
             "gpu": {"id": body["gpu"]["id"]}, "createdAt": self.clock()}
        self.created.append(body)
        self.pods.append(d)
        return d

    async def terminate_pod(self, pid):
        self.terminated.append(pid)
        self.pods = [p for p in self.pods if p["id"] != pid]


class ScaleRig:
    def __init__(self, pods=(), fail_gpus=(), **kw):
        self.kw, self.pods, self.fail_gpus = kw, pods, fail_gpus

    async def __aenter__(self):
        self.ping_status = 204

        async def ping(_):
            return web.Response(status=self.ping_status)
        app = web.Application()
        app.router.add_get("/{id}/ping", ping)
        self.runner, url = await start(app)
        self.clock = Clock()
        self.cfg = cfg(POD_IMAGE="lee101/omniserve-llm:test", POD_URL_TEMPLATE=url + "/{id}", **self.kw)
        self.session = aiohttp.ClientSession()
        self.router = lo.Router(self.cfg, self.session, clock=self.clock)
        self.api = FakeAPI(self.clock, self.pods, self.fail_gpus)
        self.scaler = lo.Scaler(self.cfg, self.router, self.api, clock=self.clock)
        return self

    async def run(self, seconds, load):
        for _ in range(int(seconds // 15)):
            left = load
            for p in self.router.pods.values():
                share = min(left, p.capacity) if p.ready and not p.draining else 0
                p.gauge.add(share - p.gauge.n)
                left -= share
            self.router.total.add(load - self.router.total.n)
            self.clock.t += 15
            await self.scaler.tick()

    async def __aexit__(self, *a):
        await self.session.close()
        await self.runner.cleanup()


def reasons(rig):
    return [d["msg"].split()[1:4] for d in rig.scaler.decisions]


def test_start_on_sustained_load_once():
    async def go():
        async with ScaleRig(fail_gpus={"NVIDIA GeForce RTX 4090"}) as rig:
            await rig.run(300, 12)
            assert rig.api.created == []
            await rig.run(600, 12)
            assert len(rig.api.created) == 1
            body = rig.api.created[0]
            assert body["name"].startswith("omniserve-llm-overflow-")
            assert body["gpu"] == {"id": "NVIDIA GeForce RTX 5090", "count": 1, "minCudaVersion": "12.9"}
            assert body["image"] == "lee101/omniserve-llm:test" and body["ports"] == ["8000/http"]
            assert body["env"]["HF_TOKEN"] == "hf_test" and body["env"]["PORT"] == "8000"
            assert body["env"]["OMNISERVE_NATIVE_SECRET"] == lo.pod_secret(rig.cfg, body["name"])
            p = rig.router.pods["pod1"]
            assert not p.ready and rig.router.free_pods() == []
            rig.ping_status = 200
            await rig.run(3600, 12)
            assert p.ready and len(rig.api.created) == 1 and rig.api.terminated == []
    asyncio.run(go())


def test_no_start_on_short_spike():
    async def go():
        async with ScaleRig() as rig:
            await rig.run(120, 20)
            await rig.run(1800, 0)
            assert rig.api.created == [] and rig.scaler.long.v < 1
    asyncio.run(go())


def test_terminate_on_idle_after_window():
    async def go():
        async with ScaleRig(POD_IDLE_S=1200) as rig:
            rig.ping_status = 200
            await rig.run(900, 12)
            assert rig.api.created and rig.router.pods["pod1"].ready
            await rig.run(900, 0)
            assert rig.api.terminated == []
            await rig.run(1500, 0)
            assert rig.api.terminated == ["pod1"] and rig.router.pods == {}
            assert ["stop", "pod=pod1", "reason=idle"] in reasons(rig)
    asyncio.run(go())


def test_terminate_on_ttl_under_load():
    async def go():
        async with ScaleRig(POD_TTL_S=3600) as rig:
            rig.ping_status = 200
            await rig.run(900, 12)
            created = rig.clock.t
            await rig.run(3000, 12)
            assert rig.api.terminated == []
            await rig.run(900, 12)
            assert "pod1" in rig.api.terminated and rig.clock.t - created > 3600
            assert ["stop", "pod=pod1", "reason=ttl"] in reasons(rig)
    asyncio.run(go())


def test_spend_cap_blocks_start_and_retires_adopted():
    async def go():
        async with ScaleRig(MAX_USD_HR=0.5) as rig:
            await rig.run(1800, 12)
            assert rig.api.created == [] and "no pod placed" in rig.scaler.blocked
        expensive = {"id": "old1", "name": "omniserve-llm-overflow-old", "status": "RUNNING", "cost": 3.0,
                     "gpu": {"id": "NVIDIA H100"}, "createdAt": "2026-10-21T00:00:00Z"}
        async with ScaleRig(pods=[expensive], MAX_USD_HR=2.0) as rig:
            await rig.run(15, 0)
            assert rig.api.terminated == ["old1"]
            assert ["stop", "pod=old1", "reason=spend_cap"] in reasons(rig)
    asyncio.run(go())


def test_adopts_existing_pods_and_ignores_foreign():
    async def go():
        clock_now = Clock().t
        ours = {"id": "mine1", "name": "omniserve-llm-overflow-beef", "status": "RUNNING", "cost": 0.89,
                "gpu": {"id": "NVIDIA GeForce RTX 4090"}, "createdAt": clock_now - 600}
        foreign = {"id": "other", "name": "someone-else", "status": "RUNNING", "cost": 5.0, "createdAt": clock_now}
        async with ScaleRig(pods=[ours, foreign]) as rig:
            rig.ping_status = 200
            await rig.run(15, 0)
            assert list(rig.router.pods) == ["mine1"]
            p = rig.router.pods["mine1"]
            assert p.ready and p.auth == {"secret": lo.pod_secret(rig.cfg, ours["name"])}
            await rig.run(1200, 12)
            assert rig.api.created == [] and rig.api.terminated == []
    asyncio.run(go())


def test_boot_timeout_terminates():
    async def go():
        async with ScaleRig(POD_BOOT_TIMEOUT_S=600) as rig:
            await rig.run(600, 12)
            assert rig.api.created
            await rig.run(700, 12)
            assert "pod1" in rig.api.terminated
            assert ["stop", "pod=pod1", "reason=boot_timeout"] in reasons(rig)
    asyncio.run(go())
