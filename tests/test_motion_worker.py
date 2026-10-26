import asyncio
import base64
import socket
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

import motion_worker as mw


def _resolve(monkeypatch, mapping):
    def fake(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        family = socket.AF_INET6 if ":" in mapping[host] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (mapping[host], port))]

    monkeypatch.setattr(mw.socket, "getaddrinfo", fake)


@pytest.fixture(autouse=True)
def strict_urls(monkeypatch):
    monkeypatch.setattr(mw, "ALLOW_PRIVATE_URLS", False)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/v.mp4", "http:///v.mp4", "http://user:pw@example.com/v.mp4", "not a url",
])
def test_rejects_bad_scheme_and_credentials(url):
    with pytest.raises(mw.MotionError) as exc:
        mw.assert_public_url(url)
    assert exc.value.status == 422


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.5.5", "169.254.169.254", "::1", "fd00::1", "0.0.0.0"])
def test_rejects_hosts_resolving_to_non_public_addresses(monkeypatch, ip):
    _resolve(monkeypatch, {"evil.example.com": ip})
    with pytest.raises(mw.MotionError) as exc:
        mw.assert_public_url("https://evil.example.com/v.mp4")
    assert exc.value.status == 422 and "public" in exc.value.message


def test_rejects_when_any_resolved_address_is_private(monkeypatch):
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", port))]

    monkeypatch.setattr(mw.socket, "getaddrinfo", fake)
    with pytest.raises(mw.MotionError):
        mw.assert_public_url("https://rebind.example.com/v.mp4")


def test_rejects_unresolvable_host(monkeypatch):
    _resolve(monkeypatch, {})
    with pytest.raises(mw.MotionError) as exc:
        mw.assert_public_url("https://nope.example.com/v.mp4")
    assert exc.value.status == 422


def test_accepts_public_host(monkeypatch):
    _resolve(monkeypatch, {"videos.example.com": "93.184.216.34", "v6.example.com": "2606:2800:220:1:248:1893:25c8:1946"})
    mw.assert_public_url("https://videos.example.com/v.mp4")
    mw.assert_public_url("http://v6.example.com:8080/v.mp4")


def test_private_urls_allowed_only_by_env(monkeypatch):
    monkeypatch.setattr(mw, "ALLOW_PRIVATE_URLS", True)
    mw.assert_public_url("http://127.0.0.1:8799/v.mp4")
    with pytest.raises(mw.MotionError):
        mw.assert_public_url("file:///etc/passwd")


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=(b"video",)):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self.closed = False

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise mw.requests.HTTPError(f"{self.status_code}")

    def iter_content(self, _size):
        yield from self._chunks

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_fetch_video_downloads_and_writes(monkeypatch, tmp_path):
    _resolve(monkeypatch, {"videos.example.com": "93.184.216.34"})
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("allow_redirects")))
        return FakeResponse(chunks=(b"abc", b"def"))

    monkeypatch.setattr(mw.requests, "get", fake_get)
    dest = tmp_path / "in.mp4"
    mw.fetch_video("https://videos.example.com/v.mp4", None, dest)
    assert dest.read_bytes() == b"abcdef" and calls == [("https://videos.example.com/v.mp4", False)]


def test_fetch_video_blocks_redirect_to_metadata_service(monkeypatch, tmp_path):
    _resolve(monkeypatch, {"videos.example.com": "93.184.216.34", "169.254.169.254": "169.254.169.254"})
    fetched = []

    def fake_get(url, **kwargs):
        fetched.append(url)
        return FakeResponse(302, {"location": "http://169.254.169.254/latest/meta-data/"})

    monkeypatch.setattr(mw.requests, "get", fake_get)
    with pytest.raises(mw.MotionError) as exc:
        mw.fetch_video("https://videos.example.com/v.mp4", None, tmp_path / "in.mp4")
    assert exc.value.status == 422 and fetched == ["https://videos.example.com/v.mp4"]


def test_fetch_video_follows_public_redirect_and_limits_hops(monkeypatch, tmp_path):
    _resolve(monkeypatch, {"a.example.com": "93.184.216.34", "b.example.com": "93.184.216.35"})

    def one_hop(url, **kwargs):
        if url.startswith("https://a."):
            return FakeResponse(301, {"location": "https://b.example.com/real.mp4"})
        return FakeResponse(chunks=(b"ok",))

    monkeypatch.setattr(mw.requests, "get", one_hop)
    dest = tmp_path / "in.mp4"
    mw.fetch_video("https://a.example.com/v.mp4", None, dest)
    assert dest.read_bytes() == b"ok"

    monkeypatch.setattr(mw.requests, "get", lambda url, **kw: FakeResponse(302, {"location": "/again"}))
    with pytest.raises(mw.MotionError) as exc:
        mw.fetch_video("https://a.example.com/v.mp4", None, tmp_path / "loop.mp4")
    assert exc.value.status == 422 and "redirected" in exc.value.message


def test_fetch_video_size_limit_and_http_errors(monkeypatch, tmp_path):
    _resolve(monkeypatch, {"videos.example.com": "93.184.216.34"})
    monkeypatch.setattr(mw, "MAX_DOWNLOAD_BYTES", 5)
    monkeypatch.setattr(mw.requests, "get", lambda url, **kw: FakeResponse(chunks=(b"abc", b"def")))
    with pytest.raises(mw.MotionError) as exc:
        mw.fetch_video("https://videos.example.com/v.mp4", None, tmp_path / "big.mp4")
    assert exc.value.status == 413

    monkeypatch.setattr(mw.requests, "get", lambda url, **kw: FakeResponse(404))
    with pytest.raises(mw.MotionError) as exc:
        mw.fetch_video("https://videos.example.com/v.mp4", None, tmp_path / "missing.mp4")
    assert exc.value.status == 502


def test_fetch_video_base64_and_missing_input(tmp_path):
    dest = tmp_path / "in.mp4"
    mw.fetch_video(None, base64.b64encode(b"hello").decode(), dest)
    assert dest.read_bytes() == b"hello"
    with pytest.raises(mw.MotionError) as exc:
        mw.fetch_video(None, "***not base64***", tmp_path / "bad.mp4")
    assert exc.value.status == 422
    with pytest.raises(mw.MotionError) as exc:
        mw.fetch_video(None, None, tmp_path / "none.mp4")
    assert exc.value.status == 422


class StubEngine:
    def __init__(self):
        self.loaded = True
        self.ready = True
        self.jobs = 0
        self.load_seconds = 1.5
        self.warm_seconds = 4.25
        self.calls = []
        self.error = None

    def run(self, video, static_cam=True, max_seconds=0.0):
        self.calls.append((Path(video).read_bytes(), static_cam, max_seconds))
        if self.error:
            raise self.error
        t = 30
        joints = np.zeros((t, 77, 3))
        idx = {n: i for i, n in enumerate(mw.motion_format.SOMA_NAMES)}
        for name, p in {"Hips": (0, 0.95, 0), "Chest": (0, 1.25, 0), "LeftShoulder": (0.18, 1.4, 0),
                        "RightShoulder": (-0.18, 1.4, 0), "LeftToeBase": (0.1, 0.04, 0.1), "RightToeBase": (-0.1, 0.04, 0.1),
                        "LeftFoot": (0.1, 0.1, 0), "RightFoot": (-0.1, 0.1, 0)}.items():
            joints[:, idx[name]] = p
        probs = np.zeros((t, 6))
        probs[:, :4] = 0.9
        params = {"body_pose": np.zeros((t, 228)), "global_orient": np.zeros((t, 3)), "transl": np.zeros((t, 3)),
                  "identity_coeffs": np.zeros((t, 45)), "scale_params": np.zeros((t, 69))}
        return {"fps_in": 24.0, "frames": t, "joints77": joints, "params": params, "contact_probs": probs,
                "timings": {"detect": 0.1}, "gpu_seconds": 0.5}


class AsgiClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def call():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(call())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr(mw, "engine", stub)
    monkeypatch.setattr(mw, "ALLOW_PRIVATE_URLS", True)
    return AsgiClient(mw.create_app()), stub


def test_health_reports_loading_state(client):
    c, stub = client
    health = c.get("/health").json()
    assert health["status"] == "ok" and health["warmup_seconds"] == 4.25
    stub.ready = False
    assert c.get("/health").json()["status"] == "loading"


def test_motion_endpoint_returns_library_format(client):
    c, stub = client
    body = {"video_base64": base64.b64encode(b"vid").decode(), "name": "Backflip", "prompt": "a flip",
            "max_seconds": 10, "include_soma": True}
    resp = c.post("/v1/motion/video", json=body)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert stub.calls == [(b"vid", True, 10.0)]
    assert out["backend"] == "gem-x" and out["gpu_seconds"] == 0.5 and out["frames_in"] == 30
    motion = out["motion"]
    assert motion["format"] == "ardy-skeleton-motion-1" and motion["id"] == "Backflip" and motion["prompt"] == "a flip"
    assert motion["fps"] == 20.0 and len(motion["positions"]) == motion["frames"] and len(motion["positions"][0]) == 27
    assert out["soma"]["format"] == "soma-motion-1" and out["soma"]["frames"] == 30

    resp = c.post("/v1/motion/video", json={"video_base64": base64.b64encode(b"vid").decode()})
    assert "soma" not in resp.json()


@pytest.mark.parametrize("body,status", [
    ({}, 422),
    ({"video_base64": "dmlk", "backend": "nope"}, 422),
    ({"video_base64": "dmlk", "fps_out": 1}, 422),
    ({"video_base64": "dmlk", "max_seconds": 999}, 422),
])
def test_motion_endpoint_validation(client, body, status):
    c, stub = client
    assert c.post("/v1/motion/video", json=body).status_code == status
    assert stub.calls == []


@pytest.mark.parametrize("error,status", [
    (mw.MotionError(413, "video is 12.0s; max_seconds is 10"), 413),
    (mw.MotionError(422, "video has fewer than 2 frames"), 422),
    (mw.MotionError(503, "motion model is still loading"), 503),
])
def test_motion_errors_pass_status_and_message(client, error, status):
    c, stub = client
    stub.error = error
    resp = c.post("/v1/motion/video", json={"video_base64": "dmlk"})
    assert resp.status_code == status and resp.json() == {"error": error.message}


def test_json_body_accepted_whatever_the_content_type(client):
    c, stub = client
    payload = '{"video_base64": "%s", "max_seconds": 3}' % base64.b64encode(b"vid").decode()
    for content_type in ("application/x-www-form-urlencoded", "text/plain", "application/json"):
        resp = c.post("/v1/motion/video", content=payload, headers={"Content-Type": content_type})
        assert resp.status_code == 200, (content_type, resp.text)
    resp = c.post("/v1/motion/video", content=payload)
    assert resp.status_code == 200 and [call[2] for call in stub.calls] == [3.0] * 4


@pytest.mark.parametrize("body", [b"", b"not json", b"[1, 2]", b'"text"', b'{"video_base64": '])
def test_malformed_body_is_a_clear_422(client, body):
    c, stub = client
    resp = c.post("/v1/motion/video", content=body)
    assert resp.status_code == 422 and resp.json()["error"] == "invalid request" and resp.json()["detail"]
    assert stub.calls == []


def test_field_errors_name_the_field(client):
    c, _ = client
    resp = c.post("/v1/motion/video", content=b'{"video_base64": "dmlk", "max_seconds": 999}')
    assert resp.status_code == 422 and "max_seconds" in resp.json()["detail"]


def test_engine_refuses_work_until_warm():
    engine = mw.GemxEngine()
    with pytest.raises(mw.MotionError) as exc:
        engine.run(Path("missing.mp4"))
    assert exc.value.status == 503
    with pytest.raises(mw.MotionError) as exc:
        engine.run(Path("missing.mp4"), warmup=True)
    assert exc.value.status == 503
    engine.loaded = True
    with pytest.raises(mw.MotionError) as exc:
        engine.run(Path("missing.mp4"))
    assert exc.value.status == 503
