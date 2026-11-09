#!/usr/bin/env python3
"""POST /v1/images/upscales contract. Without weights: auth, validation and the
503 for unconfigured weights (no GPU needed). With OMNISERVE_TEST_ESRGAN_DIR
holding RealESRGAN_x4plus.pth (+ optional _anime_6B / x2plus) and a GPU: real
2x/4x upscales, alpha, jpeg/webp, oversize 413."""
import base64, json, os, socket, struct, subprocess, sys, time, urllib.error, urllib.request, zlib

SECRET = "upscale-test-secret"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def png(w, h, alpha=False):
    ch = 4 if alpha else 3
    rows = b"".join(b"\x00" + bytes(((x * 7 + y * 3 + c * 50) & 255) if c < 3 else (255 if x < w // 2 else 0)
                                    for x in range(w) for c in range(ch)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def data_url(b):
    return "data:image/png;base64," + base64.b64encode(b).decode()


def post(port, body, secret=SECRET):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/upscales", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "Authorization": f"Bearer {secret}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def start(binary, extra):
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update({"OMNISERVE_NATIVE_BIND": "127.0.0.1", "OMNISERVE_NATIVE_SECRET": SECRET,
                "OMNISERVE_NATIVE_GUARD_JUDGE": "0", **extra})
    proc = subprocess.Popen([binary, "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(150):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return proc, port
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("FAIL: gateway exited during startup")
            time.sleep(0.2)
    raise SystemExit("FAIL: gateway never ready")


def expect(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def png_dims(b):
    expect(b[:8] == b"\x89PNG\r\n\x1a\n", "png magic")
    w, h, depth, ctype = struct.unpack(">IIBB", b[16:26])
    return w, h, ctype


def main():
    binary = os.environ.get("OMNISERVE_NATIVE_BIN")
    if not binary or not os.path.exists(binary):
        print("skip: OMNISERVE_NATIVE_BIN not set")
        return 0
    small = data_url(png(64, 48))
    proc, port = start(binary, {})
    try:
        code, body, _ = post(port, {"image_url": small}, secret="wrong")
        expect(code == 401, f"bad secret -> 401, got {code}")
        for bad in ({}, {"image_url": "http://x.io/a.png"}, {"image_url": small, "scale": 3},
                    {"image_url": small, "model": "photo"}, {"image_url": small, "output_format": "gif"}):
            code, body, _ = post(port, bad)
            expect(code == 400 and b"error" in body, f"{bad} -> 400, got {code} {body[:80]}")
        code, body, _ = post(port, {"image_url": small})
        expect(code == 503 and b"not configured" in body, f"no weights -> 503, got {code} {body[:80]}")
    finally:
        proc.terminate()
        proc.wait(10)

    wdir = os.environ.get("OMNISERVE_TEST_ESRGAN_DIR")
    if not wdir or not os.path.exists(os.path.join(wdir, "RealESRGAN_x4plus.pth")):
        print("ok (gpu cases skipped: OMNISERVE_TEST_ESRGAN_DIR not set)")
        return 0
    extra = {"OMNISERVE_NATIVE_ESRGAN_GENERAL": os.path.join(wdir, "RealESRGAN_x4plus.pth")}
    for env, name in (("OMNISERVE_NATIVE_ESRGAN_ANIME", "RealESRGAN_x4plus_anime_6B.pth"),
                      ("OMNISERVE_NATIVE_ESRGAN_GENERAL_X2", "RealESRGAN_x2plus.pth")):
        if os.path.exists(os.path.join(wdir, name)):
            extra[env] = os.path.join(wdir, name)
    if os.environ.get("OMNISERVE_NATIVE_SD_LIB"):
        extra["OMNISERVE_NATIVE_SD_LIB"] = os.environ["OMNISERVE_NATIVE_SD_LIB"]
    proc, port = start(binary, extra)
    try:
        for scale in (2, 4):
            code, body, hdr = post(port, {"image_url": small, "scale": scale})
            expect(code == 200 and hdr["Content-Type"] == "image/png", f"scale {scale} -> 200 png, got {code} {body[:120]}")
            expect(png_dims(body)[:2] == (64 * scale, 48 * scale), f"scale {scale} dims {png_dims(body)}")
            expect(hdr["X-Upscale-Width"] == str(64 * scale) and hdr["X-Upscale-Height"] == str(48 * scale)
                   and hdr["X-Upscale-Scale"] == str(scale) and hdr["X-Upscale-Model"] == "general"
                   and int(hdr["X-Upscale-Ms"]) >= 0, f"headers {dict(hdr)}")
        code, body, hdr = post(port, {"image_url": data_url(png(33, 21, alpha=True)), "scale": 2})
        expect(code == 200 and png_dims(body) == (66, 42, 6), f"alpha png keeps RGBA at odd size, got {code} {png_dims(body) if code == 200 else body[:80]}")
        code, body, hdr = post(port, {"image_url": small, "output_format": "jpeg"})
        expect(code == 200 and hdr["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8", f"jpeg {code}")
        code, body, hdr = post(port, {"image_url": small, "output_format": "webp"})
        expect(code == 200 and hdr["Content-Type"] == "image/webp" and body[8:12] == b"WEBP", f"webp {code}")
        if "OMNISERVE_NATIVE_ESRGAN_ANIME" in extra:
            code, body, hdr = post(port, {"image_url": small, "model": "anime"})
            expect(code == 200 and hdr["X-Upscale-Model"] == "anime", f"anime {code}")
        code, body, _ = post(port, {"image_url": data_url(png(1500, 1500)), "scale": 4})
        expect(code == 413, f"oversize scale 4 -> 413, got {code}")
        code, body, _ = post(port, {"image_url": data_url(png(4200, 16)), "scale": 2})
        expect(code == 413, f"output side > 8192 -> 413, got {code}")
        code, body, _ = post(port, {"image_url": "https://127.0.0.1/a.png"})
        expect(code == 400 and b"non-public" in body, f"ssrf -> 400, got {code} {body[:80]}")
    finally:
        proc.terminate()
        proc.wait(10)
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
