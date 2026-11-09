"""Worker end-to-end with and without the native encoder.

For each input and format: run remove_background with Pillow encoding, record the
exact pre-encode images, re-encode those with liboencode, and compare decoded
pixels (png must be bit-exact, webp is compared against the Pillow decode).
Then time both paths end to end.

    OENCODE_LIB=build/liboencode.so OMATTE_LIB=.../libomatte.so python3 matte/bench/encode_eval.py [image ...]
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "workers"))
import birefnet_worker as w

DEFAULTS = ["gs2k.png", "out/flowers.png", "out/inner.png", "out/nonkey.png", "out/real_A_navy.png", "out/real_C_navy.png"]


def decoded(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


def run(img, req, native, record=None, n=3):
    w.NATIVE_ENCODE = native
    real = w.encode_image
    if record is not None:
        def tap(rgba, fmt):
            record.append((Image.fromarray(rgba) if isinstance(rgba, np.ndarray) else rgba.copy(), fmt))
            return real(rgba, fmt)
        w.encode_image = tap
    try:
        w.remove_background(img, req)
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = w.remove_background(img, req)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    finally:
        w.encode_image = real
    return min(times) * 1e3, out


def main():
    assert w.oencode is not None and w.oencode.available(), "liboencode not loaded"
    w.load_model()
    base = Path(os.getenv("BENCH_DIR", ROOT))
    worst = 0
    for name in sys.argv[1:] or DEFAULTS:
        img = Image.open(base / name).convert("RGB")
        for fmt in ("png", "webp"):
            req = w.RemoveBackgroundRequest(image_url="x", output_format=fmt, background="#141a38", cache=False)
            seen = []
            pil_ms, pil_out = run(img, req, False, seen)
            nat_ms, nat_out = run(img, req, True)
            exact = True
            for rgba, f in seen[-len(pil_out):]:
                w.NATIVE_ENCODE = False
                ref = w.encode_image(rgba, f)[0]
                w.NATIVE_ENCODE = True
                got = w.encode_image(rgba, f)[0]
                a, b = decoded(ref), decoded(got)
                diff = int(np.abs(a.astype(int) - b.astype(int)).max())
                worst = max(worst, diff)
                if f == "png":
                    exact &= bool(np.array_equal(decoded(got), np.asarray(rgba.convert("RGBA")))) and diff == 0
                else:
                    exact &= diff == 0
            e2e = max(float(np.abs(decoded(pil_out[k]).astype(int) - decoded(nat_out[k]).astype(int)).max()) for k in pil_out)
            print(f"{name:22s} {img.size} {fmt:4s} pil {pil_ms:6.0f} ms  native {nat_ms:6.0f} ms  x{pil_ms / nat_ms:4.2f}"
                  f"  encoder_exact={exact} e2e_maxdiff={e2e:.0f} bytes {sum(map(len, pil_out.values()))}->{sum(map(len, nat_out.values()))}")
    print("worst encoder pixel diff:", worst)


if __name__ == "__main__":
    main()
