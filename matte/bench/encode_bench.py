"""Native vs Pillow encode: speed, size and decoded-pixel parity.

    OENCODE_LIB=build/liboencode.so python3 matte/bench/encode_bench.py [image ...]
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import oencode

DEFAULTS = ["matte/bench/out/e2e_base_cutout.png", "matte/bench/out/e2e_base_composite.png", "matte/bench/gs2k.png"]


def best(fn, n=5):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        times.append(time.perf_counter() - t)
    return min(times) * 1e3, r


def blacked(arr):
    out = arr.copy()
    out[out[:, :, 3] == 0, :3] = 0
    return out


def main():
    paths = sys.argv[1:] or DEFAULTS
    for path in paths:
        im = Image.open(path).convert("RGBA")
        arr = np.asarray(im)
        print(f"== {path} {im.size}")
        pil_ms, pil = best(lambda: (lambda b: (im.save(b, format="PNG", compress_level=3), b.getvalue())[1])(io.BytesIO()))
        print(f"png pil L3        {pil_ms:7.1f} ms {len(pil):>9} B")
        for level in (1, 2, 3, 6):
            ms, data = best(lambda: oencode.png(arr, level))
            ok = np.array_equal(np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")), arr)
            print(f"png native L{level:<2}     {ms:7.1f} ms {len(data):>9} B  lossless={ok}")
        for method in (2, 4):
            pil_ms, pw = best(lambda: (lambda b: (Image.fromarray(blacked(arr), "RGBA").save(
                b, format="WEBP", quality=85, method=method, exact=False), b.getvalue())[1])(io.BytesIO()), 3)
            a = np.asarray(Image.open(io.BytesIO(pw)).convert("RGBA"))
            for af in (1, 0):
                ms, nw = best(lambda: oencode.webp(arr, 85, method, af), 3)
                b = np.asarray(Image.open(io.BytesIO(nw)).convert("RGBA"))
                print(f"webp m{method} pil {pil_ms:7.1f} ms {len(pw):>9} B | native af{af} {ms:7.1f} ms {len(nw):>9} B"
                      f"  bytes_equal={pw == nw} pixels_equal={np.array_equal(a, b)}")


if __name__ == "__main__":
    main()
