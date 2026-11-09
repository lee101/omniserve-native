"""Per-stage timings of the worker's cutout path (CUDA-synchronised, best of N).

    BIREFNET_VRAM_BROKER_URL= OMATTE_LIB=... OENCODE_LIB=... [BIREFNET_REFINER=...] \
        python3 matte/bench/stage_prof.py [--workers DIR] [--image gs2k.webp] [--sizes 1024,2048] [--n 7]

--workers points at another copy of workers/ so two revisions can be profiled
with the same script. Stages nest: the indented ones are part of the one above.
"""
import argparse
import base64
import io
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--workers", default=str(ROOT.parents[1] / "workers"))
parser.add_argument("--image", default=str(ROOT / "gs2k.webp"))
parser.add_argument("--sizes", default="1024,2048")
parser.add_argument("--formats", default="webp,png")
parser.add_argument("--n", type=int, default=7)
parser.add_argument("--composite", action="store_true")
args = parser.parse_args()
sys.argv = sys.argv[:1]
sys.path.insert(0, args.workers)

import torch
from PIL import Image

import birefnet_worker as w

STAGES = [
    ("read_image", 0), ("_prepare_input", 0), ("_forward", 0), ("_finish_mask", 0),
    ("image_to_device", 1), ("estimate_foreground_background", 0), ("composite_over", 0),
    ("to_pil_rgb", 0), ("device_to_image", 1), ("_pack_rgba", 0), ("encode_image", 0),
]
times = defaultdict(float)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wrap(name):
    real = getattr(w, name, None)
    if real is None:
        return

    def timed(*a, **k):
        sync()
        started = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            sync()
            times[name] += (time.perf_counter() - started) * 1e3

    setattr(w, name, timed)


def main():
    w.load_model()
    for name, _ in STAGES:
        wrap(name)
    print(f"engine={w.runtime.engine} native_encode={w.NATIVE_ENCODE} refiner={bool(w.matte_refine and w.matte_refine.refiner_loaded())}"
          f" webp_method={w.WEBP_METHOD}/{w.WEBP_METHOD_LARGE} torch={torch.__version__}")
    source = Image.open(args.image).convert("RGB")
    for size in map(int, args.sizes.split(",")):
        image = source if max(source.size) == size else source.resize((size, size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=1)
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        for fmt in args.formats.split(","):
            extra = {"background": "#141a38"} if args.composite else {}
            request = w.RemoveBackgroundRequest(image_url=url, output_format=fmt, cache=False, **extra)
            runs, totals = [], []
            for i in range(args.n + 2):
                times.clear()
                sync()
                started = time.perf_counter()
                w.remove_background(w.read_image(url), request)
                sync()
                if i >= 2:
                    totals.append((time.perf_counter() - started) * 1e3)
                    runs.append(dict(times))
            rest = [t - r.get("_forward", 0.0) for t, r in zip(totals, runs)]
            print(f"== {size} {fmt}{' +composite' if args.composite else ''}: e2e min {min(totals):.0f} med {statistics.median(totals):.0f} ms"
                  f" | without forward min {min(rest):.0f} med {statistics.median(rest):.0f} ms")
            accounted = 0.0
            for name, depth in STAGES:
                values = [r[name] for r in runs if name in r]
                if not values:
                    continue
                if depth == 0:
                    accounted += min(values)
                print(f"   {'  ' * depth}{name:32s} min {min(values):6.1f} med {statistics.median(values):6.1f}")
            print(f"   {'(other, from mins)':32s}     {min(totals) - accounted:6.1f}")


main()
