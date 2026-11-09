"""Output identity and interleaved timing of two revisions of the worker in one process.

    BIREFNET_VRAM_BROKER_URL= OMATTE_LIB=... OENCODE_LIB=... [BIREFNET_REFINER=...] \
        python3 matte/bench/ab_identity.py BASE_WORKER.py [image ...] [--n 7] [--sizes 1024,2048]

BASE_WORKER.py is another revision's birefnet_worker.py (e.g. `git show REV:workers/birefnet_worker.py`);
both use this checkout's helper modules. PNG outputs must decode to identical pixels; WebP
outputs must be byte-identical unless BIREFNET_WEBP_ALPHA_METHOD is set, then pixel-identical.
Exit status 1 on any difference.
"""
import argparse
import importlib.util
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("base")
parser.add_argument("images", nargs="*")
parser.add_argument("--n", type=int, default=7)
parser.add_argument("--sizes", default="1024,2048")
args = parser.parse_args()
sys.argv = sys.argv[:1]
sys.path.insert(0, str(ROOT.parents[1] / "workers"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_model()
    return module


def decoded(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


def timed(module, image, request):
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = module.remove_background(image, request)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e3, out


def main():
    base = load("birefnet_worker_base", args.base)
    new = load("birefnet_worker_new", ROOT.parents[1] / "workers" / "birefnet_worker.py")
    print(f"engine={new.runtime.engine} refiner={bool(new.matte_refine and new.matte_refine.refiner_loaded())} perf={new.health()['perf']}")
    failures = 0
    variants = {"cutout": {}, "nodecon": {"decontaminate": False}, "composite": {"background": "#141a38"},
                "background": {"return_background": True}}
    for path in args.images or [str(ROOT / "gs2k.webp")]:
        source = Image.open(path).convert("RGB")
        for size in map(int, args.sizes.split(",")):
            scale = size / max(source.size)
            image = source if scale == 1 else source.resize(
                (round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
            for fmt in ("webp", "png"):
                for name, extra in variants.items():
                    request = new.RemoveBackgroundRequest(image_url="x", output_format=fmt, cache=False, **extra)
                    times = {"base": [], "new": []}
                    for i in range(args.n + 1):
                        for tag, module in (("base", base), ("new", new)) if i % 2 else (("new", new), ("base", base)):
                            ms, out = timed(module, image, request)
                            if i:
                                times[tag].append(ms)
                            if tag == "base":
                                ref = out
                            else:
                                got = out
                    verdict = []
                    for key in ref:
                        if fmt == "webp" and ref[key] == got[key]:
                            verdict.append(f"{key}:bytes")
                        elif np.array_equal(decoded(ref[key]), decoded(got[key])):
                            verdict.append(f"{key}:pixels({len(got[key]) / len(ref[key]) - 1:+.1%})")
                        else:
                            delta = np.abs(decoded(ref[key]).astype(int) - decoded(got[key]).astype(int))
                            verdict.append(f"{key}:DIFF(max {delta.max()}, mean {delta.mean():.4f})")
                            failures += 1
                    b, n = times["base"], times["new"]
                    print(f"{Path(path).name:18s} {image.size[0]}x{image.size[1]} {fmt:4s} {name:10s} base min {min(b):5.0f} med {statistics.median(b):5.0f}"
                          f" | new min {min(n):5.0f} med {statistics.median(n):5.0f} ms | {' '.join(verdict)}", flush=True)
    print("FAIL" if failures else "identical")
    raise SystemExit(1 if failures else 0)


main()
