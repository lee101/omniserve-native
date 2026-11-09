"""HTTP probe for a running BiRefNet worker (stdlib + Pillow only).

    python3 -I matte/bench/http_probe.py [--url http://127.0.0.1:9094] [--image gs2k.webp] [--n 7]
                                         [--sizes 1024,2048] [--conc 1,2,4] [--save DIR]

Per size and format: cutout only (binary response, X-Compute-Ms is the server
time) and cutout + composite on a flat colour (JSON with base64). Then
requests/s at each concurrency for the cutout request.
"""
import argparse
import base64
import io
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def data_url(image, size):
    if max(image.size) != size:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def post(url, body):
    started = time.perf_counter()
    request = urllib.request.Request(url + "/v1/images/background-removals", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        server = response.headers.get("X-Compute-Ms")
    return (time.perf_counter() - started) * 1e3, float(server) if server else None, raw


def stats(values):
    return f"min {min(values):6.0f} med {statistics.median(values):6.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9094")
    parser.add_argument("--image", default=os.path.join(HERE, "gs2k.webp"))
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--sizes", default="1024,2048")
    parser.add_argument("--conc", default="1,2,4")
    parser.add_argument("--formats", default="webp,png")
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    source = Image.open(args.image).convert("RGB")
    print(json.dumps({k: v for k, v in json.load(urllib.request.urlopen(args.url + "/health")).items()
                      if k in ("engine", "input_size", "matte_refiner", "webp_method", "native_encode", "perf")}))
    for size in map(int, args.sizes.split(",")):
        src = data_url(source, size)
        for fmt in args.formats.split(","):
            variants = {
                "cutout": {},
                "cutout-nodecon": {"decontaminate": False},
                "cutout+composite": {"background": "#141a38"},
            }
            for name, extra in variants.items():
                body = json.dumps({"image_url": src, "output_format": fmt, "cache": False, **extra}).encode()
                post(args.url, body)
                runs = [post(args.url, body) for _ in range(args.n)]
                wall = [r[0] for r in runs]
                server = [r[1] for r in runs if r[1] is not None]
                line = f"{size} {fmt:4s} {name:17s} wall {stats(wall)} ms"
                if server:
                    line += f" | server {stats(server)} ms"
                print(line + f" | resp {len(runs[-1][2]) / 1e6:.2f} MB", flush=True)
                if args.save:
                    os.makedirs(args.save, exist_ok=True)
                    with open(os.path.join(args.save, f"{size}_{fmt}_{name}.out"), "wb") as handle:
                        handle.write(runs[-1][2])
        body = json.dumps({"image_url": src, "output_format": "webp", "cache": False}).encode()
        for conc in map(int, args.conc.split(",")):
            best = []
            with ThreadPoolExecutor(conc) as pool:
                for _ in range(max(3, args.n // 2)):
                    count = conc * 4
                    started = time.perf_counter()
                    list(pool.map(lambda _: post(args.url, body), range(count)))
                    best.append(count / (time.perf_counter() - started))
            print(f"{size} webp cutout conc {conc}: max {max(best):5.2f} med {statistics.median(best):5.2f} req/s", flush=True)


if __name__ == "__main__":
    main()
