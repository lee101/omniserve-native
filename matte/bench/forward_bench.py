"""BiRefNet forward only: eager vs torch.compile modes and batch sizes, interleaved rounds.

    BIREFNET_VRAM_BROKER_URL= python3 matte/bench/forward_bench.py [--modes default,max-autotune-no-cudagraphs]
                                                                  [--batches 1,2,4] [--rounds 7] [--size 1024]

Prints min/median ms per forward and per image, and the mask delta of each variant against eager.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--modes", default="default")
parser.add_argument("--batches", default="1,2,4")
parser.add_argument("--rounds", type=int, default=7)
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--image", default=str(ROOT / "gs2k.webp"))
args = parser.parse_args()
sys.argv = sys.argv[:1]
sys.path.insert(0, str(ROOT.parents[1] / "workers"))

import torch
from PIL import Image

import birefnet_worker as w


def main():
    w.TORCH_COMPILE = False
    w.load_model()
    eager = w.runtime.eager_model
    image = Image.open(args.image).convert("RGB")
    variants = {"eager": eager}
    for mode in filter(None, args.modes.split(",")):
        started = time.perf_counter()
        try:
            compiled = torch.compile(eager, mode=mode, fullgraph=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=w.runtime.dtype):
                for _ in range(3):
                    compiled(w._prepare_input(image, args.size))
            variants[f"compile-{mode}"] = compiled
            print(f"compile-{mode}: ready in {time.perf_counter() - started:.0f} s", flush=True)
        except Exception as error:
            print(f"compile-{mode}: failed ({type(error).__name__}: {str(error)[:200]})", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=w.runtime.dtype):
        one = w._prepare_input(image, args.size)
        reference = torch.sigmoid(w._unwrap_prediction(eager(one))).float()
        for batch in map(int, args.batches.split(",")):
            tensor = one.expand(batch, -1, -1, -1).contiguous(memory_format=torch.channels_last)
            times = {name: [] for name in variants}
            deltas = {}
            for name, model in variants.items():
                if batch > 1 and name != "eager":
                    continue
                out = torch.sigmoid(w._unwrap_prediction(model(tensor))).float()
                deltas[name] = float((out[:1] - reference).abs().max())
            for _ in range(args.rounds):
                for name, model in variants.items():
                    if name not in deltas:
                        continue
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    model(tensor)
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - started) * 1e3)
            for name, values in times.items():
                if values:
                    print(f"batch {batch} {name:36s} min {min(values):6.1f} med {statistics.median(values):6.1f} ms"
                          f" | per image min {min(values) / batch:6.1f} | max mask delta vs eager b1 {deltas[name]:.5f}", flush=True)
    print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


main()
