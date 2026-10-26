#!/usr/bin/env python3
"""Lossless-acceleration gate for the shared Anima sampler.

Runs fixed-seed requests through workloads.anime_control_core.sample and reports
per-stage timings plus a SHA-256 of the raw output pixels. `--ref DIR` compares
against a saved reference run (hash equality, else max/mean abs pixel error).

  python tools/anime_accel_bench.py --out ref/            # record reference
  python tools/anime_accel_bench.py --ref ref/ --out cand/ # gate a candidate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = [
    {"prompt": "1girl, solo, silver hair, red eyes, school uniform, cherry blossoms", "seed": 7},
    {"prompt": "1boy, solo, armor, sword, castle, dramatic lighting", "negative_prompt": "lowres, bad anatomy", "seed": 11},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="768x768")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ref", type=Path)
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.split("x"))

    import torch

    from workloads import anime_control_core as core

    pipe = core.finalize_pipeline("bench", core.build_pipeline("bench"))
    pipe.compiled_sizes.add((width, height))
    result = {"size": args.size, "steps": args.steps, "cases": []}
    first = time.perf_counter()
    for index, case in enumerate(CASES):
        values = {"width": width, "height": height, "num_inference_steps": args.steps, **case}
        common = core.common_values(values)
        runs, stable = [], True
        for run in range(args.repeat):
            marks = core.Marks()
            with torch.inference_mode():
                image = core.sample(pipe, common, marks=marks)
            runs.append(dict(marks.values))
            pixels = np.asarray(image.convert("RGB"))
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if run == 0:
                first_pixels, first_digest = pixels, digest
            elif digest != first_digest:
                stable = False
            if index == 0 and run == 0:
                result["first_request_s"] = round(time.perf_counter() - first, 2)
        warm = runs[1:] or runs
        entry = {
            "seed": case["seed"],
            "sha256": first_digest,
            "deterministic": stable,
            "denoise_s": round(min(r["denoise"] for r in warm), 3),
            "decode_s": round(min(r["decode"] for r in warm), 3),
            "text_s": round(min(r["text"] for r in warm), 3),
        }
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            np.save(args.out / f"case{index}.npy", first_pixels)
        if args.ref:
            ref = np.load(args.ref / f"case{index}.npy")
            diff = np.abs(ref.astype(np.int16) - first_pixels.astype(np.int16))
            entry["identical"] = bool(diff.max() == 0)
            entry["max_abs_err"] = int(diff.max())
            entry["mean_abs_err"] = round(float(diff.mean()), 5)
        result["cases"].append(entry)
    result["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(result, indent=1), flush=True)
    if args.out:
        (args.out / "result.json").write_text(json.dumps(result, indent=1))
    return 0 if all(c.get("identical", True) for c in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
