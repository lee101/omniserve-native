#!/usr/bin/env python3
"""Gate exact edit-result caching with uncached renders and key mutations."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image

from image_parity_bench import decode_image_response


def encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8793")
    parser.add_argument("--route", default="/v1/images/img2img")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90908)
    parser.add_argument("--prompt", default="Change the robot material to gold. Keep its shape and background unchanged.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB").resize((512, 512))
    changed = source.copy()
    changed.putpixel((0, 0), (255, 0, 0))
    base = {"prompt": args.prompt,
            "image_base64": encode(source), "size": "512x512", "strength": 0.6,
            "steps": args.steps, "seed": args.seed, "guidance_scale": args.guidance}
    variants = [({}, "original"), ({"seed": args.seed + 1}, "seed"),
                ({"image_base64": encode(changed)}, "source"),
                ({"prompt": "Make the robot red. Keep its shape and background unchanged."}, "prompt"),
                ({"strength": 0.8}, "strength"), ({"steps": args.steps + 1}, "steps"),
                ({"size": "512x576"}, "size"), ({"guidance_scale": 1.5}, "guidance")]
    rows, failures = [], []
    for variant, name in variants[:args.limit]:
        samples = []
        for repeat, cache in enumerate((False, True, True, True, True)):
            payload = {**base, **variant, "cache": cache}
            req = urllib.request.Request(args.base + args.route, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "X-Omniserve-Tier": "background"})
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=900) as response:
                image, meta = decode_image_response(response.read(),
                    response.headers.get("Content-Type", ""), 900)
            meta["wall_ms"] = (time.perf_counter() - start) * 1000
            meta["sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
            samples.append(meta)
            if repeat == 0:
                image.save(args.output / f"{name}.png")
            if cache and bool((meta.get("cache") or {}).get("hit")) != (repeat >= 2):
                failures.append(f"{name}/{repeat}: unexpected cache status")
        if len({sample["sha256"] for sample in samples}) != 1:
            failures.append(f"{name}: cache differs from uncached render")
        row = {"case": name, "samples": samples,
               "speedup": samples[0]["wall_ms"] / statistics.median(s["wall_ms"] for s in samples[2:])}
        rows.append(row)
        (args.output / "report.json").write_text(json.dumps(
            {"rows": rows, "failures": failures, "complete": False}, indent=2, allow_nan=False))
        print(json.dumps({"case": name, "speedup": row["speedup"], "failures": failures}), flush=True)
    (args.output / "report.json").write_text(json.dumps(
        {"rows": rows, "failures": failures, "complete": True}, indent=2, allow_nan=False))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
