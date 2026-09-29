#!/usr/bin/env python3
"""Repeat native img2img requests and save a source/output sheet for human review.

Pixel consistency and source retention are diagnostics, not instruction-following
scores. No automatic semantic quality claim is made by this harness.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_parity_bench import decode_image_response, entropy, global_ssim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8793")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats required")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB").resize((512, 512))
    data = io.BytesIO()
    source.save(data, format="PNG")
    encoded = base64.b64encode(data.getvalue()).decode()
    rows = []
    thumbnails = [("Source", source)]
    failures = []
    for strength in (0.4, 0.6, 0.8):
        for style, prompt in (
            ("watercolor", "Watercolor painting of a translucent glass robot toy on a white background"),
            ("gold", "Product photo of a golden metal robot toy on a white seamless studio background"),
        ):
            timings, hashes = [], []
            for repeat in range(args.repeats):
                payload = {"prompt": prompt, "image_base64": encoded,
                           "size": "512x512", "strength": strength,
                           "steps": 9, "guidance_scale": 0, "seed": 90826}
                request = urllib.request.Request(args.base + "/v1/images/img2img",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json",
                             "X-Omniserve-Tier": "background"})
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=240) as response:
                    image, meta = decode_image_response(response.read(),
                        response.headers.get("Content-Type", ""), 240)
                timings.append((time.perf_counter() - start) * 1000)
                hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                image.save(args.output / f"{style}-{strength}-{repeat}.png")
                if repeat == 0:
                    thumbnails.append((f"{style} strength={strength}", image))
            array = np.asarray(image)
            row = {"style": style, "strength": strength, "wall_ms": timings,
                   "median_ms": statistics.median(timings), "pixel_hashes": hashes,
                   "repeat_exact": len(set(hashes)) == 1,
                   "entropy": entropy(image), "stddev": float(array.std()),
                   "source_ssim": global_ssim(np.asarray(source), array)}
            if not row["repeat_exact"] or row["entropy"] < 3 or row["stddev"] < 8:
                failures.append(f"{style}-{strength}: consistency or nonblank gate failed")
            rows.append(row)
            print(json.dumps(row), flush=True)
    sheet = Image.new("RGB", (4 * 256, 2 * 280), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(thumbnails):
        x, y = (i % 4) * 256, (i // 4) * 280
        sheet.paste(image.resize((256, 256)), (x, y + 24))
        draw.text((x + 4, y + 4), label, fill="black")
    sheet.save(args.output / "contact-sheet.png")
    report = {"rows": rows, "failures": failures,
              "quality_status": "requires human source/edit review; metrics are diagnostics"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
