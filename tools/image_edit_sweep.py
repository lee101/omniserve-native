#!/usr/bin/env python3
"""Repeated same-seed edit sweep. All timed generation requests bypass result cache.

The first step/guidance combination is the reference. Retention metrics are
diagnostics; inspect the saved source/output sheets for edit adherence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from image_cache_bench import encode
from image_parity_bench import decode_image_response, entropy, global_ssim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8793")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 12, 8])
    parser.add_argument("--guidances", type=float, nargs="+", default=[2.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[90908])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prompt", action="append", help="repeat for each instruction")
    parser.add_argument(
        "--reference-dir", type=Path, help="compare to an earlier dense sweep"
    )
    parser.add_argument("--reference-steps", type=int, default=20)
    parser.add_argument("--expect-easycache", type=float, default=0)
    args = parser.parse_args()
    if args.repeats < 2 or any(s < 1 for s in args.steps):
        parser.error("need positive steps and at least 2 repeats")
    prompts = args.prompt or [
        "Change the text on the sign from 'flux.cpp' to 'edit.cpp'. Keep the cat and background unchanged.",
        "Change the background to light blue. Keep the cat and the sign unchanged.",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB").resize((512, 512))
    source.save(args.output / "source.png")
    report = {
        "complete": False,
        "rows": [],
        "failures": [],
        "source": str(args.source.resolve()),
        "source_pixel_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "quality_status": "retention and repeat diagnostics; requires visual edit-adherence review",
    }
    references, tiles = {}, [("source", source)]

    def save() -> None:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )

    try:
        for prompt_index, prompt in enumerate(prompts):
            for seed in args.seeds:
                samples = {}
                # Alternate profile order on repeat to reduce temporal bias.
                profiles = [(s, g) for s in args.steps for g in args.guidances]
                for repeat in range(args.repeats):
                    for steps, guidance in (
                        profiles if repeat % 2 == 0 else reversed(profiles)
                    ):
                        key = (prompt_index, seed, steps, guidance)
                        payload = {
                            "prompt": prompt,
                            "image_base64": encode(source),
                            "size": "512x512",
                            "steps": steps,
                            "guidance_scale": guidance,
                            "seed": seed,
                            "cache": False,
                            "teleport": False,
                        }
                        request = urllib.request.Request(
                            args.base + "/v1/images/edits",
                            data=json.dumps(payload).encode(),
                            headers={
                                "Content-Type": "application/json",
                                "X-Omniserve-Tier": "background",
                            },
                        )
                        start = time.perf_counter()
                        with urllib.request.urlopen(request, timeout=900) as response:
                            image, meta = decode_image_response(
                                response.read(),
                                response.headers.get("Content-Type", ""),
                                900,
                            )
                        elapsed = (time.perf_counter() - start) * 1000
                        name = f"p{prompt_index}-s{seed}-steps{steps}-cfg{guidance}-r{repeat}"
                        image.save(args.output / (name + ".png"))
                        baseline_key = (prompt_index, seed)
                        if baseline_key not in references:
                            if args.reference_dir:
                                reference_name = (
                                    f"p{prompt_index}-s{seed}-steps{args.reference_steps}"
                                    f"-cfg{args.guidances[0]}-r0.png"
                                )
                                references[baseline_key] = Image.open(
                                    args.reference_dir / reference_name
                                ).convert("RGB")
                            else:
                                references[baseline_key] = image
                        baseline = np.asarray(references[baseline_key]).astype(
                            np.float64
                        )
                        array = np.asarray(image).astype(np.float64)
                        mse = float(np.mean((array - baseline) ** 2))
                        row = {
                            "case": name,
                            "prompt": prompt,
                            "seed": seed,
                            "steps": steps,
                            "guidance": guidance,
                            "repeat": repeat,
                            "wall_ms": elapsed,
                            "metadata": meta,
                            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "baseline_ssim": global_ssim(baseline, array),
                            "baseline_psnr": float(10 * np.log10(255**2 / mse))
                            if mse
                            else None,
                            "baseline_exact": mse == 0,
                            "entropy": entropy(image),
                            "stddev": float(array.std()),
                        }
                        if row["entropy"] < 3 or row["stddev"] < 8:
                            report["failures"].append(
                                name + ": blank/low-information image"
                            )
                        if (meta.get("cache") or {}).get("hit") or (
                            meta.get("teleport") or {}
                        ).get("used"):
                            report["failures"].append(name + ": cache bypass failed")
                        observed = (meta.get("denoiser_cache") or {}).get(
                            "threshold", 0
                        )
                        if abs(observed - args.expect_easycache) > 1e-6:
                            report["failures"].append(
                                name + ": wrong denoiser cache configuration"
                            )
                        report["rows"].append(row)
                        samples.setdefault(key, []).append(row)
                        if repeat == 0:
                            tiles.append(
                                (
                                    f"p{prompt_index} seed{seed} {steps}steps cfg{guidance}",
                                    image,
                                )
                            )
                        save()
                        print(
                            json.dumps(
                                {
                                    "case": name,
                                    "wall_ms": elapsed,
                                    "baseline_ssim": row["baseline_ssim"],
                                }
                            ),
                            flush=True,
                        )
                for key, rows in samples.items():
                    if len({r["pixel_sha256"] for r in rows}) != 1:
                        report["failures"].append(
                            f"{key}: repeated uncached outputs differ"
                        )
                report.setdefault("summaries", []).extend(
                    {
                        "prompt_index": key[0],
                        "seed": key[1],
                        "steps": key[2],
                        "guidance": key[3],
                        "median_ms": statistics.median(row["wall_ms"] for row in rows),
                        "wall_ms": [row["wall_ms"] for row in rows],
                    }
                    for key, rows in samples.items()
                )
        report["complete"] = True
    finally:
        sheet = Image.new("RGB", (3 * 256, ((len(tiles) + 2) // 3) * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (label, image) in enumerate(tiles):
            x, y = i % 3 * 256, i // 3 * 280
            sheet.paste(image.resize((256, 256)), (x, y + 24))
            draw.text((x + 3, y + 3), label, fill="black")
        sheet.save(args.output / "contact-sheet.png")
        save()
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
