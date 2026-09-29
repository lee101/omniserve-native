#!/usr/bin/env python3
"""Smoke-test the ra2 worker plugin against a real GPU and real weights.

Run on the shared box with the model volume staged (symlinks are fine):

    RA2_MODELS_DIR=/tmp/ra2-models \
    RA2_SD_LIB=/vfast/data/code/stable-diffusion.cpp-master/build/bin/libstable-diffusion.so \
    RA2_PARAMS_BACKEND=te=cpu \
    python3 tools/ra2_worker_smoke.py

It proves four things a build cannot: the resident context loads the pinned
weight set, text-to-image returns a decodable image in the OmniServe response
shape, a second call reuses the loaded context (the whole point of the ctypes
binding over sd-cli), and an `image_base64` request takes the reference-edit
path.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workloads import qwen_image  # noqa: E402


def show(label: str, result: dict) -> dict:
    payload = {key: value for key, value in result.items() if key != "outputs"}
    print(f"{label}: {json.dumps(payload)[:400]}", flush=True)
    assert result["model"] == "qwen-image-2.1", result["model"]
    assert result["data"] and result["data"][0]["b64_json"], "no image in response"
    assert result["outputs"] and result["outputs"][0]["content_type"].startswith("image/")
    assert "omniserve" not in result  # the runtime adds that, not the plugin
    return result


def decode(result: dict, index: int = 0):
    from PIL import Image

    raw = base64.b64decode(result["data"][index]["b64_json"])
    image = Image.open(io.BytesIO(raw))
    image.load()
    print(f"  image {image.format} {image.size} {image.mode} {len(raw) / 1024:.1f} KiB", flush=True)
    return image


def main() -> int:
    prompt = os.getenv("RA2_SMOKE_PROMPT", "a red fox sitting in snow, photograph")
    started = time.monotonic()
    first = show("t2i", qwen_image.handler({"input": {
        "prompt": prompt, "width": 512, "height": 512, "steps": 4, "seed": 7,
    }}))
    decode(first)
    print(f"  cold wall {time.monotonic() - started:.1f}s (load {first['timings']['load_ms']} ms)", flush=True)

    started = time.monotonic()
    second = show("t2i-warm", qwen_image.handler({"input": {
        "prompt": prompt + ", golden hour", "width": 512, "height": 512, "steps": 4, "seed": 8,
    }}))
    decode(second)
    warm = time.monotonic() - started
    print(f"  warm wall {warm:.1f}s", flush=True)
    if qwen_image._state.get("load_ms") and qwen_image._state["load_ms"] > warm * 1000:
        print("  load was reused: warm call is faster than the first load", flush=True)

    edit = show("edit", qwen_image.handler({"input": {
        "prompt": "repaint this as a watercolour illustration",
        "image_base64": first["data"][0]["b64_json"],
        "width": 512, "height": 512, "steps": 4, "seed": 9,
    }}))
    decode(edit)
    assert edit["timings"]["reference_edit"] is True, edit["timings"]
    print(f"edit wall {edit['timings']['sample_ms']} ms sampling", flush=True)

    png = show("png", qwen_image.handler({"input": {
        "prompt": prompt, "width": 512, "height": 512, "steps": 4, "output_format": "png", "seed": 11,
        # The manifest runtime stamps its own keys into the input it hands the
        # plugin; they are plumbing, not caller input, so they must be ignored.
        "_omniserve_profile": "default", "_omniserve_retry": 0,
    }}))
    assert png["format"] == "png" and png["outputs"][0]["content_type"] == "image/png"
    decode(png)

    for bad, why in (
        ({"prompt": ""}, "empty prompt"),
        ({"prompt": "ok", "width": 100}, "bad width"),
        ({"prompt": "ok", "bogus_field": 1}, "unknown field"),
        ({"prompt": "ok", "width": 512, "height": 512, "image_base64": "!!!not base64!!"}, "bad image"),
    ):
        try:
            qwen_image.handler({"input": bad})
        except ValueError as error:
            print(f"  rejected {why}: {error}", flush=True)
        else:
            raise AssertionError(f"{why} was accepted")

    print("ra2 worker smoke ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
