#!/usr/bin/env python3
"""Qwen multitenant worker bench: same corpus/seeds locally (in-process) or on a RunPod endpoint.

local:  python3 tools/qwen_mt_bench.py --out DIR local
runpod: python3 tools/qwen_mt_bench.py --out DIR --ref LOCALDIR runpod --endpoint ID [--seq ra2:0,ra2:1,edit:0]
Outputs PNGs, per-job timings (RunPod delay/execution split) and global SSIM/PSNR vs the reference dir.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from image_parity_bench import pixel_metrics  # noqa: E402

SOURCE = Path("/nvme0n1-disk/code/stable-diffusion.cpp/assets/flux/flux1-dev-q8_0.png")
RA2 = [
    ("A red fox sitting in fresh snow at dawn, soft golden light, detailed fur, photograph", 424242),
    ("A cozy coffee shop storefront with a chalkboard sign that reads \"OPEN LATE\", watercolor illustration", 777),
    ("Portrait of an elderly fisherman with a weathered face and a knitted blue cap, studio lighting", 90210),
]
EDIT = [
    ("Change the text on the sign from 'flux.cpp' to 'edit.cpp'. Keep the cat and background unchanged.", 90908),
    ("Change the background to light blue. Keep the cat and the sign unchanged.", 90908),
]


def source_b64() -> str:
    buf = io.BytesIO()
    Image.open(SOURCE).convert("RGB").resize((512, 512)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def case(key: str, src: str) -> dict:
    task, index = key.split(":")
    if task == "probe":
        return {"task": "probe"}
    prompt, seed = (RA2 if task == "ra2" else EDIT)[int(index)]
    body = {"task": task, "prompt": prompt, "seed": seed, "output_format": "png"}
    if task == "ra2":
        body.update(width=1024, height=1024, steps=20)
    else:
        body.update(width=512, height=512, steps=20, guidance_scale=2.5, image_base64=src)
    return body


def runpod_call(endpoint: str, body: dict, timeout: float) -> tuple[dict, dict]:
    key = os.environ["RUNPOD_API_KEY"]

    def req(method, path, data=None):
        r = urllib.request.Request(f"https://api.runpod.ai/v2/{endpoint}{path}", method=method,
                                   data=None if data is None else json.dumps(data).encode(),
                                   headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                            "User-Agent": "omniserve-bench/1.0"})
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.load(resp)

    started = time.monotonic()
    job = req("POST", "/run", {"input": body})
    while time.monotonic() - started < timeout:
        state = req("GET", "/status/" + job["id"])
        if state.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(1.0)
    else:
        req("POST", "/cancel/" + job["id"])
        state = {"status": "BENCH_TIMEOUT"}
    meta = {"job": job["id"], "status": state.get("status"), "wall_ms": int((time.monotonic() - started) * 1000),
            "delay_ms": state.get("delayTime"), "exec_ms": state.get("executionTime"),
            "worker": state.get("workerId"), "error": str(state.get("error"))[:300] if state.get("error") else None}
    return state.get("output") or {}, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--seq", default="ra2:0,ra2:1,ra2:2,edit:0,edit:1")
    ap.add_argument("--timeout", type=float, default=900)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("local")
    rp = sub.add_parser("runpod")
    rp.add_argument("--endpoint", required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = source_b64()
    if a.mode == "local":
        from workloads import qwen_mt
        started = time.monotonic()
        qwen_mt.boot()
        print(json.dumps({"boot_ms": int((time.monotonic() - started) * 1000)}), flush=True)
    rows = []
    for n, key in enumerate(a.seq.split(",")):
        body = case(key, src)
        if a.mode == "local":
            t = time.monotonic()
            out = qwen_mt.handler({"input": body})
            meta = {"wall_ms": int((time.monotonic() - t) * 1000), "status": "COMPLETED"}
        else:
            out, meta = runpod_call(a.endpoint, body, a.timeout)
        row = {"n": n, "case": key, **meta, "timings": out.get("timings"), "error": out.get("error") or meta.get("error")}
        if key.startswith("probe"):
            row["probe"] = out
        data = (out.get("data") or [{}])[0].get("b64_json")
        if data:
            image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
            name = key.replace(":", "-") + ".png"
            image.save(a.out / f"{n:02d}-{name}")
            if not (a.out / name).exists():
                image.save(a.out / name)
            if a.ref and (a.ref / name).exists():
                m = pixel_metrics(Image.open(a.ref / name).convert("RGB"), image)
                row["metrics"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()
                                  if k in ("identical", "psnr_db", "ssim_global", "edge_cosine")}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.out / "results.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
