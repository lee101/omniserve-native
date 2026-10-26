#!/usr/bin/env python3
"""Benchmark an isolated native Qwen lane; retain images, requests and timings."""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.request


CASES = [
    ("portrait", "Editorial photograph of an attractive adult woman aged 28 wearing a red jacket, "
     "waist-up, both hands holding a ceramic coffee cup, natural skin texture, "
     "soft window light, cozy cafe background, realistic anatomy, detailed fingers."),
    ("couple", "Candid full-body photograph of an attractive adult couple aged 30 walking along "
     "a beach at sunset, a man in a white linen shirt and a woman in a blue summer dress, "
     "holding hands, realistic proportions, natural skin texture, warm cinematic light."),
    ("text", 'A coffee shop entrance with a chalkboard reading "Qwen Coffee, $2 per cup".'),
]


def request(port, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=Path("build-qwen/omniserve-native"))
    ap.add_argument("--dit", type=Path, required=True)
    ap.add_argument("--llm", type=Path, required=True)
    ap.add_argument("--vision", type=Path, required=True)
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--lora", type=Path)
    ap.add_argument("--profile", choices=["lightning", "viggle", "base"], default="lightning")
    ap.add_argument("--params-backend", default="*=cpu")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vae-tiling", action="store_true")
    ap.add_argument("--vae-tile", default="64x32")
    ap.add_argument("--vae-overlap", type=float, default=.5)
    ap.add_argument("--edit-source", type=Path)
    ap.add_argument("--output-format", choices=["webp", "png"], default="webp")
    args = ap.parse_args()
    if args.repeat < 1 or not 1 <= args.cases <= len(CASES):
        ap.error("repeat must be positive; cases must be 1..3")
    if args.size < 64 or args.size % 64 or not 1 <= args.steps <= 100:
        ap.error("size must be a positive multiple of 64; steps must be 1..100")
    try:
        tiles = [int(value) for value in args.vae_tile.split("x")]
        if len(tiles) != 2 or any(not 32 <= value <= 4096 for value in tiles):
            raise ValueError
    except ValueError:
        ap.error("vae-tile must be WIDTHxHEIGHT with dimensions 32..4096")
    if not 0 <= args.vae_overlap <= .95:
        ap.error("vae-overlap must be between 0 and 0.95")
    if args.edit_source and not args.edit_source.is_file():
        ap.error("edit-source file missing")
    for key in ("binary", "dit", "llm", "vision", "vae"):
        if not getattr(args, key).is_file():
            ap.error(f"{key} file missing: {getattr(args, key)}")
    if args.profile != "base" and (not args.lora or not args.lora.is_file()):
        ap.error("distilled profiles require --lora")
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_")}
    config = {
        "BIND": "127.0.0.1", "SD_MODEL": str(args.dit.resolve()),
        "SD_DIFFUSION_MODEL": str(args.dit.resolve()), "SD_LLM": str(args.llm.resolve()),
        "SD_LLM_VISION": str(args.vision.resolve()), "SD_VAE": str(args.vae.resolve()),
        "SD_PARAMS_BACKEND": args.params_backend, "SD_REFERENCE_EDIT": "1",
        "SD_VAE_TILING": str(int(args.vae_tiling)), "SD_NOTCH": "0",
        "SD_CACHE_MODE": "off", "SD_TELEPORT": "0", "SD_WARMUP": "0",
        "SD_LOG_LEVEL": "2",
        "SD_IMAGE_FORMAT": args.output_format,
    }
    tile_x, tile_y = (str(value) for value in tiles)
    config.update({"SD_VAE_TILE_X": tile_x, "SD_VAE_TILE_Y": tile_y,
                   "SD_VAE_TILE_OVERLAP": str(args.vae_overlap)})
    if args.profile != "base":
        config["SD_ACCEL_PROFILE"] = args.profile
        config["SD_DEFAULT_LORA"] = str(args.lora.resolve())
        if args.profile == "viggle":
            config["SD_TURBO_NODES"] = "1,0.9375,0.875,0.75,0.5,0.25"
    env.update({"OMNISERVE_NATIVE_" + k: v for k, v in config.items()})
    result = {"config": config, "runs": []}
    if args.lora:
        result["lora_sha256"] = hashlib.sha256(args.lora.read_bytes()).hexdigest()
    report = args.out / "results.json"
    with (args.out / "native.log").open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve()), "--port", str(port)],
                                   env=env, stdout=log, stderr=log)
        try:
            start = time.perf_counter()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"native gateway exited; see {args.out / 'native.log'}")
                try:
                    status = request(port, "/status", timeout=2)
                    if not status.get("diffusion", {}).get("ready"):
                        raise RuntimeError("diffusion backend failed to load")
                    break
                except (OSError, TimeoutError):
                    if time.perf_counter() - start > 180:
                        raise TimeoutError("gateway startup timed out")
                    time.sleep(0.25)
            result["startup_s"] = time.perf_counter() - start
            result["status"] = status

            def render(name, prompt, image=None):
                body = {"prompt": prompt, "width": args.size, "height": args.size,
                        "seed": 42, "steps": args.steps, "guidance_scale": 1,
                        "cache": False, "teleport": False, "notch": False,
                        "output_format": args.output_format}
                if image:
                    body["image_base64"] = base64.b64encode(image.read_bytes()).decode()
                (args.out / f"{name}-request.json").write_text(json.dumps(
                    {**body, **({"image_base64": str(image)} if image else {})}, indent=2))
                durations = []
                for index in range(args.repeat):
                    started = time.perf_counter()
                    response = request(port, "/v1/images/edits" if image else "/v1/images/generations", body)
                    elapsed = time.perf_counter() - started
                    item = response["data"][0]
                    target = args.out / f"{name}-{index}.png"
                    from PIL import Image
                    raw = base64.b64decode(item["b64_json"])
                    Image.open(io.BytesIO(raw)).convert("RGB").save(target)
                    entry = {"case": name, "repeat": index, "wall_s": elapsed,
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "metadata": {k: v for k, v in item.items() if k != "b64_json"}}
                    result["runs"].append(entry)
                    report.write_text(json.dumps(result, indent=2))
                    print(json.dumps(entry), flush=True)
                    durations.append(elapsed)
                return statistics.median(durations[1:] or durations)

            result["warm_median_s"] = {}
            for name, prompt in CASES[:args.cases]:
                result["warm_median_s"][name] = render(name, prompt)
            result["warm_median_s"]["edit"] = render(
                "edit", "Change the red jacket to dark blue. Keep the person's identity, "
                "pose, hands, cup, background and lighting unchanged.",
                args.edit_source or args.out / "portrait-0.png")
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            report.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
