#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, struct, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAST = os.path.expanduser("~/models-fast/qwen-image-2.1")
CLI = os.environ.get("SD_CLI", "/vfast/data/code/sdcpp-qwen-prefixkv/build-86/bin/sd-cli")


def load_lat(p):
    b = open(p, "rb").read()
    nd = struct.unpack("<i", b[:4])[0]
    shape = struct.unpack(f"<{nd}q", b[4:4 + 8 * nd])
    return np.frombuffer(b[4 + 8 * nd:], dtype="<f4").reshape(shape)


def run(a):
    prompts = json.load(open(a.prompts))
    os.makedirs(a.out, exist_ok=True)
    steps = list(range(1, a.steps + 1))
    for pk, pv in prompts.items():
        d = f"{a.out}/{pk}"
        if os.path.exists(f"{d}/log.txt") and os.path.exists(f"{d}/lat_step{a.steps}.lat"):
            continue
        os.makedirs(d, exist_ok=True)
        cmd = [CLI, "--diffusion-model", f"{FAST}/qwen-image-2.1-Q4_K_M.gguf", "--llm", f"{FAST}/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
               "--vae", f"{FAST}/vae/qwen_image_2.1_vae_bf16.safetensors", "--cfg-scale", "1.0", "--sampling-method", "euler",
               "--steps", str(a.steps), "-W", "1024", "-H", "1024", "--seed", "0", "--vae-tiling", "-v",
               "-p", pv, "-o", f"{d}/img.png", "--latent-save-steps", ",".join(map(str, steps)), "--latent-save-prefix", f"{d}/lat"]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        open(f"{d}/log.txt", "w").write(r.stdout + r.stderr)
        print(pk, r.returncode, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=f"{HERE}/sweep_prompts_hard.json")
    ap.add_argument("--steps", type=int, default=30)
    run(ap.parse_args())
