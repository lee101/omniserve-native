#!/usr/bin/env python3
"""Hires-fix ablation on the MOSS 4-step server: base render, lanczos upscale, img2img refine (/v1/images/img2img + strength).
Variants are (tag, base_px, up, steps, strength); up=0 means base only. Writes <out>/<tag>/<key>_s<seed>.png, results.jsonl, sheets/.

    python3 qualitybench/moss_hires_ab.py --out qualitybench/out/hires-ab [--url http://127.0.0.1:8798] [--seeds 0,1]
"""
import argparse, base64, io, json, os, signal, subprocess, time
import moss_sweep as m
from PIL import Image

VARIANTS = [("b1024", 1024, 0, 0, 0), ("b1024_hr15_s35", 1024, 1.5, 8, 0.35), ("b1024_hr15_s50", 1024, 1.5, 8, 0.5),
            ("b768", 768, 0, 0, 0), ("b768_hr15_s40", 768, 1.5, 8, 0.4), ("b1024_hr125_s40", 1024, 1.25, 8, 0.4)]


def b64png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=f"{m.HERE}/moss_lora_ab.json")
    ap.add_argument("--styles", default="people,anime")
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--url", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cfg = json.load(open(a.prompts))
    suffix = {"people": ", natural skin texture, sharp eyes, correct anatomy", "anime": ", anime character, dynamic composition, clean anatomy"}
    jobs = [(f"{st}_{k}", t + suffix.get(st, "")) for st in a.styles.split(",") for k, t in cfg[st]["prompts"].items()]
    vs = [v for v in VARIANTS if not a.only or v[0] in a.only.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rp = f"{a.out}/results.jsonl"
    done = {(r["tag"], r["prompt"], r["seed"]) for r in map(json.loads, open(rp))} if os.path.exists(rp) else set()
    log = open(rp, "a")
    proc = None
    base = a.url
    if not base:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
        env.update(m.BASE_ENV); env["OMNISERVE_NATIVE_SD_DIFFUSION_MODEL"] = os.environ.get("MODEL", m.STEP4)
        proc = subprocess.Popen([m.BIN, "--port", str(a.port)], env=env, stdout=open(f"{a.out}/server.log", "a"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        assert m.wait_ready(a.port, proc), "server failed"
        base = f"http://127.0.0.1:{a.port}"
    gen = lambda body: Image.open(io.BytesIO(base64.b64decode(m.post(f"{base}/v1/images/generations", body)["data"][0]["b64_json"]))).convert("RGB")
    edit = lambda body: Image.open(io.BytesIO(base64.b64decode(m.post(f"{base}/v1/images/img2img", body)["data"][0]["b64_json"]))).convert("RGB")
    for px in sorted({v[1] for v in vs}):
        gen(dict(prompt="warmup", width=px, height=px, steps=2, seed=1, guidance_scale=1.0, output_format="png"))
    cache = {}
    try:
        for key, text in jobs:
            for sd in [int(s) for s in a.seeds.split(",")]:
                for tag, px, up, steps, strength in vs:
                    if (tag, key, sd) in done:
                        continue
                    t0 = time.time()
                    if (px, key, sd) not in cache:
                        cache[(px, key, sd)] = (gen(dict(prompt=text, width=px, height=px, steps=4, seed=sd, guidance_scale=1.0,
                                                         cache_threshold=0, output_format="png")), time.time() - t0)
                    im, tb = cache[(px, key, sd)]
                    th = 0.0
                    if up:
                        W = int(px * up) // 64 * 64
                        t1 = time.time()
                        im = edit(dict(prompt=text, image_base64=b64png(im.resize((W, W), Image.LANCZOS)), strength=strength,
                                       width=W, height=W, steps=steps, seed=sd + 1000, guidance_scale=1.0, output_format="png"))
                        th = time.time() - t1
                    os.makedirs(f"{a.out}/{tag}", exist_ok=True)
                    im.save(f"{a.out}/{tag}/{key}_s{sd}.png")
                    log.write(json.dumps(dict(tag=tag, prompt=key, seed=sd, base_s=round(tb, 2), hires_s=round(th, 2), wall=round(tb + th, 2))) + "\n"); log.flush()
                    print(f"{tag} {key} s{sd} base {tb:.1f}s hires {th:.1f}s", flush=True)
                cache = {k: v for k, v in cache.items() if k[1:] == (key, sd)}
    finally:
        if proc:
            os.killpg(proc.pid, signal.SIGTERM)
    print("HIRES AB DONE", flush=True)


if __name__ == "__main__":
    main()
