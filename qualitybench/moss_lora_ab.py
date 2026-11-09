#!/usr/bin/env python3
"""LoRA A/B on one resident MOSS 4-step server (resumable). Config: moss_lora_ab.json (styles -> prompts, variants).
Writes <out>/<style>/<tag>/<prompt>_s<seed>.png, results.jsonl, sheets/<style>_<prompt>_s<seed>.jpg.

    python3 qualitybench/moss_lora_ab.py --out qualitybench/out/lora-ab [--styles anime] [--seeds 0,1] [--port 8798]
    --url http://127.0.0.1:8793 reuses a running server instead of spawning one.
"""
import argparse, base64, json, os, signal, subprocess, time
import moss_sweep as m


def sheet(out, style, key, seed, tags, size=384):
    from PIL import Image, ImageDraw
    ims = [(t, Image.open(f).convert("RGB").resize((size, size))) for t in tags
           if os.path.exists(f := f"{out}/{style}/{t}/{key}_s{seed}.png")]
    if not ims:
        return
    cols = min(len(ims), 6)
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cols * size, rows * (size + 20)), "white")
    d = ImageDraw.Draw(S)
    for i, (t, im) in enumerate(ims):
        x, y = (i % cols) * size, (i // cols) * (size + 20)
        d.text((x + 4, y + 3), t, fill="black")
        S.paste(im, (x, y + 20))
    os.makedirs(f"{out}/sheets", exist_ok=True)
    S.save(f"{out}/sheets/{style}_{key}_s{seed}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=f"{m.HERE}/moss_lora_ab.json")
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--url", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--styles", default="")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    L = cfg.pop("_lora")
    styles = [s for s in cfg if not a.styles or s in a.styles.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rp = f"{a.out}/results.jsonl"
    done = {(r["style"], r["tag"], r["prompt"], r["seed"]) for r in map(json.loads, open(rp))} if os.path.exists(rp) else set()
    log = open(rp, "a")
    proc = None
    base = a.url
    if not base:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
        env.update(m.BASE_ENV)
        env["OMNISERVE_NATIVE_SD_DIFFUSION_MODEL"] = os.environ.get("MODEL", m.STEP4)
        proc = subprocess.Popen([m.BIN, "--port", str(a.port)], env=env, stdout=open(f"{a.out}/server.log", "a"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        assert m.wait_ready(a.port, proc), "server failed"
        base = f"http://127.0.0.1:{a.port}"
    url = f"{base}/v1/images/generations"
    m.post(url, dict(prompt="warmup", width=a.size, height=a.size, steps=2, seed=1, guidance_scale=1.0, output_format="png"))
    try:
        for st in styles:
            c = cfg[st]
            for key, text in c["prompts"].items():
                for sd in seeds:
                    for v in c["variants"]:
                        tag = v["tag"]
                        if (st, tag, key, sd) in done:
                            continue
                        body = dict(prompt=v.get("prefix", "") + text + v.get("suffix", ""), width=a.size, height=a.size,
                                    steps=v.get("steps", a.steps), seed=sd, guidance_scale=v.get("cfg", 1.0),
                                    cache_threshold=0, output_format="png")
                        if v.get("loras"):
                            body["loras"] = [dict(path=L[n], scale=s) for n, s in v["loras"]]
                        if v.get("negative"):
                            body["negative_prompt"] = v["negative"]
                        t0 = time.time()
                        try:
                            r = m.post(url, body)
                        except Exception as e:
                            print(f"[{st}] {tag} {key} ERR {e}", flush=True)
                            continue
                        wall = time.time() - t0
                        os.makedirs(f"{a.out}/{st}/{tag}", exist_ok=True)
                        open(f"{a.out}/{st}/{tag}/{key}_s{sd}.png", "wb").write(base64.b64decode(r["data"][0]["b64_json"]))
                        log.write(json.dumps(dict(style=st, tag=tag, prompt=key, seed=sd, text=body["prompt"], wall=round(wall, 2))) + "\n")
                        log.flush()
                        print(f"[{st}] {tag} {key} s{sd} {wall:.1f}s", flush=True)
                    sheet(a.out, st, key, sd, [v["tag"] for v in c["variants"]])
    finally:
        if proc:
            os.killpg(proc.pid, signal.SIGTERM)
    print("LORA AB DONE", flush=True)


if __name__ == "__main__":
    main()
