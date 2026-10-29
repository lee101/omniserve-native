#!/usr/bin/env python3
"""MOSS-Image style-LoRA / preset sweep on one resident 4-step server (resumable).

Per style (beauty, anime, gouache, psychedelic) it renders plain vs LoRA vs LoRA+preset (positive suffix,
negative prompt at CFG 1.5) and writes <out>/<style>/<tag>/<prompt>_s<seed>.webp, results.jsonl and
contact sheets in <out>/sheets/. Presets: moss_style_presets.json, prompts: moss_style_prompts.json.

    python3 qualitybench/moss_style_sweep.py --out results/moss-style-20261030 [--styles anime] [--seeds 0,1]
"""
import argparse, base64, json, os, signal, subprocess, time
import moss_sweep as m

NEG_CFG = 1.5


def variants(style, p, neg_all):
    neg = f"{neg_all}, {p['negative']}"
    pos = lambda t: f"{t}, {p['positive_suffix']}"
    trig = lambda t: f"{p['trigger']}, {t}" if p["trigger"] else t
    lora = lambda path, sc=None: [dict(path=path, scale=sc if sc is not None else p["scale"])]
    v = [("plain", lambda t: (t, {}, 1.0)),
         ("plain_pos", lambda t: (pos(t), {}, 1.0)),
         ("plain_neg", lambda t: (pos(t), dict(negative_prompt=neg), NEG_CFG)),
         ("lora", lambda t: (trig(t), dict(loras=lora(p["lora"])), 1.0)),
         ("lora_pos", lambda t: (trig(pos(t)), dict(loras=lora(p["lora"])), 1.0)),
         ("lora_neg", lambda t: (trig(pos(t)), dict(loras=lora(p["lora"]), negative_prompt=neg), NEG_CFG)),
         ("lora07_pos", lambda t: (trig(pos(t)), dict(loras=lora(p["lora"], 0.7)), 1.0))]
    if "alt_lora" in p:
        v.append(("alt_pos", lambda t: (trig(pos(t)), dict(loras=lora(p["alt_lora"])), 1.0)))
    return v


def sheet(out, style, key, seed, tags):
    from PIL import Image, ImageDraw
    ims = []
    for t in tags:
        for ext in ("webp", "png"):
            f = f"{out}/{style}/{t}/{key}_s{seed}.{ext}"
            if os.path.exists(f):
                ims.append((t, Image.open(f).convert("RGB").resize((384, 384))))
                break
    if not ims:
        return
    cols = 4
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cols * 384, rows * 404), "white")
    d = ImageDraw.Draw(S)
    for i, (t, im) in enumerate(ims):
        x, y = (i % cols) * 384, (i // cols) * 404
        d.text((x + 4, y + 3), t, fill="black")
        S.paste(im, (x, y + 20))
    os.makedirs(f"{out}/sheets", exist_ok=True)
    S.save(f"{out}/sheets/{style}_{key}_s{seed}.jpg", quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--styles", default="")
    ap.add_argument("--extra-seeds", default="2,3", help="extra seeds for the anime 'legs' prompt")
    a = ap.parse_args()
    presets = json.load(open(f"{m.HERE}/moss_style_presets.json"))
    prompts = json.load(open(f"{m.HERE}/moss_style_prompts.json"))
    neg_all = presets.pop("_global_negative")
    styles = [s for s in presets if not a.styles or s in a.styles.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    extra = [int(s) for s in a.extra_seeds.split(",") if s]
    os.makedirs(a.out, exist_ok=True)
    rp = f"{a.out}/results.jsonl"
    done = set()
    if os.path.exists(rp):
        done = {(r["style"], r["tag"], r["prompt"], r["seed"]) for r in map(json.loads, open(rp))}
    log = open(rp, "a")
    full = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    full.update(m.BASE_ENV)
    full["OMNISERVE_NATIVE_SD_DIFFUSION_MODEL"] = m.STEP4
    proc = subprocess.Popen([m.BIN, "--port", str(a.port)], env=full, stdout=open(f"{a.out}/server.log", "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    assert m.wait_ready(a.port, proc), "server failed"
    url = f"http://127.0.0.1:{a.port}/v1/images/generations"
    m.post(url, dict(prompt="warmup", width=1024, height=1024, steps=2, seed=1, guidance_scale=1.0, output_format="png"))
    for st in styles:
        p = presets[st]
        for key, text in prompts[st].items():
            ss = seeds + (extra if st == "anime" and key == "legs" else [])
            for sd in ss:
                for tag, fn in variants(st, p, neg_all):
                    if (st, tag, key, sd) in done:
                        continue
                    t, over, cfg = fn(text)
                    body = dict(prompt=t, width=1024, height=1024, steps=4, seed=sd, guidance_scale=cfg,
                                cache_threshold=0, output_format="png", **over)
                    t0 = time.time()
                    try:
                        r = m.post(url, body)
                    except Exception as e:
                        print(f"[{st}] {tag} {key} ERR {e}", flush=True)
                        continue
                    blob = base64.b64decode(r["data"][0]["b64_json"])
                    ext = "webp" if blob[8:12] == b"WEBP" else "png"
                    os.makedirs(f"{a.out}/{st}/{tag}", exist_ok=True)
                    open(f"{a.out}/{st}/{tag}/{key}_s{sd}.{ext}", "wb").write(blob)
                    log.write(json.dumps(dict(style=st, tag=tag, prompt=key, seed=sd, wall=round(time.time() - t0, 2))) + "\n")
                    log.flush()
                    print(f"[{st}] {tag} {key} s{sd} {time.time() - t0:.1f}s", flush=True)
                sheet(a.out, st, key, sd, [t for t, _ in variants(st, p, neg_all)])
    os.killpg(proc.pid, signal.SIGTERM)
    print("STYLE SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
