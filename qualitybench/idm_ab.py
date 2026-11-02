#!/usr/bin/env python3
"""Render the SFW text/layout prompts plus style prompts on one lane; build contact sheets for A/B.
uv run --with pillow qualitybench/idm_ab.py render PORT OUTDIR [SEED] | sheet OUT.jpg DIR_A DIR_B [DIR_C...]"""
import base64, json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))

def prompts():
    p = [(k, v) for k, v in json.load(open(f"{HERE}/moss_prompts_sfw.json")).items()] if isinstance(json.load(open(f"{HERE}/moss_prompts_sfw.json")), dict) else []
    s = json.load(open(f"{HERE}/moss_style_prompts.json"))
    for fam, n in (("beauty", ["p1", "p2"]), ("anime", ["legs", "p1", "p3"]), ("gouache", ["fox"]), ("psychedelic", ["ram"])):
        p += [(f"{fam}_{k}", s[fam][k]) for k in n]
    return p

def render(port, out, seed):
    os.makedirs(out, exist_ok=True)
    ts = {}
    for i, (k, text) in enumerate(prompts()):
        body = json.dumps({"prompt": text, "width": 1024, "height": 1024, "seed": seed, "output_format": "webp"}).encode()
        t = time.time()
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/generations", body, {"Content-Type": "application/json"}), timeout=1800))
        open(f"{out}/{k}.webp", "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
        ts[k] = round(time.time() - t, 2)
        print(k, ts[k], flush=True)
    json.dump(ts, open(f"{out}/times.json", "w"))

def sheet(out, dirs):
    from PIL import Image, ImageDraw
    keys = [k for k, _ in prompts() if all(os.path.exists(f"{d}/{k}.webp") for d in dirs)]
    S = 384
    img = Image.new("RGB", (S * len(dirs), S * len(keys)), "white")
    for r, k in enumerate(keys):
        for c, d in enumerate(dirs):
            im = Image.open(f"{d}/{k}.webp").convert("RGB").resize((S, S))
            ImageDraw.Draw(im).text((6, 6), f"{os.path.basename(d.rstrip('/'))}:{k}", fill="yellow")
            img.paste(im, (S * c, S * r))
    img.save(out, quality=85)

if __name__ == "__main__":
    a = sys.argv[1:]
    render(a[1], a[2], int(a[3]) if len(a) > 3 else 7) if a[0] == "render" else sheet(a[1], a[2:])
