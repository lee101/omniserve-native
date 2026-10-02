#!/usr/bin/env python3
import urllib.error, argparse, base64, io, json, os, sys, time, urllib.request
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8792/v1/images/generations")
ap.add_argument("--secret", default="localdev")
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.json"))
ap.add_argument("--only", default="")
ap.add_argument("--steps", default="20")
ap.add_argument("--guidance", default="1.0")
ap.add_argument("--size", default="1024x1024")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tag", required=True)
ap.add_argument("--fmt", default="png")
ap.add_argument("--extra", default="{}")
a = ap.parse_args()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", a.tag)
os.makedirs(out, exist_ok=True)
P = json.load(open(a.prompts))
if a.only:
    P = {k: v for k, v in P.items() if k in a.only.split(",")}
w, h = map(int, a.size.split("x"))
tiles = []
for k, p in P.items():
    body = dict(prompt=p, width=w, height=h, steps=int(a.steps), seed=a.seed,
                guidance_scale=float(a.guidance), output_format=a.fmt, **json.loads(a.extra))
    t = time.time()
    req = urllib.request.Request(a.url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-API-Key": a.secret})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        print(f"{a.tag} {k} HTTP {e.code} {e.read()[:300]}", flush=True)
        continue
    wall = time.time() - t
    d = r["data"][0]
    im = Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))).convert("RGB")
    im.save(f"{out}/{k}.png")
    tiles.append((k, im))
    print(f"{a.tag} {k} wall={wall:.1f}s inf={d.get('inference_time_ms')}ms cache={r.get('denoiser_cache')}", flush=True)
s = 512
sheet = Image.new("RGB", (s * len(tiles), s + 18), "white")
dr = ImageDraw.Draw(sheet)
for i, (k, im) in enumerate(tiles):
    sheet.paste(im.resize((s, s)), (i * s, 18))
    dr.text((i * s + 4, 3), f"{a.tag}:{k}", fill="black")
sheet.save(f"{out}/_sheet.png")
