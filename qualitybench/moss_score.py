#!/usr/bin/env python3
"""Score a moss_sweep.py run: LPIPS/PSNR/CLIP vs the dense 50-step reference of each family, steady-state
latency, and per-prompt contact sheets.

    uv run --with torch --with lpips --with open_clip_torch --with pillow --with numpy qualitybench/moss_score.py results/moss-20261029
"""
import collections, json, os, statistics, sys
import numpy as np, torch, lpips, open_clip
from PIL import Image, ImageDraw

out = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
REF = {"sfw": "d50", "nsfw": "d50"}
prompts = {f: json.load(open(f"{HERE}/moss_prompts_{f}.json")) for f in ("sfw", "nsfw")}
rows = [json.loads(l) for l in open(f"{out}/results.jsonl")]
by = collections.defaultdict(lambda: collections.defaultdict(dict))  # family -> tag -> (prompt, seed) -> row
for r in rows:
    by[r["family"]][r["tag"]][(r["prompt"], r["seed"])] = r

torch.set_num_threads(max(1, os.cpu_count() // 2))
net = lpips.LPIPS(net="alex", verbose=False)
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()


def load(r):
    return Image.open(f"{out}/{r['file']}").convert("RGB")


def lp(a, b):
    t = lambda im: torch.from_numpy(np.asarray(im.resize((512, 512)), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    with torch.no_grad():
        return float(net(t(a), t(b)))


def psnr(a, b):
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    m = float(((x - y) ** 2).mean())
    return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)


def clip(im, text):
    with torch.no_grad():
        i = model.encode_image(pre(im)[None]); t = model.encode_text(tok([text[:300]]))
        i /= i.norm(dim=-1, keepdim=True); t /= t.norm(dim=-1, keepdim=True)
    return float((i @ t.T)[0, 0])


summary = []
for fam, tags in by.items():
    ref = tags.get(REF[fam], {})
    for tag, items in tags.items():
        lps, pss, cl, walls, infs = [], [], [], [], []
        for key, r in items.items():
            im = load(r)
            cl.append(clip(im, prompts[fam][key[0]]))
            walls.append(r["wall"])
            if r.get("inf_ms"):
                infs.append(r["inf_ms"] / 1000)
            if key in ref and tag != REF[fam]:
                lps.append(lp(im, load(ref[key]))); pss.append(psnr(im, load(ref[key])))
        summary.append(dict(family=fam, tag=tag, server=next(iter(items.values()))["server"], steps=next(iter(items.values()))["steps"],
                            extra=next(iter(items.values()))["extra"], n=len(items),
                            wall_median=round(statistics.median(walls), 2), inf_median=round(statistics.median(infs), 2) if infs else None,
                            lpips=round(statistics.mean(lps), 4) if lps else None, psnr=round(statistics.mean(pss), 2) if pss else None,
                            clip=round(statistics.mean(cl), 4)))
        print(summary[-1], flush=True)
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
with open(f"{out}/summary.csv", "w") as f:
    f.write("family,tag,steps,n,wall_median,inf_median,lpips,psnr,clip\n")
    for s in summary:
        f.write(",".join(str(s[k]) for k in ("family", "tag", "steps", "n", "wall_median", "inf_median", "lpips", "psnr", "clip")) + "\n")

# contact sheets: one per (family, prompt, seed); columns = tags
os.makedirs(f"{out}/sheets", exist_ok=True)
for fam, tags in by.items():
    keys = sorted({k for t in tags.values() for k in t})
    order = [t for t in tags]
    for key in keys:
        cells = [(t, tags[t][key]) for t in order if key in tags[t]]
        if len(cells) < 2:
            continue
        S = 384
        sheet = Image.new("RGB", (S * min(len(cells), 5), (S + 18) * ((len(cells) + 4) // 5)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (t, r) in enumerate(cells):
            x, y = S * (i % 5), (S + 18) * (i // 5)
            sheet.paste(load(r).resize((S, S)), (x, y + 18))
            d.text((x + 4, y + 3), f"{t}  {r['wall']:.1f}s", fill="black")
        sheet.save(f"{out}/sheets/{fam}_{key[0]}_s{key[1]}.jpg", quality=88)
print("SCORE DONE")
