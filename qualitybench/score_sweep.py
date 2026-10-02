#!/usr/bin/env python3
import base64, collections, csv, io, json, os, sys
import numpy as np, torch, lpips, open_clip
from PIL import Image

out = sys.argv[1]
prompts = json.load(open(os.environ.get("SWEEP_PROMPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_prompts.json")))
REF = "base30_dense"
rows = [json.loads(l) for l in open(f"{out}/results.jsonl")]
by = collections.defaultdict(dict)
meta = {}
for r in rows:
    by[r["tag"]][r["prompt"]] = r
    meta[r["tag"]] = r

torch.set_num_threads(max(1, os.cpu_count() // 2))
net = lpips.LPIPS(net="alex", verbose=False)
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()


def load(p):
    return Image.open(p).convert("RGB")


def t(im):
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()[None] / 127.5 - 1


def sharp(im):
    g = np.asarray(im.convert("L"), float)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def nyq(im):
    g = np.asarray(im.convert("L").crop((256, 256, 768, 768)), float)
    g = g - g.mean()
    w = np.hanning(512)[:, None] * np.hanning(512)[None]
    F = np.abs(np.fft.fftshift(np.fft.fft2(g * w))) ** 2
    y, x = np.mgrid[:512, :512]
    r = np.maximum(abs(y - 256), abs(x - 256)) / 512
    return float(F[r > 0.45].sum() / F.sum())


with torch.no_grad():
    text = {k: model.encode_text(tok([v])) for k, v in prompts.items()}
    text = {k: v / v.norm(dim=-1, keepdim=True) for k, v in text.items()}

per = []
for tag, d in by.items():
    for pk, r in d.items():
        p = next((q for q in (f"{out}/{tag}/{pk}.webp", f"{out}/{tag}/{pk}.png") if os.path.exists(q)), f"{out}/{tag}/{pk}.png")
        refp = next((q for q in (f"{out}/{REF}/{pk}.webp", f"{out}/{REF}/{pk}.png") if os.path.exists(q)), f"{out}/{REF}/{pk}.png")
        if not os.path.exists(p):
            continue
        im = load(p)
        with torch.no_grad():
            e = model.encode_image(pre(im)[None])
            e = e / e.norm(dim=-1, keepdim=True)
            clip = float((e @ text[pk].T).item())
            lp = psnr = None
            if os.path.exists(refp) and tag != REF:
                ref = load(refp)
                lp = float(net(t(ref), t(im)).item())
                mse = float(((t(ref) - t(im)) ** 2).mean().item())
                psnr = 10 * np.log10(4 / (mse + 1e-9))
        per.append(dict(tag=tag, prompt=pk, wall=r["wall"], clip=clip, lpips=lp, psnr=psnr, sharp=sharp(im), nyq=nyq(im)))
        print(tag, pk, f"clip={clip:.3f}", flush=True)

json.dump(per, open(f"{out}/per_image.json", "w"), indent=1)
agg = []
refs = {p["prompt"]: p for p in per if p["tag"] == REF}
for tag in by:
    ps = [p for p in per if p["tag"] == tag]
    if not ps:
        continue
    m = lambda k: float(np.mean([p[k] for p in ps if p[k] is not None])) if any(p[k] is not None for p in ps) else None
    sr = [p["sharp"] / refs[p["prompt"]]["sharp"] for p in ps if p["prompt"] in refs]
    cd = [p["clip"] - refs[p["prompt"]]["clip"] for p in ps if p["prompt"] in refs]
    agg.append(dict(tag=tag, server=meta[tag]["server"], steps=meta[tag]["steps"], extra=meta[tag]["extra"], n=len(ps),
                    wall=m("wall"), lpips=m("lpips"), psnr=m("psnr"), clip=m("clip"),
                    clip_delta=float(np.mean(cd)) if cd else None, sharp_ratio=float(np.mean(sr)) if sr else None,
                    nyq=m("nyq")))
agg.sort(key=lambda a: a["wall"])
json.dump(agg, open(f"{out}/summary.json", "w"), indent=1)
with open(f"{out}/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "server", "steps", "n", "sec_per_img", "lpips_vs_dense30", "psnr", "clip", "clip_delta", "sharp_ratio", "nyquist"])
    for a in agg:
        w.writerow([a["tag"], a["server"], a["steps"], a["n"], f"{a['wall']:.1f}",
                    "" if a["lpips"] is None else f"{a['lpips']:.3f}", "" if a["psnr"] is None else f"{a['psnr']:.1f}",
                    f"{a['clip']:.4f}", "" if a["clip_delta"] is None else f"{a['clip_delta']:+.4f}",
                    "" if a["sharp_ratio"] is None else f"{a['sharp_ratio']:.2f}", f"{a['nyq']:.1e}"])
print(f"{'tag':26s} {'s/img':>6s} {'lpips':>6s} {'psnr':>5s} {'clip':>6s} {'dclip':>7s} {'sharp':>5s}")
for a in agg:
    f = lambda v, s: "   -  " if v is None else s % v
    print(f"{a['tag']:26s} {a['wall']:6.1f} {f(a['lpips'], '%6.3f')} {f(a['psnr'], '%5.1f')} {a['clip']:6.3f} {f(a['clip_delta'], '%+7.4f')} {f(a['sharp_ratio'], '%5.2f')}")
