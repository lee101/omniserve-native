"""Synthesise training composites from teacher cutouts and run BiRefNet on each.

    python3 matte/train/build.py --per 8

samples/<sid>_<j>.webp   composite I (q95)
samples/<sid>_<j>.png    BiRefNet probability (8-bit)
samples/<sid>_<j>.json   {"subject", "flip", "kind", ...}; GT is teacher/<sid>_{alpha,fg}.png (flipped)

Backdrops: generated scenes, solid/gradient colours, chroma screens. Screens add spill
F + s * K near the edge with a random strength, so removing it has exact ground truth.
"""
import argparse, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TEACHER_NO_MAIN", "1")
from teacher import DATA, birefnet, dev, load, mr  # noqa: E402

SCREEN = {"green": (0.15, 0.75, 0.12), "blue": (0.05, 0.35, 0.95)}


def gt(sid, flip):
    a = torch.from_numpy(np.asarray(Image.open(os.path.join(DATA, "teacher", sid + "_alpha.png")))).to(dev).float() / 255
    f = load(os.path.join(DATA, "teacher", sid + "_fg.png"))
    if flip:
        a, f = a.flip(-1), f.flip(-1)
    return a, f


def backdrop(rng, H, W, scenes):
    u = rng.random()
    yy = torch.linspace(0, 1, H, device=dev)[:, None].expand(H, W)
    if u < 0.5 and scenes:
        b = load(rng.choice(scenes))
        s = max(H / b.shape[1], W / b.shape[2]) * rng.uniform(1.0, 1.6)
        b = F.interpolate(b[None], scale_factor=s, mode="bilinear", align_corners=False, antialias=True)[0]
        y0, x0 = rng.randrange(b.shape[1] - H + 1), rng.randrange(b.shape[2] - W + 1)
        return b[:, y0:y0 + H, x0:x0 + W].contiguous(), "scene", None
    if u < 0.7:
        c0 = torch.tensor([rng.random() for _ in range(3)], device=dev)[:, None, None]
        c1 = torch.tensor([rng.random() for _ in range(3)], device=dev)[:, None, None]
        return (c0 * (1 - yy) + c1 * yy).clamp(0, 1), "colour", None
    key = rng.choice(list(SCREEN))
    k = torch.tensor(SCREEN[key], device=dev)
    k = (k * rng.uniform(0.75, 1.1) + torch.tensor([rng.uniform(-0.06, 0.06) for _ in range(3)], device=dev)).clamp(0, 1)
    b = k[:, None, None] * (1 - 0.25 * rng.random() * yy) + 0.015 * torch.randn(1, H, W, device=dev)
    return b.clamp(0, 1), "screen", (key, k)


def build_one(sid, j, scenes, out):
    rng = random.Random(hash((sid, j)) & 0xFFFFFFFF)
    if j == 0:
        # The render itself: real screen light, real backlit halos, with the teacher's clean GT.
        img = load(os.path.join(DATA, "subjects", sid + ".webp"))
        save_sample(img, {"subject": sid, "flip": False, "kind": "orig"}, f"{sid}_{j}", out)
        return
    flip = rng.random() < 0.5
    a, f = gt(sid, flip)
    H, W = a.shape
    b, kind, screen = backdrop(rng, H, W, scenes)
    fin = f
    meta = {"subject": sid, "flip": flip, "kind": kind}
    if screen is not None:
        r = max(4, H // rng.choice([8, 12, 16, 24]))
        prox = mr._box((1 - a)[None, None], r)[0, 0]
        prox = (prox * 2).clamp(0, 1) * (a > 0.002).float()
        # Up to backlit-halo strength: a rim lit by the screen goes past the key colour itself.
        strength = rng.uniform(0.15, 1.6) if rng.random() < 0.5 else rng.uniform(0.15, 0.8)
        fin = (f + strength * prox * screen[1][:, None, None]).clamp(0, 1)
        meta.update(key=screen[0], spill=round(strength, 3))
    elif kind == "scene" and rng.random() < 0.5:
        cast = b.mean((1, 2))
        r = max(4, H // 24)
        prox = (mr._box((1 - a)[None, None], r)[0, 0] * 2).clamp(0, 1)
        strength = rng.uniform(0.05, 0.3)
        fin = (f * (1 - strength * prox) + strength * prox * cast[:, None, None]).clamp(0, 1)
        meta.update(cast=round(strength, 3))
    img = (a * fin + (1 - a) * b).clamp(0, 1)
    save_sample(img, meta, f"{sid}_{j}", out)


def save_sample(img, meta, name, out):
    prob = birefnet(img)
    Image.fromarray((img * 255 + .5).byte().permute(1, 2, 0).cpu().numpy()).save(os.path.join(out, name + ".webp"), quality=95, method=4)
    Image.fromarray((prob * 255 + .5).byte().cpu().numpy()).save(os.path.join(out, name + ".png"))
    json.dump(meta, open(os.path.join(out, name + ".json"), "w"))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=8)
    a = ap.parse_args()
    tdir = os.path.join(DATA, "teacher")
    out = os.path.join(DATA, "samples")
    os.makedirs(out, exist_ok=True)
    sdir = os.path.join(DATA, "scenes")
    scenes = [os.path.join(sdir, n) for n in sorted(os.listdir(sdir)) if n.endswith(".webp")] if os.path.isdir(sdir) else []
    kept = sorted(n[:-10] for n in os.listdir(tdir) if n.endswith("_alpha.png"))
    made = 0
    for sid in kept:
        for j in range(a.per):
            if os.path.exists(os.path.join(out, f"{sid}_{j}.json")):
                continue
            build_one(sid, j, scenes, out)
            made += 1
    print(f"build: {made} new samples from {len(kept)} subjects, {len(scenes)} scenes")


if __name__ == "__main__":
    main()
