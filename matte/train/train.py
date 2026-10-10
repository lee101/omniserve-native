"""Train a small matte refiner: (image, BiRefNet probability) -> (alpha, clean foreground).

    python3 matte/train/train.py --iters 15000 --out ~/matte-data/runs/r1

Alpha is predicted as a logit residual on the network probability, foreground as a bounded
residual on the image, so an untrained model returns BiRefNet's alpha and the observed colours.
Held-out subjects (every 10th) are scored against BiRefNet and the hand-built K pipeline.
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workers"))
os.environ.setdefault("OMATTE_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build-cuda", "libomatte.so"))
from matte_refine import Refiner  # noqa: E402

DATA = os.path.expanduser(os.environ.get("MATTE_DATA", "~/matte-data"))


def subjects_split():
    ids = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(DATA, "samples", "*.json"))})
    val = ids[::10]
    return [i for i in ids if i not in val], val


def load_sample(name):
    base = os.path.join(DATA, "samples", name)
    meta = json.load(open(base + ".json"))
    img = np.asarray(Image.open(base + ".webp").convert("RGB"))
    prob = np.asarray(Image.open(base + ".png"))
    sid = meta["subject"]
    a = np.asarray(Image.open(os.path.join(DATA, "teacher", sid + "_alpha.png")))
    f = np.asarray(Image.open(os.path.join(DATA, "teacher", sid + "_fg.png")).convert("RGB"))
    if meta["flip"]:
        a, f = a[:, ::-1], f[:, ::-1]
    return img, prob, a, f


class Crops(torch.utils.data.Dataset):
    def __init__(self, names, crop, n):
        self.names, self.crop, self.n = names, crop, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random()
        orig = [n for n in self.names if n.endswith("_0")] if not hasattr(self, "_orig") else self._orig
        self._orig = orig
        pool = orig if orig and rng.random() < 0.3 else self.names
        img, prob, a, f = load_sample(rng.choice(pool))
        t = [torch.from_numpy(np.ascontiguousarray(x)).float() / 255 for x in (img, prob, a, f)]
        x = torch.cat([t[0].permute(2, 0, 1), t[1][None], t[2][None], t[3].permute(2, 0, 1)], 0)
        s = rng.uniform(max(0.5, self.crop / min(x.shape[1:])), 1.0)
        if s < 0.97:
            x = F.interpolate(x[None], scale_factor=s, mode="bilinear", align_corners=False, antialias=True)[0]
        H, W = x.shape[1:]
        c = min(self.crop, H, W)
        soft = ((x[4] > 0.02) & (x[4] < 0.98)).nonzero()
        if len(soft) and rng.random() < 0.85:
            y, xx = soft[rng.randrange(len(soft))].tolist()
            y0 = min(max(0, y - c // 2 + rng.randrange(-c // 4, c // 4 + 1)), H - c)
            x0 = min(max(0, xx - c // 2 + rng.randrange(-c // 4, c // 4 + 1)), W - c)
        else:
            y0, x0 = rng.randrange(H - c + 1), rng.randrange(W - c + 1)
        x = x[:, y0:y0 + c, x0:x0 + c].contiguous()
        g = rng.uniform(0.8, 1.25)
        x[0:3] = x[0:3] ** g
        x[5:8] = x[5:8] ** g
        return x


def grad(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def losses(alpha, fg, a, f):
    la = (alpha - a).abs().mean()
    (ay, ax), (gy, gx) = grad(alpha), grad(a)
    lg = (ay - gy).abs().mean() + (ax - gx).abs().mean()
    w = (a > 0.01).float()
    lf = ((fg - f).abs().mean(1, keepdim=True) * w).sum() / w.sum().clamp_min(1)
    return la, lg, lf


@torch.no_grad()
def evaluate(model, names, dev, limit=60, baselines=False):
    """Alpha SAD/grad/edge and foreground error on held-out subjects, by backdrop kind."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workers"))
    os.environ.setdefault("OMATTE_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build-cuda", "libomatte.so"))
    import matte_refine as mr
    model.eval()
    acc = {}
    for name in names[:limit]:
        kind = json.load(open(os.path.join(DATA, "samples", name + ".json")))["kind"]
        img, prob, a, f = (torch.from_numpy(np.ascontiguousarray(x)).float().to(dev) / 255 for x in load_sample(name))
        img, f = img.permute(2, 0, 1)[None], f.permute(2, 0, 1)[None]
        prob, a = prob[None, None], a[None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            alpha, fg = model(img, prob)
        cands = {"ref": (alpha.float(), fg.float())}
        if baselines:
            cands["net"] = (prob, img)
            ka = mr.refine_alpha(img[0], prob[0, 0], key=True)
            kf, _ = mr.recolor(img[0], ka, max(ka.shape))
            cands["k"] = (ka[None, None], kf[None])
        edge = (a > 0.02) & (a < 0.98)
        w = (a > 0.01).float()
        for tag, (al, ff) in cands.items():
            (ay, ax), (gy, gx) = grad(al), grad(a)
            row = {"sad": float((al - a).abs().sum() / 1000),
                   "grad": float(((ay - gy).abs().mean() + (ax - gx).abs().mean()) * 1000),
                   "edge": float((al - a).abs()[edge].mean() * 100),
                   "fg": float(((ff - f).abs().mean(1, keepdim=True) * w).sum() / w.sum() * 100)}
            for k, v in row.items():
                for grp in ("all", kind):
                    acc.setdefault(f"{grp}/{tag}_{k}", []).append(v)
    model.train()
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(acc.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(DATA, "runs", time.strftime("r%m%d-%H%M")))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    tr_ids, va_ids = subjects_split()
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(DATA, "samples", "*.json")))
    tr = [n for n in names if n.split("_")[0] in tr_ids]
    va = [n for n in names if n.split("_")[0] in va_ids]
    print(f"train {len(tr)} samples / {len(tr_ids)} subjects, val {len(va)} / {len(va_ids)}", flush=True)
    dl = torch.utils.data.DataLoader(Crops(tr, a.crop, a.iters * a.bs), batch_size=a.bs, num_workers=a.workers,
                                     persistent_workers=True, prefetch_factor=4)
    model = Refiner().to(dev).to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    print("val@0", json.dumps(evaluate(model, va, dev, baselines=True)), flush=True)
    t0 = time.time()
    run = []
    for it, x in enumerate(dl, 1):
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
        img, prob, ga, gf = x[:, 0:3], x[:, 3:4], x[:, 4:5], x[:, 5:8]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            alpha, fg = model(img, prob)
        la, lg, lf = losses(alpha.float(), fg.float(), ga, gf)
        loss = la + lg + lf
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run.append([la.item(), lg.item(), lf.item()])
        if it % 200 == 0:
            m = np.mean(run, 0); run = []
            print(f"it {it} a={m[0]:.4f} g={m[1]:.4f} f={m[2]:.4f} {(time.time() - t0) / it:.3f}s/it", flush=True)
        if it % 2500 == 0 or it == a.iters:
            torch.save(model.state_dict(), os.path.join(a.out, "refiner.pt"))
            print(f"val@{it}", json.dumps(evaluate(model, va, dev)), flush=True)
        if it >= a.iters:
            break


if __name__ == "__main__":
    main()
