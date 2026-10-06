"""Side by side on real renders: BiRefNet bilinear, hand-built K path, learned refiner.

    python3 matte/train/compare.py ~/matte-data/runs/v1/refiner.pt out.jpg img1 img2 ...
"""
import os, sys, time
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, args = sys.argv[:1], sys.argv[1:]
from teacher import birefnet, dev, load, mr  # noqa: E402
from PIL import Image


@torch.inference_mode()
def main(weights, out, paths):
    mr.load_refiner(weights, dev)
    navy = torch.tensor([0.08, 0.10, 0.22], device=dev)[:, None, None]
    rows, tim = [], {"k": [], "learned": []}
    for p in paths:
        img = load(p)
        H, W = img.shape[1:]
        s = 1024 / max(H, W)
        prob_lo = birefnet(F.interpolate(img[None], size=(round(H * s), round(W * s)), mode="area")[0])
        net = F.interpolate(prob_lo[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        torch.cuda.synchronize(); t = time.perf_counter()
        ka = mr.refine_alpha(img, prob_lo, key=True)
        kf, _ = mr.recolor(img, ka, 1024)
        torch.cuda.synchronize(); tim["k"].append(time.perf_counter() - t); t = time.perf_counter()
        la, lf = mr.refine_learned(img, prob_lo)
        torch.cuda.synchronize(); tim["learned"].append(time.perf_counter() - t)
        tiles = [img, net * img + (1 - net) * navy, ka * kf + (1 - ka) * navy, la * lf + (1 - la) * navy]
        row = torch.cat([F.interpolate(x[None], size=(512, round(512 * W / H)), mode="area")[0] for x in tiles], 2)
        rows.append(row)
    w = max(r.shape[2] for r in rows)
    sheet = torch.cat([F.pad(r, (0, w - r.shape[2])) for r in rows], 1)
    Image.fromarray((sheet.clamp(0, 1) * 255 + .5).byte().permute(1, 2, 0).cpu().numpy()).save(out, quality=88)
    print({k: f"{1000 * sum(v[1:]) / max(1, len(v) - 1):.0f} ms" for k, v in tim.items()})


main(args[0], args[1], args[2:])
