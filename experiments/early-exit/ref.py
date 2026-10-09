#!/usr/bin/env python3
"""ref.py DIR : LPIPS between the prod EasyCache render and the same request with EasyCache off (error prod already accepts)."""
import json, sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image


def t(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).permute(2, 0, 1)[None] / 127.5 - 1


root = Path(sys.argv[1])
lp = lpips.LPIPS(net="alex", verbose=False).eval()
res = {}
for l in open(root / "log.jsonl"):
    m = json.loads(l)
    if "nocache_of" not in m or not (root / m["nocache_of"] / "final.png").exists():
        continue
    with torch.no_grad():
        res.setdefault(m["kind"], []).append(float(lp(t(root / m["id"] / "final.png"), t(root / m["nocache_of"] / "final.png"))))
for k, v in res.items():
    v = np.array(v)
    print(f"{k}: n={len(v)} easycache-vs-off LPIPS mean {v.mean():.4f} median {np.median(v):.4f} p90 {np.quantile(v, 0.9):.4f} max {v.max():.4f}")
