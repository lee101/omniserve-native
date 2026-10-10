#!/usr/bin/env python3
"""sheet.py --root DIR --items 'qwen/qt00012@17,...' --out sheet.jpg : rows of [full run | exit image | |diff|x4]."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(p, size):
    im = Image.open(p).convert("RGB")
    return im.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--labels", default="")
    a = ap.parse_args()
    items = [i for i in a.items.split(",") if i]
    labels = a.labels.split("|") if a.labels else [""] * len(items)
    S = a.size
    sheet = Image.new("RGB", (3 * S, len(items) * (S + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for r, (it, lab) in enumerate(zip(items, labels)):
        key, step = it.split("@")
        jd = Path(a.root) / key
        full = load(jd / "final.png", S)
        ex = load(jd / f"k{int(step):02d}.png", S) if (jd / f"k{int(step):02d}.png").exists() else full
        diff = Image.fromarray(np.clip(np.abs(np.asarray(full, np.int16) - np.asarray(ex, np.int16)) * 4, 0, 255).astype(np.uint8))
        y = r * (S + 18)
        for c, im in enumerate([full, ex, diff]):
            sheet.paste(im, (c * S, y + 18))
        d.text((4, y + 3), f"{it} {lab}", fill="black")
    sheet.save(a.out, quality=88)


if __name__ == "__main__":
    main()
