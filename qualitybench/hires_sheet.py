#!/usr/bin/env python3
"""Center-crop comparison across hires variants at equal output scale: <out>/sheets/<key>_s<seed>.jpg."""
import os, sys
from PIL import Image, ImageDraw
out = sys.argv[1]
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(d for d in os.listdir(out) if os.path.isdir(f"{out}/{d}") and d != "sheets")
keys = sorted({f[:-4] for t in tags for f in os.listdir(f"{out}/{t}") if f.endswith(".png")})
os.makedirs(f"{out}/sheets", exist_ok=True)
C = 512
for k in keys:
    ims = []
    for t in tags:
        f = f"{out}/{t}/{k}.png"
        if not os.path.exists(f):
            continue
        im = Image.open(f).convert("RGB").resize((1536, 1536), Image.LANCZOS)  # common scale
        x, y = 768 - C // 2, int(1536 * 0.33) - C // 2
        ims.append((t, im.crop((x, y, x + C, y + C))))
    S = Image.new("RGB", (len(ims) * C, C + 20), "white"); d = ImageDraw.Draw(S)
    for i, (t, im) in enumerate(ims):
        d.text((i * C + 4, 3), t, fill="black"); S.paste(im, (i * C, 20))
    S.save(f"{out}/sheets/{k}.jpg", quality=90)
