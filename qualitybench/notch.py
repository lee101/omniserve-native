import numpy as np
from PIL import Image
B = np.array([-1, 6, -15, 20, -15, 6, -1], np.float32) / 64

def notch(a):
    a = a.astype(np.float32) / 255
    p = np.pad(a, ((3, 3), (3, 3), (0, 0)), mode="edge")
    h, w = a.shape[:2]
    bx = sum(B[i] * p[3:3 + h, i:i + w] for i in range(7))
    by = sum(B[i] * p[i:i + h, 3:3 + w] for i in range(7))
    bxy = sum(B[i] * B[j] * p[i:i + h, j:j + w] for i in range(7) for j in range(7))
    return (np.clip(a - bx - by + bxy, 0, 1) * 255 + .5).astype(np.uint8)

def crop(im, box=(250, 300, 506, 556), s=2):
    return im.crop(box).resize(((box[2] - box[0]) * s, (box[3] - box[1]) * s), Image.NEAREST)

if __name__ == "__main__":
    import sys
    src, out = sys.argv[1], sys.argv[2]
    im = Image.open(src).convert("RGB")
    n = Image.fromarray(notch(np.asarray(im)))
    n.save(out)
    sheet = Image.new("RGB", (1024, 512))
    sheet.paste(crop(im), (0, 0)); sheet.paste(crop(n), (512, 0))
    sheet.save(out.replace(".png", "_zoom.png"))
