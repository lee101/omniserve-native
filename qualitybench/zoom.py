import sys
from PIL import Image, ImageDraw
box = tuple(int(v) for v in sys.argv[1].split(","))
out = sys.argv[2]
items = sys.argv[3:]
w = box[2] - box[0]; s = 3 if w <= 200 else 2
cols = 3
rows = (len(items) + cols - 1) // cols
cw = w * s
sheet = Image.new("RGB", (cols * cw, rows * (cw + 14)), "white")
d = ImageDraw.Draw(sheet)
for i, p in enumerate(items):
    im = Image.open(p).convert("RGB")
    c = im.crop(box).resize((cw, cw), Image.NEAREST)
    x, y = (i % cols) * cw, (i // cols) * (cw + 14)
    sheet.paste(c, (x, y + 14)); d.text((x + 3, y + 1), p.split("/")[-2] if "out/" in p else p, fill="black")
sheet.save(out)
