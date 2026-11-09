#!/usr/bin/env python3
"""No-reference image/face quality per variant tag. Walks <dir>/**/<tag>/<key>_s<seed>.png (or --glob), detects faces
(facexlib RetinaFace), scores face crops with pyiqa topiq_nr-face + Laplacian sharpness, whole image with topiq_nr, and
CLIP-to-prompt when results.jsonl carries `text`. Writes <dir>/face_scores.jsonl and prints a per-(group, tag) table.

    uv run --with torch --with torchvision --with pyiqa --with facexlib --with open_clip_torch --with opencv-python-headless \
        qualitybench/face_score.py qualitybench/out/lora-ab
"""
import collections, glob, json, os, sys
import cv2, numpy as np, torch

d = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
import pyiqa
from facexlib.detection import init_detection_model
det = init_detection_model("retinaface_resnet50", half=False, device=dev)
face_iqa = pyiqa.create_metric("topiq_nr-face", device=dev)
img_iqa = pyiqa.create_metric("topiq_nr", device=dev)
import open_clip
cm, _, cpre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
ctok = open_clip.get_tokenizer("ViT-B-32")
from PIL import Image

texts = {}
rp = f"{d}/results.jsonl"
if os.path.exists(rp):
    for r in map(json.loads, open(rp)):
        if "text" in r:
            texts[(r.get("style", ""), r["tag"], r["prompt"], r["seed"])] = r["text"]


def t(im):
    return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(dev)


done = {}
op = f"{d}/face_scores.jsonl"
if os.path.exists(op):
    done = {r["file"]: r for r in map(json.loads, open(op))}
out = open(op, "a")
files = sorted(glob.glob(f"{d}/**/*.png", recursive=True))
for f in files:
    rel = os.path.relpath(f, d)
    if rel in done or "/sheets/" in f:
        continue
    parts = rel.split("/")
    tag, name = parts[-2], parts[-1][:-4]
    group = "/".join(parts[:-2])
    key, seed = name.rsplit("_s", 1)
    im = cv2.imread(f)
    H, W = im.shape[:2]
    with torch.no_grad():
        boxes = det.detect_faces(im, 0.8)
        faces = []
        for b in boxes:
            x0, y0, x1, y1 = b[:4]
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.3
            if s < 24:
                continue
            X0, Y0, X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
            crop = cv2.resize(im[Y0:Y1, X0:X1], (512, 512), interpolation=cv2.INTER_CUBIC)
            lap = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            try:
                fq = float(face_iqa(t(crop)))
            except AssertionError:
                fq = None
            faces.append(dict(px=int(s), iqa=fq, lap=lap))
        r = dict(file=rel, group=group, tag=tag, prompt=key, seed=int(seed), n_faces=len(faces), faces=faces,
                 img_iqa=float(img_iqa(t(cv2.resize(im, (768, int(768 * H / W)))))))
        txt = texts.get((group, tag, key, int(seed)))
        if txt:
            ie = cm.encode_image(cpre(Image.open(f).convert("RGB"))[None].to(dev)); te = cm.encode_text(ctok([txt[:300]]).to(dev))
            r["clip"] = float((ie / ie.norm() * (te / te.norm())).sum())
    out.write(json.dumps(r) + "\n"); out.flush()
    done[rel] = r

agg = collections.defaultdict(list)
for r in done.values():
    agg[(r["group"], r["tag"])].append(r)
print(f"{'group':12s} {'tag':18s} {'n':>3s} {'faces':>5s} {'faceIQA':>7s} {'faceLap':>8s} {'imgIQA':>6s} {'clip':>6s}")
for (g, tg), rs in sorted(agg.items()):
    fi = [x["iqa"] for r in rs for x in r["faces"] if x["iqa"] is not None]
    fl = [x["lap"] for r in rs for x in r["faces"]]
    cl = [r["clip"] for r in rs if "clip" in r]
    m = lambda v: f"{np.mean(v):.3f}" if v else "  -  "
    print(f"{g:12s} {tg:18s} {len(rs):3d} {sum(r['n_faces'] for r in rs):5d} {m(fi):>7s} {np.mean(fl) if fl else 0:8.0f} {m([r['img_iqa'] for r in rs]):>6s} {m(cl):>6s}")
