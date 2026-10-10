#!/usr/bin/env python3
"""label.py DIR [DIR...] --out labels.npz : per (traj, step) exit error vs the full-run output.

Metrics against final.png: LPIPS(alex), PSNR, CLIP-L/14 image cosine; per-image CLIP text score and LAION aesthetic.
"""
import argparse, glob, json, os
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP = "/home/administrator/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
AES = "/home/administrator/.cache/huggingface/hub/models--trl-lib--ddpo-aesthetic-predictor/snapshots/cf4bd10345610935a80a5d3be9a3133c50f121e4/aesthetic-model.pth"


def load_aes(dev):
    sd = torch.load(AES, map_location="cpu")
    ws = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith("weight")}, key=lambda k: int(k.split(".")[1]))
    mods = []
    for i, k in enumerate(ws):
        w = sd[k + ".weight"]
        l = torch.nn.Linear(w.shape[1], w.shape[0]); l.weight.data = w; l.bias.data = sd[k + ".bias"]
        mods.append(l)
    def f(e):
        for m in mods:
            e = m(e)
        return e.squeeze(-1)
    for m in mods:
        m.to(dev)
    return f


def rgb(path):
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
    return np.asarray(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    clip = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP)
    aes = load_aes(dev)
    prev = {}
    if os.path.exists(a.out):
        z = np.load(a.out, allow_pickle=True)
        prev = {k: z[k] for k in z.files}
    rows = {k: list(v) for k, v in prev.items()} if prev else {k: [] for k in ["traj", "step", "lpips", "psnr", "clip_img", "clip_txt", "aes", "clip_txt_final", "aes_final"]}
    done = set(rows["traj"])
    for root in a.dirs:
        meta = {json.loads(l)["id"]: json.loads(l) for l in open(Path(root) / "log.jsonl")}
        for tid, m in meta.items():
            key = f"{Path(root).name}/{tid}"
            d = Path(root) / tid
            if key in done or not (d / "final.png").exists():
                continue
            ks = sorted(glob.glob(str(d / "k*.png")))
            steps = [int(Path(k).stem[1:]) for k in ks]
            imgs = [rgb(d / "final.png")] + [rgb(k) for k in ks]
            with torch.no_grad():
                t = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
                l = lp(t[1:], t[:1].expand(len(ks), -1, -1, -1)).flatten().cpu().numpy()
                mse = ((t[1:] - t[:1]) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
                psnr = (10 * torch.log10(4 / mse)).cpu().numpy()
                inp = proc(text=[m["prompt"]], images=[Image.fromarray(i) for i in imgs], return_tensors="pt", padding=True, truncation=True)
                ie = clip.visual_projection(clip.vision_model(pixel_values=inp["pixel_values"].to(dev, clip.dtype)).pooler_output).float()
                te = clip.text_projection(clip.text_model(input_ids=inp["input_ids"].to(dev), attention_mask=inp["attention_mask"].to(dev)).pooler_output).float()
                ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
                cimg = (ie[1:] @ ie[0]).cpu().numpy()
                ctxt = (ie @ te[0]).cpu().numpy() * 100
                ae = aes(ie).cpu().numpy()
            for i, s in enumerate(steps):
                rows["traj"].append(key); rows["step"].append(s); rows["lpips"].append(l[i]); rows["psnr"].append(psnr[i])
                rows["clip_img"].append(cimg[i]); rows["clip_txt"].append(ctxt[i + 1]); rows["aes"].append(ae[i + 1])
                rows["clip_txt_final"].append(ctxt[0]); rows["aes_final"].append(ae[0])
    np.savez(a.out, **{k: np.array(v) for k, v in rows.items()})
    print(len(rows["traj"]), "rows")


if __name__ == "__main__":
    main()
