"""Clean (alpha, foreground) ground truth from chroma-screen renders.

    python3 matte/train/teacher.py            # all subjects without a teacher output
    python3 matte/train/teacher.py --preview  # also writes previews/ grids

Per subject: BiRefNet probability, colour guided filter at native resolution, backdrop B from
omatte, then a strict colour-difference key bounded by the network mask:
    alpha = min(dilate(net), max(key, erode(net)))
The eroded network core keeps spill-lit opaque hair opaque; the key decides the soft edge and the
gaps inside hair. Foreground: omatte solve with that alpha, then full-strength vector despill
(prompts keep the key colour out of subjects). Samples whose numbers look wrong are rejected.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "workers"))
os.environ.setdefault("OMATTE_LIB", os.path.join(ROOT, "..", "..", "build-cuda", "libomatte.so"))
import matte_refine as mr
from transformers import AutoModelForImageSegmentation

DATA = os.path.expanduser(os.environ.get("MATTE_DATA", "~/matte-data"))
dev = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
_model = None


def model():
    global _model
    if _model is None:
        _model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True,
                                                               dtype=torch.float16).to(dev).eval()
    return _model


@torch.inference_mode()
def birefnet(img, size=1024):
    x = F.interpolate(img[None], size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    x = ((x - MEAN) / STD).half()
    with torch.autocast("cuda", dtype=torch.float16):
        p = model()(x)
    p = p[-1] if isinstance(p, (list, tuple)) else p
    p = torch.sigmoid(p.float())
    return F.interpolate(p, size=img.shape[1:], mode="bilinear", align_corners=False)[0, 0]


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).to(dev).permute(2, 0, 1).float() / 255


def save_gray(x, path):
    Image.fromarray((x.clamp(0, 1) * 255 + .5).byte().cpu().numpy()).save(path)


def save_rgb(x, path, **kw):
    Image.fromarray((x.clamp(0, 1) * 255 + .5).byte().permute(1, 2, 0).cpu().numpy()).save(path, **kw)


def pool(x, r, op="max"):
    x = x[None, None]
    y = F.max_pool2d(x, 2 * r + 1, 1, r) if op == "max" else -F.max_pool2d(-x, 2 * r + 1, 1, r)
    return y[0, 0]


@torch.inference_mode()
def teach(img, key_name):
    H, W = img.shape[1:]
    prob = birefnet(img)
    net = mr.guided_upsample(img, prob[None], img, 4, 1e-4)[0].clamp(0, 1)
    _, bg = mr._solve(img, net)
    k = 1 if key_name == "green" else 2
    o = [c for c in range(3) if c != k]
    excess = img[k] - torch.maximum(img[o[0]], img[o[1]])
    excess_bg = (bg[k] - torch.maximum(bg[o[0]], bg[o[1]])).clamp_min(mr.KEYABLE_EXCESS)
    keyed = (1 - excess / excess_bg).clamp(0, 1)
    r = max(2, H // 128)
    deep = pool((net > 0.5).float(), 3 * r, "min")
    core = mr._box(deep[None, None], 2 * r)[0, 0] * net
    hull = pool(net, r, "max")
    alpha = torch.minimum(hull, torch.maximum(keyed, core))
    fg, bg2 = mr._solve(img, alpha)
    band = mr._edge_band(alpha, (H, W), max(4, H // 16))
    fg = mr.despill(fg, bg2, k, o, mr.despill_weight(alpha, band))
    soft = (alpha > 0.02) & (alpha < 0.98)
    stats = {
        "coverage": float((alpha > 0.5).float().mean()),
        "soft_share": float(soft.float().mean()),
        "key_net_disagree": float((keyed - net).abs()[(net > 0.02) & (net < 0.98)].mean()) if bool(((net > 0.02) & (net < 0.98)).any()) else 0.0,
        "residual_spill": float((mr.spill_map(fg, bg2.mean((1, 2)), k, o) * alpha).sum() / alpha.sum().clamp_min(1)),
        "bg_key_excess": float((bg[k] - torch.maximum(bg[o[0]], bg[o[1]]))[net < 0.05].mean()) if bool((net < 0.05).any()) else 0.0,
    }
    return alpha, fg, net, stats


def reject(st):
    if st["coverage"] < 0.03 or st["coverage"] > 0.85:
        return "coverage"
    if st["bg_key_excess"] < 0.2:
        return "backdrop not keyable"
    if st["residual_spill"] > 0.02:
        return "spill left"
    if st["key_net_disagree"] > 0.45:
        return "key vs net"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    src = os.path.join(DATA, "subjects")
    out = os.path.join(DATA, "teacher")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(DATA, "previews"), exist_ok=True)
    rows = []
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        meta = json.load(open(os.path.join(src, name)))
        sid = meta["id"]
        done = os.path.join(out, sid + ".json")
        if os.path.exists(done) and not a.redo:
            continue
        t = time.perf_counter()
        img = load(os.path.join(src, sid + ".webp"))
        alpha, fg, net, st = teach(img, meta["key"])
        why = reject(st)
        st.update(id=sid, key=meta["key"], rejected=why, ms=round((time.perf_counter() - t) * 1000))
        if not why:
            save_gray(alpha, os.path.join(out, sid + "_alpha.png"))
            save_rgb(fg, os.path.join(out, sid + "_fg.png"))
        if a.preview:
            navy = torch.tensor([0.08, 0.10, 0.22], device=dev)[:, None, None]
            comp = alpha * fg + (1 - alpha) * navy
            naive = net * img + (1 - net) * navy
            row = torch.cat([img, naive, comp, alpha[None].expand(3, -1, -1)], 2)
            save_rgb(F.interpolate(row[None], scale_factor=0.5, mode="area")[0], os.path.join(DATA, "previews", sid + ".jpg"), quality=88)
        json.dump(st, open(done, "w"))
        rows.append(st)
        print(json.dumps(st), flush=True)
    ok = sum(1 for r in rows if not r["rejected"])
    print(f"teacher: {ok}/{len(rows)} kept")


if __name__ == "__main__":
    main()
