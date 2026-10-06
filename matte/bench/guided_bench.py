import os, sys, time, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "workers"))
os.environ.setdefault("OMATTE_LIB", os.path.join(ROOT, "..", "..", "build-cuda", "libomatte.so"))
import omatte
from transformers import AutoModelForImageSegmentation

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True,
                                                      dtype=torch.float16).to(dev).eval()
model = model.to(memory_format=torch.channels_last)


def sync():
    torch.cuda.synchronize()


@torch.inference_mode()
def birefnet(img, size):
    x = F.interpolate(img[None], size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    x = ((x - MEAN) / STD).half().contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16):
        p = model(x)
    p = p[-1] if isinstance(p, (list, tuple)) else p
    return torch.sigmoid(p.float())[0, 0]


def box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def guided_coeffs(I, p, r, eps):
    """Colour guided filter (He et al.): per-pixel linear map p ~ A^T I + b, box-averaged."""
    C = p.shape[0]
    I4, p4 = I[None], p[None]
    mI, mp = box(I4, r)[0], box(p4, r)[0]
    II = torch.stack([I[i] * I[j] for i in range(3) for j in range(3)])[None]
    Ip = torch.stack([I[i] * p[c] for i in range(3) for c in range(C)])[None]
    var = (box(II, r)[0] - torch.stack([mI[i] * mI[j] for i in range(3) for j in range(3)]))
    cov = (box(Ip, r)[0] - torch.stack([mI[i] * mp[c] for i in range(3) for c in range(C)]))
    h, w = I.shape[1:]
    S = var.permute(1, 2, 0).reshape(h, w, 3, 3) + eps * torch.eye(3, device=I.device)
    A = torch.linalg.solve(S, cov.permute(1, 2, 0).reshape(h, w, 3, C))
    b = mp - torch.einsum("hwic,ihw->chw", A, mI)
    A = A.reshape(h, w, 3 * C).permute(2, 0, 1)
    return box(A[None], r)[0], box(b[None], r)[0]


def guided_upsample(I_lo, p_lo, I_hi, r, eps):
    A, b = guided_coeffs(I_lo, p_lo, r, eps)
    H, W = I_hi.shape[1:]
    A = F.interpolate(A[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    b = F.interpolate(b[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    C = p_lo.shape[0]
    A = A.reshape(3, C, H, W)
    return (A * I_hi[:, None]).sum(0) + b


def despill(f, a, b, r=24):
    """Pull the key colour out of the edge band only: where the subject is soft or near the edge,
    the channel the backdrop is strongest in is limited to the mean of the other two."""
    key = b.mean((1, 2))
    k = int(key.argmax())
    o = [c for c in range(3) if c != k]
    band = 1 - box(a[None, None], r)[0, 0]
    band = (band * 4).clamp(0, 1) * (a < 0.995).float().maximum(band)
    limit = torch.maximum(f[o[0]], f[o[1]])
    excess = (f[k] - limit).clamp_min(0) * band
    f = f.clone()
    f[k] = f[k] - excess
    return f


def keyed(img, I_lo, a_net, f_lo, b_lo, lo, r, eps, band_px=64):
    """Colour-difference key from the full-resolution pixels, trusted only near the subject's edge.
    The network alpha calls a backlit, spill-lit hair halo opaque; on a keyable backdrop the
    backdrop-channel excess (G - max(R, B) for green) measures the real coverage per pixel."""
    H, W = img.shape[1:]
    b_hi = up(b_lo, H, W)
    key = b_lo.mean((1, 2))
    k = int(key.argmax())
    o = [c for c in range(3) if c != k]
    d = img[k] - torch.maximum(img[o[0]], img[o[1]])
    d_b = b_hi[k] - torch.maximum(b_hi[o[0]], b_hi[o[1]])
    keyable = (d_b > 0.15).float()
    a_key = (1 - d / d_b.clamp_min(0.15)).clamp(0, 1)
    rr = max(1, band_px * H // 2048)
    outside = (a_net < 0.5).float()[None, None]
    band = (F.max_pool2d(outside, 2 * rr + 1, stride=1, padding=rr)[0, 0] > 0).float()
    band = box(band[None, None], rr // 2)[0, 0] * keyable
    a = a_net - band * (a_net - torch.minimum(a_net, a_key))
    f_lo2, b_lo2 = fb(I_lo, down(a[None], lo)[0])
    f = (img + (1 - a) * up(f_lo2 - b_lo2, H, W)).clamp(0, 1)
    excess = (f[k] - torch.maximum(f[o[0]], f[o[1]])).clamp_min(0) * band
    f = f.clone()
    f[k] = f[k] - excess
    return a, f


def fb(img_chw, alpha):
    f, b = omatte.estimate_foreground_torch(img_chw.permute(1, 2, 0).contiguous(), alpha.contiguous(),
                                            return_background=True)
    return f.permute(2, 0, 1), b.permute(2, 0, 1)


def down(x, s):
    return F.interpolate(x[None], size=(s, s), mode="area")[0]


def up(x, H, W):
    return F.interpolate(x[None], size=(H, W), mode="bilinear", align_corners=False)[0]


def run(method, img, lo=1024, r=None, eps=None):
    H, W = img.shape[1:]
    t = {}
    sync(); t0 = time.perf_counter()
    if method == "N":
        a = birefnet(img, H); sync(); t["seg"] = time.perf_counter() - t0
        f, _ = fb(img, a)
    else:
        a_lo = birefnet(img, lo); sync(); t["seg"] = time.perf_counter() - t0
        if method == "A":
            a = up(a_lo[None], H, W)[0].clamp(0, 1)
            f, _ = fb(img, a)
        else:
            I_lo = down(img, lo)
            r = r or max(2, lo // 256)
            eps = eps or 1e-4
            a = guided_upsample(I_lo, a_lo[None], img, r, eps)[0].clamp(0, 1)
            a_lo2 = down(a[None], lo)[0]
            f_lo, b_lo = fb(I_lo, a_lo2)
            f_g = guided_upsample(I_lo, f_lo, img, r, eps * 0.1).clamp(0, 1)
            if method == "K":
                a, f = keyed(img, I_lo, a, f_lo, b_lo, lo, r, eps)
            elif method == "B":
                f = f_g
            elif method in ("D", "E"):
                f = img + (1 - a) * up(f_lo - b_lo, H, W)
                if method == "E":
                    f = despill(f.clamp(0, 1), a, up(b_lo, H, W))
            else:
                b_hi = up(b_lo, H, W)
                f_d = ((img - (1 - a) * b_hi) / a.clamp_min(1e-3)).clamp(0, 1)
                w = ((a - 0.1) / 0.4).clamp(0, 1)
                w = w * w * (3 - 2 * w)
                f = w * f_d + (1 - w) * f_g
    sync(); t["total"] = time.perf_counter() - t0
    return a.clamp(0, 1), f.clamp(0, 1), t


def load(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).to(dev).permute(2, 0, 1).float() / 255


def to_img(x):
    x = x if x.dim() == 3 else x[None].expand(3, -1, -1)
    return Image.fromarray((x.clamp(0, 1) * 255 + .5).byte().permute(1, 2, 0).cpu().numpy())


def spill(f, a):
    m = (a > 0.05) & (a < 0.95)
    s = (f[1] - torch.maximum(f[0], f[2])).clamp_min(0)
    return float(s[m].mean())


def metrics(a, f, ga, gf):
    m = ga > 0.02
    w = ga[m]
    ga_dx, ga_dy = ga[:, 1:] - ga[:, :-1], ga[1:] - ga[:-1]
    a_dx, a_dy = a[:, 1:] - a[:, :-1], a[1:] - a[:-1]
    grad = float(((a_dx - ga_dx).abs().mean() + (a_dy - ga_dy).abs().mean()) / 2 * 1000)
    edge = (ga > 0.02) & (ga < 0.98)
    return {"alpha_sad_k": float((a - ga).abs().sum() / 1000), "alpha_mse_e3": float(((a - ga) ** 2).mean() * 1000),
            "alpha_edge_mae_e2": float((a - ga).abs()[edge].mean() * 100), "alpha_grad_e3": grad,
            "fg_mae_e2": float(((f - gf).abs().mean(0)[m] * w).sum() / w.sum() * 100),
            "fg_edge_mae_e2": float((f - gf).abs().mean(0)[edge].mean() * 100),
            "spill_e3": spill(f, ga) * 1000}


def crops(name, a, f, img, boxes, bg=(0.08, 0.10, 0.22)):
    H, W = a.shape
    back = torch.tensor(bg, device=dev)[:, None, None].expand(3, H, W)
    comp = a * f + (1 - a) * back
    naive = a * img + (1 - a) * back
    out = []
    for (x, y, s) in boxes:
        out.append([to_img(c[:, y:y + s, x:x + s]) for c in (comp, naive)] + [to_img(a[y:y + s, x:x + s])])
    return comp, out


@torch.inference_mode()
def main():
    img = load(os.path.join(ROOT, "gs2k.png"))
    H, W = img.shape[1:]
    boxes = [(260, 1150, 512), (560, 120, 512), (1500, 900, 512)]
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    methods = ["A", "D", "E", "K", "N"]
    for m in methods: run(m, img)
    res, timing = {}, {}
    for m in methods:
        ts = [run(m, img)[2] for _ in range(3)]
        a, f, _ = run(m, img)
        res[m] = (a, f)
        timing[m] = {k: round(min(t[k] for t in ts) * 1000, 1) for k in ts[0]}
    print("timing ms (2048^2, 3090 Ti):", json.dumps(timing))
    tiles = []
    for m in methods:
        a, f = res[m]
        comp, cs = crops(m, a, f, img, boxes)
        to_img(comp).save(os.path.join(ROOT, "out", f"real_{m}_navy.png"))
        Image.fromarray(np.dstack([np.asarray(to_img(f)), np.asarray(to_img(a))[..., 0]]), "RGBA").save(
            os.path.join(ROOT, "out", f"real_{m}_cutout.png"))
        tiles.append(cs)
        print(f"real {m}: spill_e3={spill(f, a) * 1000:.2f}")
    s = 512
    sheet = Image.new("RGB", (s * 3 * len(boxes), s * len(methods)), "white")
    for i, cs in enumerate(tiles):
        for j, (comp, naive, al) in enumerate(cs):
            sheet.paste(comp, (j * 3 * s, i * s)); sheet.paste(al, (j * 3 * s + s, i * s)); sheet.paste(naive, (j * 3 * s + 2 * s, i * s))
    sheet.save(os.path.join(ROOT, "out", "real_sheet.png"))

    ga, gf = res["N"]
    ga = ((ga - 0.02) / 0.96).clamp(0, 1)
    yy = torch.linspace(0, 1, H, device=dev)[:, None]
    gb = torch.stack([0.12 + 0.10 * yy.expand(H, W), 0.70 - 0.15 * yy.expand(H, W), 0.10 + 0.05 * yy.expand(H, W)])
    gb = gb + 0.02 * torch.randn(1, H, W, device=dev)
    gf = gf.clone()
    gf[1] = torch.minimum(gf[1], torch.maximum(gf[0], gf[2]) * 1.05)
    syn = (ga * gf + (1 - ga) * gb).clamp(0, 1)
    to_img(syn).save(os.path.join(ROOT, "out", "syn_input.png"))
    print("synthetic GT: alpha=N-alpha stretched, F=N-fg with green suppressed, B=green gradient+noise")
    rows = {}
    for m in methods:
        a, f, t = run(m, syn)
        rows[m] = metrics(a, f, ga, gf)
        print(f"syn {m}:", " ".join(f"{k}={v:.3f}" for k, v in rows[m].items()))
    for lo, r, eps in [(1024, 4, 1e-4), (1024, 4, 1e-3), (1024, 8, 1e-4), (1024, 2, 1e-5), (768, 3, 1e-4), (512, 2, 1e-4)]:
        a, f, t = run("D", syn, lo, r, eps)
        print(f"syn D lo={lo} r={r} eps={eps}: total={t['total'] * 1000:.0f}ms", " ".join(f"{k}={v:.3f}" for k, v in metrics(a, f, ga, gf).items()))


main()
