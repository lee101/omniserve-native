"""High-resolution matte from a low-resolution network pass.

BiRefNet sees the image at ~1024 px. For a larger image the old path stretched
that alpha bilinearly and solved foreground colours at full size. Here every
expensive step stays at network resolution and only per-pixel linear maps are
applied at full size, so strand detail comes from the original pixels:

- alpha: colour guided filter (He et al.) fitted at low resolution, its linear
  coefficients upsampled and applied to the full-resolution image.
- keyed backdrops (green/blue screen): the network calls a spill-lit, backlit
  hair halo opaque; near the subject's edge the backdrop-channel excess of the
  full-resolution pixels (G - max(R, B) for green) gives the real coverage.
- colours: F - B is smooth, so the low-resolution solve is upsampled as a
  correction, F = I + (1 - alpha) * up(F_lo - B_lo), with no division by alpha.
  On a keyed backdrop the remaining key-channel excess is removed near the edge.

Tensors are channel-first float32 on the GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import omatte

KEYABLE_EXCESS = 0.15
KEYABLE_SHARE = 0.6
DESPILL_RESTORE = 0.5


def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def _resize(x, size, mode="bilinear"):
    if mode == "area":
        return F.interpolate(x[None], size=size, mode="area")[0]
    return F.interpolate(x[None], size=size, mode=mode, align_corners=False)[0]


def low_size(height: int, width: int, side: int) -> tuple[int, int]:
    scale = side / max(height, width)
    return max(8, round(height * scale)), max(8, round(width * scale))


def guided_upsample(guide_lo, p_lo, guide_hi, r: int, eps: float):
    """p ~ A^T I + b fitted per window on (guide_lo, p_lo), applied to guide_hi."""
    c = p_lo.shape[0]
    h, w = guide_lo.shape[1:]
    mean_i = _box(guide_lo[None], r)[0]
    mean_p = _box(p_lo[None], r)[0]
    ii = torch.stack([guide_lo[i] * guide_lo[j] for i in range(3) for j in range(3)])
    ip = torch.stack([guide_lo[i] * p_lo[k] for i in range(3) for k in range(c)])
    var = _box(ii[None], r)[0] - torch.stack([mean_i[i] * mean_i[j] for i in range(3) for j in range(3)])
    cov = _box(ip[None], r)[0] - torch.stack([mean_i[i] * mean_p[k] for i in range(3) for k in range(c)])
    sigma = var.permute(1, 2, 0).reshape(h, w, 3, 3) + eps * torch.eye(3, device=guide_lo.device)
    a = torch.linalg.solve(sigma, cov.permute(1, 2, 0).reshape(h, w, 3, c))
    b = mean_p - torch.einsum("hwic,ihw->chw", a, mean_i)
    a = _box(a.reshape(h, w, 3 * c).permute(2, 0, 1)[None], r)[0]
    b = _box(b[None], r)[0]
    size = guide_hi.shape[1:]
    a = _resize(a, size).reshape(3, c, *size)
    return (a * guide_hi[:, None]).sum(0) + _resize(b, size)


def _solve(image_lo, alpha_lo):
    fg, bg = omatte.estimate_foreground_torch(image_lo.permute(1, 2, 0).contiguous(),
                                              alpha_lo.contiguous(), return_background=True)
    return fg.permute(2, 0, 1), bg.permute(2, 0, 1)


def _key(bg_lo, alpha_lo):
    """Index of the backdrop's key channel, or None when the backdrop is not a screen."""
    key = int(bg_lo.mean((1, 2)).argmax())
    others = [c for c in range(3) if c != key]
    excess = bg_lo[key] - torch.maximum(bg_lo[others[0]], bg_lo[others[1]])
    clear = alpha_lo < 0.05
    if int(clear.sum()) < 64:
        return None, others
    share = float((excess[clear] > KEYABLE_EXCESS).float().mean())
    return (key if share >= KEYABLE_SHARE else None), others


def _edge_band(alpha_lo, size, radius_lo: int):
    """1 within radius of the background side of the edge, fading inward."""
    outside = (alpha_lo < 0.5).float()[None, None]
    near = F.max_pool2d(outside, 2 * radius_lo + 1, stride=1, padding=radius_lo)
    near = _box(near, max(1, radius_lo // 2))[0]
    return _resize(near, size)[0].clamp(0, 1)


def refine_alpha(image_hi, prob_lo, *, key: bool = True, radius: int | None = None,
                 eps: float = 1e-4, band_lo: int = 24):
    """image_hi (3, H, W) in [0, 1]; prob_lo (h, w) network probability at any size."""
    size_hi = image_hi.shape[1:]
    size_lo = low_size(*size_hi, max(prob_lo.shape))
    image_lo = _resize(image_hi, size_lo, "area")
    prob_lo = _resize(prob_lo[None], size_lo)
    r = radius or max(2, max(size_lo) // 256)
    alpha = guided_upsample(image_lo, prob_lo, image_hi, r, eps)[0].clamp(0, 1)
    if not key:
        return alpha
    alpha_lo = _resize(alpha[None], size_lo, "area")[0]
    _, bg_lo = _solve(image_lo, alpha_lo)
    k, others = _key(bg_lo, alpha_lo)
    if k is None:
        return alpha
    bg_hi = _resize(bg_lo, size_hi)
    excess = image_hi[k] - torch.maximum(image_hi[others[0]], image_hi[others[1]])
    excess_bg = (bg_hi[k] - torch.maximum(bg_hi[others[0]], bg_hi[others[1]])).clamp_min(KEYABLE_EXCESS)
    keyed = (1 - excess / excess_bg).clamp(0, 1)
    band = _edge_band(alpha_lo, size_hi, band_lo)
    return alpha - band * (alpha - torch.minimum(alpha, keyed))


def despill(fg, bg_lo, k: int, others, weight):
    """Remove screen light as a vector: F_seen = F + s * K, K the backdrop colour.

    s is what cancels the key channel's excess over the mean of the other two
    (green spill on hair: lime (0.75, 0.95, 0.15) -> brown, not the yellow or
    orange a per-channel clamp gives). Part of K's luminance comes back as a
    neutral highlight so backlit rims stay bright.
    """
    key = bg_lo.mean((1, 2))
    key_excess = float(key[k] - (key[others[0]] + key[others[1]]) * 0.5)
    if key_excess <= 0.05:
        return fg
    excess = fg[k] - (fg[others[0]] + fg[others[1]]) * 0.5
    s = (excess / key_excess).clamp_min(0) * weight
    luma = float(0.299 * key[0] + 0.587 * key[1] + 0.114 * key[2])
    return (fg - s * key[:, None, None] + s * (luma * DESPILL_RESTORE)).clamp(0, 1)


def recolor(image_hi, alpha_hi, side: int, *, band_lo: int = 96, want_background: bool = False):
    """Foreground (and backdrop) at full resolution from a low-resolution solve."""
    size_hi = image_hi.shape[1:]
    size_lo = low_size(*size_hi, side)
    image_lo = _resize(image_hi, size_lo, "area")
    alpha_lo = _resize(alpha_hi[None], size_lo, "area")[0]
    fg_lo, bg_lo = _solve(image_lo, alpha_lo)
    fg = (image_hi + (1 - alpha_hi) * _resize(fg_lo - bg_lo, size_hi)).clamp(0, 1)
    k, others = _key(bg_lo, alpha_lo)
    if k is not None:
        fg = despill(fg, bg_lo, k, others, _edge_band(alpha_lo, size_hi, band_lo))
    bg = _resize(bg_lo, size_hi).clamp(0, 1) if want_background else None
    return fg, bg
