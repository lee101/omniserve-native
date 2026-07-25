#!/usr/bin/env python3
"""Generate deterministic reference fixtures for a C/CUDA port of
pymatting's `estimate_foreground_ml`.

Outputs into matte/fixtures/:
    <case>_image.npy   float32 (H, W, 3) in [0, 1], C-contiguous
    <case>_alpha.npy   float32 (H, W)    in [0, 1], C-contiguous
    <case>_fg_ref.npy  float32 (H, W, 3) pymatting foreground
    <case>_bg_ref.npy  float32 (H, W, 3) pymatting background
    <case>_meta.json   shapes, dtypes, params, md5s, pyramid level sizes

Usage:
    .venv/bin/python gen_fixtures.py [--no-verify]

--verify (default on) also runs the literal pure-Python transcription in
ref_python.py and records its max abs deviation from pymatting in the meta
file. That is the check that the written spec in README.md is correct.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pymatting
from pymatting import estimate_foreground_ml

from ref_python import estimate_fb_ml_py, pyramid_levels

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

SEED = 20260725

# pymatting defaults, restated explicitly so the fixture records what was used.
PARAMS = dict(
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    gradient_weight=1.0,
)


def md5(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# case A: 64x64, soft-edged disc alpha over a smooth two-colour composite
# --------------------------------------------------------------------------
def case_a():
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = xx / (w - 1), yy / (h - 1)

    # Smooth, distinct foreground and background colour fields.
    fg = np.stack([0.90 - 0.25 * v, 0.35 + 0.30 * u, 0.15 + 0.20 * u * v], axis=-1)
    bg = np.stack([0.05 + 0.10 * u, 0.20 + 0.15 * v, 0.75 - 0.20 * u], axis=-1)

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)
    # radius 20 px, 4 px feather -> a genuine soft band of fractional alpha
    alpha = 1.0 - smoothstep(18.0, 22.0, r)

    image = alpha[..., None] * fg + (1.0 - alpha[..., None]) * bg
    return f32(np.clip(image, 0, 1)), f32(np.clip(alpha, 0, 1))


# --------------------------------------------------------------------------
# case B: 128x96 greenscreen composite with deliberate green spill
# --------------------------------------------------------------------------
def case_b(rng):
    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    green = np.zeros((h, w, 3))
    green[..., 1] = 1.0  # pure green backdrop

    # Foreground object: rounded superellipse blob, textured colour.
    cy, cx = 46.0, 62.0
    ry, rx = 30.0, 38.0
    d = (np.abs((yy - cy) / ry) ** 2.5 + np.abs((xx - cx) / rx) ** 2.5) ** (1 / 2.5)
    # feathered border: alpha ramps over d in [0.88, 1.06]
    alpha = 1.0 - smoothstep(0.88, 1.06, d)

    tex = 0.5 + 0.5 * np.sin(xx / 5.0) * np.sin(yy / 7.0)
    fg = np.stack(
        [
            0.80 - 0.20 * tex,
            0.18 + 0.10 * tex,
            0.55 + 0.25 * (yy / (h - 1)),
        ],
        axis=-1,
    )
    fg += 0.02 * rng.standard_normal(fg.shape)  # slight grain
    fg = np.clip(fg, 0, 1)

    image = alpha[..., None] * fg + (1.0 - alpha[..., None]) * green

    # Deliberate green colour bleed / spill in a band hugging the border,
    # extending INSIDE the fully-opaque region (d in [0.70, 1.06]). This is the
    # artefact foreground estimation is supposed to be exercised on: the observed
    # pixel is greener than the true foreground even where alpha == 1.
    spill_band = smoothstep(0.70, 0.92, d) * (1.0 - smoothstep(1.00, 1.06, d))
    spill = 0.45 * spill_band * alpha
    image[..., 1] += spill * (1.0 - image[..., 1])
    image[..., 0] *= 1.0 - 0.35 * spill
    image[..., 2] *= 1.0 - 0.35 * spill

    return f32(np.clip(image, 0, 1)), f32(np.clip(alpha, 0, 1))


# --------------------------------------------------------------------------
# case C: 33x17 odd size, random noise alpha (non-power-of-two pyramid stress)
# --------------------------------------------------------------------------
def case_c(rng):
    h, w = 17, 33
    image = rng.random((h, w, 3))
    alpha = rng.random((h, w))
    # guarantee the alpha > 0.9 and alpha < 0.1 populations used for the initial
    # mean colours are both non-empty regardless of draw
    alpha[0, :4] = 1.0
    alpha[-1, -4:] = 0.0
    return f32(np.clip(image, 0, 1)), f32(np.clip(alpha, 0, 1))


CASES = {"A": case_a, "B": case_b, "C": case_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the slow pure-Python spec cross-check")
    args = ap.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    summary = []
    for name, fn in CASES.items():
        image, alpha = fn() if fn is case_a else fn(rng)
        h, w, depth = image.shape

        fg, bg = estimate_foreground_ml(image, alpha, return_background=True, **PARAMS)
        fg, bg = f32(fg), f32(bg)

        arrays = {
            f"{name}_image": image,
            f"{name}_alpha": alpha,
            f"{name}_fg_ref": fg,
            f"{name}_bg_ref": bg,
        }
        for key, arr in arrays.items():
            np.save(FIXTURES / f"{key}.npy", arr)

        meta = {
            "case": name,
            "seed": SEED,
            "shape": {"h": h, "w": w, "depth": depth},
            "image_shape": list(image.shape),
            "alpha_shape": list(alpha.shape),
            "dtype": "float32",
            "layout": "C-contiguous, image/fg/bg are HWC RGB in [0,1], alpha is HW in [0,1]",
            "params": PARAMS,
            "return_background": True,
            "pymatting_version": pymatting.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "pyramid_levels": [{"i_level": i, "h": lh, "w": lw,
                                "n_iter": PARAMS["n_small_iterations"]
                                if (lw <= PARAMS["small_size"] and lh <= PARAMS["small_size"])
                                else PARAMS["n_big_iterations"]}
                               for i, (lh, lw) in enumerate(pyramid_levels(h, w))],
            "md5": {k: md5(v) for k, v in arrays.items()},
        }

        if not args.no_verify:
            pf, pb = estimate_fb_ml_py(image, alpha, **PARAMS)
            meta["pure_python_spec_check"] = {
                "fg_max_abs_diff": float(np.abs(pf - fg).max()),
                "bg_max_abs_diff": float(np.abs(pb - bg).max()),
            }

        with open(FIXTURES / f"{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        summary.append((name, image.shape, alpha.shape, meta))
        print(f"case {name}: image {image.shape} alpha {alpha.shape} "
              f"levels={[(l['h'], l['w'], l['n_iter']) for l in meta['pyramid_levels']]}")
        if "pure_python_spec_check" in meta:
            print(f"  spec check vs pymatting: fg {meta['pure_python_spec_check']['fg_max_abs_diff']:.3e} "
                  f"bg {meta['pure_python_spec_check']['bg_max_abs_diff']:.3e}")
        print(f"  alpha range [{alpha.min():.3f}, {alpha.max():.3f}], "
              f"fractional px = {int(((alpha > 0.01) & (alpha < 0.99)).sum())}")

    print(f"\nwrote {len(list(FIXTURES.glob('*')))} files to {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
