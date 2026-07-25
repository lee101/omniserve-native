#!/usr/bin/env python3
"""Run the C implementation (src/omatte.c) as an eval candidate, via ctypes.

Avoids needing a C main() just to shuffle .npy files around.

    <prog> <image.npy> <alpha.npy> <out.npy> [fg|bg] [sequential|red_black] [threads]

Build the shared object first (any of these work):

    cc -O2 -shared -fPIC -Iinclude src/omatte.c -o matte/libomatte.so -lm -lpthread

Then, from matte/:

    .venv/bin/python eval_foreground.py --case B --target fg \
        --candidate-cmd ".venv/bin/python ctypes_candidate.py {image} {alpha} {out} fg sequential"
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_LIB = HERE / "libomatte.so"

ORDER = {"sequential": 0, "red_black": 1}


class OmatteParams(ctypes.Structure):
    _fields_ = [
        ("regularization", ctypes.c_float),
        ("n_small_iterations", ctypes.c_int),
        ("n_big_iterations", ctypes.c_int),
        ("small_size", ctypes.c_int),
        ("gradient_weight", ctypes.c_float),
        ("threads", ctypes.c_int),
        ("order", ctypes.c_int),
    ]


def load_lib(path=None):
    p = Path(path or DEFAULT_LIB)
    if not p.is_file():
        raise SystemExit(f"shared object not found: {p}\nbuild it, see {__file__} docstring")
    lib = ctypes.CDLL(str(p))
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.omatte_estimate_fb.restype = ctypes.c_int
    lib.omatte_estimate_fb.argtypes = [
        f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(OmatteParams), f32p, f32p,
    ]
    return lib


def run(lib, image, alpha, order="sequential", threads=0):
    image = np.ascontiguousarray(image, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    h, w, depth = image.shape

    F = np.zeros_like(image)
    B = np.zeros_like(image)
    params = OmatteParams(1e-5, 10, 2, 32, 1.0, threads, ORDER[order])

    f32p = ctypes.POINTER(ctypes.c_float)
    rc = lib.omatte_estimate_fb(
        image.ctypes.data_as(f32p), alpha.ctypes.data_as(f32p),
        h, w, depth, ctypes.byref(params),
        F.ctypes.data_as(f32p), B.ctypes.data_as(f32p),
    )
    if rc != 0:
        raise SystemExit(f"omatte_estimate_fb returned {rc}")
    return F, B


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    image_path, alpha_path, out_path = sys.argv[1:4]
    target = sys.argv[4] if len(sys.argv) > 4 else "fg"
    order = sys.argv[5] if len(sys.argv) > 5 else "sequential"
    threads = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    lib = load_lib()
    F, B = run(lib, np.load(image_path), np.load(alpha_path), order, threads)
    np.save(out_path, F if target == "fg" else B)
    return 0


if __name__ == "__main__":
    sys.exit(main())
