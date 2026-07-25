"""ctypes binding for the C/CUDA foreground estimator (include/omatte.h).

Used by the BiRefNet worker to remove backdrop colour bleed from the edges of a
cutout: the segmentation model gives alpha, this recovers the true foreground
colour so a green screen (or any strong backdrop) stops tinting the fringe.

Build the library with:

    cmake -S . -B build-cuda -DWITH_MATTE_CUDA=ON && cmake --build build-cuda -j

Set OMATTE_LIB to point at a specific libomatte.so, otherwise the usual build
directories are searched.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ORDER_SEQUENTIAL = 0
ORDER_RED_BLACK = 1

REPO = Path(__file__).resolve().parents[1]
_SEARCH = [
    os.getenv("OMATTE_LIB", ""),
    str(REPO / "build-cuda" / "libomatte.so"),
    str(REPO / "build-full" / "libomatte.so"),
    str(REPO / "build" / "libomatte.so"),
    str(REPO / "matte" / "libomatte.so"),
    "libomatte.so",
]


class _Params(ctypes.Structure):
    _fields_ = [
        ("regularization", ctypes.c_float),
        ("n_small_iterations", ctypes.c_int),
        ("n_big_iterations", ctypes.c_int),
        ("small_size", ctypes.c_int),
        ("gradient_weight", ctypes.c_float),
        ("threads", ctypes.c_int),
        ("order", ctypes.c_int),
    ]


class MatteUnavailable(RuntimeError):
    """Raised when libomatte.so cannot be loaded."""


_lib = None
_lib_path = None


def _load():
    global _lib, _lib_path
    if _lib is not None:
        return _lib

    errors = []
    for candidate in _SEARCH:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as err:
            errors.append(f"{candidate}: {err}")
            continue

        lib.omatte_estimate_fb.restype = ctypes.c_int
        lib.omatte_estimate_fb.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_Params), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        lib.omatte_estimate_fb_cuda.restype = ctypes.c_int
        lib.omatte_estimate_fb_cuda.argtypes = lib.omatte_estimate_fb.argtypes
        lib.omatte_cuda_available.restype = ctypes.c_bool
        lib.omatte_cuda_available.argtypes = []
        _lib, _lib_path = lib, candidate
        return _lib

    raise MatteUnavailable("libomatte.so not found; tried:\n  " + "\n  ".join(errors))


def library_path() -> str | None:
    """Path of the loaded library, or None if it has not been loaded yet."""
    return _lib_path


def cuda_available() -> bool:
    try:
        return bool(_load().omatte_cuda_available())
    except MatteUnavailable:
        return False


def estimate_foreground(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    use_cuda: bool | None = None,
    regularization: float = 1e-5,
    n_small_iterations: int = 10,
    n_big_iterations: int = 2,
    small_size: int = 32,
    gradient_weight: float = 1.0,
    threads: int = 0,
    order: int | None = None,
    return_background: bool = False,
):
    """Estimates foreground (and optionally background) colours.

    image: (h, w, d) float in [0, 1]; alpha: (h, w) float in [0, 1].
    Falls back to the threaded CPU path when CUDA is unavailable.
    """
    lib = _load()

    image = np.ascontiguousarray(image, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    if image.ndim != 3 or alpha.ndim != 2 or image.shape[:2] != alpha.shape:
        raise ValueError(f"shape mismatch: image {image.shape}, alpha {alpha.shape}")

    h, w, depth = image.shape
    if use_cuda is None:
        use_cuda = cuda_available()
    if order is None:
        # CUDA only implements the checkerboard order; on CPU keep the
        # reference sweep unless the caller asks for the parallel one.
        order = ORDER_RED_BLACK if use_cuda else ORDER_SEQUENTIAL

    params = _Params(
        regularization=regularization,
        n_small_iterations=n_small_iterations,
        n_big_iterations=n_big_iterations,
        small_size=small_size,
        gradient_weight=gradient_weight,
        threads=threads,
        order=order,
    )

    fg = np.empty_like(image)
    bg = np.empty_like(image) if return_background else None
    as_ptr = lambda array: array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))  # noqa: E731

    call = lib.omatte_estimate_fb_cuda if use_cuda else lib.omatte_estimate_fb
    rc = call(as_ptr(image), as_ptr(alpha), h, w, depth, ctypes.byref(params),
              as_ptr(fg), as_ptr(bg) if bg is not None else None)

    if rc == -3 and use_cuda:
        # Library built without CUDA: retry on the CPU rather than failing.
        rc = lib.omatte_estimate_fb(as_ptr(image), as_ptr(alpha), h, w, depth,
                                    ctypes.byref(params), as_ptr(fg),
                                    as_ptr(bg) if bg is not None else None)
    if rc != 0:
        raise RuntimeError(f"omatte_estimate_fb failed with rc={rc}")

    return (fg, bg) if return_background else fg
