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
        # Device-pointer entry points. Buffers are passed as plain integers
        # (torch .data_ptr()), so nothing here needs a numpy view of GPU memory.
        if hasattr(lib, "omatte_estimate_fb_cuda_device"):
            lib.omatte_estimate_fb_cuda_device.restype = ctypes.c_int
            lib.omatte_estimate_fb_cuda_device.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(_Params), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.omatte_composite_cuda_device.restype = ctypes.c_int
            lib.omatte_composite_cuda_device.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.omatte_cuda_workspace_bytes.restype = ctypes.c_size_t
            lib.omatte_cuda_workspace_bytes.argtypes = []
            lib.omatte_cuda_release_workspace.restype = None
            lib.omatte_cuda_release_workspace.argtypes = []
        lib.omatte_composite_image.restype = None
        lib.omatte_composite_image.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ]
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


def device_api_available() -> bool:
    """True when the library exposes the device-pointer entry points."""
    try:
        return cuda_available() and hasattr(_load(), "omatte_estimate_fb_cuda_device")
    except MatteUnavailable:
        return False


def workspace_bytes() -> int:
    """Device memory the cached pyramid workspace is holding."""
    try:
        lib = _load()
    except MatteUnavailable:
        return 0
    return int(lib.omatte_cuda_workspace_bytes()) if hasattr(lib, "omatte_cuda_workspace_bytes") else 0


def release_workspace() -> None:
    try:
        lib = _load()
    except MatteUnavailable:
        return
    if hasattr(lib, "omatte_cuda_release_workspace"):
        lib.omatte_cuda_release_workspace()


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


def _check_device_tensor(tensor, name: str):
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype.itemsize != 4 or not tensor.is_floating_point():
        raise ValueError(f"{name} must be float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def estimate_foreground_torch(
    image,
    alpha,
    *,
    return_background: bool = False,
    regularization: float = 1e-5,
    n_small_iterations: int = 10,
    n_big_iterations: int = 2,
    small_size: int = 32,
    gradient_weight: float = 1.0,
    stream=None,
):
    """Foreground estimation on tensors that are already on the GPU.

    BiRefNet produces its alpha on the device and the composite is consumed
    there, so the numpy entry point above pays four transfers - alpha down,
    image and alpha back up, F and B down again - that the caller never asked
    for. On a 1024x1024 cutout that is ~28 MB of round trip against ~1 ms of
    actual solving. This path moves nothing.

    image: (h, w, depth) float32 CUDA tensor in [0, 1]
    alpha: (h, w)        float32 CUDA tensor in [0, 1]

    Returns fg, or (fg, bg) when return_background is set. Both are new tensors
    on the same device. `stream` defaults to torch's current stream, so the pass
    is ordered against the segmentation work that produced the alpha without a
    device-wide synchronise.
    """
    import torch

    lib = _load()
    if not hasattr(lib, "omatte_estimate_fb_cuda_device"):
        raise MatteUnavailable("libomatte.so was built without the CUDA device API")

    image = _check_device_tensor(image.contiguous(), "image")
    alpha = _check_device_tensor(alpha.contiguous(), "alpha")
    if image.dim() != 3 or alpha.dim() != 2 or image.shape[:2] != alpha.shape:
        raise ValueError(f"shape mismatch: image {tuple(image.shape)}, alpha {tuple(alpha.shape)}")

    h, w, depth = (int(v) for v in image.shape)
    params = _Params(
        regularization=regularization,
        n_small_iterations=n_small_iterations,
        n_big_iterations=n_big_iterations,
        small_size=small_size,
        gradient_weight=gradient_weight,
        threads=0,
        order=ORDER_RED_BLACK,
    )

    fg = torch.empty_like(image)
    bg = torch.empty_like(image) if return_background else None
    if stream is None:
        stream = torch.cuda.current_stream(image.device).cuda_stream

    rc = lib.omatte_estimate_fb_cuda_device(
        ctypes.c_void_p(image.data_ptr()), ctypes.c_void_p(alpha.data_ptr()),
        h, w, depth, ctypes.byref(params),
        ctypes.c_void_p(fg.data_ptr()),
        ctypes.c_void_p(bg.data_ptr()) if bg is not None else None,
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"omatte_estimate_fb_cuda_device failed with rc={rc}")
    return (fg, bg) if return_background else fg


def composite_torch(fg, alpha, background=None, background_rgb=None, out=None, stream=None):
    """alpha * fg + (1 - alpha) * background, on the GPU.

    background is a (h, w, depth) CUDA tensor, or None to use background_rgb (a
    sequence of `depth` floats), or neither for black. `out` may be `fg`.
    """
    import torch

    lib = _load()
    if not hasattr(lib, "omatte_composite_cuda_device"):
        raise MatteUnavailable("libomatte.so was built without the CUDA device API")

    fg = _check_device_tensor(fg.contiguous(), "fg")
    alpha = _check_device_tensor(alpha.contiguous(), "alpha")
    h, w, depth = (int(v) for v in fg.shape)
    if background is not None:
        background = _check_device_tensor(background.contiguous(), "background")
        if tuple(background.shape) != (h, w, depth):
            raise ValueError("background must match the foreground shape")
    if out is None:
        out = torch.empty_like(fg)
    else:
        _check_device_tensor(out, "out")

    solid = None
    if background_rgb is not None:
        solid = (ctypes.c_float * depth)(*[float(v) for v in background_rgb][:depth])
    if stream is None:
        stream = torch.cuda.current_stream(fg.device).cuda_stream

    rc = lib.omatte_composite_cuda_device(
        ctypes.c_void_p(fg.data_ptr()), ctypes.c_void_p(alpha.data_ptr()),
        ctypes.c_void_p(background.data_ptr()) if background is not None else None,
        solid, h, w, depth,
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"omatte_composite_cuda_device failed with rc={rc}")
    return out
