"""ctypes binding for the native PNG/WebP encoder (include/oencode.h).

Set OENCODE_LIB to point at a specific liboencode.so. `available()` is False
when the library is missing, and callers fall back to Pillow.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SEARCH = [
    os.getenv("OENCODE_LIB", ""),
    str(REPO / "build-cuda" / "liboencode.so"),
    str(REPO / "build-full" / "liboencode.so"),
    str(REPO / "build" / "liboencode.so"),
    "liboencode.so",
]
THREADS = max(1, int(os.getenv("OENCODE_THREADS", str(min(8, os.cpu_count() or 1)))))


def _load():
    for path in _SEARCH:
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        u8p = ctypes.POINTER(ctypes.c_uint8)
        out = ctypes.POINTER(u8p)
        lib.oenc_png.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, out, ctypes.POINTER(ctypes.c_size_t)]
        lib.oenc_png.restype = ctypes.c_int
        lib.oenc_webp.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, out, ctypes.POINTER(ctypes.c_size_t)]
        lib.oenc_webp.restype = ctypes.c_int
        lib.oenc_free.argtypes = [ctypes.c_void_p]
        lib.oenc_free.restype = None
        return lib
    return None


_lib = _load()


def available() -> bool:
    return _lib is not None


def _take(ptr, size: int) -> bytes:
    try:
        return ctypes.string_at(ptr, size)
    finally:
        _lib.oenc_free(ptr)


def png(pixels: np.ndarray, level: int = 3, threads: int = THREADS) -> bytes:
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    out = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_size_t()
    if _lib.oenc_png(arr.ctypes.data, w, h, c, level, threads, ctypes.byref(out), ctypes.byref(size)):
        raise RuntimeError("native png encode failed")
    return _take(out, size.value)


def webp(rgba: np.ndarray, quality: float, method: int, alpha_filter: int = 0, blank_transparent: bool = True,
         threads: int = THREADS) -> bytes:
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, c = arr.shape
    if c != 4:
        raise ValueError("webp expects RGBA")
    out = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_size_t()
    if _lib.oenc_webp(arr.ctypes.data, w, h, w * 4, quality, method, alpha_filter, int(blank_transparent), threads,
                      ctypes.byref(out), ctypes.byref(size)):
        raise RuntimeError("native webp encode failed")
    return _take(out, size.value)
