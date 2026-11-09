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
        if hasattr(lib, "oenc_webp_decode"):
            ip = ctypes.POINTER(ctypes.c_int)
            lib.oenc_webp_info.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ip, ip, ip]
            lib.oenc_webp_info.restype = ctypes.c_int
            lib.oenc_webp_decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.oenc_webp_decode.restype = ctypes.c_int
        if hasattr(lib, "oenc_png_decode"):
            ip = ctypes.POINTER(ctypes.c_int)
            lib.oenc_png_info.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ip, ip, ip]
            lib.oenc_png_info.restype = ctypes.c_int
            lib.oenc_png_decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_int, ctypes.c_int]
            lib.oenc_png_decode.restype = ctypes.c_int
        if hasattr(lib, "oenc_png_mt"):
            lib.oenc_png_mt.argtypes = lib.oenc_png.argtypes
            lib.oenc_png_mt.restype = ctypes.c_int
        if hasattr(lib, "oenc_webp2"):
            lib.oenc_webp2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, out,
                                       ctypes.POINTER(ctypes.c_size_t)]
            lib.oenc_webp2.restype = ctypes.c_int
        return lib
    return None


_lib = _load()


def available() -> bool:
    return _lib is not None


def has_png_mt() -> bool:
    return _lib is not None and hasattr(_lib, "oenc_png_mt")


def has_webp2() -> bool:
    return _lib is not None and hasattr(_lib, "oenc_webp2")


def has_webp_decode() -> bool:
    return _lib is not None and hasattr(_lib, "oenc_webp_decode")


def webp_size(data: bytes) -> tuple[int, int, bool] | None:
    """(width, height, has_alpha) of a still WebP this module can decode, else None
    (not WebP, animated, or carrying EXIF that may hold an orientation)."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    w, h, a = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if _lib.oenc_webp_info(data, len(data), ctypes.byref(w), ctypes.byref(h), ctypes.byref(a)):
        return None
    return w.value, h.value, bool(a.value)


def webp_decode(data: bytes, width: int, height: int, channels: int = 3, threads: int = THREADS) -> np.ndarray:
    """(height, width, channels) uint8, RGB or RGBA, the pixels Pillow's decoder gives."""
    out = np.empty((height, width, channels), dtype=np.uint8)
    if _lib.oenc_webp_decode(data, len(data), out.ctypes.data, out.nbytes, width * channels, channels, threads):
        raise RuntimeError("native webp decode failed")
    return out


def has_png_decode() -> bool:
    return _lib is not None and hasattr(_lib, "oenc_png_decode")


def png_size(data: bytes) -> tuple[int, int, bool] | None:
    """(width, height, has_alpha) of a PNG on the fast path (8-bit RGB/RGBA, not interlaced,
    no tRNS/eXIf/APNG), else None."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h, c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if _lib.oenc_png_info(data, len(data), ctypes.byref(w), ctypes.byref(h), ctypes.byref(c)):
        return None
    return w.value, h.value, c.value == 4


def png_decode(data: bytes, width: int, height: int, channels: int = 3) -> np.ndarray:
    """(height, width, channels) uint8; RGBA sources lose their alpha when channels is 3."""
    out = np.empty((height, width, channels), dtype=np.uint8)
    if _lib.oenc_png_decode(data, len(data), out.ctypes.data, out.nbytes, width * channels, channels):
        raise RuntimeError("native png decode failed")
    return out


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


def png_mt(pixels: np.ndarray, level: int = 1, threads: int = THREADS) -> bytes:
    """zlib level (1..9), one filter+deflate strip per thread. Same pixels as png(), other bytes."""
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    out = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_size_t()
    if _lib.oenc_png_mt(arr.ctypes.data, w, h, c, level, threads, ctypes.byref(out), ctypes.byref(size)):
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


def webp2(rgba: np.ndarray, quality: float, method: int, alpha_method: int = -1, alpha_filter: int = 0,
          blank_transparent: bool = True, threads: int = THREADS) -> bytes:
    """webp() with a threaded colour conversion and the alpha plane as its own job.

    Same bytes as webp() when alpha_method is -1 or equals method; a lower alpha_method
    costs alpha bytes only (alpha is lossless at every setting, colour is untouched).
    """
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, c = arr.shape
    if c != 4:
        raise ValueError("webp expects RGBA")
    out = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_size_t()
    if _lib.oenc_webp2(arr.ctypes.data, w, h, w * 4, quality, method, alpha_method, alpha_filter,
                       int(blank_transparent), threads, ctypes.byref(out), ctypes.byref(size)):
        raise RuntimeError("native webp encode failed")
    return _take(out, size.value)
