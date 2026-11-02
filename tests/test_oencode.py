#!/usr/bin/env python3
import io
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("skipping: numpy or Pillow is not installed")
    raise SystemExit(0) if __name__ == "__main__" else unittest.SkipTest("dependencies unavailable")

import oencode

needs_lib = unittest.skipUnless(oencode.available(), "liboencode.so is not built")


def decode(data):
    return np.asarray(Image.open(io.BytesIO(data)))


def scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) * 255 // max(h + w - 2, 1)], -1)
    rgb = (rgb + rng.integers(0, 6, rgb.shape)).clip(0, 255).astype(np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    alpha[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
    edge = rng.integers(1, 255, (h, w), dtype=np.uint8)
    ring = (alpha == 0) & (np.roll(alpha, 2, 0) > 0)
    alpha[ring] = edge[ring]
    return np.dstack([rgb, alpha])


@needs_lib
class PngTest(unittest.TestCase):
    def test_rgba_lossless_at_every_level(self):
        arr = scene(61, 97)
        for level in (1, 2, 3, 6, 9, 12):
            with self.subTest(level=level):
                np.testing.assert_array_equal(decode(oencode.png(arr, level)), arr)

    def test_rgb_and_gray(self):
        arr = scene(33, 40)
        np.testing.assert_array_equal(decode(oencode.png(arr[:, :, :3])), arr[:, :, :3])
        np.testing.assert_array_equal(decode(oencode.png(arr[:, :, 0])), arr[:, :, 0])

    def test_tiny_and_thread_counts(self):
        for h, w in ((1, 1), (1, 9), (7, 1), (3, 5)):
            arr = scene(h, w, seed=h * 31 + w) if h > 3 and w > 3 else np.random.default_rng(h + w).integers(0, 256, (h, w, 4), dtype=np.uint8)
            for threads in (1, 2, 8, 64):
                with self.subTest(h=h, w=w, threads=threads):
                    np.testing.assert_array_equal(decode(oencode.png(arr, 2, threads)), arr)

    def test_noncontiguous_input(self):
        arr = scene(40, 60)[:, ::2]
        np.testing.assert_array_equal(decode(oencode.png(arr)), arr)

    def test_deterministic_across_thread_counts(self):
        arr = scene(128, 128)
        self.assertEqual(oencode.png(arr, 2, 1), oencode.png(arr, 2, 8))

    def test_bad_channels(self):
        with self.assertRaises(RuntimeError):
            oencode.png(np.zeros((4, 4, 5), np.uint8))

    def test_smaller_than_pillow_default(self):
        arr = scene(256, 256)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", compress_level=3)
        self.assertLessEqual(len(oencode.png(arr, 2)), len(buf.getvalue()))


@needs_lib
class WebpTest(unittest.TestCase):
    def reference(self, arr, method):
        blank = arr.copy()
        blank[blank[:, :, 3] == 0, :3] = 0
        buf = io.BytesIO()
        Image.fromarray(blank, "RGBA").save(buf, format="WEBP", quality=85, method=method, exact=False)
        return buf.getvalue()

    def test_opaque_matches_pillow_bytes(self):
        arr = scene(96, 96)
        arr[:, :, 3] = 255
        for method in (0, 2, 4):
            self.assertEqual(oencode.webp(arr, 85, method), self.reference(arr, method))

    def test_alpha_is_lossless_for_every_filter(self):
        arr = scene(130, 150)
        for method in (2, 4):
            for alpha_filter in (0, 1, 2):
                with self.subTest(method=method, alpha_filter=alpha_filter):
                    out = decode(oencode.webp(arr, 85, method, alpha_filter))
                    np.testing.assert_array_equal(out[:, :, 3], arr[:, :, 3])

    def test_pixels_match_pillow_decode(self):
        arr = scene(130, 150)
        for method in (2, 4):
            np.testing.assert_array_equal(decode(oencode.webp(arr, 85, method)), decode(self.reference(arr, method)))

    def test_hidden_rgb_does_not_affect_output(self):
        arr = scene(64, 64)
        dirty = arr.copy()
        dirty[arr[:, :, 3] == 0, :3] = (200, 30, 90)
        self.assertEqual(oencode.webp(dirty, 85, 2), oencode.webp(arr, 85, 2))

    def test_threading_does_not_change_bytes(self):
        arr = scene(128, 128)
        self.assertEqual(oencode.webp(arr, 85, 2, 0, True, 1), oencode.webp(arr, 85, 2, 0, True, 4))

    def test_requires_rgba(self):
        with self.assertRaises(ValueError):
            oencode.webp(np.zeros((4, 4, 3), np.uint8), 85, 2)


if __name__ == "__main__":
    unittest.main()
