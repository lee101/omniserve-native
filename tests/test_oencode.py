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


@needs_lib
@unittest.skipUnless(oencode.has_png_mt(), "liboencode.so predates oenc_png_mt")
class PngParallelTest(unittest.TestCase):
    def test_lossless_for_every_level_layout_and_thread_count(self):
        arr = scene(301, 517)
        for pixels in (arr, arr[:, :, :3], arr[:, :, 0], arr[:, :, [0, 3]]):
            for level in (1, 4, 9):
                for threads in (1, 2, 8, 64):
                    with self.subTest(shape=pixels.shape, level=level, threads=threads):
                        np.testing.assert_array_equal(decode(oencode.png_mt(pixels, level, threads)), pixels)

    def test_tiny(self):
        for h, w in ((1, 1), (1, 9), (7, 1), (3, 5)):
            arr = np.random.default_rng(h + w).integers(0, 256, (h, w, 4), dtype=np.uint8)
            for threads in (1, 8):
                with self.subTest(h=h, w=w, threads=threads):
                    np.testing.assert_array_equal(decode(oencode.png_mt(arr, 1, threads)), arr)

    def test_incompressible_strips(self):
        arr = np.random.default_rng(3).integers(0, 256, (700, 900, 4), dtype=np.uint8)
        np.testing.assert_array_equal(decode(oencode.png_mt(arr, 1, 16)), arr)

    def test_close_to_single_stream_size(self):
        arr = scene(1024, 1024)
        self.assertLess(len(oencode.png_mt(arr, 4, 16)), len(oencode.png(arr, 2)) * 1.1)

    def test_bad_channels(self):
        with self.assertRaises(RuntimeError):
            oencode.png_mt(np.zeros((4, 4, 5), np.uint8))


@needs_lib
@unittest.skipUnless(oencode.has_webp2(), "liboencode.so predates oenc_webp2")
class WebpSplitTest(unittest.TestCase):
    def test_same_bytes_as_single_call(self):
        for h, w in ((130, 150), (131, 149), (64, 64), (1, 1), (2, 3), (517, 301)):
            arr = scene(h, w) if h > 3 else np.random.default_rng(h + w).integers(0, 256, (h, w, 4), dtype=np.uint8)
            for method in (0, 2, 4):
                for alpha_filter in (0, 1):
                    for threads in (1, 2, 8, 64):
                        with self.subTest(h=h, w=w, method=method, alpha_filter=alpha_filter, threads=threads):
                            self.assertEqual(oencode.webp2(arr, 85, method, -1, alpha_filter, True, threads),
                                             oencode.webp(arr, 85, method, alpha_filter, True, 8))

    def test_opaque_and_unblanked_match(self):
        arr = scene(200, 120)
        self.assertEqual(oencode.webp2(arr, 85, 2, -1, 0, False), oencode.webp(arr, 85, 2, 0, False))
        arr[:, :, 3] = 255
        self.assertEqual(oencode.webp2(arr, 85, 4), oencode.webp(arr, 85, 4))

    def test_opaque_strip_beside_transparent_strip(self):
        arr = scene(512, 96)
        arr[:256, :, 3] = 255
        for threads in (2, 8):
            self.assertEqual(oencode.webp2(arr, 85, 2, -1, 0, True, threads), oencode.webp(arr, 85, 2))

    def test_alpha_method_changes_bytes_not_pixels(self):
        arr = scene(260, 300)
        reference = decode(oencode.webp(arr, 85, 4))
        for alpha_method in (0, 1, 6):
            for alpha_filter in (0, 1):
                with self.subTest(alpha_method=alpha_method, alpha_filter=alpha_filter):
                    out = decode(oencode.webp2(arr, 85, 4, alpha_method, alpha_filter))
                    np.testing.assert_array_equal(out, reference)
                    np.testing.assert_array_equal(out[:, :, 3], arr[:, :, 3])

    def test_noncontiguous_input(self):
        arr = scene(80, 120)[:, ::2]
        self.assertEqual(oencode.webp2(arr, 85, 2), oencode.webp(arr, 85, 2))


@needs_lib
@unittest.skipUnless(oencode.has_png_decode(), "liboencode.so predates oenc_png_decode")
class PngDecodeTest(unittest.TestCase):
    def pillow(self, arr, **options):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", **options)
        return buf.getvalue()

    def test_matches_pillow_for_every_filter_and_layout(self):
        rng = np.random.default_rng(11)
        images = [scene(61, 97), scene(128, 31, seed=3), rng.integers(0, 256, (40, 53, 4), dtype=np.uint8),
                  rng.integers(0, 256, (1, 1, 4), dtype=np.uint8), rng.integers(0, 256, (5, 1, 4), dtype=np.uint8),
                  rng.integers(0, 256, (1, 7, 4), dtype=np.uint8), rng.integers(0, 256, (9, 2, 4), dtype=np.uint8)]
        for index, rgba in enumerate(images):
            for pixels in (rgba, np.ascontiguousarray(rgba[:, :, :3])):
                encodings = [oencode.png(pixels, 1), self.pillow(pixels), self.pillow(pixels, compress_level=0)]
                if oencode.has_png_mt():
                    encodings.append(oencode.png_mt(pixels, 4, 8))
                for which, data in enumerate(encodings):
                    with self.subTest(image=index, channels=pixels.shape[2], encoding=which):
                        w, h, has_alpha = oencode.png_size(data)
                        self.assertEqual((h, w, has_alpha), (*pixels.shape[:2], pixels.shape[2] == 4))
                        reference = Image.open(io.BytesIO(data))
                        np.testing.assert_array_equal(oencode.png_decode(data, w, h, 3), np.asarray(reference.convert("RGB")))
                        np.testing.assert_array_equal(oencode.png_decode(data, w, h, 4), np.asarray(reference.convert("RGBA")))

    def test_each_filter_type_alone(self):
        import struct
        import zlib

        rng = np.random.default_rng(2)
        for channels in (3, 4):
            arr = rng.integers(0, 256, (12, 19, channels), dtype=np.uint8)
            h, w = arr.shape[:2]
            for kind in range(5):
                rows = []
                prev = np.zeros(w * channels, np.int32)
                for y in range(h):
                    cur = arr[y].reshape(-1).astype(np.int32)
                    left = np.concatenate([np.zeros(channels, np.int32), cur[:-channels]])
                    upleft = np.concatenate([np.zeros(channels, np.int32), prev[:-channels]])
                    p = left + prev - upleft
                    pa, pb, pc = abs(p - left), abs(p - prev), abs(p - upleft)
                    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, upleft))
                    predictor = [0, left, prev, (left + prev) // 2, paeth][kind]
                    rows.append(bytes([kind]) + ((cur - predictor) & 0xff).astype(np.uint8).tobytes())
                    prev = cur

                def chunk(tag, body):
                    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

                packed = zlib.compress(b"".join(rows))
                data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2 if channels == 3 else 6, 0, 0, 0))
                        + chunk(b"IDAT", packed[:40]) + chunk(b"IDAT", packed[40:]) + chunk(b"IEND", b""))
                with self.subTest(channels=channels, filter=kind):
                    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(data))), arr)
                    np.testing.assert_array_equal(oencode.png_decode(data, w, h, channels), arr)

    def test_leaves_other_pngs_to_the_caller(self):
        arr = scene(40, 40)
        rgb = Image.fromarray(arr[:, :, :3])
        exif = Image.Exif()
        exif[0x0112] = 6
        others = {
            "palette": (rgb.quantize(16), {}),
            "gray": (Image.fromarray(arr[:, :, 0]), {}),
            "gray-alpha": (Image.fromarray(arr[:, :, [0, 3]], "LA"), {}),
            "16-bit": (Image.fromarray(arr[:, :, 0].astype(np.uint16) * 257), {}),
            "transparency": (rgb, {"transparency": (1, 2, 3)}),
            "exif": (rgb, {"exif": exif.tobytes()}),
            "animated": (rgb, {"save_all": True, "append_images": [rgb.rotate(90)]}),
        }
        for name, (image, options) in others.items():
            with self.subTest(name=name):
                buf = io.BytesIO()
                image.save(buf, format="PNG", **options)
                self.assertIsNone(oencode.png_size(buf.getvalue()))
        self.assertIsNone(oencode.png_size(b"RIFF" + bytes(60)))
        self.assertIsNotNone(oencode.png_size(self.pillow(arr, dpi=(72, 72), icc_profile=b"x" * 32)))

    def test_corrupt_data_fails(self):
        data = bytearray(self.pillow(scene(64, 64)))
        w, h, _ = oencode.png_size(bytes(data))
        with self.assertRaises(RuntimeError):
            oencode.png_decode(bytes(data[:len(data) // 2]), w, h)
        data[len(data) // 2] ^= 0x55
        with self.assertRaises(RuntimeError):
            oencode.png_decode(bytes(data), w, h)


@needs_lib
@unittest.skipUnless(oencode.has_webp_decode(), "liboencode.so predates oenc_webp_decode")
class WebpDecodeTest(unittest.TestCase):
    def encoded(self, arr, **options):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="WEBP", **options)
        return buf.getvalue()

    def test_matches_pillow_pixels(self):
        arr = scene(131, 149)
        cases = [self.encoded(arr, quality=85), self.encoded(arr[:, :, :3], quality=90),
                 self.encoded(arr, lossless=True), self.encoded(arr[:, :, :3], lossless=True),
                 self.encoded(scene(1, 1)[:, :, :3], quality=80)]
        for i, data in enumerate(cases):
            for threads in (1, 8):
                with self.subTest(case=i, threads=threads):
                    w, h, has_alpha = oencode.webp_size(data)
                    reference = Image.open(io.BytesIO(data))
                    self.assertEqual((w, h), reference.size)
                    self.assertEqual(has_alpha, reference.mode == "RGBA")
                    np.testing.assert_array_equal(oencode.webp_decode(data, w, h, 3, threads),
                                                  np.asarray(reference.convert("RGB")))
                    np.testing.assert_array_equal(oencode.webp_decode(data, w, h, 4, threads),
                                                  np.asarray(reference.convert("RGBA")))

    def test_leaves_other_inputs_to_the_caller(self):
        arr = scene(40, 40)
        png = io.BytesIO()
        Image.fromarray(arr).save(png, format="PNG")
        self.assertIsNone(oencode.webp_size(png.getvalue()))
        self.assertIsNone(oencode.webp_size(b"RIFF" + bytes(4) + b"WEBP" + bytes(40)))
        animated = io.BytesIO()
        frames = [Image.fromarray(arr), Image.fromarray(arr[::-1].copy())]
        frames[0].save(animated, format="WEBP", save_all=True, append_images=frames[1:], duration=50)
        self.assertIsNone(oencode.webp_size(animated.getvalue()))
        exif = Image.Exif()
        exif[0x0112] = 6
        self.assertIsNone(oencode.webp_size(self.encoded(arr[:, :, :3], quality=80, exif=exif.tobytes())))

    def test_truncated_data_fails(self):
        data = self.encoded(scene(64, 64), quality=80)
        w, h, _ = oencode.webp_size(data)
        with self.assertRaises(RuntimeError):
            oencode.webp_decode(data[:len(data) // 2], w, h)


if __name__ == "__main__":
    unittest.main()
