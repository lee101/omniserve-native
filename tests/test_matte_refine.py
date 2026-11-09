#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

try:
    import torch
    import torch.nn.functional as F
    import matte_refine
except (ImportError, OSError) as error:
    print(f"skipping: {error}")
    raise SystemExit(0) if __name__ == "__main__" else unittest.SkipTest(str(error))


def edge_band_reference(alpha_lo, size, radius_lo):
    outside = (alpha_lo < 0.5).float()[None, None]
    near = F.max_pool2d(outside, 2 * radius_lo + 1, stride=1, padding=radius_lo)
    near = matte_refine._box(near, max(1, radius_lo // 2))[0]
    return matte_refine._resize(near, size)[0].clamp(0, 1)


class EdgeBandTest(unittest.TestCase):
    def test_separable_dilation_is_the_square_one(self):
        generator = torch.Generator().manual_seed(5)
        for h, w in ((64, 96), (131, 77), (40, 40)):
            alpha = torch.rand(h, w, generator=generator)
            alpha[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1.0
            for radius in (1, 6, 24, 96):
                with self.subTest(h=h, w=w, radius=radius):
                    size = (h * 2, w * 2)
                    self.assertTrue(torch.equal(matte_refine._edge_band(alpha, size, radius),
                                                edge_band_reference(alpha, size, radius)))


if __name__ == "__main__":
    unittest.main()
