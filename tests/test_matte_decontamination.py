#!/usr/bin/env python3
"""End-to-end check that colour decontamination actually removes backdrop bleed.

Builds a synthetic green-screen composite with a known foreground colour and a
soft alpha edge, runs the C/CUDA foreground estimator through the same ctypes
wrapper the BiRefNet worker uses, and asserts that the recovered colours are
closer to the truth than the observed composite is.

Skips (exit 0) when numpy or libomatte.so is unavailable, so CPU-only CI without
a build still passes.
"""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("skipping: numpy is not installed")
    raise SystemExit(0)

try:
    import omatte
    omatte.cuda_available()  # forces the library load
    if omatte.library_path() is None:
        raise omatte.MatteUnavailable("library not loaded")
except Exception as error:  # pragma: no cover - library not built
    print(f"skipping: {error}")
    raise SystemExit(0)


FOREGROUND = np.array([0.85, 0.35, 0.25], dtype=np.float32)
BACKDROP = np.array([0.05, 0.95, 0.10], dtype=np.float32)  # green screen


def make_case(h=96, w=128, feather=6.0):
    """Solid object with a feathered edge composited over a green screen."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    distance = np.maximum(np.abs(yy - h / 2) / (h / 3), np.abs(xx - w / 2) / (w / 3))
    alpha = np.clip((1.0 - distance) * feather, 0.0, 1.0).astype(np.float32)

    truth = np.broadcast_to(FOREGROUND, (h, w, 3)).astype(np.float32)
    observed = alpha[..., None] * truth + (1.0 - alpha[..., None]) * BACKDROP
    return observed.astype(np.float32), alpha, truth


class DecontaminationTest(unittest.TestCase):
    def setUp(self):
        self.observed, self.alpha, self.truth = make_case()
        # The fringe is where the bleed lives: partially transparent pixels.
        self.fringe = (self.alpha > 0.05) & (self.alpha < 0.95)
        self.assertGreater(self.fringe.sum(), 100, "fixture should have a real soft edge")

    def test_fringe_colour_is_recovered(self):
        estimated = omatte.estimate_foreground(self.observed, self.alpha)

        observed_error = np.abs(self.observed[self.fringe] - self.truth[self.fringe]).mean()
        estimated_error = np.abs(estimated[self.fringe] - self.truth[self.fringe]).mean()

        self.assertLess(estimated_error, observed_error * 0.25,
                        f"expected a large improvement, got {observed_error:.4f} -> {estimated_error:.4f}")
        self.assertLess(estimated_error, 0.05)

    def test_green_bleed_is_removed(self):
        estimated = omatte.estimate_foreground(self.observed, self.alpha)

        observed_green = self.observed[self.fringe][:, 1].mean()
        estimated_green = estimated[self.fringe][:, 1].mean()

        self.assertGreater(observed_green, 0.5, "fixture should start with heavy green bleed")
        self.assertLess(estimated_green, FOREGROUND[1] + 0.1)

    def test_opaque_interior_is_untouched(self):
        estimated = omatte.estimate_foreground(self.observed, self.alpha)
        opaque = self.alpha >= 0.999
        self.assertLess(np.abs(estimated[opaque] - self.truth[opaque]).max(), 0.02)

    def test_backends_agree(self):
        cpu = omatte.estimate_foreground(self.observed, self.alpha, use_cuda=False,
                                         order=omatte.ORDER_RED_BLACK, threads=4)
        cpu_single = omatte.estimate_foreground(self.observed, self.alpha, use_cuda=False,
                                                order=omatte.ORDER_RED_BLACK, threads=1)
        # Thread count must never change the answer.
        self.assertEqual(np.abs(cpu - cpu_single).max(), 0.0)

        if omatte.cuda_available():
            gpu = omatte.estimate_foreground(self.observed, self.alpha, use_cuda=True)
            self.assertLess(np.abs(gpu - cpu).max(), 1e-5)

    def test_recomposite_over_new_background(self):
        """A cutout placed on a white page should not show a green halo."""
        estimated = omatte.estimate_foreground(self.observed, self.alpha)
        white = np.ones(3, dtype=np.float32)

        alpha3 = self.alpha[..., None]
        naive = alpha3 * self.observed + (1 - alpha3) * white
        cleaned = alpha3 * estimated + (1 - alpha3) * white
        ideal = alpha3 * self.truth + (1 - alpha3) * white

        self.assertLess(np.abs(cleaned - ideal).mean(), np.abs(naive - ideal).mean() * 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
