"""workloads/qwen_image.py: gateway tier fields are accepted and the Nyquist notch behaves."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workloads"))
import qwen_image as q

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError:  # the worker image carries both; skip on a bare host
    np = None


class WorkerInputs(unittest.TestCase):
    def test_gateway_fields_are_not_rejected(self):
        seen = {}
        original = q.generate
        q.generate = lambda values: seen.update(values) or {"ok": True}
        try:
            body = {"prompt": "x", "turbo": False, "notch": True, "cache_threshold": 0.15, "cache_end": 0.9,
                    "teleport": True, "cache": False, "model": "ra2", "quality": "hq", "steps": 20}
            self.assertEqual(q.handler({"input": body}), {"ok": True})
            self.assertEqual(seen["turbo"], False)
        finally:
            q.generate = original

    def test_unknown_inputs_still_rejected(self):
        with self.assertRaises(ValueError):
            q.handler({"input": {"prompt": "x", "bogus": 1}})


@unittest.skipIf(np is None, "numpy/Pillow not installed")
class Notch(unittest.TestCase):
    def test_removes_two_pixel_lattice_and_keeps_flat_areas(self):
        h = w = 64
        base = np.full((h, w, 3), 120.0)
        checker = ((np.indices((h, w)).sum(0) % 2) * 2 - 1)[..., None] * 20.0
        out = np.asarray(q.nyquist_notch(Image.fromarray((base + checker).astype(np.uint8)))).astype(float)
        self.assertLess(abs(out[8:-8, 8:-8] - 120).max(), 1.5)
        flat = np.asarray(q.nyquist_notch(Image.fromarray(base.astype(np.uint8))))
        self.assertEqual(int(flat.min()), 120)
        self.assertEqual(int(flat.max()), 120)

    def test_matches_c_taps(self):
        taps = np.array(q._NOTCH_TAPS)
        self.assertEqual(int(taps.sum()), 64)  # DC gain 1 after the 64 * 64 scale


if __name__ == "__main__":
    unittest.main()
