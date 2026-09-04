#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
spec = importlib.util.spec_from_file_location("depth_anything_worker", ROOT / "workers" / "depth_anything_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"depth_map": "data:image/png;base64,eA==", "model": worker.MODEL_ID}


class DepthWorkerTest(unittest.TestCase):
    def test_normalization_clips_outliers_and_preserves_near_white(self):
        depth = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 1000.0]], dtype=np.float32)
        normalized = worker.normalize_depth(depth, True)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreater(normalized[1, 2], normalized[0, 0])
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_png16_preserves_more_than_eight_bits(self):
        values = np.linspace(0, 1, 1024, dtype=np.float32).reshape(32, 32)
        content, media_type = worker.encode_depth(values, "png16")
        decoded = np.asarray(Image.open(worker.io.BytesIO(content)))
        self.assertEqual(media_type, "image/png")
        self.assertGreater(len(np.unique(decoded)), 256)

    def test_fast_png16_is_pixel_exact(self):
        values = np.random.default_rng(0).random((128, 128), dtype=np.float32)
        content, _ = worker.encode_depth(values, "png16")
        decoded = np.asarray(Image.open(worker.io.BytesIO(content)))
        expected = np.rint(values * 65535).astype(np.uint16)
        np.testing.assert_array_equal(decoded, expected)

    def test_busy_worker_spills_to_runpod(self):
        request = worker.DepthRequest(image_url="https://cdn.test/image.webp")
        with mock.patch.object(worker, "OVERFLOW_URL", "https://api.runpod.test/v1/depth-estimations"), mock.patch.object(worker.session, "post", return_value=Response()) as post:
            result = worker.run_overflow(request)
        self.assertEqual(result["worker"], "runpod")
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
