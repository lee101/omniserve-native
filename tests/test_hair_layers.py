#!/usr/bin/env python3
"""Hair-layer path: cutout bootstrap, SAM2 box nesting, skip caching, cache keys.

GPU/model stack is stubbed. Skips (exit 0) when the worker deps are missing.
"""

import inspect
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "workers"))

try:
    import numpy as np
    from PIL import Image
    import test_birefnet_jobs as base  # noqa: E402  (stubs torchvision/transformers, loads worker)
except SystemExit:
    raise
except ImportError as error:  # pragma: no cover
    print(f"skipping: {error}")
    raise SystemExit(0)

import hair_layers  # noqa: E402
import object_store  # noqa: E402

worker = base.worker


class EnsureCutoutTest(unittest.TestCase):
    def test_opaque_input_goes_through_the_real_remove_background_signature(self):
        opaque = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
        seen = {}

        def fake_segment(image, threshold):
            seen["threshold"] = threshold
            return None, np.full((32, 32), 255, dtype=np.uint8)

        with mock.patch.object(worker, "read_image_rgba", return_value=opaque), \
                mock.patch.object(worker, "segment", fake_segment):
            out = worker.ensure_cutout_rgba("https://x.test/a.jpg")
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.size, (32, 32))
        self.assertEqual(seen["threshold"], 0.0)

    def test_already_cutout_input_is_returned_untouched(self):
        cut = Image.new("RGBA", (8, 8), (1, 2, 3, 0))
        with mock.patch.object(worker, "read_image_rgba", return_value=cut), \
                mock.patch.object(worker, "remove_background", side_effect=AssertionError):
            self.assertIs(worker.ensure_cutout_rgba("u"), cut)

    def test_signature_is_image_and_request(self):
        self.assertEqual(list(inspect.signature(worker.remove_background).parameters),
                         ["image", "request"])


class ProduceHairLayersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = (object_store.CACHE_DIR, object_store.BUCKET)
        object_store.CACHE_DIR = self.tmp
        object_store.BUCKET = ""
        self.addCleanup(self._restore)

    def _restore(self):
        object_store.CACHE_DIR, object_store.BUCKET = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skipped_result_is_cached(self):
        calls = []

        def split(image):
            calls.append(1)
            return {"skipped": True, "coverage": 0.01, "backend": "silhouette-split",
                    "sam2_status": "unavailable", "front": None, "back": None}

        fake = types.SimpleNamespace(split_hair_layers=split)
        request = worker.HairLayersRequest(image_url="https://x.test/bald.jpg")
        cutout = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        with mock.patch.object(worker, "hair_layers", fake), \
                mock.patch.object(worker, "object_store", object_store), \
                mock.patch.object(worker, "CACHE_ENABLED", True), \
                mock.patch.object(worker, "read_image_rgba", return_value=cutout):
            first = worker.produce_hair_layers(request)
            second = worker.produce_hair_layers(request)
        self.assertEqual(len(calls), 1)
        self.assertTrue(first["skipped"] and not first["cached"])
        self.assertTrue(second["skipped"] and second["cached"])
        self.assertEqual(second["backend"], "silhouette-split")
        self.assertAlmostEqual(second["coverage"], 0.01)


class BackdropKeyTest(unittest.TestCase):
    def test_intermediate_key_depends_on_output_shaping_params(self):
        keys = []
        with mock.patch.object(worker, "object_store") as store:
            store.cache_key.side_effect = lambda hint, params, **kw: repr(sorted(params.items()))
            store.put.side_effect = lambda key, payload, ctype: key
            image = Image.new("RGB", (4, 4))
            for threshold, decon in ((0.0, True), (0.5, True), (0.0, False)):
                request = worker.RemoveBackgroundRequest(
                    image_url="u", foreground_threshold=threshold, decontaminate=decon)
                keys.append(worker.publish_intermediate(
                    image, request.image_url, worker._backdrop_cache_params(request)))
        self.assertEqual(len(set(keys)), 3)
        for field in ("threshold", "decontaminate", "model", "input_size"):
            self.assertIn(field, keys[0])


class Sam2Test(unittest.TestCase):
    def _runtime(self, processor):
        torch = types.SimpleNamespace(inference_mode=mock.MagicMock())
        model = mock.MagicMock()
        return {"processor": processor, "model": model, "device": "cpu", "torch": torch}

    def test_boxes_are_nested_three_levels(self):
        seen = {}

        def processor(images, input_boxes, return_tensors):
            seen["boxes"] = input_boxes
            raise RuntimeError("stop here")

        alpha = np.zeros((40, 40), dtype=np.uint8)
        alpha[5:35, 10:30] = 255
        with mock.patch.object(hair_layers, "load_sam2", return_value=self._runtime(processor)):
            mask, status = hair_layers.sam2_hair_mask_status(Image.new("RGB", (40, 40)), alpha)
        boxes = seen["boxes"]
        self.assertIsNone(mask)
        self.assertEqual(status, "failed")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(boxes[0]), 1)
        self.assertEqual(len(boxes[0][0]), 4)
        self.assertTrue(all(isinstance(v, float) for v in boxes[0][0]))

    def test_failure_is_reported_in_split_result(self):
        rgba = Image.new("RGBA", (40, 40), (90, 60, 40, 255))
        with mock.patch.object(hair_layers, "sam2_hair_mask_status", return_value=(None, "failed")):
            result = hair_layers.split_hair_layers(rgba)
        self.assertEqual(result["backend"], "silhouette-split")
        self.assertEqual(result["sam2_status"], "failed")

    def test_real_processor_accepts_the_nesting(self):
        try:
            from transformers import Sam2ImageProcessorFast, Sam2Processor
            processor = Sam2Processor(image_processor=Sam2ImageProcessorFast())
        except Exception as error:
            self.skipTest(f"sam2 processor unavailable: {error}")
        out = processor(images=Image.new("RGB", (64, 64)), input_boxes=[[[1.0, 2.0, 30.0, 40.0]]],
                        return_tensors="pt")
        self.assertEqual(tuple(out["input_boxes"].shape), (1, 1, 4))

    def test_load_sam2_is_serialised(self):
        import threading
        import time
        loads = []
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        class Auto:
            @staticmethod
            def from_pretrained(model_id):
                loads.append(1)
                time.sleep(0.05)
                m = mock.MagicMock()
                m.to.return_value = m
                return m

        fake_tf = types.SimpleNamespace(AutoModel=Auto, AutoProcessor=Auto)
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_tf}), \
                mock.patch.object(hair_layers, "_sam2", None), \
                mock.patch.object(hair_layers, "_sam2_failed", False), \
                mock.patch.object(hair_layers, "HAIR_BACKEND", "auto"):
            threads = [threading.Thread(target=hair_layers.load_sam2) for _ in range(4)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.assertEqual(len(loads), 2, "processor + model once, not once per thread")


if __name__ == "__main__":
    unittest.main(verbosity=2)
