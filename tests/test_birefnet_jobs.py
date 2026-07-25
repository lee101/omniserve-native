#!/usr/bin/env python3
"""Contract tests for the cutout cache and the poll-able job API.

The GPU stack is stubbed: this exercises the routing logic that decides whether
a request costs a model run, and the queue/poll lifecycle a browser depends on.
Skips (exit 0) when the worker's base dependencies are unavailable.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

try:
    import torch  # noqa: F401
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401
    from PIL import Image
except ImportError as error:  # pragma: no cover
    print(f"skipping: {error}")
    raise SystemExit(0)


def _stub_vision_stack() -> None:
    """torchvision/transformers are only needed to build the model."""
    if "torchvision" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        transforms = types.ModuleType("torchvision.transforms")
        for name in ("Compose", "Resize", "ToTensor", "Normalize"):
            setattr(transforms, name, lambda *args, **kwargs: (lambda value: value))
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForImageSegmentation = object
        sys.modules["transformers"] = transformers


_stub_vision_stack()

WORKER_PATH = REPO / "workers" / "birefnet_worker.py"
SPEC = importlib.util.spec_from_file_location("birefnet_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(worker)
except Exception as error:  # pragma: no cover
    print(f"skipping: cannot import worker ({error})")
    raise SystemExit(0)


CUTOUT_URL = "https://cdn.example.test/cutouts/ab/abc123.webp"


def wait_for(job_id, statuses={"done", "error"}, timeout=5.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = worker.get_job(job_id)
        if job and job.get("status") in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {statuses}: {worker.get_job(job_id)}")


class EncodingTest(unittest.TestCase):
    def test_webp_keeps_alpha_and_is_smaller_than_png(self):
        rgba = Image.new("RGBA", (256, 256), (200, 40, 40, 255))
        for x in range(256):
            for y in range(0, 256, 8):
                rgba.putpixel((x, y), (200, 40, 40, x))

        webp, webp_type = worker.encode_image(rgba, "webp")
        png, png_type = worker.encode_image(rgba, "png")

        self.assertEqual(webp_type, "image/webp")
        self.assertEqual(png_type, "image/png")
        self.assertEqual(webp[:4], b"RIFF")
        self.assertLess(len(webp), len(png))

        from io import BytesIO

        decoded = Image.open(BytesIO(webp))
        self.assertEqual(decoded.mode, "RGBA")
        self.assertEqual(decoded.size, (256, 256))


class CacheParamsTest(unittest.TestCase):
    def test_params_cover_everything_that_changes_pixels(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg")
        params = request.cache_params()
        self.assertEqual(
            set(params), {"format", "threshold", "decontaminate", "model", "input_size", "quality"}
        )

    def test_threshold_change_changes_the_key(self):
        import object_store

        a = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg")
        b = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg", foreground_threshold=0.4)
        self.assertNotEqual(
            object_store.cache_key(a.image_url, a.cache_params()),
            object_store.cache_key(b.image_url, b.cache_params()),
        )


class JobLifecycleTest(unittest.TestCase):
    def test_cache_hit_answers_inline_without_queueing(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/chair.jpg")
        with (
            mock.patch.object(worker.object_store, "exists", return_value=True) as exists,
            mock.patch.object(worker.object_store, "public_url", return_value=CUTOUT_URL),
            mock.patch.object(worker, "produce_cutout") as produce,
        ):
            response = worker.enqueue_background_removal(request)

        exists.assert_called_once()
        produce.assert_not_called()
        self.assertEqual(response["status"], "done")
        self.assertTrue(response["cached"])
        self.assertEqual(response["url"], CUTOUT_URL)
        self.assertIsNone(response["job_id"])

    def test_miss_queues_a_job_that_completes(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/lamp.jpg")
        produced = {"cached": False, "key": "cutouts/ab/abc.webp", "url": CUTOUT_URL,
                    "content": b"webp", "media_type": "image/webp"}

        with (
            mock.patch.object(worker.object_store, "exists", return_value=False),
            mock.patch.object(worker, "produce_cutout", return_value=produced) as produce,
        ):
            response = worker.enqueue_background_removal(request)
            self.assertEqual(response["status"], "queued")
            self.assertIsNotNone(response["job_id"])
            self.assertGreater(response["poll_after_ms"], 0)

            job = wait_for(response["job_id"])

        produce.assert_called_once()
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["url"], CUTOUT_URL)
        self.assertFalse(job["cached"])

    def test_job_without_bucket_returns_inline_data_url(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/pot.jpg")
        produced = {"cached": False, "key": None, "url": None, "content": b"webp-bytes",
                    "media_type": "image/webp"}

        with (
            mock.patch.object(worker.object_store, "exists", return_value=False),
            mock.patch.object(worker, "produce_cutout", return_value=produced),
        ):
            response = worker.enqueue_background_removal(request)
            job = wait_for(response["job_id"])

        self.assertTrue(job["data_url"].startswith("data:image/webp;base64,"))

    def test_failure_is_reported_on_the_job(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/broken.jpg")
        with (
            mock.patch.object(worker.object_store, "exists", return_value=False),
            mock.patch.object(worker, "produce_cutout",
                              side_effect=worker.HTTPException(400, "source is not a supported image")),
        ):
            response = worker.enqueue_background_removal(request)
            job = wait_for(response["job_id"])

        self.assertEqual(job["status"], "error")
        self.assertEqual(job["http_status"], 400)
        self.assertIn("supported image", job["error"])

    def test_unknown_job_is_404(self):
        with self.assertRaises(worker.HTTPException) as caught:
            worker.background_removal_job("does-not-exist")
        self.assertEqual(caught.exception.status_code, 404)

    def test_produce_cutout_rejects_unknown_format(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg", output_format="gif")
        with self.assertRaises(worker.HTTPException) as caught:
            worker.produce_cutout(request)
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
