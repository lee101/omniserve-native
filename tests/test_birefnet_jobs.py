#!/usr/bin/env python3
"""Contract tests for the cutout cache and the poll-able job API.

The GPU stack is stubbed: this exercises the routing logic that decides whether
a request costs a model run, and the queue/poll lifecycle a browser depends on.
Skips (exit 0) when the worker's base dependencies are unavailable.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

from gpu_admission import AdaptiveCudaGuard

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
    def test_webp_is_the_default_output_format(self):
        self.assertEqual(worker.DEFAULT_FORMAT, "webp")
        self.assertEqual(worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg").output_format,
                         "webp")
        self.assertEqual(worker.ForegroundGenerationRequest(prompt="person").output_format,
                         "webp")

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

    def test_webp_blackens_rgb_under_fully_transparent_pixels(self):
        rgba = Image.new("RGBA", (3, 1), (240, 30, 10, 0))
        rgba.putpixel((1, 0), (12, 34, 56, 255))

        normalized = worker._black_fully_transparent(rgba)

        self.assertEqual(normalized.getpixel((0, 0)), (0, 0, 0, 0))
        self.assertEqual(normalized.getpixel((1, 0)), (12, 34, 56, 255))
        self.assertEqual(normalized.getpixel((2, 0)), (0, 0, 0, 0))


class CacheParamsTest(unittest.TestCase):
    def test_params_cover_everything_that_changes_pixels(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg")
        params = request.cache_params()
        self.assertEqual(
            set(params),
            {"format", "threshold", "decontaminate", "model", "input_size", "quality",
             "webp_method", "dtype", "return_background", "background",
             "background_prompt", "background_strength", "transparent_rgb"},
        )
        self.assertEqual(params["transparent_rgb"], "black")

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
        produced = {"cached": False, "media_type": "image/webp",
                    "artifacts": {"cutout": {"key": "cutouts/ab/abc.webp", "url": CUTOUT_URL,
                                             "content": b"webp"}}}

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
        produced = {"cached": False, "media_type": "image/webp",
                    "artifacts": {"cutout": {"key": None, "url": None,
                                             "content": b"webp-bytes"}}}

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


class ForegroundGenerationTest(unittest.TestCase):
    def test_cutout_can_use_an_in_memory_stage_one_image(self):
        image = Image.new("RGB", (8, 8), "white")
        request = worker.RemoveBackgroundRequest(image_url="generated:test")
        produced = {"cutout": b"cutout-webp"}
        with (
            mock.patch.object(worker, "object_store", None),
            mock.patch.object(worker, "read_image", side_effect=AssertionError("decoded again")),
            mock.patch.object(worker, "remove_background", return_value=produced) as remove,
        ):
            result = worker.produce_cutout(request, source_image=image)

        remove.assert_called_once_with(image, request)
        self.assertFalse(result["cached"])
        self.assertEqual(result["artifacts"]["cutout"]["content"], b"cutout-webp")

    def test_generation_and_cutout_are_one_job(self):
        request = worker.ForegroundGenerationRequest(
            prompt="full body harbour worker on a plain backdrop", width=640, height=960, seed=42,
        )
        generated = Image.new("RGB", (640, 960), "white")
        produced = {"cached": False, "media_type": "image/webp",
                    "artifacts": {"cutout": {"key": None, "url": None,
                                                "content": b"cutout-webp"}}}
        with (
            mock.patch.object(worker, "generate_backdrop", return_value=generated) as generate,
            mock.patch.object(worker, "produce_cutout", return_value=produced) as cutout,
        ):
            response = worker.enqueue_foreground_generation(request)
            job = wait_for(response["job_id"])

        generate.assert_called_once_with(request.prompt, 640, 960, None, 0.0, 42)
        cutout.assert_called_once()
        source_request = cutout.call_args.args[0]
        self.assertTrue(source_request.image_url.startswith("generated:"))
        self.assertIs(cutout.call_args.kwargs["source_image"], generated)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["seed"], 42)
        self.assertEqual(job["source_width"], 640)
        self.assertTrue(job["data_url"].startswith("data:image/webp;base64,"))

    def test_generation_rejects_unknown_format_before_queueing(self):
        request = worker.ForegroundGenerationRequest(prompt="person", output_format="gif")
        with self.assertRaises(worker.HTTPException) as caught:
            worker.enqueue_foreground_generation(request)
        self.assertEqual(caught.exception.status_code, 400)


class ModelOptimizationTest(unittest.TestCase):
    def test_compiled_runtime_failure_falls_back_to_eager(self):
        class Broken:
            def __call__(self, _tensor):
                raise RuntimeError("compiled graph failed")

        class Eager:
            def __call__(self, tensor):
                return [torch.zeros((1, 1, tensor.shape[-2], tensor.shape[-1]))]

        previous = (worker.runtime.model, worker.runtime.eager_model, worker.runtime.transform,
                    worker.runtime.dtype, worker.runtime.engine, worker.runtime.compile_error)
        try:
            worker.runtime.model = Broken()
            worker.runtime.eager_model = Eager()
            worker.runtime.transform = lambda _image: torch.zeros((3, 8, 8))
            worker.runtime.dtype = torch.float32
            with mock.patch.object(worker, "DEVICE", "cpu"):
                device_mask, host_mask = worker.segment(Image.new("RGB", (8, 8)), 0.0)
            self.assertIsNone(device_mask)
            self.assertEqual(host_mask.shape, (8, 8))
            self.assertIs(worker.runtime.model, worker.runtime.eager_model)
            self.assertIn("compiled graph failed", worker.runtime.compile_error)
            self.assertIn("compile-fallback", worker.runtime.engine)
        finally:
            (worker.runtime.model, worker.runtime.eager_model, worker.runtime.transform,
             worker.runtime.dtype, worker.runtime.engine,
             worker.runtime.compile_error) = previous

    def test_model_lease_uses_shared_broker(self):
        granted = mock.Mock(status_code=200)
        granted.json.return_value = {"granted": True, "lease_id": "lv-biref", "mb": 3584}
        granted.raise_for_status.return_value = None
        released = mock.Mock()
        released.raise_for_status.return_value = None
        with (
            mock.patch.object(worker, "VRAM_BROKER_URL", "http://127.0.0.1:8791"),
            mock.patch.object(worker.requests, "post", side_effect=[granted, released]) as post,
        ):
            worker.runtime.vram_lease = worker._acquire_model_vram_lease()
            worker._release_model_vram_lease()
        self.assertEqual(post.call_count, 2)
        self.assertTrue(post.call_args_list[0].args[0].endswith("/v1/gpu/lease"))
        self.assertTrue(post.call_args_list[1].args[0].endswith("/v1/gpu/release"))


class AdaptiveCudaGuardTest(unittest.TestCase):
    def test_oom_raises_floor_and_cools_down(self):
        guard = AdaptiveCudaGuard(
            1536, oom_margin_mib=512, backoff_seconds=5.0, backoff_max_seconds=60.0,
        )
        guard.note_oom(free_mib=2000, total_mib=32000, now=100.0)
        cooling = guard.capacity(free_mib=10000, total_mib=32000, now=102.0)
        recovered = guard.capacity(free_mib=2600, total_mib=32000, now=106.0)

        self.assertFalse(cooling["ready"])
        self.assertEqual(cooling["cooldown_seconds"], 3.0)
        self.assertEqual(cooling["required_free_mib"], 2512)
        self.assertTrue(recovered["ready"])
        self.assertEqual(recovered["total_ooms"], 1)

    def test_guard_recovers_after_sustained_success(self):
        guard = AdaptiveCudaGuard(1536, oom_margin_mib=512, recovery_successes=2)
        guard.note_oom(free_mib=2000, total_mib=32000, now=100.0)
        guard.note_success()
        guard.note_success()
        state = guard.capacity(free_mib=2100, total_mib=32000, now=1000.0)

        self.assertTrue(state["ready"])
        self.assertEqual(state["required_free_mib"], 2000)
        self.assertEqual(state["backoff_level"], 0)


@unittest.skipIf(worker.video_matting is None, "video matting dependencies unavailable")
class VideoBackgroundRemovalTest(unittest.TestCase):
    def test_video_encoder_blacks_transparent_rgb(self):
        import numpy as np

        rgb = np.array([[[240, 30, 10], [12, 34, 56]]], dtype=np.uint8)
        alpha = np.array([[0, 255]], dtype=np.uint8)

        changed = worker.video_matting._black_transparent_rgb(rgb, alpha)

        self.assertEqual(changed, 1)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(rgb[0, 1], [12, 34, 56])

    def request(self):
        return worker.VideoBackgroundRequest(
            video_url="https://cdn.example.test/person.webm",
            output_upload_url="https://upload.example.test/result",
            output_public_url="https://cdn.example.test/result.webm",
        )

    def test_person_route_completes_with_private_video(self):
        result = {"fallback_required": False, "route": "local-rvm-person",
                  "video_url": "https://cdn.example.test/result.webm",
                  "duration_seconds": 2.0, "metrics": {"inference_fps": 40.0}}
        with mock.patch.object(worker.video_matting, "process", return_value=result):
            response = worker.enqueue_video_background_removal(self.request())
            job = wait_for(response["job_id"])
        self.assertEqual(job["status"], "done")
        self.assertFalse(job["fallback_required"])
        self.assertEqual(job["route"], "local-rvm-person")

    def test_non_person_route_requests_standby_without_error(self):
        result = {"fallback_required": True, "fallback_reason": "no person detected",
                  "route": "standby-general-matting", "duration_seconds": 2.0,
                  "metrics": {"person_detection": {"detected": False}}}
        with mock.patch.object(worker.video_matting, "process", return_value=result):
            response = worker.enqueue_video_background_removal(self.request())
            job = wait_for(response["job_id"])
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["fallback_required"])
        self.assertEqual(job["route"], "standby-general-matting")

    def test_full_local_queue_rejects_for_provider_spillover(self):
        seeded = []
        now = __import__("time").time()
        with worker._jobs_lock:
            for index in range(worker.VIDEO_JOB_MAX_PENDING):
                job_id = f"capacity-test-{index}"
                seeded.append(job_id)
                worker._jobs[job_id] = {"job_id": job_id, "kind": "video", "status": "queued",
                                        "created": now, "updated": now}
        try:
            with self.assertRaises(worker.HTTPException) as caught:
                worker.enqueue_video_background_removal(self.request())
            self.assertEqual(caught.exception.status_code, 429)
            self.assertEqual(caught.exception.headers["Retry-After"], "5")
        finally:
            with worker._jobs_lock:
                for job_id in seeded:
                    worker._jobs.pop(job_id, None)

    def test_health_exposes_queue_gpu_and_throughput(self):
        with (
            mock.patch.object(worker.torch.cuda, "is_available", return_value=True),
            mock.patch.object(worker.torch.cuda, "mem_get_info", return_value=(2 << 30, 32 << 30)),
            mock.patch.object(worker.torch.cuda, "memory_allocated", return_value=512 << 20),
            mock.patch.object(worker.torch.cuda, "memory_reserved", return_value=768 << 20),
            mock.patch.object(worker.video_matting, "health",
                              return_value={"person_detector": "ready", "rvm_loaded": False}),
        ):
            capacity = worker.health()["video_capacity"]
        self.assertTrue(capacity["accepting"])
        self.assertEqual(capacity["max_pending"], worker.VIDEO_JOB_MAX_PENDING)
        self.assertEqual(capacity["gpu"]["free_mib"], 2048)
        self.assertIn("average_seconds", capacity["stats"])

    def test_rvm_lease_uses_the_shared_native_broker(self):
        granted = mock.Mock()
        granted.json.return_value = {"granted": True, "lease_id": "lv-test", "mb": 1536}
        granted.raise_for_status.return_value = None
        released = mock.Mock()
        released.raise_for_status.return_value = None
        with (
            mock.patch.object(worker.video_matting, "RVM_VRAM_BROKER_URL", "http://127.0.0.1:8791"),
            mock.patch.object(worker.video_matting.requests, "post",
                              side_effect=[granted, released]) as post,
        ):
            lease = worker.video_matting._acquire_vram_lease(1536)
            worker.video_matting._release_vram_lease(lease)

        self.assertTrue(lease["granted"])
        self.assertEqual(lease["lease_id"], "lv-test")
        self.assertEqual(post.call_count, 2)
        self.assertTrue(post.call_args_list[0].args[0].endswith("/v1/gpu/lease"))
        self.assertTrue(post.call_args_list[1].args[0].endswith("/v1/gpu/release"))

    def test_error_event_is_structured_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            with mock.patch.object(worker, "EVENTS_PATH", str(path)):
                worker._emit_event("error", "video_job_error", job_id="safe-id", error="boom")
            payload = json.loads(path.read_text().strip())
        self.assertEqual(payload["event"], "video_job_error")
        self.assertEqual(payload["job_id"], "safe-id")


class BackgroundRequestTest(unittest.TestCase):
    def test_artifacts_do_not_share_a_cache_key(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                                 return_background=True)
        self.assertNotEqual(worker.artifact_key(request, "cutout"),
                            worker.artifact_key(request, "background"))

    def test_asking_for_the_backdrop_changes_the_cutout_key(self):
        """Otherwise a plain cutout would serve from a run that also solved the
        backdrop, or worse, the reverse."""
        plain = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg")
        with_bg = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                                 return_background=True)
        self.assertNotEqual(worker.artifact_key(plain, "cutout"),
                            worker.artifact_key(with_bg, "cutout"))

    def test_backdrop_prompt_changes_the_composite_key(self):
        a = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                           background_prompt="a beach at dusk")
        b = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                           background_prompt="a studio backdrop")
        self.assertNotEqual(worker.artifact_key(a, "composite"),
                            worker.artifact_key(b, "composite"))

    def test_generated_backdrop_is_refused_on_the_synchronous_endpoint(self):
        """Diffusion holds a gateway permit for its whole duration, so it has to
        go through the job lane or it occupies an interactive slot."""
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                                 background_prompt="a beach at dusk")
        with self.assertRaises(worker.HTTPException) as caught:
            worker.background_removal(request)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("/jobs", caught.exception.detail)

    def test_wants_extras_only_when_there_is_more_than_one_image(self):
        plain = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg")
        transparent = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                                     background="transparent")
        self.assertFalse(plain.wants_extras())
        self.assertTrue(transparent.wants_extras())
        self.assertTrue(worker.RemoveBackgroundRequest(
            image_url="https://x.test/a.jpg", return_background=True).wants_extras())

    def test_colour_parsing(self):
        self.assertEqual(worker.parse_colour("#fff"), (1.0, 1.0, 1.0))
        self.assertEqual(worker.parse_colour("#000000"), (0.0, 0.0, 0.0))
        self.assertEqual(worker.parse_colour("white"), (1.0, 1.0, 1.0))
        red = worker.parse_colour("#FF0000")
        self.assertEqual(red, (1.0, 0.0, 0.0))
        self.assertIsNone(worker.parse_colour("https://x.test/bg.jpg"))
        self.assertIsNone(worker.parse_colour("#12345"))

    def test_multi_artifact_result_is_json(self):
        request = worker.RemoveBackgroundRequest(image_url="https://x.test/a.jpg",
                                                 return_background=True)
        produced = {"cached": False, "media_type": "image/webp", "artifacts": {
            "cutout": {"key": "k1", "url": CUTOUT_URL, "content": b"a"},
            "background": {"key": "k2", "url": None, "content": b"b"},
        }}
        with mock.patch.object(worker, "produce_cutout", return_value=produced):
            response = worker.background_removal(request)

        self.assertEqual(response.media_type, "application/json")
        import json as _json
        body = _json.loads(response.body)
        self.assertEqual(body["cutout"]["url"], CUTOUT_URL)
        self.assertTrue(body["background"]["data_url"].startswith("data:image/webp;base64,"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
