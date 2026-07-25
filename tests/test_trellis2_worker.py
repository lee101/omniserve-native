#!/usr/bin/env python3
"""Fast contract tests for the cold-start 3D adapter."""

from __future__ import annotations

from http import HTTPStatus
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


WORKER_PATH = Path(__file__).parents[1] / "workers" / "trellis2_worker.py"
SPEC = importlib.util.spec_from_file_location("trellis2_worker", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class IntegerValueTests(unittest.TestCase):
    def test_accepts_canonical_integers(self) -> None:
        self.assertEqual(worker.integer_value({"seed": 7}, "seed", 42), 7)
        self.assertEqual(worker.integer_value({"seed": "7"}, "seed", 42), 7)
        self.assertEqual(worker.integer_value({}, "seed", 42), 42)

    def test_rejects_ambiguous_values(self) -> None:
        for value in (True, 3.5, "3.5", "03", [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.integer_value({"seed": value}, "seed", 42)


class RequestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "model": worker.MODEL_TRELLIS,
            "image_url": "https://assets.example.test/object.webp",
        }
        self.public_url = "http://127.0.0.1:8791/v1/3d"

    def run_validation(self, **updates: object) -> tuple[int, dict]:
        payload = dict(self.payload)
        payload.update(updates)
        with mock.patch.object(worker, "validate_public_url", side_effect=lambda value: value):
            return worker.run_job(payload, self.public_url)

    def test_rejects_non_integer_resolution_without_touching_gpu(self) -> None:
        with mock.patch.object(worker, "gpu_memory_mib") as gpu:
            status, body = self.run_validation(resolution="large")
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertIn("integer", body["error"])
        gpu.assert_not_called()

    def test_rejects_out_of_range_decimation(self) -> None:
        status, body = self.run_validation(decimation_target=9_999)
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertIn("between", body["error"])

    def test_busy_gpu_returns_retry_metadata(self) -> None:
        with (
            mock.patch.object(worker, "runtime_installed", return_value=True),
            mock.patch.object(worker, "model_dependency_issue", return_value=None),
            mock.patch.object(worker, "gpu_memory_mib", return_value=(8_000, 32_607)),
        ):
            status, body = self.run_validation(resolution=512)
        self.assertEqual(status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(body["error"], "gpu_busy")
        self.assertEqual(body["retry_after_seconds"], 60)

    def test_missing_optional_model_fails_before_gpu_probe(self) -> None:
        with (
            mock.patch.object(worker, "runtime_installed", return_value=False),
            mock.patch.object(worker, "gpu_memory_mib") as gpu,
        ):
            status, body = self.run_validation(model=worker.MODEL_PIXAL, resolution=1024)
        self.assertEqual(status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(body["error"], "model_not_installed")
        gpu.assert_not_called()

    def test_missing_gated_dependency_fails_before_gpu_probe(self) -> None:
        with (
            mock.patch.object(worker, "runtime_installed", return_value=True),
            mock.patch.object(
                worker,
                "model_dependency_issue",
                return_value={
                    "dependency": worker.DINO_TRELLIS,
                    "approval_url": f"https://huggingface.co/{worker.DINO_TRELLIS}",
                    "message": "approval required",
                },
            ),
            mock.patch.object(worker, "gpu_memory_mib") as gpu,
        ):
            status, body = self.run_validation(resolution=512)
        self.assertEqual(status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(body["error"], "model_dependency_missing")
        self.assertEqual(body["dependency"], worker.DINO_TRELLIS)
        gpu.assert_not_called()

    def test_rejects_private_image_url(self) -> None:
        status, body = worker.run_job(
            {
                "model": worker.MODEL_TRELLIS,
                "image_url": "http://127.0.0.1/private.png",
            },
            self.public_url,
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertIn("private", body["error"])


class GPUCoordinatorTests(unittest.TestCase):
    def test_holds_wrap_job_and_release_in_reverse_order(self) -> None:
        calls: list[tuple[str, str]] = []
        with (
            mock.patch.dict(
                worker.os.environ,
                {"OMNISERVE_3D_GPU_COORDINATORS": "http://image-a,http://image-b"},
            ),
            mock.patch.object(
                worker,
                "post_gpu_coordinator",
                side_effect=lambda base, action: calls.append((base, action)),
            ),
        ):
            status, body = worker.run_with_gpu_holds(
                lambda: (HTTPStatus.OK, {"status": "generated"})
            )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(body["status"], "generated")
        self.assertEqual(
            calls,
            [
                ("http://image-a", "hold"),
                ("http://image-b", "hold"),
                ("http://image-b", "release"),
                ("http://image-a", "release"),
            ],
        )

    def test_partial_hold_failure_releases_and_defers(self) -> None:
        calls: list[tuple[str, str]] = []

        def post(base: str, action: str) -> None:
            calls.append((base, action))
            if base == "http://image-b" and action == "hold":
                raise OSError("unavailable")

        with (
            mock.patch.dict(
                worker.os.environ,
                {"OMNISERVE_3D_GPU_COORDINATORS": "http://image-a,http://image-b"},
            ),
            mock.patch.object(worker, "post_gpu_coordinator", side_effect=post),
        ):
            status, body = worker.run_with_gpu_holds(
                lambda: self.fail("job must not run without all GPU holds")
            )
        self.assertEqual(status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(body["error"], "gpu_coordination_failed")
        self.assertEqual(
            calls,
            [
                ("http://image-a", "hold"),
                ("http://image-b", "hold"),
                ("http://image-a", "release"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
