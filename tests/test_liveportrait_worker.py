#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
spec = importlib.util.spec_from_file_location("liveportrait_worker", ROOT / "workers" / "liveportrait_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def response_payload():
    return {"images": {key: f"data:image/png;base64,{key}" for key in worker.FRAME_KEYS}}


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return response_payload()


class LivePortraitWorkerTest(unittest.TestCase):
    def setUp(self):
        with worker.cache_lock:
            worker.cache.clear()
            worker.inflight.clear()

    def test_validates_profiles(self):
        with self.assertRaises(worker.HTTPException) as error:
            worker.run_upstream(worker.ExpressionPackRequest(image_url="https://cdn.test/a.webp", shape_profile="missing"))
        self.assertEqual(error.exception.status_code, 400)

    def test_reuses_pack_without_second_upstream_request(self):
        request = worker.ExpressionPackRequest(image_url="https://cdn.test/a.webp")
        with mock.patch.object(worker.session, "post", return_value=Response()) as post:
            first = worker.expression_pack(request)
            second = worker.expression_pack(request)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(set(second["images"]), set(worker.FRAME_KEYS))

    def test_rotating_signed_url_reuses_cached_pack(self):
        first_request = worker.ExpressionPackRequest(
            image_url="https://cdn.test/a.webp?X-Amz-Signature=old&Expires=1"
        )
        second_request = worker.ExpressionPackRequest(
            image_url="https://cdn.test/a.webp?X-Amz-Signature=new&Expires=2"
        )
        with mock.patch.object(worker.session, "post", return_value=Response()) as post:
            first = worker.expression_pack(first_request)
            second = worker.expression_pack(second_request)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
