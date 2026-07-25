#!/usr/bin/env python3
"""Cache-key and local-storage behaviour for worker outputs.

No network and no GPU: this covers the part that decides whether a request is
answered from the cache or costs a model run.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

import object_store  # noqa: E402


PARAMS = {"format": "webp", "threshold": 0.0, "decontaminate": True, "quality": 85}


class CacheKeyTest(unittest.TestCase):
    def test_same_input_same_key(self):
        first = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        second = object_store.cache_key("https://cdn.example.com/a/chair.jpg", dict(PARAMS))
        self.assertEqual(first, second)

    def test_key_layout_is_content_addressed(self):
        key = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        prefix, shard, name = key.split("/")
        self.assertEqual(prefix, "cutouts")
        self.assertEqual(len(shard), 2)
        self.assertTrue(name.endswith(".webp"))
        self.assertTrue(name.startswith(shard))

    def test_params_that_change_pixels_change_the_key(self):
        base = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        for field, value in (("threshold", 0.5), ("decontaminate", False), ("quality", 92), ("format", "png")):
            changed = dict(PARAMS)
            changed[field] = value
            self.assertNotEqual(base, object_store.cache_key("https://cdn.example.com/a/chair.jpg", changed),
                                f"{field} must affect the cache key")

    def test_signed_url_noise_is_ignored(self):
        """The same asset behind a rotating signature is one cache entry."""
        plain = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        signed = object_store.cache_key(
            "https://cdn.example.com/a/chair.jpg?X-Amz-Signature=abc123&Expires=1699999999", PARAMS)
        self.assertEqual(plain, signed)

    def test_meaningful_query_is_kept(self):
        plain = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        resized = object_store.cache_key("https://cdn.example.com/a/chair.jpg?w=512", PARAMS)
        self.assertNotEqual(plain, resized)

    def test_params_that_merely_start_like_a_signature_are_kept(self):
        """"sepia"/"svg"/"session" must not be mistaken for Azure SAS fields."""
        plain = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        for query in ("sepia=1", "svg=1", "session=abc", "search=red", "spread=2"):
            self.assertNotEqual(
                plain,
                object_store.cache_key(f"https://cdn.example.com/a/chair.jpg?{query}", PARAMS),
                f"{query} changes the fetched bytes and must change the key",
            )

    def test_azure_and_google_signing_params_are_dropped(self):
        plain = object_store.cache_key("https://cdn.example.com/a/chair.jpg", PARAMS)
        for query in ("se=2026-01-01&sp=r&sv=2021&sig=xyz", "X-Goog-Signature=abc&X-Goog-Expires=900"):
            self.assertEqual(
                plain,
                object_store.cache_key(f"https://cdn.example.com/a/chair.jpg?{query}", PARAMS),
                f"{query} is signing noise",
            )

    def test_query_order_does_not_matter(self):
        one = object_store.cache_key("https://x.test/i.jpg?a=1&b=2", PARAMS)
        two = object_store.cache_key("https://x.test/i.jpg?b=2&a=1", PARAMS)
        self.assertEqual(one, two)

    def test_different_images_do_not_collide(self):
        keys = {
            object_store.cache_key(f"https://cdn.example.com/{name}.jpg", PARAMS)
            for name in ("a", "b", "c", "d")
        }
        self.assertEqual(len(keys), 4)


class LocalStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._saved_dir = object_store.CACHE_DIR
        object_store.CACHE_DIR = self.tmp
        # Force the local backend regardless of the machine's R2 environment.
        self._saved_bucket = object_store.BUCKET
        object_store.BUCKET = ""

    def tearDown(self):
        object_store.CACHE_DIR = self._saved_dir
        object_store.BUCKET = self._saved_bucket
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        key = object_store.cache_key("https://x.test/i.jpg", PARAMS)
        self.assertFalse(object_store.exists(key))
        self.assertIsNone(object_store.get(key))

        self.assertIsNone(object_store.put(key, b"webp-bytes"), "local backend has no public URL")
        self.assertTrue(object_store.exists(key))
        self.assertEqual(object_store.get(key), b"webp-bytes")

    def test_describe_reports_local_backend(self):
        described = object_store.describe()
        self.assertEqual(described["backend"], "local")
        self.assertEqual(described["bucket"], str(self.tmp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
