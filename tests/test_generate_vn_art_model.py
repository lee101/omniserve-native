#!/usr/bin/env python3
"""The VN art tool must ask for its model on the route that reads one.

The gateway only consults the body's `model` on /v1/images/generations (and
/v1/images/edits, /v1/images/img2img). /v1/images/backgrounds proxies straight
to the art lane, so a model name sent there is dropped on the floor. A silent
drop is the failure worth defending against: the art gets made by the wrong
model and the manifest still claims the right one.
"""

import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import generate_vn_art

PIXEL = ("data:image/png;base64,"
         "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class ModelRoutingTest(unittest.TestCase):
    def setUp(self):
        self.backgrounds = []
        self.foregrounds = []
        self.routes = []

        def post_image(url, payload, _timeout=360):
            self.routes.append(url)
            self.backgrounds.append(payload)
            return Image.new("RGB", (8, 8))

        def request_json(_url, _method="GET", payload=None, timeout=360):
            if payload is None:
                return {"status": "done", "data_url": PIXEL}
            self.foregrounds.append(payload)
            return {"job_id": "job-1", "poll_after_ms": 1}

        self._saved = (generate_vn_art.post_image, generate_vn_art.request_json)
        generate_vn_art.post_image = post_image
        generate_vn_art.request_json = request_json

    def tearDown(self):
        generate_vn_art.post_image, generate_vn_art.request_json = self._saved

    def test_named_model_goes_to_the_route_that_reads_it(self):
        generate_vn_art.render_background("http://g", "a kitchen", 1, 8, 8, "qwen-image-2.1")

        self.assertEqual(self.routes, ["http://g/v1/images/generations"])
        self.assertEqual(self.backgrounds[0]["model"], "qwen-image-2.1")

    def test_no_model_keeps_the_backgrounds_route(self):
        generate_vn_art.render_background("http://g", "a kitchen", 1, 8, 8)

        self.assertEqual(self.routes, ["http://g/v1/images/backgrounds"])
        self.assertNotIn("model", self.backgrounds[0])

    def test_model_rides_alongside_the_existing_knobs(self):
        generate_vn_art.render_background("http://g", "a sauna", 7, 8, 8, "qwen-image-2.1")

        payload = self.backgrounds[0]
        self.assertEqual(payload["prompt"], "a sauna")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["width"], 8)
        self.assertEqual(payload["height"], 8)
        self.assertEqual(payload["num_inference_steps"], 9)
        self.assertTrue(payload["teleport"])

    def test_sprite_cutouts_carry_no_model(self):
        generate_vn_art.render_foreground("http://g", "a porter", 1, 8, 8, 1)

        self.assertNotIn("model", self.foregrounds[0])

    def test_cutout_job_is_queued_and_polled(self):
        image = generate_vn_art.render_foreground("http://g", "a porter", 3, 8, 8, 1)

        self.assertEqual(self.foregrounds[0]["seed"], 3)
        self.assertEqual(self.foregrounds[0]["output_format"], "webp")
        self.assertEqual(image.size, (1, 1))


if __name__ == "__main__":
    unittest.main()
