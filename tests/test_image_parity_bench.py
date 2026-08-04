import base64
import io
import json

import numpy as np
from PIL import Image

from tools.image_parity_bench import (
    decode_image_response,
    entropy,
    pixel_metrics,
)


def image_bytes(color=(20, 80, 160)):
    image = Image.new("RGB", (64, 64), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_openai_base64_envelope():
    encoded = base64.b64encode(image_bytes()).decode()
    body = json.dumps({"data": [{"b64_json": encoded, "seed": 42}]}).encode()
    image, meta = decode_image_response(body, "application/json", 1)
    assert image.size == (64, 64)
    assert meta["transport"] == "b64_json"


def test_decode_raw_image():
    image, meta = decode_image_response(image_bytes(), "image/png", 1)
    assert image.size == (64, 64)
    assert meta["transport"] == "raw"


def test_pixel_metrics_identical_and_changed():
    reference = Image.new("RGB", (64, 64), (10, 20, 30))
    identical = pixel_metrics(reference, reference.copy())
    assert identical["identical"] is True
    assert identical["psnr_db"] == float("inf")
    changed = pixel_metrics(reference, Image.new("RGB", (64, 64), (20, 30, 40)))
    assert changed["identical"] is False
    assert changed["mse"] == 100.0
    assert changed["histogram_intersection"] == 0.0


def test_entropy_distinguishes_flat_from_noise():
    flat = Image.new("L", (64, 64), 10).convert("RGB")
    rng = np.random.default_rng(42)
    noise = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
    assert entropy(flat) == 0.0
    assert entropy(noise) > 7.0
