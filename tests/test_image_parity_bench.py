import base64
import io
import json

import numpy as np
from PIL import Image

from tools.image_parity_bench import (
    decode_image_response,
    entropy,
    pixel_metrics,
    request_economics,
)


def image_bytes(color=(20, 80, 160)):
    image = Image.new("RGB", (64, 64), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_openai_base64_envelope():
    encoded = base64.b64encode(image_bytes()).decode()
    body = json.dumps({
        "data": [{
            "b64_json": encoded,
            "seed": 42,
            "inference_time_ms": 7841,
            "format": "png",
        }]
    }).encode()
    image, meta = decode_image_response(body, "application/json", 1)
    assert image.size == (64, 64)
    assert meta["transport"] == "b64_json"
    assert meta["inference_time_ms"] == 7841
    assert meta["format"] == "png"


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


def test_request_economics_prefers_server_inference_time():
    result = request_economics(
        {"inference_time_ms": 7841, "wall_ms": 22686},
        "1024x1024",
        1000,
    )
    assert result["timing_source"] == "inference_time_ms"
    assert result["billed_ms"] == 7841
    assert abs(result["cost_per_megapixel_usd"] - 0.002845419) < 1e-9


def test_request_economics_falls_back_to_wall_time():
    result = request_economics({"wall_ms": 1000}, "1000x1000", 2628)
    assert result["timing_source"] == "wall_ms"
    assert result["cost_per_megapixel_usd"] == 0.001
