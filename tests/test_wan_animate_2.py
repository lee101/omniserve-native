import importlib

import pytest


def test_inputs_support_runpod_and_nested_envelopes():
    worker = importlib.import_module("workloads.wan_animate_2")
    assert worker._inputs({"input": {"workload": "wan-animate-2"}})["workload"] == "wan-animate-2"
    assert worker._inputs({"input": {"input": {"prompt": "p"}}})["prompt"] == "p"


def test_dimensions_are_bounded_and_wan_aligned():
    worker = importlib.import_module("workloads.wan_animate_2")
    assert worker._dimensions({}) == (640, 800)
    assert worker._dimensions({"width": 1280, "height": 720}) == (1280, 720)
    with pytest.raises(ValueError, match="divisible by 16"):
        worker._dimensions({"width": 641, "height": 800})
    with pytest.raises(ValueError, match="921600"):
        worker._dimensions({"width": 1280, "height": 1280})


def test_distilled_profile_rejects_cfg(monkeypatch):
    worker = importlib.import_module("workloads.wan_animate_2")
    monkeypatch.setattr(worker, "_public_url", lambda value, field: str(value))
    with pytest.raises(ValueError, match="guidance_scale"):
        worker.handler({"input": {
            "character_image_url": "https://cdn.example/character.png",
            "driving_video_url": "https://cdn.example/dance.mp4",
            "prompt": "A silver robot in a white studio",
            "guidance_scale": 2,
            "_omniserve_profile": "small",
            "output_upload_url": "https://upload.example/result",
            "output_public_url": "https://cdn.example/result.mp4",
        }})
