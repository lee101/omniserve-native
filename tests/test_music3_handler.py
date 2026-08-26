import importlib.util
import io
from pathlib import Path
import wave

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "music3" / "handler.py"
SPEC = importlib.util.spec_from_file_location("music3_handler", MODULE_PATH)
music3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(music3)


def test_normalize_prompt_defaults_to_instrumental():
    request = music3.normalize_request({"input": {"prompt": "Warm synthwave at 105 BPM", "duration": 10, "seed": 7}})
    assert request["max_new_tokens"] == 250
    assert request["seed"] == 7
    assert request["lyrics"].startswith("[Intro]\n")
    assert "instrumental" in request["instructions"].lower()


def test_normalize_preserves_lyrics_and_caps_duration():
    request = music3.normalize_request({"input": {"lyrics": "[Verse]\nHello", "instructions": "Acoustic folk", "duration": 30}})
    assert request["lyrics"] == "[Verse]\nHello"
    assert request["max_new_tokens"] == 750
    with pytest.raises(ValueError, match="between 1"):
        music3.normalize_request({"input": {"prompt": "x", "duration": 361}})


def test_wav_statistics_reports_signal_health():
    sample_rate = 32000
    timeline = np.arange(sample_rate, dtype=np.float64) / sample_rate
    mono = (0.25 * np.sin(2 * np.pi * 440 * timeline) * 32767).astype("<i2")
    stereo = np.column_stack([mono, mono]).reshape(-1)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(stereo.tobytes())
    stats = music3.wav_statistics(output.getvalue())
    assert stats["sample_rate_hz"] == 32000
    assert stats["channels"] == 2
    assert stats["duration_seconds"] == 1.0
    assert stats["clipped_samples"] == 0
    assert stats["stereo_correlation"] == 1.0
