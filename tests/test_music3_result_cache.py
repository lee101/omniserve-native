import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import wave


ROOT = Path(__file__).parents[1]


def _field(digest, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _cache_key(instructions: str, lyrics: str, frames: int, seed: int, scope: str) -> str:
    digest = hashlib.sha256()
    for value in (
        b"omniserve-music3-result-v1",
        scope.encode(),
        b"MiniMaxAI/MiniMax-Music3",
        instructions.encode(),
        lyrics.encode(),
        struct.pack(">Q", frames),
        struct.pack(">Q", seed),
    ):
        _field(digest, value)
    return digest.hexdigest()


def test_worker_returns_exact_cached_wav_without_starting_model(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    request = {
        "input": {
            "instructions": "Acoustic folk",
            "lyrics": "[Verse]\nHello",
            "duration": 10,
            "seed": 7,
        }
    }
    job = tmp_path / "job.json"
    job.write_text(json.dumps(request), encoding="utf-8")
    scope = "test-release|dtype=bfloat16|serve=|threshold=75|retries=1"
    key = _cache_key("Acoustic folk", "[Verse]\nHello", 250, 7, scope)
    wav = cache / f"{key}.wav"
    with wave.open(str(wav), "wb") as output:
        output.setparams((2, 2, 32000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0\0\0" * 32000)
    original = wav.read_bytes()
    (cache / f"{key}.meta").write_text("8 2\n", encoding="ascii")
    env = {
        **os.environ,
        "MUSIC3_RESULT_CACHE": "1",
        "MUSIC3_RESULT_CACHE_NAMESPACE": "test-release",
        "MUSIC3_RESULT_CACHE_DIR": str(cache),
        "MUSIC3_WARM_START": "0",
    }

    completed = subprocess.run(
        [str(ROOT / "build/music3c"), "--job-file", str(job)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result["seed"] == 8
    assert result["metrics"]["quality_original_seed"] == 7
    assert result["metrics"]["quality_attempts"] == 2
    assert result["metrics"]["exact_result_cache_hit"] is True
    assert result["metrics"]["generation_seconds"] == 0
    assert base64.b64decode(result["outputs"][0]["data"]) == original
