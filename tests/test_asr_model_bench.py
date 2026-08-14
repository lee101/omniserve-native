import json
import urllib.request
from pathlib import Path
from unittest import mock

from tools.asr_model_bench import openai_backend


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"text": "measured transcript"}).encode()


def test_openai_backend_sends_model_audio_and_auth(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-audio")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TRANSCRIBE_URL", "https://example.test/v1/audio/transcriptions")

    with mock.patch.object(urllib.request, "urlopen", return_value=Response()) as opened:
        text = openai_backend("gpt-transcribe")(Path(clip))

    assert text == "measured transcript"
    request = opened.call_args.args[0]
    assert request.full_url == "https://example.test/v1/audio/transcriptions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert b"gpt-transcribe" in request.data
    assert b"RIFF-audio" in request.data
