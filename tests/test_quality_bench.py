import json
import urllib.request
from unittest import mock

from tools.quality_bench import Bench


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }).encode()


def test_force_local_marks_chat_request_only():
    bench = Bench("http://127.0.0.1:8791", None, 1, force_local=True)
    with mock.patch.object(urllib.request, "urlopen", return_value=Response()) as opened:
        status, text, _, _ = bench.chat("hello")

    assert status == 200
    assert text == "ok"
    request = opened.call_args.args[0]
    assert request.get_header("X-omniserve-internal") == "local"


def test_default_chat_does_not_force_local():
    bench = Bench("http://127.0.0.1:8791", None, 1)
    with mock.patch.object(urllib.request, "urlopen", return_value=Response()) as opened:
        bench.chat("hello")

    request = opened.call_args.args[0]
    assert request.get_header("X-omniserve-internal") is None
