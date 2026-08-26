import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def worker():
    return importlib.import_module("workloads.anima_image")


class _StubImage:
    def save(self, buffer, format, **kwargs):  # noqa: A002 - Pillow's keyword name
        buffer.write(b"stub-" + format.encode("ascii"))


def test_manifest_registers_the_workload():
    manifest = json.loads(Path("workloads/workloads.json").read_text(encoding="utf-8"))
    entry = manifest["anima-2.9b"]
    assert entry["module"] == "workloads.anima_image"
    assert entry["required_mib"] >= 12000
    assert "anima" in entry["aliases"]


def test_inputs_support_runpod_and_nested_envelopes(worker):
    assert worker._inputs({"input": {"prompt": "p"}})["prompt"] == "p"
    assert worker._inputs({"input": {"input": {"prompt": "q"}}})["prompt"] == "q"


def test_dimensions_are_bounded_and_patch_aligned(worker):
    assert worker._dimension(None, 832) == 832
    assert worker._dimension(1024, 832) == 1024
    with pytest.raises(ValueError, match="divisible by 16"):
        worker._dimension(833, 832)
    with pytest.raises(ValueError, match="between 512 and 1536"):
        worker._dimension(2048, 832)


def test_seed_normalisation(worker):
    assert worker._seed(7) == 7
    assert 0 <= worker._seed(-1) <= worker.MAX_SEED
    assert worker._seed(worker.MAX_SEED + 2) == 1


def test_checkpoint_remap_covers_blocks_and_top_level(worker):
    remapped = worker.remap_checkpoint({
        "net.blocks.3.self_attn.q_proj.weight": 1,
        "net.blocks.3.mlp.layer1.weight": 2,
        "net.x_embedder.proj.1.weight": 3,
        "net.t_embedder.0.linear_1.weight": 4,
        "net.unrelated": 5,
    })
    assert remapped == {
        "transformer_blocks.3.attn1.to_q.weight": 1,
        "transformer_blocks.3.ff.net.0.proj.weight": 2,
        "patch_embed.proj.weight": 3,
        "time_embed.t_embedder.linear_1.weight": 4,
    }


def test_handler_requires_the_commercial_license(worker, monkeypatch):
    monkeypatch.delenv("APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED", raising=False)
    with pytest.raises(RuntimeError, match="APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED"):
        worker.handler({"input": {"prompt": "a cat"}})


def test_handler_rejects_unknown_inputs(worker, monkeypatch):
    monkeypatch.setenv("APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED", "1")
    with pytest.raises(ValueError, match="unknown Anima inputs: lora_url"):
        worker.handler({"input": {"prompt": "a cat", "lora_url": "https://example/lora"}})


def test_handler_returns_the_app_nz_envelope(worker, monkeypatch):
    monkeypatch.setenv("APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED", "1")
    monkeypatch.setattr(
        worker, "generate", lambda pipe, values: (_StubImage(), 99, {"accel": "dense"})
    )
    result = worker.handler({"input": {"prompt": "a cat", "output_format": "png"}}, pipe=object())
    assert result["seed"] == 99
    assert result["outputs"][0]["filename"] == "anima-99.png"
    assert result["outputs"][0]["content_type"] == "image/png"
    assert result["bytes"] == len(b"stub-PNG")
    assert result["accel"] == "dense"
    assert result["teleport"] is None


def test_teleport_fraction_is_validated(worker, monkeypatch):
    monkeypatch.setenv("APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED", "1")
    with pytest.raises(ValueError, match="teleport_fraction"):
        worker.generate(object(), {"prompt": "a cat", "teleport": True, "teleport_fraction": 5})


def test_new_accel_inputs_are_accepted(worker, monkeypatch):
    unknown = sorted({"teleport", "teleport_fraction", "accel", "fused"} - worker.ALLOWED_INPUTS)
    assert unknown == []


def test_handler_rejects_unsupported_formats(worker, monkeypatch):
    monkeypatch.setenv("APPNZ_ANIMA_COMMERCIAL_LICENSE_ACCEPTED", "1")
    with pytest.raises(ValueError, match="output_format"):
        worker.handler({"input": {"prompt": "a cat", "output_format": "gif"}}, pipe=object())


def test_prompt_embeds_are_cached(worker):
    calls = []

    class _StubPipe:
        def _encode_prompt(self, texts, device, dtype, tokens):
            calls.append(texts[0])
            return object()

    worker._embed_cache.clear()
    pipe = _StubPipe()
    assert worker._cached_embeds(pipe, "a cat", "cuda", "bf16") is worker._cached_embeds(
        pipe, "a cat", "cuda", "bf16"
    )
    assert calls == ["a cat"]
