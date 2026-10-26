import base64
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def pose():
    return importlib.import_module("workloads.anime_pose_control")


def _adapter_state(rank=2):
    state = {}
    for module, (out_f, in_f) in {
        "blocks.3.self_attn.q_proj": (8, 8),
        "blocks.3.self_attn.output_proj": (8, 8),
        "blocks.3.mlp.layer1": (16, 8),
        "blocks.3.adaln_modulation_mlp.2": (24, 4),
    }.items():
        state[f"diffusion_model.{module}.lora_A.weight"] = torch.randn(rank, in_f)
        state[f"diffusion_model.{module}.lora_B.weight"] = torch.randn(out_f, rank)
    state["diffusion_model.llm_adapter.blocks.1.self_attn.o_proj.lora_A.weight"] = torch.randn(rank, 8)
    state["diffusion_model.llm_adapter.blocks.1.self_attn.o_proj.lora_B.weight"] = torch.randn(8, rank)
    state["diffusion_model.control_embedder.proj.weight"] = torch.randn(8, 64)
    state["diffusion_model.control_embedder.proj.bias"] = torch.randn(8)
    return state


def test_manifest_registers_the_workload():
    manifest = json.loads(Path("workloads/workloads.json").read_text(encoding="utf-8"))
    entry = manifest["anime-pose-control"]
    assert entry["module"] == "workloads.anime_pose_control"
    assert "anime-pose-control" in entry["aliases"]


def test_split_adapter_maps_cosmos_names_to_diffusers(pose):
    blocks, adapter, control = pose.split_adapter(_adapter_state())
    assert set(blocks) == {
        "transformer_blocks.3.attn1.to_q.weight",
        "transformer_blocks.3.attn1.to_out.0.weight",
        "transformer_blocks.3.ff.net.0.proj.weight",
        "transformer_blocks.3.norm3.linear_2.weight",
    }
    assert set(adapter) == {"blocks.1.self_attn.o_proj.weight"}
    assert set(control) == {"weight", "bias"}


def test_split_adapter_rejects_unknown_and_incomplete_keys(pose):
    state = _adapter_state()
    state["diffusion_model.blocks.0.bogus.lora_A.weight"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="unmapped"):
        pose.split_adapter(state)
    state = _adapter_state()
    del state["diffusion_model.blocks.3.mlp.layer1.lora_B.weight"]
    with pytest.raises(ValueError, match="incomplete"):
        pose.split_adapter(state)
    state = _adapter_state()
    del state["diffusion_model.control_embedder.proj.bias"]
    with pytest.raises(ValueError, match="control_embedder"):
        pose.split_adapter(state)


def test_merge_matches_unmerged_lora_forward(pose):
    layer = torch.nn.Linear(8, 8, bias=False)
    holder = torch.nn.Module()
    holder.layer = layer
    a, b = torch.randn(2, 8), torch.randn(8, 2)
    x = torch.randn(3, 8)
    expected = layer(x) + x @ a.T @ b.T
    assert pose.merge_lora(holder, {"layer.weight": {"A": a, "B": b}}) == 1
    assert torch.allclose(layer(x), expected, atol=1e-5)
    with pytest.raises(KeyError):
        pose.merge_lora(holder, {"missing.weight": {"A": a, "B": b}})


def test_control_tokens_match_patch_embed_layout(pose):
    embedder = torch.nn.Linear(64, 8)
    latent = torch.randn(1, 16, 1, 4, 6)
    tokens = pose.control_tokens(embedder, latent)
    assert tokens.shape == (1, 1, 2, 3, 8)
    patch = latent[0, :, 0, 2:4, 4:6].reshape(16, 2, 2).reshape(-1)
    assert torch.allclose(tokens[0, 0, 1, 2], embedder(patch), atol=1e-5)


def test_zero_control_is_a_no_op(pose):
    inner = torch.nn.Linear(4, 4)
    slot = pose.controlled_patch_embed(inner)
    x = torch.randn(1, 1, 2, 3, 4)
    assert torch.equal(slot(x), inner(x))
    slot.tokens = torch.zeros(1, 1, 2, 3, 4)
    assert torch.equal(slot(x), inner(x))
    slot.tokens = torch.ones(1, 1, 2, 3, 4)
    assert torch.allclose(slot(x), inner(x) + 1)


def _skeleton_points():
    keypoints = np.zeros((133, 2))
    scores = np.zeros(133)
    for index, (x, y) in enumerate(
        [(50, 10), (45, 8), (55, 8), (40, 10), (60, 10), (35, 30), (65, 30), (30, 50), (70, 50),
         (28, 70), (72, 70), (42, 60), (58, 60), (40, 80), (60, 80), (40, 95), (60, 95)]
    ):
        keypoints[index] = (x, y)
        scores[index] = 0.9
    return keypoints, scores


def test_render_scales_keypoints_to_canvas(pose):
    keypoints, scores = _skeleton_points()
    canvas = pose.render_skeleton(keypoints, scores, 256, 128, 100)
    assert canvas.shape == (128, 256, 3)
    assert canvas[10 * 128 // 100, 50 * 256 // 100].sum() > 0
    assert canvas[:, :20].sum() == 0
    for style in pose.STYLES:
        assert pose.render_skeleton(keypoints, scores, 128, 128, 100, style).any()
    with pytest.raises(ValueError, match="skeleton_style"):
        pose.render_skeleton(keypoints, scores, 128, 128, 100, "nope")


def test_render_toggles_hands_and_face(pose):
    keypoints, scores = _skeleton_points()
    keypoints[91:133] = (20, 20)
    scores[91:133] = 0.9
    with_hands = pose.render_skeleton(keypoints, scores, 100, 100, 100)
    without = pose.render_skeleton(keypoints, scores, 100, 100, 100, hands=False)
    assert with_hands[20, 20].any() and not without[20, 20].any()


def test_detect_picks_largest_person_and_rejects_empty(pose):
    big, big_scores = _skeleton_points()
    small, small_scores = big * 0.3, big_scores.copy()
    image = np.zeros((64, 64, 3), np.uint8)
    chosen, _ = pose.detect_pose(image, 100, estimator=lambda _: (np.stack([small, big]), np.stack([small_scores, big_scores])))
    assert np.allclose(chosen, big)
    with pytest.raises(pose.NoPersonError):
        pose.detect_pose(image, 100, estimator=lambda _: (np.zeros((0, 133, 2)), np.zeros((0, 133))))
    weak = np.zeros(133)
    with pytest.raises(pose.NoPersonError):
        pose.detect_pose(image, 100, estimator=lambda _: (big[None], weak[None]))


def test_reference_decoding(pose):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (1, 2, 3)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    assert pose.decode_reference("data:image/png;base64," + encoded).shape == (6, 8, 3)
    for bad in ("", "not base64!!", base64.b64encode(b"plain text").decode()):
        with pytest.raises(ValueError):
            pose.decode_reference(bad)


def test_handler_rejects_unknown_inputs_and_formats(pose):
    with pytest.raises(ValueError, match="unknown pose inputs"):
        pose.handler({"input": {"prompt": "p", "extra": 1}}, pipe=object())
    with pytest.raises(ValueError, match="output_format"):
        pose.handler({"input": {"prompt": "p", "output_format": "gif"}}, pipe=object())


def test_cfg_cutoff_is_validated(pose):
    with pytest.raises(ValueError, match="cfg_cutoff"):
        pose.generate(object(), {"prompt": "p", "cfg_cutoff": 0.05})


def test_warmup_sizes_parse(pose):
    assert pose._sizes("1024x1024, 768x512,") == [(1024, 1024), (768, 512)]


def test_skeleton_image_is_fitted_to_the_output_size(pose):
    skeleton = np.zeros((40, 30, 3), np.uint8)
    skeleton[10:20, 5:10] = (255, 0, 0)
    fitted = pose.fit_skeleton(skeleton, 64, 96)
    assert fitted.shape == (96, 64, 3) and fitted.max() == 255 and fitted[0, 0].sum() == 0
    assert "skeleton_image" in pose.ALLOWED_INPUTS
