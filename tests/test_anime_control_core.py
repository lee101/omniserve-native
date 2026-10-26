import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def core():
    return importlib.import_module("workloads.anime_control_core")


@pytest.fixture(scope="module")
def regional():
    return importlib.import_module("workloads.anime_scene_control")


def test_split_lora_accepts_down_up_alpha_and_scales_by_alpha_over_rank(core):
    state = {
        "diffusion_model.blocks.2.self_attn.q_proj.lora_down.weight": torch.randn(4, 8),
        "diffusion_model.blocks.2.self_attn.q_proj.lora_up.weight": torch.randn(8, 4),
        "diffusion_model.blocks.2.self_attn.q_proj.alpha": torch.tensor(2.0),
        "diffusion_model.blocks.2.mlp.layer1.lora_down.weight": torch.randn(4, 8),
        "diffusion_model.blocks.2.mlp.layer1.lora_up.weight": torch.randn(16, 4),
        "diffusion_model.blocks.2.mlp.layer1.alpha": torch.tensor(4.0),
        "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 8),
        "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight": torch.randn(8, 4),
        "diffusion_model.control_embedder.proj.weight": torch.zeros(1),
    }
    blocks, adapter, extras = core.split_lora(state)
    assert blocks["transformer_blocks.2.attn1.to_q.weight"]["scale"] == 0.5
    assert blocks["transformer_blocks.2.ff.net.0.proj.weight"]["scale"] == 1.0
    assert adapter["blocks.0.self_attn.q_proj.weight"]["scale"] == 1.0
    assert set(extras) == {"control_embedder.proj.weight"}


def test_merge_applies_scale_and_strength(core):
    holder = torch.nn.Module()
    holder.layer = torch.nn.Linear(8, 8, bias=False)
    before = holder.layer.weight.detach().clone()
    a, b = torch.randn(2, 8), torch.randn(8, 2)
    core.merge_lora(holder, {"layer.weight": {"A": a, "B": b, "scale": 0.5}}, strength=2.0)
    assert torch.allclose(holder.layer.weight, before + b @ a, atol=1e-5)


def test_common_values_defaults_and_bounds(core):
    values = core.common_values({"prompt": " hi ", "seed": 5})
    assert values["prompt"] == "hi" and values["steps"] == 24 and values["width"] == values["height"] == 1024
    assert values["cutoff"] == core.CFG_CUTOFF and values["seed"] == 5
    for bad in ({"prompt": ""}, {"prompt": "p", "width": 2048}, {"prompt": "p", "num_inference_steps": 4},
                {"prompt": "p", "guidance_scale": 12}, {"prompt": "p", "cfg_cutoff": 0.1},
                {"prompt": "p", "width": 1024, "height": 1280}):
        with pytest.raises(ValueError):
            core.common_values(bad)


def test_encode_outputs_returns_image_and_side_artifacts(core):
    from PIL import Image

    result = core.encode_outputs(Image.new("RGB", (8, 8)), {"mask": np.zeros((4, 4, 3), np.uint8)}, 7, "png", "x")
    assert [item["filename"] for item in result["outputs"]] == ["x-7.png", "x-7-mask.png"]
    with pytest.raises(ValueError, match="output_format"):
        core.encode_outputs(Image.new("RGB", (8, 8)), {}, 7, "gif", "x")


def test_region_mask_renders_rectangles_on_black(regional):
    mask = regional.region_mask(
        [{"x": 0, "y": 0, "w": 0.5, "h": 1, "color": "#ff0000"}, {"x": 0.5, "y": 0, "w": 0.5, "h": 0.5, "color": "#0000ff"}], 64, 32
    )
    assert mask.shape == (32, 64, 3)
    assert tuple(mask[10, 5]) == (255, 0, 0) and tuple(mask[5, 40]) == (0, 0, 255) and tuple(mask[25, 40]) == (0, 0, 0)
    for bad in ([], "x", [{"x": 2, "y": 0, "w": 1, "h": 1}], [{"x": 0, "y": 0, "w": 1}],
                [{"x": 0, "y": 0, "w": 1, "h": 1, "color": "red"}]):
        with pytest.raises(ValueError):
            regional.region_mask(bad, 64, 32)


def test_split_weights_routes_conditioning_and_layers(regional):
    tensors = {
        "lllite_conditioning1.conv1.weight": 1,
        "lllite_dit_blocks_3_self_attn_k_proj.down.weight": 2,
        "lllite_dit_blocks_3_self_attn_k_proj.depth_embed": 3,
    }
    conditioning, layers = regional.split_weights(tensors)
    assert conditioning == {"conv1.weight": 1}
    assert layers == {(3, "to_k"): {"down.weight": 2, "depth_embed": 3}}
    with pytest.raises(ValueError, match="unexpected"):
        regional.split_weights({"lllite_dit_blocks_3_cross_attn_q_proj.down.weight": 1})


def test_conditioning_net_emits_one_embedding_per_token(regional):
    net = regional.conditioning_module().eval()
    out = net(torch.randn(1, 3, 64, 96))
    assert out.shape == (1, (64 // 16) * (96 // 16), 64)


def test_layer_is_identity_at_zero_strength_and_residual_otherwise(regional):
    class State:
        cond_emb = torch.randn(1, 5, 64)
        strength = torch.zeros(())

    w = {
        "down.weight": torch.randn(64, 32), "down.bias": torch.randn(64),
        "mid.weight": torch.randn(64, 128), "mid.bias": torch.randn(64),
        "cond_to_film.weight": torch.randn(128, 64), "cond_to_film.bias": torch.randn(128),
        "up.weight": torch.randn(32, 64), "up.bias": torch.randn(32), "depth_embed": torch.randn(64),
    }
    layer = regional.Layer(w, State)
    x = torch.randn(2, 5, 32)
    assert torch.equal(layer(None, (x,))[0], x)
    State.strength = torch.ones(())
    assert not torch.allclose(layer(None, (x,))[0], x)


def test_regional_handler_rejects_unknown_inputs(regional):
    with pytest.raises(ValueError, match="unknown scene inputs"):
        regional.handler({"input": {"prompt": "p", "extra": 1}}, pipe=object())


@pytest.fixture(scope="module")
def depth():
    return importlib.import_module("workloads.anime_depth_control")


@pytest.fixture(scope="module")
def sketch():
    return importlib.import_module("workloads.anime_sketch_control")


def test_depth_key_conversion_maps_control_blocks(depth):
    tensors = {
        "control_embedder.proj.1.weight": 0,
        "control_blocks.0.before_proj.weight": 1,
        "control_blocks.0.after_proj.bias": 2,
        "control_blocks.0.block.self_attn.q_proj.weight": 3,
        "control_blocks.0.block.adaln_modulation_mlp.2.weight": 4,
        "control_blocks.1.block.mlp.layer1.weight": 5,
        "control_blocks.1.block.cross_attn.q_norm.weight": 6,
    }
    embedder, blocks = depth.convert_keys(tensors)
    assert embedder == {"weight": 0}
    assert blocks[0] == {
        "before_proj.weight": 1, "after_proj.bias": 2,
        "attn1.to_q.weight": 3, "norm3.linear_2.weight": 4,
    }
    assert blocks[1] == {"ff.net.0.proj.weight": 5, "attn2.norm_q.weight": 6}
    for bad in ({"mystery": 1}, {"control_blocks.0.block.bogus.weight": 1}):
        with pytest.raises(ValueError):
            depth.convert_keys(bad)
    with pytest.raises(ValueError, match="contiguous"):
        depth.convert_keys({"control_embedder.proj.1.weight": 0, "control_blocks.2.after_proj.bias": 1})


def test_depth_and_sketch_handlers_reject_unknown_inputs(depth, sketch):
    with pytest.raises(ValueError, match="unknown depth inputs"):
        depth.handler({"input": {"prompt": "p", "extra": 1}}, pipe=object())
    with pytest.raises(ValueError, match="unknown sketch inputs"):
        sketch.handler({"input": {"prompt": "p", "lora_strength": 1}}, pipe=object())


def test_canny_map_is_white_on_black_at_output_size(sketch):
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, 32:] = 255
    edges = sketch.canny_map(image, 128, 96)
    assert edges.shape == (96, 128, 3)
    assert edges.max() == 255 and edges[:, :40].max() == 0
    assert set(np.unique(edges)) <= {0, 255}


@pytest.mark.parametrize("batch", [1, 2])
def test_hoisted_timesteps_match_per_step_expand_cast(batch):
    steps = 24
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32) ** 1.7
    hoisted = sigmas[:steps].to(torch.bfloat16)[:, None].expand(steps, batch).contiguous()
    for index in range(steps):
        legacy = sigmas[index].expand(batch).to(torch.bfloat16)
        assert torch.equal(hoisted[index], legacy)
        assert hoisted[index].is_contiguous() and hoisted[index].shape == (batch,)
