#!/usr/bin/env python3
"""Rewrite Z-Image LoRAs into the diffusion_model.* lora_A/lora_B layout sd.cpp applies fully.

Handles kohya lora_unet_* (fused qkv/out or diffusers to_q/to_k/to_v/to_out_0, FFN, adaLN),
PEFT (base_model.model.* and bare layers.*, *.default.weight) and ai-toolkit/diffusers files.
Rejects non-Z-Image adapters (SDXL text-encoder, Flux/Klein double_blocks). Writes a sibling
<stem>.sdcpp.safetensors; never touches the source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DIM = 3840
FFN = 10240
ADALN_IN = 256
BLOCKS = {"layers": 30, "noise_refiner": 2, "context_refiner": 2}
SUBMODULES = {
    "attention.qkv": (3 * DIM, DIM),
    "attention.out": (DIM, DIM),
    "attention.to_q": (DIM, DIM),
    "attention.to_k": (DIM, DIM),
    "attention.to_v": (DIM, DIM),
    "attention.to_out.0": (DIM, DIM),
    "feed_forward.w1": (FFN, DIM),
    "feed_forward.w2": (DIM, FFN),
    "feed_forward.w3": (FFN, DIM),
    "adaLN_modulation.0": (4 * DIM, ADALN_IN),
}
NO_ADALN = {"context_refiner"}


def module_shapes() -> dict[str, tuple[int, int]]:
    shapes = {}
    for block, count in BLOCKS.items():
        for i in range(count):
            for sub, shape in SUBMODULES.items():
                if sub.startswith("adaLN") and block in NO_ADALN:
                    continue
                shapes[f"{block}.{i}.{sub}"] = shape
    return shapes


MODULES = module_shapes()
KOHYA = {name.replace(".", "_"): name for name in MODULES}
SUFFIXES = (
    (".lora_A.default.weight", "lora_A"),
    (".lora_B.default.weight", "lora_B"),
    (".lora_A.weight", "lora_A"),
    (".lora_B.weight", "lora_B"),
    (".lora_down.weight", "lora_A"),
    (".lora_up.weight", "lora_B"),
    (".alpha", "alpha"),
)
FOREIGN = re.compile(r"(^|[._])(lora_te\d?_|te\d?\.|text_model|double_blocks|single_blocks|single_transformer_blocks|transformer_blocks|input_blocks|down_blocks)")


class NotZImage(ValueError):
    pass


def split_key(key: str) -> tuple[str, str]:
    for suffix, kind in SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], kind
    raise ValueError(f"unsupported tensor suffix: {key}")


def module_name(stem: str) -> str:
    if FOREIGN.search(stem):
        raise NotZImage(f"not a Z-Image DiT adapter: {stem}")
    if stem.startswith("lora_unet_"):
        name = KOHYA.get(stem[len("lora_unet_"):])
        if name is None:
            raise ValueError(f"unknown kohya module: {stem}")
        return name
    for prefix in ("base_model.model.", "diffusion_model.", "transformer.", "model.diffusion_model."):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    if stem not in MODULES:
        raise ValueError(f"unknown Z-Image module: {stem}")
    return stem


def convert(src: Path, alpha: float | None = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    groups: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(src), "pt") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            stem, kind = split_key(key)
            name = module_name(stem)
            slot = groups.setdefault(name, {})
            if kind in slot:
                raise ValueError(f"duplicate {kind} for {name}")
            slot[kind] = handle.get_tensor(key)
    out: dict[str, torch.Tensor] = {}
    for name, slot in sorted(groups.items()):
        if "lora_A" not in slot or "lora_B" not in slot:
            raise ValueError(f"incomplete LoRA pair for {name}: {sorted(slot)}")
        down, up = slot["lora_A"], slot["lora_B"]
        rank = down.shape[0]
        if down.dim() != 2 or up.dim() != 2 or up.shape[1] != rank:
            raise ValueError(f"bad LoRA shapes for {name}: {tuple(down.shape)} {tuple(up.shape)}")
        if (up.shape[0], down.shape[1]) != MODULES[name]:
            raise ValueError(f"{name}: delta {(up.shape[0], down.shape[1])} != weight {MODULES[name]}")
        base = f"diffusion_model.{name}"
        out[f"{base}.lora_A.weight"] = down.contiguous()
        out[f"{base}.lora_B.weight"] = up.contiguous()
        value = slot.get("alpha")
        if value is not None:
            out[f"{base}.alpha"] = value.reshape(()).to(torch.float32)
        elif alpha is not None:
            out[f"{base}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    metadata = {k: v for k, v in metadata.items() if len(v) < 4096}
    metadata["sdcpp_converted_from"] = src.name
    return out, metadata


def target_path(src: Path) -> Path:
    return src.with_name(src.name[: -len(".safetensors")] + ".sdcpp.safetensors")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--alpha", type=float, help="alpha for files without per-module alpha (PEFT lora_alpha)")
    ap.add_argument("--peft-config", type=Path, help="adapter_config.json supplying lora_alpha")
    ap.add_argument("--out", type=Path, help="output path (single input only)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    alpha = args.alpha
    if args.peft_config:
        cfg = json.loads(args.peft_config.read_text())
        if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            raise SystemExit("rslora/dora/rank_pattern/alpha_pattern unsupported")
        alpha = float(cfg["lora_alpha"])
    if args.out and len(args.files) != 1:
        raise SystemExit("--out needs exactly one input")
    status = 0
    for src in args.files:
        dst = args.out or target_path(src)
        if dst.resolve() == src.resolve():
            raise SystemExit(f"refusing to overwrite source {src}")
        try:
            tensors, metadata = convert(src, alpha)
        except NotZImage as exc:
            print(json.dumps({"src": str(src), "status": "not_zimage", "error": str(exc)}))
            status = 2
            continue
        if dst.exists() and not args.force:
            print(json.dumps({"src": str(src), "status": "exists", "dst": str(dst)}))
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        save_file(tensors, str(tmp), metadata)
        tmp.replace(dst)
        print(json.dumps({"src": str(src), "status": "converted", "dst": str(dst), "tensors": len(tensors)}))
    return status


if __name__ == "__main__":
    sys.exit(main())
