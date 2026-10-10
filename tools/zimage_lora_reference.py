#!/usr/bin/env python3
"""Diffusers reference renders for Z-Image LoRA parity, bounded to a 12 GB CPU budget.

Applies each ORIGINAL LoRA file to the diffusers ZImageTransformer2DModel as unmerged
PEFT-style forward hooks with its native semantics (kohya alpha/rank, fused native qkv split
into to_q/to_k/to_v rows, PEFT lora_alpha/r), independent of sd.cpp and zimage_lora_convert.py.

  encode  --jobs J --work W          text-encoder pass, saves prompt embeddings
  convert --work W                   bf16 copy of the transformer for mmap (NVMe)
  render  --jobs J --work W --out O  denoise + decode with mmap'd bf16 weights
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
import struct
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "Tongyi-MAI/Z-Image-Turbo"
DIM = 3840
KOHYA = {
    "attention_qkv": "attention.qkv", "attention_out": "attention.to_out.0",
    "attention_to_q": "attention.to_q", "attention_to_k": "attention.to_k",
    "attention_to_v": "attention.to_v", "attention_to_out_0": "attention.to_out.0",
    "feed_forward_w1": "feed_forward.w1", "feed_forward_w2": "feed_forward.w2",
    "feed_forward_w3": "feed_forward.w3", "adaLN_modulation_0": "adaLN_modulation.0",
}


def snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, local_files_only=True))


def lora_modules(path: Path, alpha_override: float | None):
    groups: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), "pt") as handle:
        for key in handle.keys():
            m = re.match(r"^(.*?)\.(lora_A(?:\.default)?\.weight|lora_B(?:\.default)?\.weight|lora_down\.weight|lora_up\.weight|alpha)$", key)
            if not m:
                raise ValueError(key)
            kind = {"lora_down.weight": "A", "lora_up.weight": "B", "alpha": "alpha"}.get(m.group(2)) or m.group(2)[5]
            groups.setdefault(m.group(1), {})[kind] = handle.get_tensor(key).float()
    for stem, slot in groups.items():
        name = stem.removeprefix("base_model.model.").removeprefix("diffusion_model.")
        if name.startswith("lora_unet_"):
            m = re.match(r"lora_unet_(layers|noise_refiner|context_refiner)_(\d+)_(.*)$", name)
            name = f"{m.group(1)}.{m.group(2)}.{KOHYA[m.group(3)]}"
        if name.endswith("attention.out"):
            name = name[: -len("out")] + "to_out.0"
        down, up = slot["A"], slot["B"]
        alpha = float(slot["alpha"]) if "alpha" in slot else alpha_override
        scale = alpha / down.shape[0] if alpha is not None else 1.0
        if name.endswith("attention.qkv"):
            base = name[: -len("qkv")]
            for i, part in enumerate(("to_q", "to_k", "to_v")):
                yield base + part, down, up[i * DIM:(i + 1) * DIM], scale
        else:
            yield name, down, up, scale


def encode(args) -> None:
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16, local_files_only=True)
    prompts = sorted({job["prompt"] for job in json.loads(args.jobs.read_text())})
    embeds = {}
    with torch.no_grad():
        for prompt in prompts:
            pe, _ = pipe.encode_prompt(prompt=prompt, device="cpu", do_classifier_free_guidance=False)
            embeds[prompt] = [t.float().clone() for t in pe]
            print("encoded", prompt[:60], flush=True)
    args.work.mkdir(parents=True, exist_ok=True)
    torch.save(embeds, args.work / "embeds.pt")


def convert_bf16(args) -> None:
    src = snapshot() / "transformer"
    dst = args.work / "transformer-bf16"
    dst.mkdir(parents=True, exist_ok=True)
    for shard in sorted(src.glob("diffusion_pytorch_model-*.safetensors")):
        out = dst / shard.name
        if out.exists():
            continue
        with safe_open(str(shard), "pt") as handle:
            tensors = {k: handle.get_tensor(k).to(torch.bfloat16) for k in handle.keys()}
        save_file(tensors, str(out))
        del tensors
        print("converted", shard.name, flush=True)


def mmap_state_dict(directory: Path) -> dict[str, torch.Tensor]:
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    state = {}
    for path in sorted(directory.glob("*.safetensors")):
        handle = open(path, "rb")
        n = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(n))
        header.pop("__metadata__", None)
        buf = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
        base = 8 + n
        for key, info in header.items():
            start, end = info["data_offsets"]
            dtype = dtypes[info["dtype"]]
            count = (end - start) // torch.empty((), dtype=dtype).element_size()
            tensor = torch.frombuffer(buf, dtype=dtype, count=count, offset=base + start) if count else torch.empty(0, dtype=dtype)
            state[key] = tensor.reshape(info["shape"])
    return state


def linear_forward(self, x):
    out = F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
    for down, up, scale in self.lora:
        out = out + (x @ down.t()) @ up.t() * scale
    return out


def render(args) -> None:
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
    snap = snapshot()
    cfg = ZImageTransformer2DModel.load_config(str(snap / "transformer"))
    with torch.device("meta"):
        transformer = ZImageTransformer2DModel.from_config(cfg)
    state = mmap_state_dict(args.work / "transformer-bf16")
    linear_weights = {f"{n}.weight" for n, m in transformer.named_modules() if isinstance(m, torch.nn.Linear)}
    state = {k: (v if k in linear_weights else v.float().clone()) for k, v in state.items()}
    transformer.load_state_dict(state, assign=True, strict=True)
    for buffer_name, buffer in list(transformer.named_buffers()):
        if buffer.is_meta:
            raise RuntimeError(f"meta buffer {buffer_name}")
    modules = dict(transformer.named_modules())
    for module in modules.values():
        if isinstance(module, torch.nn.Linear):
            module.lora = []
            module.forward = types.MethodType(linear_forward, module)
    transformer.eval()
    vae = AutoencoderKL.from_pretrained(str(snap / "vae"), torch_dtype=torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(snap / "scheduler"))
    pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
    embeds = torch.load(args.work / "embeds.pt")
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "runs.jsonl", "a")
    latent_shape = (1, transformer.config.in_channels, 2 * (args.size // 16), 2 * (args.size // 16))
    for job in json.loads(args.jobs.read_text()):
        dst = args.out / f"{job['name']}.png"
        if dst.exists():
            continue
        hooked = []
        if job.get("lora"):
            for name, down, up, scale in lora_modules(Path(job["lora"]), job.get("alpha")):
                modules[name].lora.append((down, up, scale * float(job.get("scale", 1.0))))
                hooked.append(modules[name])
        latents = torch.randn(latent_shape, generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32)
        t0 = time.monotonic()
        with torch.no_grad():
            image = pipe(
                prompt_embeds=embeds[job["prompt"]], height=args.size, width=args.size,
                num_inference_steps=args.steps, guidance_scale=0.0, latents=latents,
            ).images[0]
        image.save(dst)
        for module in hooked:
            module.lora = []
        log.write(json.dumps({"name": job["name"], "modules": len(hooked), "s": round(time.monotonic() - t0, 1)}) + "\n")
        log.flush()
        print(job["name"], len(hooked), round(time.monotonic() - t0, 1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("encode", "convert", "render"))
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"encode": encode, "convert": convert_bf16, "render": render}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
