# Z-Image acceleration roadmap — RTX 5090

## Shipped or staged

1. **Exact encoded-result cache (native C).** A repeated exact teleport key can
   return the stored WebP/PNG without denoising, VAE decode, or re-encoding. This
   is byte-preserving and shares eviction with the latent cache.
2. **Exact final-step latent replay (stable-diffusion.cpp).** The production
   patch preserves the Euler schedule and global step numbering. The measured
   1024² repeat was byte-identical and cost `$0.00109/MP` before encoded-result
   caching.
3. **Regional compilation (CuteDSL).** `ZIMAGE_COMPILE_MODE=regional` compiles
   repeated transformer/refiner blocks; `regional:<mode>` selects another
   Inductor mode. It remains opt-in until a production-GPU cold/warm sweep.
4. **Cost-aware parity tooling.** The image parity runner can independently set
   reference/candidate step counts and reports server inference cost per MP.

## Next production-GPU ablation order

| Priority | Candidate | Expected value | Quality gate |
| --- | --- | --- | --- |
| 1 | Transformer residency / `STREAM_LAYERS` sweep | The live 2 GiB cap makes every request pay CPU transfer overhead; a dedicated 5090 should keep more layers resident | Exact same seed, pixel metrics, VRAM headroom, p50/p95 |
| 2 | Encoded-result cache canary | Removes all model work for exact repeats | Byte-identical output and cache-key isolation |
| 3 | Regional compile, fixed size buckets | Kernel fusion with much lower cold compile cost than whole-model compile | Same-seed pixel/semantic scores; cold and warm timing |
| 4 | Native VAE tile sweep (off, 128, 64) | Reduce fixed cost at small/medium sizes when VRAM permits | Seam detector plus pixel/semantic parity |
| 5 | Attention backend sweep | PyTorch SDPA/Flash and Sage variable-length paths may help CuteDSL | Exact corruption gates and prompt/CLIP/aesthetic scores |
| 6 | Layerwise FP8 or Blackwell NVFP4 | More residency and bandwidth headroom | Corpus FID/CLIP/aesthetic and blinded review |
| 7 | First-block/cache-diffusion methods | Skip transformer blocks across nearby diffusion steps | Treat as approximate; per-prompt quality gates required |

Do not combine candidates in the first sweep. Promote one at a time, then test
interactions; otherwise a speed win and a quality loss cannot be attributed.

## Current upstream evidence

- PyTorch's Diffusers guide reports regional compilation retaining roughly the
  full-model runtime gain while reducing cold compilation from 67.4 s to 9.6 s
  in its Flux/H100 example. It also recommends dynamic shapes or fixed buckets
  to control recompilation:
  <https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/>
- Diffusers supports group offload and layerwise casting, including FP8 storage
  with higher-precision compute, but warns that custom casting and PEFT paths
  need compatibility testing:
  <https://huggingface.co/docs/diffusers/optimization/memory>
- Diffusers exposes selectable attention backends; availability and architecture
  constraints differ, so the existing variable-length Z-Image path needs a real
  compatibility sweep:
  <https://huggingface.co/docs/diffusers/v0.36.0/en/optimization/attention_backends>
- NVIDIA Model Optimizer offers FP8/NVFP4 quantization and cache diffusion. Its
  published TensorRT support matrix lists Flux, SD3, and SDXL families, not
  Z-Image, so applying it here is an experiment rather than a supported drop-in:
  <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/README.md>

The highest-confidence native-C win is residency: the current production
profile streams layers under a 2 GiB image-backend cap on a GPU shared with
other tenants. Compiler, attention, and approximate cache experiments should be
measured only after a quiet-host residency baseline, or contention will dominate
the comparison.
