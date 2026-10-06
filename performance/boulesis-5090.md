# Boulesis-v2.1-26B-A4B on the shared RTX 5090

MoE roleplay model (Gemma-4, 26B total / ~4B active, 30 layers, 128 experts,
top-8, i1-Q4_K_S, 15.1 GiB) on a shared 32 GiB RTX 5090 with ~8 GiB free.

## Environment

- Host: 72 cores (load avg ~26), 250 GB RAM, 1x RTX 5090 32 GiB.
- Co-resident: netwrck search server, ComfyUI, matting workers, Z-Image,
  production omniserve-native (port 8791, build-full/).
- Model: `/nvme0n1-disk/models/omniserve-native/Boulesis-v2.1-26B-A4B.i1-Q4_K_S.gguf`
- Per-layer experts: ~454 MiB x 30 = 12.9 GiB; dense weights 1.9 GiB;
  KV (CTX 4096, 1 ctx, q8_0) ~0.5 GiB measured.
- Test servers: port 8799, `OMNISERVE_NATIVE_VRAM_BROKER=0`,
  `OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED=0`, footprint cap 6.5 GiB.

## Method

- Placement knobs: `OMNISERVE_NATIVE_TENSOR_OVERRIDE` (`-ot` syntax,
  first match wins, user patterns precede generated MoE patterns) and
  `OMNISERVE_NATIVE_MOE_CPU_EXPERTS=<N>|all|auto` (last N layers' experts
  on CPU). `auto` scans the GGUF header, estimates GPU bytes as non-expert
  weights + GPU experts + KV + 1 GiB compute, keeps the largest expert
  prefix fitting free VRAM minus `NGL_AUTO_KEEP_FREE_MB`. `NGL=auto`
  subtracts CPU-resident expert bytes before the full-offload decision.
- Bench: `./scripts/moe_bench.sh MODEL.gguf` sweeps
  `{all,24,16,8,auto} x SPEC_DRAFT {0,4} x {1,4} streams`,
  CTX=4096, KV q8_0, 200-token roleplay completions, thinking disabled.
  TTFT = separate max_tokens=1 probe; decode tok/s excludes it.
  Configs leaving <1.5 GiB free are aborted, not measured.
- Speculation here is prompt-lookup self-speculation (ngram match against
  the prompt), no draft model.

## Exact commands

```bash
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
./scripts/moe_bench.sh /nvme0n1-disk/models/omniserve-native/Boulesis-v2.1-26B-A4B.i1-Q4_K_S.gguf
MOE_SWEEP="24 auto" SPEC_SWEEP="0 4" CTX_SWEEP="1" ./scripts/moe_bench.sh <model>  # subset rerun
curl -s localhost:8799/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["llm"])'
curl -s -X POST localhost:8799/admin/llm/swap -H 'Content-Type: application/json' \
  -d '{"path":"<model>","ngl":"auto","tensor_override":"","moe_cpu_experts":"auto","ctx":4096,"contexts":1}'
```

## Results

Sweep run 2026-09-19 ~06:50 UTC, 8.5 GiB free at start, host load ~13-21.
`./scripts/moe_bench.sh` (see header for metric definitions). tps = single
stream decode tok/s; agg = aggregate incl. prefill; gpu_MB = nvidia-smi
delta; drafts saved is prompt-lookup speculation (no draft model).

| place | spec | ctxs | load_s | gpu_MB | ttft_s | toks | tps  | agg  | spec saved | gpu/cpu exp |
| all   | 0    | 1    | 4.1    | 2785   | 1.98   | 200  | 25.6 | 20.4 | 0          | 0/30  |
| all   | 0    | 4    | 4.1    | 5035   | 2.05   | 800  | -    | 27.0 | 0          | 0/30  |
| all   | 4    | 1    | 4.1    | 2785   | 2.09   | 200  | 25.9 | 20.4 | 0          | 0/30  |
| all   | 4    | 4    | 4.1    | 5035   | 2.00   | 800  | -    | 25.8 | 2          | 0/30  |
| 24    | 0    | 1    | 6.1    | 5463   | 1.66   | 200  | 31.0 | 24.6 | 0          | 6/24  |
| 24    | 0    | 4    | ABORT (776 MiB free < 1536 floor)                     |        | 6/24  |
| 24    | 4    | 1    | 6.1    | 5463   | 1.59   | 200  | 35.1 | 27.5 | 1          | 6/24  |
| 24    | 4    | 4    | ABORT (776 MiB free < 1536 floor)                     |        | 6/24  |
| 16    | *    | *    | LOAD_FAIL x4 (14 GPU layers need ~10.3 GiB)           |        | -     |
| 8     | *    | *    | LOAD_FAIL x4 (22 GPU layers need ~13.9 GiB)           |        | -     |
| auto  | 0    | 1    | 6.1    | 5023   | 1.63   | 200  | 29.8 | 24.0 | 0          | 5/25  |
| auto  | 0    | 4    | 4.1    | 5035   | 1.96   | 800  | -    | 28.6 | 0          | 0/30  |
| auto  | 4    | 1    | 6.1    | 5023   | 1.73   | 200  | 28.6 | 22.9 | 3          | 5/25  |
| auto  | 4    | 4    | 4.1    | 5035   | 2.11   | 800  | -    | 24.2 | 1          | 0/30  |

Notes:

- Best single-stream decode: 35.1 tok/s (`MOE_CPU_EXPERTS=24`, 6 GPU
  layers, 5.3 GiB). Best aggregate: 28.6 tok/s (`auto` + 4 contexts,
  all experts CPU, 5.0 GiB, ~7 tok/s per stream).
- `auto` adapts correctly: 5/25 for 1 context, 0/30 for 4 contexts at
  this free level. Matches the independent header-based estimate.
- Prompt-lookup speculation saves ~0 calls on creative roleplay; the
  spec=4 deltas are run-to-run noise on this shared box (a reversed-order
  rerun flipped the winner), not signal. Keep `SPEC_DRAFT=0`.
- Over-full placements fail gracefully (`llm not ready`, no crash) and
  the 1.5 GiB floor abort works; nothing was measured in a degraded
  state. Estimates run ~1.2 GiB conservative (1 GiB compute reserve +
  peak-head KV math vs ~0.5 GiB measured), which is the safe direction.
- Independent repeat (run B, sole-listener-guarded harness, same script):
  all/0/1 26.8, all/4/1 33.7 (0 drafts: noise), 24/0/1 38.9, 24/4/1 37.9,
  auto/0/1 34.8 (5/25), auto/0/4 28.9 agg (0/30), auto/4/1 30.0,
  auto/4/4 27.0 agg; 16/8 LOAD_FAIL x8; 24/x/4 ABORT on the floor. Same
  ordering (24 > auto/5 > all; spec ~0), +0-25% vs run A: run-to-run
  noise on this loaded box dominates small deltas. Combined best
  single-stream: 38.9 tok/s (24/0/1); best aggregate: 28.9 tok/s
  (auto/0/4).

## Recommended production profile

`systemd/omniserve-native-boulesis.conf`: `NGL=auto`,
`MOE_CPU_EXPERTS=auto`, `CTX=4096`, 1 context, `KV_TYPE=q8_0`,
`BATCH/UBATCH=auto`, threads 16/16, `SPEC_DRAFT=0`, `LLM_SWAP_DIR` set so
the model hot-swaps. `auto` is chosen over pinned 24 because free VRAM on
this box moves between ~6 and ~8.5 GiB: pinned 24 fails closed (`llm not
ready`) when free drops, while `auto` degrades to fewer GPU experts and
`NGL=auto` further to CPU placement instead of failing. Expected: ~30
tok/s single-stream at ~5 GiB with 8 free; ~26 tok/s at ~2.8 GiB with 6
free. For a pinned conservative floor, uncomment the `=all` line in the
conf (all experts CPU, ~2.8 GiB, ~26 tok/s single / ~27 agg with 4
contexts).

## Chat template notes (Gemma-4)

- `format_gemma4_prompt` renders the plain-message subset of the GGUF's own
  `tokenizer.chat_template` exactly: BOS via add_special, `<|turn>system`
  block, per-message `<|turn>{role}` + trim + `<turn|>` (assistant maps to
  `model`, other roles pass through, tool messages skipped), consecutive
  assistants merge (open suppressed on the later, close suppressed on the
  earlier per this checkpoint's template), generation prompt
  `<|turn>model` + empty thought close when thinking is off, model-history
  thinking stripped.
- Verified: system persona + 6 alternating turns, coherent in-character
  reply, `finish_reason=stop` at `<turn|>` (EOG), no control tokens in
  output with `enable_thinking=false`.
- Thinking models may still open a `<|channel>thought>` block with thinking
  disabled; the decoder excises closed blocks, holds an unterminated block
  out of streamed output, and drops a dangling block at end of generation.
  Thought tokens still cost time (see TTFT/tok/s with thinking on vs off).
- `enable_thinking` defaults to false (`OMNISERVE_NATIVE_LLM_THINKING_DEFAULT=1`
  opts back in); roleplay callers omit the field. With thinking off the
  generation prompt closes the thought channel and any thought text the model
  still emits is excised from streaming and non-streaming output alike.

## MTP speculation (CPU-only validation)

Head: `/nvme0n1-disk/models/omniserve-native/mtp-gemma-4-26B-A4B-it-Q8_0.gguf`
(arch `gemma4-assistant`, 0.43 GiB, n_embd 2816, 4 nextn layers). Test server on
port 8797, CPU-only (`NGL=0`, `MOE_CPU_EXPERTS=all`, broker/prefetch off,
`CUDA_VISIBLE_DEVICES` empty; nvidia-smi confirmed zero VRAM on every run),
16 threads, CTX 4096, KV q8_0, one context. 4 roleplay prompts (system persona
+ 2-turn history, same shapes as the llama-server probe in `/tmp/mtp-meas`),
160 new tokens, temp 0 (top_p 1.0) and 0.9 (top_p 0.9), non-streaming
`/v1/chat/completions`. tok/s is token-weighted wall time including prefill;
acceptance from `/status.speculation` deltas per request.

| temp | config      | tok/s  | acc      | rate  |
|------|-------------|--------|----------|-------|
| 0    | off         | 8.00*  | -        | -     |
| 0    | mtp draft 1 | 8.92** | 146/253  | 57.7% |
| 0    | mtp draft 2 | 8.98   | 172/333  | 51.7% |
| 0.9  | off         | 6.16*  | -        | -     |
| 0.9  | mtp draft 1 | 6.82   | 149/270  | 55.2% |
| 0.9  | mtp draft 2 | 6.56   | 145/306  | 47.4% |

\* pooled over 3 (temp 0: 7.46/8.43/8.16) and 2 (temp 0.9: 6.19/6.14) server
restarts; the host is noisy and the first window caught the worst of it.
\** pooled over 5 restarts (9.01/9.08/8.86/8.91/8.74, ±2%), identical
draft/accept counts every run.

- Draft 1 gains +11.5% (temp 0) / +10.7% (temp 0.9) over pooled off. Draft 2
  matches draft 1 at temp 0 and loses at temp 0.9: the second position accepts
  rarely while each extra draft costs a full head decode on CPU. Default stays 1.
- `SPEC_MTP_P_MIN=0.5` accepts more (158/258, 61.2%) but runs 7% slower than
  no filter (8.48 vs 8.92): the head decode is paid before confidence is
  known, so the filter only saves the verify widening. Default 0.
- Temperature-0 outputs are NOT identical to non-speculative (all 4 prompts
  diverge; e.g. P4 flips `Oh! Oh!` to `Oh, oh!` at char 74 — a near-tie
  argmax). Cause is exactly the verify width, not MTP machinery: same-config
  runs are bit-identical across restarts for off, MTP, and prompt-lookup
  alike, and a prompt-lookup request that drafted 0 tokens came out
  bit-identical to off while every request that fired diverged. The width-2
  verify batch sums in a different order than two width-1 decodes; same
  documented effect as prefix reuse. All outputs fluent and in-character.
- No control tokens (`<|channel>`, `<channel|>`, `<|turn>`, `<turn|>`) in any
  of the 40+ validation outputs, temp 0 and 0.9, streaming and non-streaming.
- Streaming works; 4 concurrent streams (1 slot, serialized) all complete
  with identical temp-0 text; counting prompt accepts 10/10.
- Precedence: `SPEC_DRAFT=4` + head loaded reports `source=mtp`,
  `draft_max=1`; after swap-unload of the head it falls back to
  `source=prompt-lookup` with no restart.
- Swap: `{"spec_mtp_gguf":""}` unloads the head (ok:true); invalid path
  (`/tmp/evil.gguf`) rejected with ok:false and the previous model restored
  serving; mismatched head (modernbert, n_embd 768 vs 2816) logs
  `head n_embd 768 != target 2816, MTP off` and serves normally. Never crashes.
- Prompt-lookup (`SPEC_DRAFT=1`, no head) still works post-refactor: 2/6
  accepted on roleplay, 7.88 tok/s — no win on CPU, as previously documented.

GPU note: unmeasured. The head would need ~0.4 GiB beside the weights; on a
bandwidth-bound device the +11% CPU gain is a floor, not a ceiling.

## Not implemented (decode-speed ideas)

- ~~Draft-model speculative decoding~~: done via the MTP head above (+11% CPU
  on roleplay). A small dense draft model (e.g. Qwen3-0.6B) remains untested.
- Pinned hot experts on GPU (usage-counted expert cache): +10-25% by
  keeping the few hot experts of CPU layers resident instead of paging.
- Prefill-only expert offload (experts on GPU for prefill, paged for
  decode): cuts TTFT without hurting decode.
- Tighter KV estimate (per-layer head counts instead of peak): auto would
  place ~1 more expert layer on GPU at 8 GiB free.
- `--no-mmap` for CPU-resident expert buffers: llama.cpp warns mmap+CPU
  overrides cost throughput; measurable on the CPU-expert matmuls.
- Larger ubatch for the verify path when SPEC_DRAFT>0.
- IQ4_XS/IQ1-class requant of experts: smaller GPU footprint per layer,
  more GPU-resident experts at fixed VRAM.
