# OmniServe notes (scratch, read-only research, 2026-10-08)

Scope: /vfast/data/code/omniserve-native (main, branch ra2-overflow @1e966e8, clean), siblings -tiers, -lorastack, -turbo.
All paths below relative to omniserve-native unless noted. Line refs are for main HEAD.

## 0. TL;DR

- Two very different "omniserve" runtimes live in one repo:
  1. C gateway `omniserve-native` (src/main.c, 4409 lines): epoll HTTP, weighted tier admission, embedded llama.cpp + stable-diffusion.cpp (one model of each kind per process), pooled proxy to Python workers, VRAM broker, overflow to remote. NOT a multi-model host.
  2. Python manifest runtime `runtime/handler.py` + `workloads/*.py` for RunPod: multi-workload, LRU co-residency by free VRAM, brokered leases, OOM-profile fallback. No idle timer, no metrics.
- Idle unload exists only in pieces: LLM evicted under broker pressure (main.c:1673), guard judge auto-admit/evict (main.c:1773), YuE child idle exit (workers/yue_worker.py:97), BiRefNet video idle release (workers/birefnet_worker.py:396). No timer-based idle unload for LLM, embedding, SD ctx, or Python workloads.
- SD (diffusion) ctx has no unload path at all: no free_sd_ctx dlsym (src/backend_sd.c:252-261), no osd_shutdown (obackend.h:243-248).
- Utilization metrics: none. /metrics has no GPU util, no per-model residency, no busy-seconds. NVML wrapper only loads mem + process fns (src/osched.c:377-381).
- models/models.csv is legacy (4 rows), read only by src/admission.c + tests/test_admission.c + legacy Makefile/Dockerfile; the CMake server never reads it, and main.c does not parse `--models` (main.c:3955-3968 handles only --port/--help).

## 1. Architecture, languages, build, binaries

Languages
- C11 gateway: src ~21.2k lines (main.c 4409, backend_llama.c 2965, oguard.c 2746, backend_sd.c 1151, oimage.c 1091, oproxy.c 1025, ohttp.c 886, ovram.c 788, ...). include/ 19 headers.
- CUDA: cuda/omatte_cuda.cu (foreground colour decontamination; opt-in `-DWITH_MATTE_CUDA`, archs 80;86;89;90;120).
- Python 3.12 (RunPod base image): runtime/ (handler.py 354, cog_pod.py 136, local_worker.py 117), workloads/ (qwen_image 980, qwen_mt 405, video_matting 566, anima_image 389, wan_animate_2 311, matanyone 294, person_detection 158, h3_video 49), workers/ (asr, birefnet, yue, trellis2, remote_3d, object_store, gpu_admission), frontier/ (routing ledger), tools/, tests/ (~80 files, mixed C and pytest).
- Second C program `music3c/` (main.c 975, music3.c 557): RunPod job-loop worker for MiniMax-Music3, supervises an sglang-omni server child.
- Legacy C: src/admission.c, jobs.c (+sqlite), jobctl.c, breaker.c, wavwrap.c built by root `Makefile` (stale: BIN_SRC lacks obroker/oguard; `make run` passes `--models` which main.c ignores).

Build (CMakeLists.txt, CMakePresets.json)
- Presets: dev (no LTO), release (LTO, -march=native), sanitize (ASan+UBSan), ci (portable -Werror). Ninja + mold/lld + ccache auto-detected (CMakeLists.txt:24-37).
- Deps are sibling checkouts, not vendored: `../llama.cpp/build/bin/libllama.so` (CMakeLists.txt:79-96, CUDA sm_120) and `../stable-diffusion.cpp/build/bin/libstable-diffusion.so` (WITH_SD=ON, dlopen'd at runtime, src/backend_sd.c:245-251). SD feature probes: header grep for `stream_layers` and `sd_latent_replay_params_t` (CMakeLists.txt:113-128) -> OMNISERVE_SD_STREAM_LAYERS / OMNISERVE_SD_LATENT_API.
- core static lib `onative_core` (CMakeLists.txt:64-75): ocapacity ohttp oimage ojson olog omatte onsfw oproxy osched ofrontier ohost ovram obroker ospec oscale otext otune oguard.

Binaries (CMake): `omniserve-native` (main.c docs.c backend_llama.c backend_sd.c), `omniserve-asr-router` (src/asr_router.c, relay w/ health+VRAM check+transient fallback), `oguard_sweep`, `omatte` (shared lib) + `omatte_cli`, `onative_bench` (http load gen), test exes (onative_tests, frontier, guard*, image_decode, sd_cache_stub .so). Makefile: `omniserve`, `omni-job` (sqlite job queue CLI), `music3c`, `wavwrap`.

Process model (one box, many processes) - see systemd/ (units + drop-in .conf): :8791 main gateway (LLM Gemma, ModernBERT, Z-Image via embedded SD), :8792 second gateway for Qwen Image 2.1 "ra2" (embedded SD, leases VRAM from :8791 via src/obroker.c), plus workers: birefnet :9094, asr worker + router, trellis2 3D :9093, yue worker, frontier timer. Deploy by drop-ins `zzz-*.conf` overriding ExecStart (systemd/gpu-sched/*.conf).

Concurrency core: epoll reactors (default 2, main.c:4363) + blocking worker pool (default 32, main.c:4085); SO_REUSEPORT (ohttp.c:796); body cap 80 MiB (ohttp.h:11); handlers are blocking and hold a permit for the whole response.

## 2. Model registry and supported models

"Registry" is not one thing. Four declaration mechanisms:

| mechanism | where | used by |
| --- | --- | --- |
| env vars naming a file | `OMNISERVE_NATIVE_LLM_GGUF`, `_EMBEDDING_GGUF`, `_SD_MODEL`/`_SD_DIFFUSION_MODEL`+`_SD_VAE`+`_SD_LLM`, `_GUARD_JUDGE_GGUF`, `_SPEC_MTP_GGUF` (main.c:4227-4322) | C gateway, one slot per kind |
| `/v1/models` (main.c:1052) | lists whatever is loaded + each configured upstream with hardcoded family labels | OpenAI clients |
| models/models.csv `id,label,weights_mib,runtime_mib` | admission.c `load_models` | legacy only |
| workloads/workloads.json `{name:{module,required_mib,aliases,profiles[]}}` | runtime/handler.py:36 `_load_registry`; module must expose `handler(job)` and optional `release()` | RunPod serverless / local_worker / cog_pod |
| frontier/seeds.json | per-workload backend candidates, p50, $/job | frontier router (C gateway hot-reloads routing.json) |
| qwen_mt.py `TASKS` dict (qwen_mt.py:54-70) | standalone RunPod entry (NOT in workloads.json) | ra2 + edit multitenant worker |

models.csv rows (MiB weights+runtime): qwen2.5-7b-qlora 4608+2048; anima-gemma-qlora 2048+1024; video-matting-rvm 105+1695; anima-2.9b-image 5570+7230.

workloads.json `required_mib`: video-matting 2800; h3-video 43000; anima-2.9b 12800; qwen-image-2.1 12000; wan-animate-2 24576 default with profiles small 20480(+6144 scratch) / balanced 28672(+8192) / throughput 57344(+12288). Aliases e.g. ra2, h3, anima, matting.

Supported by modality
- LLM chat/completions: llama.cpp GGUF embedded. Gemma 4 roleplay q8_0 (8.0 GB file, prod default), Gemma 4 MeroMero 31B IQ4_XS (16.9 GB, NGL=20 -> ~5.8 GiB VRAM), Qwen3.5-4B q8_0 (4.5 GB), Boulesis-v2.1 26B-A4B MoE i1-Q4_K_S (15.1 GiB), Qwen3-0.6B q8 (dev). Qwen3.5 vision via external mtmd worker (mmproj-qwen3.5-4b-f16 672 MB). Optional Gemma-4 MTP draft head 0.43 GiB (README:433). Remote: vLLM/Gemma upstream via `LLM_UPSTREAM`.
- Embeddings: ModernBERT / gte-modernbert-base q8_0 (160 MB), embedded encoder, CPU default (EMBEDDING_NGL=0), cls or mean pooling.
- Guard: ShieldGemma-2B q4_k_m judge (~3.2 GB; broker need default 2304 MB, main.c:1668), gte-modernbert seed cosine, lexical rules (src/oguard.c).
- Image: Z-Image Turbo Q4_K (embedded SD; 9 steps; WebP), Qwen Image 2.1 "ra2" Q4_K_M (DiT 4.6 GB + Qwen3-VL-8B TE Q4_K_M 4.7 GB + mmproj 1.1 GB + VAE 0.6 GB = ~11 GB), Qwen Image Edit 2511 Q4_K_M (+Qwen2.5-VL-7B bf16 16.6 GB TE), Anima 2.9B (torch worker, workloads/anima_image.py), sd-turbo safetensors (README example). Legacy CuteDSL image worker upstream.
- Background removal: BiRefNet (fp16 torch worker, workers/birefnet_worker.py) + C/CUDA decontamination (src/omatte.c, cuda/omatte_cuda.cu); video matting RVM resnet50 torchscript + MatAnyone + YOLO11n person detector (workloads/video_matting.py, matanyone_matting.py, person_detection.py).
- Video: H3 (43 GB class, via app.nz cog seam, workloads/h3_video.py adapter), Wan-Animate-2 (diffusers git pin, NF4 bitsandbytes for small/balanced profiles).
- Music: YuE2-3B (workers/yue_worker.py, child process, 18 GiB admit), MiniMax-Music3 (38 GB checkpoint, sglang-omni, FP8 backbone option; music3/handler.py Python, music3c C worker).
- TTS: Supertonic/Kokoro via CPU compatibility worker :9083 (upstream only, not in this repo). STT: whisper-small.en default (WER 1.74%, RTF 0.18 on 5090, docs/asr-model-selection.md), Parakeet via workers/asr_worker.py; asr_router in C.
- 3D: TRELLIS.2-4B (+ DINOv3 gated encoder), Pixal3D (remote RunPod fallback), exclusive background lane.
- Forecast: Chronos-2 upstream (/forecast).
- Animation: NVIDIA ACE adapter upstream.

File formats / quantization
- GGUF for llama.cpp and sd.cpp (Q4_K, Q4_K_M, Q8_0, IQ4_XS, F16 mmproj); safetensors for VAE/LoRA/sd-turbo/bf16 TEs; torchscript (RVM); .pth; torch diffusers dirs (Anima/Wan). Conversion: scripts/convert_models.sh (modernbert, gemma, qwen), prepare_gemma4_model.sh (capacity-checked, IQ4_XS default; NVFP4 refused as not a llama-quantize target), prepare_gemma4_nvfp4.sh (ModelOpt, experimental).
- KV cache quant: q8_0 default on blackwell/hopper/ada/ampere (src/otune.c; table README:346). MoE placement: `TENSOR_OVERRIDE`, `MOE_CPU_EXPERTS=N|all|auto`.
- sd.cpp runtime LoRA (up to 8 per request) and per-request `lora_id` from cache dir `OMNISERVE_NATIVE_LORA_DIR` (src/oimage.c:64-240).

VRAM footprints / load times documented (measured unless marked)
- LLM Gemma roleplay q8: 4.76 GiB CUDA buffer; cold reload from warm page cache 2.7 s (performance/gpu-scheduler-2026-09-24.md). Each extra llama context costs 1.36 GiB (KV+compute) measured (systemd/omniserve-native.service comment).
- Boulesis 26B-A4B on 5090 (performance/boulesis-5090.md): all experts on CPU: 2785 MB VRAM (1 ctx), 5035 MB (4 ctx), load 4.1 s, 25.6 tok/s; 6 GPU expert layers: 5463 MB, load 6.1 s, 31-35 tok/s.
- Z-Image Turbo Q4_K (performance/prod-5090-image-residency-2026-09-30.md): streamed (STREAM_LAYERS=1, MAX_VRAM=2): 5.4 GB peak, 1024^2 p50 15.5 s; resident tile64: 12.2 GB, 5.1 s; resident untiled 17.6 GB, 4.6 s; first resident request pays ~11 s weight load; exact cache hit 1-2 ms.
- Qwen ra2: +10.3 GB per 1024^2 render, ~7.7 GB at 768^2 (lease sizing main.c:2565-2575); resident worker ctx load 410-680 ms from page cache (3090 Ti); 4090 warm 1024^2 20 step 8.8-9.5 s; 5090 turbo 8.5-9.2 s, base 30-step 10.3-11.0 s.
- Per-tenant box table (24 h, 1 s NVML): gemma+ZImage 6.4/8.4 GB idle/peak, qwen :8792 1.2/11.4, netwrck search 0.95/5.4, tts-gpu 2.7 static, birefnet 1.1/4.0 (+3584 MB standing lease), chronos 1.1, cutedsl 0.5; box p50 19.0 GB, max 31.9 of 32.6 GB. Gemma idle >10 min gaps cover 91% of the day (455 calls/24.5 h).
- YuE2: admit 18 GiB, observed PyTorch peak 7.4-7.7 GiB; first render 64.0 s runtime vs 24.9 s warm; RunPod first-job queue/init 178 s.
- BiRefNet throughput vs permits: 2.98 / 4.95 / 7.29 img/s at 1/2/4.
- Not measured anywhere: GPU decode speculation on this host, Ampere profile rows (reasoned), Music3 VRAM/loaded time (no numbers in repo), video/H3 loaded times.

## 3. Load / swap / unload / residency

C gateway (single process)
- One global of each: LLM (`g_model`, slot pool `g_slots`, backend_llama.c:1324 init, 2378 shutdown, rwlock `g_model_lifecycle_lock` at 2403-2420: writers wait for in-flight chats), embedding (`g_embed_model`, oembed_init 2425), judge (`g_judge_model`), SD ctx (`g_sd`, backend_sd.c:273). No registry of N models; co-residency means different KINDS, or different PROCESSES.
- llama weights: mmap default (llama_model_default_params; no `use_mmap`/`use_mlock` override in backend_llama.c), no pinned host RAM. Host side: `ohost_prefetch_start` does posix_fadvise(WILLNEED) in 64 MiB chunks with MemAvailable floor (src/ohost.c, main.c:4065-4078), defaults to LLM+embedding GGUF paths. sd.cpp: `OMNISERVE_NATIVE_SD_MMAP` default true, `_EAGER_LOAD`, `_AUTO_FIT`, `_PARAMS_BACKEND` (diffusion=cpu / *=cpu streaming), `_STREAM_LAYERS`, `_MAX_VRAM` (backend_sd.c:735-772).
- Admin lifecycle (loopback only, `request_is_internal`): `POST /admin/llm/unload|load|swap` (main.c:3242, 3252, 3265; aliases /admin/unload etc, routes at 3506-3513). Load/swap path must sit under `OMNISERVE_NATIVE_LLM_SWAP_DIR` (main.c:149, realpath allow-list). swap restores previous model on failure (main.c:3288). `g_llm_active_*` records the live config, seeded at boot (main.c:4237-4256).
  - GAP: `/admin/llm/unload` (main.c:3242) calls ollm_shutdown but does not set `g_gs.llm_evicted`, so the next chat gets 503 until /admin/llm/load; scheduler eviction (main.c:1706) does set it and lazy-reloads.
- Broker (src/ovram.c, ovram.h): lease/renew/release/wait ledger with TTL expiry evaluated per call, tier ranking (higher tier never starved by lower), keep-free floor (default 1024 MB), reserve declarations (`VRAM_RESERVE=name:mb`), per-pid materialisation credit (NVML per-process), young-lease (<180 s) rule, block-max 15-30 s, pressure hook `ovram_set_pressure_hook(fn, ctx, after_ms)` (ovram.h:93-99; installed main.c:4052-4055 only when NOT leasing from a remote broker). Never allocates; permission only. VRAM accounting = NVML free (`ogpu_memory_gib`, osched.c:396) minus granted-unmaterialised leases. Remote broker client: src/obroker.c (second gateway leases from :8791; fail-open if unreachable).
- Pressure eviction order (main.c:1673-1717): (1) guard judge if `judge_auto`, (2) embedded LLM if idle >= `OMNISERVE_NATIVE_EVICT_LLM_IDLE_S` (default 0 = never) and waiter tier <= `EVICT_LLM_MAX_TIER` (default sub) and `g_llm_swap_lock` trylock succeeds. Prod: 600 s idle, sub (systemd/gpu-sched/omniserve-native.zzz-gpu-sched.conf). It is pressure-only: a waiter must be blocked `VRAM_PRESSURE_AFTER_MS` (1500) first; nothing frees memory just because the model is idle.
- Lazy reload: `llm_ensure_loaded` (main.c:1725) called on the three LLM handlers (main.c:1915, 2144, 2987); leases `llm_mb` (last freed, default 6144) with `ovram_lease_wait` first so reload never lands on CPU; rejects CPU landing (main.c:1743). Embedding and SD have no lazy reload.
- Judge auto-admit (main.c:1773): thread, 15 s poll, needs 4 consecutive quiet polls with headroom >= need + margin; back-off 600 s after eviction, 1800 s after failed load.
- Background exclusive window (3D/animation/training): osched background tier requires active==0 && used_slots==0 (osched.c:fits_locked); `unload_embedded_models_for_background` (main.c:1227) shuts down embedding + LLM, `reload_embedded_models_after_background` re-inits from env (main.c:1234-1262; note: uses env, not `g_llm_active_*`, so a swapped model reverts).
- No LRU across models in C (nothing to evict among). LRU exists only for the SD teleport/result cache (`OMNISERVE_NATIVE_SD_TELEPORT_CACHE_SIZE`, default 64, max 256; backend_sd.c:735+) and guard judge verdict cache.

Python runtime (runtime/handler.py) - this is the real multi-model host
- `_loaded_modules{name: module, required_mib, profile, last_used}` (handler.py:30). `dispatch` (277) under global `_dispatch_lock`: resolve alias -> pick profile (`_profile_for` 207, counts reclaimable resident MiB) -> if not resident, `_make_room` (193): while free < required + `OMNISERVE_GPU_RESERVE_MIB`(512), release oldest `last_used` (true LRU by need) -> `_acquire_vram_lease` via `OMNISERVE_VRAM_BROKER_URL` `/v1/gpu/lease` background tier ttl 10800 (133; required only if `OMNISERVE_VRAM_BROKER_REQUIRED`) -> import module (lease before import because plugins preload CUDA at import) -> handler -> release lease. CUDA OOM: `_release_all`, drop to previous profile (`_fallback_profile` 235), retry. Response carries `omniserve.resident_workloads`, `reused`, `oom_retries`.
- Co-residency happens naturally if they fit (test_runtime.py `test_dispatch_keeps_workloads_resident_when_they_fit`). Unload only by LRU-for-space or OOM; no timer. Plugin contract: `handler(job)`, `release()` (qwen_image.py:613, wan_animate_2.py:295, h3_video.py:34).
- qwen_mt.py: 2 sd.cpp contexts (ra2, edit) in one process. `plan(total)` (153) picks placement `""` / `te=cpu` combos that fit both; `ensure` (201) loads, evicting LRU ctx until it fits (`MT_RESERVE_GIB` 1.0), else `*=cpu` streaming; `boot` (385) preloads per `MT_PRELOAD` (auto = both if plan fits). Measured: 96 GB both on GPU; 48 GB ra2 GPU + edit TE in RAM; 24 GB first edit evicts ra2 (18.9 s), then both resident with TEs in RAM (5.4 GiB free); swaps 6-19 s with page-cache weights. `_ctxs[name].used` is the LRU clock. No idle timer.
- workers: yue_worker `reap_idle` (workers/yue_worker.py:97) exits the child after `YUE_IDLE_SECONDS`=30 or if free VRAM <2048 MiB; birefnet `_idle_release_loop` (birefnet_worker.py:396) drops video models after `VIDEO_MATTING_IDLE_RELEASE_SECONDS`=300, holds renewable broker lease for the image model; asr_worker `/admin/hold|release` unloads for training windows; trellis2 one-shot with `/admin/hold` peer coordination (`OMNISERVE_3D_GPU_COORDINATORS`); gpu_admission.py `AdaptiveCudaGuard` learns a free-MiB floor + exponential cooldown after OOM.

## 4. LoRA stacking, tiers, turbo and how the siblings differ

Naming trap: "tiers" means three things.
1. Scheduler tiers paid/sub/free/background (osched.h:7). Key->tier map `OMNISERVE_NATIVE_KEY_TIERS="key=sub,..."` (main.c:247, 64 keys, key tier is a ceiling, `X-Omniserve-Tier` can only lower; public callers cannot claim background).
2. Image quality tiers (ra2): turbo vs base.
3. Offload tiers (local 5090 -> daisy 3090 -> RunPod), performance/offload-tiers-2026-09-22.md.

Image quality tiers (main)
- turbo (default for plain text-to-image when `OMNISERVE_NATIVE_SD_TURBO_NODES` + `_DEFAULT_LORA` set): Viggle Qwen-Image-2.1 v0.3 6-step r256 LoRA applied at runtime on Q4_K_M base, custom sigmas from nodes 1.0,0.9375,0.875,0.75,0.5,0.25 with resolution shift (backend_sd.c:417-430 `turbo_sigmas`, 860-866, 909-914). `"turbo":false` -> base path: steps floored to `SD_MIN_STEPS` (30), EasyCache `SD_HQ_EASYCACHE_THRESHOLD` 0.15, no LoRA. Nyquist notch on both (`SD_NOTCH=1`, `notch` request field).
- Turbo only when `!image_pixels && !lora_count && req->turbo != 2` (backend_sd.c:860): reference edits and any request carrying a LoRA fall back to base schedule.
- Per-request overrides in `oimg_req` (obackend.h:204-212): cache_threshold, cache_end, cache_off, notch, turbo, flow_shift, sampler, scheduler, extra_args (guidance_schedule counts capped at 100, oimage.c commit cd71a22). Any override bypasses result cache.
- Numbers (qualitybench/README.md): 1024^2 5090 turbo 8.5-9.2 s vs base 10.3-11.0 s; pre-merged turbo GGUF (Q6_K 10.4 s / Q5_K_M 9.8 s vs runtime LoRA 11.5 s on 3090 Ti) is better and faster but needs a second diffusion ctx (TE loaded twice), not wired.
- Same tiers exist in the RunPod worker: workloads/qwen_image.py:99-106,763-777 (RA2_TURBO*, RA2_MIN_STEPS, RA2_HQ_CACHE_THRESHOLD) and qwen_mt.py ra2 (tiers not deployed to the baked image: "needs an image layer push").

LoRA mechanics (main)
- Request: trusted loopback `loras:[{path,scale}]` max 8, scale in [-4,4]; public: `lora_id` (+`lora_filename`,`lora_scale`) resolved cache-only under `OMNISERVE_NATIVE_LORA_DIR`, registry json `OMNISERVE_NATIVE_LORA_REGISTRY` (oimage.c:64-240, 610-730). Auto NSFW LoRA on prompt word match (`NSFW_LORA_PATH`, oimage.c:259).
- Applied by sd.cpp `sd_lora_t` per request (backend_sd.c:897-913); LoRA path+scale are part of the teleport/result cache key (backend_sd.c:517-545). LoRA change does NOT reload the base ctx.
- Main behaviour: default turbo LoRA XOR request LoRAs (default applied only when `turbo` which requires lora_count==0, backend_sd.c:909).

omniserve-native-tiers
- git worktree, branch ra2-tiers @1e966e8 = same commit as main HEAD (main merged it at 493b1d4). File diff vs main is only untracked/pycache/evals/qualitybench-out. Treat as redundant copy; daisy runs `build/omniserve-native` from `~/omniserve-native-tiers` (RA2_OVERFLOW.md:2026-10-02).

omniserve-native-lorastack
- git worktree, branch ra2-lorastack @6aac6e5 (an ancestor of main) + ONE uncommitted change: src/backend_sd.c (+18/-6): new env `OMNISERVE_NATIVE_SD_TURBO_STACK` -> `g_cfg.turbo_stack`; turbo allowed when `(!lora_count || turbo_stack)`; builds `sd_lora_t stacked[lora_count+1]` = [default turbo LoRA] + request LoRAs (heap, freed on all 3 exits). Also untracked qualitybench/{prompts_v2.json,review.py,score2.py,sweep2.py} and out/lora-20261005. CMakeLists differs only because it predates main's `_sd_feature_defs` refactor. Net: lorastack = "request LoRAs compose with turbo instead of disabling it"; not merged, not deployed. Needs a test that stacked LoRAs enter cache key (they do, key hashes request loras only: the default LoRA is implicit, fine while default is global).

omniserve-turbo
- Not a git repo; snapshot dated 2026-10-01 matching branch `prod-turbo` tip 6b50579 ("distilled few-step turbo schedule and default LoRA"). Has gateway turbo (backend_sd.c:366-412) but lacks: per-request quality fields in obackend.h, HQ EasyCache threshold/min steps, `turbo`/`notch` request knobs, worker-side turbo in workloads/qwen_image.py (0 TURBO refs), qualitybench/, tools/*.sh, tests for tiers; WebP q88 and EasyCache 0.05 defaults in deploy/qwen-ra2-prod.sh. Superseded by main.

## 5. Batching, queueing, concurrency, multi-tenancy

- Single global admission scheduler `osched` (src/osched.c): `OMNISERVE_NATIVE_SLOTS` (default 1, max 64, main.c:3972) are permit units; each modality takes N permits: LLM_PERMITS 1, IMAGE_PERMITS = all slots (exclusive device), BIREFNET 1, TTS/STT/EMBED/MM/AUX 1, 3D/animation/training/art = all (main.c:3976-3993); H3 permits 0 (app.nz owns admission). Timeout 30 s default (`ADMISSION_TIMEOUT_S`, prod 150), background timeout/aging bounds (`BACKGROUND_ADMISSION_TIMEOUT_S`, `BACKGROUND_MAX_WAIT_S`).
- Order: tier rank (paid>sub>free>background), FIFO within rank, head-of-line (no overtaking), per-waiter condvars (osched.c:21-60, promote_locked 181). Background needs an otherwise idle device. try_acquire refuses while anyone is queued (osched.c:200).
- LLM: prefix-aware pool of N parallel llama contexts (`LLM_CONTEXTS`, `acquire_slot` backend_llama.c:1507: best common-prefix idle slot, else cond-wait). NO cross-request token-level batching (README roadmap item 1). Speculation off by default (prompt-lookup / MTP). One LLM globally.
- SD: single ctx, `g_sd_lock` serialises generate (backend_sd.c:274-277); `SD_MAX_BATCH` default 1 (<=8 after canary); exact-key latent teleport + encoded-result cache.
- HTTP: 2 reactors + 32 blocking workers => at most 32 in-flight requests regardless of slots; relays pooled per modality (`UPSTREAM_IDLE`).
- Multi-tenancy: shared secret or per-key tier map; trust boundary by loopback + `X-Omniserve-Internal` (stripped on relay). No per-tenant quotas, rate limits, usage accounting or per-key metrics. Access log has model+status+duration, no tenant id (olog.c).
- Overflow: standing remote per lane (`IMAGE|STT|TTS_OVERFLOW_UPSTREAM`, paid only by default), non-blocking try-acquire when configured, frontier policy (frontier/policy.py, src/ofrontier.c; routing.json hot reload) deciding queue-local vs remote; oscale/ocapacity rented-lane controller (default all lanes disabled, scale-to-zero, TTL, spend caps; src/oscale.c, ocapacity.c). Sibling image models behind one admission via `OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS="ra2=http://127.0.0.1:8792,..."` (<=16 names, main.c:4153-4190, relay at 2321-2400) - this is the only existing "multi-model hosting config" and it is multi-process.
- RunPod worker: `runpod.serverless.start({"handler": dispatch})` (handler.py:354) with no `concurrency_modifier` => 1 job per worker, `_dispatch_lock`. cog_pod.py serialises with `_run_lock`.
- Job queue alternative: sqlite `omni-job` (src/jobs.c; claim by priority + fits `required_mib`, one lease per GPU, heartbeat/retry) drained by runtime/local_worker.py through the same `dispatch`.

## 6. API surface

Gateway (route table main.c:3491-3930, README:689-775)
- Probes: `/health` `/healthz` `/liveness_check` (always 200, status ok|degraded, handle_health 669); `/readyz` `/readiness_check` (503 if LLM landed on CPU); `/status` `/backend_status` (huge JSON: upstreams, proxy_pool, tune, overflow, capacity, guard, gpu_sched; main.c:867); `/metrics` (Prometheus text, main.c:685); `/errors` (last 32 5xx ring); `/docs` `/openapi.json` (src/docs.c, static spec).
- OpenAI-compat: `/v1/models`, `/v1/chat/completions` (stream SSE), `/v1/completions`, `/v1/embeddings` (<=256 strings, `dimensions`), `/v1/images/generations|edits|img2img`, `/v1/audio/speech`, `/v1/audio/transcriptions`, plus text-generator.io legacy (`/api/v1/generate*`, autocomplete, summarization, feature-extraction, generate_speech, audio-extraction, image-caption, video-question, multimodal-generate, voice-chat). Image extras: backgrounds(+/style), background-removals, foreground-generations/jobs, segmentations, text-layers, captions, classifications. Async/long: `/v1/video/generations` (H3 via app.nz), `/v1/animations/generations`, `/v1/3d/generations`, `/v1/music/generations` (YuE2), `/v1/training/jobs[/run]`, `/v1/forecasts`, `/loras`.
- Control plane (loopback only): `/v1/gpu/vram`, `/v1/gpu/status` (ledger + sched), `/v1/host/memory`, `/v1/gpu/lease|renew|release`, `/v1/guard/classify`, `/admin/llm/unload|load|swap`.
- Auth: `secret` body/query, `X-API-Key`, `X-Rapid-API-Key`, `Authorization: Bearer`; tier header `X-Omniserve-Tier`.
- Metrics currently exported (main.c:685-860): responses_total{class}, gpu_degraded, vram_free/total_gib, access_log_dropped_total, admission_slots{state}, admission_timeouts_total{tier}, queue_ms_max{paid,free only}, capacity_spend_rate/instances{lane}, vram_headroom_mb/leased/reserved/leases/lease_total{outcome}, llm_spec_*, overflow_total{cause}, frontier_kept_local_total, upstream breaker/failure counters, image_model_relay_total{model}, guard counters. Missing: per-model loaded/resident, llm evictions/reloads (only in /status.gpu_sched), per-tier served counters (osched_stats.served exists, osched.h), per-lane request/latency histograms, GPU compute utilization, busy seconds, per-pid VRAM.
- Python/RunPod entries: `runtime/handler.py` (serverless; input `{workload|kind, execution_profile, ...}`; legacy detection video_url+output_upload_url -> video-matting), `runtime/cog_pod.py` (pod: GET `/healthz` `/health-check` `/health` `/openapi.json` `/`; POST `/predictions` `{"input":..}` -> `{"status","output"}`; port `COG_PORT` 5000), `workloads/qwen_mt.py` (own entry; `task` ra2|edit|probe), `music3/handler.py` and `music3c`, `workers/*.py` FastAPI-style local workers.
- Python workers expose no Prometheus. cog_pod has no metrics route.

## 7. RunPod integration

Images (all `ghcr.io/lee101/...` except music3 on gcr.io/appainz-2)
- Dockerfile.runpod: base pinned `ghcr.io/lee101/h3-cog@sha256:6329...` (CUDA 12.8, py3.12, runpod SDK, ffmpeg w/ libvpx-vp9); builds C `omni-job` in a throwaway stage; adds ultralytics, diffusers git pin 14f482c3, transformers 5.15, bitsandbytes 0.50.1, decord; bakes RVM + YOLO11n + MatAnyone (sha256-pinned); ENTRYPOINT `python -u /opt/omniserve/runtime/handler.py`; env `OMNISERVE_WORKLOAD_MANIFEST`, `OMNISERVE_DEFAULT_WORKLOAD=video-matting`, `OMNISERVE_VRAM_BROKER_REQUIRED=0`, caches on `/runpod-volume/omniserve/*` (HF_HOME, TORCHINDUCTOR/TRITON, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `WAN_ANIMATE_TORCH_COMPILE=1`).
- Dockerfile.runpod-sdcpp: CUDA 12.8.1-devel builder for sd.cpp pinned c678dfe7 (archs 86;89;120, shared lib), runtime on same h3-cog base + libgomp + CUDA 12 cublas/cudart carried beside the .so + NCCL path; no torch; ENTRYPOINT same handler, default workload `qwen-image-2.1`, `RA2_*` env (EasyCache 0.05, 20 steps, VAE tiling 32/0.5, WebP 85, weights `RA2_MODELS_DIR=/runpod-volume/omniserve/qwen-image-2.1` with HF fallback download). Build-time ABI check commented out (needs libcuda); worker runs selftest at start (qwen_image.py:953). Turbo LoRA fetched at first boot if reachable.
- Dockerfile.runpod-sdcpp variants: the qwen-mt image `ghcr.io/lee101/omniserve-native:qwen-mt-20260923b` is 44 GB with weights baked (built by tools/qwen_mt_layer_builder.py + qwen_mt_image.sh, not by a Dockerfile in the tree).
- music3/Dockerfile (python handler) and music3c/Dockerfile (C job loop) both on `hongccc/sglang-omni@sha256:374d0b1c...` (CUDA 13); deploy via scripts/deploy-music3-runpod.sh: derived image optional, otherwise bootstrap string in template env (worker binary + sgl patch base64), runtime cloned onto volume `/runpod-volume/omniserve/music3/sglang-omni-e0c98529-fp16`, model on regional volume (`manifold-music3-models-nc1`, 38 GB).
- YuE: image `ghcr.io/lee101/yue-cog@sha256:33d2154e...` (separate repo ../yue-cog).

Deploy plumbing: scripts/deploy-runpod.sh (hash-tagged image from runtime/workloads/src + Dockerfile sha, docker push, RunPod REST `rest.runpod.io/v1` templates + endpoints PATCH, optional network volume `manifold-shared-video-models`, GraphQL `saveEndpoint` to clear `locations` when no volume; state in `.runpod.env`). Config JSONs:
| file | endpoint name | min/max | idle s | flashboot | exec ms | cuda | GPUs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deploy/runpod.json | omniserve-native | 0/0 (paused; ManifoldGen raises) | 5 | true | 900000 | 13.0 | A40, 3090, 4090, L4, A5000, A6000, 5000 Ada, 6000 Ada, L40, L40S |
| deploy/runpod-ra2.json | omniserve-native-ra2 | 0/2 | 5 | true | 900000 | 12.8 | 4090, L40S, L40, A6000, 3090 |
| deploy/music3-runpod.json | omniserve-minimax-music3-standard | 0/1 | 20 | true | 2400000 | 13.0 | H200, H200 NVL, H100 80GB HBM3/NVL/PCIe, RTX PRO 6000 Blackwell, A100 80GB PCIe/SXM4 |
| deploy/yue-runpod.json | omniserve-yue2-quality (tmozxvnm9fuuud, tpl k2im0m20rr) | 0/1 | 20 | n/a | n/a | n/a | 4090 |
Other live endpoints in docs: d6opzplchn2w3a `omniserve-qwen-mt-overflow` (tpl 8q4mmq15ex; 4090 first then L40S/6000 Ada/L40; workersMax 3, idle 30 s, flashboot on, QUEUE_DELAY 2 s, exec 600 s; adapter env `RUNPOD_RA2_ENDPOINTS`, `RUNPOD_MT_ENDPOINTS`, `RUNPOD_EDIT_ENDPOINTS` in /etc/omniserve-runpod-overflow.env); tlofa06vj7iab7 `omniserve-ra2-overflow` (old, workersMax 0, kept for rollback); akgefm0nzzr4jo Pixal3D (flashboot turned on 09-23). tools/runpod_overflow.py is the localhost adapter (primary then RunPod fallback on connect err/5xx/403/404/429; threaded, dead-primary breaker, strips unknown input keys).

Env vars (worker side): `OMNISERVE_WORKLOAD_MANIFEST`, `_DEFAULT_WORKLOAD`, `_GPU_RESERVE_MIB`(512), `_ENFORCE_VRAM`(1), `_VRAM_BROKER_URL`, `_VRAM_BROKER_REQUIRED`; `RA2_*` (qwen_image.py:55-110), `MT_*` (qwen_mt), `MUSIC3_*` (music3c/main.c), `COG_PORT/HOST/MAX_BODY_BYTES`, `RUNPOD_AI_API_KEY`/`RUNPOD_WEBHOOK_*` (music3c polls the job API itself, ping interval `RUNPOD_PING_INTERVAL`).

Cold start measurements documented (performance/runpod-multitenant-qwen.md unless noted)
- 4090: cold ra2 wall 38.8 s (delay 28.3 + boot 17.1), warm exec 8.8-9.5 s, $0.0028/image; ready 6.8 min after endpoint create. 6000 Ada: 96.3 s cold (84.6/36.0). PRO 6000 Blackwell: 48.0 s. 5090: no capacity in 28 min.
- Baked-weights image on host that has the layers: container start + eager load of both ctxs 17-36 s billed boot; fresh host 5 min "loading container image from cache" (unbilled). Edit first call on 24 GB evicts ra2 (18.9 s load).
- Old worker (weights from HF at first job): 10.8 s delay + 55.8 s exec (about 39 s downloading 11 GB). Old dedicated endpoint: cold ~4.4 min, warm 60-80 s on mixed GPU classes.
- Canary overflow cold: 58-69 s delay then 10 s exec.
- YuE2 4090: first job queue/init 178.2 s + runtime 64.0 s (caller wall 249.8 s) vs warm 0.016 s + 24.9 s; HF cache on `/runpod-volume/huggingface-cache/hub` mount is a cold-start optimisation.
- idleTimeout economics (30 s window = $0.009/burst vs cold 17-28 s billed boot $0.005-0.009 + 30-40 s latency). FlashBoot: set true everywhere, no measured effect documented separately.
- Music3: no cold numbers in repo; `MUSIC3_WARM_START=1`, prefetch threads 8 (`MUSIC3_PREFETCH_INCLUDE=qwen_7B,flowmatching_vae.pth,dav.pth`), startup timeout 1800 s, torchinductor/triton/flashinfer caches on volume; response includes model_download/server_start/generation seconds + `server_ready_before_job` (music3.h Music3Metrics). Exact result cache 16 entries/4 GiB on volume.
- Explicit repo decision (runpod-multitenant-qwen.md "Not a general multi-workload worker"): do not merge YuE/Pixal3D into the Qwen image (44 GB image, 6-19 s evictions on 24-48 GB cards, 5-min 3D jobs ahead of 10 s image jobs at concurrency 1).

## 8. Gaps for packing many workloads into one server + small low-risk changes

Observed gaps
1. In-process kinds are singletons (LLM, embed, judge, SD). Cannot host two LLMs or two diffusion models in one process; multi-model = more processes (:8791, :8792) glued by IMAGE_MODEL_UPSTREAMS + broker.
2. No time-based idle unload anywhere in C; LLM eviction only under broker pressure; SD and embedding never unload; Python handler never unloads except for space/OOM.
3. Reload asymmetry: only LLM has lazy reload. After unloading SD/embed there is no request-path hook to bring them back.
4. Admin `unload` leaves LLM in a non-reloadable state (main.c:3242), while scheduler eviction is reloadable.
5. Residency state not exported to /metrics; no GPU util; no per-lane stats. Cannot answer "what fraction of time is each model hot / GPU busy".
6. Admission is by permits not by VRAM; VRAM leases cover only the image lane and judge/LLM reload (main.c:2565 region, 1725) - LLM requests take no lease while resident.
7. Python runtime: one job per worker, no idle timer, no metrics, `required_mib` is a static manifest guess (no learned footprint), no broker lease TTL renewal for long jobs (ttl 10800 s fixed).
8. No hot-reloadable model manifest for the C gateway (everything is env at boot; `IMAGE_MODEL_UPSTREAMS` capped at 16, read once).
9. HTTP worker pool 32 blocking threads caps concurrent long streams independent of permits.
10. Metrics buffer hazard: `char body[8192]` (main.c:697) with `len += snprintf(...)` and `ohttp_respond(req,200,...,body,len)` at the end can pass len > sizeof body once enough lanes/models are added (guards are `len < sizeof body` before each append but the append itself can overshoot). Clamp before respond.

Small, low-risk changes (ordered by risk, each independently shippable, default off)

A. Timer idle unload for the embedded LLM (reuse everything already there)
- Where: main.c, new thread `llm_idle_main` next to `judge_admit_main` (1773); start after `gs_llm_touch()` at main.c:4051 (the judge thread is only created in the judge_auto branch at 4304, so create a separate pthread unconditionally when the env is set).
- Logic: every ~10 s: if `OMNISERVE_NATIVE_LLM_IDLE_UNLOAD_S>0 && ollm_ready() && idle >= N && osched_active(app->sched)==0 && trylock(g_llm_swap_lock)`: run the same body as gpu_pressure lines 1696-1713. Factor that block into `static int gs_llm_evict_locked(const char *why)` and call it from both. Reload is already lazy via `llm_ensure_loaded` (1725).
- Touch points already there: `gs_llm_touch` at 1658 (called from llm_ensure_loaded on every LLM request, 1726/1728), counters `g_gs.llm_evictions`, JSON at 1806.
- Also fix `handle_llm_unload` (3242) to set `g_gs.llm_evicted=true`/capture config under `g_gs_lock` so admin unload is reloadable (guard behind env to keep old behaviour).
- Tests: extend tests/test_core.c style or a pytest like tests/test_internal_trust.py (stub gateway, no GGUF) - hard without a model; reuse the 0.6B model path used in performance docs.

B. SD ctx idle unload / lazy reload (needs a bit more code, still local to backend_sd.c)
- backend_sd.c:252-261 add `dlsym(lib,"free_sd_ctx")` (sd.cpp C API; used by workloads/qwen_image.py:613 via `lib.lib.free_sd_ctx`); add `osd_shutdown()` taking `g_sd_lock` (274), freeing `g_sd`, the latent/result cache (`g_latent_cache`), and keeping `g_sd_name`; add `osd_reload()` = osd_init with saved model path (osd_init is idempotent via `if (g_sd) return true` at 736; stash `model_path` in a static). Declare both in include/obackend.h next to osd_init (obackend.h:243-248).
- main.c: in handle_images before `osd_generate` (2621) call `osd_ensure_loaded(app, tier)` mirroring `llm_ensure_loaded`: lease `g_gs.sd_lease_mb` + weight MB via `ovram_lease_wait` (main.c:1832 `image_gpu_lease`), then reload. Note `osd_ready()` (backend_sd.c:794) is used as the "lane exists" test at 3598 and in overflow selection, so a lazily-unloaded ctx must keep reporting "configured" separately from "resident" or the overflow path will claim it has no model (README:597).
- Gate with `OMNISERVE_NATIVE_SD_IDLE_UNLOAD_S` (default 0).

C. Utilization metrics (pure additions, no behaviour change)
- GPU util: osched.c:377-381 add `nvmlDeviceGetUtilizationRates` dlsym (struct {unsigned gpu, memory}) beside nvml_mem_info, export `bool ogpu_utilization(int *gpu_pct, int *mem_pct)` in osched.h (next to ogpu_memory_gib, osched.h:~60). Emit `omniserve_gpu_util_percent{kind="sm|mem"}` in handle_metrics (main.c after the vram_free block ~line 722).
- Gateway busy accounting: osched.c struct (line 50) add `double busy_s, slot_s; struct timespec last_change;`; add `account_locked(s)` called before every change of `used_slots` at the four sites (osched.c:190 promote_locked, 211 try_acquire, 224 acquire fast path, 294 release) - adds `used_slots>0` time to busy_s and `used_slots*dt` to slot_s. Expose in `osched_stats` (osched.h snapshot) and as `omniserve_admission_busy_seconds_total`, `omniserve_admission_slot_seconds_total`. rate() over these = utilization of the permit pool.
- Residency/eviction: in handle_metrics add `omniserve_model_loaded{kind="llm|embedding|diffusion|judge"}` (ollm_ready/oembed_ready/osd_ready/ojudge_ready, all already used at main.c:937-938, 1062-1064), `omniserve_llm_idle_seconds`, `omniserve_llm_evictions_total`, `omniserve_llm_reloads_total`, `omniserve_llm_reload_failures_total`, `omniserve_llm_reload_ms_last` (all in g_gs, struct at main.c:1625; currently only in `/status.gpu_sched`, gpu_sched_json 1806), per-tier `omniserve_admission_served_total{tier}` (osched_stats.served already filled, osched.c:~310), complete `queue_ms_max` for sub/background (main.c:728-729 emit only paid/free).
- Per-process VRAM: `ogpu_processes` (osched.c:408) already exists; ledger JSON at /v1/gpu/status has it; add `omniserve_gpu_process_mb{pid}` only if cardinality is acceptable.
- Safety: bump `char body[8192]` (main.c:697) to 16384 and clamp `len` before `ohttp_respond` (end of handle_metrics, main.c:~855).

D. Multi-model hosting config (declarative, small)
- Immediate zero-C option: generalize what exists. `OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS` (main.c:4153-4190; routing 2321-2400, state `image_model_upstream` at 33-39, 101) already maps `model` -> sibling gateway process sharing admission. Add the same table for LLM (`OMNISERVE_NATIVE_LLM_MODEL_UPSTREAMS`) using the chat handlers (1915, 2144, 2987) and for sibling workers, reusing `relay_image_model_to`'s shape. Each sibling remains one process that owns one model; this is the cheapest way to host several models behind one admission and metrics surface.
- In-process multi-LLM would require turning backend_llama.c globals (g_model, g_slots, g_vocab, g_model_name, locks) into a struct handle: large; avoid.
- For Python workloads: add `idle_unload_s` and `prewarm` keys to workloads.json entries and a daemon thread in runtime/handler.py that, under `_dispatch_lock`, calls `_release_name(name)` for entries with `time.monotonic() - last_used > idle_unload_s` (use `_loaded_modules[name]["last_used"]`, handler.py:30-34). Add a `GET /metrics`-style text on cog_pod.py (`do_GET`, line ~80) from `_loaded_modules`: resident names, required_mib, free_mib via `_free_mib()` (83), last_used age, oom_retries counter (increment in the `_is_cuda_oom` branch, handler.py:309). For serverless set `runpod.serverless.start({"handler": dispatch, "concurrency_modifier": ...})` only if workloads are made thread-safe (they are not: `_dispatch_lock`).
- qwen_mt.py: add the same idle sweep over `_ctxs[name]["used"]` (qwen_mt.py:187) calling `free_ctx` (147); trivial and measured-safe since reload costs 6-19 s from page cache.

E. Learned footprints for the Python manifest (small)
- In dispatch (handler.py:277-340) record `free_mib_before - _free_mib()` after first load into `result["omniserve"]["measured_mib"]` (the value `free_mib_before` is already captured at ~300) and persist in `/runpod-volume/omniserve/footprints.json`; use max(required_mib, learned) in `_make_room`/`_profile_for`. Fits the existing design comment that required_mib is only an admission estimate.

Recommended order: C (metrics, observe first), A (LLM idle unload, reload path already battle-tested by the 09-24 canary: reload 1127 ms for 0.6B, 2.7 s for Gemma q8), D-python idle sweep, then B (SD unload) after confirming free_sd_ctx releases VRAM in the prod sd.cpp fork (the prod fork also carries `sd_ctx_release_vram` and `SD_RESIDENCY_EVICT_MRU`, mentioned in qualitybench/README.md but not referenced anywhere in this repo's source - check sdcpp-qwen-prefixkv before relying on it).

## 9. Misc findings worth remembering

- README says it is "every OmniServe-owned gateway runtime source file is C"; true for src/, but all model logic for non-llama/sd workloads is Python behind HTTP.
- Prod 5090 is shared by ~12 tenants at ~30/32 GB; most design choices (permits-as-exclusive for diffusion, keep-free floor, streamed Z-Image at 2 GiB cap, LLM on CPU NGL=1 in service file) follow from that, not from an ideal single-tenant server.
- Exact-repeat caches are the cheapest "throughput": Z-Image result cache 1-2 ms, Music3 exact WAV cache, YuE 2 GiB/24 h disk cache + in-flight dedupe.
- Cost/frontier layers (oscale, ocapacity, ofrontier, frontier/*.py, runpod_overflow.py) are about WHERE to run; none addresses packing density on one GPU. Packing today = broker leases + eviction hooks + LRU contexts in qwen_mt.
- Do not run `ctest` on the prod box without `CUDA_VISIBLE_DEVICES= OMNISERVE_ACCESS_LOG=0` (gpu-scheduler doc: gateways take ~3 GB VRAM and write the prod access log).
- plan/ dir was empty before this file.
