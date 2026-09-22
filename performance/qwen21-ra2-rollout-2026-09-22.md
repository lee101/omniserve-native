# Qwen Image 2.1 as "ra2": local results and prod rollout plan (2026-09-22)

Bench data and scripts: `/vfast/data/code/qwen-image-2.1-bench/` (README there has the full ablation).

## What was measured (RTX 3090 Ti, 1024x1024, GPU shared with the zimage instance)

| engine | quant | steps | accel | wall / image | notes |
|---|---|---:|---|---:|---|
| ComfyUI python (triton w4a8) | Rebels w4a8 convrot 4.2 GB | 25 | none | 22.8 s | reference quality |
| ComfyUI python | w4a8 | 25 | Taylor w4 p2 t3 | ~14.5 s | SSIM 0.992 vs dense |
| ComfyUI python | w4a8 | 25 | EasyCache 0.2 | ~11 s | SSIM 0.958 |
| ComfyUI python | w4a8 | 25 | EasyCache + Taylor | ~10 s | SSIM 0.939 |
| ComfyUI python (ComfyUI-GGUF) | Q4_K_M GGUF 4.6 GB | 25 | none | 31 s | Q8_0 GGUF does not load in ComfyUI (shape bug noted in its README) |
| sd.cpp master `sd-cli` | Q4_K_M | 25 | none | 27.5 s sampling | Q8_0 same speed (compute-bound after dequant) |
| sd.cpp master `sd-cli` | Q4_K_M | 25 | EasyCache | 13-14 s sampling | PSNR 33 / SSIM 0.97 vs dense |
| sd.cpp master `sd-cli` | Q4_K_M | 25 | Spectrum | 15 s sampling | PSNR 32-35 / SSIM 0.97-0.98 |
| sd.cpp master `sd-cli` | Q4_K_M | 25 | TaylorSeer | 18-19 s sampling | PSNR 29-33 |
| omniserve-native `build-qwen` | Q4_K_M, TE on CPU | 25 | EasyCache 0.2 | 16.1 s | end to end through the gateway |
| omniserve-native `build-qwen` | Q4_K_M, TE on CPU | 20 | EasyCache 0.2 | 14.3 s | proposed prod default |
| omniserve-native `build-qwen` | reference edit (`/v1/images/edits`, `image_base64`) | 20 | EasyCache 0.2 | 27.6 s | identity preserved, style transfer and object edits correct |
| omniserve-native (zimage, today) | Z-Image Turbo Q4_K | 9 | none | 10.1 s | current ra1 backend |

The "uncensored" GGUF repo (abenzerps) is a GGUF of the plain base Qwen-Image-2.1 weights (its README says so); it has no separate finetune. `gguf_to_w4a8.py` re-quantises any Qwen 2.1 GGUF into ComfyUI's native w4a8 convrot format (validated: matches the Rebels file at PSNR 26 / SSIM 0.93, same speed and size), so a real finetune can get the same treatment later.

Prod is an RTX 5090, so expect roughly 2-2.5x faster than the numbers above (zimage there is ~4 s).

## Code changes (all local, uncommitted)

- `omniserve-native`: `OMNISERVE_NATIVE_SD_CACHE_MODE` (easycache | taylorseer | spectrum | cache-dit | dbcache | ucache) with per-mode env knobs in `src/backend_sd.c`; `denoiser_cache.requested` now reports the mode; CMake detects whether the sd.cpp header carries the prod latent-replay API / `stream_layers` and shims them out otherwise (`OMNISERVE_SD_LATENT_API`, `OMNISERVE_SD_STREAM_LAYERS`), so the same source builds against upstream master. `deploy/qwen-ra2-prod.sh` stands up the second instance.
- `stable-diffusion.cpp-master` worktree (upstream `origin/master` c678dfe): only a CLI convenience patch (`--cache-option skip=,deriv=` for taylorseer). Upstream master already has Qwen 2.1 T2I + edit, EasyCache, TaylorSeer, Spectrum, cache-dit.
- `netwrck`: `/api/ra2`, `/api/ra2-art-generator`, `/api/ra2-image-editor`, tool pages `/tools/ra2-art-generator` and `/tools/ra2-image-editor`, api docs, examples, defaults switched to ra2 (`RA2_DEFAULT=0` reverts) in ai-art-generator, game, chat overlay and ebank JS. Env: `RA2_LOCAL_IMAGE_URL`, `RA2_LOCAL_IMAGE_SECRET`. See `netwrck/RA2_CHANGES.md`. Go + jest tests pass.
- `cutedsl-site`: services `ra2` and `ra2-edit`, default service ra2 (`RA2_DEFAULT`), frontend selector, docs. Env: `RA2_BACKEND_URL`, `RA2_BACKEND_SECRET`. See `cutedsl-site/RA2_CHANGES.md`. Go tests + tsc pass.

## Prod rollout (not started; needs decisions)

Architecture: a second omniserve-native process, `omniserve-native-qwen.service` on `127.0.0.1:8792`, built from `build-qwen` against an upstream sd.cpp master worktree. The zimage instance (8791) and its patched sd.cpp are untouched, so ra1 keeps exact-prompt teleport. Qwen gets EasyCache 0.2 (2x) and the gateway result cache; exact latent replay for Qwen would need the latent-replay patch ported onto upstream (it does not apply, 500 lines, deferred).

Blockers found on prod:

1. `/nvme0n1-disk` is 100% full (11 GB free). The Qwen models need ~11 GB (DiT 4.6, TE 4.7, mmproj 1.1, VAE 0.6) plus build output. The deploy script refuses the download step below 14 GB free. Something must be deleted or moved first (`/nvme0n1-disk/models/omniserve-native` is 138 GB).
2. VRAM: 32 GB with 24.7 GB used. A stray manually started `build-dev/omniserve-native` (pid 1429811, up 1d20h) holds 5.4 GB and `netwrckprod137` holds 5.3 GB. The Qwen instance with the text encoder on CPU needs about 6-7 GB (DiT 4.6 + VAE + compute at 1024²). Stopping the dev instance frees enough.
3. `nvcc` is at `/usr/local/cuda-12.9/bin` and `hf` at `/nvme0n1-disk/code/omniserve-native/.venv/bin`; the script points at both.

Steps once unblocked (script `deploy/qwen-ra2-prod.sh`, run on prod as administrator):

```
bash deploy/qwen-ra2-prod.sh build     # sd.cpp master worktree + omniserve build-qwen (sm_120)
bash deploy/qwen-ra2-prod.sh models    # HF downloads into /nvme0n1-disk/models/omniserve-native/qwen-image-2.1
bash deploy/qwen-ra2-prod.sh install   # systemd unit + /etc/omniserve-qwen.env (reuses the zimage secret)
bash deploy/qwen-ra2-prod.sh start && bash deploy/qwen-ra2-prod.sh smoke
```

Then set `RA2_LOCAL_IMAGE_URL=http://127.0.0.1:8792/v1/images/generations` + secret in the netwrck prod env, `RA2_BACKEND_URL=http://127.0.0.1:8792` + secret for cutedsl, deploy both, and flip back with `RA2_DEFAULT=0` if needed. The omniserve, netwrck and cutedsl-site changes still need commits and pushes before the prod checkouts can pull them.

Follow-ups worth doing: port the latent-replay patch to upstream so Qwen repeats are 3x; tune EasyCache threshold (0.15-0.3) on the 5090; try Spectrum as the default if its quality edge holds across more prompts; webp output from the Qwen instance currently comes back as png (see bench README).

## Update 2026-09-22 13:20 NZST: prod instance live

- `omniserve-native-qwen.service` (port 8792, branch `qwen-ra2`, worktree `/nvme0n1-disk/code/omniserve-native-qwen`, sd.cpp master at `/nvme0n1-disk/code/stable-diffusion.cpp-master`) is running on prod with every weight streamed from RAM (`OMNISERVE_NATIVE_SD_PARAMS_BACKEND=*=cpu`) because only ~2-8 GB VRAM is free next to the other tenants. Measured through the gateway: 1024x1024, 20 steps, EasyCache 0.2 = 10-11 s wall, WebP out (RGBA WebP fix `5f848c0`). With `te=cpu` (DiT resident) sampling is 7.0 s but the VAE decode then fails for lack of VRAM; that config needs ~8 GB free at load, i.e. the stray `build-dev/omniserve-native` (pid 1429811, 5.4 GB, manual, 2 days old) stopped.
- Secret: `/etc/omniserve-qwen.env` (random, generated at install). The zimage instance on 8791 runs without a secret.
- Disk was freed by the user (44 GB free), models live in `/nvme0n1-disk/models/omniserve-native/qwen-image-2.1/`.

Remaining wiring (user-facing, not done): netwrck prod needs `RA2_LOCAL_IMAGE_URL=http://127.0.0.1:8792/v1/images/generations` and `RA2_LOCAL_IMAGE_SECRET=<from /etc/omniserve-qwen.env>` in its environment plus a rebuild/restart from branch `ra2-art-generator`; cutedsl-site prod needs `RA2_BACKEND_URL=http://127.0.0.1:8792` and `RA2_BACKEND_SECRET` plus a deploy from its `ra2-art-generator` branch. Both flip the site default to ra2 (`RA2_DEFAULT=0` reverts).

Local optimisation loop result (ComfyUI/python, 3090 Ti): floor SSIM>=0.985/PSNR>=37 is met fastest by adaptive sigma-aware Taylor (15 real calls, 10.75 s, PSNR 38.4, SSIM 0.988); details in `qwen-image-2.1-bench/README.md`. The same error-gated skipping is being ported into sd.cpp's TaylorSeer for prod (`stable-diffusion.cpp-master` branch `qwen-adaptive-taylor`, bench in `qwen-image-2.1-bench/comparison/sdcpp_optim/`).

netwrck prod specifics: the Go server runs under supervisord (`/etc/supervisor/conf.d/netwrckprod135.conf`, program `netwrckprod135`, `environment=...RA1_LOCAL_IMAGE_URL="http://127.0.0.1:8791/v1/images/generations",RA1_LOCAL_IMAGE_SECRET=...`; the live pid currently runs `bin/netwrckprod137` on PORT 8124). To wire ra2: append `,RA2_LOCAL_IMAGE_URL="http://127.0.0.1:8792/v1/images/generations",RA2_LOCAL_IMAGE_SECRET="<value from /etc/omniserve-qwen.env>"` to that `environment=` line, build the branch (`cd /nvme0n1-disk/code/netwrck && git fetch && git checkout ra2-art-generator && cd search_server_go && go build -o bin/netwrckprod138 .`), point `command=` at the new binary, `sudo supervisorctl update && sudo supervisorctl restart netwrckprod135`, then check `/tools/ra2-art-generator` and `/api/ra2` with an API key. cutedsl-site: `cutedsl.service` (systemd) with `RA2_BACKEND_URL`/`RA2_BACKEND_SECRET` added to its environment and a rebuild from `ra2-art-generator`.

## Update 14:50 NZST: netwrck live on ra2

- Prod Qwen unit now `te=cpu` (DiT resident) + EasyCache 0.05: ~14 s per 1024² 20-step image through the gateway. The manual `build-dev/omniserve-native` LLM instance on 8799 respawns from an interactive ssh session (someone is running it by hand), so it was left alone.
- netwrck prod: branch merged on the prod checkout (prod WIP snapshot-committed first, ra2 commits shipped as a git bundle because prod has no GitHub key), JS bundles rebuilt, `bin/netwrckprod138` built, supervisor program `netwrckprod138` (PORT 8124, `RA2_LOCAL_IMAGE_URL`/`RA2_LOCAL_IMAGE_SECRET` added) replaced `netwrckprod137` (stopped, kept for rollback: `sudo supervisorctl stop netwrckprod138 && sudo supervisorctl start netwrckprod137`). Verified: `POST https://netwrck.com/api/ra2-art-generator` returns an uploaded image in ~20 s (4 credits), `/tools/ra2-art-generator`, `/tools/ra2-image-editor`, `/api-docs/ra2` serve.
- cutedsl-site prod runs a diverged local branch (`agent/review-fixes-inference`, 113 dirty files); the ra2 patch does not apply, so the service is being ported onto the prod file versions separately (`qwen-image-2.1-bench/subagent/cutedsl_prod/`).
- `localqwen.sh` now drives omniserve-native (local build-qwen instance on 8792, `--remote` = ssh tunnel to the prod ra2 instance, `--python` = experimental sampler, `--edit img` = reference edit).
