# Image offload tiers (2026-09-22)

Main prod GPU (5090, 32 GB) is shared by ~12 tenants and sits at ~30 GB used, so
both image gateways now overflow instead of failing:

```
prod :8791 Z-Image  --overflow-->  http://127.0.0.1:18793 (nginx bridge) --> https://zimage-daisy.netwrck.com --> daisy :8791 (Z-Image, RTX 3090)
prod :8792 RA2      --overflow-->  http://127.0.0.1:18794 (nginx bridge) --> https://ra2-daisy.netwrck.com    --> daisy :8792 (RA2, RTX 3090, te=cpu, ~16 s/1024)
```

* omniserve's proxy only speaks plain http, so `/etc/nginx/conf.d/omniserve-daisy-upstreams.conf`
  bridges to the Cloudflare tunnel hostnames (daisy tunnel `images3`, config
  `~/.cloudflared/config-images3.yml`, connector run by hand: `cloudflared --config ... tunnel run`).
* `OMNISERVE_NATIVE_IMAGE_OVERFLOW_PATH=passthrough` (d21b679) keeps generations and
  edits on their own routes when the upstream is another gateway.
  `OMNISERVE_NATIVE_OVERFLOW_TIERS=free,paid,sub,background` so every caller may spill
  (default is paid only). Prod :8791 config: drop-in `overflow-daisy.conf` (also swaps the
  binary to `omniserve-native-ra2overflow/build-full-ra2`); prod :8792 config: `/etc/omniserve-qwen.env`.
* Prod :8792 runs `OMNISERVE_NATIVE_SD_PARAMS_BACKEND=*=cpu` + VAE tiling (drop-in
  `ra2-vae-tiling.conf`): with ~2 GB VRAM free, resident weights failed 1024 edits and
  1536+ renders; streamed weights give 8-11 s per 1024 image and 37 s for 2048.
* daisy: `omniserve-native-qwen.service` (unit + `/etc/omniserve/daisy-qwen.env`), binary
  `build-qwen` against `stable-diffusion.cpp-master`, models in `/media/lee/pcd/models/qwen-image-2.1`.
  Setup scripts used: `/media/lee/pcd/ra2-setup.sh`, `/media/lee/pcd/ra2-models.sh`.
* Verified: 3 concurrent RA2 and 2 concurrent Z-Image requests on prod -> `saturated`
  counters 2 and 3, daisy journals show the relayed renders.
* Not done: lee-top (omniserve.how.nz, RTX 3080 16 GB) runs an LLM-only gateway with no
  image weights; lee-low refused the given password. app.nz RA2 overflow cog is in
  `error` (context deadline) and the image-upscaler cog is pinned to an offline 3090 host.

## Update 2026-09-23

* RunPod was dead: upstream deleted `abenzerps/.../qwen-image-2.1-Q4_K_M.gguf`, so every cold
  worker 404'd. The byte-identical file (sha256 833439e9...) is now mirrored at
  `netwrck/ra2` as `ra2-dit-q4_k_m.gguf`; `workloads/qwen_image.py` and
  `deploy/qwen-ra2-prod.sh` default to it and `ghcr.io/lee101/omniserve-native:ra2` bakes the env.
* app.nz (prod binary from 2026-07-15) promotes bursty cogs to a dedicated pod and never
  hedges onto serverless, so it is no longer in the path. New dedicated endpoint
  `omniserve-ra2-overflow` (`tlofa06vj7iab7`, template `7p2yctvzvk`, 4090/3090/A5000/L4,
  scale to zero, 30 s idle): cold ~4.4 min, warm ~60-80 s per 1024 image.
* `tools/runpod_overflow.py` (systemd `omniserve-runpod-overflow`): localhost adapter that
  tries `PRIMARY_UPSTREAM` first and falls back to RunPod on connect errors / 5xx / 403 / 404 / 429.
  - prod: `127.0.0.1:18797`, env `/etc/omniserve-runpod-overflow.env`, primary = daisy RA2
    via the 18794 bridge. Prod RA2 (:8792) overflow points here. Verified with daisy down:
    local 11 s, two overflow requests completed on RunPod in 80 s.
  - daisy: `127.0.0.1:18795`, env `/etc/omniserve/runpod-overflow.env`, no primary; daisy RA2
    overflow (paid tier) points here.
* daisy's 3090 hit Xid 79 (fell off the bus) at 2026-09-23 10:21 NZST; a PCI rescan hung and
  the reboot hung in shutdown, so it needs a physical power cycle. After it is back:
  `cd /media/lee/pcd/code/omniserve-native && git fetch -q && git checkout -q --detach origin/main`
  then `sudo systemctl restart omniserve-runpod-overflow` (adapter UA fix), and on prod restore the
  Z-Image overflow env lines from `/etc/omniserve-overflow-daisy.conf.disabled` into the
  `overflow-daisy.conf` drop-in and restart `omniserve-native`. Consider `nvidia-smi -pl 300`
  on daisy (both models now share one 3090; Xid 79 is often a power transient).
* lee-top: `omniserve-native-ra2.service` (:8792, `/mnt/crucial/code/omniserve-native-ra2`,
  sd.cpp master built for sm_86 with CUDA 12.0 / g++-12, weights in
  `/mnt/crucial/models/qwen-image-2.1`). Works, but the laptop GPU is firmware-capped at
  10 W / P8 (210 MHz) even on AC, so a 1024 image takes ~11 min. Not in the overflow chain.
