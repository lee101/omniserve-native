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
