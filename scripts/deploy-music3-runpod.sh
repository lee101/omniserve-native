#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_file="${MUSIC3_RUNPOD_STATE:-$root/.runpod-music3.env}"
config="$root/deploy/music3-runpod.json"
runpod_key="${RUNPOD_API_KEY:-${H3_RUNPOD_API_KEY:-}}"
base_image="${MUSIC3_BASE_IMAGE:-hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101}"
image_repository="${MUSIC3_IMAGE_REPOSITORY:-gcr.io/appainz-2/omniserve-music3}"
registry_auth_id="${MUSIC3_RUNPOD_REGISTRY_AUTH_ID:-}"
network_volume_name="${MUSIC3_RUNPOD_NETWORK_VOLUME_NAME:-manifold-music3-models-nc1}"

if [[ -f "$state_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$state_file"
  set +a
fi
if [[ -z "$runpod_key" ]]; then
  echo "ERROR: RUNPOD_API_KEY is required" >&2
  exit 1
fi
for command_name in base64 curl git jq sha256sum make cc; do
  command -v "$command_name" >/dev/null || { echo "ERROR: missing $command_name" >&2; exit 1; }
done

tag="$(sha256sum "$root/music3c/Dockerfile" "$root/music3c/main.c" "$root/music3c/music3.c" "$root/music3/sglang-music3-a100.patch" "$config" | sha256sum | cut -c1-16)"
image="$base_image"
if [[ "${MUSIC3_BUILD_DERIVED_IMAGE:-0}" == "1" ]]; then
  command -v docker >/dev/null || { echo "ERROR: missing docker" >&2; exit 1; }
  image="${image_repository}:${tag}"
  echo "Building optional derived image $image"
  docker buildx build --pull --push --platform linux/amd64 -t "$image" -f "$root/music3c/Dockerfile" "$root"
fi

# Avoid copying the upstream image's 11.42GB layer between registries. RunPod
# pulls it from its public source; the C adapter is compiled here and the
# 38GB MiniMax-Music3 checkpoint stays on the regional network volume.
# The worker ships base64 in a container env var, and that counts against the
# exec argument limit for every child the worker spawns, so it has to stay
# small: compile it inside the serving image against that image's own libc
# rather than statically or with whatever toolchain the deploy host has.
mkdir -p "$root/build"
if command -v docker >/dev/null; then
  docker run --rm --user "$(id -u):$(id -g)" -e SGLANG_OMNI_AUTO_CLONE=0 -v "$root":/src -w /src "$base_image" \
    cc -std=c17 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L \
    music3c/main.c music3c/music3.c -lm -o build/music3c-deploy
  docker run --rm -e SGLANG_OMNI_AUTO_CLONE=0 -v "$root/build":/b:ro "$base_image" \
    /b/music3c-deploy --selftest \
    || { echo "ERROR: worker binary is not runnable in $base_image" >&2; exit 1; }
else
  make -C "$root" DEPLOY_CC=/usr/bin/gcc build/music3c-deploy
fi
# jq reads the encoded payloads from files rather than argv, which the
# base64-encoded worker would overflow.
worker_b64_file="$(mktemp)"
sgl_patch_b64_file="$(mktemp)"
trap 'rm -f "$worker_b64_file" "$sgl_patch_b64_file"' EXIT
base64 -w0 "$root/build/music3c-deploy" > "$worker_b64_file"
base64 -w0 "$root/music3/sglang-music3-a100.patch" > "$sgl_patch_b64_file"
# FP8 on the Qwen3 backbone renders ~1.3x faster per second of audio on H200
# and halves what the AR stage reads per frame. Export MUSIC3_SERVE_EXTRA_ARGS
# (empty is honoured) to serve bf16 instead.
default_serve_args="--quantization fp8"

# Boot is on the request path for a scale-to-zero endpoint, so it stays free of
# per-boot installs: the runtime lives on the volume and is imported through
# PYTHONPATH, and every step logs to the volume so a failed boot is diagnosable.
bootstrap='set -Eeuo pipefail
runtime=/runpod-volume/omniserve/music3/sglang-omni-e0c98529
mkdir -p /runpod-volume/omniserve/music3 /opt/omniserve-music3
exec >>/runpod-volume/omniserve/music3/bootstrap.log 2>&1
echo "--- music3 boot $(date -u +%FT%TZ) ---"
if [[ ! -f "$runtime/pyproject.toml" ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang-omni.git "$runtime"
  git -C "$runtime" checkout e0c98529e5730f60e19251025877387b9476c8d4
fi
printf "%s" "$MUSIC3_SGL_PATCH_B64" | base64 -d > /tmp/music3-a100.patch
if git -C "$runtime" apply --check /tmp/music3-a100.patch 2>/dev/null; then
  git -C "$runtime" apply /tmp/music3-a100.patch
fi
if ! python3 -c "import sglang_omni" 2>/dev/null; then
  python3 -m pip install --no-deps --no-build-isolation -e "$runtime"
fi
printf "%s" "$MUSIC3_WORKER_B64" | base64 -d > /opt/omniserve-music3/music3c
chmod +x /opt/omniserve-music3/music3c
if ! /opt/omniserve-music3/music3c --selftest; then
  echo "worker binary is not runnable in this image"
  exit 1
fi
exec /opt/omniserve-music3/music3c'

api="https://rest.runpod.io/v1"
auth=(-H "Authorization: Bearer $runpod_key" -H "Content-Type: application/json")
volumes="$(curl --fail-with-body -sS "$api/networkvolumes" "${auth[@]}")"
volume_id="${MUSIC3_RUNPOD_NETWORK_VOLUME_ID:-$(jq -r --arg name "$network_volume_name" '[.[]|select(.name==$name)][0].id // empty' <<<"$volumes")}"
datacenter_id="${MUSIC3_RUNPOD_DATACENTER_ID:-$(jq -r --arg name "$network_volume_name" '[.[]|select(.name==$name)][0].dataCenterId // empty' <<<"$volumes")}"
if [[ -z "$volume_id" || -z "$datacenter_id" ]]; then
  echo "ERROR: Music3 requires a regional network volume for its 38GB checkpoint" >&2
  exit 1
fi

template_payload="$(jq -n --arg image "$image" --arg registry_auth "$registry_auth_id" --arg bootstrap "$bootstrap" --rawfile worker "$worker_b64_file" --rawfile sgl_patch "$sgl_patch_b64_file" --arg extra_serve_args "${MUSIC3_SERVE_EXTRA_ARGS-$default_serve_args}" '{
  imageName:$image, name:"omniserve-minimax-music3", containerDiskInGb:60,
  dockerEntrypoint:[], dockerStartCmd:["bash","-lc",$bootstrap], env:{
    MUSIC3_WORKER_B64:($worker|rtrimstr("\n")),
    MUSIC3_SGL_PATCH_B64:($sgl_patch|rtrimstr("\n")),
    MUSIC3_MODEL_ID:"MiniMaxAI/MiniMax-Music3",
    MUSIC3_MODEL_DIR:"/runpod-volume/models/minimax-music3",
    HF_HOME:"/runpod-volume/huggingface",
    TORCHINDUCTOR_CACHE_DIR:"/runpod-volume/omniserve/music3/torchinductor",
    FLASHINFER_WORKSPACE_BASE:"/runpod-volume/omniserve/music3/flashinfer",
    MUSIC3_PORT:"8000", MUSIC3_MAX_RUNNING_REQUESTS:"1", MUSIC3_ACOUSTIC_DTYPE:"bfloat16",
    MUSIC3_STARTUP_TIMEOUT_SECONDS:"1800", MUSIC3_REQUEST_TIMEOUT_SECONDS:"1800",
    PYTHONPATH:"/runpod-volume/omniserve/music3/sglang-omni-e0c98529",
    MUSIC3_SERVE_PYTHON_MODULE:"1", MUSIC3_WARM_START:"1", MUSIC3_PREFETCH_THREADS:"8",
    MUSIC3_SERVE_EXTRA_ARGS:($extra_serve_args)
  }, category:"NVIDIA",
  isPublic:false, isServerless:true,
  ports:[], readme:"OmniServe accelerated MiniMax-Music3 worker"
} | if $registry_auth != "" then .containerRegistryAuthId=$registry_auth else . end')"
template_id="${MUSIC3_RUNPOD_TEMPLATE_ID:-}"
if [[ -n "$template_id" ]]; then
  template_response="$(curl --fail-with-body -sS -X POST "$api/templates/$template_id/update" "${auth[@]}" --data "$(jq 'del(.category, .isServerless)' <<<"$template_payload")")"
else
  template_response="$(curl --fail-with-body -sS -X POST "$api/templates" "${auth[@]}" --data "$template_payload")"
  template_id="$(jq -r '.id // empty' <<<"$template_response")"
fi
[[ -n "$template_id" ]] || { echo "ERROR: RunPod returned no template ID" >&2; exit 1; }

endpoint_payload="$(jq --arg template "$template_id" --arg volume "$volume_id" --arg dc "$datacenter_id" '{
  templateId:$template, computeType:"GPU", gpuCount:1, name:.name,
  workersMin:.workersMin, workersMax:.workersMax, idleTimeout:.idleTimeout,
  flashboot:.flashboot, scalerType:.scalerType, scalerValue:.scalerValue,
  executionTimeoutMs:.executionTimeoutMs, allowedCudaVersions:.allowedCudaVersions,
  minCudaVersion:.minCudaVersion, gpuTypeIds:.gpuTypeIds,
  networkVolumeId:$volume, dataCenterIds:[$dc]
}' "$config")"
endpoint_id="${MUSIC3_RUNPOD_ENDPOINT_ID:-}"
if [[ -n "$endpoint_id" ]]; then
  endpoint_response="$(curl --fail-with-body -sS -X PATCH "$api/endpoints/$endpoint_id" "${auth[@]}" --data "$(jq 'del(.computeType)' <<<"$endpoint_payload")")"
else
  endpoint_response="$(curl --fail-with-body -sS -X POST "$api/endpoints" "${auth[@]}" --data "$endpoint_payload")"
  endpoint_id="$(jq -r '.id // empty' <<<"$endpoint_response")"
fi
[[ -n "$endpoint_id" ]] || { echo "ERROR: RunPod returned no endpoint ID" >&2; exit 1; }

umask 077
printf 'MUSIC3_RUNPOD_TEMPLATE_ID=%s\nMUSIC3_RUNPOD_ENDPOINT_ID=%s\nMUSIC3_RUNPOD_NETWORK_VOLUME_ID=%s\nMUSIC3_RUNPOD_DATACENTER_ID=%s\n' \
  "$template_id" "$endpoint_id" "$volume_id" "$datacenter_id" >"$state_file"
echo "Music3 RunPod runtime deployed"
echo "MUSIC3_RUNPOD_ENDPOINT_ID=$endpoint_id"
