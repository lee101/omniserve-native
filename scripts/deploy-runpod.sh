#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_file="${OMNISERVE_RUNPOD_STATE:-${VIDEO_BACKGROUND_RUNPOD_STATE:-$root/.runpod.env}}"
config="$root/deploy/runpod.json"
worker="$root"

if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$root/.env"
  set +a
fi
if [[ -f "$state_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$state_file"
  set +a
fi

runpod_key="${RUNPOD_API_KEY:-${H3_RUNPOD_API_KEY:-}}"
image_repository="${OMNISERVE_IMAGE_REPOSITORY:-${VIDEO_BACKGROUND_IMAGE_REPOSITORY:-ghcr.io/lee101/omniserve-native}}"
registry_auth_id="${OMNISERVE_RUNPOD_REGISTRY_AUTH_ID:-${VIDEO_BACKGROUND_RUNPOD_REGISTRY_AUTH_ID:-}}"
if [[ -z "$runpod_key" ]]; then
  echo "ERROR: RUNPOD_API_KEY or H3_RUNPOD_API_KEY is required" >&2
  exit 1
fi
for command_name in curl docker jq sha256sum; do
  command -v "$command_name" >/dev/null || { echo "ERROR: missing $command_name" >&2; exit 1; }
done

dockerfile="$worker/Dockerfile.runpod"
tag="$(find "$worker/runtime" "$worker/workloads" "$worker/src" -type f \
  ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 \
  | sort -z | xargs -0 sha256sum; sha256sum "$dockerfile")"
tag="$(sha256sum <<<"$tag" | cut -c1-16)"
image="${image_repository}:${tag}"
echo "Building $image"
docker build --pull -t "$image" -f "$dockerfile" "$worker"
docker push "$image"

api="https://rest.runpod.io/v1"
auth=(-H "Authorization: Bearer $runpod_key" -H "Content-Type: application/json")
network_volume_id="${OMNISERVE_RUNPOD_NETWORK_VOLUME_ID:-${VIDEO_BACKGROUND_RUNPOD_NETWORK_VOLUME_ID:-}}"
network_volume_datacenter="${OMNISERVE_RUNPOD_DATACENTER_ID:-${VIDEO_BACKGROUND_RUNPOD_DATACENTER_ID:-}}"
use_network_volume="${OMNISERVE_RUNPOD_USE_NETWORK_VOLUME:-${VIDEO_BACKGROUND_RUNPOD_USE_NETWORK_VOLUME:-1}}"
if [[ "$use_network_volume" != "0" && "$use_network_volume" != "1" ]]; then
  echo "ERROR: VIDEO_BACKGROUND_RUNPOD_USE_NETWORK_VOLUME must be 0 or 1" >&2
  exit 1
fi
if [[ "$use_network_volume" == "0" ]]; then
  network_volume_id=""
  network_volume_datacenter=""
elif [[ -z "$network_volume_id" ]]; then
  network_volume_name="${OMNISERVE_RUNPOD_NETWORK_VOLUME_NAME:-${VIDEO_BACKGROUND_RUNPOD_NETWORK_VOLUME_NAME:-manifold-shared-video-models}}"
  volumes_response="$(curl --fail-with-body -sS "$api/networkvolumes" "${auth[@]}")"
  network_volume_id="$(jq -r --arg name "$network_volume_name" \
    '[.[] | select(.name == $name)][0].id // empty' <<<"$volumes_response")"
  network_volume_datacenter="$(jq -r --arg name "$network_volume_name" \
    '[.[] | select(.name == $name)][0].dataCenterId // empty' <<<"$volumes_response")"
fi
if [[ -n "$network_volume_id" && -z "$network_volume_datacenter" ]]; then
  network_volume_response="$(curl --fail-with-body -sS "$api/networkvolumes/$network_volume_id" "${auth[@]}")"
  network_volume_datacenter="$(jq -r '.dataCenterId // empty' <<<"$network_volume_response")"
fi
template_payload="$(jq -n --arg image "$image" --arg registry_auth "$registry_auth_id" '{
  imageName:$image, name:"omniserve-native", category:"NVIDIA",
  containerDiskInGb:20, dockerEntrypoint:[], dockerStartCmd:[], env:{},
  isPublic:false, isServerless:true, ports:[], readme:"OmniServe multi-workload native GPU runtime"
} | if $registry_auth != "" then .containerRegistryAuthId = $registry_auth else . end')"

template_id="${OMNISERVE_RUNPOD_TEMPLATE_ID:-${VIDEO_BACKGROUND_RUNPOD_TEMPLATE_ID:-}}"
if [[ -n "$template_id" ]]; then
  template_update_payload="$(jq 'del(.category, .isServerless)' <<<"$template_payload")"
  template_response="$(curl --fail-with-body -sS -X POST "$api/templates/$template_id/update" "${auth[@]}" --data "$template_update_payload")"
else
  template_response="$(curl --fail-with-body -sS -X POST "$api/templates" "${auth[@]}" --data "$template_payload")"
  template_id="$(jq -r '.id // empty' <<<"$template_response")"
fi
if [[ -z "$template_id" ]]; then
  echo "ERROR: RunPod returned no template ID" >&2
  exit 1
fi

endpoint_payload="$(jq --arg template "$template_id" '{
  templateId:$template, computeType:"GPU", gpuCount:1,
  name:.name, workersMin:.workersMin, workersMax:.workersMax,
  idleTimeout:.idleTimeout, flashboot:.flashboot, scalerType:.scalerType,
  scalerValue:.scalerValue, executionTimeoutMs:.executionTimeoutMs,
  allowedCudaVersions:.allowedCudaVersions, minCudaVersion:.minCudaVersion,
  gpuTypeIds:.gpuTypeIds
}' "$config")"
if [[ -n "$network_volume_id" ]]; then
  endpoint_payload="$(jq --arg volume "$network_volume_id" --arg datacenter "$network_volume_datacenter" \
    '.networkVolumeId = $volume
     | if $datacenter != "" then .dataCenterIds = [$datacenter] else . end' \
    <<<"$endpoint_payload")"
elif [[ "$use_network_volume" == "0" ]]; then
  endpoint_payload="$(jq '.networkVolumeId = "" | .networkVolumeIds = [] | .dataCenterIds = []' <<<"$endpoint_payload")"
fi

endpoint_id="${OMNISERVE_RUNPOD_ENDPOINT_ID:-${VIDEO_BACKGROUND_RUNPOD_ENDPOINT_ID:-}}"
if [[ -n "$endpoint_id" ]]; then
  endpoint_update_payload="$(jq 'del(.computeType)' <<<"$endpoint_payload")"
  endpoint_response="$(curl --fail-with-body -sS -X PATCH "$api/endpoints/$endpoint_id" "${auth[@]}" --data "$endpoint_update_payload")"
else
  endpoint_response="$(curl --fail-with-body -sS -X POST "$api/endpoints" "${auth[@]}" --data "$endpoint_payload")"
  endpoint_id="$(jq -r '.id // empty' <<<"$endpoint_response")"
fi
if [[ -z "$endpoint_id" ]]; then
  echo "ERROR: RunPod returned no endpoint ID" >&2
  exit 1
fi
if [[ "$use_network_volume" == "0" ]]; then
  # The REST API accepts an empty dataCenterIds list but can retain the legacy
  # GraphQL `locations` value from a previously attached regional volume.
  # Clear it explicitly so scale-to-zero workers can allocate in any region.
  graphql_query="mutation { saveEndpoint(input: { id: \"$endpoint_id\", gpuIds: \"AMPERE_48,ADA_48_PRO\", locations: \"\", name: \"omniserve-native\", templateId: \"$template_id\", workersMax: $(jq -r .workersMax "$config"), workersMin: $(jq -r .workersMin "$config") }) { id locations } }"
  graphql_response="$(curl --fail-with-body -sS -X POST \
    -H "Content-Type: application/json" \
    "https://api.runpod.io/graphql?api_key=$runpod_key" \
    --data "$(jq -n --arg query "$graphql_query" '{query:$query}')")"
  if jq -e '.errors | length > 0' >/dev/null 2>&1 <<<"$graphql_response"; then
    echo "ERROR: RunPod could not clear the endpoint region constraint" >&2
    exit 1
  fi
  # saveEndpoint applies GraphQL defaults to scaler fields that are omitted
  # above. Reapply the authoritative REST payload after clearing `locations`
  # so request-count scaling and the configured idle timeout are preserved.
  endpoint_response="$(curl --fail-with-body -sS -X PATCH "$api/endpoints/$endpoint_id" "${auth[@]}" --data "$endpoint_update_payload")"
fi

umask 077
printf 'OMNISERVE_RUNPOD_TEMPLATE_ID=%s\nOMNISERVE_RUNPOD_ENDPOINT_ID=%s\nVIDEO_BACKGROUND_RUNPOD_TEMPLATE_ID=%s\nVIDEO_BACKGROUND_RUNPOD_ENDPOINT_ID=%s\n' \
  "$template_id" "$endpoint_id" \
  "$template_id" "$endpoint_id" >"$state_file"
if [[ -n "$network_volume_id" ]]; then
  printf 'OMNISERVE_RUNPOD_NETWORK_VOLUME_ID=%s\nOMNISERVE_RUNPOD_DATACENTER_ID=%s\nVIDEO_BACKGROUND_RUNPOD_NETWORK_VOLUME_ID=%s\nVIDEO_BACKGROUND_RUNPOD_DATACENTER_ID=%s\n' \
    "$network_volume_id" "$network_volume_datacenter" \
    "$network_volume_id" "$network_volume_datacenter" >>"$state_file"
fi
echo "OmniServe RunPod runtime deployed"
echo "OMNISERVE_RUNPOD_ENDPOINT_ID=$endpoint_id"
