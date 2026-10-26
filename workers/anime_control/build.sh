#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cog="${ANIMA_COG_DIR:-$root/../anima-cog}"
tag="${1:?usage: build.sh ghcr.io/lee101/anime-control:TAG [--push]}"
ctx="$(mktemp -d "${TMPDIR:-/tmp}/anime-control-ctx.XXXXXX")"
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/omniserve"
cp "$here"/Dockerfile "$here"/requirements.txt "$here"/download_weights.py "$ctx/"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$root/runtime" "$root/workloads" "$ctx/omniserve/"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$cog/anima_cog" "$ctx/omniserve/"
docker build -t "$tag" "$ctx"
[[ "${2:-}" == "--push" ]] && docker push "$tag"
docker image inspect "$tag" --format '{{.Id}} {{.Size}}'
