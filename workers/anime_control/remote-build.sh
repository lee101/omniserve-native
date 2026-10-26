#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cog="${ANIMA_COG_DIR:-$root/../anima-cog}"
host="${1:?usage: remote-build.sh user@host TAG [--push]}"
tag="${2:?tag}"
dir="${REMOTE_DIR:-/nvme0n1-disk/tmp/anime-control-ctx}"
ctx="$(mktemp -d "${TMPDIR:-/tmp}/anime-control-ctx.XXXXXX")"
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/omniserve"
cp "$here"/Dockerfile "$here"/requirements.txt "$here"/remote-run.sh "$ctx/"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$root/runtime" "$root/workloads" "$ctx/omniserve/"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$cog/anima_cog" "$ctx/omniserve/"
ssh "$host" "mkdir -p $dir"
rsync -a --delete "$ctx/" "$host:$dir/"
ssh -o ServerAliveInterval=20 "$host" "nohup bash $dir/remote-run.sh $tag ${3:-} > $dir/run.log 2>&1 < /dev/null & echo started"
echo "follow with: ssh $host tail -f $dir/run.log"
