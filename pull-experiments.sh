#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote="${PROD_SSH_HOST:-dictator-prod}"
remote_dir="${PROD_EXPERIMENTS_DIR:-}"
destination="${EXPERIMENTS_DEST:-$root/experiments}"

command -v ssh >/dev/null || { echo "ERROR: ssh is required" >&2; exit 1; }
command -v rsync >/dev/null || { echo "ERROR: rsync is required" >&2; exit 1; }

if [[ -z "$remote_dir" ]]; then
  remote_dir="$(ssh -o BatchMode=yes "$remote" \
    'find /tmp -mindepth 2 -maxdepth 2 -type d -name experiments -print 2>/dev/null | sort | tail -n 1')"
fi
if [[ -z "$remote_dir" ]]; then
  echo "ERROR: no prod experiments directory found; set PROD_EXPERIMENTS_DIR" >&2
  exit 1
fi

mkdir -p "$destination"
echo "Pulling $remote:$remote_dir -> $destination"
rsync -az --mkpath "$remote:$remote_dir/" "$destination/"
