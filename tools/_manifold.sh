#!/usr/bin/env bash
# Shared ManifoldGen client for the tools/ CLIs. Sourced by every make-*.sh /
# remove-background.sh, never executed on its own.
#
#   source "$(dirname "$0")/_manifold.sh"
#
# What it provides:
#   mg_ensure_server   use the running ManifoldGen API, start it if it is down
#   mg_ensure_key      resolve a users.api_key (env, then the local database)
#   mg_request         authenticated curl; sets MG_STATUS, stores the response
#   mg_service         POST /api/service for one service id
#   mg_poll_job        follow an async audio/video job to completion
#   mg_public_url      make a local file reachable over http for the API
#   mg_get/mg_fetch    read the last response and save its artifact
#   mg_outfile         timestamped output path
#
# Environment:
#   MANIFOLDGEN_API    base URL, default http://127.0.0.1:8116
#   MANIFOLDGEN_DIR    checkout to build/start, default /vfast/data/code/manifoldgen-site
#   MANIFOLDGEN_API_KEY  bearer key; otherwise read from the local database
#   MANIFOLDGEN_NO_AUTOSTART=1  fail instead of starting a server
#   MG_OUT_DIR         directory for generated files, default results/
#   MG_TIMEOUT         seconds to wait for a generation, default 900

MG_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MG_JSON="$MG_LIB_DIR/mg_json.py"
MG_STATUS=""

mg_die() { printf '%s: %s\n' "${0##*/}" "$*" >&2; exit 1; }

mg_json() { python3 "$MG_JSON" "$@"; }

mg_api() { printf '%s' "${MANIFOLDGEN_API:-http://127.0.0.1:8116}"; }

mg_server_dir() { printf '%s' "${MANIFOLDGEN_DIR:-/vfast/data/code/manifoldgen-site}"; }

mg_port() {
  local url port
  url="$(mg_api)"
  port="${url##*:}"
  [[ "$port" == "$url" ]] && port=8116
  printf '%s' "${port%%/*}"
}

mg_is_local() {
  case "$(mg_api)" in
    http://127.0.0.1:* | http://localhost:* | http://\[::1\]:*) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------- server

mg_health() { curl -fsS --max-time 3 "$(mg_api)/api/health" >/dev/null 2>&1; }

mg_server_log() { printf '%s' "${MG_SERVER_LOG:-/tmp/manifoldgen-server.log}"; }

mg_start_server() {
  local dir port log
  dir="$(mg_server_dir)"
  port="$(mg_port)"
  log="$(mg_server_log)"
  [[ -f "$dir/server/main.go" ]] || mg_die "no ManifoldGen checkout at $dir (set MANIFOLDGEN_DIR)"
  [[ -x "$dir/server/manifoldgen-server" ]] || command -v go >/dev/null || mg_die "go is required to build the ManifoldGen server"

  printf 'starting ManifoldGen API on port %s (log: %s)\n' "$port" "$log" >&2
  local bin="$dir/server/manifoldgen-server"
  if command -v go >/dev/null && [[ ! -x "$bin" || -n "$(find "$dir/server" -name '*.go' -newer "$bin" -print -quit)" ||
    "$dir/server/go.mod" -nt "$bin" || "$dir/server/go.sum" -nt "$bin" ]]; then
    if ! (cd "$dir/server" && go build -o manifoldgen-server .) >"$log.build" 2>&1; then
      cat "$log.build" >&2
      mg_die "ManifoldGen build failed"
    fi
  fi
  # godotenv loads ../.env from the server directory; PORT/DIST_DIR here win
  # over .env because godotenv never overrides an existing variable.
  (cd "$dir/server" && PORT="$port" DIST_DIR=../frontend/out \
    nohup ./manifoldgen-server >>"$log" 2>&1 &)

  local i
  for ((i = 1; i <= 90; i++)); do
    if mg_health; then
      return 0
    fi
    if ! kill -0 "$(pgrep -f "manifoldgen-server" | head -1)" 2>/dev/null; then
      tail -20 "$log" >&2
      mg_die "ManifoldGen exited during startup"
    fi
    sleep 1
  done
  tail -20 "$log" >&2
  mg_die "ManifoldGen did not become healthy on $(mg_api)"
}

# Use the running server; start one when there is none. Every tool calls this
# before its first request so a cold box still works.
MG_SERVER_OK=""
mg_ensure_server() {
  [[ -n "$MG_SERVER_OK" ]] && return 0
  if mg_health; then
    MG_SERVER_OK=1
    return 0
  fi
  if [[ -n "${MANIFOLDGEN_NO_AUTOSTART:-}" ]]; then
    mg_die "no ManifoldGen server at $(mg_api) and MANIFOLDGEN_NO_AUTOSTART is set"
  fi
  mg_start_server
  MG_SERVER_OK=1
}

# ---------------------------------------------------------------- auth

mg_database_url() {
  local env_file line
  env_file="$(mg_server_dir)/.env"
  if [[ -r "$env_file" ]]; then
    line="$(grep -m1 '^DATABASE_URL=' "$env_file" 2>/dev/null || true)"
    if [[ -n "$line" ]]; then
      printf '%s' "${line#DATABASE_URL=}"
      return 0
    fi
  fi
  printf '%s' "postgres://manifoldgen:manifoldgen_pass_2026@localhost:5432/manifoldgen?sslmode=disable"
}

# The server accepts exactly one inbound credential: Authorization: Bearer with
# a users.api_key value. Locally that comes from the same database the server
# uses; the seed user is the account the repo's own scripts drive.
mg_key_cache() {
  printf '%s/manifoldgen/key-%s' "${XDG_CACHE_HOME:-$HOME/.cache}" "$(printf '%s' "$(mg_api)" | tr -c 'A-Za-z0-9' '_')"
}

# The key is cached per API url so a run does not pay for a psql round trip;
# mg_forget_key drops it when the API answers 401 (rotated or reseeded).
mg_forget_key() {
  rm -f "$(mg_key_cache)"
  MANIFOLDGEN_API_KEY=""
}

mg_ensure_key() {
  if [[ -n "${MANIFOLDGEN_API_KEY:-}" ]]; then
    return 0
  fi
  local cache key
  cache="$(mg_key_cache)"
  if mg_is_local && [[ -r "$cache" ]]; then
    MANIFOLDGEN_API_KEY="$(<"$cache")"
    [[ -n "$MANIFOLDGEN_API_KEY" ]] && return 0
  fi
  command -v psql >/dev/null || mg_die "set MANIFOLDGEN_API_KEY (no psql to read the local database)"
  mg_is_local || mg_die "set MANIFOLDGEN_API_KEY for a remote $(mg_api)"
  key="$(psql "$(mg_database_url)" -tAc \
    "select api_key from users where email='seed@manifoldgen.com'" 2>/dev/null | head -1)"
  [[ -n "$key" ]] || mg_die "no seed@manifoldgen.com API key; set MANIFOLDGEN_API_KEY"
  MANIFOLDGEN_API_KEY="$key"
  mkdir -p "$(dirname "$cache")" && (umask 077; printf '%s' "$key" >"$cache") || true
}

# ---------------------------------------------------------------- transport

# The last response lives in a file, never a variable: an image response is
# megabytes of base64 and a shell variable that size is both slow and unsafe to
# pass around. mg_json reads the file from stdin.
MG_BODY_FILE=""
MG_TEMPFILES=()

# One trap owns every scratch file. A script that adds its own EXIT trap would
# otherwise replace this one and leak the response body.
mg_cleanup() {
  local file
  for file in ${MG_TEMPFILES+"${MG_TEMPFILES[@]}"}; do
    rm -f "$file"
  done
}
trap mg_cleanup EXIT

mg_tempfile() { MG_TEMPFILES+=("$(mktemp)"); }

# These set globals instead of printing a path: under a command substitution
# they would run in a subshell and the caller's variable would never be set.
mg_body_file() {
  if [[ -z "$MG_BODY_FILE" ]]; then
    mg_tempfile
    MG_BODY_FILE="${MG_TEMPFILES[-1]}"
  fi
}

# mg_stash_body VAR -- copy the current response aside and point VAR at it, for
# a script that needs to read an earlier response after another request.
mg_stash_body() {
  local -n target="$1"
  mg_body_file
  mg_tempfile
  target="${MG_TEMPFILES[-1]}"
  cp "$MG_BODY_FILE" "$target"
}

mg_get() { mg_body_file; mg_json get "$@" <"$MG_BODY_FILE"; }
mg_count() { mg_body_file; mg_json count "$@" <"$MG_BODY_FILE"; }
mg_jget() { mg_body_file; mg_json jget "$@" <"$MG_BODY_FILE"; }
mg_err() { mg_body_file; mg_json err "$@" <"$MG_BODY_FILE"; }
mg_fetch() { mg_body_file; mg_json fetch "$@" <"$MG_BODY_FILE"; }

# mg_request METHOD PATH [JSON_BODY] -- stores the response and sets MG_STATUS.
mg_request() {
  local method="$1" path="$2" payload="${3:-}" status
  mg_ensure_server
  mg_ensure_key
  mg_body_file
  if [[ -n "$payload" ]]; then
    status="$(curl -sS -o "$MG_BODY_FILE" -w '%{http_code}' -X "$method" "$(mg_api)$path" \
      -H "Authorization: Bearer $MANIFOLDGEN_API_KEY" \
      -H 'Content-Type: application/json' \
      --data-binary "$payload")" || mg_die "request to $path failed"
  else
    status="$(curl -sS -o "$MG_BODY_FILE" -w '%{http_code}' -X "$method" "$(mg_api)$path" \
      -H "Authorization: Bearer $MANIFOLDGEN_API_KEY")" || mg_die "request to $path failed"
  fi
  MG_STATUS="$status"
}

# Fail with the server's own message. 202 is success for the async services.
mg_require_ok() {
  case "$MG_STATUS" in
    200 | 201 | 202) return 0 ;;
  esac
  mg_err "$MG_STATUS" >&2
  exit 1
}

mg_service() {
  mg_request POST /api/service "$1"
  mg_require_ok
}

# ---------------------------------------------------------------- jobs

# mg_poll_job audio|video JOB_ID -- follows the job to a terminal state and
# leaves the final document in the response file.
mg_poll_job() {
  local kind="$1" job_id="$2" deadline=$((SECONDS + ${MG_TIMEOUT:-900})) status
  while ((SECONDS < deadline)); do
    mg_request GET "/api/$kind-jobs/$job_id"
    case "$MG_STATUS" in
      200) ;;
      202) sleep 3; continue ;;
      *) mg_err "$MG_STATUS" >&2; exit 1 ;;
    esac
    status="$(mg_get job.status 2>/dev/null || true)"
    case "$status" in
      completed | succeeded | success | done)
        printf 'job %s: %s\n' "$job_id" "$status" >&2
        return 0
        ;;
      failed | error | cancelled | canceled)
        mg_err >&2
        exit 1
        ;;
    esac
    printf 'job %s: %s\r' "$job_id" "${status:-queued}" >&2
    sleep 3
  done
  mg_die "job $job_id did not finish within ${MG_TIMEOUT:-900}s"
}

# ---------------------------------------------------------------- input files

mg_content_type() {
  python3 - "$1" <<'PY'
import mimetypes, os, sys
kind = mimetypes.guess_type(os.path.basename(sys.argv[1]))[0]
print(kind or "application/octet-stream")
PY
}

mg_require_file() {
  [[ -n "${1:-}" ]] || mg_die "missing file argument"
  [[ -f "$1" ]] || mg_die "no such file: $1"
  printf '%s' "$1"
}

# Image, edit, upscale and voice endpoints take image_url/video_url as an
# absolute public URL and reject base64, so a local file is uploaded to the
# same R2 bucket the site uses first.
mg_public_url() {
  local file="$1" content_type encoded upload public
  file="$(mg_require_file "$file")"
  content_type="$(mg_content_type "$file")"
  encoded="$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "${file##*/}")"
  mg_request GET "/api/uploads/presign?filename=$encoded&content_type=$content_type&dataset=cli"
  mg_require_ok
  upload="$(mg_get upload_url)" || mg_die "no upload_url returned"
  public="$(mg_get public_url)" || mg_die "no public_url returned"
  curl -fsS -X PUT "$upload" -H "Content-Type: $content_type" --data-binary "@$file" \
    || mg_die "upload of $file failed"
  printf '%s' "$public"
}

# ---------------------------------------------------------------- output

# Generated artifacts land in results/ so a run from any directory keeps its
# output together and out of the working tree. MG_OUT_DIR overrides the place;
# an explicit -o/--out path is always taken literally.
mg_outfile() {
  local prefix="$1" ext="$2" dir="${MG_OUT_DIR:-results}" stamp path
  mkdir -p "$dir" || mg_die "cannot create output directory $dir"
  stamp="$(date +%Y%m%d-%H%M%S)"
  path="$dir/$prefix-$stamp.$ext"
  local i=1
  while [[ -e "$path" ]]; do
    path="$dir/$prefix-$stamp-$i.$ext"
    i=$((i + 1))
  done
  printf '%s' "$path"
}


# mg_resolve_job audio|video -- follow a 202 service response to its finished
# job document. Synchronous services are left alone.
mg_resolve_job() {
  local kind="$1" job_id
  [[ "$MG_STATUS" == 202 ]] || return 0
  job_id="$(mg_get result.job_id)" || mg_die "202 response without result.job_id"
  mg_report "queued as $job_id, polling /api/$kind-jobs/$job_id"
  mg_poll_job "$kind" "$job_id"
}


mg_report() {
  printf '%s\n' "$*" >&2
}


# mg_field PATH... -- first value present in the last response, or "?". Jobs
# nest their payload under job.result; the synchronous lanes return it flat, so
# every summary line lists both shapes.
mg_field() {
  local value
  value="$(mg_get "$@" 2>/dev/null || true)"
  printf '%s' "${value:-?}"
}

# mg_body KEY VALUE [KEY VALUE ...] -- assemble a JSON request body without any
# shell quoting hazards. Numbers and booleans keep their JSON types; an empty
# value drops the key so optional flags stay out of the request.
mg_body() {
  python3 - "$@" <<'PY'
import json, sys

args = sys.argv[1:]
body = {}
for index in range(0, len(args) - 1, 2):
    key, value = args[index], args[index + 1]
    if value == "":
        continue
    if value in ("true", "false"):
        body[key] = value == "true"
        continue
    try:
        body[key] = int(value)
        continue
    except ValueError:
        pass
    try:
        body[key] = float(value)
    except ValueError:
        body[key] = value
print(json.dumps(body))
PY
}