#!/bin/sh
set -eu

fail() {
    echo "release audit: $*" >&2
    exit 1
}

for required in LICENSE THIRD_PARTY_NOTICES.md CONTRIBUTING.md SECURITY.md CODE_OF_CONDUCT.md; do
    test -s "$required" || fail "missing required release document: $required"
done

tracked_runtime_artifacts=$(
    git ls-files | grep -E '(^|/)(build[^/]*|\.venv|__pycache__)/|\.(gguf|safetensors|pt|pth|onnx|so|a|pyc)$' || true
)
test -z "$tracked_runtime_artifacts" || {
    echo "$tracked_runtime_artifacts" >&2
    fail "tracked build, environment, bytecode, or model artifact"
}

# Five MiB is intentionally generous for source/test fixtures and small enough
# to catch an accidentally added checkpoint, generated media corpus, or binary.
git ls-files | while IFS= read -r path; do
    test -f "$path" || continue
    bytes=$(wc -c < "$path")
    test "$bytes" -le 5242880 || fail "tracked file exceeds 5 MiB: $path ($bytes bytes)"
done

secret_files=$(
    git grep -I -l -E \
        '(-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35})' \
        -- . ':!scripts/release_audit.sh' || true
)
test -z "$secret_files" || {
    echo "$secret_files" >&2
    fail "possible credential material in tracked files"
}

echo "release audit: source tree checks passed"
