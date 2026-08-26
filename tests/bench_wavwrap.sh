#!/usr/bin/env bash
# Benchmarks PCM16 -> WAV wrapping: native C library/CLI vs the Go
# implementation used by ringnz backend-go, plus a CUDA upload+wrap+download
# comparison when build/bench_wavwrap_cuda can be built.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BYTES="${BYTES:-67108864}" # 64 MiB ceiling payload
RATE="${RATE:-24000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

make -C "$ROOT" build/wavwrap build/bench_wavwrap >/dev/null
dd if=/dev/urandom of="$TMP/in.pcm" bs=1M count=$((BYTES / 1048576)) status=none

bench() { # label input_file cmd...
    local label="$1" infile="$2"; shift 2
    local best=0 run t
    for _ in 1 2 3; do
        if [ -n "$infile" ]; then
            t=$(python3 -c "import time,subprocess,sys;
t0=time.perf_counter()
with open(sys.argv[1],'rb') as f: subprocess.run(sys.argv[2:],stdin=f,stdout=subprocess.DEVNULL,check=True)
print(time.perf_counter()-t0)" "$infile" "$@")
        else
            t=$(python3 -c "import time,subprocess,sys;
t0=time.perf_counter()
subprocess.run(sys.argv[1:],stdout=subprocess.DEVNULL,check=True)
print(time.perf_counter()-t0)" "$@")
        fi
        run=$(python3 -c "print(int($BYTES/1048576/$t))")
        [ "$run" -gt "$best" ] && best=$run
    done
    printf '%-28s %8d MiB/s (best of 3)\n' "$label" "$best"
}

echo "payload sizes: real TTS replies are <=4 MiB; $((BYTES / 1048576)) MiB shown for the ceiling"
"$ROOT/build/bench_wavwrap" 2
"$ROOT/build/bench_wavwrap" $((BYTES / 1048576))

bench "C wavwrap CLI" "$TMP/in.pcm" "$ROOT/build/wavwrap" --rate "$RATE" "$TMP/in.pcm"

if command -v go >/dev/null; then
    bench "Go wavFromPCM16 CLI" "$TMP/in.pcm" go run "$ROOT/tests/wavwrap_go_ref.go"
fi

if [ -z "${WAVWRAP_SKIP_GPU:-}" ]; then
    make -C "$ROOT" build/bench_wavwrap_cuda >/dev/null 2>&1 || true
fi
if [ -x "$ROOT/build/bench_wavwrap_cuda" ]; then
    if ! "$ROOT/build/bench_wavwrap_cuda" 2; then
        echo "GPU measurement failed: no usable CUDA runtime"
    fi
else
    echo "build/bench_wavwrap_cuda not built: skipping GPU measurement"
fi
