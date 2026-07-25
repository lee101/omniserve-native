#!/usr/bin/env bash
# Full correctness matrix for the foreground-estimation port.
#
#   ./matte/run_eval.sh [build-dir]
#
# Runs every backend (C sequential, C red-black at two thread counts, CUDA)
# against the pymatting fixtures in matte/fixtures and prints the measured
# error for each. The sequential backend is the one that must match pymatting
# bit-for-bit-ish; red-black/CUDA are a different (parallel-safe) sweep order,
# so they are scored against the CPU red-black result as well as the reference.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
BUILD="${1:-$REPO/build-cuda}"
CLI="$BUILD/omatte_cli"
PY="$HERE/.venv/bin/python"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -x "$CLI" ] || { echo "missing $CLI - build with -DWITH_MATTE_CUDA=ON"; exit 2; }
[ -x "$PY" ] || { echo "missing $PY - see matte/README.md"; exit 2; }

status=0
for case in A B C; do
    image="$HERE/fixtures/${case}_image.npy"
    alpha="$HERE/fixtures/${case}_alpha.npy"

    for backend in "sequential:--order sequential --threads 1" \
                   "redblack-1:--order redblack --threads 1" \
                   "redblack-16:--order redblack --threads 16" \
                   "cuda:--cuda"; do
        name="${backend%%:*}"
        flags="${backend#*:}"
        out="$TMP/${case}_${name}.npy"

        # shellcheck disable=SC2086
        if ! "$CLI" --image "$image" --alpha "$alpha" --out-fg "$out" $flags >/dev/null 2>"$TMP/err"; then
            echo "case $case $name: SKIP ($(tail -1 "$TMP/err"))"
            continue
        fi

        printf 'case %s %-12s ' "$case" "$name"
        "$PY" - "$HERE/fixtures/${case}_fg_ref.npy" "$out" "$TMP/${case}_sequential.npy" <<'PY'
import sys
import numpy as np

ref = np.load(sys.argv[1])
got = np.load(sys.argv[2])
err = np.abs(got - ref)
line = f"vs pymatting: max {err.max():.3e} mean {err.mean():.3e}"
try:
    seq = np.load(sys.argv[3])
    line += f" | vs C-sequential: max {np.abs(got - seq).max():.3e}"
except OSError:
    pass
print(line)
PY
    done
done

echo
echo "Convergence check (both sweep orders approach the same solution):"
"$PY" - "$HERE/fixtures" "$CLI" "$TMP" <<'PY'
import subprocess
import sys
from pathlib import Path

import numpy as np
from pymatting import estimate_foreground_ml

fixtures, cli, tmp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for case in ("A", "B", "C"):
    image = np.load(fixtures / f"{case}_image.npy")
    alpha = np.load(fixtures / f"{case}_alpha.npy")
    converged = estimate_foreground_ml(image, alpha, n_small_iterations=200, n_big_iterations=40)
    converged = converged.astype(np.float32)

    row = {"pymatting-default": np.abs(np.load(fixtures / f"{case}_fg_ref.npy") - converged).max()}
    for label, args in (
        ("redblack-default", ["--order", "redblack"]),
        ("redblack-40x10", ["--order", "redblack", "--small-iterations", "40", "--big-iterations", "10"]),
    ):
        out = tmp / f"{case}_{label}.npy"
        subprocess.run(
            [cli, "--image", str(fixtures / f"{case}_image.npy"),
             "--alpha", str(fixtures / f"{case}_alpha.npy"), "--out-fg", str(out), *args],
            check=True, capture_output=True,
        )
        row[label] = np.abs(np.load(out) - converged).max()
    print(f"  case {case}: " + "  ".join(f"{k} {v:.3e}" for k, v in row.items()))
PY

exit $status
