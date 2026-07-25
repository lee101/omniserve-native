#!/usr/bin/env python3
"""Evaluate a candidate foreground/background estimate against the pymatting
reference fixtures.

Two ways to supply the candidate:

  --candidate path.npy
      An already-written .npy of shape (H, W, 3), float32/float64.

  --candidate-cmd "<shell command>"
      A command that MUST write an .npy to the path given by --candidate-out
      (a temp file if not specified). The command string is formatted with:
          {out}    destination .npy path
          {image}  path to <case>_image.npy
          {alpha}  path to <case>_alpha.npy
          {case}   case name
          {h} {w} {depth}
      e.g.
        --candidate-cmd "./build/matte_ml {image} {alpha} {out}"

Exit code 0 on PASS, 1 on FAIL (2 on usage/IO error).
"""

import argparse
import json
import math
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURES = HERE / "fixtures"


def die(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_npy(path, what):
    p = Path(path)
    if not p.is_file():
        die(f"{what} not found: {p}")
    try:
        arr = np.load(p)
    except Exception as e:  # noqa: BLE001
        die(f"failed to load {what} {p}: {e}")
    return arr


def metrics(cand, ref):
    d = np.abs(cand.astype(np.float64) - ref.astype(np.float64))
    max_abs = float(d.max())
    mean_abs = float(d.mean())
    mse = float((d * d).mean())
    rmse = math.sqrt(mse)
    # peak = 1.0 (images are in [0, 1])
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    idx = np.unravel_index(int(d.argmax()), d.shape)
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "psnr_db": psnr,
        "argmax_index": [int(i) for i in idx],
        "n_elements": int(d.size),
        "n_over_atol": None,  # filled in by caller (needs atol)
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    ap.add_argument("--case", required=True, help="fixture case name, e.g. A, B, C")
    ap.add_argument("--target", default="fg", choices=["fg", "bg"],
                    help="which reference to compare against (default: fg)")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES),
                    help=f"fixtures directory (default: {DEFAULT_FIXTURES})")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--candidate", help="path to candidate .npy")
    src.add_argument("--candidate-cmd", help="shell command that writes {out} as .npy")
    ap.add_argument("--candidate-out", help="where --candidate-cmd should write (default: temp file)")
    ap.add_argument("--atol", type=float, default=2e-3, help="max abs error threshold (default: 2e-3)")
    ap.add_argument("--mean-atol", type=float, default=2e-4, help="mean abs error threshold (default: 2e-4)")
    ap.add_argument("--report-json", help="write a JSON report to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    fixtures = Path(args.fixtures)
    ref_path = fixtures / f"{args.case}_{args.target}_ref.npy"
    image_path = fixtures / f"{args.case}_image.npy"
    alpha_path = fixtures / f"{args.case}_alpha.npy"

    ref = load_npy(ref_path, "reference")

    # ---- obtain the candidate array
    cmd_info = None
    tmpdir = None
    if args.candidate_cmd:
        if args.candidate_out:
            out_path = Path(args.candidate_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmpdir = tempfile.TemporaryDirectory()
            out_path = Path(tmpdir.name) / f"{args.case}_{args.target}.npy"
        h, w, depth = ref.shape
        cmd = args.candidate_cmd.format(
            out=shlex.quote(str(out_path)),
            image=shlex.quote(str(image_path)),
            alpha=shlex.quote(str(alpha_path)),
            case=args.case, h=h, w=w, depth=depth,
        )
        if not args.quiet:
            print(f"running: {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.stdout and not args.quiet:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
        if proc.returncode != 0:
            die(f"candidate command exited {proc.returncode}")
        cmd_info = {"cmd": cmd, "returncode": proc.returncode, "out": str(out_path)}
        cand = load_npy(out_path, "candidate (from --candidate-cmd)")
    else:
        cand = load_npy(args.candidate, "candidate")

    # ---- shape / dtype checks
    if cand.shape != ref.shape:
        die(f"shape mismatch: candidate {cand.shape} vs reference {ref.shape}", 1)
    if not np.isfinite(cand).all():
        n_bad = int((~np.isfinite(cand)).sum())
        die(f"candidate contains {n_bad} non-finite values", 1)

    m = metrics(cand, ref)
    d = np.abs(cand.astype(np.float64) - ref.astype(np.float64))
    m["n_over_atol"] = int((d > args.atol).sum())

    passed = (m["max_abs_error"] <= args.atol) and (m["mean_abs_error"] <= args.mean_atol)

    report = {
        "case": args.case,
        "target": args.target,
        "reference": str(ref_path),
        "candidate": str(args.candidate) if args.candidate else None,
        "candidate_cmd": cmd_info,
        "shape": list(ref.shape),
        "candidate_dtype": str(cand.dtype),
        "reference_dtype": str(ref.dtype),
        "thresholds": {"atol": args.atol, "mean_atol": args.mean_atol},
        "metrics": m,
        "result": "PASS" if passed else "FAIL",
    }

    if not args.quiet:
        print(f"case {args.case} target {args.target}  shape {tuple(ref.shape)}  "
              f"candidate dtype {cand.dtype}")
        print(f"  max  abs error : {m['max_abs_error']:.6e}   (atol      {args.atol:.1e})")
        print(f"  mean abs error : {m['mean_abs_error']:.6e}   (mean-atol {args.mean_atol:.1e})")
        print(f"  rmse           : {m['rmse']:.6e}")
        print(f"  psnr           : {m['psnr_db']:.2f} dB" if math.isfinite(m["psnr_db"])
              else "  psnr           : inf dB (exact match)")
        print(f"  elements > atol: {m['n_over_atol']} / {m['n_elements']}")
        print(f"  worst at (y,x,c)={tuple(m['argmax_index'])}")
        print(f"RESULT: {report['result']}")

    if args.report_json:
        p = Path(args.report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(report, f, indent=2)
        if not args.quiet:
            print(f"wrote report {p}")

    if tmpdir is not None:
        tmpdir.cleanup()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
