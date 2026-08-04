#!/usr/bin/env python3
"""Gate exact latent teleportation against an unsplit deterministic render."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from image_parity_bench import DEFAULT_CORPUS, request_image


ROOT = Path(__file__).resolve().parents[1]


def slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def pixels_sha256(image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8792")
    parser.add_argument("--secret-env", default="OMNISERVE_NATIVE_SECRET")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--teleport-start-step", type=int, default=7)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text())
    cases = corpus["cases"][: args.limit or None]
    defaults = corpus.get("defaults", {})
    run_dir = args.output_dir / f"image_teleport_{slug()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    secret = os.getenv(args.secret_env)
    rows = []
    failures = []
    for index, case in enumerate(cases):
        common = {
            **defaults,
            **{key: case[key] for key in ("prompt", "size", "seed")},
            "n": 1,
            "teleport_start_step": min(args.teleport_start_step, defaults.get("steps", 9) - 1),
        }
        baseline, baseline_meta = request_image(
            args.base, {**common, "teleport": False}, secret, args.timeout)
        prime, prime_meta = request_image(
            args.base, {**common, "teleport": True}, secret, args.timeout)
        replay, replay_meta = request_image(
            args.base, {**common, "teleport": True}, secret, args.timeout)
        baseline_sha = pixels_sha256(baseline)
        prime_sha = pixels_sha256(prime)
        replay_sha = pixels_sha256(replay)
        teleport = replay_meta.get("teleport") or {}
        row_failures = []
        if baseline_sha != prime_sha:
            row_failures.append("split prime differs from unsplit baseline")
        if prime_sha != replay_sha:
            row_failures.append("replayed pixels differ from prime")
        if not teleport.get("cache_hit"):
            row_failures.append("replay did not report an exact cache hit")
        if teleport.get("method") != "exact_prompt_latent_replay":
            row_failures.append(f"unexpected method {teleport.get('method')!r}")
        for name, image in (("baseline", baseline), ("prime", prime), ("replay", replay)):
            image.save(run_dir / f"{index:02d}_{case['id']}_{name}.png", format="PNG")
        speedup = prime_meta["wall_ms"] / replay_meta["wall_ms"]
        row = {
            "id": case["id"],
            "size": case["size"],
            "baseline": baseline_meta,
            "prime": prime_meta,
            "replay": replay_meta,
            "baseline_sha256": baseline_sha,
            "prime_sha256": prime_sha,
            "replay_sha256": replay_sha,
            "speedup": speedup,
            "failures": row_failures,
            "passed": not row_failures,
        }
        rows.append(row)
        failures.extend(f"{case['id']}: {failure}" for failure in row_failures)
        print(json.dumps({"case": case["id"], "passed": row["passed"],
                          "cache_hit": teleport.get("cache_hit"),
                          "speedup": round(speedup, 3)}, sort_keys=True), flush=True)

    report = {
        "timestamp": slug(),
        "base": args.base,
        "rows": rows,
        "median_speedup": statistics.median(row["speedup"] for row in rows),
        "failures": failures,
        "passed": not failures,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# OmniServe Exact Latent Teleport",
        "",
        f"- passed: `{report['passed']}`",
        f"- median prime/replay speedup: `{report['median_speedup']:.3f}x`",
        "",
        "| Case | Baseline s | Prime s | Replay s | Speedup | Exact |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['id']}` | {row['baseline']['wall_ms']/1000:.2f} | "
            f"{row['prime']['wall_ms']/1000:.2f} | {row['replay']['wall_ms']/1000:.2f} | "
            f"{row['speedup']:.3f}x | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""] + [f"- {failure}" for failure in failures])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": report["passed"], "run_dir": str(run_dir),
                      "median_speedup": report["median_speedup"],
                      "failures": failures}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
