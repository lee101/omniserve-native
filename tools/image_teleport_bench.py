#!/usr/bin/env python3
"""Gate exact latent teleportation against an unsplit deterministic render."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from image_parity_bench import DEFAULT_CORPUS, request_image


ROOT = Path(__file__).resolve().parents[1]


def slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def pixels_sha256(image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def request_with_retries(
    base: str,
    payload: dict,
    secret: str | None,
    timeout: float,
    *,
    retries: int,
    retry_delay: float,
):
    """Retry transient admission/backpressure responses, not semantic errors."""
    for attempt in range(retries + 1):
        try:
            return request_image(base, payload, secret, timeout)
        except RuntimeError as exc:
            retryable = "HTTP 503" in str(exc) or "HTTP 429" in str(exc)
            if not retryable or attempt >= retries:
                raise
            delay = retry_delay * (attempt + 1)
            print(
                json.dumps(
                    {
                        "retry": attempt + 1,
                        "delay_seconds": delay,
                        "reason": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8792")
    parser.add_argument("--secret-env", default="OMNISERVE_NATIVE_SECRET")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--teleport-start-step",
        type=int,
        default=None,
        help="resume step (default: final scheduled step, steps-1)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="offset corpus seeds to guarantee a fresh prime cache key",
    )
    parser.add_argument(
        "--replays",
        type=int,
        default=1,
        help="exact cache-hit replays per primed request (use >1 for an OOM/leak soak)",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error("--replays must be at least 1")
    if args.retries < 0 or args.retry_delay < 0:
        parser.error("--retries and --retry-delay must be non-negative")

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
            **{key: case[key] for key in ("prompt", "size")},
            "seed": int(case["seed"]) + args.seed_offset,
            "n": 1,
            "teleport_start_step": min(
                args.teleport_start_step
                if args.teleport_start_step is not None
                else defaults.get("steps", 9) - 1,
                defaults.get("steps", 9) - 1,
            ),
        }
        baseline, baseline_meta = request_with_retries(
            args.base, {**common, "teleport": False}, secret, args.timeout,
            retries=args.retries, retry_delay=args.retry_delay,
        )
        prime, prime_meta = request_with_retries(
            args.base, {**common, "teleport": True}, secret, args.timeout,
            retries=args.retries, retry_delay=args.retry_delay,
        )
        replay_results = [
            request_with_retries(
                args.base, {**common, "teleport": True}, secret, args.timeout,
                retries=args.retries, retry_delay=args.retry_delay,
            )
            for _ in range(args.replays)
        ]
        baseline_sha = pixels_sha256(baseline)
        prime_sha = pixels_sha256(prime)
        replay_shas = [pixels_sha256(image) for image, _ in replay_results]
        row_failures = []
        if baseline_sha != prime_sha:
            row_failures.append("split prime differs from unsplit baseline")
        prime_teleport = prime_meta.get("teleport") or {}
        if prime_teleport.get("cache_hit"):
            row_failures.append(
                "prime unexpectedly reported a cache hit; rerun with a fresh --seed-offset"
            )
        for replay_index, ((_, replay_meta), replay_sha) in enumerate(
            zip(replay_results, replay_shas, strict=True)
        ):
            teleport = replay_meta.get("teleport") or {}
            if prime_sha != replay_sha:
                row_failures.append(f"replay {replay_index} pixels differ from prime")
            if not teleport.get("cache_hit"):
                row_failures.append(f"replay {replay_index} did not report an exact cache hit")
            if teleport.get("method") != "exact_prompt_latent_replay":
                row_failures.append(
                    f"replay {replay_index} reported unexpected method "
                    f"{teleport.get('method')!r}"
                )
        for name, image in (("baseline", baseline), ("prime", prime)):
            image.save(run_dir / f"{index:02d}_{case['id']}_{name}.png", format="PNG")
        for replay_index, (replay, _) in enumerate(replay_results):
            replay.save(
                run_dir / f"{index:02d}_{case['id']}_replay_{replay_index:02d}.png",
                format="PNG",
            )
        replay_wall_ms = [meta["wall_ms"] for _, meta in replay_results]
        median_replay_wall_ms = statistics.median(replay_wall_ms)
        speedup = prime_meta["wall_ms"] / median_replay_wall_ms
        row = {
            "id": case["id"],
            "size": case["size"],
            "seed": common["seed"],
            "teleport_start_step": common["teleport_start_step"],
            "baseline": baseline_meta,
            "prime": prime_meta,
            "replay": replay_results[0][1],
            "replays": [meta for _, meta in replay_results],
            "median_replay_wall_ms": median_replay_wall_ms,
            "baseline_sha256": baseline_sha,
            "prime_sha256": prime_sha,
            "replay_sha256": replay_shas[0],
            "replay_sha256s": replay_shas,
            "speedup": speedup,
            "failures": row_failures,
            "passed": not row_failures,
        }
        rows.append(row)
        failures.extend(f"{case['id']}: {failure}" for failure in row_failures)
        print(json.dumps({"case": case["id"], "passed": row["passed"],
                          "cache_hits": sum(
                              bool((meta.get("teleport") or {}).get("cache_hit"))
                              for _, meta in replay_results
                          ),
                          "replays": args.replays,
                          "speedup": round(speedup, 3)}, sort_keys=True), flush=True)

    report = {
        "timestamp": slug(),
        "base": args.base,
        "seed_offset": args.seed_offset,
        "replays_per_case": args.replays,
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
        f"- replays per case: `{args.replays}`",
        f"- median prime/median-replay speedup: `{report['median_speedup']:.3f}x`",
        "",
        "| Case | Baseline s | Prime s | Replay s | Speedup | Exact |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['id']}` | {row['baseline']['wall_ms']/1000:.2f} | "
            f"{row['prime']['wall_ms']/1000:.2f} | {row['median_replay_wall_ms']/1000:.2f} | "
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
