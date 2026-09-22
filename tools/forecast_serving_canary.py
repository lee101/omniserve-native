#!/usr/bin/env python3
"""Bounded local Chronos single/batch parity and latency canary (stdlib only).

Run on the GPU host; uses existing worker admission, never admin unload hooks.
Synthetic data measures serving parity, NOT held-out forecasting accuracy.
"""
import argparse
import json
import math
import statistics
import subprocess
import time
import urllib.request


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.load(response)
    return result, (time.perf_counter() - start) * 1000


def forecast_values(output):
    vectors = [output["mean"]] + [output["quantiles"][q] for q in ("0.1", "0.5", "0.9")]
    if any(len(vector) != 16 for vector in vectors):
        raise ValueError("invalid forecast vector shape")
    return output["mean"] + [x for q in ("0.1", "0.5", "0.9")
                             for x in output["quantiles"][q]]


def error_metrics(reference, candidate, context):
    a, b = forecast_values(reference), forecast_values(candidate)
    if len(a) != 64 or len(b) != 64 or not all(math.isfinite(x) for x in a + b):
        raise ValueError("invalid forecast shape or non-finite output")
    errors = [abs(x - y) for x, y in zip(a, b)]
    # Diagnostic only: do not use BF16 spacing to relax the parity gate.
    ulps = [error / math.ldexp(1.0, max(-126, math.frexp(max(abs(x), abs(y)))[1] - 1) - 7)
            for x, y, error in zip(a, b, errors)]
    return {"max_abs": max(errors), "mae": statistics.mean(errors),
            "max_context_std_units": max(errors) / max(statistics.pstdev(context), 1e-12),
            "max_bf16_steps": max(ulps),
            "changed_values": sum(error != 0 for error in errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cold", action="store_true")
    parser.add_argument("--ragged", action="store_true",
                        help="use unequal context lengths to exercise missing-value padding")
    args = parser.parse_args()
    base = "http://127.0.0.1:8101"
    health, _ = call(base, "/health")
    queue = health["gpu_queue"]
    if not health["cuda"]["healthy"] or any(queue.get(k, 0) for k in
            ("active", "waiting_high", "waiting_low")):
        raise RuntimeError("worker unhealthy or busy; retry off-peak")
    cold = not health["models"]["chronos2"]
    if cold and not args.allow_cold:
        raise RuntimeError("model cold; opt in with --allow-cold")
    if health["models"].get("zimage"):
        raise RuntimeError("refusing potential co-resident model eviction")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(gpu) != 1:
        raise RuntimeError("expected one GPU")
    free, util = [int(v.strip()) for v in gpu[0].split(",")]
    if free < 4096 or util >= 85:
        raise RuntimeError("insufficient GPU headroom; retry off-peak")

    series = [[10 + k + math.sin(i / (8 + k)) for i in range(128)]
              for k in range(4)]
    if args.ragged:
        series = [values[-length:] for values, length in zip(series, (128, 97, 64, 33))]
    common = {"prediction_length": 16, "quantile_levels": [0.1, 0.5, 0.9]}
    _, first_ms = call(base, "/forecast", dict(common, values=series[0]))
    singles = []
    single_ms = []
    for values in series:
        output, elapsed = call(base, "/forecast", dict(common, values=values))
        singles.append(output)
        single_ms.append(elapsed)
    batches = []
    max_error = 0.0
    diagnostics = []
    for _ in range(3):
        batch, elapsed = call(base, "/forecast_batch", dict(common, series=series))
        batches.append(elapsed)
        assert len(batch["results"]) == len(singles)
        diagnostics.append([error_metrics(single, result, context)
                            for single, result, context in zip(singles, batch["results"], series)])
        for single, batched in zip(singles, batch["results"]):
            for a, b in [(single["mean"], batched["mean"])] + [
                    (single["quantiles"][q], batched["quantiles"][q])
                    for q in ("0.1", "0.5", "0.9")]:
                assert len(a) == len(b) == common["prediction_length"]
                for x, y in zip(a, b):
                    assert math.isfinite(x) and math.isfinite(y)
                    max_error = max(max_error, abs(x - y))
    repeatability = []
    for reference, values in zip(singles, series):
        repeated, _ = call(base, "/forecast", dict(common, values=values))
        repeatability.append(error_metrics(reference, repeated, values))
    report = {"cold_start": cold, "first_request_ms": first_ms,
              "context_lengths": [len(values) for values in series],
              "batch_error_diagnostics": diagnostics,
              "single_repeatability": repeatability,
              "single_ms": single_ms, "batch_four_ms": batches,
              "batch_speedup": sum(single_ms) / statistics.median(batches),
              "max_abs_parity_error": max_error,
              "parity_tolerance": 0.02, "parity_pass": max_error <= 0.02}
    print(json.dumps(report, indent=2), flush=True)
    if not report["parity_pass"]:
        raise SystemExit("single/batch parity failed; do not enable batching")


if __name__ == "__main__":
    main()
