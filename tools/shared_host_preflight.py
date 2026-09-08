#!/usr/bin/env python3
"""Read-only, dependency-free host snapshot and conservative GPU capacity gate.

Never loads a model, contacts a provider, or changes service/GPU state.
Capacity is an estimate, not a reservation: rerun immediately before a canary.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

GIB = 1024 ** 3


def parse_meminfo(raw: str) -> dict[str, int]:
    return {line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in raw.splitlines() if len(line.split()) >= 3
            and line.split()[2] == "kB"}


def parse_gpus(raw: str) -> list[dict]:
    rows = []
    for row in csv.reader(io.StringIO(raw)):
        if not row:
            continue
        index, name, total, free, used, utilization = [v.strip() for v in row]
        rows.append({"index": int(index), "name": name,
                     "total_gib": int(total) / 1024, "free_gib": int(free) / 1024,
                     "used_gib": int(used) / 1024,
                     "utilization_pct": int(utilization)})
    return rows


def capacity(gpu: dict, weights_gib: float, runtime_gib: float,
             reserve_gib: float, target_gib: float) -> dict:
    """Do not credit existing allocations back to the candidate model."""
    values = (weights_gib, runtime_gib, reserve_gib, target_gib)
    if any(not math.isfinite(v) or v < 0 for v in values) or target_gib == 0:
        raise ValueError("capacity sizes must be finite and nonnegative; target must be positive")
    budget = max(0.0, min(target_gib, gpu["free_gib"] - reserve_gib))
    required = weights_gib + runtime_gib
    return {"weights_gib": weights_gib, "runtime_and_kv_gib": runtime_gib,
            "reserve_gib": reserve_gib, "target_gib": target_gib,
            "candidate_budget_gib": budget, "required_gib": required,
            "fits_estimate": required <= budget,
            "shortfall_gib": max(0.0, required - budget)}


def snapshot(disk_path: str) -> dict:
    mem = parse_meminfo(Path("/proc/meminfo").read_text())
    disk = shutil.disk_usage(disk_path)
    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cpu_count": os.cpu_count(), "load_average": list(os.getloadavg()),
        "memory_gib": {key: mem.get(key, 0) / GIB for key in
                       ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")},
        "disk": {"path": str(Path(disk_path).resolve()),
                 "free_gib": disk.free / GIB, "total_gib": disk.total / GIB,
                 "used_pct": disk.used / disk.total * 100},
        "pressure": {}, "gpus": [], "warnings": [],
    }
    for resource in ("cpu", "memory", "io"):
        try:
            report["pressure"][resource] = Path(f"/proc/pressure/{resource}").read_text().strip()
        except OSError:
            report["pressure"][resource] = None
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=5, check=True)
        report["gpus"] = parse_gpus(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        report["warnings"].append("GPU telemetry unavailable; capacity cannot be approved")
    if disk.used / disk.total >= 0.95:
        report["warnings"].append("Disk at least 95% full; avoid conversion/download staging here")
    if mem.get("SwapTotal", 0) and mem.get("SwapFree", 0) / mem["SwapTotal"] < 0.05:
        report["warnings"].append("Swap nearly full; inspect PSI and swap-in/out, not swap occupancy alone")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-path", default=".")
    parser.add_argument("--model", type=Path, help="existing GGUF; stat only, no weight reads")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runtime-gib", type=float,
                        help="required with --model: measured or conservative KV/workspace allowance")
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--target-gib", type=float, default=24.0)
    args = parser.parse_args()
    if args.model and (args.runtime_gib is None or not args.model.is_file()):
        parser.error("--model must exist and requires --runtime-gib")
    for value in (args.runtime_gib, args.reserve_gib, args.target_gib):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("sizes must be finite and nonnegative")
    if args.target_gib == 0:
        parser.error("--target-gib must be positive")
    report = snapshot(args.disk_path)
    status = 0
    if args.model:
        gpu = next((g for g in report["gpus"] if g["index"] == args.gpu), None)
        report["model"] = str(args.model.resolve())
        if gpu is None:
            report["capacity"] = {"fits_estimate": False, "reason": "requested GPU unavailable"}
        else:
            report["capacity"] = capacity(gpu, args.model.stat().st_size / GIB,
                                          args.runtime_gib, args.reserve_gib, args.target_gib)
        if not report["capacity"]["fits_estimate"]:
            status = 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
