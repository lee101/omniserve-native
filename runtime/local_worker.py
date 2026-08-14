#!/usr/bin/env python3
"""Drain the native OmniServe queue through the same workload registry as RunPod."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time

from runtime.handler import _free_mib, dispatch


def _jobctl(binary: Path, *arguments: str, allow_empty: bool = False) -> dict:
    completed = subprocess.run(
        [str(binary), *arguments], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode != 0:
        if allow_empty and completed.returncode == 4:
            return {"disposition": "empty"}
        raise RuntimeError(completed.stderr.strip() or "omni-job failed")
    return json.loads(completed.stdout)


def _heartbeat(binary: Path, database: Path, key: str, worker: str, lease: int, stop: threading.Event) -> None:
    while not stop.wait(max(1, lease // 3)):
        _jobctl(binary, "heartbeat", str(database), key, worker, str(lease))


def _request(path_value: str) -> dict:
    path = Path(path_value).resolve()
    if not path.is_file() or path.stat().st_size > 1 << 20:
        raise ValueError("queue payload must name a request JSON file no larger than 1 MiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("queued request must be a JSON object")
    return value


def _write_result(directory: Path, key: str, result: dict) -> Path:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=f".{key}.", delete=False) as output:
        json.dump(result, output, separators=(",", ":"))
        output.write("\n")
        temporary = Path(output.name)
    destination = directory / f"{key}.json"
    temporary.replace(destination)
    return destination


def run(args: argparse.Namespace) -> None:
    worker = args.worker or f"{socket.gethostname()}-{os.getpid()}"
    while True:
        available = max(0, _free_mib() - args.reserve_mib)
        claimed = _jobctl(
            args.jobctl,
            "claim",
            str(args.database),
            worker,
            args.gpu,
            str(available),
            str(args.lease_seconds),
            args.kinds,
        )
        if claimed.get("disposition") == "empty":
            if args.once:
                return
            time.sleep(args.poll_seconds)
            continue
        key = claimed["key"]
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat,
            args=(args.jobctl, args.database, key, worker, args.lease_seconds, stop),
            daemon=True,
        )
        heartbeat.start()
        try:
            result_path = _write_result(args.results, key, dispatch(_request(claimed["payload"])))
            _jobctl(args.jobctl, "finish", str(args.database), key, worker, result_path.as_uri())
        except Exception as error:
            retry = int(claimed.get("attempts", 1)) < args.max_attempts
            arguments = ["fail", str(args.database), key, worker, str(error)[:500]]
            if retry:
                arguments.append("retry")
            _jobctl(args.jobctl, *arguments)
        finally:
            stop.set()
            heartbeat.join(timeout=2)
        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("/var/lib/omniserve/jobs.sqlite"))
    parser.add_argument("--jobctl", type=Path, default=Path("/opt/omniserve/bin/omni-job"))
    parser.add_argument("--results", type=Path, default=Path("/var/lib/omniserve/results"))
    parser.add_argument("--worker", default="")
    parser.add_argument("--gpu", default="host:0")
    parser.add_argument("--kinds", default="video-matting,h3-video")
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
