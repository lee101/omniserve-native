#!/usr/bin/env python3
"""Run a bounded local image backend and a command, then stop only that child.

Example: python tools/image_canary.py --model qwen --output /tmp/qwen-run --
  .venv/bin/python tools/image_cache_bench.py --base http://127.0.0.1:8793 ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import signal
import subprocess
import threading
import time
import urllib.request

from shared_host_preflight import snapshot

ROOT = Path(__file__).resolve().parents[1]


def model_environment(
    model: str, root: Path, budget: float, threads: int
) -> dict[str, str]:
    if model == "qwen":
        directory = root / "qwen-edit-2511"
        paths = {
            "SD_DIFFUSION_MODEL": directory / "qwen-image-edit-2511-Q4_K_M.gguf",
            "SD_LLM": directory
            / "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "SD_VAE": directory / "split_files/vae/qwen_image_vae.safetensors",
        }
    else:
        directory = root / "zimage-q4"
        paths = {
            "SD_DIFFUSION_MODEL": directory / "z_image_turbo-Q4_K.gguf",
            "SD_LLM": directory / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            "SD_VAE": directory / "split_files/vae/ae.safetensors",
        }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = {key: str(value) for key, value in paths.items()}
    settings.update(
        {
            "BIND": "127.0.0.1",
            "SLOTS": "1",
            "IMAGE_PERMITS": "1",
            "SD_MAX_VRAM": str(budget),
            "SD_STREAM_LAYERS": "1",
            "SD_PARAMS_BACKEND": "cpu",
            "SD_BACKEND": "diffusion=cuda0,te=cpu,vae=cpu",
            "SD_THREADS": str(threads),
            "SD_MMAP": "1",
            "SD_EAGER_LOAD": "0",
            "SD_IMAGE_FORMAT": "png",
            "SD_MIN_FREE_MB": "2048",
            "SD_REFERENCE_EDIT": "1" if model == "qwen" else "0",
            "SD_MODEL_ARGS": "qwen_image_zero_cond_t=true" if model == "qwen" else "",
            "SD_TELEPORT_CACHE_SIZE": "16",
            "SD_ZERO_GUIDANCE": "1",
            "IMAGE_PREFER_EMBEDDED": "1",
        }
    )
    return {"OMNISERVE_NATIVE_" + key: value for key, value in settings.items()}


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "zimage"), required=True)
    parser.add_argument(
        "--model-root", type=Path, default=Path("/nvme0n1-disk/models/omniserve-native")
    )
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build-edit-perf/omniserve-native"
    )
    parser.add_argument(
        "--sd-library", type=Path, help="isolated candidate libstable-diffusion.so"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--budget-gib", type=float, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--text-backend", choices=("cpu", "cuda0"), default="cpu")
    parser.add_argument("--vae-backend", choices=("cpu", "cuda0"), default="cpu")
    parser.add_argument("--easycache-threshold", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not command
        or not 0 < args.budget_gib <= 8
        or not 1 <= args.threads <= 72
        or not args.timeout > 0
        or not math.isfinite(args.timeout)
    ):
        parser.error(
            "provide a command, 0 < budget <= 8 GiB, 1..72 threads, positive timeout"
        )
    settings = model_environment(
        args.model, args.model_root, args.budget_gib, args.threads
    )
    if not 0 <= args.easycache_threshold <= 1:
        parser.error("EasyCache threshold must be in [0, 1]; zero disables it")
    settings["OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD"] = str(args.easycache_threshold)
    if args.sd_library:
        if not args.sd_library.is_file():
            parser.error("--sd-library must exist")
        settings["OMNISERVE_NATIVE_SD_LIB"] = str(args.sd_library.resolve())
    settings["OMNISERVE_NATIVE_SD_BACKEND"] = (
        f"diffusion=cuda0,te={args.text_backend},vae={args.vae_backend}"
    )
    # Detect collisions before launching: never benchmark another process on this port.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    before = snapshot(str(ROOT))
    gpu = next((g for g in before["gpus"] if g["index"] == 0), None)
    if gpu is None or gpu["free_gib"] < args.budget_gib + 3:
        raise RuntimeError(
            "insufficient current VRAM for budget plus 3 GiB scratch/reserve"
        )
    if before["memory_gib"]["MemAvailable"] < 40:
        raise RuntimeError("need 40 GiB available RAM for CPU parameters and scratch")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "before": before,
        "settings": settings,
        "command": command,
        "binary": str(args.binary.resolve()),
        "complete": False,
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update(settings)
    started = time.monotonic()

    def terminate(_signum: int, _frame: object) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    proc = worker = None
    monitor = None
    monitor_stop = threading.Event()
    report["gpu_process_mib_samples"] = []

    def sample_memory() -> None:
        while not monitor_stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=True,
                )
                for line in result.stdout.splitlines():
                    pid, memory = (value.strip() for value in line.split(","))
                    if int(pid) == proc.pid:
                        report["gpu_process_mib_samples"].append(
                            [time.monotonic() - started, int(memory)]
                        )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            monitor_stop.wait(5)

    try:
        with (args.output / "server.log").open("w") as log:
            proc = subprocess.Popen(
                [str(args.binary.resolve()), "--port", str(args.port)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            monitor = threading.Thread(target=sample_memory, daemon=True)
            monitor.start()
            deadline = time.monotonic() + min(180, args.timeout)
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"canary exited {proc.returncode}; see server.log"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/status", timeout=2
                    ) as response:
                        status = json.load(response)
                    if status["diffusion"]["ready"]:
                        report["status"] = status
                        break
                except OSError:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("canary startup deadline exceeded")
            report["load_seconds"] = time.monotonic() - started
            print(
                json.dumps({"ready": True, "load_seconds": report["load_seconds"]}),
                flush=True,
            )
            worker = subprocess.Popen(command)
            report["returncode"] = worker.wait(
                timeout=max(1, args.timeout - report["load_seconds"])
            )
            report["complete"] = True
            return report["returncode"]
    finally:
        if worker is not None:
            stop(worker)
        if proc is not None:
            stop(proc)
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=5)
        report["after"] = snapshot(str(ROOT))
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "canary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )


if __name__ == "__main__":
    raise SystemExit(main())
