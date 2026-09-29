#!/usr/bin/env python3
"""Compare CPU thread counts in sequential, isolated native canaries."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def call(base: str, path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(base + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json", "X-Omniserve-Internal": "local"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 8, 16])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    prompts = {
        "short": "List the first ten square numbers. Answer with numbers only.",
        "long": "Read the following notes. " + "The red robot is named Copper. The blue robot is named Sky. " * 32
                + "What is the red robot named? Answer with its name only.",
        "code": "Write a Python function that returns the larger of two numbers. Only output the function.",
    }
    rows = []
    for threads in args.threads:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
        env.update({"OMNISERVE_NATIVE_LLM_GGUF": str(args.model.resolve()),
                    "OMNISERVE_NATIVE_NGL": "0", "OMNISERVE_NATIVE_CTX": "4096",
                    "OMNISERVE_NATIVE_LLM_CONTEXTS": "1", "OMNISERVE_NATIVE_SLOTS": "1",
                    "OMNISERVE_NATIVE_BATCH": "512", "OMNISERVE_NATIVE_UBATCH": "128",
                    "OMNISERVE_NATIVE_KV_TYPE": "q8_0", "OMNISERVE_NATIVE_BIND": "127.0.0.1",
                    "OMNISERVE_NATIVE_LLM_THREADS": str(threads),
                    "OMNISERVE_NATIVE_LLM_THREADS_BATCH": str(threads)})
        with (args.output / f"threads-{threads}.log").open("w") as log:
            proc = subprocess.Popen([str(args.binary.resolve()), "--port", str(args.port)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"canary exited: {proc.returncode}")
                    try:
                        status = call(base, "/status")
                        if status["llm"]["ready"]:
                            break
                    except OSError:
                        pass
                    time.sleep(0.2)
                else:
                    raise TimeoutError("canary startup timed out")
                for name, prompt in prompts.items():
                    for repeat in range(3):
                        start = time.perf_counter()
                        result = call(base, "/v1/chat/completions", {
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 48, "temperature": 0, "seed": 42})
                        row = {"threads": threads, "case": name, "repeat": repeat,
                               "wall_ms": (time.perf_counter() - start) * 1000,
                               "usage": result.get("usage"),
                               "text": result["choices"][0]["message"]["content"]}
                        rows.append(row)
                        (args.output / "report.json").write_text(json.dumps(rows, indent=2))
                        print(json.dumps(row), flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
