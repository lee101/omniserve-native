"""Prove cache hits complete while the only image/GPU permit is occupied."""

import concurrent.futures
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request


def main() -> int:
    binary = os.environ["OMNISERVE_NATIVE_BIN"]
    stub = os.environ["OMNISERVE_SD_STUB"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
    env.update(
        {
            "OMNISERVE_NATIVE_SD_MODEL": "cache-test.gguf",
            "OMNISERVE_NATIVE_SD_LIB": stub,
            "OMNISERVE_NATIVE_BIND": "127.0.0.1",
            "OMNISERVE_NATIVE_SLOTS": "1",
            "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
            "OMNISERVE_NATIVE_SD_MIN_FREE_MB": "0",
            "OMNISERVE_NATIVE_SD_IMAGE_FORMAT": "png",
        }
    )

    def call(path, payload=None, timeout=5):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}" + path,
            data=json.dumps(payload).encode() if payload else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(
            [binary, "--port", str(port)], env=env, stdout=log, stderr=log
        )
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError("stub server exited")
                try:
                    if call("/status")["diffusion"]["ready"]:
                        break
                except OSError:
                    pass
                time.sleep(0.05)
            else:
                raise TimeoutError("stub server did not become ready")
            payload = {
                "prompt": "cached image",
                "size": "64x64",
                "steps": 2,
                "seed": 42,
                "cache": True,
            }
            prime = call("/v1/images/generations", payload)["data"][0]
            assert not prime["cache"]["hit"]
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                busy = pool.submit(
                    call, "/v1/images/generations", {**payload, "seed": 43}
                )
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    if call("/status")["admission"]["active"] == 1:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("uncached request did not occupy image permit")
                assert not busy.done()
                for _ in range(3):
                    hot = call("/v1/images/generations", payload, timeout=0.75)["data"][
                        0
                    ]
                    assert hot["cache"]["hit"] and hot["b64_json"] == prime["b64_json"]
                    assert not busy.done(), "cache request waited for generation"
                changed = busy.result()["data"][0]
                assert not changed["cache"]["hit"]
                assert changed["b64_json"] != prime["b64_json"]
            print("3 exact cache hits bypass occupied GPU permit; changed seed misses")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
