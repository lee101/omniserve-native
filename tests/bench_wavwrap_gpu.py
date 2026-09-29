"""Measures whether a GPU helps PCM16->WAV wrapping.

The wrap is one header write plus one contiguous copy, so the CPU path runs at
memcpy speed. A GPU round trip must pay PCIe upload + download for the same
bytes. This script times both end-to-end on identical payloads and prints the
comparison; it exits 0 either way (measurement, not assertion).
"""

import os
import sys
import time

import torch


def gpu_wrap(pcm: torch.Tensor, rate: int) -> torch.Tensor:
    n = pcm.numel()
    out = torch.empty(44 + n, dtype=torch.uint8, device="cuda")
    out[:4] = torch.frombuffer(b"RIFF", dtype=torch.uint8)
    out[4:8] = torch.tensor(list((36 + 2 * n).to_bytes(4, "little")), dtype=torch.uint8)
    out[8:12] = torch.frombuffer(b"WAVE", dtype=torch.uint8)
    out[12:16] = torch.frombuffer(b"fmt ", dtype=torch.uint8)
    out[16:20] = torch.tensor([16, 0, 0, 0], dtype=torch.uint8)
    out[20:22] = torch.tensor([1, 0], dtype=torch.uint8)
    out[22:24] = torch.tensor([1, 0], dtype=torch.uint8)
    out[24:28] = torch.tensor(list(rate.to_bytes(4, "little")), dtype=torch.uint8)
    out[28:32] = torch.tensor(list((rate * 2).to_bytes(4, "little")), dtype=torch.uint8)
    out[32:34] = torch.tensor([2, 0], dtype=torch.uint8)
    out[34:36] = torch.tensor([16, 0], dtype=torch.uint8)
    out[36:40] = torch.frombuffer(b"data", dtype=torch.uint8)
    out[40:44] = torch.tensor(list((2 * n).to_bytes(4, "little")), dtype=torch.uint8)
    out[44:] = pcm.view(torch.uint8) if pcm.dtype == torch.int16 else pcm
    return out


def main() -> int:
    path = os.environ.get("WAVWRAP_PCM")
    rate = 24000
    if path:
        host = torch.from_file(path, shared=False, dtype=torch.int16,
                               size=os.path.getsize(path) // 2)
    else:
        host = torch.randint(-32768, 32767, (128 * 1024 * 1024,), dtype=torch.int16)

    mib = host.numel() * 2 / (1 << 20)
    print(f"payload: {mib:.0f} MiB PCM16 @ {rate} Hz")

    start = time.perf_counter()
    wrapped = gpu_wrap(host.to("cuda", non_blocking=False), rate)
    torch.cuda.synchronize()
    result = wrapped.cpu()
    gpu_s = time.perf_counter() - start
    assert len(result) == 44 + host.numel() * 2
    print(f"CUDA upload+wrap+download   {mib / gpu_s:8.0f} MiB/s "
          f"(includes PCIe both ways)")

    cpu_start = time.perf_counter()
    out = bytearray(44 + host.numel() * 2)
    out[44:] = host.numpy().tobytes()
    cpu_s = time.perf_counter() - cpu_start
    assert len(out) == 44 + host.numel() * 2
    print(f"CPU memcpy (numpy baseline) {mib / cpu_s:8.0f} MiB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
