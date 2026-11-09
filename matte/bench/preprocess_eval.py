"""BIREFNET_GPU_PREPROCESS against the Pillow resize: network mask and final alpha deltas.

    BIREFNET_VRAM_BROKER_URL= OMATTE_LIB=... [BIREFNET_REFINER=...] python3 matte/bench/preprocess_eval.py [image ...]

Gate is the one the compiled model must pass at load (_validate_compiled_model):
mean |delta| of the probability map <= 0.005 and binary agreement >= 0.99.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "workers"))
import birefnet_worker as w


def network_mask(image, gpu):
    w.GPU_PREPROCESS = gpu
    with torch.inference_mode(), w._source_scope(image):
        torch.cuda.synchronize()
        started = time.perf_counter()
        tensor = w._prepare_input(image)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1e3
        return torch.sigmoid(w._unwrap_prediction(w._forward(tensor))).float(), tensor.float(), elapsed


def final_alpha(image, gpu):
    w.GPU_PREPROCESS = gpu
    with w._source_scope(image):
        return w.segment(image, 0.0)[1].astype(np.int32)


def main():
    w.load_model()
    worst_mean, worst_agree = 0.0, 1.0
    for path in sys.argv[1:] or [str(ROOT / "gs2k.webp")]:
        image = Image.open(path).convert("RGB")
        for _ in range(2):
            network_mask(image, True)
        pil, pil_in, pil_ms = min((network_mask(image, False) for _ in range(5)), key=lambda r: r[2])
        gpu, gpu_in, gpu_ms = min((network_mask(image, True) for _ in range(5)), key=lambda r: r[2])
        mean = float((pil - gpu).abs().mean())
        agree = float(((pil >= 0.5) == (gpu >= 0.5)).float().mean())
        alpha = np.abs(final_alpha(image, False) - final_alpha(image, True))
        worst_mean, worst_agree = max(worst_mean, mean), min(worst_agree, agree)
        print(f"{Path(path).name:22s} {image.size} prepare {pil_ms:5.1f} -> {gpu_ms:5.1f} ms | input max delta {float((pil_in - gpu_in).abs().max()):.4f}"
              f" | mask mean delta {mean:.6f} max {float((pil - gpu).abs().max()):.3f} agreement {agree:.6f}"
              f" | alpha/255 mean {alpha.mean():.4f} >1: {(alpha > 1).mean():.4%} >8: {(alpha > 8).mean():.4%} max {alpha.max()}")
    ok = worst_mean <= 0.005 and worst_agree >= 0.99
    print(f"gate (mean <= 0.005, agreement >= 0.99): {'pass' if ok else 'FAIL'} (worst mean {worst_mean:.6f}, agreement {worst_agree:.6f})")
    raise SystemExit(0 if ok else 1)


main()
