#!/usr/bin/env python3
"""cache_eval.py --root DIR --split split.json --out cache_eval.npz [--steps 3,5,8,...]

Endpoint-agreement exit: for each test variant and its nearest banked trajectory b (same step), decodes
final_b + (x0_k - x0_b,k) through the canary's SD_TRAJ_INJECT_DIR queue and scores LPIPS vs the variant's full run,
next to the plain x0 exit at the same step.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import capture


def write_lat(path, a):
    a = np.ascontiguousarray(a, np.float32)
    with open(str(path) + ".tmp", "wb") as f:
        f.write(np.int32(a.ndim).tobytes())
        f.write(np.array(a.shape[::-1], np.int64).tobytes())
        f.write(a.tobytes())
    os.rename(str(path) + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--feats", required=True); ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--fracs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--lib", default="/nvme0n1-disk/tmp/early-exit/build-traj/bin/libstable-diffusion.so")
    ap.add_argument("--extra-env", default="{}")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    split = json.load(open(a.split))
    F = np.load(a.feats, allow_pickle=True)
    pairs = []
    for v in split["test"]:
        m = F["traj"] == v
        if not m.any():
            continue
        z = np.load(root.parent / v / "traj.npz")
        n = len(z["step"]); real = [i for i in range(n) if not z["skipped"][i]]
        for fr in [float(x) for x in a.fracs.split(",")]:
            i = next((r for r in real if (r + 1) / n >= fr and r < n - 1), None)
            if i is None:
                continue
            nk = str(F["nn_key"][m][i]); cd = float(F["cache_d"][m][i])
            if not nk:
                continue
            pairs.append((v, i, nk, cd))
    pairs = sorted(pairs, key=lambda p: p[3])[: a.max_pairs]
    q = root / "_inject"; q.mkdir(exist_ok=True)
    for f in q.iterdir():
        f.unlink()
    for qi, (v, i, nk, cd) in enumerate(pairs):
        zv = np.load(root.parent / v / "traj.npz"); zb = np.load(root.parent / nk / "traj.npz")
        lat = zb["den"][-1].astype(np.float32) + zv["den"][i].astype(np.float32) - zb["den"][i].astype(np.float32)
        write_lat(q / f"q{qi:04d}.lat", lat)
    env0, binary, wd = capture.unit_env()
    env = {**os.environ, **env0, "OMNISERVE_NATIVE_PORT": str(a.port), "OMNISERVE_NATIVE_SECRET": "", "OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM": "",
           "OMNISERVE_ACCESS_LOG": "0", "OMNISERVE_NATIVE_GUARD_JUDGE": "0", "OMNISERVE_NATIVE_FRONTIER_LOG": "",
           "OMNISERVE_NATIVE_VRAM_OWNER": "early-exit-canary", "OMNISERVE_NATIVE_SD_LIB": a.lib,
           "SD_TRAJ_DIR": str(root / "_traj"), "SD_TRAJ_INJECT_DIR": str(q), **json.loads(a.extra_env)}
    (root / "_traj").mkdir(exist_ok=True)
    p = capture.start(binary, wd, env, a.port, open(root / "cache_eval_canary.log", "ab"))
    try:
        r = requests.post(f"http://127.0.0.1:{a.port}/v1/images/generations", timeout=900,
                          json={"prompt": "a gray square", "width": 256, "height": 256, "steps": 2, "seed": 1, "cache": False, "turbo": False})
        print("trigger", r.status_code)
    finally:
        capture.stop(p)
    import lpips, torch
    lp = lpips.LPIPS(net="alex", verbose=False).eval()
    def t(im):
        return torch.from_numpy(np.asarray(im.convert("RGB"), np.float32)).permute(2, 0, 1)[None] / 127.5 - 1
    rows = {k: [] for k in ["traj", "step", "base", "cache_d", "lpips_corr", "lpips_plain", "lpips_base"]}
    for qi, (v, i, nk, cd) in enumerate(pairs):
        f = q / f"q{qi:04d}.rgb"
        if not f.exists():
            continue
        corr = Image.fromarray(capture.read_rgb(f)[..., :3])
        fin = Image.open(root.parent / v / "final.png")
        plain = root.parent / v / f"k{i + 1:02d}.png"
        with torch.no_grad():
            rows["lpips_corr"].append(float(lp(t(corr), t(fin))))
            rows["lpips_plain"].append(float(lp(t(Image.open(plain)), t(fin))) if plain.exists() else np.nan)
            rows["lpips_base"].append(float(lp(t(Image.open(root.parent / nk / "final.png")), t(fin))))
        rows["traj"].append(v); rows["step"].append(i + 1); rows["base"].append(nk); rows["cache_d"].append(cd)
    np.savez(a.out, **{k: np.array(x) for k, x in rows.items()})
    print(len(rows["traj"]), "scored")


if __name__ == "__main__":
    main()
