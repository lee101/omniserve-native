#!/usr/bin/env python3
"""Bounded Qwen 2.1 trajectory capture through a prod-config canary gateway with the traj-dump sd.cpp lib.

capture.py --jobs jobs.jsonl --out DIR [--port 8795] [--minutes 18]
Each job: {"id","kind":"t2i"|"edit","prompt","seed","width","height","steps","src"?}
Writes DIR/<id>/{traj.npz,final.png,kNN.png}, appends DIR/log.jsonl. Resumable: done ids are skipped.
"""
import argparse, base64, glob, io, json, os, shlex, shutil, signal, subprocess, time
from pathlib import Path

import ctypes

import numpy as np
import requests
from PIL import Image

from features import pooled, traj_rows

UNIT = "omniserve-native-qwen.service"
ACCESS_LOG = "/var/log/omniserve/access.log"


def unit_env():
    raw = subprocess.run(["systemctl", "show", UNIT, "-p", "Environment", "--value"], capture_output=True, text=True).stdout
    env = dict(kv.split("=", 1) for kv in shlex.split(raw) if "=" in kv)
    exe = subprocess.run(["systemctl", "show", UNIT, "-p", "ExecStart", "--value"], capture_output=True, text=True).stdout
    binary = exe.split("path=")[1].split(";")[0].strip()
    wd = subprocess.run(["systemctl", "show", UNIT, "-p", "WorkingDirectory", "--value"], capture_output=True, text=True).stdout.strip()
    return env, binary, wd


def free_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(o.split()[0])


def own_mib(pid):
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in o.splitlines() if l.split(",")[0].strip() == str(pid))


def broker_bg_headroom():
    try:
        return int(requests.get("http://127.0.0.1:8791/v1/gpu/status", timeout=3).json()["ledger"]["headroom_mb"]["background"])
    except Exception:
        return 0


def prod_5xx_since(ts, interactive_only=False):
    n = 0
    try:
        with open(ACCESS_LOG, "rb") as f:
            f.seek(max(0, os.path.getsize(ACCESS_LOG) - 4_000_000))
            for line in f.read().decode(errors="replace").splitlines():
                p = line.split(" ", 1)
                if len(p) == 2 and p[0] >= ts and " status=5" in line and "hdrs=-" not in line:
                    if not interactive_only or " internal=0 " in line:
                        n += 1
    except OSError:
        pass
    return n


def read_rgb(path):
    with open(path, "rb") as f:
        w, h, c = np.frombuffer(f.read(12), np.int32)
        a = np.frombuffer(f.read(), np.uint8).reshape(h, w, c)
    return a


def read_traj(path):
    b = open(path, "rb").read()
    o = 0
    def i32():
        nonlocal o
        v = int(np.frombuffer(b, np.int32, 1, o)[0]); o += 4; return v
    def f32():
        nonlocal o
        v = float(np.frombuffer(b, np.float32, 1, o)[0]); o += 4; return v
    def tensor():
        nonlocal o
        nd = i32()
        shape = tuple(int(v) for v in np.frombuffer(b, np.int64, nd, o)); o += 8 * nd
        n = int(np.prod(shape))
        a = np.frombuffer(b, np.float32, n, o).reshape(shape[::-1]); o += 4 * n
        return a
    assert i32() == 0x4a415254
    n = i32()
    steps, sig, sk, xs, ds = [], [], [], [], []
    for _ in range(n):
        steps.append(i32()); sig.append(f32()); sk.append(i32()); xs.append(tensor()); ds.append(tensor())
    return dict(step=np.array(steps, np.int32), sigma=np.array(sig, np.float32), skipped=np.array(sk, np.int8),
                x=np.stack(xs), den=np.stack(ds))


def to_png(a, path):
    Image.fromarray(a[..., :3] if a.shape[2] == 4 and (a[..., 3] == 255).all() else a).save(path, compress_level=1)


def _child_setup():
    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def start(binary, wd, env, port, logf):
    if f":{port} " in subprocess.run(["ss", "-ltnH"], capture_output=True, text=True).stdout:
        raise RuntimeError(f"port {port} busy")
    p = subprocess.Popen([binary, "--port", str(port)], cwd=wd, env=env, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=_child_setup)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300:
        if p.poll() is not None:
            raise RuntimeError(f"canary exited rc={p.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/status", timeout=2).ok:
                return p
        except requests.RequestException:
            pass
        time.sleep(1)
    stop(p)
    raise RuntimeError("canary not ready")


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(30)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--minutes", type=float, default=18)
    ap.add_argument("--lib", default="/nvme0n1-disk/tmp/early-exit/build-traj/bin/libstable-diffusion.so")
    ap.add_argument("--decode-from", default="1")
    ap.add_argument("--peak-mib", type=int, default=11264)
    ap.add_argument("--extra-env", default="{}")
    ap.add_argument("--max-bg-5xx", type=int, default=10)
    a = ap.parse_args()
    deadline = time.monotonic() + a.minutes * 60
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    tdir = out / "_traj"; shutil.rmtree(tdir, ignore_errors=True); tdir.mkdir()
    done = {json.loads(l)["id"] for l in open(out / "log.jsonl")} if (out / "log.jsonl").exists() else set()
    jobs = [j for j in map(json.loads, open(a.jobs)) if j["id"] not in done]
    if not jobs:
        print("nothing to do"); return
    recent = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 300))
    for _ in range(12):
        fm, bg, n5 = free_mib(), broker_bg_headroom(), prod_5xx_since(recent, True)
        if fm >= a.peak_mib + 4096 and bg >= a.peak_mib + 512 and n5 == 0:
            break
        time.sleep(10)
    else:
        print(f"skip: free {fm} MiB, broker bg headroom {bg} MiB, interactive 5xx last 5 min {n5}"); return
    env0, binary, wd = unit_env()
    env = {**os.environ, **env0, "OMNISERVE_NATIVE_PORT": str(a.port), "OMNISERVE_NATIVE_SECRET": "",
           "OMNISERVE_NATIVE_IMAGE_OVERFLOW_UPSTREAM": "", "OMNISERVE_ACCESS_LOG": "0", "OMNISERVE_NATIVE_GUARD_JUDGE": "0",
           "OMNISERVE_NATIVE_FRONTIER_LOG": "", "OMNISERVE_NATIVE_VRAM_OWNER": "early-exit-canary",
           "OMNISERVE_NATIVE_SD_LIB": a.lib, "SD_TRAJ_DIR": str(tdir), "SD_TRAJ_DECODE_FROM": a.decode_from,
           **json.loads(a.extra_env)}
    started = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    logf = open(out / "canary.log", "ab")
    p = start(binary, wd, env, a.port, logf)
    src_cache = {}
    n_ok = 0
    peak_box = [0]
    import threading
    def sampler():
        while p.poll() is None:
            peak_box[0] = max(peak_box[0], own_mib(p.pid))
            time.sleep(0.5)
    threading.Thread(target=sampler, daemon=True).start()
    try:
        for j in jobs:
            if time.monotonic() > deadline - 60:
                break
            bad, inter = prod_5xx_since(started), prod_5xx_since(started, True)
            if inter >= 1 or bad >= a.max_bg_5xx:
                print(f"ABORT prod 5xx={bad} interactive={inter}"); break
            body = {"prompt": j["prompt"], "width": j["width"], "height": j["height"], "steps": j["steps"], "seed": j["seed"],
                    "cache": False, "output_format": "png", "n": 1}
            path = "generations"
            if j["kind"] == "edit":
                sp = out / j["src"] / "final.png"
                if not sp.exists():
                    continue
                if j["src"] not in src_cache:
                    src_cache[j["src"]] = base64.b64encode(sp.read_bytes()).decode()
                body["image_base64"] = src_cache[j["src"]]
                path = "edits"
            else:
                body["turbo"] = False
            body.update(j.get("body", {}))
            for f in tdir.iterdir():
                f.unlink()
            t0 = time.monotonic()
            try:
                r = requests.post(f"http://127.0.0.1:{a.port}/v1/images/{path}", json=body, timeout=600,
                                  headers={"X-Omniserve-Tier": "background"})
            except requests.RequestException as e:
                print("req fail", j["id"], e); continue
            wall = time.monotonic() - t0
            trajs = sorted(glob.glob(str(tdir / "*.traj")))
            if r.status_code != 200 or len(trajs) != 1:
                print("bad", j["id"], r.status_code, r.text[:200], len(trajs)); continue
            base = trajs[0][:-5]
            jd = out / j["id"]; jd.mkdir(exist_ok=True)
            z = read_traj(trajs[0])
            rows = traj_rows(z)
            z["x"] = z["x"].astype(np.float16); z["den"] = z["den"].astype(np.float16)
            np.savez(jd / "traj.npz", rows=json.dumps(rows), px=pooled(z["x"]), pd=pooled(z["den"]), **z)
            for f in glob.glob(base + ".*.rgb"):
                tag = f[len(base) + 1:-4]
                to_png(read_rgb(f), jd / f"{tag}.png")
            with open(out / "log.jsonl", "a") as lf:
                lf.write(json.dumps({**{k: v for k, v in j.items()}, "wall_s": round(wall, 2), "t": time.time()}) + "\n")
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"{n_ok} done, last {wall:.1f}s", flush=True)
    finally:
        stop(p)
    print(f"captured {n_ok} in run; total {len(done) + n_ok}; canary peak seen {peak_box[0]} MiB")


if __name__ == "__main__":
    main()
