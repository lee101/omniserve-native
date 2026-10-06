#!/usr/bin/env python3
import argparse, base64, io, json, os, signal, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
FAST = f"{HOME}/models-fast/qwen-image-2.1"
BIN = os.environ.get("SWEEP_BIN", f"{REPO}/build-qwen/omniserve-native")
SD_LIB = os.environ.get("SWEEP_SD_LIB", "/vfast/data/code/stable-diffusion.cpp-master/build/bin/libstable-diffusion.so")
LORA = f"{FAST}/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors"
Q6 = f"{FAST}/Qwen-Image-2.1-viggle-turbo-v0.3-6step-Q6_K.gguf"
Q5 = f"{FAST}/Qwen-Image-2.1-viggle-turbo-v0.3-6step-Q5_K_M.gguf"
N6 = "1.0,0.9375,0.875,0.75,0.5,0.25"
N4 = "1.0,0.875,0.625,0.25"
N8 = "1.0,0.9375,0.875,0.8125,0.75,0.625,0.5,0.25"

BASE_ENV = {
    "OMNISERVE_NATIVE_BIND": "127.0.0.1",
    "OMNISERVE_NATIVE_SLOTS": "1",
    "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
    "OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED": "1",
    "OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": f"{FAST}/qwen-image-2.1-Q4_K_M.gguf",
    "OMNISERVE_NATIVE_SD_LLM": f"{FAST}/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "OMNISERVE_NATIVE_SD_LLM_VISION": f"{FAST}/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    "OMNISERVE_NATIVE_SD_VAE": f"{FAST}/vae/qwen_image_2.1_vae_bf16.safetensors",
    "OMNISERVE_NATIVE_SD_EAGER_LOAD": "1",
    "OMNISERVE_NATIVE_SD_VAE_TILING": "1",
    "OMNISERVE_NATIVE_SD_ZERO_GUIDANCE": "1.0",
    "OMNISERVE_NATIVE_SD_LIB": SD_LIB,
    "OMNISERVE_NATIVE_SD_MIN_FREE_MB": "256",
    "OMNISERVE_NATIVE_SD_NOTCH": "1",
    "OMNISERVE_NATIVE_SD_MIN_STEPS": "0",
    "OMNISERVE_NATIVE_SD_CACHE_MODE": "easycache",
    "OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD": "0.15",
    "OMNISERVE_NATIVE_GUARD": "0",
    "OMNISERVE_NATIVE_GUARD_JUDGE": "0",
}


def turbo_env(nodes=N6, scale="1.0"):
    return {"OMNISERVE_NATIVE_SD_TURBO_NODES": nodes, "OMNISERVE_NATIVE_SD_DEFAULT_LORA": LORA,
            "OMNISERVE_NATIVE_SD_DEFAULT_LORA_SCALE": scale}


def premerged(path, nodes=N6):
    return {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": path, "OMNISERVE_NATIVE_SD_TURBO_NODES": nodes}


def base(tag, steps, **extra):
    return (tag, steps, dict(turbo=False, **extra))


MAIN = [
    base("base30_dense", 30, cache_threshold=0),
    base("base20_dense", 20, cache_threshold=0),
    base("base24_dense", 24, cache_threshold=0),
    base("base30_ec05", 30, cache_threshold=0.05),
    base("base30_ec08", 30, cache_threshold=0.08),
    base("base30_ec10", 30, cache_threshold=0.10),
    base("base30_ec15", 30, cache_threshold=0.15),
    base("base30_ec20", 30, cache_threshold=0.20),
    base("base30_ec30", 30, cache_threshold=0.30),
    base("base24_ec10", 24, cache_threshold=0.10),
    base("base24_ec15", 24, cache_threshold=0.15),
    base("base24_ec20", 24, cache_threshold=0.20),
    base("base20_ec10", 20, cache_threshold=0.10),
    base("base20_ec15", 20, cache_threshold=0.15),
    base("base20_ec20", 20, cache_threshold=0.20),
    base("base16_ec15", 16, cache_threshold=0.15),
    base("base16_ec20", 16, cache_threshold=0.20),
    base("base30_ec15_end50", 30, cache_threshold=0.15, cache_end=0.5),
    base("base30_ec15_end70", 30, cache_threshold=0.15, cache_end=0.7),
    base("base30_ec15_end85", 30, cache_threshold=0.15, cache_end=0.85),
    base("base30_ec15_nonotch", 30, cache_threshold=0.15, notch=False),
    ("turbo6", 3, {}),
    ("turbo6_nonotch", 3, {"notch": False}),
]

SERVERS = [
    ("main", {**turbo_env()}, MAIN),
    ("turbo_nodes4", turbo_env(N4), [("turbo4", 3, {})]),
    ("turbo_nodes8", turbo_env(N8), [("turbo8", 3, {})]),
    ("turbo_lora07", turbo_env(N6, "0.7"), [("turbo6_lora07", 3, {})]),
    ("turbo_lora13", turbo_env(N6, "1.3"), [("turbo6_lora13", 3, {})]),
    ("turbo_lora05", turbo_env(N6, "0.5"), [("turbo6_lora05", 3, {})]),
    ("premerged_q6", premerged(Q6), [("premerged_q6_turbo6", 3, {})]),
    ("premerged_q5", premerged(Q5), [("premerged_q5_turbo6", 3, {})]),
    ("cache_ucache", {"OMNISERVE_NATIVE_SD_CACHE_MODE": "ucache"},
     [base("ucache30_t15", 30, cache_threshold=0.15), base("ucache30_t30", 30, cache_threshold=0.30)]),
    ("cache_taylor2", {"OMNISERVE_NATIVE_SD_CACHE_MODE": "taylorseer", "OMNISERVE_NATIVE_SD_TAYLORSEER_SKIP_INTERVAL": "2"},
     [base("taylor30_skip2", 30, cache_threshold=0.15)]),
    ("cache_taylor3", {"OMNISERVE_NATIVE_SD_CACHE_MODE": "taylorseer", "OMNISERVE_NATIVE_SD_TAYLORSEER_SKIP_INTERVAL": "3",
                       "OMNISERVE_NATIVE_SD_TAYLORSEER_DERIVATIVES": "2"},
     [base("taylor30_skip3_d2", 30, cache_threshold=0.15)]),
    ("cache_spectrum", {"OMNISERVE_NATIVE_SD_CACHE_MODE": "spectrum"}, [base("spectrum30", 30, cache_threshold=0.15)]),
    ("cache_dbcache", {"OMNISERVE_NATIVE_SD_CACHE_MODE": "dbcache"}, [base("dbcache30", 30, cache_threshold=0.15)]),
    ("cache_cachedit", {"OMNISERVE_NATIVE_SD_CACHE_MODE": "cache-dit"}, [base("cachedit30", 30, cache_threshold=0.15)]),
]


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def wait_ready(port, proc, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            post(f"http://127.0.0.1:{port}/v1/images/generations",
                 {"prompt": "warmup", "width": 256, "height": 256, "steps": 4, "seed": 1, "turbo": False,
                  "cache_threshold": 0, "output_format": "webp"}, 600)
            return True
        except Exception:
            time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--servers", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--prompts", default=f"{HERE}/sweep_prompts.json")
    a = ap.parse_args()
    prompts = json.load(open(a.prompts))
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
    w, h = map(int, a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/results.jsonl", "a")
    done = set()
    try:
        for l in open(f"{a.out}/results.jsonl"):
            r = json.loads(l)
            done.add((r["tag"], r["prompt"]))
    except Exception:
        pass
    want = set(a.servers.split(",")) if a.servers else None
    for name, env, variants in SERVERS:
        if want and name not in want:
            continue
        todo = [(t, s, e) for t, s, e in variants if (not a.tags or t in a.tags.split(",")) and any((t, k) not in done for k in prompts)]
        if not todo:
            continue
        full = dict(os.environ)
        for k in list(full):
            if k.startswith("OMNISERVE_NATIVE_SD_") or k == "OMNISERVE_NATIVE_SECRET":
                del full[k]
        full.update(BASE_ENV)
        full.update(env)
        full["OMNISERVE_NATIVE_PORT"] = str(a.port)
        sl = open(f"{a.out}/server_{name}.log", "w")
        proc = subprocess.Popen([BIN, "--port", str(a.port)], env=full, stdout=sl, stderr=sl, start_new_session=True)
        print(f"[{name}] starting pid {proc.pid}", flush=True)
        if not wait_ready(a.port, proc):
            print(f"[{name}] failed to become ready", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            continue
        url = f"http://127.0.0.1:{a.port}/v1/images/generations"
        for tag, steps, extra in todo:
            os.makedirs(f"{a.out}/{tag}", exist_ok=True)
            for pk, pv in prompts.items():
                if (tag, pk) in done:
                    continue
                body = dict(prompt=pv, width=w, height=h, steps=steps, seed=a.seed, guidance_scale=1.0,
                            output_format="png", **extra)
                t0 = time.time()
                try:
                    r = post(url, body)
                except urllib.error.HTTPError as e:
                    print(f"[{name}] {tag} {pk} HTTP {e.code} {e.read()[:200]}", flush=True)
                    continue
                except Exception as e:
                    print(f"[{name}] {tag} {pk} ERR {e}", flush=True)
                    continue
                wall = time.time() - t0
                d = r["data"][0]
                blob = base64.b64decode(d["b64_json"]); open(f"{a.out}/{tag}/{pk}." + ("webp" if blob[8:12] == b"WEBP" else "png"), "wb").write(blob)
                rec = dict(server=name, tag=tag, prompt=pk, steps=steps, extra=extra, wall=round(wall, 2),
                           inf_ms=d.get("inference_time_ms"), cache=r.get("denoiser_cache"))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{name}] {tag} {pk} wall={wall:.1f}s inf={d.get('inference_time_ms')}ms cache={r.get('denoiser_cache')}", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(3)
    print("SWEEP DONE", flush=True)


main()
