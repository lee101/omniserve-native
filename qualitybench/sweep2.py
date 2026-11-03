#!/usr/bin/env python3
import argparse, base64, json, os, signal, subprocess, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAST = os.path.expanduser("~/models-fast/qwen-image-2.1")
BIN = f"{REPO}/build-qwen/omniserve-native"
SD_LIB = os.environ.get("SWEEP_SD_LIB", "/vfast/data/code/sdcpp-qwen-prefixkv/build-86/bin/libstable-diffusion.so")
FIX = f"{FAST}/qwen-image-2.1-fix-1.0-comfy.safetensors"
T256 = f"{FAST}/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors"
T128 = f"{FAST}/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
N6 = "1.0,0.9375,0.875,0.75,0.5,0.25"

ENV = {
    "OMNISERVE_NATIVE_BIND": "127.0.0.1", "OMNISERVE_NATIVE_SLOTS": "1", "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
    "OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED": "1",
    "OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": f"{FAST}/qwen-image-2.1-Q4_K_M.gguf",
    "OMNISERVE_NATIVE_SD_LLM": f"{FAST}/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "OMNISERVE_NATIVE_SD_VAE": f"{FAST}/vae/qwen_image_2.1_vae_bf16.safetensors",
    "OMNISERVE_NATIVE_SD_EAGER_LOAD": "1", "OMNISERVE_NATIVE_SD_VAE_TILING": "1", "OMNISERVE_NATIVE_SD_VAE_TILE_X": os.environ.get("SWEEP_TILE", "32"), "OMNISERVE_NATIVE_SD_VAE_TILE_Y": os.environ.get("SWEEP_TILE", "32"), "OMNISERVE_NATIVE_SD_ZERO_GUIDANCE": "1.0",
    "OMNISERVE_NATIVE_SD_LIB": SD_LIB, "OMNISERVE_NATIVE_SD_MIN_FREE_MB": "256", "OMNISERVE_NATIVE_SD_NOTCH": "1",
    "OMNISERVE_NATIVE_SD_MIN_STEPS": "0", "OMNISERVE_NATIVE_SD_CACHE_MODE": "easycache",
    "OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD": "0.15", "OMNISERVE_NATIVE_GUARD": "0", "OMNISERVE_NATIVE_GUARD_JUDGE": "0",
    "OMNISERVE_NATIVE_SD_PARAMS_BACKEND": "te=cpu", "OMNISERVE_NATIVE_SD_TURBO_NODES": N6, "OMNISERVE_NATIVE_SD_TURBO_STACK": "1",
}


def L(*pairs):
    return [{"path": p, "scale": s} for p, s in pairs]


def B(steps, ec, *loras):
    return (steps, dict(turbo=False, cache_threshold=ec, **({"loras": L(*loras)} if loras else {})))


def T(*loras):
    return (3, dict(turbo=True, **({"loras": L(*loras)} if loras else {})))


CONFIGS = {
    "B30_dense": B(30, 0),
    "B30_ec15": B(30, 0.15),
    "B20_ec15": B(20, 0.15),
    "B30_ec15_fix05": B(30, 0.15, (FIX, 0.5)),
    "B30_ec15_fix10": B(30, 0.15, (FIX, 1.0)),
    "B24_ec15_fix07": B(24, 0.15, (FIX, 0.7)),
    "B20_ec15_fix05": B(20, 0.15, (FIX, 0.5)),
    "B20_ec15_fix10": B(20, 0.15, (FIX, 1.0)),
    "T6": T((T256, 1.0)),
    "T6_fix03": T((T256, 1.0), (FIX, 0.3)),
    "T6_fix06": T((T256, 1.0), (FIX, 0.6)),
    "T6_fix10": T((T256, 1.0), (FIX, 1.0)),
    "T6_r128": T((T128, 1.0)),
    "T6_s07_fix05": T((T256, 0.7), (FIX, 0.5)),
    "H16_tl04": B(16, 0, (T256, 0.4)),
    "H12_tl05": B(12, 0, (T256, 0.5)),
    "H16_tl04_fix05": B(16, 0, (T256, 0.4), (FIX, 0.5)),
    "H20_tl03_fix05": B(20, 0, (T256, 0.3), (FIX, 0.5)),
}


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--only", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--prompts", default=f"{HERE}/prompts_v2.json")
    ap.add_argument("--cfg", default="")
    a = ap.parse_args()
    prompts = json.load(open(a.prompts))
    if a.only:
        prompts = {k: prompts[k] for k in a.only.split(",")}
    cfgs = CONFIGS
    if a.cfg:
        cfgs = {k: tuple(v) for k, v in json.load(open(a.cfg)).items()}
    if a.tags:
        cfgs = {k: v for k, v in cfgs.items() if k in a.tags.split(",")}
    w, h = map(int, a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)
    rp = f"{a.out}/results.jsonl"
    done = set()
    if os.path.exists(rp):
        for l in open(rp):
            r = json.loads(l)
            done.add((r["tag"], r["prompt"], r["seed"]))
    log = open(rp, "a")
    full = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE")}
    full.update(ENV)
    full["OMNISERVE_NATIVE_PORT"] = str(a.port)
    proc = subprocess.Popen([BIN, "--port", str(a.port)], env=full, stdout=open(f"{a.out}/server.log", "a"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{a.port}/v1/images/generations"
    end = time.time() + 900
    while True:
        if proc.poll() is not None or time.time() > end:
            print("server failed", flush=True)
            return
        try:
            post(url, {"prompt": "warmup", "width": 256, "height": 256, "steps": 4, "seed": 1, "turbo": False,
                       "cache_threshold": 0, "output_format": "webp"}, 600)
            break
        except Exception:
            time.sleep(3)
    print("ready", flush=True)
    try:
        for seed in map(int, a.seeds.split(",")):
            for pk, pv in prompts.items():
                for tag, (steps, extra) in cfgs.items():
                    if (tag, pk, seed) in done:
                        continue
                    d_ = f"{a.out}/{tag}"
                    os.makedirs(d_, exist_ok=True)
                    body = dict(prompt=pv, width=w, height=h, steps=steps, seed=seed, guidance_scale=1.0,
                                output_format="png", **extra)
                    t0 = time.time()
                    try:
                        r = post(url, body)
                    except urllib.error.HTTPError as e:
                        print(f"{tag} {pk} HTTP {e.code} {e.read()[:200]}", flush=True)
                        continue
                    except Exception as e:
                        print(f"{tag} {pk} ERR {e}", flush=True)
                        continue
                    wall = time.time() - t0
                    dd = r["data"][0]
                    blob = base64.b64decode(dd["b64_json"])
                    ext = "webp" if blob[8:12] == b"WEBP" else "png"
                    open(f"{d_}/{pk}_s{seed}.{ext}", "wb").write(blob)
                    log.write(json.dumps(dict(tag=tag, prompt=pk, seed=seed, steps=steps, wall=round(wall, 2),
                                              inf_ms=dd.get("inference_time_ms"), ext=ext)) + "\n")
                    log.flush()
                    print(f"{tag} {pk} s{seed} {wall:.1f}s", flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
    print("SWEEP DONE", flush=True)


main()
