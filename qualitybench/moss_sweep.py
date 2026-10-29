#!/usr/bin/env python3
"""MOSS-Image acceleration sweep (SFW and NSFW families), resumable.

Each server group is one omniserve-native process (model + optional default LoRA); its variants are
request grids (steps / cfg / EasyCache) sent to that resident server, so timings are steady state
(a discarded warmup render precedes them). Results: <out>/results.jsonl and <out>/<family>/<tag>/<prompt>.webp.

    uv run qualitybench/moss_sweep.py --out results/moss-20261029 [--servers sfw_4step] [--only sign,portrait]
"""
import argparse, base64, json, os, signal, subprocess, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
M = os.environ.get("MOSS_DIR", "/vfast/data/models/moss")
BIN = os.environ.get("SWEEP_BIN", "/vfast/data/code/omniserve-native-moss-build/omniserve-native")
SD_LIB = os.environ.get("SWEEP_SD_LIB", "/vfast/data/code/sdcpp-moss/build/bin/libstable-diffusion.so")
DENSE = f"{M}/sft/MOSS-Image-SFT-Q4_K.gguf"
DENSE_Q8 = f"{M}/sft/MOSS-Image-SFT-Q8_0.gguf"
STEP4 = f"{M}/sft4/MOSS-Image-SFT-4step-v2.0-merged-Q4_0.gguf"
NSFW_LORA = f"{M}/lora/MOSS-Image-NSFW-Uncensored-LoRA-F16.gguf"
NSFW_LORA_ALT = f"{M}/lora/MOSS-Image-NSFW-Uncensored-LoRA-Alternative-1750-F16.gguf"

BASE_ENV = {
    "OMNISERVE_NATIVE_BIND": "127.0.0.1",
    "OMNISERVE_NATIVE_SLOTS": "1",
    "OMNISERVE_NATIVE_IMAGE_PERMITS": "1",
    "OMNISERVE_NATIVE_IMAGE_PREFER_EMBEDDED": "1",
    "OMNISERVE_NATIVE_GUARD": "0",
    "OMNISERVE_NATIVE_GUARD_JUDGE": "0",
    "OMNISERVE_NATIVE_SD_LIB": SD_LIB,
    "OMNISERVE_NATIVE_SD_LLM": f"{M}/te/Qwen3.5-9B-Q8_0.gguf",
    "OMNISERVE_NATIVE_SD_VAE": f"{M}/vae/flux2-vae.safetensors",
    "OMNISERVE_NATIVE_SD_TOKENIZER": f"{M}/tok/tokenizer.json",
    "OMNISERVE_NATIVE_SD_ZERO_GUIDANCE": "1.0",
    "OMNISERVE_NATIVE_SD_MIN_STEPS": "0",
    "OMNISERVE_NATIVE_SD_EAGER_LOAD": "1",
    "OMNISERVE_NATIVE_SD_MIN_FREE_MB": "256",
    "OMNISERVE_NATIVE_SD_CACHE_MODE": "easycache",
    "OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD": "0",
}


def lora_env(path, scale):
    return {"OMNISERVE_NATIVE_SD_DEFAULT_LORA": path, "OMNISERVE_NATIVE_SD_DEFAULT_LORA_SCALE": str(scale),
            "OMNISERVE_NATIVE_SD_DEFAULT_LORA_ALWAYS": "1"}


def dense_variants():
    return [("d50", 50, dict(guidance_scale=3.0, cache_threshold=0)),
            ("d30", 30, dict(guidance_scale=3.0, cache_threshold=0)),
            ("d20", 20, dict(guidance_scale=3.0, cache_threshold=0)),
            ("d30_ec15", 30, dict(guidance_scale=3.0, cache_threshold=0.15)),
            ("d30_ec30", 30, dict(guidance_scale=3.0, cache_threshold=0.30))]


def step4_variants():
    return [("s4", 4, dict(guidance_scale=1.0, cache_threshold=0)),
            ("s6", 6, dict(guidance_scale=1.0, cache_threshold=0)),
            ("s8", 8, dict(guidance_scale=1.0, cache_threshold=0)),
            ("s4_cfg15", 4, dict(guidance_scale=1.5, cache_threshold=0))]


# name, family, env, variants
SERVERS = [
    ("sfw_4step", "sfw", {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": STEP4}, step4_variants()),
    ("sfw_dense", "sfw", {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": DENSE}, dense_variants()),
    ("nsfw_4step_s08", "nsfw", {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": STEP4, **lora_env(NSFW_LORA, 0.8)},
     [("s4_l08", 4, dict(guidance_scale=1.0, cache_threshold=0)), ("s6_l08", 6, dict(guidance_scale=1.0, cache_threshold=0))]),
    ("nsfw_4step_s10", "nsfw", {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": STEP4, **lora_env(NSFW_LORA, 1.0)},
     [("s4_l10", 4, dict(guidance_scale=1.0, cache_threshold=0))]),
    ("nsfw_4step_s12", "nsfw", {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": STEP4, **lora_env(NSFW_LORA, 1.2)},
     [("s4_l12", 4, dict(guidance_scale=1.0, cache_threshold=0))]),
    ("nsfw_dense", "nsfw", {"OMNISERVE_NATIVE_SD_DIFFUSION_MODEL": DENSE, **lora_env(NSFW_LORA, 0.9)},
     [t for t in dense_variants() if t[0] in ("d50", "d30", "d20", "d30_ec15")]),
]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def wait_ready(port, proc, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).read()
            return True
        except Exception:
            time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--seeds", default="0", help="comma list; every (prompt, seed) is rendered")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--servers", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    seeds = [int(s) for s in a.seeds.split(",")]
    prompts = {f: json.load(open(f"{HERE}/moss_prompts_{f}.json")) for f in ("sfw", "nsfw")}
    if a.only:
        keep = set(a.only.split(","))
        prompts = {f: {k: v for k, v in p.items() if k in keep} for f, p in prompts.items()}
    os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/results.jsonl", "a")
    done = set()
    for l in open(f"{a.out}/results.jsonl"):
        r = json.loads(l)
        done.add((r["family"], r["tag"], r["prompt"], r["seed"]))
    want = set(a.servers.split(",")) if a.servers else None
    tags = set(a.tags.split(",")) if a.tags else None
    for name, family, env, variants in SERVERS:
        if want and name not in want:
            continue
        pr = prompts[family]
        todo = [(t, s, e) for t, s, e in variants if (not tags or t in tags)
                and any((family, t, k, sd) not in done for k in pr for sd in seeds)]
        if not todo:
            continue
        full = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_NATIVE_")}
        full.update(BASE_ENV)
        full.update(env)
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
        try:  # discarded warmup: graph build, compute buffers, first-use weight staging
            post(url, dict(prompt="warmup", width=w, height=h, steps=2, seed=1, guidance_scale=1.0, output_format="png"))
        except Exception as e:
            print(f"[{name}] warmup failed: {e}", flush=True)
        for tag, steps, extra in todo:
            os.makedirs(f"{a.out}/{family}/{tag}", exist_ok=True)
            for pk, pv in pr.items():
                for sd in seeds:
                    if (family, tag, pk, sd) in done:
                        continue
                    body = dict(prompt=pv, width=w, height=h, steps=steps, seed=sd, output_format="png", **extra)
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
                    blob = base64.b64decode(d["b64_json"])
                    ext = "webp" if blob[8:12] == b"WEBP" else "png"
                    path = f"{a.out}/{family}/{tag}/{pk}_s{sd}.{ext}"
                    open(path, "wb").write(blob)
                    rec = dict(server=name, family=family, tag=tag, prompt=pk, seed=sd, steps=steps, extra=extra,
                               wall=round(wall, 2), inf_ms=d.get("inference_time_ms"), cache=r.get("denoiser_cache"),
                               file=os.path.relpath(path, a.out))
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"[{name}] {tag} {pk} s{sd} wall={wall:.1f}s inf={d.get('inference_time_ms')}ms", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(5)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
