"""RunPod serverless job that bakes the Qwen multitenant weights into an image in-datacenter.

The build host's uplink is ~1-2 MB/s, so a 42 GB weight push from it is infeasible. This
handler runs on a throwaway serverless worker (base image ghcr.io/lee101/omniserve-native:ra2,
which already has huggingface_hub + hf_transfer): it downloads the pinned files, verifies
sha256, packs one gzip layer per file and `crane append`s them onto the base, pushing to
ghcr. Delivered via env HANDLER_B64; job input: {"token": <ghcr token>, "target": <ref>}.
"""
import hashlib, json, os, shutil, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import runpod

BASE = "ghcr.io/lee101/omniserve-native:ra2"
FILES = [
    ("edit", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", "qwen_2.5_vl_7b.safetensors",
     "cfafd739459bc86257397259f612a9aee88e5b98e85b5c0d0d1717e898b3463a"),
    ("edit", "unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf", "qwen-image-edit-2511-Q4_K_M.gguf",
     "8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723"),
    ("edit", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors",
     "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
    ("ra2", "Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
     "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
    ("ra2", "netwrck/ra2", "ra2-dit-q4_k_m.gguf", "qwen-image-2.1-Q4_K_M.gguf",
     "833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8"),
    ("ra2", "Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
     "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"),
    ("ra2", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF", "vae/qwen_image_2.1_vae_bf16.safetensors",
     "qwen_image_2.1_vae_bf16.safetensors", "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
]
WORK = "/work"
log = []


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    log.append(line)
    print(line, flush=True)


def sh(cmd, **kw):
    started = time.monotonic()
    out = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    say(f"$ {cmd[:160]} -> {out.returncode} {time.monotonic() - started:.1f}s {(out.stdout + out.stderr)[-400:]}")
    if out.returncode:
        raise RuntimeError(cmd[:120])
    return out.stdout


def fetch(item):
    task, repo, name, dest, sha = item
    from huggingface_hub import hf_hub_download
    started = time.monotonic()
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=f"{WORK}/dl")
    target = f"{WORK}/models/{task}/{dest}"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.replace(path, target)
    fetched = time.monotonic() - started
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        while chunk := handle.read(64 << 20):
            digest.update(chunk)
    ok = digest.hexdigest() == sha
    say(f"{dest} {os.path.getsize(target) / 1e9:.2f} GB fetched {fetched:.0f}s sha_ok={ok}")
    if not ok:
        raise RuntimeError(f"sha mismatch {dest}")
    layer = f"{WORK}/layer-{task}-{dest}.tar.gz"
    sh(f"tar -C {WORK} --owner=0 --group=0 --numeric-owner -cf - models/{task}/{dest} | pigz -1 -p 8 > {layer}")
    os.remove(target)
    return layer


def handler(job):
    inp = job["input"]
    started = time.monotonic()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.pop("HF_HOME", None)
    try:
        shutil.rmtree(WORK, ignore_errors=True)
        os.makedirs(WORK)
        sh("df -h / | tail -1; nproc; (command -v pigz || (apt-get update -qq && apt-get install -y -qq pigz)) >/dev/null")
        urllib.request.urlretrieve("https://github.com/google/go-containerregistry/releases/download/v0.20.3/"
                                   "go-containerregistry_Linux_x86_64.tar.gz", f"{WORK}/gcr.tgz")
        sh(f"tar -C {WORK} -xzf {WORK}/gcr.tgz crane")
        with ThreadPoolExecutor(3) as pool:
            layers = list(pool.map(fetch, FILES))
        sh(f"{WORK}/crane auth login ghcr.io -u lee101 --password-stdin", input=inp["token"])
        args = " ".join(f"-f {layer}" for layer in layers)
        sh(f"{WORK}/crane append -b {BASE} {args} -t {inp['target']}")
        digest = sh(f"{WORK}/crane digest {inp['target']}").strip()
        return {"ok": True, "digest": digest, "s": round(time.monotonic() - started), "log": log[-40:]}
    except Exception as error:
        return {"ok": False, "error": str(error), "log": log[-40:]}
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


runpod.serverless.start({"handler": handler})
