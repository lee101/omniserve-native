import os, shutil, subprocess, sys, time, urllib.request

import runpod

WORK = "/bakework"
log = []


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    log.append(line)
    print(line, flush=True)


def sh(cmd, **kw):
    started = time.monotonic()
    out = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    say(f"$ {cmd[:140]} -> {out.returncode} {time.monotonic() - started:.1f}s {(out.stdout + out.stderr)[-300:]}")
    if out.returncode:
        raise RuntimeError(cmd[:120])
    return out.stdout


H3_FETCH = r'''
import os, sys, time
sys.path.insert(0, "/src")
os.chdir("/src")
from h3_model_profile import resolve_model_profile
from weights import ensure_weights
p = resolve_model_profile(None)
lazy = os.getenv("H3_LAZY_REF2VA", "").lower() in {"1", "true", "yes", "on"}
face = p.has("face_refine") and os.getenv("H3_FACE_REFINE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
t = time.time()
got = ensure_weights(include_ref2va=p.has("ref2va") and not lazy, include_face_refine=face, include_turbo=p.has("turbo"), quant=os.getenv("H3_QUANT"))
print("profile", p.name, sorted(p.models), "files", len(got), "s", round(time.time() - t))
'''

PIXAL_FETCH = r'''
import os, sys, time
sys.path.insert(0, "/src")
os.environ["HF_HOME"] = "/hf"
import subprocess
if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"]).returncode == 0:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
else:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
import torch
from huggingface_hub import snapshot_download
from weights import MODELS
t = time.time()
for repo, kwargs in MODELS:
    snapshot_download(repo, **kwargs)
naf = os.path.join(torch.hub.get_dir(), "checkpoints", "naf_release.pth")
os.makedirs(os.path.dirname(naf), exist_ok=True)
if not os.path.exists(naf):
    torch.hub.download_url_to_file("https://github.com/valeoai/NAF/releases/download/model/naf_release.pth", naf)
print("hub", torch.hub.get_dir(), "s", round(time.time() - t))
'''


def handler(job):
    inp = job["input"]
    started = time.monotonic()
    mode = inp["mode"]
    roots = {"h3": ["weights"], "pixal": ["hf", "root/.cache/torch/hub/checkpoints"]}[mode]
    try:
        shutil.rmtree(WORK, ignore_errors=True)
        os.makedirs(WORK)
        sh("df -h / | tail -1; nproc; (command -v pigz || (apt-get update -qq && apt-get install -y -qq pigz)) >/dev/null")
        urllib.request.urlretrieve("https://github.com/google/go-containerregistry/releases/download/v0.20.3/"
                                   "go-containerregistry_Linux_x86_64.tar.gz", f"{WORK}/gcr.tgz")
        sh(f"tar -C {WORK} -xzf {WORK}/gcr.tgz crane")
        script = f"{WORK}/fetch.py"
        with open(script, "w") as handle:
            handle.write(H3_FETCH if mode == "h3" else PIXAL_FETCH)
        sh(f"python -u {script}")
        sh("du -sh " + " ".join("/" + r for r in roots) + " 2>/dev/null; df -h / | tail -1")
        big = [p for p in sh("find " + " ".join("/" + r for r in roots) + " -type f -size +1G").split() if p]
        excl = f"{WORK}/exclude.txt"
        with open(excl, "w") as handle:
            handle.write("".join(p.lstrip("/") + "\n" for p in big))
        layers = [f"{WORK}/layer-00-small.tar.gz"]
        sh(f"tar -C / --owner=0 --group=0 --numeric-owner -X {excl} -cf - {' '.join(roots)} | pigz -1 > {layers[0]}")
        for index, path in enumerate(big, 1):
            layer = f"{WORK}/layer-{index:02d}.tar.gz"
            sh(f"tar -C / --owner=0 --group=0 --numeric-owner -cf - {path.lstrip('/')} | pigz -1 > {layer} && rm -f {path}")
            layers.append(layer)
        sh(f"{WORK}/crane auth login ghcr.io -u lee101 --password-stdin", input=inp["token"])
        sh(f"{WORK}/crane append -b {inp['base']} " + " ".join(f"-f {layer}" for layer in layers) + f" -t {inp['target']}")
        for env in inp.get("env", []):
            sh(f"{WORK}/crane mutate {inp['target']} --env {env} -t {inp['target']}")
        digest = sh(f"{WORK}/crane digest {inp['target']}").strip()
        sizes = sh(f"du -ch {WORK}/layer-*.tar.gz | tail -1").strip()
        return {"ok": True, "digest": digest, "layers": len(layers), "size": sizes, "s": round(time.monotonic() - started), "log": log[-30:]}
    except Exception as error:
        return {"ok": False, "error": str(error), "log": log[-30:]}
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


runpod.serverless.start({"handler": handler})
