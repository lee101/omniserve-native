"""Render every edit case under every config against a reference-edit lane.

uv run --with pillow python qualitybench/edit/run.py --url http://127.0.0.1:8793 --configs configs.json [--only a,b] [--cases x,y]
Outputs out/<config>/<case>.<ext>, appends out/results.jsonl; existing outputs are skipped (resumable).
"""
import argparse, base64, io, json, math, os, time, urllib.request
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def size_for(w, h, policy, mp):
    if policy == "square":
        return 1024, 1024
    s = math.sqrt(mp * 1024 * 1024 / (w * h))
    snap = lambda v: max(256, min(2048, int(v * s / 64 + 0.5) * 64))
    return snap(w), snap(h)


def ext_of(b):
    return "webp" if b[:4] == b"RIFF" else "png" if b[:4] == b"\x89PNG" else "jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8793")
    ap.add_argument("--configs", default=os.path.join(HERE, "configs.json"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cases = json.load(open(os.path.join(HERE, "cases.json")))
    configs = json.load(open(a.configs))
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only.split(",")}
    if a.cases:
        cases = [c for c in cases if c["id"] in a.cases.split(",")]
    log = open(os.path.join(a.out, "results.jsonl"), "a") if os.path.isdir(a.out) else None
    if not log:
        os.makedirs(a.out)
        log = open(os.path.join(a.out, "results.jsonl"), "a")
    for name, cfg in configs.items():
        cfg = dict(cfg)
        policy, mp = cfg.pop("size_policy", "aspect"), cfg.pop("megapixels", 1.0)
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        for c in cases:
            if any(f.startswith(c["id"] + ".") for f in os.listdir(os.path.join(a.out, name))):
                continue
            src = open(os.path.join(HERE, "src", c["file"]), "rb").read()
            w, h = Image.open(io.BytesIO(src)).size
            ow, oh = size_for(w, h, policy, mp)
            body = {"prompt": c["instruction"], "image_base64": base64.b64encode(src).decode(),
                    "width": ow, "height": oh, "seed": a.seed, "n": 1, **cfg}
            req = urllib.request.Request(a.url + "/v1/images/edits", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t = time.time()
            try:
                resp = json.load(urllib.request.urlopen(req, timeout=900))
                img = base64.b64decode(resp["data"][0]["b64_json"])
                err = None
            except Exception as e:
                img, err, resp = None, str(e)[:300], {}
            wall = time.time() - t
            if img:
                open(os.path.join(a.out, name, f"{c['id']}.{ext_of(img)}"), "wb").write(img)
            rec = {"config": name, "case": c["id"], "wall_s": round(wall, 2), "w": ow, "h": oh, "error": err,
                   "server_ms": resp.get("elapsed_ms") or resp.get("inference_time_ms")}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"{name:24s} {c['id']:18s} {ow}x{oh} wall={wall:6.1f}s {err or ''}", flush=True)


if __name__ == "__main__":
    main()
