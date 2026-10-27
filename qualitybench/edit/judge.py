"""VLM judge for edit outputs (GEdit-Bench style SC/PQ plus preservation), cached in out/judge.jsonl.

GEMINI_API_KEY=... uv run --with pillow python qualitybench/edit/judge.py [--model gemini-3.1-pro-preview] [--only cfg,..]
Prints per-config means: SC (instruction followed), PQ (natural, artifact-free), PR (unedited content kept),
O = sqrt(SC*PQ) as in GEdit-Bench, and mean wall seconds from results.jsonl.
"""
import argparse, base64, concurrent.futures as cf, io, json, math, os, urllib.request
from collections import defaultdict
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = """You are a strict expert judge of AI image edits. Image 1 is the ORIGINAL, image 2 is the EDITED result.
Edit instruction: "{instruction}"

Score each 0-10 (integers, 10 = flawless, be harsh and use the full range):
- sc: semantic compliance. Did the edit do exactly what was asked, completely, with nothing missing? Misspelled or wrong text, partial edits, edits to the wrong object score low.
- pq: perceptual quality of the EDITED image alone: realism/naturalness, sharpness, no artifacts (smears, grid/lattice patterns, melted hands/faces, seams, oversaturation, oversharpening, blur, colour casts).
- pr: preservation. Is everything the instruction did not ask to change kept identical to the original (identity, faces, pose, composition, framing/crop, background, colours, fine detail)? Reframing, zoom, aspect distortion or identity drift score low.
Return JSON: {{"sc": int, "pq": int, "pr": int, "notes": "one short sentence naming the main flaw"}}"""


def jpeg_b64(path, edge=1024):
    im = Image.open(path).convert("RGB")
    im.thumbnail((edge, edge))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def judge(model, src, out, instruction):
    body = {"contents": [{"parts": [
        {"text": PROMPT.format(instruction=instruction)},
        {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64(src)}},
        {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64(out)}}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
            text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))
            d = json.loads(text[text.find("{"):text.rfind("}") + 1])
            return {k: d[k] for k in ("sc", "pq", "pr", "notes")}
        except Exception as e:
            err = str(e)
    return {"error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cases = {c["id"]: c for c in json.load(open(os.path.join(HERE, "cases.json")))}
    cache_path = os.path.join(a.out, "judge.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            if "error" not in r:
                cache[(r["config"], r["case"], r["model"])] = r
    todo = []
    configs = sorted(d for d in os.listdir(a.out) if os.path.isdir(os.path.join(a.out, d)))
    if a.only:
        configs = [c for c in configs if c in a.only.split(",")]
    for cfg in configs:
        for f in os.listdir(os.path.join(a.out, cfg)):
            cid = f.rsplit(".", 1)[0]
            if cid in cases and (cfg, cid, a.model) not in cache:
                todo.append((cfg, cid, os.path.join(a.out, cfg, f)))
    with open(cache_path, "a") as log, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(judge, a.model, os.path.join(HERE, "src", cases[cid]["file"]), path,
                          cases[cid]["instruction"]): (cfg, cid) for cfg, cid, path in todo}
        for fut in cf.as_completed(futs):
            cfg, cid = futs[fut]
            r = {"config": cfg, "case": cid, "model": a.model, **fut.result()}
            log.write(json.dumps(r) + "\n")
            log.flush()
            if "error" not in r:
                cache[(cfg, cid, a.model)] = r
    walls = defaultdict(list)
    rp = os.path.join(a.out, "results.jsonl")
    if os.path.exists(rp):
        for line in open(rp):
            r = json.loads(line)
            if not r.get("error"):
                walls[r["config"]].append(r["wall_s"])
    rows = []
    for cfg in configs:
        js = [v for (c, _, m), v in cache.items() if c == cfg and m == a.model]
        if not js:
            continue
        sc, pq, pr = (sum(j[k] for j in js) / len(js) for k in ("sc", "pq", "pr"))
        o = sum(math.sqrt(j["sc"] * j["pq"]) for j in js) / len(js)
        w = walls[cfg][-len(js):]
        rows.append((cfg, len(js), sc, pq, pr, o, sum(w) / len(w) if w else float("nan")))
    print(f"{'config':26s} {'n':>3s} {'SC':>5s} {'PQ':>5s} {'PR':>5s} {'O':>5s} {'wall':>6s}")
    for r in sorted(rows, key=lambda r: -r[5]):
        print(f"{r[0]:26s} {r[1]:3d} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:5.2f} {r[6]:6.1f}")


if __name__ == "__main__":
    main()
