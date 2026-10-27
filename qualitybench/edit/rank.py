"""Comparative judge: per case, every config's output shown together (shuffled, anonymous) next to the source.

GEMINI_API_KEY=... uv run --with pillow python qualitybench/edit/rank.py --configs a,b,c [--rounds 2]
Absolute single-image scores cluster at 7; side-by-side scoring separates weak edits from strong ones.
Cached in out/rank.jsonl keyed by (case, config set, round).
"""
import argparse, concurrent.futures as cf, json, math, os, random, urllib.request
from collections import defaultdict
from judge import jpeg_b64

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = """You are a strict expert judge of AI image edits. The first image is the ORIGINAL. Then {n} candidate edits follow, labelled {labels}.
Edit instruction: "{instruction}"

Compare the candidates against each other and score each 0-10 (integers, use the full range, the best and worst should differ when they visibly differ):
- sc: does it fully and clearly do what was asked (a timid, barely visible change scores low; wrong text scores low)?
- pq: realism and image quality of the candidate alone (artifacts, melted details, blur, oversharpening, washed out or oversaturated colours).
- pr: everything not mentioned in the instruction kept the same as the original (identity, pose, framing, background, colours).
Return JSON: {{"scores": {{"A": {{"sc": int, "pq": int, "pr": int}}, ...}}, "best": "label", "notes": "one sentence on what separates them"}}"""


def call(model, parts):
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    err = None
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=400))
            text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))
            return json.loads(text[text.find("{"):text.rfind("}") + 1])
        except Exception as e:
            err = str(e)
    return {"error": err}


def rank_case(model, case, configs, rnd, out):
    files = {}
    for cfg in configs:
        d = os.path.join(out, cfg)
        f = next((x for x in os.listdir(d) if x.startswith(case["id"] + ".")), None) if os.path.isdir(d) else None
        if f:
            files[cfg] = os.path.join(d, f)
    order = sorted(files)
    random.Random(f"{case['id']}-{rnd}").shuffle(order)
    labels = [chr(65 + i) for i in range(len(order))]
    parts = [{"text": PROMPT.format(n=len(order), labels=", ".join(labels), instruction=case["instruction"])},
             {"text": "ORIGINAL:"}, {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64(os.path.join(HERE, "src", case["file"]), 768)}}]
    for lab, cfg in zip(labels, order):
        parts += [{"text": f"Candidate {lab}:"}, {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64(files[cfg], 768)}}]
    r = call(model, parts)
    if "error" in r:
        return r
    scores = {cfg: r["scores"].get(lab) for lab, cfg in zip(labels, order)}
    return {"scores": scores, "best": dict(zip(labels, order)).get(r.get("best")), "notes": r.get("notes", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    configs = a.configs.split(",")
    key = ",".join(sorted(configs))
    cases = json.load(open(os.path.join(HERE, "cases.json")))
    path = os.path.join(a.out, "rank.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("set") == key and r.get("model") == a.model and "error" not in r:
                done[(r["case"], r["round"])] = r
    todo = [(c, rnd) for c in cases for rnd in range(a.rounds) if (c["id"], rnd) not in done]
    with open(path, "a") as log, cf.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(rank_case, a.model, c, configs, rnd, a.out): (c, rnd) for c, rnd in todo}
        for fut in cf.as_completed(futs):
            c, rnd = futs[fut]
            r = {"set": key, "model": a.model, "case": c["id"], "round": rnd, **fut.result()}
            log.write(json.dumps(r) + "\n")
            log.flush()
            if "error" not in r:
                done[(c["id"], rnd)] = r
            else:
                print("error", c["id"], r["error"][:200])
    agg, best, per_case = defaultdict(list), defaultdict(int), defaultdict(lambda: defaultdict(list))
    for (cid, _), r in done.items():
        if r.get("best"):
            best[r["best"]] += 1
        for cfg, s in r["scores"].items():
            if s:
                agg[cfg].append(s)
                per_case[cid][cfg].append(math.sqrt(s["sc"] * s["pq"]) * 0.75 + s["pr"] * 0.25)
    print(f"{'config':22s} {'n':>3s} {'SC':>5s} {'PQ':>5s} {'PR':>5s} {'O':>5s} {'O+PR':>5s} {'best':>4s}")
    rows = []
    for cfg, ss in agg.items():
        sc, pq, pr = (sum(s[k] for s in ss) / len(ss) for k in ("sc", "pq", "pr"))
        o = sum(math.sqrt(s["sc"] * s["pq"]) for s in ss) / len(ss)
        rows.append((cfg, len(ss), sc, pq, pr, o, 0.75 * o + 0.25 * pr, best[cfg]))
    for r in sorted(rows, key=lambda r: -r[6]):
        print(f"{r[0]:22s} {r[1]:3d} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:5.2f} {r[6]:5.2f} {r[7]:4d}")
    print("\nper case (0.75*O + 0.25*PR):")
    cfgs = [r[0] for r in sorted(rows, key=lambda r: -r[6])]
    print("case".ljust(17) + "".join(c[:11].rjust(12) for c in cfgs))
    for cid, d in per_case.items():
        print(cid.ljust(17) + "".join((f"{sum(d[c]) / len(d[c]):.1f}" if d[c] else "-").rjust(12) for c in cfgs))


if __name__ == "__main__":
    main()
