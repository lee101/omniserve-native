#!/usr/bin/env python3
"""make_jobs.py --n-t2i N --n-edit M --size S --out jobs.jsonl [--n-var V] [--n-nocache C] [--seed 0] [--prefix q]

Variants: same seed, prompt + small modifier (near-duplicate requests for the cache rule).
No-cache: duplicates with EasyCache off, the reference for the error prod already accepts."""
import argparse, json, random

EDITS = ["make it a snowy winter scene", "turn it into a watercolor painting", "add a small red umbrella",
         "change the time of day to night", "make it look like a pencil sketch", "replace the sky with a dramatic sunset",
         "add falling cherry blossom petals", "make the colors black and white except for red", "turn it into a pixel art game scene",
         "add a cat sitting in the foreground", "make it look like an oil painting by a renaissance master", "add heavy fog",
         "change the season to autumn with orange leaves", "make it a neon cyberpunk scene", "remove all people",
         "add the text \"OPEN\" on a sign", "make it look like a vintage 1970s photo", "turn it into a claymation scene",
         "add rain and wet reflections", "make it underwater with bubbles", "zoom out to show more of the surroundings",
         "make the main subject wear sunglasses", "convert to a flat vector illustration", "add a rainbow in the background"]
SHORT = ["a cat", "a red apple on a table", "mountain lake at dawn", "portrait of an old man", "a robot", "city street at night",
         "a bowl of ramen", "a castle", "sunflower field", "a dragon", "a logo that says \"BREW\"", "a cozy cabin interior",
         "an astronaut riding a horse", "a poster that says \"SUMMER SALE 50% OFF\"", "macro shot of a dew drop on a leaf",
         "a dog wearing a party hat", "anime girl with blue hair", "a minimalist chair", "the word \"HELLO\" in neon", "a forest path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-t2i", type=int, required=True)
    ap.add_argument("--n-edit", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--t2i-steps", type=int, default=30)
    ap.add_argument("--edit-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="q")
    ap.add_argument("--n-var", type=int, default=0)
    ap.add_argument("--n-nocache", type=int, default=0)
    ap.add_argument("--prompts", default="/nvme0n1-disk/code/manifoldgen-farm/shard-1.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines = open(a.prompts).readlines()
    jobs = []
    for i in range(a.n_t2i):
        p = rng.choice(SHORT) if rng.random() < 0.12 else json.loads(rng.choice(lines))["prompt"]
        jobs.append({"id": f"{a.prefix}t{i:05d}", "kind": "t2i", "prompt": p, "seed": rng.randrange(1 << 31),
                     "width": a.size, "height": a.size, "steps": a.t2i_steps})
    t2i = [j["id"] for j in jobs]
    edits = []
    for i in range(a.n_edit):
        edits.append({"id": f"{a.prefix}e{i:05d}", "kind": "edit", "prompt": rng.choice(EDITS), "seed": rng.randrange(1 << 31),
                      "width": a.size, "height": a.size, "steps": a.edit_steps, "src": t2i[i % len(t2i)]})
    k = max(1, len(jobs) // max(1, len(edits))) if edits else 0
    res, ei = [], 0
    for idx, j in enumerate(jobs):
        res.append(j)
        if edits and idx >= 4 and idx % k == 0 and ei < len(edits):
            src_idx = rng.randrange(0, idx - 1)
            edits[ei]["src"] = jobs[src_idx]["id"]
            res.append(edits[ei]); ei += 1
    res += edits[ei:]
    MODS = [", highly detailed", ", soft lighting", ", 35mm photo", ", vibrant colors", ", at dusk", ", minimalist"]
    for i in range(a.n_var):
        b = jobs[rng.randrange(len(jobs))]
        res.append({**b, "id": f"{a.prefix}v{i:05d}", "prompt": b["prompt"] + rng.choice(MODS), "variant_of": b["id"]})
    pool = [j for j in res if j["kind"] in ("t2i", "edit") and "variant_of" not in j]
    for i in range(a.n_nocache):
        b = pool[rng.randrange(len(pool))]
        res.append({**b, "id": f"{a.prefix}n{i:05d}", "body": {"cache_threshold": 0}, "nocache_of": b["id"]})
    with open(a.out, "w") as f:
        for j in res:
            f.write(json.dumps(j) + "\n")
    print(len(res), "jobs")


if __name__ == "__main__":
    main()
