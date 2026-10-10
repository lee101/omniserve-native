#!/usr/bin/env python3
"""make_split.py DIR --out split.json [--test-frac 0.25] : grouped train/test split.

Edits are grouped with their source render. Near-duplicate variants go to test with their base forced into train
(the bank then holds the earlier request, as in serving). No-cache reference runs are excluded.
"""
import argparse, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.25); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    group, variants, forced_train = {}, [], set()
    for root in a.dirs:
        name = Path(root).name
        for l in open(Path(root) / "log.jsonl"):
            m = json.loads(l)
            k = f"{name}/{m['id']}"
            if "nocache_of" in m or not m["id"].startswith("q"):
                continue
            if "variant_of" in m:
                variants.append(k); forced_train.add(f"{name}/{m['variant_of']}")
                continue
            group[k] = f"{name}/{m['src']}" if m["kind"] == "edit" else k
    forced_train = {group.get(b, b) for b in forced_train}
    gids = sorted(set(group.values()))
    rng.shuffle(gids)
    test_g = set(g for g in gids[: int(len(gids) * a.test_frac)] if g not in forced_train)
    train = [k for k, g in group.items() if g not in test_g]
    test = [k for k, g in group.items() if g in test_g] + variants
    json.dump({"train": train, "test": test, "group": {**group, **{v: v for v in variants}}, "variants": variants},
              open(a.out, "w"), indent=0)
    print(len(train), "train", len(test), "test", len(variants), "variants")


if __name__ == "__main__":
    main()
