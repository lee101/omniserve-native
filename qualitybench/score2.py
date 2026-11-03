#!/usr/bin/env python3
import collections, json, os, sys, urllib.request
import numpy as np, torch, lpips, open_clip
from PIL import Image

out = sys.argv[1]
REF = os.environ.get("REF", "B30_dense")
prompts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_v2.json")))
rows = [json.loads(l) for l in open(f"{out}/results.jsonl")]
head = os.path.expanduser("~/.cache/aesthetic_l14.pth")
if not os.path.exists(head):
    urllib.request.urlretrieve("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth", head)
torch.set_num_threads(max(1, os.cpu_count() // 2))
net = lpips.LPIPS(net="alex", verbose=False)
model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-L-14")
model.eval()
lin = torch.nn.Linear(768, 1)
lin.load_state_dict(torch.load(head, map_location="cpu"))


def path(r, tag=None):
    return f"{out}/{tag or r['tag']}/{r['prompt']}_s{r['seed']}.{r['ext']}"


refs = {(r["prompt"], r["seed"]): r for r in rows if r["tag"] == REF}
per = []
with torch.no_grad():
    for r in rows:
        im = Image.open(path(r)).convert("RGB")
        e = model.encode_image(pre(im)[None])
        e = e / e.norm(dim=-1, keepdim=True)
        aes = float(lin(e)[0, 0])
        tx = model.encode_text(tok([prompts[r["prompt"]]]))
        tx = tx / tx.norm(dim=-1, keepdim=True)
        clip = float((e * tx).sum())
        g = np.asarray(im.convert("L"), float)
        sharp = float((g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).var())
        lp = None
        rr = refs.get((r["prompt"], r["seed"]))
        if rr and r["tag"] != REF:
            f = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1).float()[None] / 127.5 - 1
            lp = float(net(f(path(r)), f(path(rr))))
        per.append(dict(tag=r["tag"], prompt=r["prompt"], seed=r["seed"], s=(r["inf_ms"] or 0) / 1000, aes=aes, clip=clip, sharp=sharp, lpips=lp))
json.dump(per, open(f"{out}/scores.json", "w"))
agg = collections.defaultdict(list)
for p in per:
    agg[p["tag"]].append(p)
m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"{'tag':22s} {'n':>3s} {'sec':>6s} {'aes':>6s} {'clip':>6s} {'sharp':>7s} {'lpips':>6s}")
for t, v in sorted(agg.items(), key=lambda kv: m([x["s"] for x in kv[1]])):
    print(f"{t:22s} {len(v):3d} {m([x['s'] for x in v]):6.1f} {m([x['aes'] for x in v]):6.2f} {m([x['clip'] for x in v]):6.3f} {m([x['sharp'] for x in v]):7.0f} {m([x['lpips'] for x in v if x['lpips'] is not None]):6.3f}")
cat = collections.defaultdict(lambda: collections.defaultdict(list))
for p in per:
    cat[p["tag"]][p["prompt"].split("_")[0]].append(p["aes"])
print("\naesthetic by category (" + ",".join(sorted({c for d in cat.values() for c in d})) + ")")
for t in agg:
    print(f"{t:22s} " + " ".join(f"{c}={m(v):.2f}" for c, v in sorted(cat[t].items())))
