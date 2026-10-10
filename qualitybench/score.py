import sys, re, glob, os, collections, numpy as np, torch, lpips
from PIL import Image
ref = sys.argv[1]; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
net = lpips.LPIPS(net="alex", verbose=False)
tim = collections.defaultdict(dict)
for l in open(f"{out}/frontier.log"):
    m = re.match(r"(\S+) (\S+) wall=([\d.]+)s", l)
    if m: tim[m[1]][m[2]] = float(m[3])
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2,0,1).float()[None]/127.5-1
rows = []
for tag in tim:
    if tag == ref: continue
    L, P, T = [], [], []
    for k, t in tim[tag].items():
        a, b = f"{out}/{ref}/{k}.png", f"{out}/{tag}/{k}.png"
        if not (os.path.exists(a) and os.path.exists(b)): continue
        x, y = load(a), load(b)
        with torch.no_grad(): L.append(net(x, y).item())
        P.append(10*np.log10(4/(((x-y)**2).mean().item()+1e-9))); T.append(t)
    if L: rows.append((tag, np.mean(T), np.mean(L), np.mean(P), len(L)))
rt = np.mean(list(tim[ref].values()))
print(f"ref {ref} {rt:.1f}s")
for r in sorted(rows, key=lambda r: r[1]): print(f"{r[0]:18s} {r[1]:5.1f}s  lpips={r[2]:.3f}  psnr={r[3]:.1f}  n={r[4]}")
