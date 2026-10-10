import sys, os, numpy as np, torch, lpips
from PIL import Image
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); ref = sys.argv[1]
net = lpips.LPIPS(net="alex", verbose=False)
def load(p): return torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2,0,1).float()[None]/127.5-1
ks = ["fox2","trio","duo","quad"]
print("tag".ljust(16), *[k.ljust(7) for k in ks])
for t in sys.argv[2:]:
    v = []
    for k in ks:
        with torch.no_grad(): v.append(net(load(f"{out}/{ref}/{k}.png"), load(f"{out}/{t}/{k}.png")).item())
    print(t.ljust(16), *[f"{x:.3f}".ljust(7) for x in v])
