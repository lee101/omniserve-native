import io, os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "workers"))
os.environ.setdefault("OMATTE_LIB", os.path.join(ROOT, "..", "..", "build-cuda", "libomatte.so"))
import torch
from PIL import Image
import birefnet_worker as w

w.load_model()
img = Image.open(os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "gs2k.png")).convert("RGB")
for refine in (False, True):
    w.REFINE = refine
    req = w.RemoveBackgroundRequest(image_url="x", output_format=os.environ.get("FMT","png"), background="#141a38", cache=False)
    for _ in range(2): w.remove_background(img, req)
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = w.remove_background(img, req); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    tag = "refine" if refine else "base"
    for k, v in out.items():
        Image.open(io.BytesIO(v)).save(os.path.join(ROOT, "out", f"e2e_{tag}_{k}.png"))
    print(f"{tag}: {img.size} min {min(ts)*1000:.0f} ms median {sorted(ts)[2]*1000:.0f} ms (incl png encode) artifacts={list(out)}")
