import os, sys, time
sys.path.insert(0, "workers")
os.environ.setdefault("OMATTE_LIB", "build-cuda/libomatte.so")
import torch
from PIL import Image
import birefnet_worker as w, matte_refine as mr
w.load_model()
img = Image.open(sys.argv[1]).convert("RGB").resize((2048, 2048), Image.LANCZOS)
orig = mr._key
seen = []
mr._key = lambda b, a: (lambda r: (seen.append(r[0]), r)[1])(orig(b, a))
req = w.RemoveBackgroundRequest(image_url="x", output_format="webp", background="#ffffff", cache=False)
w.remove_background(img, req)
torch.cuda.synchronize(); t = time.perf_counter(); out = w.remove_background(img, req); torch.cuda.synchronize()
print("key channel decisions:", seen[-2:], f"{(time.perf_counter()-t)*1000:.0f} ms")
import io; Image.open(io.BytesIO(out["composite"])).resize((768, 768)).save("matte/bench/out/nonkey.png")
