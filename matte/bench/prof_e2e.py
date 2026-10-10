import cProfile, pstats, os, sys
sys.argv = [sys.argv[0]]
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "workers"))
os.environ.setdefault("OMATTE_LIB", os.path.join(ROOT, "..", "..", "build-cuda", "libomatte.so"))
from PIL import Image
import birefnet_worker as w
w.load_model()
img = Image.open(os.path.join(ROOT, "gs2k.png")).convert("RGB")
fmt = os.environ.get("FMT", "png")
req = w.RemoveBackgroundRequest(image_url="x", output_format=fmt, background="#141a38", cache=False)
for _ in range(2): w.remove_background(img, req)
pr = cProfile.Profile(); pr.enable()
for _ in range(3): w.remove_background(img, req)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
