"""Extract the GEdit-Bench (stepfun-ai/GEdit-Bench) source images listed in cases.json into src/.

uv run --with pyarrow --with pillow python qualitybench/edit/prep.py SHARD.arrow ...
Shards: https://huggingface.co/datasets/stepfun-ai/GEdit-Bench/resolve/main/data-0000N-of-00009.arrow
Sources are saved as JPEG q95 with the long edge capped at 2048 (prod callers send what users upload).
"""
import io, json, os, sys
import pyarrow as pa
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
cases = json.load(open(os.path.join(HERE, "cases.json")))
want = {c["key"]: c for c in cases}
os.makedirs(os.path.join(HERE, "src"), exist_ok=True)
for path in sys.argv[1:]:
    with pa.memory_map(path) as f:
        try:
            table = pa.ipc.open_stream(f).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(f).read_all()
    for row in table.select(["key", "input_image_raw", "input_image"]).to_pylist():
        c = want.get(row["key"])
        if not c:
            continue
        img = row["input_image_raw"] or row["input_image"]
        im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        im.thumbnail((2048, 2048), Image.LANCZOS)
        im.save(os.path.join(HERE, "src", c["file"]), "JPEG", quality=95)
        print(c["id"], im.size)
missing = [c["id"] for c in cases if not os.path.exists(os.path.join(HERE, "src", c["file"]))]
print("missing:", missing)
