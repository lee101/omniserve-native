#!/usr/bin/env python3
"""End-to-end route smoke on a running routed lane: posts prompts, saves images, prints wall time.
Check the server's stderr for "[lora-route] <name>" lines.   python3 qualitybench/route_smoke.py http://127.0.0.1:8799 OUTDIR"""
import base64, json, os, sys, time, urllib.request
url, out = sys.argv[1], sys.argv[2]
P = [("digitalpainting", "a detailed digital painting of a knight resting against a mossy wall at dusk"),
     ("digitalpainting", "fantasy illustration of an elf archer in an autumn forest"),
     ("animepaint", "painterly anime illustration of a girl on a train platform at dusk, cinematic lighting"),
     ("anime", "cel shaded anime boy with headphones on a rooftop"),
     ("people", "a woman jogging along a river path in the morning"),
     ("beauty", "studio portrait of a woman with a nose ring and wet hair"),
     ("-", "a bowl of fruit on a wooden table")]
os.makedirs(out, exist_ok=True)
for i, (exp, p) in enumerate(P):
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{url}/v1/images/generations",
        json.dumps(dict(prompt=p, width=1024, height=1024, seed=7, output_format="png")).encode(), {"Content-Type": "application/json"}), timeout=900))
    open(f"{out}/{i:02d}_{exp}.png", "wb").write(base64.b64decode(r["data"][0]["b64_json"]))
    print(f"{i:02d} expect={exp:16s} {time.time() - t0:.1f}s  {p[:60]}", flush=True)
