#!/usr/bin/env python3
"""summarize.py fit_dir : test-set frontier, max real-eval saving per policy at a p95-LPIPS cap (oracle threshold on test, upper bound)."""
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1] + "/curve.csv")))
caps = [0.02, 0.03, 0.05, 0.1]
best = defaultdict(lambda: [0.0] * len(caps))
for r in rows:
    if float(r["budget"]) != float(rows[0]["budget"]) and r["policy"] in ("fixed", "dx", "d0", "d0_hyst"):
        continue
    k = (r["kind"], r["policy"])
    for i, c in enumerate(caps):
        if float(r["lpips_p95"]) <= c:
            best[k][i] = max(best[k][i], float(r["evals_saved_frac"]))
print("kind policy " + " ".join(f"p95<={c}" for c in caps))
for (kind, pol), v in sorted(best.items()):
    print(f"{kind:5s} {pol:28s} " + " ".join(f"{100 * x:6.1f}%" for x in v))
