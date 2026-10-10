#!/usr/bin/env python3
import json, math, os, re, sys
import numpy as np
from traj import load_lat

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
KS = (3, 5, 8, 12, 16, 20, 24, 27)
prompts = [p for p in sorted(os.listdir(out)) if os.path.exists(f"{out}/{p}/lat_step{N}.lat")]


def sigmas(mu, n):
    s = [math.exp(mu) / (math.exp(mu) + (1 / (1 - i / n) - 1)) if i < n else 0.0 for i in range(n + 1)]
    return s


rows = {}
for p in prompts:
    log = open(f"{out}/{p}/log.txt").read()
    mu = float(re.search(r"Flux scheduler: .*?mu=([0-9.]+)", log).group(1))
    sg = sigmas(mu, N)
    z = {k: load_lat(f"{out}/{p}/lat_step{k}.lat").astype(np.float64).ravel() for k in range(1, N + 1)}
    x0 = {}
    for k in range(1, N):
        v = (z[k + 1] - z[k]) / (sg[k + 1] - sg[k])
        x0[k] = z[k] - sg[k] * v
    x0[N] = z[N]
    ref = x0[N]
    nr = np.linalg.norm(ref)
    err = {k: float(np.linalg.norm(x0[k] - ref) / nr) for k in range(1, N + 1)}
    dlt = {k: float(np.linalg.norm(x0[k + 1] - x0[k]) / nr) for k in range(1, N)}
    cut = {}
    for tau in (0.02, 0.05, 0.10):
        cut[tau] = next((k for k in range(1, N + 1) if all(err[j] < tau for j in range(k, N + 1))), N)
    rows[p] = dict(mu=mu, err=err, delta=dlt, cut={str(t): c for t, c in cut.items()})
json.dump(rows, open(f"{out}/traj_summary.json", "w"))
print("x0-pred rel error vs final, by completed steps")
print("prompt".ljust(12), " ".join(f"k={k:<3d}" for k in KS), " cut@2% 5% 10%")
for p, r in rows.items():
    print(p.ljust(12), " ".join(f"{r['err'][k]:5.3f}" for k in KS), "  ", " ".join(f"{r['cut'][str(t)]:>3d}" for t in (0.02, 0.05, 0.10)))
print("step-to-step x0 change (rel), by step")
for p, r in rows.items():
    print(p.ljust(12), " ".join(f"{r['delta'][k]:5.3f}" for k in KS))
