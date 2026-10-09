#!/usr/bin/env python3
"""features.py DIR [DIR...] --split split.json --out feats.npz

Per (traj, step) exit features. Cache/bank features use only train trajectories (leave-one-out for train rows).
split.json: {"train": [keys], "test": [keys]} with key = "<dirname>/<id>".
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def rel(a, b, eps=1e-8):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def pooled(t, k=4):
    x = torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)).reshape(-1, *t.shape[-3:])
    x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
    return F.avg_pool2d(x, k).flatten(1).numpy()


def traj_rows(z):
    st, sig, sk = z["step"], z["sigma"], z["skipped"]
    x = z["x"].astype(np.float32); d = z["den"].astype(np.float32)
    n = len(st)
    rows = []
    real = [i for i in range(n) if not sk[i]]
    for i in range(n):
        pr = [j for j in real if j < i]
        j = pr[-1] if pr else None
        jj = pr[-2] if len(pr) > 1 else None
        f = {"step": int(st[i]), "N": n, "frac": st[i] / n, "sigma": float(sig[i]), "skipped": int(sk[i]),
             "dx": rel(x[i], x[i - 1]) if i else 1.0,
             "d0": rel(d[i], d[j]) if j is not None else 1.0,
             "taylor": 1.0, "d0_prev": 1.0}
        if j is not None and jj is not None:
            pred = d[j] + (d[j] - d[jj]) * (sig[i] - sig[j]) / (sig[j] - sig[jj] + 1e-12)
            f["taylor"] = rel(d[i], pred)
            f["d0_prev"] = rel(d[j], d[jj])
        rows.append(f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--knn", type=int, default=5)
    a = ap.parse_args()
    split = json.load(open(a.split))
    train = set(split["train"])
    keys, meta = [], {}
    for root in a.dirs:
        for l in open(Path(root) / "log.jsonl"):
            m = json.loads(l)
            k = f"{Path(root).name}/{m['id']}"
            if "nocache_of" in m or not m["id"].startswith("q"):
                continue
            if (Path(root) / m["id"] / "traj.npz").exists():
                keys.append(k); meta[k] = (Path(root) / m["id"], m)
    bank_x, bank_d, bank_key = {}, {}, {}
    per = {}
    for k in keys:
        z = np.load(meta[k][0] / "traj.npz")
        per[k] = json.loads(str(z["rows"])) if "rows" in z.files else traj_rows(z)
        px, pd = (z["px"], z["pd"]) if "px" in z.files else (pooled(z["x"]), pooled(z["den"]))
        per[k + "#px"], per[k + "#pd"] = px, pd
        if k in train:
            sig = (meta[k][1]["kind"], len(z["step"]), z["x"].shape[-2:])
            for i in range(len(z["step"])):
                bank_x.setdefault((sig, i), []).append(px[i]); bank_d.setdefault((sig, i), []).append(pd[i])
                bank_key.setdefault((sig, i), []).append(k)
    for b in (bank_x, bank_d):
        for s in b:
            b[s] = torch.from_numpy(np.stack(b[s])).cuda() if torch.cuda.is_available() else torch.from_numpy(np.stack(b[s]))
    out = {c: [] for c in ["traj", "kind", "step", "N", "frac", "sigma", "skipped", "dx", "d0", "d0_prev", "taylor",
                           "cache_x", "cache_d", "density_d", "nn_key"]}
    for k in keys:
        m = meta[k][1]
        px, pd = per[k + "#px"], per[k + "#pd"]
        n = len(per[k])
        for i, f in enumerate(per[k]):
            sig = (m["kind"], n, None)
            bk = next((s for s in bank_x if s[0][0] == m["kind"] and s[0][1] == n and s[1] == i), None)
            cx = cd = dens = 1.0; nk = ""
            if bk is not None:
                qx = torch.from_numpy(px[i]).to(bank_x[bk].device); qd = torch.from_numpy(pd[i]).to(bank_d[bk].device)
                dxs = (bank_x[bk] - qx).norm(dim=1) / (qx.norm() + 1e-8)
                dds = (bank_d[bk] - qd).norm(dim=1) / (qd.norm() + 1e-8)
                if k in train:
                    self_idx = bank_key[bk].index(k)
                    dxs[self_idx] = float("inf"); dds[self_idx] = float("inf")
                cx = float(dxs.min()); best = int(dds.argmin()); cd = float(dds[best]); nk = bank_key[bk][best]
                kk = min(a.knn, len(dds) - (1 if k in train else 0))
                dens = float(dds.topk(kk, largest=False).values.mean()) if kk > 0 else 1.0
            out["traj"].append(k); out["kind"].append(m["kind"])
            for c in ["step", "N", "frac", "sigma", "skipped", "dx", "d0", "d0_prev", "taylor"]:
                out[c].append(f[c])
            out["cache_x"].append(cx); out["cache_d"].append(cd); out["density_d"].append(dens); out["nn_key"].append(nk)
    np.savez(a.out, **{c: np.array(v) for c, v in out.items()})
    print(len(out["traj"]), "rows,", len(keys), "trajs")


if __name__ == "__main__":
    main()
