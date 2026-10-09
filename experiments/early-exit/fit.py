#!/usr/bin/env python3
"""fit.py --feats feats.npz --labels labels.npz --split split.json --out DIR [--budget 0.1] [--alpha 0.05]

Fits P(exit error <= budget) (logistic regression and LightGBM), picks the probability threshold on grouped CV
of the train split so the per-request violation rate stays <= alpha, and evaluates exit policies on the test split.
Taylor disagreement only vetoes. Writes DIR/report.json, DIR/curve.csv, DIR/model_<kind>.txt (sd.cpp format).
"""
import argparse, json, os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE = ["frac", "lsigma", "ldx", "ld0", "ld0_prev"]
CACHE = ["lcache_x", "lcache_d", "ldensity_d"]
TAYLOR = ["ltaylor"]


def load(a):
    f = np.load(a.feats, allow_pickle=True); l = np.load(a.labels, allow_pickle=True)
    F = {k: f[k] for k in f.files}; L = {k: l[k] for k in l.files}
    idx = {(t, int(s)): i for i, (t, s) in enumerate(zip(L["traj"], L["step"]))}
    n = len(F["traj"])
    for k in ["lpips", "psnr", "clip_img", "clip_txt", "aes", "clip_txt_final", "aes_final"]:
        v = np.full(n, np.nan)
        for i, (t, s) in enumerate(zip(F["traj"], F["step"])):
            j = idx.get((t, int(s)))
            if j is not None:
                v[i] = L[k][j]
        F[k] = v
    last = F["step"] == F["N"]
    F["lpips"][last] = 0; F["psnr"][last] = 99; F["clip_img"][last] = 1
    F["clip_txt"][last] = F["clip_txt_final"][last]; F["aes"][last] = F["aes_final"][last]
    eps = 1e-6
    F["lsigma"] = np.log(F["sigma"] + eps)
    for k in ["dx", "d0", "d0_prev", "taylor", "cache_x", "cache_d", "density_d"]:
        F["l" + k] = np.log(F[k] + eps)
    return F


def X(F, cols, m):
    return np.stack([F[c][m] for c in cols], 1)


def simulate(F, m_traj, score, thr, veto=None, vthr=np.inf, min_frac=0.0):
    """First real step with score >= thr (and taylor <= vthr) exits; returns per-traj (steps_saved, real_saved, row)."""
    out = []
    trajs = F["traj"]
    for t in m_traj:
        rows = F["_rows"][t]
        ex = rows[-1]
        for i in rows:
            if F["skipped"][i] or F["step"][i] == F["N"][i] or F["frac"][i] < min_frac:
                continue
            if np.isnan(F["lpips"][i]):
                continue
            if score[i] >= thr and (veto is None or veto[i] <= vthr):
                ex = i; break
        after = [i for i in rows if F["step"][i] > F["step"][ex]]
        out.append((F["N"][ex] - F["step"][ex], sum(1 for i in after if not F["skipped"][i]), ex))
    return out


def summarize(F, res, budget):
    ex = np.array([r[2] for r in res])
    lp = F["lpips"][ex]
    nreal = np.array([sum(1 for i in F["_rows"][F["traj"][e]] if not F["skipped"][i]) for e in ex])
    return {"n": len(res), "steps_saved": float(np.mean([r[0] for r in res])),
            "steps_saved_frac": float(np.mean([r[0] / F["N"][r[2]] for r in res])),
            "evals_saved_frac": float(np.mean(np.array([r[1] for r in res]) / nreal)),
            "exit_rate": float(np.mean([r[0] > 0 for r in res])),
            "lpips_mean": float(lp.mean()), "lpips_p95": float(np.quantile(lp, 0.95)), "lpips_max": float(lp.max()),
            "violation": float(np.mean(lp > budget)),
            "clip_img_mean": float(F["clip_img"][ex].mean()), "clip_img_min": float(F["clip_img"][ex].min()),
            "clip_txt_delta": float(np.nanmean(F["clip_txt"][ex] - F["clip_txt_final"][ex])),
            "aes_delta": float(np.nanmean(F["aes"][ex] - F["aes_final"][ex])),
            "worst": [str(F["traj"][e]) + f"@{int(F['step'][e])}" for e in ex[np.argsort(-lp)[:8]]]}


def pick(F, trajs, score, budget, alpha, veto=None, vgrid=(np.inf,), min_frac=0.0):
    """Lowest threshold (most saving) whose violation rate on these trajs is <= alpha."""
    best = None
    for vt in vgrid:
        for thr in np.unique(np.quantile(score[np.isfinite(score)], np.linspace(0, 1, 201))):
            res = simulate(F, trajs, score, thr, veto, vt, min_frac)
            lp = F["lpips"][[r[2] for r in res]]
            if np.mean(lp > budget) <= alpha:
                s = np.mean([r[0] for r in res])
                if best is None or s > best[0]:
                    best = (s, thr, vt)
                break
    return best


def cv_scores(F, cols, cand, train_trajs, kind, model, budget, folds=5):
    s = np.full(len(F["traj"]), -np.inf)
    groups = np.array([F["group"][i] for i in np.where(cand)[0]])
    idx = np.where(cand)[0]
    y = (F["lpips"][idx] <= budget).astype(int)
    folds = min(folds, len(set(groups)))
    if folds < 2:
        return s
    for tr, te in GroupKFold(folds).split(idx, y, groups):
        if len(set(y[tr])) < 2:
            s[idx[te]] = float(y[tr][0])
            continue
        clf = make(model).fit(X(F, cols, idx[tr]), y[tr])
        s[idx[te]] = clf.predict_proba(X(F, cols, idx[te]))[:, 1]
    return s


def make(model):
    if model == "lr":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)


def export_lr(clf, cols, thr, vthr, min_frac, path, n_steps=0):
    sc, lr = clf.steps[0][1], clf.steps[1][1]
    w = lr.coef_[0] / sc.scale_
    b = lr.intercept_[0] - float((lr.coef_[0] * sc.mean_ / sc.scale_).sum())
    logit_thr = float(np.log(thr / (1 - thr))) if 0 < thr < 1 else (1e9 if thr >= 1 else -1e9)
    with open(path, "w") as f:
        f.write("# sd.cpp early-exit model: exit when b + sum(w*feature) >= logit and ltaylor <= veto\n")
        f.write(f"bias {b:.9g}\nlogit {logit_thr:.9g}\nveto_ltaylor {vthr if np.isfinite(vthr) else 1e9:.9g}\nmin_frac {min_frac:.9g}\nn_steps {n_steps}\n")
        for c, v in zip(cols, w):
            f.write(f"w {c} {v:.9g}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--budget", type=float, default=0.1); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--budgets", default="0.05,0.1,0.15,0.2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    F = load(a)
    split = json.load(open(a.split))
    F["_rows"] = {}
    for i, t in enumerate(F["traj"]):
        F["_rows"].setdefault(t, []).append(i)
    for t in F["_rows"]:
        F["_rows"][t].sort(key=lambda i: F["step"][i])
    F["group"] = np.array([split.get("group", {}).get(t, t) for t in F["traj"]])
    report = {}
    curve = ["kind,budget,policy,param,steps_saved,steps_saved_frac,evals_saved_frac,lpips_mean,lpips_p95,violation"]
    for kind in sorted(set(F["kind"])):
        tr = [t for t in split["train"] if t in F["_rows"] and F["kind"][F["_rows"][t][0]] == kind]
        te = [t for t in split["test"] if t in F["_rows"] and F["kind"][F["_rows"][t][0]] == kind]
        rk = {"n_train": len(tr), "n_test": len(te)}
        in_tr = np.isin(F["traj"], tr)
        cand = in_tr & (F["skipped"] == 0) & (F["step"] < F["N"]) & ~np.isnan(F["lpips"])
        for budget in [float(b) for b in a.budgets.split(",")]:
            rb = {}
            N = int(F["N"][F["_rows"][tr[0]][0]])
            sc_fixed = F["step"].astype(float)
            p = pick(F, tr, sc_fixed, budget, a.alpha)
            if p:
                rb["fixed_step"] = {"exit_step": p[1], **summarize(F, simulate(F, te, sc_fixed, p[1]), budget)}
            for name, sc in [("dx", -F["ldx"]), ("d0", -F["ld0"]), ("d0_hyst", -np.maximum(F["ld0"], F["ld0_prev"]))]:
                m_te = np.isin(F["traj"], te) & (F["skipped"] == 0)
                for thr in np.quantile(sc[m_te], np.linspace(0.3, 1, 36)):
                    r = summarize(F, simulate(F, te, sc, thr), budget)
                    curve.append(f"{kind},{budget},{name},{thr:.4f},{r['steps_saved']:.3f},{r['steps_saved_frac']:.4f},{r['evals_saved_frac']:.4f},{r['lpips_mean']:.4f},{r['lpips_p95']:.4f},{r['violation']:.4f}")
                p = pick(F, tr, sc, budget, a.alpha)
                if p:
                    rb[name] = {"thr": float(-p[1]), **summarize(F, simulate(F, te, sc, p[1]), budget)}
            vgrid = [np.inf] + list(np.quantile(F["ltaylor"][cand], [0.5, 0.75, 0.9]))
            ok_fr = F["frac"][cand & (F["lpips"] <= budget)]
            min_frac = float(max(np.quantile(ok_fr, 0.01) if len(ok_fr) else 1.0, F["frac"][cand].min()))
            rb["min_frac"] = min_frac
            for model in ["lr", "gbt"]:
                for fs_name, cols in [("base", BASE), ("base+cache", BASE + CACHE)]:
                    y = (F["lpips"] <= budget).astype(int)
                    s_cv = cv_scores(F, cols, cand, tr, kind, model, budget)
                    for vname, vg in [("", (np.inf,)), ("+taylor_veto", vgrid)]:
                        p = pick(F, tr, s_cv, budget, a.alpha, F["ltaylor"], vg, min_frac)
                        if not p:
                            continue
                        if len(set(y[cand])) < 2:
                            continue
                        clf = make(model).fit(X(F, cols, np.where(cand)[0]), y[cand])
                        s_te = np.full(len(F["traj"]), -np.inf)
                        m_te = np.isin(F["traj"], te)
                        s_te[m_te] = clf.predict_proba(X(F, cols, np.where(m_te)[0]))[:, 1]
                        key = f"{model}:{fs_name}{vname}"
                        rb[key] = {"p_thr": float(p[1]), "veto_ltaylor": float(p[2]), **summarize(F, simulate(F, te, s_te, p[1], F["ltaylor"], p[2], min_frac), budget)}
                        if model == "lr" and budget == a.budget:
                            export_lr(clf, cols, p[1], p[2], min_frac, out / f"model_{kind}_{fs_name}{vname.replace('+', '_')}.txt", N)
                        for thr in np.quantile(s_te[m_te & np.isfinite(s_te)], np.linspace(0.3, 1, 36)):
                            r = summarize(F, simulate(F, te, s_te, thr, F["ltaylor"], p[2], min_frac), budget)
                            curve.append(f"{kind},{budget},{key},{thr:.4f},{r['steps_saved']:.3f},{r['steps_saved_frac']:.4f},{r['evals_saved_frac']:.4f},{r['lpips_mean']:.4f},{r['lpips_p95']:.4f},{r['violation']:.4f}")
            for k in range(1, N + 1):
                r = summarize(F, simulate(F, te, sc_fixed, k), budget)
                curve.append(f"{kind},{budget},fixed,{k},{r['steps_saved']:.3f},{r['steps_saved_frac']:.4f},{r['evals_saved_frac']:.4f},{r['lpips_mean']:.4f},{r['lpips_p95']:.4f},{r['violation']:.4f}")
            rk[str(budget)] = rb
        report[kind] = rk
    json.dump(report, open(out / "report.json", "w"), indent=1, default=float)
    open(out / "curve.csv", "w").write("\n".join(curve) + "\n")
    for kind, rk in report.items():
        print(f"== {kind} train={rk['n_train']} test={rk['n_test']}")
        for b, rb in rk.items():
            if not b[0].isdigit():
                continue
            print(f" budget lpips<={b} alpha={a.alpha}")
            for name, r in rb.items():
                if not isinstance(r, dict):
                    print(f"  {name} {r:.3f}"); continue
                print(f"  {name:28s} saved {r['steps_saved']:5.2f} ({100*r['steps_saved_frac']:4.1f}% steps, {100*r['evals_saved_frac']:4.1f}% evals) "
                      f"lpips mean {r['lpips_mean']:.4f} p95 {r['lpips_p95']:.4f} max {r['lpips_max']:.3f} viol {100*r['violation']:4.1f}% "
                      f"clipimg {r['clip_img_mean']:.4f} dclip {r['clip_txt_delta']:+.3f} daes {r['aes_delta']:+.3f}")


if __name__ == "__main__":
    main()
