from __future__ import annotations

import json
import os
import time
from pathlib import Path

SEEDS = Path(__file__).with_name("seeds.json")
POLICIES = ("local_only", "overflow_on_busy", "fastest", "cheapest_within_deadline", "background")
GATEWAY_TIERS = ("paid", "sub", "free", "background")
MIN_SAMPLES = 5


def decide(policy: str, local_wait_ms: float, local_p50_ms: float, remote_p50_ms: float,
           deadline_ms: float | None = None, allow_overflow: bool = False) -> str:
    local_eta = local_wait_ms + local_p50_ms
    if policy == "local_only":
        return "local"
    if policy == "background":
        if allow_overflow:
            return "remote" if remote_p50_ms < local_eta else "local"
        if deadline_ms and deadline_ms > 0 and local_eta > deadline_ms and remote_p50_ms <= deadline_ms:
            return "remote"
        return "local"
    if policy == "fastest":
        return "remote" if remote_p50_ms < local_eta else "local"
    if policy == "cheapest_within_deadline":
        if deadline_ms and deadline_ms > 0:
            if local_eta <= deadline_ms:
                return "local"
            if remote_p50_ms <= deadline_ms:
                return "remote"
        return "remote" if remote_p50_ms < local_eta else "local"
    return "remote"


def pareto(candidates: list[dict]) -> list[dict]:
    live = [c for c in candidates if c.get("available", True) and c.get("quality_ok", True)
            and c.get("p50_ms") is not None and c.get("usd_per_job") is not None]
    front = []
    for c in live:
        dominated = any(o is not c and o["p50_ms"] <= c["p50_ms"] and o["usd_per_job"] <= c["usd_per_job"]
                        and (o["p50_ms"] < c["p50_ms"] or o["usd_per_job"] < c["usd_per_job"]) for o in live)
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: (c["p50_ms"], c["usd_per_job"]))


def load_seeds(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or SEEDS).read_text())


def measured(ledger, workload: str, candidate: dict, since: float) -> dict | None:
    from .ledger import percentile
    if ledger is None:
        return None
    rows = [r for r in ledger.samples(workload, since) if not r["cache_hit"]
            and str(r["status"] or "") in ("200", "ok", "COMPLETED")]
    if candidate["kind"] == "local":
        rows = [r for r in rows if r["backend"] == "local"]
        walls = [r["exec_ms"] for r in rows if r["exec_ms"]]
    else:
        rows = [r for r in rows if r["backend"] == "runpod" and r["endpoint"] == candidate.get("endpoint")]
        walls = [r["wall_ms"] for r in rows if r["wall_ms"]]
    if len(walls) < MIN_SAMPLES:
        return None
    usd = sum(r["est_usd"] or 0 for r in rows) / len(rows)
    cold = sum(1 for r in rows if r["cold"]) / len(rows)
    if candidate["kind"] == "runpod":
        usd *= ledger.billing_factor(candidate["endpoint"])
    return {"p50_ms": percentile(walls, 50), "p95_ms": percentile(walls, 95), "usd_per_job": usd,
            "n": len(walls), "cold_fraction": cold, "source": "ledger"}


def build_routing(seeds: dict, ledger=None, days: float = 7, now: float | None = None) -> dict:
    now = now or time.time()
    since = now - days * 86400
    tolerance = float(seeds.get("quality_tolerance", 0.01))
    tier_policies = dict(seeds.get("tiers", {}))
    overrides = json.loads(os.getenv("FRONTIER_TIER_OVERRIDES", "") or "{}")
    tier_policies.update(overrides)
    background_overflow = os.getenv("FRONTIER_BACKGROUND_OVERFLOW", "0") == "1"
    out = {"version": 1, "generated_at": now, "window_days": days, "workloads": {}}
    for name, spec in seeds.get("workloads", {}).items():
        reference = float(spec.get("reference_quality", 1.0))
        candidates = []
        for seed in spec.get("candidates", []):
            c = {**seed, "source": "seed"}
            got = measured(ledger, name, c, since)
            if got:
                c.update(got)
            c["quality_ok"] = float(c.get("quality", 0)) >= reference - tolerance
            candidates.append(c)
        front = pareto(candidates)
        for c in candidates:
            c["frontier"] = any(f is c for f in front)
        local = next((c for c in candidates if c["kind"] == "local" and c.get("available", True)), None)
        remotes = [c for c in candidates if c["kind"] != "local" and c.get("available", True) and c["quality_ok"]
                   and c.get("p50_ms") is not None]
        tiers = {}
        for tier in (*GATEWAY_TIERS, "priority"):
            policy = (overrides.get(tier) or spec.get("tiers", {}).get(tier) or tier_policies.get(tier)
                      or {"policy": "overflow_on_busy"})
            policy = dict(policy)
            if policy.get("policy") == "background" and background_overflow:
                policy["allow_overflow"] = True
            if policy.get("policy") not in POLICIES:
                policy["policy"] = "overflow_on_busy"
            if policy["policy"] == "cheapest_within_deadline" and not policy.get("deadline_ms"):
                policy["deadline_ms"] = spec.get("deadline_ms")
            key = (lambda c: (c["usd_per_job"], c["p50_ms"])) if policy["policy"] in (
                "cheapest_within_deadline", "local_only", "background") else (lambda c: (c["p50_ms"], c["usd_per_job"]))
            deadline = policy.get("deadline_ms")
            ordered = sorted(remotes, key=key)
            if policy["policy"] == "cheapest_within_deadline" and deadline:
                ordered = [c for c in ordered if c["p50_ms"] <= deadline] + [c for c in ordered if c["p50_ms"] > deadline]
            policy["remote_order"] = [c["id"] for c in ordered]
            tiers[tier] = policy
        best_remote = min(remotes, key=lambda c: c["p50_ms"]) if remotes else None
        entry = {"candidates": candidates, "frontier": [c["id"] for c in front], "tiers": tiers}
        if local and local.get("p50_ms") and best_remote:
            entry["gateway"] = {"local_p50_ms": round(local["p50_ms"], 1),
                                "remote_p50_ms": round(best_remote["p50_ms"], 1),
                                "tiers": {t: {k: v for k, v in tiers[t].items() if k in ("policy", "deadline_ms", "allow_overflow") and v}
                                          for t in GATEWAY_TIERS}}
        out["workloads"][name] = entry
    return out


def write_routing(routing: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(routing, indent=1, sort_keys=True))
    tmp.replace(path)
