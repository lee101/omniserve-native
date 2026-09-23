from __future__ import annotations

import json
import os
import threading
import time

from .policy import decide

DEFAULT_ROUTING = "/nvme0n1-disk/data/omniserve-frontier/routing.json"


class Router:
    def __init__(self, path: str | None = None, min_interval: float = 1.0):
        self.path = path or os.getenv("FRONTIER_ROUTING_FILE", DEFAULT_ROUTING)
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._table: dict = {}
        self._stamp = None
        self._checked = 0.0

    def table(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if now - self._checked >= self.min_interval:
                self._checked = now
                try:
                    st = os.stat(self.path)
                    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
                    if stamp != self._stamp:
                        with open(self.path) as handle:
                            self._table = json.load(handle)
                        self._stamp = stamp
                except (OSError, ValueError):
                    self._table, self._stamp = {}, None
            return self._table

    def workload(self, name: str) -> dict | None:
        return (self.table().get("workloads") or {}).get(name)

    def tier_policy(self, workload: str, tier: str) -> dict | None:
        spec = self.workload(workload)
        if not spec:
            return None
        return (spec.get("tiers") or {}).get(tier or "free") or (spec.get("tiers") or {}).get("free")

    def candidate(self, workload: str, candidate_id: str) -> dict | None:
        spec = self.workload(workload) or {}
        return next((c for c in spec.get("candidates", []) if c.get("id") == candidate_id), None)

    def decide(self, workload: str, tier: str, local_wait_ms: float, deadline_ms: float | None = None) -> str | None:
        spec = self.workload(workload)
        policy = self.tier_policy(workload, tier)
        if not spec or not policy:
            return None
        local = next((c for c in spec.get("candidates", []) if c.get("kind") == "local" and c.get("available", True)), None)
        order = policy.get("remote_order") or []
        remote = self.candidate(workload, order[0]) if order else None
        if policy["policy"] == "local_only":
            return "local"
        if not remote:
            return "local"
        if not local or local.get("p50_ms") is None:
            return "remote"
        return decide(policy["policy"], local_wait_ms, local["p50_ms"], remote["p50_ms"],
                      deadline_ms or policy.get("deadline_ms"), bool(policy.get("allow_overflow")))

    def remote_order(self, workload: str, tier: str) -> list[str]:
        policy = self.tier_policy(workload, tier) or {}
        return list(policy.get("remote_order") or [])


_ROUTER: Router | None = None


def get_router() -> Router:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = Router()
    return _ROUTER
