"""Reusable adaptive admission policy for GPU model workers."""

from __future__ import annotations

import threading
import time


class AdaptiveCudaGuard:
    """Learn a safe scratch floor and cool a lane down after CUDA OOMs.

    A fixed free-memory threshold prevents obvious overcommit, but it cannot
    account for another process growing between the check and allocation. An
    OOM raises the threshold above the free memory observed before the failed
    attempt and starts an exponential cooldown. Sustained success decays both
    conservatively so a transient spike cannot pin throughput low forever.
    """

    def __init__(
        self,
        minimum_free_mib: int,
        *,
        oom_margin_mib: int = 512,
        backoff_seconds: float = 5.0,
        backoff_max_seconds: float = 300.0,
        recovery_successes: int = 8,
    ) -> None:
        self._lock = threading.Lock()
        self.minimum_free_mib = max(1, minimum_free_mib)
        self.oom_margin_mib = max(1, oom_margin_mib)
        self.backoff_seconds = max(0.1, backoff_seconds)
        self.backoff_max_seconds = max(self.backoff_seconds, backoff_max_seconds)
        self.recovery_successes = max(1, recovery_successes)
        self.required_free_mib = self.minimum_free_mib
        self.cooldown_until = 0.0
        self.backoff_level = 0
        self.successes_since_oom = 0
        self.total_ooms = 0
        self.last_oom_at = 0.0
        self.last_oom_free_mib = 0

    def capacity(self, free_mib: int, total_mib: int, now: float | None = None) -> dict:
        current = time.monotonic() if now is None else now
        with self._lock:
            cooldown = max(0.0, self.cooldown_until - current)
            required = min(self.required_free_mib, max(self.minimum_free_mib, total_mib - 256))
            return {
                "ready": cooldown <= 0.0 and free_mib >= required,
                "required_free_mib": required,
                "cooldown_seconds": round(cooldown, 3),
                "backoff_level": self.backoff_level,
                "successes_since_oom": self.successes_since_oom,
                "total_ooms": self.total_ooms,
                "last_oom_at": self.last_oom_at,
                "last_oom_free_mib": self.last_oom_free_mib,
            }

    def note_oom(self, free_mib: int, total_mib: int, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            self.total_ooms += 1
            self.backoff_level = min(self.backoff_level + 1, 16)
            delay = min(
                self.backoff_max_seconds,
                self.backoff_seconds * (2 ** (self.backoff_level - 1)),
            )
            self.cooldown_until = max(self.cooldown_until, current + delay)
            failed_floor = max(
                self.required_free_mib + self.oom_margin_mib,
                free_mib + self.oom_margin_mib,
            )
            self.required_free_mib = min(
                failed_floor,
                max(self.minimum_free_mib, total_mib - 256),
            )
            self.successes_since_oom = 0
            self.last_oom_at = time.time()
            self.last_oom_free_mib = max(0, free_mib)

    def note_success(self) -> None:
        with self._lock:
            self.successes_since_oom += 1
            if self.successes_since_oom < self.recovery_successes:
                return
            self.successes_since_oom = 0
            self.backoff_level = max(0, self.backoff_level - 1)
            self.required_free_mib = max(
                self.minimum_free_mib,
                self.required_free_mib - self.oom_margin_mib,
            )
