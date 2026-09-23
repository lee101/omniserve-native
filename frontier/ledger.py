from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path

DEFAULT_DB = "/nvme0n1-disk/data/omniserve-frontier/ledger.db"
GPU_USD_PER_H = {"H200": 5.94, "H100": 4.80, "L40S": 1.76, "RTX6000Ada": 1.76, "A40": 1.63, "A6000": 1.63,
                 "RTX4090": 1.10, "POOL24": 0.70, "B200": 8.65, "RTX5090-local": 0.10}
ENDPOINT_USD_PER_H = {"tlofa06vj7iab7": 0.74, "tmozxvnm9fuuud": 1.10, "akgefm0nzzr4jo": 1.10, "wl0am3ahax9mi9": 0.74}
COLD_DELAY_MS = 3000.0
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, ts REAL NOT NULL, workload TEXT NOT NULL, backend TEXT NOT NULL,
  endpoint TEXT, gpu TEXT, tier TEXT, queue_ms REAL, exec_ms REAL, wall_ms REAL, cold INTEGER DEFAULT 0,
  est_usd REAL DEFAULT 0, quality_tier TEXT, status TEXT, cache_hit INTEGER DEFAULT 0, job_id TEXT, source TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS jobs_wts ON jobs(workload, ts);
CREATE TABLE IF NOT EXISTS billing(day TEXT, endpoint TEXT, amount REAL, billed_ms REAL, disk_gb REAL,
  est_usd REAL, jobs INTEGER, factor REAL, fetched REAL, PRIMARY KEY(day, endpoint));
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""
COLUMNS = ("ts", "workload", "backend", "endpoint", "gpu", "tier", "queue_ms", "exec_ms", "wall_ms", "cold",
           "est_usd", "quality_tier", "status", "cache_hit", "job_id", "source", "detail")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def rates() -> tuple[dict, dict]:
    gpu, endpoint = dict(GPU_USD_PER_H), dict(ENDPOINT_USD_PER_H)
    try:
        override = json.loads(os.getenv("FRONTIER_RATES", "") or "{}")
        gpu.update(override.get("gpu", {}))
        endpoint.update(override.get("endpoint", {}))
    except ValueError:
        pass
    return gpu, endpoint


def estimate_usd(backend: str, *, exec_ms: float = 0, queue_ms: float = 0, endpoint: str | None = None,
                 gpu: str | None = None, saturated: bool = False, cache_hit: bool = False) -> tuple[float, bool]:
    if cache_hit:
        return 0.0, False
    exec_h = max(0.0, exec_ms or 0) / 3.6e6
    if backend == "local":
        usd = exec_h * env_float("FRONTIER_LOCAL_USD_PER_H", 0.10)
        if saturated:
            usd += exec_h * env_float("FRONTIER_OPPORTUNITY_USD_PER_H", 0.70)
        return usd, False
    if backend == "runpod":
        gpu_rates, endpoint_rates = rates()
        rate = endpoint_rates.get(endpoint or "") or gpu_rates.get(gpu or "") or gpu_rates["RTX4090"]
        cold = (queue_ms or 0) > COLD_DELAY_MS
        billed_ms = (exec_ms or 0) + ((queue_ms or 0) if cold else 0)
        return rate * billed_ms / 3.6e6, cold
    return 0.0, False


class Ledger:
    def __init__(self, path: str | os.PathLike = DEFAULT_DB):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._writer: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.conn().executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    @staticmethod
    def normalize(row: dict) -> dict:
        row = dict(row)
        if row.get("ts") is None:
            row["ts"] = time.time()
        saturated = bool(row.pop("saturated", False))
        if row.get("est_usd") is None:
            usd, cold = estimate_usd(row.get("backend", ""), exec_ms=row.get("exec_ms") or 0,
                                     queue_ms=row.get("queue_ms") or 0, endpoint=row.get("endpoint"),
                                     gpu=row.get("gpu"), saturated=saturated, cache_hit=bool(row.get("cache_hit")))
            row["est_usd"] = usd
            row.setdefault("cold", int(cold))
        if row.get("wall_ms") is None and (row.get("exec_ms") is not None or row.get("queue_ms") is not None):
            row["wall_ms"] = (row.get("exec_ms") or 0) + (row.get("queue_ms") or 0)
        if isinstance(row.get("detail"), (dict, list)):
            row["detail"] = json.dumps(row["detail"], separators=(",", ":"))
        for key in ("cold", "cache_hit"):
            row[key] = int(bool(row.get(key)))
        return {key: row.get(key) for key in COLUMNS}

    def insert(self, rows: list[dict]) -> None:
        if not rows:
            return
        values = [tuple(self.normalize(r)[c] for c in COLUMNS) for r in rows]
        sql = f"INSERT INTO jobs({','.join(COLUMNS)}) VALUES({','.join('?' * len(COLUMNS))})"
        conn = self.conn()
        conn.execute("BEGIN")
        conn.executemany(sql, values)
        conn.execute("COMMIT")

    def record(self, **row) -> None:
        self._start()
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def flush(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

    def _start(self) -> None:
        if self._writer and self._writer.is_alive():
            return
        with self._lock:
            if self._writer and self._writer.is_alive():
                return
            self._writer = threading.Thread(target=self._drain, name="frontier-ledger", daemon=True)
            self._writer.start()

    def _drain(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < 200:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                self.insert(batch)
            except (sqlite3.Error, TypeError, ValueError) as exc:
                self.dropped += len(batch)
                print(f"frontier ledger write failed: {exc}", flush=True)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn().execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    def ingest_jsonl(self, path: str | os.PathLike, source: str = "gateway") -> int:
        path = Path(path)
        if not path.exists():
            return 0
        key = f"offset:{path}"
        stat = path.stat()
        state = json.loads(self.meta(key, "{}") or "{}")
        offset = state.get("offset", 0) if state.get("inode") == stat.st_ino and state.get("offset", 0) <= stat.st_size else 0
        rows = []
        with path.open("rb") as handle:
            handle.seek(offset)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                offset += len(raw)
                try:
                    item = json.loads(raw)
                except ValueError:
                    continue
                ports = {int(x) for x in os.getenv("FRONTIER_GATEWAY_PORTS", "8792").split(",") if x.strip()}
                if item.get("port") is not None and ports and int(item["port"]) not in ports:
                    continue
                backend = item.get("backend", "local")
                row = {"ts": item.get("ts"), "workload": item.get("workload", "image"), "tier": item.get("tier"),
                       "queue_ms": item.get("queue_ms"), "exec_ms": item.get("exec_ms"), "source": source,
                       "status": str(item.get("status", "")), "detail": {"reason": item.get("reason"),
                                                                          "local_wait_ms": item.get("local_wait_ms")}}
                if backend == "cache":
                    row.update(backend="local", cache_hit=1, quality_tier="exact")
                elif backend == "overflow":
                    row.update(backend="gateway-overflow", est_usd=0.0, wall_ms=item.get("exec_ms"))
                else:
                    row.update(backend="local", gpu="RTX5090-local", quality_tier="reference",
                               saturated=item.get("reason") == "queued" or (item.get("local_wait_ms") or 0) > 0)
                rows.append(row)
        self.insert(rows)
        self.set_meta(key, json.dumps({"inode": stat.st_ino, "offset": offset}))
        return len(rows)

    def reconcile(self, api_key: str, days: int = 7, endpoints: list[str] | None = None, fetch=None) -> list[dict]:
        start = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(time.time() - days * 86400))
        url = f"https://rest.runpod.io/v1/billing/endpoints?bucketSize=day&grouping=endpointId&startTime={start}"
        if fetch is None:
            request = urllib.request.Request(url, headers={"Authorization": "Bearer " + api_key,
                                                           "User-Agent": "omniserve-frontier/1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                buckets = json.load(response)
        else:
            buckets = fetch(url)
        known = set(endpoints or ENDPOINT_USD_PER_H)
        out = []
        conn = self.conn()
        for bucket in buckets:
            endpoint = bucket.get("endpointId")
            if endpoint not in known:
                continue
            day = str(bucket.get("time", ""))[:10]
            day_start = time.mktime(time.strptime(day, "%Y-%m-%d")) - time.timezone
            est, jobs = conn.execute(
                "SELECT COALESCE(SUM(est_usd),0), COUNT(*) FROM jobs WHERE backend='runpod' AND endpoint=? AND ts>=? AND ts<?",
                (endpoint, day_start, day_start + 86400)).fetchone()
            amount = float(bucket.get("amount") or 0)
            factor = amount / est if est > 0 else None
            row = {"day": day, "endpoint": endpoint, "amount": amount, "billed_ms": bucket.get("timeBilledMs"),
                   "disk_gb": bucket.get("diskSpaceBilledGB"), "est_usd": est, "jobs": jobs, "factor": factor,
                   "fetched": time.time()}
            conn.execute("INSERT OR REPLACE INTO billing(day, endpoint, amount, billed_ms, disk_gb, est_usd, jobs, factor, fetched)"
                         " VALUES(:day,:endpoint,:amount,:billed_ms,:disk_gb,:est_usd,:jobs,:factor,:fetched)", row)
            out.append(row)
        self.set_meta("last_reconcile", str(time.time()))
        return out

    def billing_factor(self, endpoint: str, days: int = 14) -> float:
        rows = self.conn().execute(
            "SELECT factor FROM billing WHERE endpoint=? AND factor IS NOT NULL AND jobs>=3 ORDER BY day DESC LIMIT ?",
            (endpoint, days)).fetchall()
        values = sorted(r[0] for r in rows)
        if not values:
            return 1.0
        return min(10.0, max(0.5, values[len(values) // 2]))

    def samples(self, workload: str, since: float) -> list[sqlite3.Row]:
        conn = self.conn()
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute("SELECT * FROM jobs WHERE workload=? AND ts>=? ORDER BY ts", (workload, since)).fetchall()
        finally:
            conn.row_factory = None

    def summary(self, hours: float = 24) -> dict:
        since = time.time() - hours * 3600
        rows = self.conn().execute(
            "SELECT workload, backend, COALESCE(endpoint,''), cache_hit, wall_ms, est_usd, status FROM jobs WHERE ts>=?",
            (since,)).fetchall()
        groups: dict = {}
        for workload, backend, endpoint, cache_hit, wall, usd, status in rows:
            name = f"{backend}:{endpoint}" if endpoint else backend
            if cache_hit:
                name += ":cache_hit"
            g = groups.setdefault(workload, {}).setdefault(name, {"jobs": 0, "errors": 0, "usd": 0.0, "wall": []})
            g["jobs"] += 1
            g["usd"] += usd or 0
            if status and not str(status).startswith("2") and status not in ("ok", "COMPLETED"):
                g["errors"] += 1
            elif wall is not None:
                g["wall"].append(wall)
        for backends in groups.values():
            for g in backends.values():
                walls = sorted(g.pop("wall"))
                g["p50_ms"] = percentile(walls, 50)
                g["p95_ms"] = percentile(walls, 95)
                g["usd"] = round(g["usd"], 6)
        billing = [dict(zip(("day", "endpoint", "amount", "est_usd", "jobs", "factor"), r)) for r in self.conn().execute(
            "SELECT day, endpoint, amount, est_usd, jobs, factor FROM billing ORDER BY day DESC, endpoint LIMIT 30")]
        return {"window_hours": hours, "generated_at": time.time(), "workloads": groups, "billing": billing,
                "dropped": self.dropped}

    def prometheus(self, hours: float = 24) -> str:
        data = self.summary(hours)
        lines = ["# HELP frontier_jobs Jobs recorded in the window.", "# TYPE frontier_jobs gauge",
                 "# HELP frontier_usd Estimated USD in the window.", "# TYPE frontier_usd gauge",
                 "# HELP frontier_wall_p50_ms Median wall time of successful jobs.", "# TYPE frontier_wall_p50_ms gauge"]
        for workload, backends in sorted(data["workloads"].items()):
            for backend, g in sorted(backends.items()):
                labels = f'workload="{workload}",backend="{backend}"'
                lines.append(f"frontier_jobs{{{labels}}} {g['jobs']}")
                lines.append(f"frontier_usd{{{labels}}} {g['usd']:.6f}")
                if g["p50_ms"] is not None:
                    lines.append(f"frontier_wall_p50_ms{{{labels}}} {g['p50_ms']:.1f}")
        for b in data["billing"][:10]:
            lines.append(f'frontier_billed_usd{{endpoint="{b["endpoint"]}",day="{b["day"]}"}} {b["amount"]:.6f}')
        lines.append(f"frontier_ledger_dropped_total {data['dropped']}")
        return "\n".join(lines) + "\n"


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


_LEDGER: Ledger | None = None
_LEDGER_LOCK = threading.Lock()


def get_ledger() -> Ledger | None:
    global _LEDGER
    if os.getenv("FRONTIER_LEDGER", "0") != "1":
        return None
    if _LEDGER is None:
        with _LEDGER_LOCK:
            if _LEDGER is None:
                try:
                    _LEDGER = Ledger(os.getenv("FRONTIER_LEDGER_DB", DEFAULT_DB))
                except (OSError, sqlite3.Error) as exc:
                    print(f"frontier ledger unavailable: {exc}", flush=True)
                    return None
    return _LEDGER


def record(**row) -> None:
    try:
        ledger = get_ledger()
        if ledger is not None:
            ledger.record(**row)
    except Exception as exc:
        print(f"frontier ledger record skipped: {type(exc).__name__}", flush=True)
