from __future__ import annotations

import hashlib
from concurrent.futures import Future
import hmac
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.getenv("YUE_COG_ROOT", str(Path(__file__).resolve().parents[2] / "yue-cog")))
from yue_runtime import MODEL_REVISION, VAE_REVISION, RUNTIME_VERSION, normalize_request
from audio_transport import prepare_output_upload
from gpu_admission import AdaptiveCudaGuard

sys.path.insert(0, os.getenv("FRONTIER_LIB", str(Path(__file__).resolve().parents[1])))
try:
    import frontier
except ImportError:
    frontier = None

TIMEOUT = 1500
CACHE = Path(os.getenv("YUE_RESULT_CACHE", "/tmp/omniserve-yue-cache"))
GUARD = AdaptiveCudaGuard(int(os.getenv("YUE_MIN_FREE_MIB", "18432")), backoff_seconds=60)
LOCAL_LOCK = threading.Lock()
SLOTS = threading.BoundedSemaphore(4)
PROCESS = None
PIPE = None
LAST_USED = 0.0
INFLIGHT = {}
INFLIGHT_LOCK = threading.Lock()
LOCAL_STARTED = 0.0
LOCAL_WAITERS = 0
REMOTE_TIMING = threading.local()


def ledger(**row) -> None:
    if frontier is not None:
        frontier.record(workload="yue", source="yue_worker", quality_tier="equal", **row)


def busy_route(tier: str) -> tuple[str | None, float]:
    if frontier is None or os.getenv("YUE_FRONTIER_ROUTING", "0") != "1" or not LOCAL_LOCK.locked():
        return None, 0.0
    router = frontier.get_router()
    local = router.candidate("yue", "local") or {}
    p50 = float(local.get("p50_ms") or 0)
    if p50 <= 0:
        return None, 0.0
    wait_ms = max(0.0, p50 - (time.monotonic() - LOCAL_STARTED) * 1000) + LOCAL_WAITERS * p50
    return router.decide("yue", tier, wait_ms), wait_ms


def cache_key(request: dict) -> str:
    identity = {"request": request, "runtime": RUNTIME_VERSION,
                "model_id": os.getenv("YUE_MODEL", "m-a-p/YuE2-3B"),
                "vae_id": os.getenv("YUE_VAE", "m-a-p/YuE2-Vae"),
                "model": os.getenv("YUE_MODEL_REVISION", MODEL_REVISION),
                "vae": os.getenv("YUE_VAE_REVISION", VAE_REVISION),
                "backend": os.getenv("YUE_BACKEND", "torch"),
                "quant": os.getenv("YUE_QUANT", "none")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gpu_memory() -> tuple[int, int]:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits", "-i", os.getenv("YUE_GPU_INDEX", "0")], timeout=3, text=True)
        return tuple(int(value.strip()) for value in output.strip().split(","))
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0, 0


def resident_mib() -> int:
    process = PROCESS
    if process is None or not process.is_alive():
        return 0
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits", "-i", os.getenv("YUE_GPU_INDEX", "0")], timeout=3, text=True)
        for line in output.splitlines():
            pid, used = (int(value.strip()) for value in line.split(","))
            if pid == process.pid:
                return used
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0


def reap_idle() -> None:
    while True:
        time.sleep(5)
        if LOCAL_LOCK.acquire(blocking=False):
            try:
                if PROCESS is not None and (time.monotonic() - LAST_USED > float(os.getenv("YUE_IDLE_SECONDS", "30"))
                                           or gpu_memory()[0] < 2048):
                    stop_local()
            finally:
                LOCAL_LOCK.release()


def child_main(connection) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("YUE_GPU_INDEX", "0")
    from yue_runtime import Runtime
    runtime = Runtime()
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        try:
            connection.send({"output": runtime.generate(request)})
        except Exception as error:
            connection.send({"error": str(error), "oom": "out of memory" in str(error).lower()})
            return


def stop_local() -> None:
    global PROCESS, PIPE
    if PROCESS is not None:
        PROCESS.terminate()
        PROCESS.join(timeout=5)
        if PROCESS.is_alive():
            PROCESS.kill()
            PROCESS.join(timeout=5)
    if PIPE is not None:
        PIPE.close()
    PROCESS, PIPE = None, None


def local_generate(request: dict, wait_s: float = 0.0) -> dict | None:
    global PROCESS, PIPE, LAST_USED, LOCAL_STARTED, LOCAL_WAITERS
    if os.getenv("YUE_LOCAL_ENABLED", "1") != "1":
        return None
    if wait_s > 0:
        LOCAL_WAITERS += 1
        try:
            acquired = LOCAL_LOCK.acquire(timeout=wait_s)
        finally:
            LOCAL_WAITERS -= 1
    else:
        acquired = LOCAL_LOCK.acquire(blocking=False)
    if not acquired:
        return None
    LOCAL_STARTED = time.monotonic()
    success = False
    try:
        free, total = gpu_memory()
        if PROCESS is not None and not PROCESS.is_alive():
            stop_local()
            free, total = gpu_memory()
        if free < 2048 or not GUARD.capacity(free + resident_mib(), total)["ready"]:
            return None
        if PROCESS is None:
            context = multiprocessing.get_context("spawn")
            PIPE, child = context.Pipe()
            PROCESS = context.Process(target=child_main, args=(child,), daemon=True)
            PROCESS.start()
            child.close()
        PIPE.send(request)
        if not PIPE.poll(TIMEOUT):
            raise TimeoutError("local music generation timed out")
        result = PIPE.recv()
        if result.get("oom"):
            GUARD.note_oom(free, total)
            return None
        if result.get("error"):
            raise RuntimeError("local music generation failed: " + result["error"])
        GUARD.note_success()
        success = True
        LAST_USED = time.monotonic()
        return {**result["output"], "backend": "local"}
    finally:
        if not success:
            stop_local()
        LOCAL_LOCK.release()


def remote_request(path: str, payload: dict | None = None) -> dict:
    endpoint = os.getenv("YUE_RUNPOD_ENDPOINT_ID", "")
    key = os.getenv("RUNPOD_API_KEY", "")
    if not endpoint or not key:
        raise RuntimeError("local GPU busy and YuE RunPod endpoint is not configured")
    base = "https://api.runpod.ai/v2/" + endpoint
    request = urllib.request.Request(base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "Omniserve-YuE/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def remote_generate(request: dict, timeout: float | None = None) -> dict:
    remaining = TIMEOUT if timeout is None else timeout
    execution_ms = 180000 if request["semantic_max_tokens"] <= 1500 else 480000
    target = prepare_output_upload(request["format"])
    state = remote_request("/run", {"input": {**request, "output_upload": target},
        "policy": {"executionTimeout": execution_ms, "ttl": max(10000, int(remaining * 1000))}})
    job_id = state.get("id")
    if not isinstance(job_id, str) or not job_id or not all(c.isalnum() or c == "-" for c in job_id):
        raise RuntimeError("RunPod returned no valid job ID")
    deadline = time.monotonic() + remaining
    terminal = False
    try:
        while time.monotonic() < deadline:
            status = state.get("status")
            if status == "COMPLETED":
                terminal = True
                REMOTE_TIMING.value = {"queue_ms": state.get("delayTime"), "exec_ms": state.get("executionTime"),
                                       "job_id": job_id, "status": status}
                output = state.get("output", {})
                if not isinstance(output, dict) or output.get("error") or output.get("audio_url") != target["audio_url"]:
                    raise RuntimeError("RunPod music job returned no audio")
                return {**output, "backend": "runpod", "remote_job_id": job_id}
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                terminal = True
                raise RuntimeError("RunPod music job " + status.lower())
            time.sleep(2)
            try:
                state = remote_request("/status/" + job_id)
            except (TimeoutError, urllib.error.URLError):
                continue
        raise TimeoutError("RunPod music generation timed out")
    finally:
        if not terminal:
            try:
                remote_request("/cancel/" + job_id, {})
            except (OSError, ValueError):
                pass


def generate_once(request: dict, tier: str = "free") -> dict:
    request = normalize_request(request)
    path = CACHE / (cache_key(request) + ".json")
    try:
        if path.stat().st_mtime >= time.time() - 86400:
            ledger(backend="local", cache_hit=1, tier=tier, status="200", exec_ms=0)
            return {**json.loads(path.read_text()), "cached": True}
        path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    if not SLOTS.acquire(blocking=False):
        raise BlockingIOError("music queue is full")
    try:
        started = time.monotonic()
        result = local_generate(request)
        wait_ms = 0.0
        if result is None:
            choice, wait_ms = busy_route(tier)
            if choice == "local":
                result = local_generate(request, wait_s=max(1.0, TIMEOUT / 2))
        if result is not None:
            elapsed = (time.monotonic() - started) * 1000
            ledger(backend="local", gpu="RTX5090-local", tier=tier, status="200", exec_ms=elapsed,
                   saturated=wait_ms > 0, detail={"local_wait_ms": wait_ms})
        else:
            REMOTE_TIMING.value = {}
            try:
                result = remote_generate(request, timeout=max(1, TIMEOUT - (time.monotonic() - started)))
            finally:
                timing = getattr(REMOTE_TIMING, "value", {}) or {}
                ledger(backend="runpod", endpoint=os.getenv("YUE_RUNPOD_ENDPOINT_ID", ""), tier=tier,
                       status=timing.get("status", "FAILED"), job_id=timing.get("job_id"),
                       queue_ms=timing.get("queue_ms"), exec_ms=timing.get("exec_ms"),
                       wall_ms=(time.monotonic() - started) * 1000, detail={"local_wait_ms": wait_ms})
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as temporary:
                json.dump(result, temporary)
            os.replace(temporary.name, path)
            entries = sorted(CACHE.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            retained = 0
            for old in entries:
                size = old.stat().st_size
                retained += size
                if retained > 2 * 1024**3 or old.stat().st_mtime < time.time() - 86400:
                    old.unlink(missing_ok=True)
        except OSError:
            pass
        return {**result, "cached": False}
    finally:
        SLOTS.release()


def generate(request: dict, tier: str = "free") -> dict:
    request = normalize_request(request)
    key = cache_key(request)
    with INFLIGHT_LOCK:
        future = INFLIGHT.get(key)
        owner = future is None
        if owner:
            future = Future()
            INFLIGHT[key] = future
    if not owner:
        shared = future.result(timeout=TIMEOUT + 60)
        ledger(backend="local", cache_hit=1, tier=tier, status="200", exec_ms=0, detail={"inflight": True})
        return {**shared, "cached": True}
    try:
        result = generate_once(request, tier)
        future.set_result(result)
        return result
    except Exception as error:
        future.set_exception(error)
        raise
    finally:
        with INFLIGHT_LOCK:
            INFLIGHT.pop(key, None)


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if self.path != "/health":
            self.respond(404, {"error": "not found"})
            return
        free, total = gpu_memory()
        local_ready = os.getenv("YUE_LOCAL_ENABLED", "1") == "1" and not LOCAL_LOCK.locked() and free >= 2048
        local_ready = local_ready and GUARD.capacity(free + resident_mib(), total)["ready"]
        self.respond(200, {"status": "ok", "local_ready": local_ready,
            "free_mib": free, "required_free_mib": GUARD.required_free_mib,
            "remote_configured": bool(os.getenv("YUE_RUNPOD_ENDPOINT_ID") and os.getenv("RUNPOD_API_KEY")),
            "runtime": RUNTIME_VERSION})

    def do_POST(self) -> None:
        if self.path != "/v1/music/generations":
            self.respond(404, {"error": "not found"})
            return
        secret = os.getenv("YUE_WORKER_SECRET", "")
        if not secret or not hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), ("Bearer " + secret).encode("utf-8")):
            self.respond(401, {"error": "invalid secret"})
            return
        try:
            self.connection.settimeout(15)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("request must be between 1 and 65536 bytes")
            request = normalize_request(json.loads(self.rfile.read(size)))
        except (ValueError, TypeError, OSError):
            self.respond(400, {"error": "invalid music request"})
            return
        try:
            tier = self.headers.get("X-Omniserve-Tier", "")
            self.respond(200, generate(request, tier) if tier else generate(request))
        except BlockingIOError:
            self.respond(429, {"error": "music queue is full"})
        except Exception as error:
            print(f"yue generation failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            self.respond(503, {"error": "Music generation unavailable; please try again later"})


if __name__ == "__main__":
    threading.Thread(target=reap_idle, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", int(os.getenv("YUE_WORKER_PORT", "9106"))), Handler).serve_forever()
