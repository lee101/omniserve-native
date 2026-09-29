#!/usr/bin/env python3
"""Manifest-driven RunPod entry point for OmniServe GPU workloads."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any

import requests

try:
    import runpod
except ModuleNotFoundError:  # Local native workers do not need the provider SDK.
    runpod = None


MANIFEST = Path(os.getenv("OMNISERVE_WORKLOAD_MANIFEST", "/opt/omniserve/workloads/workloads.json"))
RESERVE_MIB = int(os.getenv("OMNISERVE_GPU_RESERVE_MIB", "512"))
ENFORCE_VRAM = os.getenv("OMNISERVE_ENFORCE_VRAM", "1").lower() not in {"0", "false", "no"}
VRAM_BROKER_URL = os.getenv("OMNISERVE_VRAM_BROKER_URL", "").rstrip("/")
VRAM_BROKER_REQUIRED = os.getenv("OMNISERVE_VRAM_BROKER_REQUIRED", "0").lower() in {"1", "true", "yes"}
_dispatch_lock = threading.Lock()
_active_name = ""
_active_module = None
_loaded_modules: dict[str, dict] = {}
_registry = None


def _load_registry() -> dict:
    global _registry
    if _registry is not None:
        return _registry
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = {}
    for name, definition in raw.items():
        canonical = name.strip().lower()
        if not canonical or not definition.get("module") or int(definition.get("required_mib", -1)) < 0:
            raise RuntimeError(f"invalid workload definition: {name}")
        entry = {**definition, "name": canonical, "required_mib": int(definition["required_mib"])}
        profiles = []
        for order, profile in enumerate(definition.get("profiles", [])):
            profile_name = str(profile.get("name", "")).strip().lower()
            required_mib = int(profile.get("required_mib", -1))
            if not profile_name or required_mib <= 0:
                raise RuntimeError(f"invalid workload profile: {name}/{profile_name or order}")
            profiles.append({**profile, "name": profile_name, "required_mib": required_mib, "order": order})
        if profiles:
            if len({profile["name"] for profile in profiles}) != len(profiles):
                raise RuntimeError(f"duplicate workload profile: {name}")
            entry["profiles"] = profiles
        for alias in [canonical, *definition.get("aliases", [])]:
            key = str(alias).strip().lower()
            if not key or (key in registry and registry[key]["name"] != canonical):
                raise RuntimeError(f"duplicate workload alias: {alias}")
            registry[key] = entry
    _registry = registry
    return registry


def _inputs(job: dict) -> dict:
    values = job.get("input") or {}
    return values.get("input", values) if isinstance(values, dict) else {}


def _workload_name(job: dict) -> str:
    values = _inputs(job)
    explicit = str(values.get("workload") or values.get("kind") or "").strip().lower()
    if explicit:
        return explicit
    # Backward compatibility for the live Manifold endpoint.
    if values.get("video_url") and (values.get("output_upload_url") or values.get("background_color") == "transparent"):
        return "video-matting"
    return os.getenv("OMNISERVE_DEFAULT_WORKLOAD", "video-matting").strip().lower()


def _free_mib() -> int:
    errors = []
    try:
        import torch

        if torch.cuda.is_available():
            values = []
            for device in range(torch.cuda.device_count()):
                with torch.cuda.device(device):
                    free_bytes, _ = torch.cuda.mem_get_info()
                values.append(int(free_bytes // (1 << 20)))
            if values:
                return max(values)
    except Exception as error:  # container policies vary across RunPod hosts
        errors.append(f"cuda={error}")
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        values = [int(line.strip()) for line in completed.stdout.splitlines()
                  if line.strip().isdigit()]
        if values:
            return max(values)
        errors.append(f"nvidia-smi={completed.stdout.strip() or 'no devices'}")
    except Exception as error:
        errors.append(f"nvidia-smi={error}")
    raise RuntimeError("GPU memory telemetry unavailable: " + "; ".join(errors))


def _clear_cuda() -> None:
    """Return cached allocator blocks after an engine unload, when CUDA exists."""
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError):
                pass
    except (ImportError, RuntimeError):
        pass


def _acquire_vram_lease(required_mib: int, workload: str) -> dict:
    """Reserve cross-process headroom through the production native broker."""
    if not VRAM_BROKER_URL:
        return {"brokered": False, "granted": True, "lease_id": "", "mb": 0}
    owner = f"{socket.gethostname()}-{os.getpid()}-{workload}"[:31]
    try:
        response = requests.post(
            VRAM_BROKER_URL + "/v1/gpu/lease",
            json={"owner": owner, "mb": required_mib, "min_mb": required_mib,
                  "tier": "background", "ttl_s": 10800},
            timeout=(2, 5),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("granted") and payload.get("lease_id"):
            return {"brokered": True, "granted": True, "lease_id": str(payload["lease_id"]),
                    "mb": int(payload.get("mb") or 0)}
        return {"brokered": True, "granted": False, "lease_id": "", "mb": 0,
                "reason": str(payload.get("reason") or "no_headroom")}
    except (requests.RequestException, TypeError, ValueError) as error:
        if VRAM_BROKER_REQUIRED:
            return {"brokered": True, "granted": False, "lease_id": "", "mb": 0,
                    "reason": f"broker_unavailable: {str(error)[:200]}"}
        return {"brokered": False, "granted": True, "lease_id": "", "mb": 0,
                "warning": f"broker_unavailable: {str(error)[:200]}"}


def _release_vram_lease(lease: dict) -> None:
    lease_id = str(lease.get("lease_id") or "")
    if not lease_id or not VRAM_BROKER_URL:
        return
    try:
        requests.post(
            VRAM_BROKER_URL + "/v1/gpu/release",
            json={"lease_id": lease_id}, timeout=(2, 5),
        ).raise_for_status()
    except requests.RequestException:
        # The broker TTL is the crash-safe release path.
        pass


def _release_name(name: str) -> None:
    global _active_name, _active_module
    loaded = _loaded_modules.pop(name, None)
    if loaded is not None:
        release = getattr(loaded["module"], "release", None)
        if callable(release):
            release()
    if _active_name == name:
        _active_name = ""
        _active_module = None
    _clear_cuda()


def _release_active() -> None:
    """Compatibility hook used by local workers and shutdown tooling."""
    if _active_name:
        _release_name(_active_name)


def _make_room(required_mib: int) -> int:
    free_mib = _free_mib()
    needed = required_mib + RESERVE_MIB
    while free_mib < needed and _loaded_modules:
        oldest = min(_loaded_modules, key=lambda name: _loaded_modules[name]["last_used"])
        _release_name(oldest)
        free_mib = _free_mib()
    if free_mib < needed:
        raise RuntimeError(
            f"workload needs {needed} MiB including reserve; {free_mib} MiB free"
        )
    return free_mib


def _profile_for(definition: dict, requested: str = "auto", *, free_mib: int | None = None) -> dict:
    profiles = definition.get("profiles") or []
    if not profiles:
        return {**definition, "profile": "default"}
    requested = requested.strip().lower() or "auto"
    if requested != "auto":
        for profile in profiles:
            if profile["name"] == requested:
                return {**definition, **profile, "profile": profile["name"]}
        choices = ", ".join(["auto", *(profile["name"] for profile in profiles)])
        raise ValueError(f"execution_profile must be one of: {choices}")

    available = _free_mib() if free_mib is None else free_mib
    # Resident engines are reclaimable. Considering their admitted budgets here
    # avoids choosing a slow profile merely because an idle model owns the card.
    reclaimable = sum(int(item.get("required_mib", 0)) for item in _loaded_modules.values())
    potential = available + reclaimable - RESERVE_MIB
    candidates = [profile for profile in profiles if profile["required_mib"] <= potential]
    if not candidates:
        # Preserve the normal admission error, which includes measured free VRAM.
        selected = min(profiles, key=lambda profile: profile["required_mib"])
    else:
        # Manifest order is the cost/quality frontier: later fitting entries are
        # faster or higher quality, never just larger for its own sake.
        selected = max(candidates, key=lambda profile: profile["order"])
    return {**definition, **selected, "profile": selected["name"]}


def _fallback_profile(definition: dict, current: str) -> str:
    profiles = definition.get("profiles") or []
    for index, profile in enumerate(profiles):
        if profile["name"] == current:
            return profiles[max(0, index - 1)]["name"]
    return current


def _is_cuda_oom(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(error).lower()
    return "cuda" in message and ("out of memory" in message or "cuda error: memory allocation" in message)


def _job_with_profile(job: dict, profile: str, retry: int) -> dict:
    prepared: dict[str, Any] = dict(job)
    raw_input = prepared.get("input")
    if isinstance(raw_input, dict):
        envelope = dict(raw_input)
        if isinstance(envelope.get("input"), dict):
            values = dict(envelope["input"])
            values["_omniserve_profile"] = profile
            values["_omniserve_retry"] = retry
            envelope["input"] = values
        else:
            envelope["_omniserve_profile"] = profile
            envelope["_omniserve_retry"] = retry
        prepared["input"] = envelope
    return prepared


def _release_all() -> None:
    for resident in list(_loaded_modules):
        _release_name(resident)


def dispatch(job: dict) -> dict:
    global _active_name, _active_module
    requested = _workload_name(job)
    base_definition = _load_registry().get(requested)
    if base_definition is None:
        supported = sorted({item["name"] for item in _load_registry().values()})
        raise ValueError(f"unsupported workload {requested!r}; supported: {', '.join(supported)}")
    with _dispatch_lock:
        requested_profile = str(_inputs(job).get("execution_profile") or "auto")
        retry = 0
        while True:
            definition = _profile_for(base_definition, requested_profile)
            name = definition["name"]
            resident = _loaded_modules.get(name)
            # A different resource profile can imply different quantization and
            # offload hooks, so it is a real engine switch rather than reuse.
            if resident is not None and resident.get("profile") != definition["profile"]:
                _release_name(name)
                resident = None
            reused = resident is not None
            free_mib_before = None
            if not reused and ENFORCE_VRAM:
                free_mib_before = _make_room(definition["required_mib"])
            lease_mib = definition["required_mib"] if not reused else int(definition.get("scratch_mib", RESERVE_MIB))
            # Acquire cross-process admission before importing the plugin. Some
            # plugins start asynchronous CUDA preloading at import time, so a
            # post-import lease cannot prevent simultaneous loaders from OOMing.
            lease = _acquire_vram_lease(lease_mib, name)
            if not lease.get("granted"):
                raise RuntimeError(f"global GPU admission denied: {lease.get('reason', 'no_headroom')}")
            try:
                if not reused:
                    module = importlib.import_module(definition["module"])
                    _loaded_modules[name] = {
                        "module": module,
                        "required_mib": definition["required_mib"],
                        "profile": definition["profile"],
                        "last_used": time.monotonic(),
                    }
                loaded = _loaded_modules[name]
                _active_name = name
                _active_module = loaded["module"]
                result = _active_module.handler(_job_with_profile(job, definition["profile"], retry))
            except BaseException as error:
                if _is_cuda_oom(error):
                    failed_profile = definition["profile"]
                    fallback_profile = _fallback_profile(base_definition, failed_profile)
                    _release_all()
                    if fallback_profile != failed_profile:
                        requested_profile = fallback_profile
                        retry += 1
                        continue
                raise
            finally:
                _release_vram_lease(lease)
            loaded["last_used"] = time.monotonic()
            if not isinstance(result, dict):
                raise RuntimeError(f"workload {name} returned a non-object result")
            result["omniserve"] = {
                "workload": name,
                "profile": definition["profile"],
                "required_mib": definition["required_mib"],
                "runtime": "omniserve-native-v2",
                "reused": reused,
                "oom_retries": retry,
                "vram_brokered": bool(lease.get("brokered")),
                "vram_lease_mib": int(lease.get("mb") or 0),
                "resident_workloads": sorted(_loaded_modules),
            }
            if free_mib_before is not None:
                result["omniserve"]["free_mib_before"] = free_mib_before
            return result


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("the RunPod SDK is required for the serverless entry point")
    runpod.serverless.start({"handler": dispatch})
