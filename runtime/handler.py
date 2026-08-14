#!/usr/bin/env python3
"""Manifest-driven RunPod entry point for OmniServe GPU workloads."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

try:
    import runpod
except ModuleNotFoundError:  # Local native workers do not need the provider SDK.
    runpod = None


MANIFEST = Path(os.getenv("OMNISERVE_WORKLOAD_MANIFEST", "/opt/omniserve/workloads/workloads.json"))
RESERVE_MIB = int(os.getenv("OMNISERVE_GPU_RESERVE_MIB", "512"))
ENFORCE_VRAM = os.getenv("OMNISERVE_ENFORCE_VRAM", "1").lower() not in {"0", "false", "no"}
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


def dispatch(job: dict) -> dict:
    global _active_name, _active_module
    requested = _workload_name(job)
    definition = _load_registry().get(requested)
    if definition is None:
        supported = sorted({item["name"] for item in _load_registry().values()})
        raise ValueError(f"unsupported workload {requested!r}; supported: {', '.join(supported)}")
    with _dispatch_lock:
        name = definition["name"]
        reused = name in _loaded_modules
        free_mib_before = None
        if not reused:
            if ENFORCE_VRAM:
                free_mib_before = _make_room(definition["required_mib"])
            module = importlib.import_module(definition["module"])
            _loaded_modules[name] = {
                "module": module,
                "required_mib": definition["required_mib"],
                "last_used": time.monotonic(),
            }
        loaded = _loaded_modules[name]
        _active_name = name
        _active_module = loaded["module"]
        result = _active_module.handler(job)
        loaded["last_used"] = time.monotonic()
        if not isinstance(result, dict):
            raise RuntimeError(f"workload {name} returned a non-object result")
        result["omniserve"] = {
            "workload": name,
            "required_mib": definition["required_mib"],
            "runtime": "omniserve-native-v1",
            "reused": reused,
            "resident_workloads": sorted(_loaded_modules),
        }
        if free_mib_before is not None:
            result["omniserve"]["free_mib_before"] = free_mib_before
        return result


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("the RunPod SDK is required for the serverless entry point")
    runpod.serverless.start({"handler": dispatch})
