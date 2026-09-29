"""Adapter for the H3 runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_module = None


def _load():
    global _module
    if _module is not None:
        return _module
    source = Path("/src/rp_handler.py")
    if not source.is_file():
        raise RuntimeError("H3 workload is not installed in this image")
    if "/src" not in sys.path:
        sys.path.insert(0, "/src")
    spec = importlib.util.spec_from_file_location("omniserve_h3_handler", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("H3 workload could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module = module
    return module


def handler(job: dict) -> dict:
    return _load().handler(job)


def release() -> None:
    module = _module
    if module is None:
        return
    runtime = getattr(module, "_runtime", None)
    if runtime is not None:
        try:
            runtime.close()
        finally:
            module._runtime = None
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
