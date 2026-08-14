import importlib
import json
from pathlib import Path
import subprocess
import types


def test_registry_alias_and_legacy_detection(monkeypatch):
    runtime = importlib.import_module("runtime.handler")
    manifest = Path(__file__).parents[1] / "workloads" / "workloads.json"
    monkeypatch.setattr(runtime, "MANIFEST", manifest)
    monkeypatch.setattr(runtime, "_registry", None)
    registry = runtime._load_registry()
    assert registry["video_background_removal"]["name"] == "video-matting"
    job = {"input": {"video_url": "https://cdn.example/input.webm", "output_upload_url": "https://upload.example"}}
    assert runtime._workload_name(job) == "video-matting"


def test_manifest_is_generic_and_bounded():
    manifest = json.loads((Path(__file__).parents[1] / "workloads" / "workloads.json").read_text())
    assert {"video-matting", "h3-video"}.issubset(manifest)
    assert all(0 < item["required_mib"] < 50000 for item in manifest.values())


def test_free_memory_prefers_cuda_when_nvidia_smi_is_restricted(monkeypatch):
    runtime = importlib.import_module("runtime.handler")

    class Device:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        device=lambda _index: Device(),
        mem_get_info=lambda: (12 << 30, 48 << 30),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "nvidia-smi", output="[Insufficient Permissions]")
        ),
    )

    assert runtime._free_mib() == 12 * 1024


def test_dispatch_keeps_workloads_resident_when_they_fit(monkeypatch):
    runtime = importlib.import_module("runtime.handler")
    released = []

    class Module:
        def __init__(self, name):
            self.name = name

        def handler(self, _job):
            return {"route": self.name}

        def release(self):
            released.append(self.name)

    modules = {"test.video": Module("video-matting"), "test.h3": Module("h3-video")}
    registry = {
        "video-matting": {"name": "video-matting", "module": "test.video", "required_mib": 1800},
        "h3-video": {"name": "h3-video", "module": "test.h3", "required_mib": 43000},
    }
    monkeypatch.setattr(runtime, "_load_registry", lambda: registry)
    monkeypatch.setattr(runtime, "ENFORCE_VRAM", False)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: modules[name])
    runtime._loaded_modules.clear()
    runtime._active_name = ""
    runtime._active_module = None

    video = runtime.dispatch({"input": {"workload": "video-matting"}})
    h3 = runtime.dispatch({"input": {"workload": "h3-video"}})
    reused = runtime.dispatch({"input": {"workload": "video-matting"}})

    assert released == []
    assert video["omniserve"]["resident_workloads"] == ["video-matting"]
    assert h3["omniserve"]["resident_workloads"] == ["h3-video", "video-matting"]
    assert reused["omniserve"]["reused"] is True
    for name in list(runtime._loaded_modules):
        runtime._release_name(name)
