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
    assert {"video-matting", "h3-video", "wan-animate-2"}.issubset(manifest)
    assert all(0 < item["required_mib"] < 60000 for item in manifest.values())


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
    monkeypatch.setattr(runtime, "RESERVE_MIB", 0)
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


def test_auto_profile_uses_fastest_fitting_frontier(monkeypatch):
    runtime = importlib.import_module("runtime.handler")
    definition = {
        "name": "wan-animate-2", "module": "test.wan", "required_mib": 20480,
        "profiles": [
            {"name": "small", "required_mib": 20480, "order": 0},
            {"name": "balanced", "required_mib": 28672, "order": 1},
            {"name": "throughput", "required_mib": 57344, "order": 2},
        ],
    }
    runtime._loaded_modules.clear()
    monkeypatch.setattr(runtime, "RESERVE_MIB", 512)
    assert runtime._profile_for(definition, free_mib=32000)["profile"] == "balanced"
    assert runtime._profile_for(definition, free_mib=18000)["profile"] == "small"
    assert runtime._profile_for(definition, "throughput", free_mib=1000)["profile"] == "throughput"


def test_cuda_oom_unloads_and_retries_lower_profile(monkeypatch):
    runtime = importlib.import_module("runtime.handler")
    released = []
    calls = []

    class Module:
        def handler(self, job):
            profile = runtime._inputs(job)["_omniserve_profile"]
            calls.append(profile)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return {"ok": True}

        def release(self):
            released.append(True)

    definition = {
        "name": "wan-animate-2", "module": "test.wan", "required_mib": 20,
        "profiles": [
            {"name": "small", "required_mib": 20, "order": 0},
            {"name": "balanced", "required_mib": 30, "order": 1},
        ],
    }
    registry = {"wan-animate-2": definition}
    monkeypatch.setattr(runtime, "_load_registry", lambda: registry)
    monkeypatch.setattr(runtime, "ENFORCE_VRAM", False)
    monkeypatch.setattr(runtime, "RESERVE_MIB", 0)
    monkeypatch.setattr(runtime, "_free_mib", lambda: 100)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _name: Module())
    monkeypatch.setattr(runtime, "_clear_cuda", lambda: None)
    monkeypatch.setattr(runtime, "_acquire_vram_lease", lambda required, name: {"granted": True, "brokered": False, "mb": 0})
    monkeypatch.setattr(runtime, "_release_vram_lease", lambda lease: None)
    runtime._loaded_modules.clear()

    result = runtime.dispatch({"input": {"workload": "wan-animate-2"}})

    assert calls == ["balanced", "small"]
    assert released == [True]
    assert result["omniserve"]["profile"] == "small"
    assert result["omniserve"]["oom_retries"] == 1
    runtime._loaded_modules.clear()


def test_repeated_cuda_oom_descends_full_profile_frontier(monkeypatch):
    runtime = importlib.import_module("runtime.handler")
    calls = []

    class Module:
        def handler(self, job):
            profile = runtime._inputs(job)["_omniserve_profile"]
            calls.append(profile)
            if profile != "small":
                raise RuntimeError("CUDA out of memory")
            return {"ok": True}

        def release(self):
            return None

    definition = {
        "name": "wan-animate-2", "module": "test.wan", "required_mib": 20,
        "profiles": [
            {"name": "small", "required_mib": 20, "order": 0},
            {"name": "balanced", "required_mib": 30, "order": 1},
            {"name": "throughput", "required_mib": 40, "order": 2},
        ],
    }
    monkeypatch.setattr(runtime, "_load_registry", lambda: {"wan-animate-2": definition})
    monkeypatch.setattr(runtime, "ENFORCE_VRAM", False)
    monkeypatch.setattr(runtime, "RESERVE_MIB", 0)
    monkeypatch.setattr(runtime, "_free_mib", lambda: 100)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _name: Module())
    monkeypatch.setattr(runtime, "_clear_cuda", lambda: None)
    monkeypatch.setattr(runtime, "_acquire_vram_lease", lambda required, name: {"granted": True, "brokered": False, "mb": 0})
    monkeypatch.setattr(runtime, "_release_vram_lease", lambda lease: None)
    runtime._loaded_modules.clear()

    result = runtime.dispatch({"input": {"workload": "wan-animate-2"}})

    assert calls == ["throughput", "balanced", "small"]
    assert result["omniserve"]["profile"] == "small"
    assert result["omniserve"]["oom_retries"] == 2
    runtime._loaded_modules.clear()


def test_broker_denial_happens_before_plugin_import(monkeypatch):
    runtime = importlib.import_module("runtime.handler")
    imported = []
    definition = {"name": "video-matting", "module": "test.video", "required_mib": 20}
    monkeypatch.setattr(runtime, "_load_registry", lambda: {"video-matting": definition})
    monkeypatch.setattr(runtime, "ENFORCE_VRAM", False)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: imported.append(name))
    monkeypatch.setattr(runtime, "_acquire_vram_lease", lambda required, name: {
        "granted": False, "brokered": True, "reason": "no_headroom", "mb": 0,
    })
    runtime._loaded_modules.clear()

    try:
        runtime.dispatch({"input": {"workload": "video-matting"}})
    except RuntimeError as error:
        assert "global GPU admission denied" in str(error)
    else:
        raise AssertionError("broker denial should fail dispatch")

    assert imported == []
    assert runtime._loaded_modules == {}
