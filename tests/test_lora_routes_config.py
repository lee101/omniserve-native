import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "deploy" / "zimage-lora-routes.json"
REGISTRY = Path("/nvme0n1-disk/code/cutedsl-site/inference/lora_registry.json")


def load():
    return json.loads(ROUTES.read_text())["routes"]


def test_routes_are_well_formed():
    routes = load()
    assert routes and routes[-1]["name"] == "default" and "keywords" not in routes[-1]
    for route in routes:
        assert 0 < len(route["loras"]) <= 4
        for lora in route["loras"]:
            assert re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", lora["id"])
            assert -4.0 <= lora.get("scale", 1.0) <= 4.0
        for word in route.get("keywords", []) + route.get("exclude", []):
            assert word == word.lower() and 0 < len(word) < 96


def test_text_prompts_are_excluded_everywhere():
    for route in load():
        assert {"text", "logo*", "typograph*", "poster*"} <= set(route["exclude"])


def test_route_loras_are_native_loadable():
    if not REGISTRY.exists():
        return
    registry = {entry["id"]: entry for entry in json.loads(REGISTRY.read_text())}
    for route in load():
        for lora in route["loras"]:
            entry = registry[lora["id"]]
            assert not entry.get("is_adult")
            header = Path(entry["path"])
            if not header.exists():
                continue
            with header.open("rb") as handle:
                size = int.from_bytes(handle.read(8), "little")
                keys = [k for k in json.loads(handle.read(size)) if k != "__metadata__"]
            assert keys and all(k.startswith("diffusion_model.") for k in keys)
