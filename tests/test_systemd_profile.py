from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shared_gpu_profile_tiles_vae_decode_for_1024_images():
    """The constrained production profile must not full-frame decode 1024px images."""
    unit = (ROOT / "systemd" / "omniserve-native.service").read_text()

    assert "Environment=OMNISERVE_NATIVE_SD_PARAMS_BACKEND=cpu" in unit
    assert "Environment=OMNISERVE_NATIVE_SD_MAX_VRAM=2" in unit
    assert "Environment=OMNISERVE_NATIVE_SD_VAE_TILING=1" in unit
    assert "Environment=OMNISERVE_NATIVE_SD_VAE_TILE_X=64" in unit
    assert "Environment=OMNISERVE_NATIVE_SD_VAE_TILE_Y=64" in unit


def test_shared_gpu_profile_runs_llm_fully_on_gpu():
    """NGL=1 serves Gemma on CPU at ~4 tok/s; 99 is the measured GPU path."""
    unit = (ROOT / "systemd" / "omniserve-native.service").read_text()

    assert "Environment=OMNISERVE_NATIVE_NGL=99" in unit
