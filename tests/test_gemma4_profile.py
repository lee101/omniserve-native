from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gemma4_profile_is_optional_and_preserves_adaptive_memory_settings():
    profile = (ROOT / "systemd" / "omniserve-native-gemma4-iq4.conf").read_text()

    assert "[Service]\n" in profile
    assert "Environment=OMNISERVE_NATIVE_NGL=20" in profile
    assert "Environment=OMNISERVE_NATIVE_KV_TYPE=q8_0" in profile
    assert "Environment=OMNISERVE_NATIVE_BATCH=auto" in profile
    assert "Environment=OMNISERVE_NATIVE_UBATCH=auto" in profile
    assert "Environment=OMNISERVE_NATIVE_LLM_CONTEXTS=1" in profile
    assert "Environment=OMNISERVE_NATIVE_LLM_GGUF=/nvme0n1-disk/models/omniserve-native/G4-MEROMERO-V2-31B-IQ4_XS.gguf" in profile


def test_gemma4_profile_does_not_change_the_default_unit():
    unit = (ROOT / "systemd" / "omniserve-native.service").read_text()

    assert "Environment=OMNISERVE_NATIVE_NGL=0" in unit
    assert "Environment=OMNISERVE_NATIVE_LLM_GGUF=/nvme0n1-disk/models/omniserve-native/gemma-roleplay-v2-q8_0.gguf" in unit
