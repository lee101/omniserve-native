from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_gateway_disables_blocking_core_dumps():
    recovery_drop_in = (
        ROOT / "systemd" / "omniserve-native-recovery.conf"
    ).read_text(encoding="utf-8")

    assert "[Service]" in recovery_drop_in
    assert "LimitCORE=0" in recovery_drop_in
