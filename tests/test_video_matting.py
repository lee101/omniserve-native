import subprocess

import pytest

from workloads import video_matting


def test_alpha_validation_forces_libvpx_and_extracts_alpha(monkeypatch, tmp_path):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video_matting.subprocess, "run", run)
    metrics = video_matting._validate_vp9_alpha(tmp_path / "foreground.webm")

    assert metrics == {"alpha_validated": True, "alpha_decoder": "libvpx-vp9"}
    assert commands[0][:7] == [
        video_matting.FFMPEG, "-hide_banner", "-loglevel", "error", "-c:v", "libvpx-vp9", "-i",
    ]
    assert "alphaextract" in commands[0]


def test_alpha_validation_rejects_an_opaque_decoder_path(monkeypatch, tmp_path):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, "", "Requested planes not available")

    monkeypatch.setattr(video_matting.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="VP9 alpha validation failed"):
        video_matting._validate_vp9_alpha(tmp_path / "foreground.webm")
