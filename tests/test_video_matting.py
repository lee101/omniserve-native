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


def test_max_quality_plan_skips_rvm_and_person_gate():
    plan = video_matting.matting_plan({"max_quality": True, "video_url": "https://cdn.example/clip.mp4"})
    assert plan["engine"] == "matanyone"
    assert plan["skip_person"] is True
    assert plan["allow_standby"] is False
    assert plan["route"] == "local-matanyone"
    assert video_matting.matting_plan({"max_quality": "true"})["engine"] == "matanyone"
    default = video_matting.matting_plan({"video_url": "https://cdn.example/clip.mp4"})
    assert default["engine"] == "rvm"
    assert default["allow_standby"] is True


def test_coarse_subject_mask_keeps_centered_subject():
    from workloads.matanyone_matting import coarse_subject_mask
    import numpy as np

    frame = np.full((64, 64, 3), 12, dtype=np.uint8)
    frame[16:48, 20:44] = (200, 40, 30)
    mask = coarse_subject_mask(frame)
    assert mask.shape == (64, 64)
    assert mask[32, 32] == 255
    assert mask[0, 0] == 0
    assert mask.mean() > 20


def test_matte_quality_rejects_black_and_empty_alpha():
    import numpy as np

    rgb = np.full((16, 16, 3), 80, dtype=np.uint8)
    alpha = np.zeros((16, 16), dtype=np.uint8)
    ok, reason = video_matting.matte_quality_ok(rgb, alpha)
    assert not ok and "coverage" in reason
    alpha[:] = 255
    rgb[:] = 0
    ok, reason = video_matting.matte_quality_ok(rgb, alpha)
    assert not ok and "black" in reason
    rgb[:] = 90
    alpha[:12, :12] = 200
    ok, reason = video_matting.matte_quality_ok(rgb, alpha)
    assert ok and reason == ""


def test_matanyone_source_is_checked_out():
    from workloads.matanyone_matting import matanyone_source

    root = matanyone_source()
    assert (root / "matanyone" / "inference" / "inference_core.py").is_file()


def test_ffmpeg_vp9_alpha_roundtrip_keeps_transparency(tmp_path):
    import numpy as np

    ffmpeg = video_matting.FFMPEG
    source = tmp_path / "rgba.webm"
    alpha = tmp_path / "alpha.raw"
    raw = tmp_path / "frames.rgba"
    frames = np.zeros((2, 16, 16, 4), dtype=np.uint8)
    frames[:, 4:12, 4:12] = (180, 40, 30, 220)
    frames[:, 0, 0] = (0, 0, 0, 0)
    raw.write_bytes(frames.tobytes())
    encoded = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgba", "-s:v", "16x16", "-r", "8", "-i", str(raw),
            "-frames:v", "2", "-an", "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0", "-y", str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if encoded.returncode != 0:
        pytest.skip(encoded.stderr or "libvpx-vp9 yuva unavailable")
    extracted = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-c:v", "libvpx-vp9",
            "-i", str(source), "-vf", "alphaextract,format=gray", "-f", "rawvideo",
            "-pix_fmt", "gray", str(alpha),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert extracted.returncode == 0, extracted.stderr
    plane = np.frombuffer(alpha.read_bytes(), dtype=np.uint8)
    assert plane.size >= 16 * 16
    assert plane.max() > 40
    assert plane.min() < 20


def test_matte_quality_rejects_black_on_varied_frames():
    import numpy as np

    for size in (8, 32, 64):
        rgb = np.full((size, size, 3), 70, dtype=np.uint8)
        alpha = np.zeros((size, size), dtype=np.uint8)
        ok, reason = video_matting.matte_quality_ok(rgb, alpha)
        assert not ok and "coverage" in reason
        rgb[:] = 1
        alpha[:] = 255
        ok, reason = video_matting.matte_quality_ok(rgb, alpha)
        assert not ok and "black" in reason
        rgb[2:size - 2, 2:size - 2] = 90
        alpha[:] = 0
        alpha[2:size - 2, 2:size - 2] = 200
        ok, reason = video_matting.matte_quality_ok(rgb, alpha)
        assert ok


def test_matte_quality_monitor_rejects_persistent_black_after_two_frames():
    import numpy as np

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    alpha = np.full((16, 16), 255, dtype=np.uint8)
    quality = video_matting.MatteQualityMonitor()
    assert quality.check(rgb, alpha) == (True, "")
    assert quality.check(rgb, alpha) == (True, "")
    ok, reason = quality.check(rgb, alpha)
    assert not ok and "3 frames" in reason
    rgb[:] = 80
    assert video_matting.MatteQualityMonitor().check(rgb, alpha) == (True, "")
