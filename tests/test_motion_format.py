import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

import motion_format as mf


def _joints(frames=30, heading=0.0, lift=0.0):
    j = np.zeros((frames, 77, 3))
    idx = {n: i for i, n in enumerate(mf.SOMA_NAMES)}
    base = {
        "Hips": (0, 0.95, 0), "Spine1": (0, 1.05, 0), "Spine2": (0, 1.15, 0), "Chest": (0, 1.25, 0),
        "Neck1": (0, 1.45, 0), "Head": (0, 1.6, 0),
        "LeftShoulder": (0.18, 1.4, 0), "RightShoulder": (-0.18, 1.4, 0),
        "LeftLeg": (0.1, 0.9, 0), "LeftShin": (0.1, 0.5, 0), "LeftFoot": (0.1, 0.1, 0), "LeftToeBase": (0.1, 0.04, 0.1),
        "RightLeg": (-0.1, 0.9, 0), "RightShin": (-0.1, 0.5, 0), "RightFoot": (-0.1, 0.1, 0), "RightToeBase": (-0.1, 0.04, 0.1),
    }
    j[:] = np.array(base["Hips"])
    for name, p in base.items():
        j[:, idx[name]] = p
    c, s = np.cos(heading), np.sin(heading)
    x, z = j[..., 0].copy(), j[..., 2].copy()
    j[..., 0] = c * x + s * z
    j[..., 2] = -s * x + c * z
    j[..., 1] += lift
    return j


def test_name_tables():
    assert len(mf.SOMA_NAMES) == 77 and len(set(mf.SOMA_NAMES)) == 77
    assert len(mf.ARDY_NAMES) == len(mf.ARDY_PARENTS) == 27
    assert mf.SOMA_NAMES[:4] == ["Hips", "Spine1", "Spine2", "Chest"]
    assert mf.SOMA_NAMES[-1] == "RightToeEnd" and mf.SOMA_NAMES[-5] == "RightLeg"


def test_ardy_positions_maps_and_interpolates():
    pos = mf.ardy_positions(_joints())
    a = {n: i for i, n in enumerate(mf.ARDY_NAMES)}
    assert pos.shape == (30, 27, 3)
    assert pos[0, a["Spine3"], 1] == pytest.approx(1.25)
    assert pos[0, a["Spine2"], 1] == pytest.approx(1.20)
    assert pos[0, a["LeftUpLeg"], 1] == pytest.approx(0.9)
    assert pos[0, a["LeftLeg"], 1] == pytest.approx(0.5)


def test_ardy_positions_rejects_bad_shape():
    with pytest.raises(ValueError):
        mf.ardy_positions(np.zeros((5, 76, 3)))


@pytest.mark.parametrize("heading", [0.0, 0.7, 2.5, -1.9])
def test_normalize_world_heading_floor_origin(heading):
    pos = mf.normalize_world(mf.ardy_positions(_joints(heading=heading, lift=0.3)))
    a = {n: i for i, n in enumerate(mf.ARDY_NAMES)}
    assert pos[0, 0, 0] == pytest.approx(0.0) and pos[0, 0, 2] == pytest.approx(0.0)
    assert min(pos[:, a["LeftToeBase"], 1].min(), pos[:, a["RightToeBase"], 1].min()) == pytest.approx(0.0, abs=1e-9)
    left_minus_right = pos[0, a["LeftShoulder"]] - pos[0, a["RightShoulder"]]
    assert left_minus_right[0] > 0.3 and abs(left_minus_right[2]) < 1e-6


def test_resample_keeps_duration_and_endpoints():
    pos = np.zeros((25, 2, 3))
    pos[:, 0, 0] = np.arange(25)
    out = mf.resample(pos, 24.0, 20.0)
    assert out.shape[0] == 21
    assert out[0, 0, 0] == 0 and out[-1, 0, 0] == pytest.approx(24.0)
    assert mf.resample(pos, 20.0, 20.0) is pos


def test_foot_contacts_stance_and_flight():
    pos = mf.ardy_positions(_joints(frames=40))
    pos = mf.normalize_world(pos)
    standing = mf.foot_contacts(pos, 20.0)
    assert standing.shape == (40, 4) and standing.min() == 1.0
    pos[10:25, :, 1] += 1.0
    flying = mf.foot_contacts(pos, 20.0)
    assert flying[12:23].max() == 0.0
    assert flying[:6].min() == 1.0 and flying[-6:].min() == 1.0


def test_build_ardy_motion_schema():
    motion = mf.build_ardy_motion(_joints(frames=48), 24.0, 20.0, id="x/y", name="Y", prompt="p")
    assert motion["format"] == "ardy-skeleton-motion-1"
    assert motion["fps"] == 20.0 and motion["frames"] == len(motion["positions"]) == len(motion["footContacts"]) == 40
    assert len(motion["positions"][0]) == 27 and len(motion["positions"][0][0]) == 3
    assert motion["skeleton"]["parents"][0] == -1 and motion["id"] == "x/y"


def test_soma_payload_shapes():
    t = 12
    payload = mf.soma_payload(
        {"body_pose": np.zeros((t, 228)), "global_orient": np.zeros((t, 3)), "transl": np.zeros((t, 3)),
         "identity_coeffs": np.zeros((t, 45)), "scale_params": np.zeros((t, 69))}, 24.0)
    assert payload["frames"] == t and len(payload["body_pose"][0]) == 228
    assert len(payload["identity_coeffs"]) == 45 and len(payload["scale_params"]) == 69
    assert len(payload["body_joint_names"]) == 76 and payload["root_joint"] == "Hips"


def _drifted():
    pos = mf.ardy_positions(_joints(frames=60))
    pos[30:, :, 1] += 0.45
    pos[10:30, :, 1] += np.sin(np.linspace(0, np.pi, 20))[:, None] + np.linspace(0, 0.45, 20, endpoint=False)[:, None]
    contacts = np.zeros((60, 4), bool)
    contacts[:10] = True
    contacts[30:] = True
    return pos, contacts


def test_snap_to_floor_removes_post_landing_float():
    pos, contacts = _drifted()
    out = mf.snap_to_floor(pos, contacts, smooth=1)
    toe = mf.ARDY_NAMES.index("LeftToeBase")
    assert out[3, toe, 1] == pytest.approx(0.03) and out[45, toe, 1] == pytest.approx(0.03)
    assert out[20, toe, 1] > 0.4
    assert np.abs(np.diff(out[:, toe, 1])).max() < 0.2


def test_snap_to_floor_none_without_contacts():
    pos, _ = _drifted()
    assert mf.snap_to_floor(pos, np.zeros((60, 4), bool)) is None


def test_build_with_contact_probs_returns_to_standing_height():
    joints = _joints(frames=60)
    joints[30:, :, 1] += 0.45
    joints[10:30, :, 1] += 1.0
    probs = np.zeros((60, 6))
    probs[:10, :4] = 0.95
    probs[30:, :4] = 0.95
    motion = mf.build_ardy_motion(joints, 24.0, 20.0, contact_probs=probs)
    hips_end = motion["positions"][-1][0][1]
    assert hips_end == pytest.approx(0.94, abs=0.03)
    assert motion["footContacts"][0] == [1.0, 1.0, 1.0, 1.0] and motion["footContacts"][-1] == [1.0, 1.0, 1.0, 1.0]
    assert max(motion["footContacts"][14:22][i][1] for i in range(8)) == 0.0


def test_build_with_zero_contact_probs_falls_back():
    motion = mf.build_ardy_motion(_joints(frames=30), 24.0, 20.0, contact_probs=np.zeros((30, 6)))
    assert motion["frames"] == len(motion["footContacts"]) and motion["format"] == mf.FORMAT
