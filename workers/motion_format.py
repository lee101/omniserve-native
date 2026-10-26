"""SOMA world joints to the ardy-skeleton-motion-1 library format (numpy only)."""

from __future__ import annotations

import numpy as np

FORMAT = "ardy-skeleton-motion-1"

ARDY_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd", "RightHandThumb1",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]
ARDY_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25]


def _soma_names() -> list[str]:
    names = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"]
    for side in ("Left", "Right"):
        names += [f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand", f"{side}HandThumb1",
                  f"{side}HandThumb2", f"{side}HandThumb3", f"{side}HandThumbEnd"]
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            names += [f"{side}Hand{finger}{i}" for i in (1, 2, 3, 4)] + [f"{side}Hand{finger}End"]
    for side in ("Left", "Right"):
        names += [f"{side}Leg", f"{side}Shin", f"{side}Foot", f"{side}ToeBase", f"{side}ToeEnd"]
    return names


SOMA_NAMES = _soma_names()
_SOMA_INDEX = {name: i for i, name in enumerate(SOMA_NAMES)}

_DIRECT = {
    "Hips": "Hips", "Spine": "Spine1", "Spine1": "Spine2", "Spine3": "Chest", "Neck": "Neck1", "Head": "Head",
    "RightUpLeg": "RightLeg", "RightLeg": "RightShin", "RightFoot": "RightFoot", "RightToeBase": "RightToeBase",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftShin", "LeftFoot": "LeftFoot", "LeftToeBase": "LeftToeBase",
    "RightHandEnd": "RightHandMiddleEnd", "LeftHandEnd": "LeftHandMiddleEnd",
}
for _side in ("Right", "Left"):
    for _part in ("Shoulder", "Arm", "ForeArm", "Hand", "HandThumb1"):
        _DIRECT[f"{_side}{_part}"] = f"{_side}{_part}"

_A = {name: i for i, name in enumerate(ARDY_NAMES)}
CONTACT_JOINTS = [_A["LeftFoot"], _A["LeftToeBase"], _A["RightFoot"], _A["RightToeBase"]]
CONTACT_HEIGHT = np.array([0.12, 0.07, 0.12, 0.07])
CONTACT_SPEED = 0.6
CONTACT_REST_HEIGHT = np.array([0.09, 0.03, 0.09, 0.03])
CONTACT_PROB = 0.5


def ardy_positions(joints77: np.ndarray) -> np.ndarray:
    """(T, 77, 3) SOMA FK joints to (T, 27, 3); ARDY Spine2 is the midpoint of SOMA Spine2 and Chest."""
    joints77 = np.asarray(joints77, dtype=np.float64)
    if joints77.ndim != 3 or joints77.shape[1:] != (77, 3):
        raise ValueError(f"expected (T, 77, 3) joints, got {joints77.shape}")
    out = np.empty((joints77.shape[0], len(ARDY_NAMES), 3))
    for i, name in enumerate(ARDY_NAMES):
        if name == "Spine2":
            out[:, i] = 0.5 * (joints77[:, _SOMA_INDEX["Spine2"]] + joints77[:, _SOMA_INDEX["Chest"]])
        else:
            out[:, i] = joints77[:, _SOMA_INDEX[_DIRECT[name]]]
    return out


def snap_to_floor(pos: np.ndarray, contacts: np.ndarray, smooth: int = 5) -> np.ndarray | None:
    """Offset y so contacting feet rest on y=0, interpolating the offset through flight; None without contacts."""
    contacts = np.asarray(contacts, dtype=bool)
    feet = pos[:, CONTACT_JOINTS, 1] - CONTACT_REST_HEIGHT
    touching = contacts.any(axis=1)
    if not touching.any():
        return None
    floor = np.where(contacts, feet, np.inf).min(axis=1)
    frames = np.arange(pos.shape[0])
    offset = np.interp(frames, frames[touching], -floor[touching])
    if smooth > 1 and pos.shape[0] > smooth:
        pad = smooth // 2
        padded = np.pad(offset, pad, mode="edge")
        offset = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")
    out = pos.copy()
    out[..., 1] += offset[:, None]
    return out


def normalize_world(pos: np.ndarray, ground: bool = True) -> np.ndarray:
    """Floor at y=0, hips at the xz origin on frame 0, frame-0 facing along +z (y-up, metres)."""
    pos = pos.copy()
    if ground:
        toes = pos[:, [_A["LeftToeBase"], _A["RightToeBase"]], 1]
        pos[..., 1] -= np.percentile(toes, 2)
    pos[..., 0] -= pos[0, 0, 0]
    pos[..., 2] -= pos[0, 0, 2]
    left_to_right = pos[0, _A["LeftShoulder"]] - pos[0, _A["RightShoulder"]]
    forward = np.cross(np.array([left_to_right[0], 0.0, left_to_right[2]]), np.array([0.0, 1.0, 0.0]))
    norm = np.hypot(forward[0], forward[2])
    if norm > 1e-6:
        fx, fz = forward[0] / norm, forward[2] / norm
        c, s = fz, fx
        x, z = pos[..., 0].copy(), pos[..., 2].copy()
        pos[..., 0] = c * x - s * z
        pos[..., 2] = s * x + c * z
    return pos


def resample(pos: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    if abs(fps_in - fps_out) < 1e-6 or pos.shape[0] < 2:
        return pos
    t_in = np.arange(pos.shape[0]) / fps_in
    count = max(2, int(round((pos.shape[0] - 1) / fps_in * fps_out)) + 1)
    t_out = np.minimum(np.arange(count) / fps_out, t_in[-1])
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, pos.shape[0] - 2)
    w = ((t_out - t_in[idx]) / (t_in[idx + 1] - t_in[idx]))[:, None, None]
    return pos[idx] * (1 - w) + pos[idx + 1] * w


def foot_contacts(pos: np.ndarray, fps: float) -> np.ndarray:
    """(T, 4) float contacts [L_heel, L_toe, R_heel, R_toe] from height and speed."""
    feet = pos[:, CONTACT_JOINTS]
    speed = np.zeros(feet.shape[:2])
    if feet.shape[0] > 1:
        speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * fps
        speed[0] = speed[1]
    contact = (feet[..., 1] < CONTACT_HEIGHT) & (speed < CONTACT_SPEED)
    if contact.shape[0] >= 3:
        padded = np.pad(contact.astype(np.int8), ((1, 1), (0, 0)), mode="edge")
        contact = (padded[:-2] + padded[1:-1] + padded[2:]) >= 2
    return contact.astype(np.float64)


def _round(arr: np.ndarray, digits: int = 5) -> list:
    return np.round(arr, digits).tolist()


def build_ardy_motion(joints77: np.ndarray, fps_in: float, fps_out: float = 20.0,
                      contact_probs: np.ndarray | None = None, **meta) -> dict:
    pos = ardy_positions(joints77)
    snapped = None
    if contact_probs is not None:
        probs = np.asarray(contact_probs, dtype=np.float64)[:, :4]
        snapped = snap_to_floor(pos, probs > CONTACT_PROB)
    pos = normalize_world(snapped if snapped is not None else pos, ground=snapped is None)
    pos = resample(pos, fps_in, fps_out)
    if snapped is not None:
        contacts = resample(probs[:, None, :], fps_in, fps_out)[:, 0, :] > CONTACT_PROB
        contacts = contacts.astype(np.float64)
    else:
        contacts = foot_contacts(pos, fps_out)
    return {
        "format": FORMAT,
        **meta,
        "fps": float(fps_out),
        "frames": int(pos.shape[0]),
        "skeleton": {"names": ARDY_NAMES, "parents": ARDY_PARENTS},
        "positions": _round(pos),
        "footContacts": contacts.tolist(),
    }


def soma_payload(params: dict, fps: float) -> dict:
    """Native SOMA parameters (axis-angle) for rotation-based retargeting."""
    out = {"format": "soma-motion-1", "fps": float(fps), "root_joint": SOMA_NAMES[0], "body_joint_names": SOMA_NAMES[1:]}
    for key in ("body_pose", "global_orient", "transl"):
        out[key] = _round(np.asarray(params[key], dtype=np.float64), 4)
    out["frames"] = len(out["transl"])
    for key in ("identity_coeffs", "scale_params"):
        out[key] = _round(np.asarray(params[key], dtype=np.float64).mean(axis=0), 4)
    return out
