"""Weighted pose similarity on qpos and threshold-based thinning, used by motion_vocab to drop
near-duplicate frames inside a command bucket, and a left-right symmetry error to rank frames
before thinning.

The features are scaled by sqrt(weight) per block, so the Euclidean distance between two feature
rows is the weighted pose distance. Nothing is forwarded through mujoco: the model only provides
joint names and qpos addresses.
"""

import functools
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import mujoco
import numpy as np
import yaml

ENV_MODEL = Path(__file__).resolve().parents[3] / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
ROBOT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "robot" / "g1" / "g1_29dof.yaml"


@functools.lru_cache(maxsize=None)
def env_model() -> mujoco.MjModel:
    """The G1 model buffer qpos refers to (free base, 29 dof)."""
    return mujoco.MjModel.from_xml_path(str(ENV_MODEL))


def default_pose(model: mujoco.MjModel) -> np.ndarray:
    """(nq,) qpos with the joints at the default joint angles of the robot config (a left-right
    symmetric stance, slightly crouched), qpos0 elsewhere."""
    angles = yaml.safe_load(ROBOT_CONFIG.read_text())["robot"]["init_state"]["default_joint_angles"]
    qpos = model.qpos0.copy()
    for name, angle in angles.items():
        qpos[model.joint(name).qposadr[0]] = angle
    return qpos


# Closer to the pelvis weighs more, legs more than arms.
DEFAULT_WEIGHTS = {
    "base_quat": 4.0,
    "base_height": 4.0,
    "joints": {
        "waist": 3.0,
        "hip": 2.5,
        "knee": 2.0,
        "ankle": 1.0,
        "shoulder": 0.8,
        "elbow": 0.5,
        "wrist": 0.2,
    },
}


def remove_yaw(quat: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions with the heading (ZYX yaw) removed, sign fixed to w >= 0."""
    w, x, y, z = quat.T
    half_yaw = 0.5 * np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(half_yaw), np.sin(half_yaw)
    # (c, 0, 0, -s) * (w, x, y, z): undo the yaw rotation about the world z-axis
    out = np.stack([c * w + s * z, c * x + s * y, c * y - s * x, c * z - s * w], axis=-1)
    return np.where(out[:, :1] < 0, -out, out)


def joint_weights(model: mujoco.MjModel, patterns: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """qpos addresses and weights of the hinge joints; a joint takes the first pattern in its name."""
    addr, weights = [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(j).name
        weight = next((w for pattern, w in patterns.items() if pattern in name), None)
        if weight is None:
            raise ValueError(f"No similarity weight for joint '{name}', patterns: {list(patterns)}")
        addr.append(model.jnt_qposadr[j])
        weights.append(weight)
    return np.array(addr), np.array(weights, dtype=np.float32)


def qpos_features(model: mujoco.MjModel, qpos: np.ndarray, weights: Mapping | None = None) -> np.ndarray:
    """(N, D) features of free-base qpos: yaw-free base quaternion, base height and joint angles."""
    weights = weights or DEFAULT_WEIGHTS
    addr, joint_w = joint_weights(model, weights["joints"])
    return np.concatenate(
        [
            np.sqrt(weights["base_quat"]) * remove_yaw(qpos[:, 3:7]),
            np.sqrt(weights["base_height"]) * qpos[:, 2:3],
            np.sqrt(joint_w) * qpos[:, addr],
        ],
        axis=-1,
    ).astype(np.float32)


def mirror_joints(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left-right mirror of the hinge joints: qpos addresses, partner column and sign of each.

    The mirror of a pose has q'[j] = sign[j] * q[partner[j]], partner swapping left_ / right_.
    Reflecting across the sagittal (xz) plane maps a rotation about axis a by angle t to one about
    R a by -t, so sign = -<axis of partner, R axis>, from the world axes at qpos0: pitch joints
    keep their sign, roll and yaw joints flip it.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reflect = np.array([1.0, -1.0, 1.0])
    joints = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    names = [model.joint(j).name for j in joints]
    partner, sign = [], []
    for j, name in zip(joints, names):
        other = name.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
        k = names.index(other)
        dot = float(np.dot(data.xaxis[joints[k]], reflect * data.xaxis[j]))
        if abs(abs(dot) - 1.0) > 1e-3:
            raise ValueError(f"Joints {name} and {other} are not mirror images (axis dot {dot:.3f})")
        partner.append(k)
        sign.append(-np.sign(dot))
    return np.array([model.jnt_qposadr[j] for j in joints]), np.array(partner), np.array(sign, dtype=np.float32)


def symmetry_error(
    model: mujoco.MjModel, qpos: np.ndarray, weights: Mapping | None = None, center: np.ndarray | None = None
) -> np.ndarray:
    """Left-right symmetry error of the joint angles of each qpos row on its own, (N,).

    With the mirror M of mirror_joints, a pose splits into a symmetric part s = (q + Mq) / 2 and an
    antisymmetric part a = (q - Mq) / 2. A pitch joint (mirror sign +1, the axis of the gait swing)
    may be axially symmetric (left = right) or centrally symmetric about the center c (left and right
    deviating from c by opposite amounts, e.g. the hips mid-stride), each joint on its own: its error
    is min(|a|, |s - c|) = ||l - c| - |r - c|| / 2, left and right off c by the same amount. A roll or
    yaw joint (sign -1) must be axially symmetric, error |a|: opposite deviations there lean or turn
    both limbs to one side, and the waist roll and yaw count in full. c is the qpos center,
    default_pose by default. Joints are weighted as in qpos_features, so the error is in the units
    of the pose distance.
    """
    weights = weights or DEFAULT_WEIGHTS
    center = default_pose(model) if center is None else center
    addr, partner, sign = mirror_joints(model)
    w_addr, joint_w = joint_weights(model, weights["joints"])
    assert (w_addr == addr).all()
    q = qpos[:, addr]
    mirrored = sign * q[:, partner]
    sym, anti = 0.5 * (q + mirrored), 0.5 * (q - mirrored)
    err = np.where(sign > 0, np.minimum(np.abs(anti), np.abs(sym - center[addr])), np.abs(anti))
    return np.linalg.norm(np.sqrt(joint_w) * err, axis=-1)


def _greedy_net(feat: np.ndarray, order: np.ndarray, threshold: float, k: int) -> np.ndarray:
    """Walk rows in order, keeping a row when it is >= threshold from every kept row; stop at k kept."""
    kept = [order[0]]
    kept_feat = np.empty((k, feat.shape[1]), dtype=feat.dtype)
    kept_feat[0] = feat[order[0]]
    for i in order[1:]:
        if len(kept) == k:
            break
        if np.min(np.sum((kept_feat[: len(kept)] - feat[i]) ** 2, axis=-1)) >= threshold**2:
            kept_feat[len(kept)] = feat[i]
            kept.append(i)
    return np.array(kept, dtype=np.int64)


def threshold_sample(
    feat: np.ndarray, k: int, min_dist: float | None = None, seed: int = 0, iters: int = 30
) -> Tuple[np.ndarray, float]:
    """Pick up to k rows pairwise at least a threshold apart, the threshold bisected to fill k.

    Rows are visited in one random order and kept greedily when no kept row is closer than the
    threshold, so the picks follow the density of the rows instead of reaching for outliers. The
    threshold is the largest found that still keeps k rows, but not below min_dist (then fewer
    than k rows are kept). Returns the picked indices and the threshold.
    """
    n = len(feat)
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64), float("inf")
    order = np.random.default_rng(seed).permutation(n)
    if n <= k and min_dist is None:
        return order, 0.0
    lo = 0.0 if min_dist is None else float(min_dist)
    if len(_greedy_net(feat, order, lo, k)) < k:
        return _greedy_net(feat, order, lo, n), lo
    # twice the farthest distance from one row bounds the diameter: keeps a single row
    hi = 2.0 * float(np.linalg.norm(feat - feat[order[0]], axis=-1).max()) + 1e-6
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if len(_greedy_net(feat, order, mid, k)) >= k:
            lo = mid
        else:
            hi = mid
    return _greedy_net(feat, order, lo, k), lo


def net_size(feat: np.ndarray, threshold: float, seed: int = 0) -> int:
    """Rows threshold_sample keeps at this threshold with no quota (same visiting order)."""
    if len(feat) == 0:
        return 0
    order = np.random.default_rng(seed).permutation(len(feat))
    return len(_greedy_net(feat, order, threshold, len(feat)))


def shared_threshold(feats: Sequence[np.ndarray], total: int, seed: int = 0, iters: int = 30) -> float:
    """Largest threshold at which thinning every feature set at that one threshold keeps total rows in all."""
    feats = [f for f in feats if len(f)]
    if sum(len(f) for f in feats) <= total:
        return 0.0
    lo, hi = 0.0, 2.0 * max(float(np.linalg.norm(f - f[0], axis=-1).max()) for f in feats) + 1e-6
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sum(net_size(f, mid, seed) for f in feats) >= total:
            lo = mid
        else:
            hi = mid
    return lo
