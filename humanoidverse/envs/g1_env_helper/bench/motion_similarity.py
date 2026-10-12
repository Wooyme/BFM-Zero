"""Weighted pose similarity on qpos and threshold-based thinning, used by motion_vocab to drop
near-duplicate frames inside a command bucket.

The features are scaled by sqrt(weight) per block, so the Euclidean distance between two feature
rows is the weighted pose distance. Nothing is forwarded through mujoco: the model only provides
joint names and qpos addresses.
"""

from typing import Mapping, Tuple

import mujoco
import numpy as np

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
