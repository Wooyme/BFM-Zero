"""Per-frame motion metrics and a searchable subset database used by RewardWrapperHV.reward_rank.

Each metric reads one quantity from a forwarded mujoco.MjData. To add a metric, register a
function in METRICS; reward_rank and MotionDatabase pick it up by name with no other changes.
Angles are in degrees, lengths in m, velocities in m/s (deg/s for angular rates).
"""

import dataclasses
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import joblib
import mujoco
import numpy as np

from humanoidverse.envs.g1_env_helper.rewards import get_center_of_mass_linvel, get_xmat, get_xpos, rot2eul

MetricFn = Callable[[mujoco.MjModel, mujoco.MjData], float]
Range = Tuple[float | None, float | None]


def _heading_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Pelvis linear velocity in the yaw-only (heading) frame: [forward, left, up]."""
    # free joint: qvel[0:3] is the world-frame linear velocity
    yaw = rot2eul(get_xmat(model, data, "pelvis"))[2]
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = data.qvel[0:3]
    return np.array([c * vx + s * vy, -s * vx + c * vy, vz])


def _world_angvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    # free joint: qvel[3:6] is the angular velocity in the body frame
    return get_xmat(model, data, "pelvis") @ data.qvel[3:6]


def _euler_deg(model: mujoco.MjModel, data: mujoco.MjData, body: str, axis: int) -> float:
    return float(np.rad2deg(rot2eul(get_xmat(model, data, body))[axis]))


METRICS: Dict[str, MetricFn] = {
    # base (pelvis) position / velocity
    "base_height": lambda m, d: float(get_xpos(m, d, "pelvis")[2]),
    "base_speed": lambda m, d: float(np.linalg.norm(d.qvel[0:2])),
    "base_vx": lambda m, d: float(_heading_velocity(m, d)[0]),
    "base_vy": lambda m, d: float(_heading_velocity(m, d)[1]),
    "base_vz": lambda m, d: float(d.qvel[2]),
    # direction of horizontal motion relative to heading, same convention as move-ego-<angle>-<speed>
    "move_angle": lambda m, d: float(np.rad2deg(np.arctan2(*_heading_velocity(m, d)[[1, 0]]))),
    # same quantity LocomotionReward scores on
    "com_speed": lambda m, d: float(np.linalg.norm(get_center_of_mass_linvel(m, d)[0:2])),
    "yaw_rate": lambda m, d: float(np.rad2deg(_world_angvel(m, d)[2])),
    # orientation
    "base_roll": lambda m, d: _euler_deg(m, d, "pelvis", 0),
    "base_pitch": lambda m, d: _euler_deg(m, d, "pelvis", 1),
    "torso_roll": lambda m, d: _euler_deg(m, d, "torso_link", 0),
    "torso_pitch": lambda m, d: _euler_deg(m, d, "torso_link", 1),
    # body part heights
    "torso_height": lambda m, d: float(get_xpos(m, d, "torso_link")[2]),
    "left_hand_height": lambda m, d: float(get_xpos(m, d, "left_rubber_hand")[2]),
    "right_hand_height": lambda m, d: float(get_xpos(m, d, "right_rubber_hand")[2]),
    "left_knee_height": lambda m, d: float(get_xpos(m, d, "left_knee_link")[2]),
    "right_knee_height": lambda m, d: float(get_xpos(m, d, "right_knee_link")[2]),
    "left_foot_height": lambda m, d: float(get_xpos(m, d, "left_ankle_roll_link")[2]),
    "right_foot_height": lambda m, d: float(get_xpos(m, d, "right_ankle_roll_link")[2]),
}


def _metrics_worker(x, model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    qpos, qvel = x
    fns = [METRICS[n] for n in names]
    data = mujoco.MjData(model)
    out = np.zeros((qpos.shape[0], len(names)), dtype=np.float32)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        out[i] = [fn(model, data) for fn in fns]
    return out


def compute_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    names: Sequence[str],
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
) -> Dict[str, np.ndarray]:
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}, available: {list(METRICS)}")
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    f = functools.partial(_metrics_worker, model=model, names=list(names))
    if max_workers == 1:
        result = [f(a) for a in args]
    elif process_executor:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context(process_context)) as exe:
            result = list(exe.map(f, args))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            result = list(exe.map(f, args))
    values = np.concatenate(result)
    return {n: values[:, j] for j, n in enumerate(names)}


def range_mask(metrics: Mapping[str, np.ndarray], ranges: Mapping[str, Range]) -> np.ndarray:
    """Frames whose metrics all lie in [lo, hi] (inclusive); None means unbounded."""
    n = len(next(iter(metrics.values())))
    mask = np.ones(n, dtype=bool)
    for name, (lo, hi) in ranges.items():
        if name not in metrics:
            raise KeyError(f"Metric '{name}' not in database, available: {list(metrics)}")
        if lo is not None:
            mask &= metrics[name] >= lo
        if hi is not None:
            mask &= metrics[name] <= hi
    return mask


def _take(x: Any, idx: np.ndarray) -> Any:
    if isinstance(x, Mapping):
        return {k: _take(v, idx) for k, v in x.items()}
    return x[idx]


@dataclasses.dataclass
class MotionDatabase:
    """A set of frames with their metric tags. All operations return a new MotionDatabase.

    metrics: name -> (N,) values
    source_index: (N,) index of each frame in the data reward_rank sampled from the buffer
    frames: per-frame data kept for later use (qpos, qvel, action, next_obs / B, ...)
    ranges: the range query that produced this subset
    """

    metrics: Dict[str, np.ndarray]
    source_index: np.ndarray
    frames: Dict[str, Any] = dataclasses.field(default_factory=dict)
    ranges: Dict[str, Range] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.source_index)

    @property
    def metric_names(self) -> List[str]:
        return list(self.metrics)

    def select(self, idx: np.ndarray) -> "MotionDatabase":
        # contiguous copy: torch tensors in frames cannot be indexed with negative-stride arrays
        idx = np.ascontiguousarray(idx)
        return MotionDatabase(
            metrics={k: v[idx] for k, v in self.metrics.items()},
            source_index=self.source_index[idx],
            frames=_take(self.frames, idx),
            ranges=dict(self.ranges),
        )

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """One frame as a record: metric values, source_index and frame data."""
        rec = {k: float(v[i]) for k, v in self.metrics.items()}
        rec["source_index"] = int(self.source_index[i])
        rec.update(_take(self.frames, i))
        return rec

    def filter(self, ranges: Mapping[str, Range] | None = None, **kw: Range) -> "MotionDatabase":
        """Narrow further, e.g. db.filter(base_pitch=(-5, 5))."""
        ranges = {**(ranges or {}), **kw}
        db = self.select(np.flatnonzero(range_mask(self.metrics, ranges)))
        db.ranges.update(ranges)
        return db

    def where(self, fn: Callable[[Dict[str, np.ndarray]], np.ndarray]) -> "MotionDatabase":
        """Arbitrary boolean predicate, e.g. db.where(lambda m: m["left_hand_height"] > m["right_hand_height"])."""
        return self.select(np.flatnonzero(fn(self.metrics)))

    def sort(self, by: str | Sequence[str], descending: bool = False) -> "MotionDatabase":
        """Sort by one metric, or several (first name is the primary key)."""
        keys = [by] if isinstance(by, str) else list(by)
        order = np.lexsort([self.metrics[k] for k in reversed(keys)])
        return self.select(order[::-1] if descending else order)

    def top_k(self, by: str, k: int, descending: bool = True) -> "MotionDatabase":
        return self.sort(by, descending=descending).select(np.arange(min(k, len(self))))

    def nearest(self, target: Mapping[str, float], k: int = 10, weights: Mapping[str, float] | None = None) -> "MotionDatabase":
        """k frames closest to target values, distance measured in per-metric std units."""
        weights = weights or {}
        dist = np.zeros(len(self))
        for name, value in target.items():
            v = self.metrics[name]
            std = v.std() + 1e-8
            dist += weights.get(name, 1.0) * ((v - value) / std) ** 2
        order = np.argsort(dist)[:k]
        return self.select(order)

    def describe(self):
        return self.to_dataframe().describe()

    def to_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(self.metrics)
        df.insert(0, "source_index", self.source_index)
        return df

    def save(self, path: str | Path) -> None:
        joblib.dump({f.name: getattr(self, f.name) for f in dataclasses.fields(self)}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MotionDatabase":
        return cls(**joblib.load(path))
