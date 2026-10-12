"""Run a QueryFunction over buffer frames and keep the matches in a searchable MotionDatabase.

Used by RewardWrapperHV.query_motions. Queries live in g1_env_helper/queries; metrics that tag
each matching frame (for sorting / filtering afterwards) are METRICS in queries/metrics.py.
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

from humanoidverse.envs.g1_env_helper.queries import METRICS, QueryFunction, parse_bounds

Range = Tuple[float | None, float | None]


def _query_worker(x, model: mujoco.MjModel, query: QueryFunction, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    offset, qpos, qvel = x
    fns = [METRICS[n] for n in names]
    data = mujoco.MjData(model)
    idx, out = [], []
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        if query.match(model, data):
            idx.append(offset + i)
            out.append([fn(model, data) for fn in fns])
    return np.array(idx, dtype=np.int64), np.array(out, dtype=np.float32).reshape(len(idx), len(names))


def run_query(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    query: QueryFunction,
    names: Sequence[str],
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Indices of the frames matching query, and the metrics in names for those frames."""
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}, available: {list(METRICS)}")
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(i, qpos[i : i + chunk_size], qvel[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    f = functools.partial(_query_worker, model=model, query=query, names=list(names))
    if max_workers == 1:
        result = [f(a) for a in args]
    elif process_executor:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context(process_context)) as exe:
            result = list(exe.map(f, args))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            result = list(exe.map(f, args))
    idx = np.concatenate([r[0] for r in result])
    values = np.concatenate([r[1] for r in result])
    return idx, {n: values[:, j] for j, n in enumerate(names)}


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
    source_index: (N,) index of each frame in the data query_motions sampled from the buffer
    frames: per-frame data kept for later use (qpos, qvel, action, next_obs / B, ...)
    query: the QueryFunction that produced this subset
    """

    metrics: Dict[str, np.ndarray]
    source_index: np.ndarray
    frames: Dict[str, Any] = dataclasses.field(default_factory=dict)
    query: QueryFunction | None = None

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
            query=self.query,
        )

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """One frame as a record: metric values, source_index and frame data."""
        rec = {k: float(v[i]) for k, v in self.metrics.items()}
        rec["source_index"] = int(self.source_index[i])
        rec.update(_take(self.frames, i))
        return rec

    def filter(self, **bounds: float | None) -> "MotionDatabase":
        """Narrow further with min_<metric> / max_<metric>, e.g. db.filter(min_base_pitch=-5, max_base_pitch=5)."""
        return self.select(np.flatnonzero(range_mask(self.metrics, parse_bounds(bounds, self.metrics))))

    def where(self, fn: str | Callable[[Dict[str, np.ndarray]], np.ndarray]) -> "MotionDatabase":
        """Arbitrary boolean predicate, e.g. db.where(lambda m: m["left_hand_height"] > m["right_hand_height"]),
        or a pandas.DataFrame.query expression over the metrics and source_index,
        e.g. db.where("base_speed > 0.5 and abs(move_angle) < 30")."""
        if isinstance(fn, str):
            return self.select(self.to_dataframe().query(fn).index.to_numpy())
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
