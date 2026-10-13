"""Per-frame metric tables of buffer frames, and MotionDatabase to keep and search selected frames.

compute_metrics evaluates METRICS (queries/metrics.py) on every frame; motion queries then select
rows with SQL through sql (DuckDB), see humanoidverse/query/motion_query.py.
"""

import dataclasses
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Sequence, Tuple

import joblib
import mujoco
import numpy as np
import pandas as pd

from humanoidverse.envs.g1_env_helper.queries import METRICS

Range = Tuple[float | None, float | None]

QPOS_PREFIX = "qpos_"


def sql(statement: str, params: Mapping[str, Any] | None = None, **tables: pd.DataFrame) -> pd.DataFrame:
    """Run a DuckDB SQL statement over DataFrames registered under their keyword names, with
    $name parameters bound from params, e.g.
    sql("SELECT * FROM frames WHERE base_speed > $v", {"v": 0.5}, frames=df)."""
    import duckdb

    con = duckdb.connect()
    try:
        for name, df in tables.items():
            con.register(name, df)
        return (con.execute(statement, dict(params)) if params else con.sql(statement)).df()
    finally:
        con.close()


def qpos_columns(nq: int) -> List[str]:
    return [f"{QPOS_PREFIX}{i}" for i in range(nq)]


def qpos_of(df: pd.DataFrame) -> np.ndarray:
    """(N, nq) qpos kept in the qpos_<i> columns of a frame table."""
    nq = sum(c.startswith(QPOS_PREFIX) for c in df.columns)
    return df[qpos_columns(nq)].to_numpy()


def _metrics_worker(x, model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    qpos, qvel = x
    fns = [METRICS[n] for n in names]
    data = mujoco.MjData(model)
    out = np.empty((qpos.shape[0], len(names)), dtype=np.float64)
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
    """The metrics in names of every frame, (N,) float64 each."""
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}, available: {list(METRICS)}")
    if qpos.shape[0] == 0:
        return {n: np.zeros(0) for n in names}
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


def parse_bounds(cfg: Mapping[str, float | None], names: Collection[str]) -> Dict[str, Range]:
    """{"min_x": 1, "max_x": 2, "max_y": 3} -> {"x": (1, 2), "y": (None, 3)}, for x, y in names."""
    bounds = {}
    for key, value in cfg.items():
        side, _, name = key.partition("_")
        if side not in ("min", "max") or name not in names:
            raise ValueError(f"Expected min_<name> / max_<name>, got '{key}'; names: {list(names)}")
        lo, hi = bounds.get(name, (None, None))
        bounds[name] = (value, hi) if side == "min" else (lo, value)
    return bounds


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
    source_index: (N,) index of each frame in the source it was selected from (buffer / LAFAN frames)
    frames: per-frame data kept for later use (qpos, qvel, action, next_obs / B, ...)
    labels: text columns stored as codes in metrics -> their values (code i is labels[name][i])
    recipe: the query that produced this set (SQL condition / query script), for provenance
    """

    metrics: Dict[str, np.ndarray]
    source_index: np.ndarray
    frames: Dict[str, Any] = dataclasses.field(default_factory=dict)
    labels: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    recipe: str | None = None

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
            labels=self.labels,
            recipe=self.recipe,
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
        """A SQL WHERE condition over the metrics and source_index (DuckDB, as in motion_query recipes),
        e.g. db.where("base_speed > 0.5 AND abs(move_angle) < 30"), or a boolean predicate on the
        metrics, e.g. db.where(lambda m: m["left_hand_height"] > m["right_hand_height"])."""
        if isinstance(fn, str):
            df = self.to_dataframe()
            df.insert(0, "row_", np.arange(len(df)))
            return self.select(sql(f"SELECT row_ FROM frames WHERE {fn} ORDER BY row_", frames=df)["row_"].to_numpy())
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

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metrics)
        df.insert(0, "source_index", self.source_index)
        return df

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({f.name: getattr(self, f.name) for f in dataclasses.fields(self)}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MotionDatabase":
        state = joblib.load(path)
        state.pop("query", None)  # databases saved before recipes kept a QueryFunction here
        return cls(**state)
