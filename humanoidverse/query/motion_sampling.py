"""Samplers that pick n rows of a query's frame table, written as a chain of stages, outermost first:

    query.sample(4000)
    .partition("grp", {"stand": 0.05, "translate": 0.85, "turn": 0.1}, spacing_floor=0.5)
    .stratify(translate=grid(base_vx=0.15, base_vy=0.15, limit=1.05), turn=grid(yaw_rate=0.1, limit=1.05))
    .prefer("symmetry_error", keep=0.5)
    .spread(weights)

    partition  split n over the values of a column (shares, or equal)
    stratify   split n equally over the cells of a grid on some metrics (uniform coverage)
    prefer     keep the rows lowest in a column (a share, but never fewer than n)
    spread     keep n rows pairwise apart in weighted qpos distance (similarity thinning)

A stage after a partition can differ per value of it: stratify(translate=..., turn=...) stratifies
those values and passes the others through. The chain builds a tree of the Sampler classes below
(Partition, Stratify, Prefer, Spread, Head). SymmetryError is a column function for
Query.with_columns, giving the column prefer sorts on.
"""

import abc
import dataclasses
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import rich

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import qpos_of
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model, net_size, qpos_features, shared_threshold, symmetry_error, threshold_sample


def allocate(quota: Sequence[float], cap: Sequence[int], total: int) -> list[int]:
    """Water-fill total over the buckets in proportion to quota, none above its cap.

    A bucket whose proportional share exceeds its cap keeps its cap, and the rest is shared again
    among the others, until no share exceeds its cap (then fewer than total if all are capped).
    """
    quota, cap = np.asarray(quota, dtype=np.float64), np.asarray(cap, dtype=np.int64)
    out = np.zeros(len(quota), dtype=np.int64)
    free = quota > 0
    while free.any():
        share = (total - out[~free].sum()) * quota / quota[free].sum()
        over = free & (cap < share)
        if not over.any():
            out[free] = np.round(share[free]).astype(np.int64)
            break
        out[over] = cap[over]
        free &= ~over
    return out.tolist()


@dataclasses.dataclass(frozen=True)
class Cell:
    """Grid axis of Stratify: cells of width size aligned at 0, values beyond +-limit in the outermost cells."""

    size: float
    limit: float


def grid(limit: float, **sizes: float) -> dict[str, Cell]:
    """Grid of Stratify, e.g. grid(base_vx=0.15, base_vy=0.15, limit=1.05)."""
    return {name: Cell(size, limit) for name, size in sizes.items()}


def grid_cells(df: pd.DataFrame | Mapping[str, np.ndarray], cells: Mapping[str, Cell]) -> np.ndarray:
    """(N, len(cells)) cell index of each row on the grid of cells."""
    keys = []
    for name, cell in cells.items():
        n = int(np.ceil(cell.limit / cell.size - 1e-6))
        keys.append(np.clip(np.floor(np.asarray(df[name]) / cell.size), -n, n - 1).astype(np.int64))
    return np.stack(keys, axis=-1)


def print_uniformity(cells: Mapping[str, Cell], keys: np.ndarray, counts: np.ndarray, cell_quota: Sequence[int], target: int):
    """Grid cells of a Stratify: how many are full, how many give all they have, and the grid."""
    cell_quota = np.asarray(cell_quota)
    short = cell_quota >= counts
    full = cell_quota[~short]
    print(
        f"Stratify {list(cells)}: {cell_quota.sum()}/{target} frames over {len(keys)} cells, "
        f"{(~short).sum()} at {full.max() if len(full) else '-'} frames, {short.sum()} short (all {cell_quota[short].sum()} candidates taken)"
    )
    names = list(cells)
    if len(names) == 2:
        lo, hi = keys.min(axis=0), keys.max(axis=0)
        grid = np.zeros(hi - lo + 1, dtype=np.int64)
        grid[tuple((keys - lo).T)] = cell_quota
        a, b = cells[names[0]], cells[names[1]]
        print(f"  rows {names[0]} from {lo[0] * a.size:+.2f}, cols {names[1]} from {lo[1] * b.size:+.2f}, step {a.size}/{b.size}")
        for r in grid:
            print("  " + " ".join(f"{v:4d}" for v in r))
    elif len(names) == 1:
        size = cells[names[0]].size
        print("  " + " ".join(f"{k[0] * size:+.2f}:{q}" for k, q in zip(keys, cell_quota)))


class Sampler(abc.ABC):
    """Picks up to n rows of a candidate table; returns their positions (sorted) in df."""

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        """min_dist: floor on the pose distance of the kept rows, passed down by Partition's spacing_floor."""

    def pools(self, df: pd.DataFrame) -> list[np.ndarray]:
        """Row sets the sampler would thin with no quota, for Partition's shared threshold."""
        return [np.arange(len(df))]

    def capacity(self, df: pd.DataFrame, dist: float) -> int:
        """Rows the sampler can keep at pose distance dist; unbounded samplers keep all."""
        return len(df)


@dataclasses.dataclass
class Head(Sampler):
    """The first n rows, in table order."""

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        return np.arange(min(n, len(df)))


@dataclasses.dataclass
class Spread(Sampler):
    """n rows pairwise apart in weighted qpos distance (bench/motion_similarity.threshold_sample).

    The distance threshold is bisected to fill n, but not below min_dist (then fewer rows).
    """

    weights: Mapping | None = None
    min_dist: float | None = None
    seed: int = 0

    def features(self, df: pd.DataFrame) -> np.ndarray:
        return qpos_features(env_model(), qpos_of(df), self.weights)

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        floors = [d for d in (self.min_dist, min_dist) if d is not None]
        sel, _ = threshold_sample(self.features(df), n, max(floors) if floors else None, self.seed)
        return np.sort(sel)

    def capacity(self, df: pd.DataFrame, dist: float) -> int:
        return net_size(self.features(df), dist, self.seed)


@dataclasses.dataclass
class Prefer(Sampler):
    """Sort: the keep share of rows lowest in column by, but never fewer than n, handed to then."""

    by: str
    keep: float
    then: Sampler

    def pool(self, df: pd.DataFrame, n: int = 0) -> np.ndarray:
        k = min(len(df), max(int(np.ceil(self.keep * len(df))), n))
        return np.sort(np.argsort(df[self.by].to_numpy(), kind="stable")[:k])

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        pool = self.pool(df, n)
        return pool[self.then(df.iloc[pool], n, min_dist)]

    def pools(self, df: pd.DataFrame) -> list[np.ndarray]:
        pool = self.pool(df)
        return [pool[p] for p in self.then.pools(df.iloc[pool])]

    def capacity(self, df: pd.DataFrame, dist: float) -> int:
        pool = self.pool(df)
        return self.then.capacity(df.iloc[pool], dist)


@dataclasses.dataclass
class Stratify(Sampler):
    """Uniform coverage of some metrics: grid cells share n equally, a cell short of rows gives all it has.

    Coverage comes first: the cells ignore the min_dist of a Partition's spacing_floor and the
    sampler is never capped by it.
    """

    cells: Mapping[str, Cell]
    then: Sampler

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        if len(df) == 0 or n <= 0:
            return np.zeros(0, dtype=np.int64)
        keys, inverse, counts = np.unique(grid_cells(df, self.cells), axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        cell_quota = allocate([1] * len(keys), counts.tolist(), n)
        sel = []
        for c, q in enumerate(cell_quota):
            members = np.flatnonzero(inverse == c)
            sel.append(members[self.then(df.iloc[members], q)])
        print_uniformity(self.cells, keys, counts, cell_quota, n)
        return np.sort(np.concatenate(sel))

    def pools(self, df: pd.DataFrame) -> list[np.ndarray]:
        return self.then.pools(df)


@dataclasses.dataclass
class Partition(Sampler):
    """Split n over the values of column by: in proportion to share, or equally; one sampler per value.

    then is one sampler for all values, or {value: sampler} with default for the values it lacks
    (None: those values are an error). With spacing_floor, a partition too dense
    to keep its quota with rows spacing_floor * shared threshold apart keeps fewer (Sampler.capacity),
    and its leftover goes to the others in proportion to their quota; the shared threshold is the one
    distance threshold that, thinning the pools of all partitions at once, keeps n rows.
    report: metrics whose range is printed per partition.
    """

    by: str
    then: Sampler | Mapping[str, Sampler]
    default: Sampler | None = None
    share: Mapping[str, float] | None = None
    spacing_floor: float | None = None
    weights: Mapping | None = None
    seed: int = 0
    report: Sequence[str] = ("base_vx", "base_vy", "yaw_rate")

    def groups(self, df: pd.DataFrame) -> dict:
        values = df[self.by].to_numpy()
        present = sorted(pd.unique(values).tolist())
        keys = present if self.share is None else list(self.share)
        unknown = set(present) - set(keys)
        if unknown:
            raise ValueError(f"Partition {self.by}: values {sorted(unknown)} missing from share {list(keys)}")
        return {k: np.flatnonzero(values == k) for k in keys}

    def child(self, key) -> Sampler:
        if not isinstance(self.then, Mapping):
            return self.then
        if key not in self.then and self.default is None:
            raise ValueError(f"Partition {self.by}: no sampler for value {key!r}")
        return self.then.get(key, self.default)

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        groups = self.groups(df)
        keys = list(groups)
        if self.share is None:
            quota = [n / len(keys)] * len(keys)
        else:
            quota = [n * self.share[k] for k in keys]
        quota = [int(round(q)) for q in quota]
        rows = [{self.by: k, "candidates": len(groups[k])} for k in keys]
        if self.spacing_floor:
            feats = [qpos_features(env_model(), qpos_of(df.iloc[groups[k][p]]), self.weights) for k in keys for p in self.child(k).pools(df.iloc[groups[k]])]
            shared = shared_threshold(feats, n, self.seed)
            floor = self.spacing_floor * shared
            cap = [self.child(k).capacity(df.iloc[groups[k]], floor) for k in keys]
            quota = allocate(quota, cap, n)
            min_dist = max(floor, min_dist or 0.0)
            print(f"Partition {self.by}: shared threshold {shared:.3f}, spacing floor {floor:.3f}")
            for row, c in zip(rows, cap):
                row["cap"] = c
        sel = []
        for k, q, row in zip(keys, quota, rows):
            g = groups[k]
            picked = g[self.child(k)(df.iloc[g], q, min_dist)]
            sel.append(picked)
            row.update({"quota": q, "selected": len(picked)})
            for name in self.report:
                if name in df:
                    v = df[name].to_numpy()[picked]
                    row[name] = f"[{v.min():+.2f}, {v.max():+.2f}]" if len(v) else "-"
            if len(picked) < q and not isinstance(self.child(k), Stratify):
                print(f"WARNING: {self.by} {k} has {len(picked)} frames for a quota of {q}")
        rich.print(pd.DataFrame(rows).to_string())
        return np.sort(np.concatenate(sel)) if sel else np.zeros(0, dtype=np.int64)

    def pools(self, df: pd.DataFrame) -> list[np.ndarray]:
        return [g[p] for k, g in self.groups(df).items() for p in self.child(k).pools(df.iloc[g])]

    def capacity(self, df: pd.DataFrame, dist: float) -> int:
        return sum(self.child(k).capacity(df.iloc[g], dist) for k, g in self.groups(df).items())


@dataclasses.dataclass(frozen=True)
class SymmetryError:
    """Column function: the better of axial and central left-right symmetry error of each row
    (bench/motion_similarity.symmetry_error), central about the mean pose of its by partition."""

    by: str | None = None
    weights: Mapping | None = None

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        qpos = qpos_of(df)
        err = np.zeros(len(df), dtype=np.float32)
        parts = [np.arange(len(df))] if self.by is None else [np.flatnonzero(df[self.by].to_numpy() == k) for k in pd.unique(df[self.by])]
        for rows in parts:
            if len(rows):
                axial, central = symmetry_error(env_model(), qpos[rows], self.weights)
                err[rows] = np.minimum(axial, central)
        return err


@dataclasses.dataclass(frozen=True)
class Chain(Sampler):
    """A sampler written as a chain of stages, see the module docstring; each method returns a new Chain.

    Started from Query.sample(n), collect / to_database run the query and keep the n sampled rows;
    started from sample(), it is a Sampler to call on a table. It builds a tree of Samplers
    (build()); a chain that does not end in spread keeps the first rows of its last stage (Head).
    """

    stages: tuple = ()
    query: object = None  # motion_query.Query
    n: int | None = None

    def _add(self, kind: str, *args, per_value: Mapping | None = None, **kwargs) -> "Chain":
        if self.stages and self.stages[-1][0] == "spread":
            raise ValueError("spread must be the last stage")
        return dataclasses.replace(self, stages=self.stages + ((kind, args, kwargs, dict(per_value or {})),))

    def partition(self, by: str, share: Mapping[str, float] | None = None, spacing_floor: float | None = None, report: Sequence[str] = ("base_vx", "base_vy", "yaw_rate")) -> "Chain":
        """Split n over the values of column by, in proportion to share or equally. spacing_floor: see Partition."""
        return self._add("partition", by, share=share, spacing_floor=spacing_floor, report=tuple(report))

    def stratify(self, cells: Mapping[str, Cell] | None = None, **per_value: Mapping[str, Cell]) -> "Chain":
        """Uniform over the grid cells (see grid); or per value of the enclosing partition, others passed through."""
        if (cells is None) == (not per_value):
            raise ValueError("stratify takes cells, or cells per value of the enclosing partition")
        return self._add("stratify", cells, per_value=per_value)

    def prefer(self, by: str, keep: float) -> "Chain":
        """Keep the keep share of rows lowest in column by, but never fewer than n."""
        return self._add("prefer", by, keep)

    def spread(self, weights: Mapping | None = None, min_dist: float | None = None, seed: int = 0) -> "Chain":
        """Keep n rows pairwise apart in weighted qpos distance (bench/motion_similarity.py)."""
        return self._add("spread", weights=weights, min_dist=min_dist, seed=seed)

    def build(self) -> Sampler:
        return _build(self.stages, None)

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        return self.build()(df, n, min_dist)

    def _query(self):
        if self.query is None:
            raise ValueError("Start the chain from Query.sample(n) to collect it")
        return self.query.apply(_Take(self.build(), self.n))

    def collect(self) -> pd.DataFrame:
        """Run the query and keep the n rows the chain samples."""
        return self._query().collect()

    def to_database(self):
        """The sampled rows as a MotionDatabase, see Query.to_database."""
        return self._query().to_database()


@dataclasses.dataclass(frozen=True)
class _Take:
    """Query step keeping the rows a sampler picks."""

    sampler: Sampler
    n: int

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.sampler(df, self.n)]


def sample() -> Chain:
    """Start a sampler chain without a query: sample().partition(...)...(df, n)."""
    return Chain()


def _build(stages: tuple, value) -> Sampler:
    """Sampler of stages, for value of the enclosing partition (None outside one / for other values)."""
    if not stages:
        return Head()
    (kind, args, kwargs, per_value), rest = stages[0], stages[1:]
    if kind == "spread":
        return Spread(**kwargs)
    if kind == "prefer":
        return Prefer(*args, then=_build(rest, value))
    if kind == "stratify":
        cells = per_value.get(value) if per_value else args[0]
        return Stratify(cells, then=_build(rest, value)) if cells is not None else _build(rest, value)
    # partition: one child per value named by share or by the per-value stages up to the next partition
    by = args[0]
    values = set(kwargs["share"] or ())
    for stage in rest:
        if stage[0] == "partition":
            break
        values |= set(stage[3])
    weights = next((stage[2]["weights"] for stage in rest if stage[0] == "spread"), None)
    return Partition(
        by,
        then={v: _build(rest, v) for v in sorted(values)},
        default=_build(rest, None),
        share=kwargs["share"],
        spacing_floor=kwargs["spacing_floor"],
        weights=weights,
        report=kwargs["report"],
    )
