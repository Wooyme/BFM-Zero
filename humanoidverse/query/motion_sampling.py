"""Samplers that pick n rows of a query's frame table, written as a chain of stages, outermost first:

    query.sample(4000)
    .partition("grp", {"stand": 0.05, "translate": 0.85, "turn": 0.1}, spacing_floor=0.5)
    .stratify({"translate": grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05)), "turn": grid(yaw_rate=0.1, range=(-1.05, 1.05))}, by="grp")
    .prefer("symmetry_error", keep=0.5)
    .spread(weights)

    partition  split n over the values of a column (shares, or equal)
    stratify   split n equally over the cells of a grid on some metrics (uniform coverage)
    prefer     keep the rows lowest in a column (a share, but never fewer than n)
    spread     keep n rows pairwise apart in weighted qpos distance (similarity thinning)

With plot (a folder), each stratify saves the distribution of its picks over the grid as
stratify_<partition values>.png. A stratify can differ per value of an enclosing partition, named by its column: stratify({"translate":
..., "turn": ...}, by="grp") stratifies those values of grp and passes the others through; without by,
its grid applies to all rows. The chain builds a tree of the Sampler classes below
(Partition, Stratify, Prefer, Spread, Head). SymmetryError is a column function for
Query.with_columns, giving the column prefer sorts on.
"""

import abc
import dataclasses
import itertools
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import rich

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import qpos_of
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model, net_size, parallel_map, qpos_features, shared_threshold, symmetry_error, threshold_sample


def allocate(quota: Sequence[float], cap: Sequence[int], total: int) -> list[int]:
    """Water-fill total over the buckets in proportion to quota, none above its cap.

    A bucket whose proportional share exceeds its cap keeps its cap, and the rest is shared again
    among the others, until no share exceeds its cap (then fewer than total if all are capped).
    Shares are rounded by largest remainder, so they sum to exactly total.
    """
    quota, cap = np.asarray(quota, dtype=np.float64), np.asarray(cap, dtype=np.int64)
    out = np.zeros(len(quota), dtype=np.int64)
    free = quota > 0
    while free.any():
        share = (total - out[~free].sum()) * quota / quota[free].sum()
        over = free & (cap < share)
        if not over.any():
            # largest remainder: floor each share, then one more to the largest fractional parts
            idx = np.flatnonzero(free)
            base = np.floor(share[idx]).astype(np.int64)
            left = int(round(total - out[~free].sum() - base.sum()))
            base[np.argsort(-(share[idx] - base), kind="stable")[:left]] += 1
            out[idx] = base
            break
        out[over] = cap[over]
        free &= ~over
    return out.tolist()


@dataclasses.dataclass(frozen=True)
class Cell:
    """Grid axis of Stratify: cells of width size from lo to hi (edges at lo + k * size), values outside
    [lo, hi] in the outermost cells; e.g. Cell(0.15, -1.05, 1.05) is 7 cells each side of 0."""

    size: float
    lo: float
    hi: float

    @property
    def count(self) -> int:
        return int(np.ceil((self.hi - self.lo) / self.size - 1e-6))

    def edge(self, k: int) -> float:
        """Lower edge of cell k."""
        return self.lo + k * self.size


def grid(range: tuple[float, float], **sizes: float) -> dict[str, Cell]:
    """Grid of Stratify, the same range on each axis, e.g. grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05))."""
    return {name: Cell(size, *range) for name, size in sizes.items()}


def grid_cells(df: pd.DataFrame | Mapping[str, np.ndarray], cells: Mapping[str, Cell]) -> np.ndarray:
    """(N, len(cells)) cell index of each row on the grid of cells."""
    keys = []
    for name, cell in cells.items():
        keys.append(np.clip(np.floor((np.asarray(df[name]) - cell.lo) / cell.size), 0, cell.count - 1).astype(np.int64))
    return np.stack(keys, axis=-1)


def print_uniformity(cells: Mapping[str, Cell], keys: np.ndarray, counts: np.ndarray, cell_quota: Sequence[int], target: int):
    """Grid cells of a Stratify: how many are full, how many give all they have, and the grid."""
    cell_quota = np.asarray(cell_quota)
    short = cell_quota >= counts
    full = cell_quota[~short]
    print(
        f"Stratify {list(cells)}: {cell_quota.sum()}/{target} frames from {counts.sum()} candidates over {len(keys)} cells, "
        f"{(~short).sum()} at {full.max() if len(full) else '-'} frames, {short.sum()} short (all {cell_quota[short].sum()} candidates taken)"
    )
    names = list(cells)
    if len(names) == 2:
        lo, hi = keys.min(axis=0), keys.max(axis=0)
        grid = np.zeros(hi - lo + 1, dtype=np.int64)
        grid[tuple((keys - lo).T)] = cell_quota
        a, b = cells[names[0]], cells[names[1]]
        print(f"  rows {names[0]} from {a.edge(lo[0]):+.2f}, cols {names[1]} from {b.edge(lo[1]):+.2f}, step {a.size}/{b.size}")
        for r in grid:
            print("  " + " ".join(f"{v:4d}" for v in r))
    elif len(names) == 1:
        cell = cells[names[0]]
        print("  " + " ".join(f"{cell.edge(k[0]):+.2f}:{q}" for k, q in zip(keys, cell_quota)))


def plot_uniformity(cells: Mapping[str, Cell], keys: np.ndarray, counts: np.ndarray, selected: Sequence[int], path: Path, title: str):
    """Picks of a Stratify over its grid, saved to path: per axis, the picks (bars) and candidates
    (dots, log scale) of each cell summed over the other axes; per pair of axes, a heatmap of the
    picks (blank: no candidates)."""
    from matplotlib.figure import Figure  # no pyplot: no backend, safe in worker processes

    names = list(cells)
    selected = np.asarray(selected, dtype=np.float64)
    pairs = list(itertools.combinations(range(len(names)), 2))
    ncols = max(len(names), len(pairs))
    fig = Figure(figsize=(4.5 * ncols, 3.8 * (1 + bool(pairs))), layout="constrained")
    axes = fig.subplots(1 + bool(pairs), ncols, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, name in enumerate(names):
        cell = cells[name]
        x = cell.lo + (np.arange(cell.count) + 0.5) * cell.size
        ax = axes[0, i]
        ax.set_visible(True)
        ax.bar(x, np.bincount(keys[:, i], weights=selected, minlength=cell.count), width=0.9 * cell.size)
        ax.set_xlabel(name)
        ax.set_ylabel("selected")
        cand = np.bincount(keys[:, i], weights=counts, minlength=cell.count)
        twin = ax.twinx()
        twin.plot(x[cand > 0], cand[cand > 0], "k.--", lw=0.8)
        twin.set_yscale("log")
        twin.set_ylabel("candidates")
    for j, (a, b) in enumerate(pairs):
        ca, cb = cells[names[a]], cells[names[b]]
        heat = np.zeros((ca.count, cb.count))
        cand = np.zeros((ca.count, cb.count))
        np.add.at(heat, (keys[:, a], keys[:, b]), selected)
        np.add.at(cand, (keys[:, a], keys[:, b]), counts)
        heat[cand == 0] = np.nan
        ax = axes[1, j]
        ax.set_visible(True)
        extent = (ca.lo, ca.edge(ca.count), cb.lo, cb.edge(cb.count))
        im = ax.imshow(heat.T, origin="lower", extent=extent, aspect="auto", cmap="viridis", interpolation="nearest")
        if heat.size <= 400:
            for (u, v), h in np.ndenumerate(heat):
                if not np.isnan(h):
                    ax.text(ca.edge(u) + ca.size / 2, cb.edge(v) + cb.size / 2, f"{h:.0f}", ha="center", va="center", fontsize=6, color="w")
        ax.set_xlabel(names[a])
        ax.set_ylabel(names[b])
        fig.colorbar(im, ax=ax, label="selected")
    fig.suptitle(f"{title}: {int(selected.sum())} selected from {int(counts.sum())} candidates over {len(keys)} cells")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"  saved {path}")


def _sample(args) -> np.ndarray:
    sampler, df, n = args
    return sampler(df, n)


def _capacity(args) -> int:
    sampler, df, dist = args
    return sampler.capacity(df, dist)


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
    plot: Path | None = None  # folder of the distribution plot
    name: str = "all"  # values of the enclosing partitions, naming the plot

    def __call__(self, df: pd.DataFrame, n: int, min_dist: float | None = None) -> np.ndarray:
        if len(df) == 0 or n <= 0:
            return np.zeros(0, dtype=np.int64)
        keys, inverse, counts = np.unique(grid_cells(df, self.cells), axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        cell_quota = allocate([1] * len(keys), counts.tolist(), n)
        # cells are independent: sampled on the shared pool, the largest first
        members = [np.flatnonzero(inverse == c) for c in range(len(keys))]
        cells = sorted(range(len(keys)), key=lambda c: -len(members[c]))
        picked = parallel_map(_sample, [(self.then, df.iloc[members[c]], cell_quota[c]) for c in cells])
        sel = [members[c][p] for c, p in zip(cells, picked)]
        print_uniformity(self.cells, keys, counts, cell_quota, n)
        if self.plot is not None:
            selected = np.zeros(len(keys), dtype=np.int64)
            selected[cells] = [len(p) for p in picked]
            stem = re.sub(r"[^\w=.-]+", "_", self.name)
            plot_uniformity(self.cells, keys, counts, selected, Path(self.plot) / f"stratify_{stem}.png", f"Stratify {self.name}")
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
            cap = parallel_map(_capacity, [(self.child(k), df.iloc[groups[k]], floor) for k in keys])
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
    """Column function: the left-right symmetry error of each row (bench/motion_similarity.symmetry_error),
    pitch joints axially or centrally symmetric about the default pose, roll and yaw joints axially."""

    weights: Mapping | None = None

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        return symmetry_error(env_model(), qpos_of(df), self.weights).astype(np.float32)


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

    def _add(self, kind: str, *args, **kwargs) -> "Chain":
        if self.stages and self.stages[-1][0] == "spread":
            raise ValueError("spread must be the last stage")
        return dataclasses.replace(self, stages=self.stages + ((kind, args, kwargs),))

    def partition(self, by: str, share: Mapping[str, float] | None = None, spacing_floor: float | None = None, report: Sequence[str] = ("base_vx", "base_vy", "yaw_rate")) -> "Chain":
        """Split n over the values of column by, in proportion to share or equally. spacing_floor: see Partition."""
        return self._add("partition", by, share=share, spacing_floor=spacing_floor, report=tuple(report))

    def stratify(self, cells: Mapping, by: str | None = None, plot: Path | None = None) -> "Chain":
        """Uniform over the grid cells (see grid), for all rows; with by, cells is {value: grid} per value
        of the enclosing partition on column by, the values it lacks passed through. plot: folder of
        the distribution plot of each grid (plot_uniformity)."""
        return self._add("stratify", dict(cells), by=by, plot=plot)

    def prefer(self, by: str, keep: float) -> "Chain":
        """Keep the keep share of rows lowest in column by, but never fewer than n."""
        return self._add("prefer", by, keep)

    def spread(self, weights: Mapping | None = None, min_dist: float | None = None, seed: int = 0) -> "Chain":
        """Keep n rows pairwise apart in weighted qpos distance (bench/motion_similarity.py)."""
        return self._add("spread", weights=weights, min_dist=min_dist, seed=seed)

    def build(self) -> Sampler:
        return _build(self.stages, {})

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


def _build(stages: tuple, values: Mapping[str, object]) -> Sampler:
    """Sampler of stages, for the values {column: value} of the enclosing partitions."""
    if not stages:
        return Head()
    (kind, args, kwargs), rest = stages[0], stages[1:]
    if kind == "spread":
        return Spread(**kwargs)
    if kind == "prefer":
        return Prefer(*args, then=_build(rest, values))
    if kind == "stratify":
        by = kwargs["by"]
        if by is None:
            cells = args[0]
        elif by not in values:
            raise ValueError(f"stratify by {by!r}: no enclosing partition on {by!r}")
        else:
            cells = args[0].get(values[by])
        if cells is None:
            return _build(rest, values)
        name = ",".join(f"{k}={v}" for k, v in values.items() if v is not None) or "all"
        return Stratify(cells, then=_build(rest, values), plot=kwargs["plot"], name=name)
    # partition: one child per value named by share or by the per-value stages up to the next partition on by
    by = args[0]
    share = kwargs["share"]
    named = set()
    for stage in rest:
        if stage[0] == "partition" and stage[1][0] == by:
            break
        if stage[0] == "stratify" and stage[2]["by"] == by:
            named |= set(stage[1][0])
    if share is not None and named - set(share):
        raise ValueError(f"Partition {by}: stratify values {sorted(named - set(share))} missing from share {list(share)}")
    weights = next((stage[2]["weights"] for stage in rest if stage[0] == "spread"), None)
    return Partition(
        by,
        then={v: _build(rest, {**values, by: v}) for v in sorted(set(share or ()) | named)},
        default=_build(rest, {**values, by: None}),
        share=share,
        spacing_floor=kwargs["spacing_floor"],
        weights=weights,
        report=kwargs["report"],
    )
