"""Select frames of a replay buffer (or of LAFAN motions) with a chain of SQL-like operations
over their per-frame metrics (METRICS in envs/g1_env_helper/queries/metrics.py).

    db = (
        Source(model_folder).scan()                       # lazy: nothing runs until collect / to_database
        .where("base_height >= 0.6 AND base_speed BETWEEN 0.5 AND 1.0")
        .with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))")
        .case("bucket", {"slow": "cmd_speed < 0.5", "fast": "cmd_speed >= 0.5"})
        .order_by("base_speed DESC")
        .to_database()
    )

Conditions and expressions are DuckDB SQL over the columns of the table: frame (index in the
source), origin (where the frame comes from, "<file>:<frame index>"), every metric and
qpos_0 .. qpos_<nq-1>. When a query is easier to read as one statement,
write it raw, inline or from a .sql file, with $name parameters:

    Source(model_folder).scan().sql('''
        SELECT *, CASE WHEN base_speed < $slow THEN 'slow' ELSE 'fast' END AS tier
        FROM frames
        WHERE base_height >= 0.6 AND abs(move_angle) <= 30
    ''', per_chunk=True, slow=0.7)

The leading row-local operations (where, case, with_columns given SQL expressions, sql with
per_chunk) run while scanning, chunk by chunk, so the buffer is never held whole; the others
(order_by, limit, sql, Python column functions, sample) run on the rows that pass them.
sample(n) continues the chain with the samplers of motion_sampling.

Static stability (the pose_hold score of a pose, query/stability.py) is a column like the others,
only evaluated in Isaac Sim when not cached, so put it after the cheap conditions:

    Source(model_folder).scan().where("base_speed <= 0.1").stable(90).to_database()

This is a library: the queries themselves are scripts in scripts/queries.
"""

from concurrent.futures import ProcessPoolExecutor
import dataclasses
from fnmatch import fnmatch
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping

import h5py
import mujoco
import numpy as np
import pandas as pd
import torch

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase, QPOS_PREFIX, compute_metrics, qpos_columns, sql
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model
from humanoidverse.envs.g1_env_helper.queries import METRICS

HUMANOIDVERSE_DIR = Path(__file__).resolve().parent.parent

# Python column function: the table -> one value per row
ColumnFn = Callable[[pd.DataFrame], np.ndarray]


class H5Frames:
    """Frames of a saved replay buffer read straight from its buffer.hdf5, a slice at a time.

    The reduced buffer (N, ...) keeps next-qpos / next-observation-*; the trajectory buffer
    (T, E, ...) keeps qpos / observation-* of the same step, flattened to frame t * E + e.
    """

    def __init__(self, path: Path):
        self.file = h5py.File(path, "r")
        self.name = _display_path(path)
        self.prefix = "next-" if "next-qpos" in self.file else ""
        qpos = self.file[f"{self.prefix}qpos"]
        self.lead = qpos.shape[:-1]
        self.num_frames = int(np.prod(self.lead))
        self.obs_keys = [k for k in self.file if k.startswith(f"{self.prefix}observation-")]

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        """Frames [start, stop) of key; for (T, E, ...) whole rows of E frames, so starts are row aligned."""
        ds = self.file[key]
        if len(self.lead) == 1:
            return ds[start:stop]
        e = self.lead[1]
        return ds[start // e : -(-stop // e)].reshape(-1, *ds.shape[2:])[: stop - start]

    def gather(self, key: str, idx: np.ndarray) -> np.ndarray:
        """Frames idx (sorted, unique) of key."""
        ds = self.file[key]
        if len(self.lead) == 1:
            return ds[idx]
        e = self.lead[1]
        rows = idx // e
        out = np.empty((len(idx), *ds.shape[2:]), dtype=ds.dtype)
        for row in np.unique(rows):
            sel = rows == row
            out[sel] = ds[row][idx[sel] % e]
        return out

    def gather_frames(self, idx: np.ndarray) -> dict:
        """qpos / qvel / action / next_obs of frames idx, laid out like query_motions keep_frames."""
        frames = {name: self.gather(f"{self.prefix}{name}", idx) for name in ("qpos", "qvel")}
        if "action" in self.file:
            frames["action"] = self.gather("action", idx)
        frames["next_obs"] = {k.split("-")[-1]: self.gather(k, idx) for k in self.obs_keys}
        return frames

    def origins(self, idx: np.ndarray) -> np.ndarray:
        """"<buffer.hdf5>:<frame>" of frames idx."""
        return np.char.add(f"{self.name}:", np.asarray(idx).astype(str))


class LafanFrames:
    """Frames of LAFAN motions sampled at the env dt by the motion library of an Isaac env.

    Same interface as H5Frames (a flat (N, ...) source): qpos / qvel for the query, and the
    backward observations of get_backward_observation (velocity_multiplier 1) as next_obs, so
    goal inference sees the same inputs as for frames of the replay buffer. motion_frame is the
    frame of the motion file nearest to each sample, as pose_hold's frame_index counts them.
    """

    prefix = ""

    def __init__(self, model_folder: Path, files: list[Path], device: str = "cuda"):
        from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
        from humanoidverse.utils.helpers import get_backward_observation
        from humanoidverse.utils.torch_utils import quat_rotate_inverse

        with open(model_folder / "config.json") as f:
            env_config = json.load(f)["env"]
        use_root_height_obs = env_config.get("root_height_obs", False)
        env_config["hydra_overrides"] += ["env.config.max_episode_length_s=10000", "env.config.headless=True", "simulator=isaacsim"]
        env_config["disable_domain_randomization"] = True
        env_config["disable_obs_noise"] = True
        data = {"qpos": [], "qvel": [], "motion": [], "motion_time": [], "motion_frame": []}
        next_obs = []
        # the motion library loads files lazily by index, so they must outlive the loop
        with tempfile.TemporaryDirectory() as motion_dir:
            for file in files:
                (Path(motion_dir) / file.name).symlink_to(file.resolve())
            env_config["lafan_tail_path"] = motion_dir
            wrapped_env, _ = HumanoidVerseIsaacConfig(**env_config).build(1)
            env = wrapped_env._env
            lib = env._motion_lib
            for i in range(lib._num_unique_motions):
                env.set_is_evaluating(i)  # loads motion i into slot 0
                name = Path(str(lib.curr_motion_keys[0])).stem
                obs, _ = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=1.0)
                n = next(iter(obs.values())).shape[0]
                times = torch.arange(n, device=env.device) * env.dt
                state = lib.get_motion_state(torch.zeros(n, dtype=torch.long, device=env.device), times)
                root_rot = state["root_rot"]  # xyzw
                body_ang_vel = quat_rotate_inverse(root_rot, state["root_ang_vel"], w_last=True)
                qpos = torch.cat([state["root_pos"], root_rot[:, [3, 0, 1, 2]], state["dof_pos"]], dim=-1)
                qvel = torch.cat([state["root_vel"], body_ang_vel, state["dof_vel"]], dim=-1)
                data["qpos"].append(qpos.cpu().numpy())
                data["qvel"].append(qvel.cpu().numpy())
                data["motion"].append(np.full(n, name))
                data["motion_time"].append(times.cpu().numpy())
                data["motion_frame"].append(torch.round(times * lib._motion_fps[0]).long().cpu().numpy())
                next_obs.append({k: v.cpu().numpy() for k, v in obs.items()})
                print(f"  {name}: {n} frames")
        self.data = {k: np.concatenate(v) for k, v in data.items()}
        self.next_obs = {k: np.concatenate([o[k] for o in next_obs]) for k in next_obs[0]}
        self.num_frames = len(self.data["qpos"])
        self.lead = (self.num_frames,)

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        return self.data[key][start:stop]

    def gather_frames(self, idx: np.ndarray) -> dict:
        """qpos / qvel / next_obs of frames idx, with the motion name and time (s) of each."""
        frames = {k: v[idx] for k, v in self.data.items()}
        frames["next_obs"] = {k: v[idx] for k, v in self.next_obs.items()}
        return frames

    def origins(self, idx: np.ndarray) -> np.ndarray:
        """"<motion>:<motion_frame>" of frames idx."""
        return np.char.add(np.char.add(self.data["motion"][idx].astype(str), ":"), self.data["motion_frame"][idx].astype(str))


def lafan_files(lafan_dir: Path, patterns: str) -> list[Path]:
    """Motion files of lafan_dir whose name matches one of the comma-separated glob patterns."""
    globs = [p.strip() for p in patterns.split(",") if p.strip()]
    files = [f for f in sorted(Path(lafan_dir).glob("*.pkl")) if any(fnmatch(f.stem, g) for g in globs)]
    if not files:
        raise ValueError(f"No motion in {lafan_dir} matches {globs}")
    return files


def _display_path(path: Path) -> str:
    """path relative to the repository when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(HUMANOIDVERSE_DIR.parent)) if path.is_relative_to(HUMANOIDVERSE_DIR.parent) else str(path)


def frame_table(start: int, stride: int, qpos: np.ndarray, metrics: dict, origin: np.ndarray) -> pd.DataFrame:
    """Rows of consecutive frames start, start + stride, ...: frame, origin, metrics and qpos_<i>."""
    df = pd.DataFrame({"frame": start + np.arange(len(qpos), dtype=np.int64) * stride, "origin": origin, **metrics})
    return pd.concat([df, pd.DataFrame(qpos.astype(np.float32), columns=qpos_columns(qpos.shape[1]))], axis=1)


@dataclasses.dataclass(frozen=True)
class SqlOp:
    """A DuckDB statement over the table `frames`; row_local when it looks at one row at a time.

    params: $name parameters, tables: more tables the statement reads, as (name, value) pairs.
    """

    statement: str
    row_local: bool
    params: tuple = ()
    tables: tuple = ()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return sql(self.statement, dict(self.params), frames=df, **dict(self.tables))


@dataclasses.dataclass(frozen=True)
class SqlColumnOp:
    """Set column name to a row-local SQL expression, replacing the column if present."""

    name: str
    expression: str
    row_local: bool = True

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = f"* EXCLUDE ({self.name})" if self.name in df.columns else "*"
        return sql(f"SELECT {keep}, {self.expression} AS {self.name} FROM frames", frames=df)


@dataclasses.dataclass(frozen=True)
class ColumnOp:
    """Set column name to a Python column function of the table."""

    name: str
    fn: ColumnFn
    row_local: bool = False

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{self.name: np.asarray(self.fn(df))})


@dataclasses.dataclass(frozen=True)
class StabilityOp:
    """Add columns of the pose_hold scores of the rows' qpos ({name: cache column}), from the stability cache."""

    columns: tuple  # (name, cache column) pairs
    cache: object  # StabilityCache, opened on first use
    row_local: bool = False

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        from humanoidverse.envs.g1_env_helper.bench.motion_metrics import qpos_of
        from humanoidverse.query.stability import StabilityCache

        if df.empty:
            return df.assign(**{name: np.empty(0, dtype=np.float32) for name, _ in self.columns})
        if "origin" not in df.columns:
            raise ValueError("Stability needs the origin column of the frame table")
        cache = self.cache() if self.cache is StabilityCache else self.cache
        scores = cache.scores(qpos_of(df), df["origin"].to_numpy().astype(str), tuple(column for _, column in self.columns))
        return df.assign(**{name: scores[column].to_numpy().astype(np.float32) for name, column in self.columns})


@dataclasses.dataclass(frozen=True)
class Query:
    """A lazy chain of operations over the frame table of a source; each method returns a new Query.

    Run it with collect (the table) or to_database (a MotionDatabase with the frame data). frames
    is a frame source (H5Frames / LafanFrames), or a table already collected to query further.
    """

    frames: object  # H5Frames | LafanFrames | pd.DataFrame
    ops: tuple = ()
    chunk: int = 100_000
    max_frames: int | None = None
    stride: int = 1
    max_workers: int = os.cpu_count() or 1

    def _then(self, op) -> "Query":
        return dataclasses.replace(self, ops=self.ops + (op,))

    def where(self, condition: str, **params) -> "Query":
        """Keep the rows matching a SQL condition; successive wheres AND together. params bind
        $name parameters, e.g. where("base_speed >= $v", v=speed)."""
        return self._then(SqlOp(f"SELECT * FROM frames WHERE {condition}", row_local=True, params=tuple(params.items())))

    def with_columns(self, **columns: str | ColumnFn) -> "Query":
        """Add or replace columns: a SQL expression (row-local) or a Python column function, e.g.
        with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))", symmetry_error=SymmetryError())."""
        query = self
        for name, column in columns.items():
            query = query._then(SqlColumnOp(name, column) if isinstance(column, str) else ColumnOp(name, column))
        return query

    def case(self, name: str, cases: Mapping[str, str], default: str | None = None) -> "Query":
        """Column name holding the label of the first condition a row matches ({label: condition}), else default."""
        whens = " ".join(f"WHEN {cond} THEN {_literal(label)}" for label, cond in cases.items())
        return self._then(SqlColumnOp(name, f"CASE {whens} ELSE {_literal(default)} END"))

    def order_by(self, *keys: str) -> "Query":
        """Sort the rows, e.g. order_by("base_speed DESC", "frame")."""
        return self._then(SqlOp(f"SELECT * FROM frames ORDER BY {', '.join(keys)}", row_local=False))

    def limit(self, n: int) -> "Query":
        return self._then(SqlOp(f"SELECT * FROM frames LIMIT {int(n)}", row_local=False))

    def sql(self, statement: str | Path, per_chunk: bool = False, **bindings) -> "Query":
        """A raw DuckDB statement over the table `frames` (the rows reaching it), e.g. a whole query
        with CTEs, CASE, windows and joins. It must keep the frame, origin and qpos_<i> columns (SELECT *).

        statement: the SQL, or the Path of a .sql file holding it.
        per_chunk: the statement looks at one row at a time (no ORDER BY, LIMIT, DISTINCT, GROUP BY,
            aggregates or window functions), so at the head of the chain it runs while scanning,
            chunk by chunk, like where. Otherwise it runs once on all the rows that reach it.
        bindings: a DataFrame binds a table of that name, any other value a $name parameter, e.g.
            sql("SELECT f.* FROM frames f JOIN keep USING (frame) WHERE base_speed > $v", keep=df, v=0.5).
        """
        if isinstance(statement, Path):
            statement = statement.read_text()
        tables = tuple((k, v) for k, v in bindings.items() if isinstance(v, pd.DataFrame))
        params = tuple((k, v) for k, v in bindings.items() if not isinstance(v, pd.DataFrame))
        return self._then(SqlOp(statement, row_local=per_chunk, params=params, tables=tables))

    def with_stability(self, columns: Mapping[str, str] | None = None, cache=None) -> "Query":
        """Add the static stability of each row: the pose_hold score (0..100) of the BFM policy holding
        its pose from rest, read from the stability cache, the poses not cached yet evaluated in Isaac
        Sim batches and added to it. Not row-local, so it runs on the rows that reach it.

        columns: {name: cache column} (columns of query.stability.SCORE_COLUMNS, e.g. base_score,
            joint_score, distance_score), default {"stability": "score"}.
        cache: a StabilityCache (another model, evaluation settings or batch size), default its defaults.
        """
        from humanoidverse.query.stability import StabilityCache

        columns = columns or {"stability": "score"}
        return self._then(StabilityOp(tuple(columns.items()), cache or StabilityCache))

    def stable(self, min_score: float, cache=None) -> "Query":
        """Keep the rows whose static stability (column stability, see with_stability) is >= min_score."""
        return self.with_stability(cache=cache).where("stability >= $min_score", min_score=float(min_score))

    def sample(self, n: int):
        """Continue with samplers that pick n rows: .sample(n).partition(...).stratify(...).prefer(...).spread(...)."""
        from humanoidverse.query.motion_sampling import Chain

        return Chain(query=self, n=n)

    def apply(self, op: Callable[[pd.DataFrame], pd.DataFrame]) -> "Query":
        """Any DataFrame -> DataFrame step on the rows that reach it (not row-local)."""
        return self._then(op)

    def collect(self) -> pd.DataFrame:
        """Run the chain: scan with the leading row-local operations, then the rest."""
        n_local = next((i for i, op in enumerate(self.ops) if not getattr(op, "row_local", False)), len(self.ops))
        df = self._scan(self.ops[:n_local])
        for op in self.ops[n_local:]:
            df = op(df)
        return df.reset_index(drop=True)

    def to_database(self) -> MotionDatabase:
        """The collected rows as a MotionDatabase, with their frame data gathered from the source.

        Numeric columns become metrics; text columns (e.g. bucket) become the code of each value in
        its sorted values, kept in labels. origin is left out: source_index identifies the frame.
        """
        if isinstance(self.frames, pd.DataFrame):
            raise ValueError("A query over a table keeps no frame data, use collect")
        return to_database(self.frames, self.collect())

    def _scan(self, ops: tuple) -> pd.DataFrame:
        if isinstance(self.frames, pd.DataFrame):
            df = self.frames
            for op in ops:
                df = op(df)
            return df
        frames, names = self.frames, list(METRICS)
        stop = frames.num_frames if self.max_frames is None else min(self.max_frames, frames.num_frames)
        chunk = self.chunk
        if len(frames.lead) > 1:
            chunk = max(1, chunk // frames.lead[1]) * frames.lead[1]
        print(f"Scanning {stop} of {frames.num_frames} frames")
        start_t = time.time()
        # one pool for all chunks: spawning a worker re-imports the package
        pool = ProcessPoolExecutor(self.max_workers, mp_context=multiprocessing.get_context("spawn")) if self.max_workers > 1 else None
        try:
            parts = self._scan_chunks(frames, names, ops, stop, chunk, pool)
        finally:
            if pool is not None:
                pool.shutdown()
        df = pd.concat(parts, ignore_index=True)
        print(f"{len(df)} frames (done in {time.time() - start_t:.0f}s)")
        return df

    def _scan_chunks(self, frames, names: list, ops: tuple, stop: int, chunk: int, pool) -> list:
        parts = []
        for start in range(0, stop, chunk):
            end = min(start + chunk, stop)
            qpos = frames.read(f"{frames.prefix}qpos", start, end)[:: self.stride]
            qvel = frames.read(f"{frames.prefix}qvel", start, end)[:: self.stride]
            metrics = compute_metrics(env_model(), qpos, qvel, names, max_workers=self.max_workers, executor=pool)
            df = frame_table(start, self.stride, qpos, metrics, frames.origins(np.arange(start, end)[:: self.stride]))
            for op in ops:  # conditions see the metrics in float64, kept rows store them in float32
                df = op(df)
            parts.append(df.astype({n: np.float32 for n in names}))
            print(f"  frames [{start}, {end}): {len(df)} match")
        return parts


def _literal(value: str | None) -> str:
    return "NULL" if value is None else "'" + str(value).replace("'", "''") + "'"


def to_database(frames, df: pd.DataFrame) -> MotionDatabase:
    """MotionDatabase of the rows of df, with their frame data gathered from frames."""
    idx = df["frame"].to_numpy()
    order = np.argsort(idx, kind="stable")  # gather reads in source order
    kept = frames.gather_frames(idx[order])
    inverse = np.argsort(order)
    kept = {k: ({kk: vv[inverse] for kk, vv in v.items()} if isinstance(v, dict) else v[inverse]) for k, v in kept.items()}
    metrics, labels = {}, {}
    for c in df.columns:
        if c in ("frame", "origin") or c.startswith(QPOS_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            metrics[c] = df[c].to_numpy().astype(np.float32)
        else:
            labels[c] = sorted(df[c].dropna().unique())
            metrics[c] = pd.Categorical(df[c], categories=labels[c]).codes.astype(np.float32)
    return MotionDatabase(metrics=metrics, source_index=idx, frames=kept, labels=labels)


@dataclasses.dataclass
class Source:
    """Where a query script takes its frames from, and how it scans them (the script's CLI arguments).

    Args:
        model_folder: model folder holding the checkpoint (and its buffers).
        buffer: buffer folder (default: checkpoint/buffers/train_reduced, else train).
        lafan: take the frames from the LAFAN motions of lafan_dir whose name matches one of these
            comma-separated globs instead of the buffer, e.g. "walk*,run*". The motions are sampled
            at the env dt by the motion library of an Isaac env, which also gives their backward
            observations for latent inference.
        chunk: frames read from the source per step.
        max_frames: only scan the first max_frames frames.
        stride: only scan every stride-th frame.
        max_workers: processes computing the metrics (default: every core).
    """

    model_folder: Path
    buffer: Path | None = None
    lafan: str | None = None
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof"
    chunk: int = 100_000
    max_frames: int | None = None
    stride: int = 1
    max_workers: int = os.cpu_count() or 1
    device: str = "cuda"

    def open(self):
        model_folder = Path(self.model_folder)
        if self.lafan is not None:
            files = lafan_files(self.lafan_dir, self.lafan)
            print(f"Loading {len(files)} LAFAN motions: {[f.stem for f in files]}")
            return LafanFrames(model_folder, files, device=self.device)
        buffer = self.buffer
        if buffer is None:
            buffer = model_folder / "checkpoint/buffers/train_reduced"
            if not buffer.is_dir():
                buffer = model_folder / "checkpoint/buffers/train"
        print(f"Buffer {buffer}")
        return H5Frames(Path(buffer) / "buffer.hdf5")

    def scan(self) -> Query:
        """Start a query over the frames of this source."""
        return Query(self.open(), chunk=self.chunk, max_frames=self.max_frames, stride=self.stride, max_workers=self.max_workers)
