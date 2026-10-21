"""Static stability of a pose: the pose_hold score (pose_hold.py) of the BFM policy holding it from rest.

The scores live in a SQLite cache (.cache/pose_hold.sqlite) that the query reads; poses it does not
hold yet are evaluated in batches by a pose_hold Isaac Sim subprocess and added to it.

A pose is keyed by what the evaluation sees of it: pose_hold resets the root to the origin, sets the
height by ground clearance, and the policy and floor ignore heading, so the key is the yaw-free root
quaternion and the joint angles, quantized to QUANTUM. The evaluated pose is the dequantized key, so
a cached score belongs to exactly the poses that share its key. Scores of different evaluators (model
weights, duration, clearance, score config) are kept apart by their config id. Each score keeps the
origin of the pose it was computed for ("<file>:<frame index>") and whether StabilityCache evaluated
it (else it was imported from a ranking.csv).

    uv run python -m humanoidverse.query.stability import [--ranking data/pose_hold/lafan_29dof/ranking.csv]
    uv run python -m humanoidverse.query.stability stats
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from humanoidverse.pose_hold import DEFAULT_MODEL, DEFAULT_OUTPUT, ROOT, ScoreConfig, sha256

CACHE = ROOT / ".cache" / "pose_hold.sqlite"
QUANTUM = 1e-4  # rad, per quaternion component and joint angle
MODEL_FILES = (
    "config.json",
    "config.yaml",
    "checkpoint/model/config.json",
    "checkpoint/model/init_kwargs.json",
    "checkpoint/model/model.safetensors",
)
# Columns of a cached score, as pose_hold_eval writes them to ranking.csv
SCORE_COLUMNS = (
    "score", "base_score", "joint_score", "pose_score", "distance_score", "cost", "valid",
    "base_error_mean_deg", "base_error_max_deg", "joint_rmse_mean", "joint_error_max",
    "end_displacement", "max_displacement", "tail_base_speed", "height_shift",
)  # fmt: skip

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS scores (
    config TEXT NOT NULL,
    key INTEGER NOT NULL,
    {", ".join(f"{c} REAL" for c in SCORE_COLUMNS)},
    origin TEXT,
    evaluated INTEGER,
    PRIMARY KEY (config, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS configs (config TEXT PRIMARY KEY, description TEXT);
CREATE TABLE IF NOT EXISTS imports (path TEXT PRIMARY KEY, mtime_ns INTEGER, config TEXT, rows INTEGER);
CREATE TABLE IF NOT EXISTS file_hashes (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, sha256 TEXT);
"""


def canonical(qpos: np.ndarray) -> np.ndarray:
    """(N, 4 + nj) yaw-free root quaternion (wxyz, w >= 0) and joints of (N, 7 + nj) qpos, quantized to QUANTUM steps."""
    q = np.asarray(qpos, dtype=np.float64)
    quat = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    w, x, y, z = quat.T
    half = -0.5 * np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(half), np.sin(half)  # (c, 0, 0, s) * quat removes the heading
    quat = np.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), axis=1)
    quat *= np.where(quat[:, :1] < 0, -1.0, 1.0)
    return np.round(np.concatenate((quat, q[:, 7:]), axis=1) / QUANTUM).astype(np.int32)


def pose_keys(codes: np.ndarray) -> np.ndarray:
    """(N,) int64 key of each row of canonical codes."""
    codes = np.ascontiguousarray(codes)
    return np.array([int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little", signed=True) for row in codes], dtype=np.int64)


def pose_of(codes: np.ndarray) -> np.ndarray:
    """(N, 7 + nj) qpos evaluated for canonical codes: root at the origin (pose_hold sets the height)."""
    values = codes.astype(np.float64) * QUANTUM
    quat = values[:, :4] / np.linalg.norm(values[:, :4], axis=1, keepdims=True)
    root = np.zeros((len(codes), 3))
    root[:, 2] = 0.8
    return np.concatenate((root, quat, values[:, 4:]), axis=1).astype(np.float32)


def config_id(model_hashes, duration: float, clearance: float, score_config: ScoreConfig) -> tuple[str, str]:
    """Id of an evaluator and its description, independent of where the model folder is."""
    description = json.dumps(
        {"model": sorted(model_hashes), "duration": duration, "clearance": clearance, "score_config": asdict(score_config)},
        sort_keys=True,
    )
    return hashlib.sha256(description.encode()).hexdigest()[:16], description


def connect(path: Path = CACHE) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    if "evaluated" not in {row[1] for row in con.execute("PRAGMA table_info(scores)")}:
        # caches from before the column marked evaluated scores by origin = 'evaluated'
        with con:
            con.execute("ALTER TABLE scores ADD COLUMN evaluated INTEGER")
            con.execute("UPDATE scores SET evaluated = (origin = 'evaluated')")
    return con


def file_hash(con: sqlite3.Connection, path: Path) -> str:
    """sha256 of a file, remembered in the cache by size and mtime (the checkpoint is 3 GB)."""
    st = Path(path).stat()
    row = con.execute("SELECT size, mtime_ns, sha256 FROM file_hashes WHERE path = ?", (str(path),)).fetchone()
    if row is not None and row[:2] == (st.st_size, st.st_mtime_ns):
        return row[2]
    digest = sha256(path)
    with con:
        con.execute("INSERT OR REPLACE INTO file_hashes VALUES (?, ?, ?, ?)", (str(path), st.st_size, st.st_mtime_ns, digest))
    return digest


@dataclasses.dataclass
class StabilityCache:
    """pose_hold scores of poses, read from the cache and evaluated when missing.

    Args:
        model_folder: the BFM model holding the poses (default: pose_hold's).
        path: SQLite cache file.
        duration / clearance / score_config: pose_hold evaluation settings (its defaults).
        num_envs: parallel Isaac Sim environments of an evaluation.
        device: device of the evaluation.
        ranking: pose_hold ranking.csv imported into the cache on first use, if present.
    """

    model_folder: Path = DEFAULT_MODEL
    path: Path = CACHE
    duration: float = 5.0
    clearance: float = 0.01
    score_config: ScoreConfig = ScoreConfig()
    num_envs: int = 2048
    device: str = "cuda:0"
    ranking: Path | None = DEFAULT_OUTPUT / "ranking.csv"

    def __post_init__(self):
        self.model_folder = Path(self.model_folder).resolve()
        with connect(self.path) as con:
            hashes = [file_hash(con, self.model_folder / f) for f in MODEL_FILES if (self.model_folder / f).exists()]
            self.config, description = config_id(hashes, self.duration, self.clearance, self.score_config)
            con.execute("INSERT OR IGNORE INTO configs VALUES (?, ?)", (self.config, description))
        if self.ranking is not None and Path(self.ranking).exists():
            import_ranking(Path(self.ranking), self.path)

    def lookup(self, keys: np.ndarray, columns=SCORE_COLUMNS) -> pd.DataFrame:
        """Cached columns of the unique keys found, indexed by key."""
        con = connect(self.path)
        try:
            con.execute("CREATE TEMP TABLE wanted (key INTEGER PRIMARY KEY)")
            con.executemany("INSERT OR IGNORE INTO wanted VALUES (?)", ((int(k),) for k in keys))
            found = pd.read_sql_query(
                f"SELECT s.key, {', '.join('s.' + c for c in columns)} FROM scores s JOIN wanted USING (key) WHERE s.config = ?",
                con,
                params=(self.config,),
            )
        finally:
            con.close()
        return found.set_index("key")

    def scores(self, qpos: np.ndarray, origins: np.ndarray, columns=SCORE_COLUMNS) -> pd.DataFrame:
        """(N,) rows of columns for poses qpos, evaluating the ones missing from the cache first.

        origins: (N,) "<file>:<frame index>" of each pose, kept with the scores it evaluates.
        """
        codes = canonical(qpos)
        keys = pose_keys(codes)
        found = self.lookup(keys, columns)
        missing = ~np.isin(keys, found.index.to_numpy())
        _, first = np.unique(keys[missing], return_index=True)
        todo = np.flatnonzero(missing)[first]
        print(f"Stability: {len(qpos)} poses, {len(np.unique(keys))} distinct, {len(todo)} not cached")
        if len(todo):
            self.evaluate(codes[todo], keys[todo], np.asarray(origins).astype(str)[todo])
            found = self.lookup(keys, columns)
        if not np.isin(keys, found.index.to_numpy()).all():
            raise RuntimeError("Evaluation left poses without a cached score")
        return found.loc[keys].reset_index(drop=True)

    def evaluate(self, codes: np.ndarray, keys: np.ndarray, origins: np.ndarray):
        """Score the poses of codes with pose_hold in an Isaac Sim subprocess (a query may already run
        an Isaac app, and there is one per process), which writes each finished batch to the cache."""
        with tempfile.TemporaryDirectory() as tmp:
            poses = Path(tmp) / "poses.npz"
            np.savez(poses, codes=codes, keys=keys, origins=origins)
            command = [
                sys.executable, "-m", "humanoidverse.query.stability", "evaluate", str(poses),
                "--cache", str(self.path), "--config", self.config, "--model-folder", str(self.model_folder),
                "--duration", str(self.duration), "--clearance", str(self.clearance),
                "--num-envs", str(min(self.num_envs, len(codes))), "--device", self.device,
            ]  # fmt: skip
            for f in dataclasses.fields(ScoreConfig):
                command += ["--" + f.name.replace("_", "-"), repr(getattr(self.score_config, f.name))]
            subprocess.run(command, check=True, cwd=ROOT)


def import_ranking(ranking: Path, path: Path = CACHE, force: bool = False) -> int:
    """Add the scores of a pose_hold run (its ranking.csv, motion_db.pkl and manifest.json) to the cache.

    Returns the number of rows read; 0 when this ranking.csv was already imported unchanged.
    """
    from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase

    ranking = Path(ranking).resolve()
    folder = ranking.parent
    mtime = ranking.stat().st_mtime_ns
    with connect(path) as con:
        row = con.execute("SELECT mtime_ns FROM imports WHERE path = ?", (str(ranking),)).fetchone()
    if row is not None and row[0] == mtime and not force:
        return 0
    identity = json.loads((folder / "manifest.json").read_text())["identity"]
    config, description = config_id(
        identity["model_files"].values(), identity["duration"], identity["clearance"], ScoreConfig(**identity["score_config"])
    )
    print(f"Importing {ranking} as config {config}")
    df = pd.read_csv(ranking, usecols=["source_index", "motion_name", "frame_index", *SCORE_COLUMNS])
    db = MotionDatabase.load(folder / "motion_db.pkl")
    qpos = pd.DataFrame({"source_index": db.source_index, "i": np.arange(len(db))})
    df = df.merge(qpos, on="source_index", how="inner")
    df["valid"] = df["valid"].astype(str).str.lower().isin(("true", "1", "1.0")).astype(float)
    df["key"] = pose_keys(canonical(np.asarray(db.frames["qpos_original"])[df["i"].to_numpy()]))
    df["origin"] = df["motion_name"] + ":" + df["frame_index"].astype(str)
    rows = [(config, int(r.key), *(float(getattr(r, c)) for c in SCORE_COLUMNS), r.origin, 0) for r in df.itertuples()]
    with connect(path) as con:
        con.execute("INSERT OR IGNORE INTO configs VALUES (?, ?)", (config, description))
        # an evaluated score of the dequantized pose takes precedence over an imported nearby one
        con.executemany(f"INSERT OR IGNORE INTO scores VALUES ({', '.join('?' * (len(SCORE_COLUMNS) + 4))})", rows)
        con.execute("INSERT OR REPLACE INTO imports VALUES (?, ?, ?, ?)", (str(ranking), mtime, config, len(rows)))
    print(f"  {len(rows)} rows, {df['key'].nunique()} distinct poses")
    return len(rows)


def run_evaluation(args):
    """Subprocess entry: pose_hold the poses of args.poses in batches, writing each batch to the cache."""
    import torch

    from humanoidverse.pose_hold import GoalPolicy, Observations, load_training_config
    from humanoidverse.pose_hold_eval import evaluate_batch, make_backend

    torch.set_num_threads(1)
    torch.manual_seed(0)
    with np.load(args.poses) as data:
        codes, keys, origins = data["codes"], data["keys"], data["origins"]
    qpos = pose_of(codes)
    score_config = ScoreConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(ScoreConfig)})
    config = load_training_config(args.model_folder)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    backend = make_backend(config, args.num_envs, args.clearance, args.device)
    start = time.perf_counter()
    try:
        policy = GoalPolicy(args.model_folder, args.device)
        observations = Observations(config, args.device)
        with torch.inference_mode():
            for offset in range(0, len(qpos), args.num_envs):
                ix = slice(offset, offset + args.num_envs)
                result = evaluate_batch(backend, policy, observations, qpos[ix], args.duration, score_config)
                rows = [
                    (args.config, int(k), *(float(result[c][i]) for c in SCORE_COLUMNS), str(o), 1)
                    for i, (k, o) in enumerate(zip(keys[ix], origins[ix]))
                ]
                with connect(args.cache) as con:
                    con.executemany(f"INSERT OR REPLACE INTO scores VALUES ({', '.join('?' * (len(SCORE_COLUMNS) + 4))})", rows)
                done = min(offset + args.num_envs, len(qpos))
                print(f"  stability {done}/{len(qpos)} poses, {done / (time.perf_counter() - start):.1f} poses/s", flush=True)
    finally:
        backend.close()


def stats(path: Path):
    with connect(path) as con:
        for config, description in con.execute("SELECT config, description FROM configs"):
            n, mean = con.execute("SELECT count(*), avg(score) FROM scores WHERE config = ?", (config,)).fetchone()
            counts = dict(con.execute("SELECT evaluated, count(*) FROM scores WHERE config = ? GROUP BY 1", (config,)).fetchall())
            print(f"{config}: {n} poses ({counts.get(1, 0)} evaluated, {counts.get(0, 0)} imported), mean score {mean}")
            print(f"  {description}")
        for path, _, config, rows in con.execute("SELECT * FROM imports"):
            print(f"imported {path} -> {config} ({rows} rows)")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="add a pose_hold ranking.csv to the cache")
    imp.add_argument("--cache", type=Path, default=CACHE)
    imp.add_argument("--ranking", type=Path, default=DEFAULT_OUTPUT / "ranking.csv")
    imp.add_argument("--force", action="store_true", help="import again even if unchanged")
    sub.add_parser("stats", help="poses cached per evaluator").add_argument("--cache", type=Path, default=CACHE)
    ev = sub.add_parser("evaluate", help="internal: score poses in batches (run by StabilityCache)")
    ev.add_argument("poses", type=Path)
    ev.add_argument("--cache", type=Path, default=CACHE)
    ev.add_argument("--config", required=True)
    ev.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL)
    ev.add_argument("--duration", type=float, default=5.0)
    ev.add_argument("--clearance", type=float, default=0.01)
    ev.add_argument("--num-envs", type=int, default=2048)
    ev.add_argument("--device", default="cuda:0")
    for f in dataclasses.fields(ScoreConfig):
        ev.add_argument("--" + f.name.replace("_", "-"), type=float, default=f.default)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "import":
        import_ranking(args.ranking, args.cache, args.force)
    elif args.command == "stats":
        stats(args.cache)
    else:
        if not math.isfinite(args.duration) or args.duration <= 0 or args.num_envs < 1:
            raise ValueError("Require a positive duration and num-envs")
        run_evaluation(args)
