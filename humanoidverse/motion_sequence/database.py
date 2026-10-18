"""Sequence indices and source-frame access; no model or simulator is needed."""

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "humanoidverse/data/lafan_29dof"
DEFAULT_RANKING = ROOT / "data/pose_hold/lafan_29dof/ranking.csv"
DEFAULT_OUTPUT = ROOT / "data/motion_sequence/lafan_29dof"
SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_spans(scores: np.ndarray, threshold: float = 83.0) -> list[tuple[str, int, int]]:
    """Return (kind, inclusive start, inclusive end), allowing shared stable endpoints.

    Stable runs are maximal, including singletons. Each gap between stable runs
    gives one transition from the preceding run's last frame to the next's first.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not math.isfinite(threshold) or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Require a finite threshold and a finite one-dimensional score array")
    changes = np.diff(np.r_[False, scores > threshold, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
    spans = [("stable", int(start), int(end)) for start, end in zip(starts, ends)]
    spans += [("transition", int(start), int(end)) for start, end in zip(ends[:-1], starts[1:])]
    return sorted(spans, key=lambda span: (span[1], span[0] != "stable"))


@dataclass(frozen=True)
class Sequence:
    """One segment in a source motion; frame numbers are zero-based and inclusive."""

    sequence_id: int
    kind: str
    source_file: str  # Relative to SequenceDatabase.data_path.
    motion_name: str
    start_frame: int
    end_frame: int
    fps: float
    start_score: float
    end_score: float
    min_score: float
    mean_score: float

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration(self) -> float:
        """Elapsed time between the first and last frame (zero for a singleton)."""
        return (self.num_frames - 1) / self.fps

    def to_record(self) -> dict:
        return asdict(self) | {"num_frames": self.num_frames, "duration": self.duration}


@dataclass
class SequenceDatabase:
    """Small, trusted joblib index with score snapshots and references to original motions.

    Only the most recently accessed source file and converted clip are cached.
    No latent, rollout, or copied motion arrays are persisted.
    """

    data_path: str
    ranking_path: str
    ranking_sha256: str
    threshold: float
    source_sha256: dict[str, str]
    scores: dict[tuple[str, str], np.ndarray]
    sequences: list[Sequence]
    _cached_file: str | None = field(default=None, init=False, repr=False)
    _cached_motions: dict = field(default_factory=dict, init=False, repr=False)
    _cached_key: tuple[str, str] | None = field(default=None, init=False, repr=False)
    _cached_qpos: np.ndarray | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.sequences)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "data_path": self.data_path,
                "ranking_path": self.ranking_path,
                "ranking_sha256": self.ranking_sha256,
                "threshold": self.threshold,
                "source_sha256": self.source_sha256,
                "scores": self.scores,
                "sequences": [asdict(sequence) for sequence in self.sequences],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SequenceDatabase":
        """Load a trusted index written by save(); never unpickle untrusted input."""
        state = joblib.load(path)
        if state.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported sequence database schema")
        state["sequences"] = [Sequence(**row) for row in state["sequences"]]
        return cls(**state)

    def get_frames(self, sequence_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return original WXYZ qpos and scores in temporal order, including both endpoints."""
        from scripts.render_bfm_motion import load_motions, motion_qpos

        if not 0 <= sequence_id < len(self):
            raise IndexError(f"Sequence index must be in [0, {len(self) - 1}]")
        sequence = self.sequences[sequence_id]
        source_file = sequence.source_file
        key = (source_file, sequence.motion_name)
        if self._cached_file != source_file:
            path = Path(self.data_path) / source_file
            if sha256(path) != self.source_sha256[source_file]:
                raise ValueError(f"Source changed since extraction: {path}")
            motions = load_motions(path)
            self._cached_file, self._cached_motions = source_file, motions
            self._cached_key, self._cached_qpos = None, None
        if self._cached_key != key:
            qpos, fps = motion_qpos(self._cached_motions[sequence.motion_name])
            if fps != sequence.fps or len(qpos) != len(self.scores[key]):
                raise ValueError(f"Source frame count/FPS does not match the index: {key}")
            self._cached_key, self._cached_qpos = key, qpos
        span = slice(sequence.start_frame, sequence.end_frame + 1)
        return self._cached_qpos[span], self.scores[key][span]

    def summary(self) -> dict:
        groups = {}
        for kind in ("stable", "transition"):
            sequences = [sequence for sequence in self.sequences if sequence.kind == kind]
            lengths = np.array([sequence.num_frames for sequence in sequences], dtype=np.int64)
            groups[kind] = {
                "count": len(lengths),
                "singletons": int((lengths == 1).sum()),
                "min_frames": int(lengths.min()) if len(lengths) else None,
                "median_frames": float(np.median(lengths)) if len(lengths) else None,
                "p95_frames": float(np.quantile(lengths, 0.95)) if len(lengths) else None,
                "max_frames": int(lengths.max()) if len(lengths) else None,
                "max_duration": max((sequence.duration for sequence in sequences), default=None),
            }
        leading, trailing = 0, 0
        for scores in self.scores.values():
            stable = np.flatnonzero(scores > self.threshold)
            # A wholly unstable clip is counted once, as an unclosed leading interval.
            leading += int(stable[0]) if len(stable) else len(scores)
            trailing += len(scores) - 1 - int(stable[-1]) if len(stable) else 0
        return {
            "schema_version": SCHEMA_VERSION,
            "data_path": self.data_path,
            "ranking_path": self.ranking_path,
            "ranking_sha256": self.ranking_sha256,
            "source_sha256": self.source_sha256,
            "threshold": self.threshold,
            "stable_condition": "score > threshold",
            "source_files": len(self.source_sha256),
            "motions": len(self.scores),
            "frames": sum(len(scores) for scores in self.scores.values()),
            "stable_frames": sum(int((scores > self.threshold).sum()) for scores in self.scores.values()),
            "sequences": len(self),
            "groups": groups,
            "unclosed_leading_frames": leading,
            "unclosed_trailing_frames": trailing,
        }
