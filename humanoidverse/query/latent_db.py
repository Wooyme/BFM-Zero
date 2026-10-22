"""Generated latents with the motion the policy makes of them, as a frame source for motion queries.

A row is a latent z given to the policy (e.g. optimized by CEM) and the state its rollout reaches. Unlike LAFAN
motions or a replay buffer, z is not what the backward map infers from that state (goal_inference of next_obs
gives another latent), so both are kept: z is the command, next_obs what the backward map sees of the result.
Rows are independent, not consecutive frames of a motion, so every row keeps its own velocities: qvel holds the
base linear velocity (world frame, qvel[0:3]), the base angular velocity (body frame, qvel[3:6]) and the joint
velocities, as the qvel of the other sources, and next_obs the body velocities.

    LatentDatabase(z=z, qpos=qpos, qvel=qvel, next_obs=obs).save("latent_db.pkl")
    Source(model_folder, latents=Path("latent_db.pkl")).scan().where("base_height >= 0.6").to_database()

As a source (LatentFrames), the origin of row i is "<file stem>:<i>", and to_database keeps z in frames, which
motion_latent.infer_z returns as is. The viewer (humanoidverse.motion_viewer) opens a latent database directly.
"""

import dataclasses
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase

KIND = "latent_db"


@dataclasses.dataclass
class LatentDatabase:
    """Latents and the rollout state of each; every array has one row per latent.

    z: (N, z_dim) latent given to the policy.
    qpos: (N, 7 + nj) state the rollout reached; qvel: (N, 6 + nj) its velocities (see the module doc).
    next_obs: {key: (N, ...)} backward-map observation of that state (state, last_action, privileged_state).
    recipe: how the latents were generated, for provenance.
    """

    z: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    next_obs: dict
    recipe: str | None = None

    def __post_init__(self):
        n = len(self.z)
        arrays = {"qpos": self.qpos, "qvel": self.qvel, **{f"next_obs.{k}": v for k, v in self.next_obs.items()}}
        wrong = {k: len(v) for k, v in arrays.items() if len(v) != n}
        if wrong:
            raise ValueError(f"LatentDatabase arrays must have {n} rows like z, got {wrong}")

    def __len__(self) -> int:
        return len(self.z)

    def select(self, idx: np.ndarray) -> "LatentDatabase":
        idx = np.ascontiguousarray(idx)
        return dataclasses.replace(self, z=self.z[idx], qpos=self.qpos[idx], qvel=self.qvel[idx],
                                   next_obs={k: v[idx] for k, v in self.next_obs.items()})

    def to_motion_database(self) -> MotionDatabase:
        """The rows as a MotionDatabase (rollout states as frames, z kept in frames["z"]), e.g. for the viewer."""
        frames = {"qpos": self.qpos, "qvel": self.qvel, "next_obs": self.next_obs, "z": self.z}
        return MotionDatabase(metrics={}, source_index=np.arange(len(self)), frames=frames, recipe=self.recipe)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": KIND} | {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LatentDatabase":
        state = joblib.load(path)
        if state.pop("kind", None) != KIND:
            raise ValueError(f"{path} is not a LatentDatabase")
        return cls(**state)


def load_database(path: str | Path) -> MotionDatabase:
    """A MotionDatabase pickle, or a LatentDatabase pickle as its MotionDatabase."""
    state = joblib.load(path)
    if state.pop("kind", None) == KIND:
        return LatentDatabase(**state).to_motion_database()
    state.pop("query", None)  # as MotionDatabase.load
    return MotionDatabase(**state)


class LatentFrames:
    """Rows of a LatentDatabase as a flat frame source (same interface as H5Frames / LafanFrames)."""

    prefix = ""

    def __init__(self, path: Path):
        self.db = LatentDatabase.load(path)
        self.name = Path(path).stem
        self.num_frames = len(self.db)
        self.lead = (self.num_frames,)

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        return getattr(self.db, key)[start:stop]

    def gather_frames(self, idx: np.ndarray) -> dict:
        """qpos / qvel / next_obs / z of rows idx."""
        db = self.db
        return {"qpos": db.qpos[idx], "qvel": db.qvel[idx], "z": db.z[idx],
                "next_obs": {k: v[idx] for k, v in db.next_obs.items()}}

    def origins(self, idx: np.ndarray) -> np.ndarray:
        """"<file stem>:<row>" of rows idx."""
        return np.char.add(f"{self.name}:", np.asarray(idx).astype(str))
