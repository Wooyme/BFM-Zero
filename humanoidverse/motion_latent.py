import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase


def infer_z(model, frames: dict, per_frame: bool, device="cuda") -> np.ndarray:
    """z of MotionDatabase frames: one (1, z_dim) for all of them, or (N, z_dim) with per_frame.

    Uses the kept backward embeddings B when present, otherwise next_obs (reward inference with
    reward 1 on every frame, or goal inference per frame).
    """
    with torch.no_grad():
        if "B" in frames:
            B = torch.as_tensor(frames["B"], dtype=torch.float32, device=device)
            z = model.project_z(B if per_frame else B.sum(dim=0, keepdim=True))
        elif "next_obs" in frames:
            next_obs = tree_map(lambda x: torch.as_tensor(x, dtype=torch.float32), frames["next_obs"])
            n = len(next(iter(tree_flatten(next_obs)[0])))
            if per_frame:
                batch_size = model.cfg.inference_batch_size
                z = torch.cat(
                    [
                        model.goal_inference(tree_map(lambda x: x[i : i + batch_size].to(device), next_obs))
                        for i in range(0, n, batch_size)
                    ]
                )
            else:
                z = model.reward_inference(next_obs, torch.ones(n, 1)).reshape(1, -1)
        else:
            raise ValueError("Frames keep no next_obs or B, rerun motion_query with keep_frames")
    return z.float().cpu().numpy()


def load_model(model_folder: Path, device: str = "cuda"):
    model = load_model_from_checkpoint_dir(Path(model_folder) / "checkpoint", device=device)
    model.to(device)
    model.eval()
    return model


def infer_z_by(model_folder: Path, db: MotionDatabase, by: str, per_frame: bool = True, device: str = "cuda") -> dict:
    """{label: z} for each value of the text column by of db (see MotionDatabase.labels), labels in sorted order."""
    model = load_model(model_folder, device)
    z = {}
    for code, label in enumerate(db.labels[by]):
        sub = db.select(np.flatnonzero(db.metrics[by] == code))
        if len(sub):
            z[label] = infer_z(model, sub.frames, per_frame=per_frame, device=device)
    return z


def main(
    model_folder: Path,
    db: Path | None = None,
    where: str | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    per_frame: bool = False,
    name: str | None = None,
    output: Path | None = None,
    device="cuda",
):
    """Turn the frames of a MotionDatabase (from motion_query) into latent z.

    By default all selected frames give one z, the backward embedding averaged over them (reward
    inference with reward 1 on the selected frames); with per_frame, each frame gives its own z
    (goal inference).

    Args:
        db: MotionDatabase pickle (default: model_folder/motion_query/motion_db.pkl).
        where: SQL condition (DuckDB) over the metric columns and source_index, e.g.
            "base_speed > 0.5 and abs(move_angle) < 30 and not (torso_pitch > 20)".
        sort_by: metric to sort by (descending unless ascending), applied after where.
        limit: keep the first limit frames after where / sort_by.
        sample: keep this many frames drawn uniformly without replacement after where / sort_by / limit
            (with seed), in their original order.
        per_frame: one z per frame instead of one z for the whole selection.
        name: key of the z in the output file (default: the where expression, or "all").
        output: joblib file of {name: z}, z of shape (1, z_dim) or (N, z_dim); an existing file
            is updated, so several selections can be collected in one file
            (default: model_folder/motion_latent/latent_z.pkl).
    """
    model_folder = Path(model_folder)
    db_path = Path(db) if db is not None else model_folder / "motion_query" / "motion_db.pkl"
    output = Path(output) if output is not None else model_folder / "motion_latent" / "latent_z.pkl"
    name = name or where or "all"

    motion_db = MotionDatabase.load(db_path)
    print(f"Loaded {len(motion_db)} frames from {db_path}")
    if where is not None:
        motion_db = motion_db.where(where)
    if sort_by is not None:
        motion_db = motion_db.sort(sort_by, descending=not ascending)
    if limit is not None:
        motion_db = motion_db.select(np.arange(min(limit, len(motion_db))))
    if sample is not None and sample < len(motion_db):
        motion_db = motion_db.select(np.sort(np.random.default_rng(seed).choice(len(motion_db), sample, replace=False)))
    if len(motion_db) == 0:
        raise ValueError(f"No frames selected by where={where!r}")
    print(f"Selected {len(motion_db)} frames")

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()

    z = infer_z(model, motion_db.frames, per_frame, device)

    output.parent.mkdir(parents=True, exist_ok=True)
    z_dict = joblib.load(output) if output.exists() else {}
    z_dict[name] = z
    joblib.dump(z_dict, output)
    print(f"Saved z {z.shape} as {name!r} in {output}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
