"""Goal latent z of every frame of the LAFAN motions, the dataset of the residual VQ tokenizer.

Each frame gives z = project_z(B(next_obs)), next_obs the backward observation of the frame with its
velocities kept (velocity_multiplier 1, unlike goal_inference, which zeroes them). Needs Isaac.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np

from humanoidverse.motion_latent import infer_z, load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, LafanFrames, lafan_files


def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    output: Path | None = None,
    device: str = "cuda",
):
    """Save z (N, z_dim) with the motion name, time (s), qpos and qvel of each frame.

    Args:
        model_folder: model folder holding config.json and checkpoint/.
        motions: comma-separated globs of the motion files of lafan_dir to take.
        output: npz file (default: model_folder/latent_vq/goal_z.npz).
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else model_folder / "latent_vq" / "goal_z.npz"
    frames = LafanFrames(model_folder, lafan_files(lafan_dir, motions), device=device)
    model = load_model(model_folder, device)
    z = infer_z(model, {"next_obs": frames.next_obs}, per_frame=True, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, z=z, **frames.data)
    norm = np.linalg.norm(z, axis=-1)
    print(f"Saved z {z.shape} of {len(np.unique(frames.data['motion']))} motions to {output} (|z| {norm.min():.2f}..{norm.max():.2f})")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
