"""Export stable pose_hold frames as one holosoma motion of static goal poses with their BFM-Zero goal latents.

Frames are those of humanoidverse/pose_hold_eval.py output (parts/*.npz) that are valid with score > min_score:
the frozen actor, given z = goal_inference of the stationary frame, held the pose. Each frame becomes one row of a
holosoma MotionLoader npz (fps, joint/body names, joint_pos [xyz, wxyz, joints], zero joint_vel, body_pos_w,
body_quat_w (wxyz), zero body velocities) from the motion library's forward kinematics of initial_qpos, and z_track
holds the stored goal latent of that frame. Rows are unrelated poses: a consumer must hold one row, never play them.

Needs Isaac (for the motion library's skeleton); run from the repository root:
    python -m scripts.export_holosoma_pose_hold --model-folder new_model_for_training_code_inference \
        --motion-path data/lafan_29dof_10s-clipped.pkl \
        --output ../holosoma/src/holosoma/holosoma/data/motions/g1_29dof/bfm_pose_hold/pose_hold_score83.npz
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import glob
from pathlib import Path

import numpy as np
import torch

from experiments.hand_compose import Kinematics
from experiments.tracking_window import build_env

POSE_HOLD = "data/pose_hold/lafan_29dof/parts"
# MuJoCo qpos joint order of pose_hold initial_qpos (summary.json configuration.robot.dof_names).
POSE_DOFS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def load_frames(parts: str, min_score: float) -> dict[str, np.ndarray]:
    """initial_qpos, latent, source_index and score of the valid frames above min_score, in source order."""
    keys = ("initial_qpos", "latent", "source_index", "score")
    out = {k: [] for k in keys}
    for path in sorted(glob.glob(f"{parts}/*.npz")):
        with np.load(path) as data:
            keep = data["valid"] & (data["score"] > min_score)
            for k in keys:
                out[k].append(data[k][keep])
    out = {k: np.concatenate(v) for k, v in out.items()}
    order = np.argsort(out["source_index"])
    return {k: v[order] for k, v in out.items()}


def main(model_folder: Path, motion_path: Path, output: Path, min_score: float = 83.0, chunk: int = 4096):
    """FK every selected frame and write the static-pose motion file."""
    frames = load_frames(POSE_HOLD, min_score)
    wrapped_env, _ = build_env(model_folder, str(Path(motion_path).resolve()), 1)
    env = wrapped_env._env
    joint_names = list(env.simulator.dof_names)
    body_names = list(env.simulator.body_names)
    if joint_names != POSE_DOFS:
        raise RuntimeError("Motion library joint order differs from the pose_hold qpos order")
    kin = Kinematics(env._motion_lib.mesh_parsers, "cpu")

    qpos = torch.as_tensor(frames["initial_qpos"], dtype=torch.float32)
    root_pos, root_rot_xyzw, dof = qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]
    pos, rot = [], []
    for i in range(0, len(qpos), chunk):
        s = slice(i, i + chunk)
        out = kin.cpu.fk_batch(kin._pose(root_rot_xyzw[s], dof[s])[None], root_pos[s][None])
        pos.append(out.global_translation_extend[0, :, : len(body_names)])
        rot.append(out.global_rotation_extend[0, :, : len(body_names)])
    body_pos, body_rot_xyzw = torch.cat(pos), torch.cat(rot)
    # The root body is the free joint itself.
    error = (body_pos[:, 0] - root_pos).abs().max().item()
    if error > 1e-5:
        raise RuntimeError(f"FK root differs from qpos root by {error} m")

    n = len(qpos)
    arrays = {
        "fps": np.array([round(1.0 / env.dt)], dtype=np.int64),
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
        "joint_pos": qpos.numpy(),
        "joint_vel": np.zeros((n, 6 + len(joint_names)), dtype=np.float32),
        "body_pos_w": body_pos.numpy(),
        "body_quat_w": body_rot_xyzw[..., [3, 0, 1, 2]].numpy(),
        "body_lin_vel_w": np.zeros_like(body_pos.numpy()),
        "body_ang_vel_w": np.zeros_like(body_pos.numpy()),
        "z_track": frames["latent"].astype(np.float32),
        "source_index": frames["source_index"],
        "pose_hold_score": frames["score"].astype(np.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    print(f"{n} frames with score > {min_score} written to {output}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
