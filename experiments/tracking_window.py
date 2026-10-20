"""Experiment: how the tracking window T changes the latent z and what the policy does with it.

For every T of windows, z_T(f) = project_z(mean B(next_obs[f : f + T])) (truncated at the motion end,
velocities kept), FBModel.tracking_inference with seq_length T; T = 1 is per-frame goal z. Each rollout
starts at frame t of a LAFAN motion (qpos and qvel of the reference) and plays the latents as tracking
does: the action after the state at frame t + k uses z_T(t + 1 + k). All T of a start run side by side
from the same state, so their z gaps can be compared with the gaps of their actions and trajectories.
Needs Isaac; run from the repository root: python -m experiments.tracking_window. Results and write-up:
experiments/docs/tracking_window/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from humanoidverse.motion_latent import load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, lafan_files

# Per-step rollout metrics, (starts, windows, horizon) each; *_base compare with the rollout of windows[0].
STEP_METRICS = ("z_cos_base", "z_l2_base", "z_cos_last", "action_base", "dof_base", "root_xy_base", "body_base",
                "body_ref", "body_ref_local", "dof_ref")
DOCS = Path(__file__).resolve().parent / "docs" / "tracking_window"
LOST = 0.5  # m of mean global body-position error, the evaluation threshold of terminate_when_motion_far


def window_z(B: torch.Tensor, frame: torch.Tensor, end: torch.Tensor, window: int, project) -> torch.Tensor:
    """project(mean(B[f : min(f + window, e)])) for each frame f and exclusive motion end e (rows of B)."""
    offsets = torch.arange(window, device=B.device)
    rows = frame[:, None] + offsets
    mask = (rows < end[:, None]).to(B.dtype)
    rows = torch.minimum(rows, end[:, None] - 1)
    return project((B[rows] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))


def rollout_starts(lengths: list[int], horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """(motion, first frame) of the rollouts: every stride frames while frame t + horizon is in the motion."""
    motion, frame = [], []
    for m, n in enumerate(lengths):
        t = np.arange(0, n - horizon, stride)
        motion.append(np.full(len(t), m))
        frame.append(t)
    return np.concatenate(motion), np.concatenate(frame)


def category(name: str) -> str:
    """Motion category from a LAFAN name, e.g. dance1_subject1 -> dance."""
    return re.match(r"[A-Za-z]+", name).group(0)


class Reference:
    """Backward observations and reference states of every frame of the motions, flat over motions."""

    def __init__(self, env, model, use_root_height_obs: bool):
        from humanoidverse.utils.helpers import get_backward_observation

        lib = env._motion_lib
        env.set_is_evaluating(0)
        if not lib.all_motions_loaded:
            raise ValueError(f"{lib._num_unique_motions} motions do not fit in {env.num_envs} envs, raise --num-envs")
        names, B, parts = [], [], {k: [] for k in ("body_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")}
        num_bodies = env.simulator._rigid_body_pos.shape[1]
        for i in range(lib._num_unique_motions):
            names.append(Path(str(lib.curr_motion_keys[i])).stem)
            obs, d = get_backward_observation(env, i, use_root_height_obs=use_root_height_obs, velocity_multiplier=1.0)
            with torch.no_grad():
                B.append(model.backward_map(obs).float())
            parts["body_pos"].append(d["ref_body_pos"][:, :num_bodies])
            parts["root_rot"].append(d["ref_body_rots"][:, 0])  # xyzw
            parts["root_vel"].append(d["ref_body_vels"][:, 0])
            parts["root_ang_vel"].append(d["ref_body_angular_vels"][:, 0])
            parts["dof_pos"].append(d["dof_pos"])
            parts["dof_vel"].append(d["ref_dof_vel"])
            print(f"  {names[-1]}: {len(B[-1])} frames")
        self.names = names
        self.lengths = [len(b) for b in B]
        self.offset = torch.as_tensor(np.r_[0, np.cumsum(self.lengths)[:-1]], device=env.device)
        self.B = torch.cat(B)
        for k, v in parts.items():
            setattr(self, k, torch.cat(v))

    def root_states(self, rows: torch.Tensor, wxyz: bool) -> torch.Tensor:
        rot = self.root_rot[rows][:, [3, 0, 1, 2]] if wxyz else self.root_rot[rows]
        return torch.cat([self.body_pos[rows, 0], rot, self.root_vel[rows], self.root_ang_vel[rows]], dim=-1)


def build_env(model_folder: Path, motion_dir: str, num_envs: int):
    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    with open(model_folder / "config.json") as f:
        env_config = json.load(f)["env"]
    env_config["hydra_overrides"] += [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        "simulator=isaacsim",
        "env.config.lie_down_init=False",
    ]
    env_config["disable_domain_randomization"] = True
    env_config["disable_obs_noise"] = True
    env_config["lafan_tail_path"] = motion_dir
    wrapped_env, _ = HumanoidVerseIsaacConfig(**env_config).build(num_envs)
    return wrapped_env, env_config.get("root_height_obs", False)


@torch.no_grad()
def run_batch(wrapped_env, model, ref: Reference, windows: list[int], rows: torch.Tensor, ends: torch.Tensor, horizon: int) -> tuple[dict, np.ndarray]:
    """Roll out every window from each start row (absolute frame index); metrics of shape (len(rows), W, horizon) and the mean (W, W) z cosine."""
    env = wrapped_env._env
    num_envs, W, n = env.num_envs, len(windows), len(rows)
    # env e = start (e // W) with window windows[e % W]; padding envs copy the first start
    env_rows = torch.cat([rows.repeat_interleave(W), rows[:1].repeat(num_envs - n * W)])
    env_ends = torch.cat([ends.repeat_interleave(W), ends[:1].repeat(num_envs - n * W)])
    env_window = torch.as_tensor(windows, device=env.device).repeat(n)
    env_window = torch.cat([env_window, env_window[:1].repeat(num_envs - n * W)])

    dof_states = torch.stack([ref.dof_pos[env_rows], ref.dof_vel[env_rows]], dim=-1)
    target_states = {"dof_states": dof_states, "root_states": ref.root_states(env_rows, wxyz=env.config.simulator.config.name == "isaacsim")}
    env_ids = torch.arange(num_envs, dtype=torch.long, device=env.device)
    env.reset_envs_idx(env_ids, target_states=target_states)
    zero = torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32, device=env.device)
    wrapped_env.step(zero, to_numpy=False)
    observation = wrapped_env._get_g1env_observation(to_numpy=False)
    if not torch.allclose(env.simulator.dof_pos, ref.dof_pos[env_rows], atol=1e-4):
        raise RuntimeError("Reset did not reach the reference joint positions")

    def latents(k: int) -> torch.Tensor:
        frame = torch.minimum(env_rows + 1 + k, env_ends - 1)
        z = torch.empty((num_envs, ref.B.shape[1]), device=env.device)
        for w in set(windows):
            sel = env_window == w
            z[sel] = window_z(ref.B, frame[sel], env_ends[sel], w, model.project_z)
        return z

    def per_start(x: torch.Tensor) -> torch.Tensor:
        return x[: n * W].reshape(n, W, *x.shape[1:])

    out = {k: torch.empty((n, W, horizon), device=env.device) for k in STEP_METRICS}
    cos_sum = torch.zeros((W, W), device=env.device)
    origins = env.env_origins
    for k in range(horizon):
        z = latents(k)
        zs = per_start(z)
        out["z_cos_base"][:, :, k] = torch.nn.functional.cosine_similarity(zs, zs[:, :1], dim=-1)
        out["z_l2_base"][:, :, k] = (zs - zs[:, :1]).norm(dim=-1)
        out["z_cos_last"][:, :, k] = torch.nn.functional.cosine_similarity(zs, zs[:, -1:], dim=-1)
        cos_sum += torch.nn.functional.cosine_similarity(zs[:, :, None], zs[:, None], dim=-1).sum(0)
        action = model.act(observation, z, mean=True)
        observation, _, terminated, truncated, _ = wrapped_env.step(action, to_numpy=False)
        if (terminated | truncated).any():
            raise RuntimeError("An env was reset during the rollout; terminations must be off")
        target = torch.minimum(env_rows + 1 + k, env_ends - 1)
        body = per_start(env.simulator._rigid_body_pos - origins[:, None])
        body_ref = per_start(ref.body_pos[target])
        dof, dof_ref = per_start(env.simulator.dof_pos), per_start(ref.dof_pos[target])
        acts = per_start(action)
        out["action_base"][:, :, k] = (acts - acts[:, :1]).norm(dim=-1)
        out["dof_base"][:, :, k] = (dof - dof[:, :1]).square().mean(-1).sqrt()
        out["root_xy_base"][:, :, k] = (body[:, :, 0, :2] - body[:, :1, 0, :2]).norm(dim=-1)
        out["body_base"][:, :, k] = (body - body[:, :1]).norm(dim=-1).mean(-1)
        out["body_ref"][:, :, k] = (body - body_ref).norm(dim=-1).mean(-1)
        local, local_ref = body - body[:, :, :1], body_ref - body_ref[:, :, :1]
        out["body_ref_local"][:, :, k] = (local - local_ref).norm(dim=-1).mean(-1)
        out["dof_ref"][:, :, k] = (dof - dof_ref).square().mean(-1).sqrt()
    out = {k: v.cpu().numpy() for k, v in out.items()}
    # all windows of a start act first from the same state: the network's own output gap
    out["open_loop_action"] = out["action_base"][:, :, 0]
    return out, (cos_sum / (n * horizon)).cpu().numpy()


def summarize(result: dict, windows: list[int], names: list[str], motion: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """metrics.json, the per-window table and the per-motion means."""
    horizon = result["body_ref"].shape[-1]
    lost = (result["body_ref"] > LOST).any(-1)
    marks = sorted({k for k in (10, 25, 50, horizon) if k <= horizon})
    rows = []
    for j, w in enumerate(windows):
        row = {
            "window": w,
            "z_cos_base": result["z_cos_base"][:, j].mean(),
            "z_cos_last": result["z_cos_last"][:, j].mean(),
            "open_loop_action": result["open_loop_action"][:, j].mean(),
            "action_base": result["action_base"][:, j].mean(),
        }
        for k in marks:
            row[f"body_base@{k}"] = result["body_base"][:, j, k - 1].mean()
        row["body_ref"] = result["body_ref"][:, j].mean()
        row["body_ref_local"] = result["body_ref_local"][:, j].mean()
        row["dof_ref"] = result["dof_ref"][:, j].mean()
        row["lost"] = lost[:, j].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    # Does a bigger z gap give a bigger behavior gap? Spearman over (start, window), base window excluded
    gap = pd.DataFrame({
        "z_gap": (1 - result["z_cos_base"][:, 1:]).mean(-1).ravel(),
        "body_gap": result["body_base"][:, 1:].mean(-1).ravel(),
        "category": np.repeat([category(names[m]) for m in motion], len(windows) - 1),
    })
    spearman = {"all": gap.z_gap.corr(gap.body_gap, method="spearman")}
    spearman |= {c: g.z_gap.corr(g.body_gap, method="spearman") for c, g in gap.groupby("category")}
    per_motion = []
    for m in np.unique(motion):
        sel = motion == m
        for j, w in enumerate(windows):
            per_motion.append({
                "motion": names[m],
                "category": category(names[m]),
                "window": w,
                "starts": int(sel.sum()),
                **{k: result[k][sel, j].mean() for k in STEP_METRICS},
                "open_loop_action": result["open_loop_action"][sel, j].mean(),
                "lost": lost[sel, j].mean(),
            })
    metrics = {"windows": windows, "horizon": horizon, "starts": len(motion), "table": table.to_dict("records"),
               "spearman_z_gap_body_gap": spearman}
    return metrics, table, pd.DataFrame(per_motion)


def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    windows: list[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    horizon: int = 50,
    start_stride: int = 10,
    num_envs: int = 1024,
    output: Path | None = None,
    device: str = "cuda",
):
    """Roll out the tracking latents of each window from the same reference states and compare them.

    Args:
        model_folder: model folder holding config.json and checkpoint/.
        motions: comma-separated globs of the motion files of lafan_dir to take.
        windows: tracking windows T; the *_base metrics compare with the first one.
        horizon: controller steps per rollout (50 = 1 s).
        start_stride: frames between the first frames of the rollouts of a motion.
        num_envs: Isaac envs; each batch rolls num_envs // len(windows) starts.
        output: output folder (default: experiments/docs/tracking_window), holding rollouts.npz (gitignored),
            metrics.json and per_motion.csv.
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    per_batch = num_envs // len(windows)
    if per_batch == 0:
        raise ValueError("num_envs must be at least the number of windows")
    model = load_model(model_folder, device)
    # the motion library loads files lazily by index, so they must outlive the run
    with tempfile.TemporaryDirectory() as motion_dir:
        for file in lafan_files(lafan_dir, motions):
            (Path(motion_dir) / file.name).symlink_to(file.resolve())
        wrapped_env, use_root_height_obs = build_env(model_folder, motion_dir, num_envs)
        ref = Reference(wrapped_env._env, model, use_root_height_obs)
        motion, frame = rollout_starts(ref.lengths, horizon, start_stride)
        print(f"{len(motion)} starts of {len(ref.names)} motions, {per_batch} per batch of {num_envs} envs")
        offset = ref.offset.cpu().numpy()
        lengths = np.asarray(ref.lengths)
        parts, cos = [], []
        for b in range(0, len(motion), per_batch):
            m = motion[b : b + per_batch]
            rows = torch.as_tensor(offset[m] + frame[b : b + per_batch], device=ref.B.device)
            ends = torch.as_tensor(offset[m] + lengths[m], device=ref.B.device)
            out, c = run_batch(wrapped_env, model, ref, windows, rows, ends, horizon)
            parts.append(out)
            cos.append((c, len(m)))
            print(f"  {b + len(m)}/{len(motion)} starts")
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cos_matrix = sum(c * n for c, n in cos) / sum(n for _, n in cos)
    metrics, table, per_motion = summarize(result, windows, ref.names, motion)
    metrics["z_cos_matrix"] = cos_matrix.tolist()
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "rollouts.npz", motion=np.asarray(ref.names)[motion], frame=frame, windows=np.asarray(windows),
             z_cos_matrix=cos_matrix, **{k: v.astype(np.float16) for k, v in result.items()})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    per_motion.to_csv(output / "per_motion.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(table.to_string(index=False))
        print("mean cos(z_T, z_T'):")
        print(pd.DataFrame(cos_matrix, index=windows, columns=windows).to_string())
    print("Spearman(1 - cos(z_T, z_base), body distance to the base rollout):",
          {k: round(v, 3) for k, v in metrics["spearman_z_gap_body_gap"].items()})
    print(f"Saved {output / 'rollouts.npz'}, metrics.json and per_motion.csv")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
