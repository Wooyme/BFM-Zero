import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import json
import torch
import joblib
import mediapy as media
import numpy as np
from torch.utils._pytree import tree_map
from tqdm import tqdm

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation

# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent


def main(model_folder: Path, data_path: Path | None = None, headless: bool = True, device="cuda", simulator: str = "isaacsim", save_mp4: bool=False, disable_dr: bool = False, disable_obs_noise: bool = False, episode_len: int = 500, video_folder: str | None = None):

    model_folder = Path(model_folder)
    video_folder = Path(video_folder) if video_folder is not None else model_folder / "goal_inference" / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = "model"
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"
    # import ipdb; ipdb.set_trace()
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
        z_dim=model.cfg.archi.z_dim,
        history=('history_actor' in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {output_dir}/{model_name}.onnx")
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    num_envs = 1
    wrapped_env, _ = env_cfg.build(num_envs)
    env = wrapped_env._env
    motion_keys = {k:i for i,k in enumerate(path.split('/')[-1].split('.')[0] for path in wrapped_env._env._motion_lib._motion_data_keys)}
    print("="*80)
    print(env.config.simulator)
    print("-"*80)
    
    # Try to find goal_frames JSON file
    goal_json_paths = [
        HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "goal_frames_lafan29dof.json",
        HUMANOIDVERSE_DIR / "data" / "goal_frames_lafan29dof.json",
    ]
    goal_json = None
    for path in goal_json_paths:
        if path.exists():
            goal_json = str(path)
            break
    
    if goal_json is None:
        raise FileNotFoundError(
            f"Could not find goal_frames_lafan29dof.json. Searched in: {goal_json_paths}"
        )
    
    with open(goal_json, "r") as f:
        goals_to_evaluate = json.load(f)
    pbar = tqdm(goals_to_evaluate, leave=False, disable=False)
    z_dict = {}
    with torch.no_grad():
        for goal in pbar:
            env.set_is_evaluating(motion_keys[goal["motion_name"]])
                # we visulize the first env
            gobs, gobs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0)
            num_frames = next(iter(gobs.values())).shape[0]
            frame_pbar = tqdm(goal["frames"], leave=False, disable=False, desc="frames")
            for frame_idx in frame_pbar:
                if frame_idx >= num_frames:
                    pbar.write(f"  Skipping frame_idx {frame_idx} (motion has {num_frames} frames)")
                    continue
                goal_name = f"{goal['motion_name']}_{frame_idx}"
                goal_observation = {k: v[frame_idx][None,...] for k,v in gobs.items()}
                goal_observation = tree_map(lambda x: torch.tensor(x, device=model.device, dtype=torch.float32), goal_observation)

                z_dict[goal_name] = model.goal_inference(goal_observation).cpu().numpy()
    path = model_folder / "goal_inference"
    path.mkdir(exist_ok=True)
    with open(os.path.join(path, "goal_reaching.pkl"), "wb") as f:
        joblib.dump(z_dict, f)

    if save_mp4:
        rgb_renderer = IsaacRendererWithMuJoco(render_size=256)

    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)
    observation, info = wrapped_env.reset(to_numpy=False)

    frames = []
    counter = 0
    # episode_len: number of sim steps for the optional goal-reaching video (~12 it/s → 5000 ≈ 7 min)
    _pbar = tqdm(desc="steps", disable=False, leave=False, total=episode_len)
    goal_idx = -1
    goal_names = list(z_dict.keys())

    while counter < episode_len:
        if counter % 100 == 0:
            goal_idx = (goal_idx + 1) % len(goal_names)
            print(f"Switching to goal {goal_names[goal_idx]} at step {counter}")
            z = z_dict[goal_names[goal_idx]].copy()
            z = torch.tensor(z, device=model.device, dtype=torch.float32)

        action = model.act(observation, z.repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
        if save_mp4:
            frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
        counter += 1
        _pbar.update(1)
    _pbar.close()
    if save_mp4:
        media.write_video(video_folder / "goal.mp4", frames, fps=50)
        print("Saved video for goal")

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
