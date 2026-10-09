import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import json
import torch
import joblib
import mediapy as media
import rich
import warnings
import h5py
import time
from collections.abc import Mapping

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, _to_torch
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.envs.g1_env_helper.bench import RewardEvaluationHV, RewardWrapperHV
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation

# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent

def make_reward_agent(model, model_folder: Path, num_samples: int) -> RewardWrapperHV:
    print("Loading the replay buffer...", end=" ", flush=True)
    start_t = time.time()
    buffer_path = model_folder / "checkpoint/buffers/train_reduced"
    if buffer_path.is_dir():
        # Load reduced buffer if that exists
        dataset = DictBuffer.load(buffer_path, device="cpu")
        print("Loaded reduced buffer")
    else:
        # Try loading the original dataset
        buffer_path = model_folder / "checkpoint/buffers/train"
        dataset = TrajectoryDictBufferMultiDim.load(buffer_path, device="cpu")
        print("Loaded original buffer")
    # dataset = fast_load_buffer(model_folder / "checkpoint/buffers/train", device="cpu")
    print(f"done in {time.time()-start_t}s")
    inference_function = "reward_wr_inference"
    return RewardWrapperHV(
        model=model,
        inference_dataset=dataset,
        num_samples_per_inference=num_samples,
        inference_function=inference_function,
        max_workers=24,
        process_executor=True,
        env_model=str(HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"),
    )


def main(model_folder: Path, data_path: Path | None = None, headless: bool = True, device="cuda", simulator: str = "isaacsim", save_mp4: bool=False, episode_length: int=500, video_folder: str | None = None, disable_dr: bool = False, disable_obs_noise: bool = False, num_samples: int = 150_000, n_inferences: int = 1, skip_rollouts: bool = False):
    model_folder = Path(model_folder)
    video_folder = Path(video_folder) if video_folder is not None else model_folder / "reward_inference" / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(data_path)

    if not Path(config["env"]["lafan_tail_path"]).exists():
        config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    # config["env"]["hydra_overrides"].append("env.config.lie_down_init=True")
    # config["env"]["hydra_overrides"].append("env.config.lie_down_init_prob=1")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    rich.print(config["env"])
    num_envs = 1
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs)

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
    tasks = [
        # stand
        "move-ego-0-0",
        "move-ego-low0.5-0-0",

        # locomotion medium
        "move-ego-0-0.7",
        # "move-ego-90-0.7",
        # "move-ego-180-0.7",
        # "move-ego--90-0.7",

        # "move-ego-low0.6-0-0.7",

        # locomotion slow
        "move-ego-0-0.3",
        "move-ego-90-0.3",
        "move-ego-180-0.3",
        "move-ego--90-0.3",
        
        # locomotion fast
        # "move-ego-0-1",
        # "move-ego-90-1",
        # "move-ego-180-1",
        # "move-ego--90-1",

        # spin
        "rotate-z-5-0.5",
        "rotate-z--5-0.5",
    
        # raise arms
        "raisearms-l-l",
        "raisearms-l-m",
        "raisearms-m-l",
        "raisearms-m-m",


        # move + arms
        "move-arms-0-0.7-m-m",
        "move-arms-90-0.7-m-m",
        "move-arms-180-0.4-m-m",
        "move-arms--90-0.7-m-m",
        "move-arms-0-0.7-l-m",
        "move-arms-90-0.7-l-m",
        "move-arms-180-0.4-l-m",
        "move-arms--90-0.7-l-m",
        "move-arms-0-0.7-m-l",
        "move-arms-90-0.7-m-l",
        "move-arms-180-0.4-m-l",
        "move-arms--90-0.7-m-l",
        "move-arms-0-0.7-l-l",
        "move-arms-90-0.7-l-l",
        "move-arms-180-0.4-l-l",
        "move-arms--90-0.7-l-l",

        # spin + arms
        "spin-arms-5-l-l",
        "spin-arms--5-l-l",
        "spin-arms-5-l-m",
        "spin-arms--5-l-m",
        # "spin-arms-5-m-m",
        # "spin-arms--5-m-m",
        "spin-arms-5-m-l",
        "spin-arms--5-m-l",

        # sit
        "crouch-0",
        "crouch-0.25",
        "sitonground",
    ]

    reward_eval_agent = make_reward_agent(model, model_folder, num_samples)
    z_dict = {}
    for r in range(n_inferences):
        for task in tasks:
            print(f"Started inference for {task}...", end=" ", flush=True)
            start_t = time.time()
            z = reward_eval_agent.reward_inference(task=task)
            z_dict[task] = z_dict.get(task, []) + [z.cpu()]
            print(f"done in {time.time()-start_t}s")

            path = model_folder / "reward_inference"
            path.mkdir(exist_ok=True)
            with open(os.path.join(path, "reward_locomotion.pkl"), "wb") as f:
                joblib.dump(z_dict, f)
            print(f"Saved file at {path}/reward_locomotion.pkl")

    # z_dict = joblib.load(model_folder / "reward_inference/reward_locomotion.pkl")

    if not skip_rollouts:
        print("Generating videos...")
        if save_mp4:
            rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
        for task in tasks:
            frames = []
            for z in z_dict[task]:
                z = z.repeat(num_envs, 1).to(device)
                
                observation, info = wrapped_env.reset(to_numpy=False, reset_to_default_pose=True)
                if save_mp4:
                    frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
                for i in range(episode_length):
                    action = model.act(observation, z, mean=True)
                    observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)

                    if save_mp4:
                        frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
            if save_mp4:
                file = video_folder / f"{task}.mp4"
                media.write_video(file, frames, fps=50)
                print(f"Saved video for {task}: {file}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
