# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Sequence

import gymnasium
import numpy as np
import ot
import torch
from humenv import CustomManager
from humenv.bench.gym_utils.episodes import Episode
from humenv.misc.motionlib import MotionBuffer
from packaging.version import Version
from hydra.utils import instantiate
from tqdm import tqdm

from humanoidverse.utils.g1_env_config import G1EnvConfigsType
os.environ["ISAAC_USE_GPU_PIPELINE"] = "0"
QVEL_IDX: int = 23

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):

    def cast_obs_wrapper(env) -> gymnasium.Wrapper:
        return gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32))
else:

    def cast_obs_wrapper(env) -> gymnasium.Wrapper:
        return gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32), env.observation_space)


xpos_bodies = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    # "left_wrist_roll_link",
    # "left_wrist_pitch_link",
    # "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    # "right_wrist_roll_link",
    # "right_wrist_pitch_link",
    # "right_wrist_yaw_link",
]


@dataclasses.dataclass(kw_only=True)
class TrackingEvaluationHV:
    # motion_ids: int | List[int] | None = None
    # motion_base_path: str | None = None
    # environment parameters
    num_envs: int = 1
    env: Any = None
    mp_context: str = "forkserver"
    device: str = "cuda"

    # def __post_init__(self) -> None:
        
        

    def run(self, agent: Any, disable_tqdm: bool = False) -> Dict[str, Any]:
        # ids = self.motion_buffer.get_motion_ids()
        self.motion_ids = list(range(len(self.env._motion_lib._curr_motion_ids)))
        # np.random.shuffle(ids)  # shuffle the ids to evenly distribute the motions, as different datasets have different motion length
        num_workers = min(self.num_envs, len(self.motion_ids))
        num_workers = 1
        motions_per_worker = np.array_split(self.motion_ids, num_workers)

        f = functools.partial(
            _async_tracking_worker,
            env=self.env,
            # motion_buffer=self.motion_buffer,
            disable_tqdm=disable_tqdm,
        )
        if num_workers == 1:
            metrics = f((motions_per_worker[0], 0, agent))
        else:
            prev_omp_num_th = os.environ.get("OMP_NUM_THREADS", None)
            os.environ["OMP_NUM_THREADS"] = "1"
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=multiprocessing.get_context(self.mp_context),
            ) as pool:
                inputs = [(x, y, agent) for x, y in zip(motions_per_worker, range(len(motions_per_worker)))]
                list_res = pool.map(f, inputs)
                metrics = {}
                for el in list_res:
                    metrics.update(el)
            if prev_omp_num_th is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = prev_omp_num_th
        return metrics

    def close(self) -> None:
        if self.mp_manager is not None:
            self.mp_manager.shutdown()

from humanoidverse.utils.helpers import get_backward_observation

def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    motion_ids = torch.tensor(motion_ids, device=device)
    num_motions = len(motion_ids)

    assigned = motion_ids[torch.arange(num_envs, device=device) % num_motions]

    motion_to_envs = {}
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs.setdefault(motion_id, []).append(env_id)

    return assigned, motion_to_envs

def _async_tracking_worker(inputs, env, disable_tqdm: bool = False):
    motion_ids, pos, agent = inputs
    # import ipdb; ipdb.set_trace()
    # motion_ids = torch.tensor(motion_ids, device=env.device)
    env._motion_lib.load_motions_for_evaluation()

    metrics = {}
    num_envs = env.num_envs


    # Assign motions to envs in grouped fashion
    assigned_motions, motion_to_envs = group_assign_motions_to_envs_with_map(motion_ids, num_envs)
    

    ctx_dict = {}
    tracking_targets = {}
    target_xpos_dict = {}
    dof_states_list = [None] * num_envs
    root_states_list = [None] * num_envs

    # Precompute context and target states per motion
    for m_id in motion_ids.tolist():
        tracking_target, tracking_target_dict = get_backward_observation(env, m_id)
        z = agent.model.backward_map(tracking_target)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        ctx = agent.model.project_z(z)
        ctx_dict[m_id] = ctx
        tracking_targets[m_id] = tracking_target.cpu()
        # import ipdb; ipdb.set_trace()

        ref_root_init_state = torch.cat([
            tracking_target_dict["ref_body_pos"][0, 0],
            tracking_target_dict["ref_body_rots"][0, 0],
            tracking_target_dict["ref_body_vels"][0, 0],
            tracking_target_dict["ref_body_angular_vels"][0, 0]
        ])
        # import ipdb; ipdb.set_trace()
        for env_id in motion_to_envs[m_id]:
            dof_init_state = torch.zeros_like(env.simulator.dof_state.view(num_envs, -1, 2)[0])
            dof_init_state[..., 0] = tracking_target_dict["dof_pos"][0]
            dof_init_state[..., 1] = tracking_target_dict["ref_dof_vel"][0]
            dof_states_list[env_id] = dof_init_state
            root_states_list[env_id] = ref_root_init_state
            target_xpos_dict[env_id] = tracking_target_dict["ref_body_pos"][:, :len(xpos_bodies)]

    for i in range(num_envs):
        if dof_states_list[i] is None:
            dof_states_list[i] = torch.zeros_like(env.simulator.dof_state.view(num_envs, -1, 2)[0])
        if root_states_list[i] is None:
            root_states_list[i] = torch.zeros_like(root_states_list[0])

    target_states = {
        "dof_states": torch.stack(dof_states_list),
        "root_states": torch.stack(root_states_list)
    }

    env_ids = list(range(num_envs))
    
    with torch.no_grad():
        observation, info = env.reset_envs_idx(env_ids, target_states=target_states)
        # hard code 
        observation, reward, reset, info = env.step({"actions": env.actions})
        # info = {}
        obs_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in observation.items()}
        info_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in info.items()}

        _episode = Episode()
        _episode.initialise(obs_cpu["actor_obs"], info_cpu)

        xpos_log = [env.simulator._rigid_body_pos.reshape(num_envs, -1, 3)]

        max_ctx_len = max([ctx.shape[0] for ctx in ctx_dict.values()])
        for step in tqdm(range(max_ctx_len), desc="Tracking Evaluation"):

            ctx_batch = []
            for env_id in range(num_envs):
                m_id = assigned_motions[env_id].item()
                ctx = ctx_dict[m_id]
                ctx_t = ctx[step % ctx.shape[0]]
                ctx_batch.append(ctx_t)
            ctx_batch = torch.stack(ctx_batch)
            action = agent.act(observation['actor_obs'], ctx_batch, mean=False)
            observation, reward, reset, info = env.step({"actions": torch.tensor(action, device=env.device, dtype=torch.float32)})

            xpos_log.append(env.simulator._rigid_body_pos.reshape(num_envs, -1, 3))

            _episode.add(
                observation["actor_obs"].cpu(),
                reward.cpu(),
                action,
                reset.cpu(),
                reset.cpu(),
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in info.items()}
            )

        episode_data = _episode.get()
        episode_data["xpos"] = torch.stack(xpos_log)[:-1]
        assert(env._motion_lib._num_unique_motions <= num_envs)
        eval_tqdm = tqdm(range(env._motion_lib._num_unique_motions))
        for env_id in eval_tqdm:
            m_id = assigned_motions[env_id].item()
            tmp = {
                "xpos": episode_data["xpos"][0:ctx_dict[m_id].shape[0], env_id],
                "target_xpos": target_xpos_dict[env_id],
                "tracking_target": tracking_targets[m_id],
                "motion_id": m_id,
                "motion_file": env._motion_lib.curr_motion_keys[m_id],
                "observation": episode_data["observation"][0:(ctx_dict[m_id].shape[0]+1), env_id]
            }
            # import ipdb; ipdb.set_trace()
            metrics.update(_calc_metrics(tmp))

    return metrics

QPOS_START = 23+3
QPOS_END = 23+3+23

def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor):
    # keep only pose part of the observations
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {"emd": transport_cost}


def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats["proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats["distance"] = dist.mean()
    stats["success"] = in_bounds_mask.min().float()
    return stats


def _calc_metrics(ep):
    metr = {}
    # keep only qpos states
    next_obs = torch.tensor(ep["observation"][1:, QPOS_START:QPOS_END], dtype=torch.float32)
    tracking_target = torch.tensor(ep["tracking_target"][:, QPOS_START:QPOS_END], dtype=torch.float32)
    dist_prox_res = distance_proximity(next_obs=next_obs, tracking_target=tracking_target)
    metr.update(dist_prox_res)
    emd_res = emd_numpy(next_obs=next_obs, tracking_target=tracking_target)
    metr.update(emd_res)
    # add distance wrt xpos
    xpos = torch.tensor(ep["xpos"], dtype=torch.float32).view(ep["xpos"].shape[0], -1)  # [keyframes, 72]
    target_xpos = torch.tensor(ep["target_xpos"], dtype=torch.float32).view(ep["target_xpos"].shape[0], -1)

    dist_prox_res = distance_proximity(next_obs=xpos, tracking_target=target_xpos)
    for k, v in dist_prox_res.items():
        metr[f"{k}_xpos"] = v
    # import ipdb; ipdb.set_trace()
    metr["emd_xpos"] = emd_numpy(next_obs=xpos, tracking_target=target_xpos)["emd"]
    jpos_pred = xpos.reshape(xpos.shape[0], -1, 3)  # length x 23 x 3
    jpos_gt = target_xpos.reshape(target_xpos.shape[0], -1, 3)  # length x 23 x 3
    metr["success_phc_linf_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1) <= 0.5).float()
    metr["success_phc_mean_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1).mean(dim=-1) <= 0.5).float()
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}
