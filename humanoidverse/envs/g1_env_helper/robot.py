import collections
import copy
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import gymnasium
import mujoco
import numpy as np

from humanoidverse.utils.g1_env_config import G1EnvConfig, get_g1_robot_xml_root
from humenv.misc.motionlib import MotionBuffer

from . import init
from . import rewards as g1_rewards
from .base import G1Base

END_FREEJOINT_QPOS = 7
END_FREEJOINT_QVEL = 6


class ObsType(Enum):
    proprioceptive = 0
    pixel = 1
    proprioceptive_pixel = 2


class StateInit(Enum):
    Default = 0
    Fall = 1
    MoCap = 2
    DefaultAndFall = 3
    MoCapAndFall = 4


# Single scene for all tasks (no separate terrain XMLs).
_SCENE_XML = "scene_29dof_freebase_mujoco.xml"


def task_to_xml(task_name: str) -> Path:
    """Resolve task name to the single G1 scene XML path (package data dir)."""
    root = get_g1_robot_xml_root()
    return root / _SCENE_XML


class G1Env(G1Base):
    def __init__(
        self,
        config: G1EnvConfig = G1EnvConfig(),
        shared_motion_lib: MotionBuffer | None = None,
    ):
        xml_task = config.xml_task
        if "<mujoco" in xml_task:
            xml = xml_task
        else:
            xml = task_to_xml(xml_task).as_posix()
        super().__init__(
            xml_path=xml,
            config=config,
        )
        self._xml_task = xml_task

        self.set_task(config.task)
        self._post_init()

        if self.spec is None:
            from gymnasium.envs.registration import EnvSpec

            self.spec = EnvSpec(config.name)

        self.motion_buffer = shared_motion_lib  # unused for kinematic Default init; keep for API

        # Action space: fixed bounds (no ctrl_config)
        _action_clip = 2.0
        self.action_space = gymnasium.spaces.Box(
            low=-np.ones(self.action_size) * _action_clip,
            high=np.ones(self.action_size) * _action_clip,
            shape=(self.action_size,),
            dtype=np.float64,
        )

        obs, _ = self.reset(seed=self.seed)
        # to keep the same order of the observation dict
        space_dict = OrderedDict()
        for k, v in obs.items():
            if k == "pixel":
                space_dict[k] = gymnasium.spaces.Box(
                    low=0, high=255, shape=(self._config.render_width, self._config.render_height, 3), dtype=np.uint8
                )
            else:
                space_dict[k] = gymnasium.spaces.Box(-np.inf * np.ones_like(v), np.inf * np.ones_like(v), dtype=v.dtype)
        self.observation_space = gymnasium.spaces.Dict(space_dict)

    def _post_init(self) -> None:
        self._obs_type = ObsType[self._config.obs_type]
        self._init_q = np.array(self._mj_model.keyframe("stand").qpos)
        self._default_pose = np.array(self._mj_model.keyframe("stand").qpos[END_FREEJOINT_QPOS:])
        # Support both XML naming conventions: imu_in_pelvis (old freebase) or imu (g1_29dof_mujoco)
        try:
            self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id
        except KeyError:
            self._pelvis_imu_site_id = self._mj_model.site("imu").id

        # Kinematic only: no torque/PD state
        n = self.action_size
        self._lowers, self._uppers = self._mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # get id of torso (for render)
        self.torso_id = self._mj_model.body("torso_link").id
        assert self.torso_id > -1

        # Domain randomization (delay/repeat set by G1EnvRand when used)
        self._observation_noise_level = self._config.noise_config.level
        self._ctrl_config_delay = 0
        self._ctrl_config_repeat_probability = 0.0
        assert not (self._ctrl_config_delay > 0 and self._ctrl_config_repeat_probability > 0), (
            "Control delay and repeat probability cannot both be > 0."
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed = seed
        if options:
            # Only 29-DOF (36-D qpos: 7 free + 29 joints) is supported; no 23-DOF.
            if "qpos" in options:
                q = np.asarray(options["qpos"]).ravel()
                if q.size != self._mj_model.nq:
                    raise ValueError(
                        f"qpos size {q.size} does not match model nq {self._mj_model.nq}. "
                        "Only 36-D qpos (7 free + 29 joints) is supported."
                    )
            self._mj_data = init(self._mj_model, **options)
        else:
            self._mj_data = self.reset_humanoid(self.np_random)

        # Action delay in the episode
        if self._ctrl_config_delay > 0:
            self._action_queue = collections.deque([np.zeros(self.action_size) for i in range(self._ctrl_config_delay + 1)])

        # Phase, freq=U(1.0, 1.5)
        # gait_freq = self.np_random.uniform(low=1.25, high=1.5)
        # phase_dt = 2 * np.pi * self.dt * gait_freq
        # phase = np.array([0, np.pi])
        info = self._get_reset_info()
        obs = self._get_obs(self._mj_data, info, self.np_random, last_action=info["last_act"])

        return obs, info

    def _get_reset_info(self) -> Dict[str, Any]:
        info = {
            "step": 0,
            "last_act": np.zeros(self._mj_model.nu),
            "last_last_act": np.zeros(self._mj_model.nu),
            "motor_targets": np.zeros(self._mj_model.nu),
            "push": np.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": 1,
            "aux_rewards": {},
            "qpos": self._mj_data.qpos.copy(),
            "qvel": self._mj_data.qvel.copy(),
        }
        costs = self._sim2real_costs(self._mj_data, info["last_act"], info["last_last_act"], np.zeros(2))
        for aux_reward in self._config.aux_reward_config.names:
            info["aux_rewards"][aux_reward] = costs[aux_reward]
        self._info = copy.deepcopy(info)
        return info

    def reset_humanoid(self, rng):
        """Kinematic-only: default stand pose with optional xy/yaw randomization."""
        qpos = self._init_q.copy()
        qvel = np.zeros(self._mj_model.nv)
        dxy = rng.uniform(size=(2,), low=-0.5, high=0.5)
        qpos[0:2] = qpos[0:2] + dxy
        yaw = rng.uniform(low=-3.14, high=3.14)
        quat, new_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), yaw)
        mujoco.mju_mulQuat(new_quat, qpos[3:7], quat)
        qpos[3:END_FREEJOINT_QPOS] = new_quat
        qpos[END_FREEJOINT_QPOS:] = qpos[END_FREEJOINT_QPOS:] * rng.uniform(size=(self._mj_model.nq - 7,), low=0.5, high=1.5)
        qvel[0:END_FREEJOINT_QVEL] = rng.uniform(size=(END_FREEJOINT_QVEL,), low=-0.5, high=0.5)
        # Kinematic: do not set ctrl (model may have nu != nq-7)
        return init(self._mj_model, qpos=qpos, qvel=qvel)

    def step(self, action: np.ndarray):
        # Kinematic: action as target joint position offset, no dynamics
        action_scale = 0.25
        self._mj_data.qpos[END_FREEJOINT_QPOS:] = self._default_pose + action_scale * np.clip(action, -2.0, 2.0)
        self._mj_data.qvel[END_FREEJOINT_QVEL:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._info["motor_targets"] = np.zeros(self._mj_model.nu)

        obs = self._get_obs(self._mj_data, self._info, rng=self.np_random, last_action=action)
        done = self._get_termination(self._mj_data)

        # reward related information (kinematic: no contact)
        reward = self.task.compute(self._mj_model, self._mj_data)
        costs = self._sim2real_costs(self._mj_data, action, self._info["last_act"], np.zeros(2))
        for aux_reward in self._config.aux_reward_config.names:
            self._info["aux_rewards"][aux_reward] = costs[aux_reward]

        self._info["push"] = np.zeros(2)
        self._info["step"] += 1
        self._info["push_step"] += 1
        self._info["last_last_act"] = self._info["last_act"]
        self._info["last_act"] = action
        self._info["qpos"] = self._mj_data.qpos.copy()
        self._info["qvel"] = self._mj_data.qvel.copy()
        terminated = False
        truncated = bool(done)
        return obs, reward, terminated, truncated, copy.deepcopy(self._info)

    def _get_termination(self, data: mujoco.MjData) -> np.ndarray:
        # TODO
        return False

    def _get_obs(self, data: mujoco.MjData, info: dict[str, Any], rng: np.random.RandomState, last_action: np.ndarray) -> OrderedDict:
        # projected gravity along the z axis
        gravity = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])
        noisy_gravity = (
            gravity + rng.uniform(-1, 1, size=gravity.shape) * self._observation_noise_level * self._config.noise_config.scales.gravity
        )

        # joint positions except for the freejoint (i.e., floating root) which has a non-local representation (xyz (global) + quaternion for orientation)
        joint_angles = data.qpos[END_FREEJOINT_QPOS:]
        noisy_joint_angles = (
            joint_angles
            + rng.uniform(-1, 1, size=joint_angles.shape) * self._observation_noise_level * self._config.noise_config.scales.joint_pos
        )
        # joint velocies except for the freejoint (i.e., floating root) which has a non-local representation
        joint_vel = data.qvel[END_FREEJOINT_QVEL:]
        noisy_joint_vel = (
            joint_vel
            + rng.uniform(-1, 1, size=joint_vel.shape) * self._observation_noise_level * self._config.noise_config.scales.joint_vel
        )
        # root angular velocities [for a freejoint we have linear vel + angular vel]
        root_vel = data.qvel[3:END_FREEJOINT_QVEL]
        noisy_root_vel = (
            root_vel + rng.uniform(-1, 1, size=root_vel.shape) * self._observation_noise_level * self._config.noise_config.scales.joint_vel
        )

        # cos = np.cos(info["phase"])
        # sin = np.sin(info["phase"])
        # phase = np.concatenate([cos, sin])
        state = np.hstack([
            noisy_joint_angles - self._default_pose,
            noisy_joint_vel,
            noisy_gravity,
            noisy_root_vel,
        ])

        match self._obs_type:
            case ObsType.proprioceptive:
                obs = OrderedDict(state=state, last_action=last_action.copy())
            case ObsType.pixel:
                obs = OrderedDict(pixel=self.render(camera="back"))
            case ObsType.proprioceptive_pixel:
                obs = OrderedDict(state=state, last_action=last_action.copy(), pixel=self.render(camera="back"))
            case _:
                raise ValueError(f"Unknown observation type: {self._obs_type}")

        return obs

    def set_task(self, task: g1_rewards.RewardFunction | str | None) -> None:
        if task is None:
            self.task = g1_rewards.ZeroReward()
        elif isinstance(task, str):
            self.task = make_from_name(task)
        else:
            self.task = task

    ####################################################
    # Sim2Real reward functions
    ####################################################

    def _sim2real_costs(self, data, action, last_action, contact) -> dict[str, np.ndarray]:
        # Kinematic env: no dynamics costs, return zeros for all aux reward names
        names = self._config.aux_reward_config.names
        return {k: np.array(0.0) for k in names}


def make_from_name(
    name: str | None = None,
):
    return g1_rewards.make_from_name(name)
