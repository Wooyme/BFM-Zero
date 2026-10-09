import copy
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import gymnasium
import mujoco
import numpy as np
from ml_collections.config_dict import config_dict

from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root
from . import init, step
from .base import G1Base
from .robot import END_FREEJOINT_QPOS, END_FREEJOINT_QVEL, ObsModel


def quat_rotate_inverse_numpy(q, v):
    # shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c


def default_config_29dof() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        render_height=400,
        render_width=400,
        render_task=False,
        camera="track",
        obs_model="state",
        soft_joint_pos_limit_factor=0.95,
        soft_torque_limit_factor=0.95,
        use_23_dof=False,  # Use 23 dof instead of 29 dof
        init_config=config_dict.create(
            state_init="Default",
        ),
        noise_config=config_dict.create(
            level=0.0,  # Set to 0.0 to disable noise, 1.0 to enable it.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,
                torque=0.1,
            ),
            torque_range=[0.5, 1.5],
        ),
        ctrl_repeat_config=config_dict.create(repeat_ctrl_probability=0.0),
        ctrl_config=config_dict.create(
            action_scale=0.25,
            clip_torques=True,
            action_clip_value=5.0,  # it is used if use_range_limits is False
            dof_effort_limit_list=[
                88.0,
                88.0,
                88.0,
                139.0,
                50.0,
                50.0,
                88.0,
                88.0,
                88.0,
                139.0,
                50.0,
                50.0,
                88.0,
                50.0,
                50.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                5.0,
                5.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                5.0,
                5.0,
            ],
            # PD Drive parameters [N*m/rad]
            stiffness=config_dict.create(
                hip_yaw=100,
                hip_roll=100,
                hip_pitch=100,
                knee=200,
                ankle_pitch=20,
                ankle_roll=20,
                waist_yaw=400,
                waist_roll=400,
                waist_pitch=400,
                shoulder_pitch=90,
                shoulder_roll=60,
                shoulder_yaw=20,
                elbow=60,
                wrist=4,
            ),
            # [N*m/rad]  # [N*m*s/rad]
            damping=config_dict.create(
                hip_yaw=2.5,
                hip_roll=2.5,
                hip_pitch=2.5,
                knee=5.0,
                ankle_pitch=0.2,
                ankle_roll=0.1,
                waist_yaw=5.0,
                waist_roll=5.0,
                waist_pitch=5.0,
                shoulder_pitch=2.0,
                shoulder_roll=1.0,
                shoulder_yaw=0.4,
                elbow=1.0,
                wrist=0.2,
            ),
        ),
    )


# to extract the 23 dof from the 29 dof humanoid
QPOS_IDX_23_IN_29 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32]
QVEL_IDX_23_IN_29 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31]
ACT_IDX_23_IN_29 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25]


class G1Env29dof(G1Base):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config_29dof(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        xml = get_g1_robot_xml_root() / "scene_29dof_freebase_noadditional_actuators.xml"
        super().__init__(
            xml_path=xml,
            config=config,
            config_overrides=config_overrides,
        )
        self.model = self._mj_model
        self._post_init()
        self.action_space = gymnasium.spaces.Box(
            low=-np.ones(self.action_size),
            high=np.ones(self.action_size),
            shape=(self.action_size,),
            dtype=np.float64,
        )

        self._init_q = np.array(self._mj_model.keyframe("stand").qpos)
        self._default_pose_23 = np.array(self._mj_model.keyframe("stand").qpos[QPOS_IDX_23_IN_29][END_FREEJOINT_QPOS:])
        self._default_pose_29 = np.array(self._mj_model.keyframe("stand").qpos[END_FREEJOINT_QPOS:])
        obs, _ = self.reset(seed=self.seed)
        # to keep the same order of the observation dict
        space_dict = OrderedDict()
        for k, v in obs.items():
            space_dict[k] = gymnasium.spaces.Box(-np.inf * np.ones_like(v), np.inf * np.ones_like(v), dtype=v.dtype)
        self.observation_space = gymnasium.spaces.Dict(space_dict)

    def _post_init(self):
        self._obs_model = ObsModel[self._config.obs_model]

        # controller: model-sized defaults only (no config physics)
        n = self.real_action_size
        self.torque_limits = np.full(n, 100.0)
        self.p_gains = np.full(n, 50.0)
        self.d_gains = np.full(n, 5.0)
        self._kp_scale = np.ones_like(self.p_gains)
        self._kd_scale = np.ones_like(self.p_gains)
        self._soft_torque_limits = self.torque_limits * self._config.soft_torque_limit_factor
        self._torque_lim_scale_noise = np.ones_like(self.torque_limits)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        if options:
            self._mj_data = init(self._mj_model, **options)
        else:
            self._mj_data = self.reset_humanoid(self.np_random)

        info = {
            "step": 0,
            "last_act": np.zeros(self.action_size),
            "last_last_act": np.zeros(self.action_size),
            "motor_targets": np.zeros(self._mj_model.nu),
            "aux_rewards": {},
            "qpos": self._mj_data.qpos.copy(),
            "qvel": self._mj_data.qvel.copy(),
        }

        self._last_extended_action = np.zeros(self.real_action_size)  # last action used in the step
        obs = self._get_obs(self._mj_data, info, self.np_random, last_action=info["last_act"])
        self._info = copy.deepcopy(info)
        return obs, info

    def reset_humanoid(self, rng):
        def _default():
            qpos = self._init_q.copy()
            qvel = np.zeros(self._mj_model.nv)
            # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
            dxy = rng.uniform(size=(2,), low=-0.5, high=0.5)
            qpos[0:2] = qpos[0:2] + dxy
            yaw = rng.uniform(low=-3.14, high=3.14)
            quat, new_quat = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), yaw)
            mujoco.mju_mulQuat(new_quat, qpos[3:7], quat)
            qpos[3:END_FREEJOINT_QPOS] = new_quat
            # qpos[END_FREEJOINT_QPOS:]=*U(0.5, 1.5)
            qpos[END_FREEJOINT_QPOS:] = qpos[END_FREEJOINT_QPOS:] * rng.uniform(size=(self._mj_model.nq - 7,), low=0.5, high=1.5)
            # d(xyzrpy)=U(-0.5, 0.5)
            qvel[0:END_FREEJOINT_QVEL] = rng.uniform(size=(END_FREEJOINT_QVEL,), low=-0.5, high=0.5)
            return qpos, qvel

        mujoco.mj_resetData(self._mj_model, self._mj_data)
        qpos, qvel = _default()
        return init(self._mj_model, qpos=qpos, qvel=qvel, ctrl=np.zeros(self._mj_model.nu))

    def step(self, action: np.ndarray):
        action = action * self._config.ctrl_config.action_clip_value
        if self._config.use_23_dof:
            assert len(action) == 23, f"Action should be of length 23, got {len(action)}"
        if action.shape == (23,):
            # in case we control the 23 dof of the humanoid
            extendend_action = np.zeros(self.real_action_size)
            extendend_action[ACT_IDX_23_IN_29] = action
        elif action.shape == (self.real_action_size,):
            extendend_action = action
        else:
            raise ValueError(f"Action shape {action.shape} is not valid. Expected (23,) or ({self._mj_model.nu - 6},).")

        if self.np_random.random() < self._config.ctrl_repeat_config.repeat_ctrl_probability:
            # if control delay is active, use the last action
            # Sticky actions: Instead of always simulating the action passed to the environment, there is a small probability that the previously executed action is used instead
            extendend_action = self._last_extended_action
            action = self._info["last_act"]

        self._last_extended_action = extendend_action.copy()

        for _ in range(self.n_substeps):
            motor_targets = self._compute_torques(extendend_action)
            step(self._mj_model, self._mj_data, motor_targets, 1)
        self._info["motor_targets"] = self._mj_data.qfrc_actuator[END_FREEJOINT_QVEL:].copy()

        obs = self._get_obs(self._mj_data, self._info, rng=self.np_random, last_action=action)
        done = self._get_termination(self._mj_data)

        reward = 0.0
        # self._info["push"] = push
        # self._info["step"] += 1
        # self._info["push_step"] += 1
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
        base_quat = data.qpos[3:7].copy().reshape(1, -1)  # quaternion representing the base orientation
        v = np.array([[0, 0, -1]])
        gravity = quat_rotate_inverse_numpy(base_quat, v).ravel()  # gravity vector in the base frame
        noisy_gravity = (
            gravity + rng.uniform(-1, 1, size=gravity.shape) * self._config.noise_config.level * self._config.noise_config.scales.gravity
        )

        # joint positions except for the freejoint (i.e., floating root) which has a non-local representation (xyz (global) + quaternion for orientation)
        joint_angles = data.qpos[QPOS_IDX_23_IN_29][END_FREEJOINT_QPOS:]
        noisy_joint_angles = (
            joint_angles
            + rng.uniform(-1, 1, size=joint_angles.shape) * self._config.noise_config.level * self._config.noise_config.scales.joint_pos
        )
        # joint velocies except for the freejoint (i.e., floating root) which has a non-local representation
        joint_vel = data.qvel[QVEL_IDX_23_IN_29][END_FREEJOINT_QVEL:]
        noisy_joint_vel = (
            joint_vel
            + rng.uniform(-1, 1, size=joint_vel.shape) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel
        )
        # root angular velocities [for a freejoint we have linear vel + angular vel]
        root_vel = data.qvel[3:END_FREEJOINT_QVEL]
        noisy_root_vel = (
            root_vel
            + rng.uniform(-1, 1, size=root_vel.shape) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel
        )

        state = [
            noisy_joint_angles - self._default_pose_23,  # 23
            noisy_joint_vel,  # 23
            noisy_gravity,  # 3
            noisy_root_vel,  # 3
            # phase, # 4
        ]
        state = np.hstack(state)

        privileged_state = np.zeros(357)

        # TODO not very scalable approach, need to rework this obs creation
        if self._obs_model == ObsModel.priv_posrotlinangvel_action:
            return OrderedDict(
                state=state,
                last_action=last_action.copy(),
                privileged_state=privileged_state,
            )
        else:  # this is to guarantee that if we concatenate we keep this order
            return OrderedDict(
                state=state,
                privileged_state=privileged_state,
            )

    def _compute_torques(self, actions):
        actions_scaled = actions * self._config.ctrl_config.action_scale
        # F(t) = K_p (\bar{q}(t) - q(t)) - K_d (\dot{q}(t)) where \bar{q}(t) is the desired position
        # q(t) is the joint position and \dot{q}(t) is the joint velocity
        # the floating root (i.e., freejoint corresponding to the pelvis) is not actuated. By being a freejoint we need to
        # remove the first 7 dimensions from qpos and 6 dimensions from qvel
        torques = (
            self._kp_scale * self.p_gains * (actions_scaled + self._default_pose_29 - self._mj_data.qpos[END_FREEJOINT_QPOS:])
            - self._kd_scale * self.d_gains * self._mj_data.qvel[END_FREEJOINT_QVEL:]
        )
        noisy_torques = (
            torques
            + (2 * self.np_random.uniform(size=torques.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.torque
            * self._torque_lim_scale_noise  # used in domain randomization to scale the range (shape=torque_limits.shape)
            * self.torque_limits
        )
        if self._config.ctrl_config.clip_torques:
            # torque limit does not change with the domain randomization
            # These limits are already enforced by mujoco
            # (because autolimits="true" and we specify actuatorfrcrange for each joint) but better to have it also here
            noisy_torques = np.clip(noisy_torques, -self.torque_limits, self.torque_limits)
        # add 6 initial actions for the free base (i.e., floating root) which are not controlled
        noisy_torques = np.concatenate(
            (
                np.zeros(6),  # free base actions (not controlled)
                noisy_torques,
            )
        )
        return noisy_torques

    @property
    def action_size(self) -> int:
        if self._config.use_23_dof:
            return 23
        else:
            return self.real_action_size

    @property
    def real_action_size(self) -> int:
        """Returns the real action size, which is the number of actuated joints."""
        return self._mj_model.nu - 6  # Exclude the free base actions