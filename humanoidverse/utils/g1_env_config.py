# G1 env config and xml root for inference/bench (data/robots/g1).

import importlib.util
from pathlib import Path
from typing import Any, Literal

import typing as tp

import gymnasium
import pydantic
from pydantic import Field

from humenv import CustomManager
from humenv.misc.motionlib import MotionBuffer


def get_g1_robot_xml_root() -> Path:
    """Path to humanoidverse/data/robots/g1 (scene xmls, meshes, goal_frames). Resolved from package location."""
    spec = importlib.util.find_spec("humanoidverse")
    if spec is not None and spec.origin is not None:
        pkg_dir = Path(spec.origin).resolve().parent
    else:
        pkg_dir = Path(__file__).resolve().parent.parent
    return pkg_dir / "data" / "robots" / "g1"


class BaseConfig(pydantic.BaseModel):
    """Base class for model configurations."""

    model_config = pydantic.ConfigDict(extra="forbid", strict=True, use_enum_values=True, frozen=True)
    name: Literal["BaseConfig"] = "BaseConfig"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name") or cls.name == "BaseConfig":
            cls.name = cls.__name__
            cls.__annotations__["name"] = Literal[cls.__name__]
        else:
            cls.__annotations__["name"] = Literal[(cls.__name__,) + cls.__annotations__["name"].__args__]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in config {self.__class__.__name__}")

    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"The object {self} did not have valid build function. Did you forget to define it?")


def flatten_frame_stack(obs):
    for k, v in obs.items():
        obs[k] = v.reshape(-1)
    return obs


class StateInitConfig(BaseConfig):
    state_init: tp.Literal["Default", "Fall", "MoCap", "DefaultAndFall", "MoCapAndFall"] = "Default"
    fall_prob: float = 0.2
    motions: str | None | tp.List[str] = None
    motion_base_path: str | None = None


class PushConfig(BaseConfig):
    enable: bool = False
    interval_range: tp.List[float] = Field(default_factory=lambda: [5.0, 10.0])
    magnitude_range: tp.List[float] = Field(default_factory=lambda: [0.1, 2.0])


class NoiseScalesConfig(BaseConfig):
    joint_pos: float = 0.03
    joint_vel: float = 1.5
    gravity: float = 0.05
    linvel: float = 0.1
    gyro: float = 0.2
    torque: float = 0.1


class NoiseConfig(BaseConfig):
    level: float = 0.0
    scales: NoiseScalesConfig = NoiseScalesConfig()


class AuxRewardConfig(BaseConfig):
    names: tp.List[str] = Field(
        default_factory=lambda: [
            "penalty_torques", "penalty_dof_acc", "penalty_dof_vel", "penalty_action_rate",
            "limits_dof_pos", "limits_dof_vel", "limits_torque", "penalty_slippage",
        ]
    )


class G1EnvConfig(BaseConfig):
    name: tp.Literal["g1env"] = "g1env"
    seed: tp.Optional[int] = None
    vectorization_mode: tp.Literal["sync", "async"] = "async"
    mp_context: tp.Optional[str] = None
    max_episode_steps: int | None = 500
    task: str | None = None
    xml_task: str = "asap_flat_terrain_old"
    init_config: StateInitConfig = StateInitConfig()
    camera: str = "track"
    render_height: int = 400
    render_width: int = 400
    render_task: bool = False
    add_time: bool = True
    frame_stack: int = 1
    flatten_obs: bool = False
    add_action_rescaling: bool = True
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    obs_model: str = "state"
    obs_type: str = "proprioceptive"
    soft_joint_pos_limit_factor: float = 0.95
    soft_torque_limit_factor: float = 0.95
    soft_dof_vel_limit_factor: float = 0.95
    # 29 DOF velocity limits (same order as dof_effort_limit_list)
    dof_vel_limit_list: tp.List[float] = Field(
        default_factory=lambda: [
            32.0, 32.0, 32.0, 20.0, 37.0, 37.0, 32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
            32.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
        ]
    )
    feet_geoms: tp.List[str] = Field(default_factory=lambda: ["left_foot", "right_foot"])
    feet_linvel_sensors: tp.List[str] = Field(default_factory=lambda: ["left_foot_global_linvel", "right_foot_global_linvel"])
    noise_config: NoiseConfig = NoiseConfig()
    push_config: PushConfig = PushConfig()
    aux_reward_config: AuxRewardConfig = AuxRewardConfig()

    @property
    def object_class(self):
        from humanoidverse.envs.g1_env_helper.robot import G1Env
        return G1Env

    def build(self, num_envs: int = 1) -> tp.Tuple[gymnasium.Env, tp.Any]:
        assert num_envs >= 1
        wrappers = []
        if self.max_episode_steps is not None:
            wrappers.append(lambda env: gymnasium.wrappers.TimeLimit(env, max_episode_steps=self.max_episode_steps))
        if self.add_action_rescaling:
            wrappers.append(lambda env: gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1))
        if self.frame_stack > 1:
            wrappers.append(lambda env: gymnasium.wrappers.FrameStackObservation(env, stack_size=self.frame_stack, padding_type="reset"))
            wrappers.append(
                lambda env: gymnasium.wrappers.TransformObservation(
                    env, flatten_frame_stack,
                    gymnasium.spaces.Dict({k: gymnasium.spaces.utils.flatten_space(v) for k, v in env.observation_space.items()}),
                )
            )
        if self.flatten_obs:
            wrappers.append(gymnasium.wrappers.FlattenObservation)
        if self.add_time:
            wrappers.append(lambda env: gymnasium.wrappers.TimeAwareObservation(env, flatten=False))
        mp_info = None
        shared_lib = None
        if num_envs > 1:
            if self.init_config.motions is not None:
                manager = CustomManager()
                manager.start()
                shared_lib = manager.MotionBuffer(files=self.init_config.motions, base_path=self.init_config.motion_base_path)
                mp_info = {"manager": manager, "motion_buffer": shared_lib}
            envs = [self.create_single_env(wrappers=wrappers, shared_motion_lib=shared_lib) for _ in range(num_envs)]
            if self.vectorization_mode == "sync":
                env = gymnasium.vector.SyncVectorEnv(envs)
            elif self.vectorization_mode == "async":
                env = gymnasium.vector.AsyncVectorEnv(envs, context=self.mp_context)
            else:
                raise ValueError(f"Unknown vectorization mode: {self.vectorization_mode}.")
        else:
            env = self.create_single_env(wrappers=wrappers, shared_motion_lib=None)()
        env.reset(seed=self.seed)
        return env, {"mp_info": mp_info}

    def create_single_env(
        self, wrappers: tp.Optional[tp.List[tp.Callable]] = None, shared_motion_lib: tp.Optional[MotionBuffer] = None
    ) -> tp.Callable:
        def trunk():
            env = self.object_class(config=self, shared_motion_lib=shared_motion_lib)
            if wrappers is not None:
                for wrapper in wrappers:
                    env = wrapper(env)
            return env
        return trunk


class TorqueLimScaleConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[0.5, 1.5])


class CtrlDelayConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[int] | tp.Tuple[int, int] = Field(default=[0, 2])


class CtrlRepeatConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[0.25, 0.25])


class TerrainRandomizationConfig(BaseConfig):
    enable: bool = Field(False)
    terrains: tp.List[str] = Field(default_factory=lambda: ["asap_flat_terrain", "asap_rough_terrain"])


class FrictionRandomizationConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[0.4, 1.0])


class FrictionLossRandomizationConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[0.5, 2.0])


class ArmatureRandomizationConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[1.0, 1.05])


class MassRandomizationConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[0.9, 1.1])


class TorsoMassRandomizationConfig(BaseConfig):
    enable: bool = Field(False)
    range: tp.List[float] | tp.Tuple[float, float] = Field(default=[-1.0, 1.0])


class DomainRandomizationConfig(BaseConfig):
    noise_level_randomization: bool = Field(default=True)
    torque_lim_scale_config: TorqueLimScaleConfig = TorqueLimScaleConfig()
    ctrl_delay_config: CtrlDelayConfig = CtrlDelayConfig()
    ctrl_repeat_config: CtrlRepeatConfig = CtrlRepeatConfig()
    terrain_randomization: TerrainRandomizationConfig = TerrainRandomizationConfig()
    friction_randomization: FrictionRandomizationConfig = FrictionRandomizationConfig()
    frictionloss_randomization: FrictionLossRandomizationConfig = FrictionLossRandomizationConfig()
    armature_randomization: ArmatureRandomizationConfig = ArmatureRandomizationConfig()
    mass_randomization: MassRandomizationConfig = MassRandomizationConfig()
    torso_mass_randomization: TorsoMassRandomizationConfig = TorsoMassRandomizationConfig()


class G1EnvRandConfig(G1EnvConfig):
    name: tp.Literal["g1envrand"] = "g1envrand"
    domain_rand_config: DomainRandomizationConfig = DomainRandomizationConfig()

    @property
    def object_class(self):
        from humanoidverse.envs.g1_env_helper.robot_random import G1EnvRand
        return G1EnvRand


G1EnvConfigsType = tp.Union[G1EnvConfig, G1EnvRandConfig]
