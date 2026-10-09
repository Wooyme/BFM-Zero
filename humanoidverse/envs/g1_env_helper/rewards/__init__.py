"""G1 rewards, discovered automatically from the modules in this package.

Add a module containing a concrete RewardFunction subclass and implement
reward_from_name(name), returning None for names the class does not support.
No registration or changes to the inference script are required.
"""

import importlib
import inspect
import pkgutil

# Preserve the public imports previously provided by rewards.py.
from .base import (
    ALIGNMENT_BOUNDS,
    COORD_TO_INDEX,
    GRAVITY_VECTOR,
    REWARD_LIMITS,
    RewardFunction,
    add_arrow_from_xpos_to_direction,
    add_visual_arrow,
    get_center_of_mass_linvel,
    get_sensor_data,
    get_torso_upright,
    get_xmat,
    get_xpos,
    quat_rotate_inverse_numpy,
    rot2eul,
)
from .arms import ArmsReward, MoveArmsReward, SpinArmsReward
from .jump import JumpReward
from .locomotion import LocomotionReward
from .posture import SitOnGroundReward, ToTheKnee
from .rotation import RotationReward
from .zero import ZeroReward

__all__ = [
    "ALIGNMENT_BOUNDS",
    "COORD_TO_INDEX",
    "GRAVITY_VECTOR",
    "REWARD_LIMITS",
    "RewardFunction",
    "ArmsReward",
    "JumpReward",
    "LocomotionReward",
    "MoveArmsReward",
    "RotationReward",
    "SitOnGroundReward",
    "SpinArmsReward",
    "ToTheKnee",
    "ZeroReward",
    "add_arrow_from_xpos_to_direction",
    "add_visual_arrow",
    "get_center_of_mass_linvel",
    "get_sensor_data",
    "get_torso_upright",
    "get_xmat",
    "get_xpos",
    "quat_rotate_inverse_numpy",
    "rot2eul",
    "get_reward_classes",
    "make_from_name",
]


def get_reward_classes() -> list[type[RewardFunction]]:
    """Load all reward modules and return their concrete classes once each."""
    reward_classes = []
    seen = set()
    modules = sorted(pkgutil.walk_packages(__path__, prefix=__name__ + "."), key=lambda module: module.name)
    for module_info in modules:
        module = importlib.import_module(module_info.name)
        for _, reward_cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(reward_cls, RewardFunction)
                and not inspect.isabstract(reward_cls)
                and reward_cls not in seen
            ):
                reward_classes.append(reward_cls)
                seen.add(reward_cls)
    return reward_classes


def make_from_name(name: str | None = None) -> RewardFunction:
    """Return the first reward accepting name, or raise for an unknown name."""
    if isinstance(name, str):
        for reward_cls in get_reward_classes():
            reward = reward_cls.reward_from_name(name)
            if reward is not None:
                return reward
    raise ValueError(f"Unknown reward name: {name}")
