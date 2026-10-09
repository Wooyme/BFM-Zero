import abc
from typing import Collection, Dict, Mapping, Tuple

import mujoco
import numpy as np


def in_bounds(value: float, lo: float | None, hi: float | None) -> bool:
    """lo <= value <= hi, with None meaning unbounded."""
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def parse_bounds(cfg: Mapping[str, float | None], names: Collection[str]) -> Dict[str, Tuple[float | None, float | None]]:
    """{"min_x": 1, "max_x": 2, "max_y": 3} -> {"x": (1, 2), "y": (None, 3)}, for x, y in names."""
    bounds = {}
    for key, value in cfg.items():
        side, _, name = key.partition("_")
        if side not in ("min", "max") or name not in names:
            raise ValueError(f"Expected min_<name> / max_<name>, got '{key}'; names: {list(names)}")
        lo, hi = bounds.get(name, (None, None))
        bounds[name] = (value, hi) if side == "min" else (lo, value)
    return bounds


class QueryFunction(abc.ABC):
    """Selects frames from a motion buffer. Mirrors RewardFunction, with a bool in place of a reward.

    Subclasses are dataclasses built from YAML (see queries.load_query); name bounds min_<x> / max_<x>,
    with None for an open bound.
    """

    @abc.abstractmethod
    def match(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> bool: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> bool:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        return self.match(model, data)
