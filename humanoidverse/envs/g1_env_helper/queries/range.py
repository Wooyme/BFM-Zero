import dataclasses
from typing import Dict, Tuple

import mujoco

from .base import QueryFunction, in_bounds, parse_bounds
from .metrics import METRICS


@dataclasses.dataclass
class RangeQuery(QueryFunction):
    """Frames whose METRICS all lie in bounds given as min_<metric> / max_<metric>.

    YAML:
        type: RangeQuery
        min_base_speed: 0.5
        max_base_speed: 1.0
        max_torso_pitch: 10
    """

    bounds: Dict[str, Tuple[float | None, float | None]] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_config(cls, **cfg: float | None) -> "RangeQuery":
        return cls(bounds=parse_bounds(cfg, METRICS))

    def match(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> bool:
        return all(in_bounds(METRICS[name](model, data), lo, hi) for name, (lo, hi) in self.bounds.items())
