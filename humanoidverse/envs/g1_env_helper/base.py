from typing import Optional

import gymnasium
import mujoco
import numpy as np
from pathlib import Path

from humanoidverse.utils.g1_env_config import G1EnvConfigsType

def get_mujoco_model(xml: str) -> mujoco.MjModel:
    if "<mujoco" in xml:
        return mujoco.MjModel.from_xml_string(xml)
    p = Path(xml).resolve()
    if not p.exists():
        raise RuntimeError(f"XML file not found: {p}")
    return mujoco.MjModel.from_xml_path(str(p))

class G1Base(gymnasium.Env):
    def __init__(
        self,
        xml_path: str,
        config: G1EnvConfigsType,
        seed: int = 0,
    ):
        self._config = config.model_copy(deep=True)
        self.seed = seed

        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt

        self._xml_path = xml_path
        self._mj_model = get_mujoco_model(xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_data = mujoco.MjData(self._mj_model)

        self.renderer = None
        self.task = None

    @property
    def data(self) -> mujoco.MjData:
        """Return the current simulation data."""
        return self._mj_data

    @property
    def model(self) -> mujoco.MjModel:
        """Return the current simulation model."""
        return self._mj_model

    @property
    def dt(self) -> float:
        """Control timestep for the environment."""
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        """Simulation timestep for the environment."""
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    def render(
        self,
        camera: str | None = None,
        scene_option: Optional[mujoco.MjvOption] = None,
    ) -> np.ndarray | None:
        # We had to hijack whole render code because we need stuff between "update_scene" and "render"
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self._mj_model,
                width=self._config.render_width,
                height=self._config.render_height,
            )
            mujoco.mj_forward(self._mj_model, self._mj_data)
        camera = camera or self._config.camera
        self.renderer.update_scene(self._mj_data, camera=camera, scene_option=scene_option)
        if self._config.render_task:
            self._render_task()

        pixels = self.renderer.render()
        return pixels

    def _render_task(self):
        pass

    # # Sensor readings.

    # def get_gravity(self, data: mujoco.MjData, frame: str) -> np.ndarray:
    #     """Return the gravity vector in the world frame."""
    #     return asap_g1.get_sensor_data(self._mj_model, data, f"{GRAVITY_SENSOR}_{frame}")

    # def get_global_linvel(self, data: mujoco.MjData, frame: str) -> np.ndarray:
    #     """Return the linear velocity of the robot in the world frame."""
    #     return asap_g1.get_sensor_data(self._mj_model, data, f"{GLOBAL_LINVEL_SENSOR}_{frame}")

    # def get_global_angvel(self, data: mujoco.MjData, frame: str) -> np.ndarray:
    #     """Return the angular velocity of the robot in the world frame."""
    #     return asap_g1.get_sensor_data(self._mj_model, data, f"{GLOBAL_ANGVEL_SENSOR}_{frame}")

    # def get_local_linvel(self, data: mujoco.MjData, frame: str) -> np.ndarray:
    #     """Return the linear velocity of the robot in the local frame."""
    #     return get_sensor_data(self._mj_model, data, f"{LOCAL_LINVEL_SENSOR}_{frame}")

    # def get_accelerometer(self, data: mujoco.MjData, frame: str) -> np.ndarray:
    #     """Return the accelerometer readings in the local frame."""
    #     return asap_g1.get_sensor_data(self._mj_model, data, f"{ACCELEROMETER_SENSOR}_{frame}")

    # def get_gyro(self, data: mujoco.MjData, frame: str) -> np.ndarray:
    #     """Return the gyroscope readings in the local frame."""
    #     return asap_g1.get_sensor_data(self._mj_model, data, f"{GYRO_SENSOR}_{frame}")