from typing import Any

import mujoco

from humenv.misc.motionlib import MotionBuffer

from humanoidverse.utils.g1_env_config import G1EnvRandConfig
from .robot import END_FREEJOINT_QVEL, G1Env, task_to_xml


class G1EnvRand(G1Env):
    def __init__(
        self,
        config: G1EnvRandConfig = G1EnvRandConfig(),
        shared_motion_lib: MotionBuffer | None = None,
    ):
        super().__init__(config=config, shared_motion_lib=shared_motion_lib)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # terrain randomization
        if self._config.domain_rand_config.terrain_randomization.enable:
            new_xml_task = self.rand_terrain()

            if self._xml_task != new_xml_task:
                # Load the xml associated to the new terrain
                self._xml_task = new_xml_task
                self._xml_path = (task_to_xml(new_xml_task).as_posix(),)

                self._mj_model = mujoco.MjModel.from_xml_path(task_to_xml(new_xml_task).as_posix())
                self._mj_data = mujoco.MjData(self._mj_model)

        # randomize dynamics parameters
        self.rand_dynamics()

        # determine the noise level of the episode
        # self._config.noise_config.level = self.rand_step_noise_level()
        self._observation_noise_level = self.rand_step_noise_level()

        # determine the amount of delay in the episode (in steps)
        # self._config.ctrl_config.delay = self.rand_action_delay()
        # self._config.ctrl_config.repeat_probability = self.rand_action_repeat()
        self._ctrl_config_delay = self.rand_action_delay()
        self._ctrl_config_repeat_probability = self.rand_action_repeat()

        assert not (self._ctrl_config_delay > 0 and self._ctrl_config_repeat_probability > 0), (
            "Action delay and action repeat cannot be both enabled at the same time. Please set one of them to 0."
        )

        # randomize torque
        # scale limits: *U(lower, upper)
        if self._config.domain_rand_config.torque_lim_scale_config.enable:
            self._torque_lim_scale_noise = self.np_random.uniform(
                self._config.domain_rand_config.torque_lim_scale_config.range[0],
                self._config.domain_rand_config.torque_lim_scale_config.range[1],
                size=self.action_size,
            )

        obs, info = super().reset(seed=seed, options=options)

        return obs, info

    def rand_terrain(self):
        # randomize over available terrains
        terrain_type = self.np_random.integers(0, len(self._config.domain_rand_config.terrain_randomization.terrains))
        return self._config.domain_rand_config.terrain_randomization.terrains[terrain_type]

    def rand_step_noise_level(self):
        noise_level = 0.0

        if self._config.domain_rand_config.noise_level_randomization is True:
            # randomize between the noisy or deterministic step
            if self.np_random.integers(0, 2) == 0:
                noise_level = 0.0
            else:
                noise_level = 1.0

        return noise_level

    def rand_dynamics(self):
        # Floor / foot friction: =U(0.4, 1.0).
        if self._config.domain_rand_config.friction_randomization.enable:
            friction = self.np_random.uniform(
                low=self._config.domain_rand_config.friction_randomization.range[0],
                high=self._config.domain_rand_config.friction_randomization.range[1],
            )
            # apply the friction to the foot contact pairs only on tangential direction
            self._mj_model.pair_friction[self.feet_foot_contact_pairs_ids, 0:2] = friction

        njnt = self._mj_model.njnt

        # Scale static friction: *U(0.5, 2.0).
        if self._config.domain_rand_config.frictionloss_randomization.enable:
            frictionloss = self._mj_model.dof_frictionloss[END_FREEJOINT_QVEL:] * self.np_random.uniform(
                low=self._config.domain_rand_config.frictionloss_randomization.range[0],
                high=self._config.domain_rand_config.frictionloss_randomization.range[1],
                size=(njnt - 1,),
            )
            self._mj_model.dof_frictionloss[END_FREEJOINT_QVEL:] = frictionloss

        # Scale armature: *U(1.0, 1.05).
        if self._config.domain_rand_config.armature_randomization.enable:
            armature = self._mj_model.dof_armature[END_FREEJOINT_QVEL:] * self.np_random.uniform(
                low=self._config.domain_rand_config.armature_randomization.range[0],
                high=self._config.domain_rand_config.armature_randomization.range[1],
                size=(njnt - 1,),
            )
            self._mj_model.dof_armature[END_FREEJOINT_QVEL:] = armature

        # Scale all link masses: *U(0.9, 1.1).
        if self._config.domain_rand_config.mass_randomization.enable:
            # Scale all link masses
            dmass = self.np_random.uniform(
                low=self._config.domain_rand_config.mass_randomization.range[0],
                high=self._config.domain_rand_config.mass_randomization.range[1],
                size=(self._mj_model.nbody,),
            )
            self._mj_model.body_mass[:] = self._mj_model.body_mass * dmass

        # Add mass to torso: +U(-1.0, 1.0).
        if self._config.domain_rand_config.torso_mass_randomization.enable:
            dmass = self.np_random.uniform(
                low=self._config.domain_rand_config.torso_mass_randomization.range[0],
                high=self._config.domain_rand_config.torso_mass_randomization.range[1],
            )
            self._mj_model.body_mass[self.torso_id] = self._mj_model.body_mass[self.torso_id] + dmass

    def rand_action_delay(self):
        # Randomize the action delay
        if self._config.domain_rand_config.ctrl_delay_config.enable is False:
            return 0
        else:
            delay = self.np_random.integers(
                low=self._config.domain_rand_config.ctrl_delay_config.range[0],
                high=self._config.domain_rand_config.ctrl_delay_config.range[1] + 1,
            )
            return int(delay)

    def rand_action_repeat(self):
        # Randomize the action delay
        if self._config.domain_rand_config.ctrl_repeat_config.enable is False:
            return 0.0
        else:
            if self._config.domain_rand_config.ctrl_repeat_config.range[0] == self._config.domain_rand_config.ctrl_repeat_config.range[1]:
                # If the range is the same, return the fixed value
                delay = self._config.domain_rand_config.ctrl_repeat_config.range[0]
            else:
                # Randomize the action delay
                delay = self.np_random.uniform(
                    low=self._config.domain_rand_config.ctrl_repeat_config.range[0],
                    high=self._config.domain_rand_config.ctrl_repeat_config.range[1],
                )
            return float(delay)


if __name__ == "__main__":
    print("TEST FOR DOMAIN RANDOMIZATION")
    config = G1EnvRandConfig()
    print(f"Domain randomization config: {config.domain_rand_config}")

    env = G1EnvRand(
        xml_task="asap_flat_terrain",
        config=config,
        config_overrides=None,
    )
    for i in range(10):
        print(f"Iteration: {i}")
        env.reset()
        # print all the parameters of the environment
        print(f"Current XML task (terrain type): {env._xml_task}")
        print(f"Noise level for the episode: {env._config.noise_config.level}")
        print(f"Contact pair friction parameters: {env._mj_model.pair_friction[0:2, 0:2]}")
        print(f"Dynamics parameters - DOF friction loss: {env._mj_model.dof_frictionloss[6:]}")
        joint_names = [env._mj_model.joint(j).name for j in range(env._mj_model.njnt)]
        print("Dynamics parameters - DOF armature:")
        for name, armature in zip(joint_names[1:], env._mj_model.dof_armature[6:]):
            print(f"  Joint: {name}, Armature: {armature}")
        print("Body masses of all links:")
        for body_id in range(env._mj_model.nbody):
            body_name = env._mj_model.body(body_id).name
            body_mass = env._mj_model.body_mass[body_id]
            print(f"  Body: {body_name}, Mass: {body_mass}")
        print(f"Mass of the torso link: {env._mj_model.body_mass[env._mj_model.body('torso_link').id]}")