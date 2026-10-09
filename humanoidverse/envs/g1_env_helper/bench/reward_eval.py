# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Any, Dict, List

import gymnasium
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.utils.g1_env_config import G1EnvConfigsType


@dataclasses.dataclass(kw_only=True)
class RewardEvaluation:
    tasks: List[str]
    num_episodes: int = 10
    env_config: G1EnvConfigsType

    def run(self, agent: Any, disable_tqdm: bool = True) -> Dict[str, Any]:
        env, mp_info = self.env_config.build(1)

        def reset_task(task):
            if not isinstance(env, (gymnasium.vector.AsyncVectorEnv, gymnasium.vector.SyncVectorEnv)):
                env.unwrapped.set_task(task)
            else:
                env.call("set_task", task)

        task_to_rewards = {task: {"reward": []} for task in self.tasks}
        for task in self.tasks:
            ctx = agent.reward_inference(task=task)
            reset_task(task)

            for _ in range(self.num_episodes):
                observation, info = env.reset()
                cumulative_reward = 0.0
                truncated = False
                terminated = False
                while not truncated and not terminated:
                    observation = tree_map(
                        lambda x: torch.tensor(x, dtype=torch.float32 if x.dtype == np.float64 else None)[None], observation
                    )
                    action = agent.act(observation, ctx).ravel()
                    observation, reward, terminated, truncated, info = env.step(action)
                    cumulative_reward += reward

            task_to_rewards[task]["reward"].append(cumulative_reward)

        env.close()
        if mp_info is not None and mp_info["mp_info"] is not None:
            mp_info["manager"].shutdown()

        return task_to_rewards