import collections
import numbers
import time
import typing as tp

import numpy as np
import pydantic
from humanoidverse.envs.g1_env_helper.bench import RewardEvaluation, TrackingEvaluation
from humanoidverse.utils.g1_env_config import G1EnvConfig, G1EnvRandConfig

from humanoidverse.agents.evaluations.base import BaseEvalConfig, extract_model
from humanoidverse.agents.wrappers.humenvbench import RewardWrapper, TrackingWrapper


class G1EnvTrackingEvaluationConfig(BaseEvalConfig):
    name: tp.Literal["g1env_tracking_eval"] = "g1env_tracking_eval"
    name_in_logs: str = "g1env_tracking_eval"
    motions: str
    motions_root: str
    tracking_env_cfg: G1EnvConfig | G1EnvRandConfig = pydantic.Field(G1EnvConfig(), discriminator="name")
    num_envs: int = 50

    def build(self):
        return G1EnvTrackingEvaluation(self)


class G1EnvTrackingEvaluation:
    def __init__(self, config: G1EnvTrackingEvaluationConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, logger, **kwargs):
        model = extract_model(agent_or_model)
        eval_agent = TrackingWrapper(model=model)
        tracking_eval = TrackingEvaluation(
            motions=self.cfg.motions,
            motion_base_path=self.cfg.motions_root,
            env_config=self.cfg.tracking_env_cfg,
            num_envs=self.cfg.num_envs,
        )
        start_t = time.time()
        print(f"Tracking started at {time.ctime(start_t)}", flush=True)
        tracking_metrics = tracking_eval.run(agent=eval_agent, disable_tqdm=True)
        duration = time.time() - start_t
        print(f"Tracking eval time: {duration}")
        wandb_dict = {}
        aggregate = collections.defaultdict(list)
        for _, metr in tracking_metrics.items():
            for k, v in metr.items():
                if isinstance(v, numbers.Number):
                    aggregate[k].append(v)
        for k, v in aggregate.items():
            wandb_dict[k] = np.mean(v)
            wandb_dict[f"{k}#std"] = np.std(v)
        wandb_dict["time"] = duration

        if logger is not None:
            for k, v in tracking_metrics.items():
                v["motion_name"] = k
                v["timestep"] = timestep
                logger.log(v)

        return tracking_metrics, wandb_dict


class G1EnvRewardEvaluationConfig(BaseEvalConfig):
    name: tp.Literal["g1env_reward_eval"] = "g1env_reward_eval"
    name_in_logs: str = "g1env_reward_eval"
    tasks: list[str]
    reward_env_cfg: G1EnvConfig | G1EnvRandConfig = pydantic.Field(G1EnvRandConfig(), discriminator="name")
    num_episodes: int = 10
    max_workers: int = 12
    process_executor: bool = True

    num_inference_workers: int = 1
    num_inference_samples: int = 50_000

    def build(self):
        return G1EnvRewardEvaluation(self)

    @classmethod
    def requires_replay_buffer(self):
        return True


class G1EnvRewardEvaluation:
    def __init__(self, config: G1EnvRewardEvaluationConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, replay_buffer, logger, **kwargs):
        # Including **kwargs to swallow extra arguments to make calls to evals easier
        inference_function: str = "reward_wr_inference"

        model = extract_model(agent_or_model)

        eval_agent = RewardWrapper(
            model=model,
            inference_dataset=replay_buffer["train"],
            num_samples_per_inference=self.cfg.num_inference_samples,
            inference_function=inference_function,
            max_workers=self.cfg.num_inference_workers,
            process_executor=False if self.cfg.num_inference_workers == 1 else True,
            make_env_fn=self.cfg.reward_env_cfg,
        )
        reward_eval = RewardEvaluation(
            tasks=self.cfg.tasks,
            num_episodes=self.cfg.num_episodes,
            env_config=self.cfg.reward_env_cfg,
        )
        start_t = time.time()
        reward_metrics = {}
        wandb_dict = {}
        if not replay_buffer["train"].empty():
            print(f"Reward started at {time.ctime(start_t)}", flush=True)
            reward_metrics = reward_eval.run(agent=eval_agent, disable_tqdm=True)
            duration = time.time() - start_t
            print(f"Reward eval time: {duration}")

            avg_return = []
            for task in reward_metrics.keys():
                wandb_dict[f"{task}/return"] = np.mean(reward_metrics[task]["reward"])
                wandb_dict[f"{task}/return#std"] = np.std(reward_metrics[task]["reward"])
                avg_return.append(reward_metrics[task]["reward"])
            wandb_dict["return"] = np.mean(avg_return)
            wandb_dict["return#std"] = np.std(avg_return)
            wandb_dict["time"] = duration

        # log reward results
        if logger is not None:
            for k, v in reward_metrics.items():
                n = len(v[list(v.keys())[0]])
                v["task"] = [k] * n
                v["timestep"] = [timestep] * n
                logger.log(v)

        return reward_metrics, wandb_dict