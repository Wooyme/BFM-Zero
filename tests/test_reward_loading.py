import ast
import importlib
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from humanoidverse.envs.g1_env_helper import rewards


class RewardLoadingTests(unittest.TestCase):
    def test_existing_reward_names(self):
        cases = {
            "ZERO": (rewards.ZeroReward, {}),
            "move-ego--90-0.3": (rewards.LocomotionReward, {"move_angle": -90, "move_speed": 0.3}),
            "move-ego-low0.5-0-0": (rewards.LocomotionReward, {"stay_low": True, "stand_height": 0.5}),
            "jump-1.4": (rewards.JumpReward, {"jump_height": 1.4}),
            "rotate-z--5-0.5": (rewards.RotationReward, {"target_ang_velocity": -5}),
            "raisearms-l-m": (rewards.ArmsReward, {"left_pose": "l", "right_pose": "m"}),
            "sitonground": (rewards.SitOnGroundReward, {"constrained_knees": True}),
            "crouch-0.25": (rewards.SitOnGroundReward, {"pelvis_height_th": 0.25, "knees_not_on_ground": True}),
            "totheknee": (rewards.ToTheKnee, {}),
            "move-arms-0-0.7-m-l": (rewards.MoveArmsReward, {"move_speed": 0.7, "right_pose": "l"}),
            "spin-arms--5-l-m": (rewards.SpinArmsReward, {"target_ang_velocity": -5}),
        }
        for name, (reward_cls, attributes) in cases.items():
            with self.subTest(name=name):
                reward = rewards.make_from_name(name)
                self.assertIs(type(reward), reward_cls)
                for key, value in attributes.items():
                    self.assertEqual(getattr(reward, key), value)
                self.assertEqual(pickle.loads(pickle.dumps(reward)), reward)

    def test_all_inference_tasks_resolve(self):
        script = Path(__file__).resolve().parents[1] / "humanoidverse" / "reward_inference.py"
        tree = ast.parse(script.read_text())
        tasks = next(
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "tasks" for target in node.targets)
        )
        for task in tasks:
            with self.subTest(task=task):
                self.assertIsInstance(rewards.make_from_name(task), rewards.RewardFunction)

    def test_unknown_name(self):
        for name in ("unknown-reward", "", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown reward name"):
                rewards.make_from_name(name)

    def test_discovers_new_scripts_and_nested_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "custom_reward.py").write_text(
                "from humanoidverse.envs.g1_env_helper.rewards import RewardFunction, ZeroReward\n"
                "class Helper:\n"
                "    pass\n"
                "class AbstractReward(RewardFunction):\n"
                "    pass\n"
                "class CustomReward(RewardFunction):\n"
                "    def compute(self, model, data):\n"
                "        return 42.0\n"
                "    @staticmethod\n"
                "    def reward_from_name(name):\n"
                "        return CustomReward() if name == 'custom-task' else None\n"
                "Alias = CustomReward\n"
            )
            nested = root / "custom_nested"
            nested.mkdir()
            (nested / "__init__.py").write_text("")
            (nested / "other.py").write_text(
                "from ..custom_reward import CustomReward\n"
                "class NestedReward(CustomReward):\n"
                "    @staticmethod\n"
                "    def reward_from_name(name):\n"
                "        return NestedReward() if name == 'nested-task' else None\n"
            )
            try:
                with patch.object(rewards, "__path__", [*rewards.__path__, directory]):
                    importlib.invalidate_caches()
                    classes = rewards.get_reward_classes()
                    self.assertEqual(len(classes), len(set(classes)))
                    self.assertEqual(sum(cls.__name__ == "CustomReward" for cls in classes), 1)
                    self.assertNotIn("AbstractReward", [cls.__name__ for cls in classes])
                    self.assertNotIn("Helper", [cls.__name__ for cls in classes])
                    self.assertEqual(rewards.make_from_name("custom-task").compute(None, None), 42.0)
                    self.assertEqual(type(rewards.make_from_name("nested-task")).__name__, "NestedReward")
                    self.assertIsInstance(rewards.make_from_name("zero"), rewards.ZeroReward)
            finally:
                for name in list(sys.modules):
                    if name.startswith(rewards.__name__ + ".custom_"):
                        del sys.modules[name]
                for name in ("custom_reward", "custom_nested"):
                    rewards.__dict__.pop(name, None)


if __name__ == "__main__":
    unittest.main()
