import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import torch

from humanoidverse.pose_hold import (
    DEFAULT_MODEL,
    Observations,
    ScoreConfig,
    Scores,
    control_parameters,
    load_training_config,
    transform_actions,
)
from humanoidverse.pose_hold_eval import load_catalog, parser, rank_values, run


def poses(n=2):
    q = torch.zeros((n, 36))
    q[:, 2], q[:, 3] = 0.8, 1
    return q


def score_constant(target, current, duration=5):
    score = Scores(target, 10, duration / 10)
    for i in range(10):
        score.add(i, {"qpos": current, "qvel": torch.zeros(len(target), 35)}, np.ones(len(target), dtype=bool))
    return score.finish()


class ScoreTests(unittest.TestCase):
    def test_exact_hold_quaternion_sign_and_yaw(self):
        target = poses(3)
        q = target.clone()
        q[1, 3:7] *= -1
        q[2, 3:7] = torch.tensor([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        result = score_constant(target, q)
        np.testing.assert_allclose(result["score"][:2], 100)
        self.assertAlmostEqual(result["base_error_mean_deg"][2], 90, places=4)
        self.assertLess(result["score"][2], result["score"][0])

    def test_formula_units_and_full_trajectory_aggregation(self):
        target = poses(1)
        q = target.clone()
        angle = np.deg2rad(15)
        q[0, 3:7] = torch.tensor([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        q[0, 7:] = 0.2
        q[0, 0] = 0.25
        result = score_constant(target, q)
        self.assertAlmostEqual(result["cost"][0], 1, places=5)
        self.assertAlmostEqual(result["score"][0], 100 / np.e, places=4)
        self.assertAlmostEqual(result["joint_rmse_mean"][0], 0.2, places=6)
        s = Scores(target, 10, 0.1)
        for i in range(10):
            q = target.clone()
            q[:, 0] = 0.2 if i < 5 else 0
            s.add(i, {"qpos": q, "qvel": torch.zeros(1, 35)}, [True])
        result = s.finish()
        self.assertEqual(result["end_displacement"][0], 0)
        self.assertAlmostEqual(result["path_length"][0], 0.4, places=6)
        self.assertAlmostEqual(result["max_displacement"][0], 0.2, places=6)

    def test_invalid_ranking_and_configuration(self):
        q = poses(2)
        q[1, 7] = float("nan")
        result = score_constant(poses(2), q)
        self.assertFalse(result["valid"][1])
        self.assertEqual(result["score"][1], 0)
        zero_quaternion = poses(1)
        zero_quaternion[:, 3:7] = 0
        self.assertFalse(score_constant(poses(1), zero_quaternion)["valid"][0])
        np.testing.assert_array_equal(rank_values(np.array([1, 1, 0.1]), np.array([True, True, False]), np.array([2, 1, 0])), [2, 1, 3])
        with self.assertRaises(ValueError):
            ScoreConfig(base_weight=0.8)


def stationary_state(qpos):
    qpos = torch.as_tensor(qpos, dtype=torch.float32).clone()
    n = len(qpos)
    body_pos = qpos[:, None, :3].repeat(1, 30, 1)
    body_pos[:, :, 2] += torch.linspace(0, 0.5, 30)
    return {
        "qpos": qpos,
        "qvel": torch.zeros(n, 35),
        "body_pos": body_pos,
        "body_rot": qpos[:, None, [4, 5, 6, 3]].repeat(1, 30, 1),
        "body_vel": torch.zeros(n, 30, 3),
        "body_ang_vel": torch.zeros(n, 30, 3),
        "valid": torch.ones(n, dtype=torch.bool),
    }


class StationaryBackend:
    """Ideal hold fixture for export/resume tests; no physics approximation."""

    def __init__(self, config, batch_size, clearance, device):
        self.config = config
        self.params = control_parameters(config)
        self.dt = 0.02
        self.metadata = {"backend": "isaacsim", "batch_size": batch_size, "fixture": True}

    def reset(self, qpos):
        qpos = np.array(qpos, copy=True)
        qpos[:, :2] = 0
        self.state = stationary_state(qpos)
        return self.state

    def step(self, action):
        applied, _ = transform_actions(action, self.config, self.params)
        return self.state, applied

    def close(self):
        pass


@unittest.skipUnless((DEFAULT_MODEL / "config.yaml").exists(), "local G1 checkpoint/config unavailable")
class ObservationExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_training_config(DEFAULT_MODEL)

    def test_action_scaling_matches_training(self):
        params = control_parameters(self.config)
        self.assertEqual(params["kp"][12], 300)
        action = torch.ones(1, 29)
        applied, target = transform_actions(action, self.config, params)
        np.testing.assert_allclose(applied, 5)
        np.testing.assert_allclose(target[0], params["default"] + 1.25 * params["effort"] / params["kp"], atol=1e-6)
        applied, _ = transform_actions(action * 100, self.config, params)
        np.testing.assert_allclose(applied, 5)

    def test_observation_contract_history_and_reference_privileged(self):
        from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max
        from humanoidverse.utils.torch_utils import my_quat_rotate, quat_mul

        q = poses(2)
        q[1, 3:7] = torch.tensor([0.9238795, 0, 0, 0.3826834])
        state = stationary_state(q)
        builder = Observations(self.config, "cpu")
        obs, s = builder.build(state, np.zeros((2, 29)), reset=True)
        self.assertEqual(
            {k: tuple(v.shape[1:]) for k, v in obs.items()},
            {"state": (64,), "last_action": (29,), "history_actor": (372,), "privileged_state": (463,)},
        )
        ext = self.config["robot"]["motion"]["extend_config"][0]
        parent = self.config["robot"]["body_names"].index(ext["parent_name"])
        offset = torch.tensor(ext["pos"]).expand(2, -1)
        rot = torch.tensor(ext["rot"])[[1, 2, 3, 0]].expand(2, -1)
        position = my_quat_rotate(rot, my_quat_rotate(s["body_rot"][:, parent], offset)) + s["body_pos"][:, parent]
        rotations = quat_mul(s["body_rot"][:, parent], rot, w_last=True)
        body_pos = torch.cat((s["body_pos"], position[:, None]), 1)
        body_rot = torch.cat((s["body_rot"], rotations[:, None]), 1)
        zero = torch.zeros_like(body_pos)
        reference = torch.cat(list(compute_humanoid_observations_max(body_pos, body_rot, zero, zero, True, True).values()), -1)
        torch.testing.assert_close(obs["privileged_state"], reference)
        initial = {k: v.clone() for k, v in obs.items()}
        for _ in range(5):
            builder.build(state, np.ones((2, 29)))
        reset, _ = builder.build(state, np.zeros((2, 29)), reset=True)
        for key in initial:
            torch.testing.assert_close(initial[key], reset[key])

    @patch("humanoidverse.pose_hold_eval.make_backend", side_effect=StationaryBackend)
    def test_resume_initial_only_outputs(self, make_backend):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "input.pkl"
            motion = {
                "dof": np.zeros((3, 29)),
                "root_trans_offset": np.tile([0, 0, 0.8], (3, 1)),
                "root_rot": np.tile([0, 0, 0, 1], (3, 1)),
                "fps": 30,
            }
            joblib.dump({"pose": motion}, data)
            args = parser().parse_args(
                [
                    "--model-folder",
                    str(DEFAULT_MODEL),
                    "--data-path",
                    str(data),
                    "-o",
                    str(tmp / "out"),
                    "--num-envs",
                    "2",
                    "--device",
                    "cpu",
                    "--duration",
                    ".02",
                ]
            )
            run(args)
            part = tmp / "out/parts/batch_000000000.npz"
            timestamp = part.stat().st_mtime_ns
            with np.load(part, allow_pickle=False) as cache:
                self.assertEqual(cache["initial_qpos"].shape, (2, 36))
                self.assertEqual(cache["latent"].shape, (2, 256))
                self.assertTrue(all(cache[k].ndim <= 2 for k in cache))
            args.resume = True
            with patch("humanoidverse.pose_hold_eval.evaluate_batch", side_effect=AssertionError("Recomputed cached batch")):
                run(args)
            self.assertEqual(part.stat().st_mtime_ns, timestamp)
            from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase

            db = MotionDatabase.load(tmp / "out/motion_db.pkl")
            self.assertEqual(len(db), 3)
            np.testing.assert_array_equal(db.metrics["rank"], [1, 2, 3])
            self.assertFalse(db.frames["qvel"].any())
            args.duration = 0.04
            with self.assertRaisesRegex(ValueError, "differ"):
                run(args)


class CatalogTests(unittest.TestCase):
    def test_original_indices_no_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "x.pkl"

            def motion(n):
                return {
                    "dof": np.zeros((n, 29)),
                    "root_trans_offset": np.tile([0, 0, 0.8], (n, 1)),
                    "root_rot": np.tile([0, 0, 0, 1], (n, 1)),
                    "fps": 30,
                }

            joblib.dump({"a": motion(5), "b": motion(4)}, source)
            catalog = load_catalog(source, stride=2)
            np.testing.assert_array_equal(catalog["source_index"], [0, 2, 4, 5, 7])
            np.testing.assert_array_equal(catalog["frame_index"], [0, 2, 4, 0, 2])
            np.testing.assert_allclose(catalog["time"], [0, 2 / 30, 4 / 30, 0, 2 / 30])


if __name__ == "__main__":
    unittest.main()
