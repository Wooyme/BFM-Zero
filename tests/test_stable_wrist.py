import math
import unittest

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from experiments.stable_wrist import WRIST, combine, wrist_offset
from experiments.stable_wrist_view import face_x
from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model
from humanoidverse.envs.g1_env_helper.queries.metrics import METRICS
from humanoidverse.query.motion_query import zero_velocity


def project(z):
    return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


def yaw_quat(yaw):  # wxyz
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


class WristMetricTest(unittest.TestCase):
    def setUp(self):
        self.model = env_model()
        self.data = mujoco.MjData(self.model)
        rng = np.random.default_rng(0)
        self.qpos = self.model.qpos0.copy()
        self.qpos[7:] = rng.uniform(-0.3, 0.3, self.model.nq - 7)

    def offset(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return np.array([METRICS[k](self.model, self.data) for k in WRIST])

    def test_yaw_and_position_invariant(self):
        moved = self.qpos.copy()
        moved[0:3] += (1.0, -2.0, 0.1)
        moved[3:7] = yaw_quat(1.3)
        np.testing.assert_allclose(self.offset(moved), self.offset(self.qpos), atol=1e-9)

    def test_matches_backend_offset(self):
        # the same quantity as the rollout computes from body positions and the xyzw root rotation
        q = self.qpos.copy()
        q[3:7] = yaw_quat(0.7)
        expected = self.offset(q)
        wrist = self.model.body("right_wrist_yaw_link").id
        pelvis = self.model.body("pelvis").id
        state = {"body_pos": torch.as_tensor(self.data.xpos[[pelvis, wrist]][None]),
                 "body_rot": torch.as_tensor(self.data.xquat[[pelvis]][:, [1, 2, 3, 0]][None])}
        np.testing.assert_allclose(wrist_offset(state, 1)[0].numpy(), expected, atol=1e-9)


class CombineTest(unittest.TestCase):
    def test_reward_inference(self):
        # reward_inference: project(sum_i r_i B_i) over the two frames with rewards (1, w)
        gen = torch.Generator().manual_seed(0)
        B_s, B_w = torch.randn(5, 8, generator=gen), torch.randn(7, 8, generator=gen)
        pair = torch.randint(5, (7,), generator=gen)
        z = combine(B_s, B_w, pair, [1.0, 0.5], project)
        self.assertEqual(z.shape, (2, 7, 8))
        for k, w in enumerate((1.0, 0.5)):
            for i in range(7):
                B = torch.stack([B_s[pair[i]], B_w[i]])
                ref = project(torch.tensor([[1.0], [w]]).T @ B)[0]
                self.assertTrue(torch.allclose(z[k, i], ref, atol=1e-6))


class ZeroVelocityTest(unittest.TestCase):
    def db(self):
        nb, ndof = 31, 29
        rng = np.random.default_rng(0)
        obs = {"state": rng.normal(size=(4, 2 * ndof + 6)).astype(np.float32),
               "privileged_state": rng.normal(size=(4, 15 * nb - 2)).astype(np.float32)}
        return MotionDatabase(metrics={}, source_index=np.arange(4),
                              frames={"qvel": np.zeros((4, ndof + 6)), "next_obs": obs}), nb, ndof

    def test_rows(self):
        db, nb, ndof = self.db()
        before = {k: v.copy() for k, v in db.frames["next_obs"].items()}
        zero_velocity(db, np.array([1, 3]))
        state, priv = db.frames["next_obs"]["state"], db.frames["next_obs"]["privileged_state"]
        for r in (0, 2):
            np.testing.assert_array_equal(state[r], before["state"][r])
            np.testing.assert_array_equal(priv[r], before["privileged_state"][r])
        for r in (1, 3):
            self.assertTrue((state[r, ndof : 2 * ndof] == 0).all() and (state[r, 2 * ndof + 3 :] == 0).all())
            np.testing.assert_array_equal(state[r, : ndof], before["state"][r, : ndof])
            np.testing.assert_array_equal(state[r, 2 * ndof : 2 * ndof + 3], before["state"][r, 2 * ndof : 2 * ndof + 3])
            self.assertTrue((priv[r, -6 * nb :] == 0).all())
            np.testing.assert_array_equal(priv[r, : -6 * nb], before["privileged_state"][r, : -6 * nb])

    def test_all(self):
        db, nb, _ = self.db()
        zero_velocity(db)
        self.assertTrue((db.frames["next_obs"]["privileged_state"][:, -6 * nb :] == 0).all())


class ViewerTest(unittest.TestCase):
    def test_face_x(self):
        q = np.zeros(36)
        q[0:3] = 1.0, 2.0, 0.7
        q[3:7] = yaw_quat(2.0)
        out = face_x(q, -1.2)
        np.testing.assert_allclose(out[0:3], (0, -1.2, 0.7))
        np.testing.assert_allclose(out[3:7], (1, 0, 0, 0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
