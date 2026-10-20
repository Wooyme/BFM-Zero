import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.hand_compose import Kinematics, body_slice, cell_keys, compose, from_heading, reward_z, solve_ik
from experiments.motion_to_z import to_heading

CONFIG = Path(__file__).resolve().parents[1] / "new_model_for_training_code_inference" / "config.yaml"


def project(z):
    return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


def random_quat(n, gen):
    return F.normalize(torch.randn(n, 4, generator=gen), dim=-1)


class HeadingTest(unittest.TestCase):
    def test_round_trip(self):
        gen = torch.Generator().manual_seed(0)
        q, v = random_quat(50, gen), torch.randn(50, 3, 3, generator=gen)
        self.assertTrue(torch.allclose(from_heading(to_heading(v, q), q), v, atol=1e-5))

    def test_yaw_only(self):
        # a quarter turn about z maps the x axis to y
        q = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
        out = from_heading(torch.tensor([[1.0, 0.0, 0.5]]), q)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 1.0, 0.5]]), atol=1e-6))


class LayoutTest(unittest.TestCase):
    def test_body_slice(self):
        # 31 bodies with root height: 1 + 30 * 3 + 31 * 12 = 463 columns, body 1 right after the height
        self.assertEqual(body_slice(1, 31, True, 463), slice(1, 4))
        self.assertEqual(body_slice(22, 31, True, 463), slice(64, 67))
        self.assertEqual(body_slice(22, 31, False, 462), slice(63, 66))
        with self.assertRaises(ValueError):
            body_slice(22, 30, True, 463)

    def test_cell_keys(self):
        p = torch.tensor([[0.05, -0.05, 0.19], [0.1, 0.0, -0.2]])
        self.assertEqual(cell_keys(p, 0.1).tolist(), [[0, -1, 1], [1, 0, -2]])


class ComposeTest(unittest.TestCase):
    def test_norm_and_identity(self):
        gen = torch.Generator().manual_seed(0)
        zw, zh = project(torch.randn(6, 256, generator=gen)), project(torch.randn(6, 256, generator=gen))
        lam = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 0.0])
        z = compose(zw, zh, lam, project)
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.full((6,), 16.0)))
        self.assertTrue(torch.allclose(z[0], zw[0], atol=1e-5))
        self.assertTrue(torch.allclose(compose(zw, zh, 1.0, project), project(zw + zh)))

    def test_reward_z_is_reward_wr_inference(self):
        gen = torch.Generator().manual_seed(0)
        B, wrist = torch.randn(500, 8, generator=gen), torch.rand(500, 3, generator=gen)
        targets = torch.rand(5, 3, generator=gen)
        z, hits = reward_z(B, wrist, targets, 0.1, project, chunk=2)
        for i, t in enumerate(targets):
            r = torch.exp(-(wrist - t).square().sum(-1, keepdim=True) / (2 * 0.1**2))
            # FBModel.reward_wr_inference: (r * softmax(10 r)).T @ B, projected
            expected = project((r * torch.softmax(10 * r, dim=0)).T @ B)[0]
            self.assertTrue(torch.allclose(z[i], expected, atol=1e-4))
            self.assertEqual(hits[i].item(), int((r > 0.5).sum()))

    def test_reward_z_weighted_mean(self):
        # without a projection: the weighted mean, so a constant delta comes back as is
        gen = torch.Generator().manual_seed(0)
        wrist, targets = torch.rand(300, 3, generator=gen), torch.rand(4, 3, generator=gen)
        delta = torch.full((300, 8), 0.3)
        z, _ = reward_z(delta, wrist, targets, 0.1, None)
        self.assertTrue(torch.allclose(z, torch.full((4, 8), 0.3), atol=1e-5))


@unittest.skipUnless(CONFIG.exists(), "needs the model config for the skeleton")
class IKTest(unittest.TestCase):
    def test_reaches_a_reachable_wrist(self):
        from omegaconf import OmegaConf

        from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        kin = Kinematics(Humanoid_Batch(OmegaConf.load(CONFIG).robot.motion), "cpu")
        gen = torch.Generator().manual_seed(0)
        n = 8
        arm = torch.tensor([15, 16, 17, 18]).expand(n, -1)  # left shoulder pitch, roll, yaw, elbow
        wrist = torch.full((n,), 22)  # left_wrist_yaw_link
        low, high = kin.low[arm], kin.high[arm]
        root_pos, root_rot = torch.randn(n, 3, generator=gen), random_quat(n, gen)
        dof = torch.zeros(n, 29)
        goal = dof.scatter(1, arm, low + (high - low) * (0.3 + 0.4 * torch.rand(n, 4, generator=gen)))
        target = kin.positions(root_pos, root_rot, goal)[torch.arange(n), wrist]
        q_mid = (low + high) / 2
        out, err = solve_ik(kin, root_pos, root_rot, dof, arm, wrist, target, q_mid, q_mid, iters=300, reg=0.0)
        self.assertLess(err.max().item(), 0.01)
        self.assertTrue(torch.equal(out[:, :15], dof[:, :15]))  # only the arm moves
        self.assertTrue(((out.gather(1, arm) >= low - 1e-6) & (out.gather(1, arm) <= high + 1e-6)).all())


if __name__ == "__main__":
    unittest.main()
