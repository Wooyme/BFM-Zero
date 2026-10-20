import unittest

import torch

from experiments.dof_noise_z import joint_noise


class JointNoiseTest(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)
        self.dof = torch.zeros(500, 10)
        self.low, self.high = torch.full((10,), -1.0), torch.full((10,), 1.0)

    def test_picks_n_joints_of_the_group(self):
        group = torch.tensor([2, 5, 7, 9])
        noisy, mask = joint_noise(self.dof, group, 3, 0.1, self.low, self.high, self.gen)
        self.assertTrue((mask.sum(1) == 3).all())
        outside = torch.ones(10, dtype=torch.bool)
        outside[group] = False
        self.assertFalse(mask[:, outside].any())
        self.assertTrue((noisy[~mask] == 0).all())
        self.assertTrue((noisy[mask] != 0).all())

    def test_whole_group_and_spread(self):
        group = torch.arange(10)
        noisy, mask = joint_noise(self.dof, group, 10, 0.2, self.low, self.high, self.gen)
        self.assertTrue(mask.all())
        self.assertAlmostEqual(noisy.std().item(), 0.2, delta=0.01)

    def test_joints_are_spread_over_the_group(self):
        group = torch.tensor([1, 4, 6])
        _, mask = joint_noise(self.dof, group, 1, 0.1, self.low, self.high, self.gen)
        counts = mask[:, group].sum(0).float()
        self.assertTrue((counts > 120).all())

    def test_clipped_to_limits(self):
        low, high = torch.full((10,), -0.05), torch.full((10,), 0.05)
        noisy, _ = joint_noise(self.dof, torch.arange(10), 10, 1.0, low, high, self.gen)
        self.assertTrue(((noisy >= low) & (noisy <= high)).all())
        self.assertTrue((noisy == high).any() and (noisy == low).any())


if __name__ == "__main__":
    unittest.main()
