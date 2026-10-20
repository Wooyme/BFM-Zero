import math
import unittest

import torch
import torch.nn.functional as F

from experiments.z_to_motion import DIRECTIONS, equivalent_cos, null_direction, perturb, variants


def sphere(n, d):
    return math.sqrt(d) * F.normalize(torch.randn(n, d), dim=-1)


class PerturbTest(unittest.TestCase):
    def test_cosine_and_norm(self):
        torch.manual_seed(0)
        z, u = sphere(6, 32), torch.randn(6, 32)
        cos = torch.tensor([1.0, 0.995, 0.98, 0.9, 0.5, 0.0])
        out = perturb(z, u, cos)
        torch.testing.assert_close(F.cosine_similarity(out, z, dim=-1), cos, atol=1e-5, rtol=0)
        torch.testing.assert_close(out.norm(dim=-1), z.norm(dim=-1))

    def test_moves_towards_u(self):
        z, u = sphere(4, 16), sphere(4, 16)
        near, far = perturb(z, u, torch.full((4,), 0.9)), perturb(z, u, torch.full((4,), 0.5))
        self.assertTrue((F.cosine_similarity(far, u, dim=-1) > F.cosine_similarity(near, u, dim=-1)).all())

    def test_cosine_one_is_identity(self):
        z = sphere(3, 8)
        torch.testing.assert_close(perturb(z, torch.zeros(3, 8), torch.ones(3)), z)


class NullDirectionTest(unittest.TestCase):
    def test_orthogonal_to_rows_and_z(self):
        torch.manual_seed(0)
        z = sphere(5, 20)
        J = torch.randn(5, 4, 20)
        zn = F.normalize(z, dim=-1)
        J = J - (J @ zn[..., None]) * zn[:, None]  # tangent Jacobian
        _, _, Vh = torch.linalg.svd(J, full_matrices=False)
        u = null_direction(Vh, z, torch.randn(5, 20))
        torch.testing.assert_close((J @ u[..., None]).abs().amax(), torch.tensor(0.0), atol=1e-5, rtol=0)
        torch.testing.assert_close((u * z).sum(-1), torch.zeros(5), atol=1e-5, rtol=0)
        torch.testing.assert_close(u.norm(dim=-1), torch.ones(5))


class VariantsTest(unittest.TestCase):
    def test_layout(self):
        var = variants([0.9, 0.8])
        self.assertEqual(var[:2], [("base", 1.0), ("rerun", 1.0)])
        self.assertEqual(len(var), 2 + 2 * len(DIRECTIONS))
        self.assertEqual(var[2:4], [(DIRECTIONS[0], 0.9), (DIRECTIONS[0], 0.8)])


class EquivalentCosTest(unittest.TestCase):
    def test_interpolates_in_one_minus_cos(self):
        self.assertAlmostEqual(equivalent_cos([0.9, 0.8], [0.02, 0.06], 0.04), 0.85)
        self.assertAlmostEqual(equivalent_cos([0.9, 0.8], [0.08, 0.1], 0.04), 0.95)

    def test_unreached(self):
        self.assertTrue(math.isnan(equivalent_cos([0.9, 0.8], [0.01, 0.02], 0.04)))


if __name__ == "__main__":
    unittest.main()
