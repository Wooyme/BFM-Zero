import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from scripts.queries.rollout_vocab import default_qpos, load_latents


class RolloutVocabularyTests(unittest.TestCase):
    def test_default_pose_uses_config_joint_order_and_wxyz_rotation(self):
        config = {'robot': {'dof_names': ['second', 'first'], 'init_state': {
            'pos': [0., 0., .8], 'rot': [0., .6, 0., .8],
            'default_joint_angles': {'first': -.1, 'second': .3}}}}
        np.testing.assert_allclose(default_qpos(config), [0., 0., .8, .8, 0., .6, 0., .3, -.1])

    def test_token_order_matches_sorted_vocabulary_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'z.pkl'
            joblib.dump({'13_stand': np.full((1, 4), 3.), '01_loco': np.full((2, 4), 1.)}, path)
            z, names, codes, within = load_latents(path)
        self.assertEqual(names, ['01_loco', '13_stand'])
        np.testing.assert_array_equal(z[:, 0], [1., 1., 3.])
        np.testing.assert_array_equal(codes, [0, 0, 1])
        np.testing.assert_array_equal(within, [0, 1, 0])

    def test_invalid_latents_are_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'z.pkl'
            for groups in ({}, {'a': np.ones(4)}, {'a': np.full((1, 4), np.nan)},
                           {'a': np.ones((1, 4)), 'b': np.ones((1, 3))}):
                with self.subTest(groups=list(groups)):
                    joblib.dump(groups, path)
                    with self.assertRaises(ValueError):
                        load_latents(path)


if __name__ == '__main__':
    unittest.main()
