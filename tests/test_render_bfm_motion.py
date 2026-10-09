import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.render_bfm_motion import motion_qpos, output_paths, select_names


class RenderMotionTests(unittest.TestCase):
    def test_qpos_order_normalization_and_frame_rate(self):
        motion = {
            "root_trans_offset": np.array([[1, 2, 3]]),
            "root_rot": np.array([[0, 0, 0.6, 0.8]]) * 2,
            "dof": np.arange(29)[None, :],
            "fps": 30,
        }
        qpos, fps = motion_qpos(motion)
        np.testing.assert_allclose(qpos[0, :7], [1, 2, 3, 0.8, 0, 0, 0.6])
        np.testing.assert_array_equal(qpos[0, 7:], np.arange(29))
        self.assertEqual((qpos.shape, fps), ((1, 36), 30))
        motion["root_rot"][:] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            motion_qpos(motion)

    def test_selection_matches_dataset_order(self):
        motions = {"z_clip0": {}, "a_clip0": {}}
        self.assertEqual(select_names(motions), ["z_clip0"])
        self.assertEqual(select_names(motions, index=1), ["a_clip0"])
        self.assertEqual(select_names(motions, motion="a_clip0"), ["a_clip0"])
        self.assertEqual(select_names(motions, all_motions=True), list(motions))
        for index in (-1, 2):
            with self.assertRaises(ValueError):
                select_names(motions, index=index)
        with self.assertRaises(ValueError):
            select_names(motions, motion="missing")

    def test_output_validation_prevents_clobber_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "x.mp4"
            existing.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                output_paths(["x"], "input.pkl", output_dir=directory)
            self.assertEqual(output_paths(["x"], "input.pkl", output_dir=directory, overwrite=True), [existing])
            with self.assertRaises(ValueError):
                output_paths(["../x"], "input.pkl", output_dir=directory)
            with self.assertRaises(ValueError):
                output_paths(["x", "y"], "input.pkl", output=existing)


if __name__ == "__main__":
    unittest.main()
