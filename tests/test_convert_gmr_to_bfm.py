import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts import convert_gmr_to_bfm as converter


def make_motion(frames=601, fps=60.0):
    times = np.arange(frames) / fps
    return {
        "root_pos": np.column_stack((times, 2 * times, 1 + times / 10)),
        "root_rot": Rotation.from_euler("z", times / 10).as_quat(),
        "dof_pos": times[:, None] * np.linspace(0, 0.1, 29)[None, :],
        "fps": fps,
    }


def toy_model(reverse=False, negative_axis=False):
    indices = list(range(29))
    if reverse:
        indices.reverse()
    children = "".join(
        f'<body name="link{i}"><joint name="joint{i}" axis="{("-1" if negative_axis else "1")} 0 0"/>'
        '<geom type="sphere" size="0.1"/></body>'
        for i in indices
    )
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size="0.1"/>{children}</body></worldbody></mujoco>'
    )


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(converter.DEFAULT_TARGET_MJCF))
        cls.indices, cls.axes = converter.joint_mapping(cls.model, cls.model)

    def test_name_mapping_reorders_source_columns(self):
        indices, axes = converter.joint_mapping(toy_model(reverse=True), toy_model())
        np.testing.assert_array_equal(indices, np.arange(28, -1, -1))
        motion = make_motion()
        motion["dof_pos"][:] = np.arange(29)
        clips, _ = converter.convert_motion("order", motion, indices, axes)
        np.testing.assert_array_equal(clips["order_clip0"]["dof"][0], np.arange(28, -1, -1))

    def test_axis_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "axes differ"):
            converter.joint_mapping(toy_model(negative_axis=True), toy_model())

    def test_sixty_fps_roundoff_and_schema(self):
        motion = make_motion(fps=59.9999999988)
        clips, report = converter.convert_motion("sample", motion, self.indices, self.axes)
        self.assertEqual((report.frames, report.clips, report.dropped), (301, 1, 1))
        self.assertEqual(list(clips), ["sample_clip0"])
        clip = clips["sample_clip0"]
        np.testing.assert_allclose(clip["root_trans_offset"], motion["root_pos"][0:600:2], atol=1e-6)
        np.testing.assert_allclose(clip["dof"], motion["dof_pos"][0:600:2], atol=1e-7)
        converter.validate_dataset(clips, self.axes)
        self.assertTrue(all(clip[k].dtype == np.float32 for k in converter.ARRAY_SHAPES))
        self.assertEqual(clip["motion_name"], "sample")

    def test_slerp_handles_quaternion_sign_and_normalization(self):
        motion = make_motion(frames=2, fps=15)
        motion["root_rot"] = Rotation.from_euler("z", [0, 90], degrees=True).as_quat() * 3
        motion["root_rot"][1] *= -1
        sampled = converter.resample_motion(motion)
        expected = Rotation.from_euler("z", [0, 45, 90], degrees=True).as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(sampled["root_rot"]).as_matrix(), expected, atol=1e-12)

    def test_noninteger_fps_does_not_extrapolate(self):
        motion = make_motion(frames=601, fps=59.94)
        sampled = converter.resample_motion(motion)
        self.assertEqual(len(sampled["root_pos"]), 301)
        np.testing.assert_allclose(sampled["root_pos"][:, 0], np.arange(301) / 30, atol=1e-12)
        self.assertLessEqual(sampled["root_pos"][-1, 0], motion["root_pos"][-1, 0])

    def test_clip_boundaries_and_short_motions(self):
        for frames, expected in [(1, (0, 1)), (299, (0, 299)), (300, (1, 0)), (599, (1, 299)), (600, (2, 0)), (601, (2, 1))]:
            with self.subTest(frames=frames):
                clips, report = converter.convert_motion("x", make_motion(frames, 30), self.indices, self.axes)
                self.assertEqual((report.clips, report.dropped), expected)
                self.assertEqual(len(clips), expected[0])
                if expected[0] == 2:
                    self.assertEqual(clips["x_clip1"]["root_trans_offset"][0, 0], 10)

    def test_invalid_inputs(self):
        cases = []
        for fps in [0, -1, np.nan, np.inf, "60", True, None]:
            motion = make_motion()
            motion["fps"] = fps
            cases.append(motion)
        for key, value in [
            ("root_rot", np.zeros((601, 4))),
            ("dof_pos", np.zeros((601, 28))),
            ("root_pos", np.zeros((600, 3))),
            ("root_pos", np.full((601, 3), np.nan)),
        ]:
            motion = make_motion()
            motion[key] = value
            cases.append(motion)
        for motion in cases:
            with self.subTest(fps=motion["fps"]), self.assertRaises(ValueError):
                converter.resample_motion(motion)

    def test_alignment_preserves_motion_and_shares_offset_across_clips(self):
        # Analytic sphere geometry: minimum root z is 2, radius is 0.1.
        model = toy_model()
        indices, axes = converter.joint_mapping(model, model)
        aligner = converter.GroundAligner(model)
        motion = make_motion(601, 30)
        motion["root_pos"][:, 2] = 2 + np.linspace(0, 1, 601)
        original = copy.deepcopy(motion)
        clips, report = converter.convert_motion("x", motion, indices, axes, aligner, ground_offset=0.02)
        self.assertAlmostEqual(report.height_shift, -1.88)
        for i in range(2):
            sl = slice(i * 300, (i + 1) * 300)
            clip = clips[f"x_clip{i}"]
            np.testing.assert_allclose(clip["root_trans_offset"][:, 2] - original["root_pos"][sl, 2], -1.88, atol=1e-7)
            np.testing.assert_allclose(clip["root_trans_offset"][:, :2], original["root_pos"][sl, :2], atol=1e-6)
        first = clips["x_clip0"]
        self.assertAlmostEqual(aligner.minimum_height(first["root_trans_offset"], first["root_rot"], first["dof"]), 0.02, places=6)
        for key in ("root_pos", "root_rot", "dof_pos"):
            np.testing.assert_array_equal(motion[key], original[key])

    def test_collision_minimum_excludes_floor_and_visual_geometry(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom type="plane" size="1 1 0.1" pos="0 0 -100"/>
          <body><freejoint/>
            <geom type="sphere" size="0.2"/>
            <geom type="cylinder" size="0.3 0.5" quat="0.70710678 0 0.70710678 0"/>
            <geom type="sphere" size="1" pos="0 0 -50" contype="0" conaffinity="0"/>
          </body></worldbody></mujoco>""")
        aligner = converter.GroundAligner(model)
        self.assertAlmostEqual(aligner.minimum_height([[0, 0, 2]], np.array([[0, 0, 0, 1]]), np.empty((1, 0))), 1.7)

    def test_target_mesh_minimum_is_translation_equivariant(self):
        aligner = converter.GroundAligner(self.model)
        motion = make_motion(2)
        low = aligner.minimum_height(motion["root_pos"], motion["root_rot"], motion["dof_pos"])
        motion["root_pos"][:, 2] += 0.123
        self.assertAlmostEqual(aligner.minimum_height(motion["root_pos"], motion["root_rot"], motion["dof_pos"]) - low, 0.123)

    def test_joblib_roundtrip_overwrite_and_failure_cleanup(self):
        clips, _ = converter.convert_motion("x", make_motion(), self.indices, self.axes)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "converted.pkl"
            converter.write_dataset(clips, output, self.axes)
            converter.validate_dataset(joblib.load(output), self.axes)
            with self.assertRaises(FileExistsError):
                converter.write_dataset(clips, output, self.axes)
            before = output.read_bytes()
            with patch.object(converter.joblib, "load", side_effect=ValueError("readback failed")):
                with self.assertRaisesRegex(ValueError, "readback failed"):
                    converter.write_dataset(clips, output, self.axes, overwrite=True)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [output])
            converter.write_dataset(clips, output, self.axes, overwrite=True)

    def test_directory_conversion_and_no_valid_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.mkdir()
            output = Path(directory) / "out.pkl"
            for name in ("z", "a"):
                with (source / f"{name}.pkl").open("wb") as stream:
                    pickle.dump(make_motion(), stream)
            reports = converter.convert_directory(source, output, converter.DEFAULT_TARGET_MJCF, height_mode="none")
            self.assertEqual([r.name for r in reports], ["a", "z"])
            self.assertEqual(list(joblib.load(output)), ["a_clip0", "z_clip0"])
            with self.assertRaises(FileExistsError):
                converter.convert_directory(source, output)
            with self.assertRaisesRegex(ValueError, "outside"):
                converter.convert_directory(source, source / "bad.pkl")
            for path in source.glob("*.pkl"):
                with path.open("wb") as stream:
                    pickle.dump(make_motion(1), stream)
            with self.assertRaisesRegex(ValueError, "No complete"):
                converter.convert_directory(source, output, converter.DEFAULT_TARGET_MJCF, height_mode="none", overwrite=True)
            self.assertEqual(list(joblib.load(output)), ["a_clip0", "z_clip0"])


if __name__ == "__main__":
    unittest.main()
