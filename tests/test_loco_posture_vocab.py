"""Geometry, quota, provenance and CEM contracts; no Isaac app required."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import mujoco
import numpy as np
import pandas as pd
import torch

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase
from humanoidverse.envs.g1_env_helper.queries import METRICS
from humanoidverse.query.motion_query import Source
from vocab.loco_posture_vocab.loco_posture_vocab import (
    AugmentedFrames,
    compensate_half_kneel,
    deduplicate_posture_latents,
    fill_candidates,
    fill_parent_indices,
    goal_latents,
    goal_modes,
    ordered_fill_tasks,
    zero_postures,
)
from vocab.loco_posture_vocab.posture_cem import Optimization, objective, should_optimize
from vocab.loco_posture_vocab.posture_vocab import (
    BUCKETS,
    CollisionGeometry,
    Postures,
    annotate,
    cell_keys,
    coverage_deficits,
    locomotion,
    materialize,
    neighbour_order,
    prepare_candidates,
    quotas,
    select_frames,
    target_cells,
    translate_to_ground,
    with_wrist_preference,
)
from vocab.loco_posture_vocab.scan_cache import identity, load_scan


def frame(h=0.35, knees=(0.06, 0.07), tilt=20, pitch=10, wrists=None, feet=(0.05, 0.06), i=0):
    if wrists is None:
        wrists = (h, h + .02)
    row = dict.fromkeys(METRICS, 0.0)
    row |= dict(
        frame=i,
        origin=f"test:{i}",
        base_height=h,
        torso_height=h + 0.25,
        torso_tilt=tilt,
        torso_pitch=pitch,
        left_knee_height=knees[0],
        right_knee_height=knees[1],
        left_wrist_height=wrists[0],
        right_wrist_height=wrists[1],
        left_foot_height=feet[0],
        right_foot_height=feet[1],
        base_vx=0.3,
        collision_min_height=0.01,
        candidate_stage="grounded",
        height_shift=0,
        symmetry_error=0.0,
    )
    row.update({f"qpos_{j}": 0.0 for j in range(36)})
    row["qpos_2"], row["qpos_3"], row["qpos_7"] = h, 1.0, 0.03 * i
    return row


def postures():
    return pd.DataFrame(
        [
            frame(h=0.15, knees=(0.25, 0.35), tilt=85, pitch=-60, wrists=(0.1, 0.2), i=0),
            frame(h=0.15, knees=(0.10, 0.20), tilt=85, pitch=60, wrists=(0.1, 0.2), i=1),
            frame(i=2),
            frame(knees=(0.10, 0.30), i=3),
            frame(h=0.55, knees=(0.22, 0.30), tilt=30, i=4),
            frame(h=0.55, knees=(0.06, 0.30), tilt=30, i=5),
            frame(h=0.75, knees=(0.33, 0.35), tilt=15, wrists=(0.8, 0.82), i=6),
        ]
    )


class FakeFrames:
    num_frames = 100

    def gather_frames(self, idx):
        assert np.all(np.diff(idx) > 0), "HDF5 requires sorted unique indices"
        n = len(idx)
        q = np.zeros((n, 36), np.float32)
        q[:, 2], q[:, 3] = 0.7, 1
        return dict(
            qpos=q,
            qvel=np.ones((n, 35), np.float32),
            z=np.full((n, 4), 99, np.float32),
            next_obs=dict(
                state=np.ones((n, 64), np.float32),
                privileged_state=np.ones((n, 463), np.float32),
                last_action=np.zeros((n, 29), np.float32),
            ),
        )


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Postures()

    def test_both_wrists_reach_band_floor_without_band_ceiling_even_when_relaxed(self):
        examples = postures().iloc[[0, 2, 4, 6]].reset_index(drop=True)
        for i in range(4):
            lo, hi = self.cfg.height_edges[i:i+2]
            for side in ('left', 'right'):
                row = examples.iloc[[i]].copy()
                row[f'{side}_wrist_height'] = lo - .01
                for slack in (0., .2):
                    self.assertTrue(annotate(row, self.cfg, slack).bucket.isna().all())
                row[f'{side}_wrist_height'] = hi + .01
                self.assertEqual(annotate(row, self.cfg).bucket.iloc[0], BUCKETS[i])
            row = examples.iloc[[i]].copy()
            row[['left_wrist_height', 'right_wrist_height']] = lo
            self.assertEqual(annotate(row, self.cfg).bucket.iloc[0], BUCKETS[i])
            row[['left_wrist_height', 'right_wrist_height']] = hi
            self.assertTrue(annotate(row, self.cfg).bucket.notna().iloc[0])

    def test_six_postures_and_rising_are_eligible(self):
        df = annotate(postures(), self.cfg)
        self.assertEqual(df.pose_kind.tolist(), ["supine", "prone", "kneel", "half_kneel", "crouch", "rising", "stand"])
        self.assertTrue(df.bucket.notna().all())

    def test_combo_rejects_same_height_wrong_pose(self):
        data = [
            frame(h=0.55, knees=(0.06, 0.07)),
            frame(feet=(0.05, 0.7)),
            frame(wrists=(1.2, 1.3)),
            frame(h=0.15, tilt=5, wrists=(0.1, 0.2)),
        ]
        self.assertTrue(annotate(pd.DataFrame(data), self.cfg).bucket.isna().all())

    def test_kneeling_support_planes_and_half_kneeling_order(self):
        rows = pd.DataFrame([
            frame(knees=(.06,.07), feet=(.035,.04)),
            frame(knees=(.10,.30), feet=(.035,.04)),
            frame(knees=(.30,.10), feet=(.04,.035)),
        ])
        self.assertEqual(annotate(rows,self.cfg).pose_kind.tolist(),['kneel','half_kneel','half_kneel'])

    def test_deep_squats_and_support_transitions_stay_excluded_when_relaxed(self):
        rows = pd.DataFrame([
            frame(knees=(.13,.14),feet=(.035,.04)),  # symmetric deep squat; old <=.18 rule accepted it
            frame(knees=(.17,.29),feet=(.035,.04)),  # asymmetric squat; low knee too far from ankles
            frame(knees=(.04,.30),feet=(.035,.04)),  # low knee is not above both ankles
            frame(knees=(.10,.12),feet=(.035,.04)),  # transitional knee separation
            frame(knees=(.10,.30),feet=(.02,.10)),  # ankles not on the same plane
            frame(knees=(.16,.17),feet=(.14,.15)),  # four elevated joints are not floor support
            frame(knees=(.128,.110),feet=(.092,.093)),  # visually rejected hand-supported crouch
            frame(knees=(.394,.136),feet=(.082,.091)),  # ordering holds but the support knee is airborne
            frame(h=.258,knees=(.330,.106),feet=(.054,.034)),  # seated getting-up transition
            frame(h=.295,knees=(.329,.104),feet=(.055,.038)),  # seated side support: high knee above pelvis
            frame(h=.281,knees=(.114,.230),feet=(.082,.081)),  # hand-supported transition, rear knee still high
        ])
        for slack in [0.,self.cfg.relax]:
            with self.subTest(slack=slack):
                self.assertTrue(annotate(rows,self.cfg,slack).bucket.isna().all())

    def test_transition_velocity_arms_and_side_swap(self):
        rows = postures()
        rows["base_vx"], rows["base_vy"], rows["yaw_rate"] = 3.0, -2.0, 4.0
        rows.loc[2, "right_wrist_height"] = 0.44  # asymmetric wrists, both inside the kneeling band
        original = annotate(rows, self.cfg)
        for part in ("wrist", "foot", "knee"):
            rows[[f"left_{part}_height", f"right_{part}_height"]] = rows[[f"right_{part}_height", f"left_{part}_height"]].to_numpy()
        swapped = annotate(rows, self.cfg)
        self.assertEqual(original.bucket.tolist(), swapped.bucket.tolist())
        self.assertEqual(original.pose_kind.tolist(), swapped.pose_kind.tolist())
        self.assertTrue(original.bucket.notna().all())

    def test_height_boundaries_do_not_overlap(self):
        rows = [frame(h=0.25), frame(h=0.45, knees=(0.2, 0.3)), frame(h=0.65, knees=(0.3, 0.35)), frame(h=1.0, knees=(0.3, 0.35), wrists=(1.,1.))]
        self.assertEqual(annotate(pd.DataFrame(rows), self.cfg).bucket.tolist(), ["11_kneel", "12_crouch", "13_stand", "13_stand"])
        with self.assertRaises(ValueError):
            Postures(height_edges=(0, 0.25, 0.25, 0.65, 1)).validate()

    def test_locomotion_excludes_stand_and_tilted_frames(self):
        rows = pd.DataFrame([frame(h=0.75, i=i) for i in range(6)])
        rows["origin"] = ["walk1_subject1:0", "walk2_subject1:0", "run1_subject1:0",
                          "run2_subject1:0", "dance1_subject1:0", "sidewalk1_subject1:0"]
        rows.loc[1, "torso_tilt"] = 60
        rows.loc[2, "base_vx"] = 0
        self.assertEqual(locomotion(rows).frame.tolist(), [0, 3])

    def test_ground_translation_preserves_orientation_and_relative_geometry(self):
        row = postures().iloc[[2]].copy()
        row["collision_min_height"] = 0.4
        translated = translate_to_ground(row, self.cfg)
        self.assertAlmostEqual(translated.base_height.iloc[0], row.base_height.iloc[0] - 0.39)
        self.assertAlmostEqual(
            translated.left_wrist_height.iloc[0] - translated.base_height.iloc[0], row.left_wrist_height.iloc[0] - row.base_height.iloc[0]
        )
        self.assertAlmostEqual(translated.qpos_2.iloc[0], row.qpos_2.iloc[0] - 0.39)
        self.assertEqual(translated.torso_tilt.iloc[0], row.torso_tilt.iloc[0])

    def test_airborne_alignment_is_only_a_fallback(self):
        data = pd.concat([postures(), postures().iloc[[2]]], ignore_index=True)
        data.loc[7, "frame"] = 7
        data.loc[7, "base_height"] += 0.5
        data.loc[7, "qpos_2"] += 0.5
        data.loc[7, "collision_min_height"] = 0.51
        for c in [c for c in data if c.endswith("_height") and c not in ("base_height", "collision_min_height")]:
            data.loc[7, c] += 0.5
        with (
            patch("vocab.loco_posture_vocab.posture_vocab.collision_heights", return_value=np.array([0.01] * 7 + [0.51])),
            patch("vocab.loco_posture_vocab.posture_vocab.coverage_deficits", return_value=([], {})),
        ):
            _, result, _, report = prepare_candidates(data, self.cfg, 6)
        self.assertEqual(report["aligned_candidates"], 0)
        self.assertFalse((result.frame == 7).any())
        with patch("vocab.loco_posture_vocab.posture_vocab.collision_heights", return_value=np.array([0.01] * 7 + [0.51])):
            _, result, _, report = prepare_candidates(data, self.cfg, 30)
        self.assertEqual(report["aligned_candidates"], 1)
        self.assertAlmostEqual(result.loc[result.frame == 7, "height_shift"].iloc[0], -0.5)

    def test_half_kneel_hole_does_not_admit_unrelated_airborne_kneel(self):
        data = pd.concat([postures(), postures().iloc[[3, 2]]], ignore_index=True)
        data['frame'] = np.arange(len(data))
        data.loc[3, ['base_height', 'qpos_2']] = .325
        data.loc[7, ['base_height', 'qpos_2']] = .425
        for name in ['qpos_2'] + [c for c in data if c.endswith('_height') and c != 'collision_min_height']:
            data.loc[8, name] += .5
        gaps = np.array([.01] * 8 + [.51])
        data['collision_min_height'] = gaps
        with patch('vocab.loco_posture_vocab.posture_vocab.collision_heights', return_value=gaps):
            _, result, _, report = prepare_candidates(data, self.cfg, 6)
        self.assertIn(['11_kneel', 'half_kneel', 7], report['missing_cells'])
        self.assertEqual(report['aligned_candidates'], 0)
        self.assertNotIn(8, result.frame.to_numpy())
        self.assertTrue((result.candidate_stage == 'grounded').all())

    def test_collision_support_ignores_floor_and_visual_geoms(self):
        for kind, size, expected in [
            ("sphere", ".1", 0.9),
            ("box", ".1 .2 .3", 0.7),
            ("capsule", ".1 .2", 0.7),
            ("cylinder", ".1 .2", 0.8),
            ("ellipsoid", ".1 .2 .3", 0.7),
        ]:
            model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><geom type="plane" size="0 0 .1"/>
              <body pos="0 0 1"><geom type="{kind}" size="{size}"/><geom type="sphere" size="10" contype="0" conaffinity="0"/>
              </body></worldbody></mujoco>''')
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(CollisionGeometry(model).minimum(data), expected, places=6)

    def test_rotated_collision_mesh_uses_surface_not_origin(self):
        vertices = " ".join(f"{x} {y} {z}" for x in [-0.1, 0.1] for y in [-0.2, 0.2] for z in [-0.3, 0.3])
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="box" vertex="{vertices}"/></asset>
            <worldbody><body pos="0 0 1" quat=".9238795325 .3826834324 0 0"><geom type="mesh" mesh="box"/>
            </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(CollisionGeometry(model).minimum(data), 1 - (0.2 + 0.3) / np.sqrt(2), places=6)


class SamplingTests(unittest.TestCase):
    def test_preference_uses_the_closer_wrist_without_symmetry(self):
        data = pd.DataFrame(dict(base_height=[.35, .35, .35], left_wrist_height=[.36, .65, .35],
                                 right_wrist_height=[1., .65, .35]))
        result = with_wrist_preference(data)
        np.testing.assert_allclose(result.wrist_pelvis_distance, [.01, .3, 0.])
        self.assertNotIn('symmetry_error', result)

    def test_exact_quota_with_short_band(self):
        balanced = quotas(BUCKETS, [2000] * 4, 2048, dict(zip(BUCKETS, [2, 2, 1, 1])))
        self.assertEqual(list(balanced.values()), [512] * 4)
        counts = quotas(BUCKETS, [5, 1300, 800, 900], 2048, dict(zip(BUCKETS, [2, 2, 1, 1])))
        self.assertEqual(sum(counts.values()), 2048)
        self.assertEqual(counts[BUCKETS[0]], 5)
        with self.assertRaisesRegex(ValueError, "Cannot allocate"):
            quotas(BUCKETS, [1, 1, 1, 1], 2048)

    def test_half_loco_and_six_kinds(self):
        cfg = Postures()
        base = pd.concat([postures() for _ in range(5)], ignore_index=True)
        base["frame"] = np.arange(len(base))
        base["qpos_7"] = np.arange(len(base)) * 0.025
        post = annotate(base, cfg)
        loco = pd.DataFrame([frame(h=0.75, i=i + 50) for i in range(20)])
        with patch.dict("os.environ", {"MOTION_WORKERS": "1"}):
            selected = select_frames(loco, post, 24, cfg)
        self.assertEqual(len(selected), 24)
        self.assertEqual(int((selected.grp == "loco").sum()), 12)
        self.assertTrue(set(["supine", "prone", "kneel", "half_kneel", "crouch", "stand"]) <= set(selected.pose_kind))

    def test_holes_are_inside_observed_support(self):
        cfg = Postures()
        rows = postures()
        rows.loc[0, ["base_height", "qpos_2"]] = 0.05
        another = rows.iloc[[0]].copy()
        another[["base_height", "qpos_2"]] = 0.15
        post = annotate(pd.concat([rows, another], ignore_index=True), cfg)
        targets = target_cells(post, cfg)
        self.assertIn(("10_lie", "supine", 2), targets)
        self.assertNotIn(("13_stand", "stand", 19), targets)
        tasks, _ = coverage_deficits(post, targets, cfg, 20)
        self.assertIn(("10_lie", "supine", 2), tasks)
        tasks = ordered_fill_tasks(tasks, post, 4)
        self.assertLessEqual(len(tasks), 4)

    def test_missing_kind_central_cell_matches_boundary_rounding(self):
        cfg = Postures()
        post = annotate(postures().drop(index=3), cfg)
        self.assertIn(('11_kneel', 'half_kneel', 7), target_cells(post, cfg))
        self.assertNotIn(('11_kneel', 'half_kneel', 6), target_cells(post, cfg))

    def test_unique_hdf_gather_and_consistent_observation_shift(self):
        selected = annotate(postures().iloc[[2, 2]].copy(), Postures())
        selected["height_shift"] = [0.0, -0.3]
        db = materialize(AugmentedFrames(FakeFrames()), selected)
        self.assertEqual(len(db), 2)
        self.assertAlmostEqual(db.frames["qpos"][1, 2], 0.4, places=6)
        self.assertAlmostEqual(db.frames["next_obs"]["privileged_state"][1, 0], 0.7, places=6)
        zero_postures(db, np.array([1]))
        self.assertTrue((db.frames["qvel"][1] == 0).all())
        self.assertTrue((db.frames["qvel"][0] == 1).all())
        self.assertTrue((db.frames["next_obs"]["state"][1, 29:58] == 0).all())


class LatentTests(unittest.TestCase):
    def test_generated_cell_uses_reached_state_and_keeps_parent(self):
        cfg, opt = Postures(), Optimization(max_fill=1, fill_attempts=1)
        post = annotate(postures(), cfg)
        target = ("13_stand", "stand", 15)
        frames = AugmentedFrames(FakeFrames())
        qpos = np.array([[post.iloc[6][f"qpos_{j}"] for j in range(36)]], dtype=np.float32)
        held = post.iloc[6]
        result = dict(
            valid=np.array([True]),
            matching=np.array([True]),
            stability_score=np.array([99.0]),
            z=np.full((1, 4), 7.0, np.float32),
            final_qpos=qpos,
            final_qvel=np.zeros((1, 35), np.float32),
            final_obs_state=np.ones((1, 64), np.float32),
            final_obs_privileged_state=np.ones((1, 463), np.float32),
            final_obs_last_action=np.zeros((1, 29), np.float32),
        )
        report = {}
        with (
            patch("vocab.loco_posture_vocab.loco_posture_vocab.coverage_deficits", return_value=([target], {})),
            patch("vocab.loco_posture_vocab.loco_posture_vocab.goal_latents", return_value=np.ones((1, 4))),
            patch("vocab.loco_posture_vocab.loco_posture_vocab.evaluate", return_value=result),
            patch("vocab.loco_posture_vocab.loco_posture_vocab.compute_metrics", return_value={k: np.array([held[k]]) for k in METRICS}),
            patch("vocab.loco_posture_vocab.loco_posture_vocab.collision_heights", return_value=np.array([0.01])),
        ):
            augmented = fill_candidates(SimpleNamespace(), frames, post, [target], cfg, opt, 12, report)
        self.assertEqual(report["fill"]["accepted"], 1)
        generated = augmented.iloc[[-1]]
        self.assertEqual(cell_keys(generated, cfg)[0], target)
        self.assertEqual(generated.parent_frame.iloc[0], 6)
        self.assertEqual(generated.frame.iloc[0], 100)
        db = materialize(frames, generated)
        np.testing.assert_array_equal(db.frames["qpos"], qpos)
        np.testing.assert_array_equal(db.frames["z"], result["z"])
        self.assertEqual(db.frames["next_obs"]["state"].shape, (1, 64))

    def test_goals_bypass_kept_z_and_B(self):
        class Policy:
            cfg = SimpleNamespace(inference_batch_size=100)

            def goal_inference(self, obs):
                n = len(obs["state"])
                assert not torch.any(obs["state"][:, 29:58])
                return torch.full((n, 4), 7.0)

        frames = FakeFrames().gather_frames(np.array([1, 2]))
        frames["B"] = np.full((2, 4), 123.0)
        db = MotionDatabase({}, np.array([1, 2]), frames)
        zero_postures(db, np.array([1]))
        with patch("humanoidverse.pose_hold.policy.GoalPolicy", return_value=Policy()):
            z = goal_latents(SimpleNamespace(model_folder=Path("."), device="cpu"), db, np.array([1]))
        np.testing.assert_array_equal(z[0], [99] * 4)
        np.testing.assert_array_equal(z[1], [7] * 4)

    def test_cell_fill_can_change_parent_height_and_uses_numeric_cell_indices(self):
        cfg, opt = Postures(), Optimization()
        g = annotate(postures().iloc[[4]], cfg)
        result = dict(valid=np.array([True]), stability_score=np.array([97.0]), height_err=np.array([0.20]))
        cells = np.array(cell_keys(g, cfg), dtype=str)
        _, match, feasible = objective(result, g, np.array(["12_crouch"]), np.array(["crouch"]), cells, cfg, opt, filling=True)
        self.assertTrue(match[0])
        self.assertTrue(feasible[0])
        _, _, normal_feasible = objective(result, g, np.array(["12_crouch"]), np.array(["crouch"]), cells, cfg, opt)
        self.assertFalse(normal_feasible[0])
        result["hold_height"] = np.array([0.45])  # only the final state hit the target cell
        _, match, _ = objective(result, g, np.array(["12_crouch"]), np.array(["crouch"]), cells, cfg, opt, filling=True)
        self.assertFalse(match[0])

    def test_stability_is_soft_and_shape_is_separate(self):
        cfg, opt = Postures(), Optimization()
        g = annotate(postures().iloc[[2, 3]], cfg)
        result = dict(valid=np.array([True, True]), stability_score=np.array([94.0, 99.0]), height_err=np.array([0.0, 0.0]))
        key, match, feasible = objective(
            result, g, np.array(["11_kneel"] * 2), np.array(["kneel"] * 2), np.array([["11_kneel", "kneel", 0]] * 2), cfg, opt
        )
        self.assertGreater(key[0], key[1])
        self.assertTrue(match[0])
        self.assertFalse(feasible[0])
        self.assertTrue(should_optimize(result, match, opt).all())

    def test_stable_deep_squat_cannot_satisfy_genuine_half_kneeling(self):
        cfg,opt=Postures(),Optimization()
        g=annotate(pd.DataFrame([frame(knees=(.10,.30)),frame(knees=(.17,.29))]),cfg,cfg.relax)
        result=dict(valid=np.ones(2,bool),stability_score=np.array([97.,100.]),height_err=np.zeros(2))
        keys,matching,feasible=objective(result,g,np.array(['11_kneel']*2),np.array(['half_kneel']*2),
                                         np.array([['11_kneel','half_kneel',7]]*2),cfg,opt)
        np.testing.assert_array_equal(matching,[True,False])
        np.testing.assert_array_equal(feasible,[True,False])
        self.assertGreater(keys[0],keys[1])

    def test_inclination_is_not_a_cell_or_cem_target(self):
        cfg, opt = Postures(), Optimization()
        rows = pd.DataFrame([frame(h=0.55, knees=(0.2, 0.3), tilt=10), frame(h=0.55, knees=(0.2, 0.3), tilt=70)])
        g = annotate(rows, cfg)
        keys = cell_keys(g, cfg)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(len(keys[0]), 3)
        result = dict(
            valid=np.array([True, True]),
            stability_score=np.array([97.0, 97.0]),
            height_err=np.zeros(2),
            hold_height=np.full(2, 0.55),
            torso_tilt_deg=np.array([10.0, 70.0]),
        )
        key, matching, feasible = objective(
            result, g, np.array(["12_crouch"] * 2), np.array(["crouch"] * 2), np.array(keys, str), cfg, opt, filling=True
        )
        np.testing.assert_array_equal(key, [key[0]] * 2)
        self.assertTrue(matching.all() and feasible.all())


    def test_half_kneel_retry_only_borrows_kneel_even_when_crouch_is_closer(self):
        heights = np.array([.35, .31, .46])
        buckets = np.array(["11_kneel", "11_kneel", "12_crouch"])
        kinds = np.array(["half_kneel", "kneel", "crouch"])
        self.assertEqual(neighbour_order(heights, buckets, kinds, .32, "11_kneel", "half_kneel")[0], 0)
        self.assertEqual(neighbour_order(heights, buckets, kinds, .32, "11_kneel", "half_kneel", attempt=1)[0], 1)
        self.assertEqual(neighbour_order(heights, buckets, kinds, .44, "11_kneel", "half_kneel", attempt=1)[0], 1)
        self.assertNotIn(2, neighbour_order(heights, buckets, kinds, .44, "11_kneel", "half_kneel"))
        self.assertEqual(len(neighbour_order(heights[2:], buckets[2:], kinds[2:], .44,
                                             "11_kneel", "half_kneel", attempt=1)), 0)
        g = annotate(pd.DataFrame([frame(h=.32)]), Postures())  # two grounded knees
        result = dict(valid=np.array([True]), stability_score=np.array([99.]), height_err=np.array([0.]))
        _, matching, feasible = objective(result, g, np.array(["11_kneel"]), np.array(["half_kneel"]),
                                          np.array([["11_kneel", "half_kneel", 6]]), Postures(), Optimization())
        self.assertFalse(matching[0])
        self.assertFalse(feasible[0])

    def test_multiple_fill_seeds_prioritize_own_half_then_only_double_kneel(self):
        post = annotate(postures(), Postures())
        target = ('11_kneel', 'half_kneel', 6)
        self.assertEqual(fill_parent_indices(post, target, Postures(), 0, 8), [3])
        self.assertEqual(fill_parent_indices(post, target, Postures(), 1, 8), [2])
        crouch_only = post.loc[post.pose_kind == 'crouch']
        self.assertEqual(fill_parent_indices(crouch_only, target, Postures(), 1, 8), [])

    def test_related_compensation_requires_height_and_stability_without_pose_filters(self):
        cfg, opt = Postures(), Optimization()
        # Knees violate half kneeling, but the wrist-height lower bound is mandatory.
        g = annotate(pd.DataFrame([frame(h=.325, knees=(.23, .24), wrists=(.3, .4))]), cfg)
        result = dict(valid=np.array([True]), stability_score=np.array([99.]),
                      height_err=np.array([0.]), hold_height=np.array([.325]))
        args = (result, g, np.array(["11_kneel"]), np.array(["half_kneel"]),
                np.array([["11_kneel", "half_kneel", 6]]), cfg, opt)
        self.assertFalse(objective(*args, filling=True)[1][0])
        _, matching, feasible = objective(*args, filling=True, height_only=np.array([True]))
        self.assertTrue(matching[0] and feasible[0])
        g.loc[0, 'w_max'] = .45
        self.assertTrue(objective(*args, filling=True, height_only=np.array([True]))[1][0])
        g.loc[0, 'w_min'] = .24
        self.assertFalse(objective(*args, filling=True, height_only=np.array([True]))[1][0])
        g.loc[0, 'w_min'] = .3
        g.loc[0, 'w_max'] = .4
        result['hold_height'][:] = .37
        self.assertFalse(objective(*args, filling=True, height_only=np.array([True]))[1][0])
        result['hold_height'][:] = .325
        result['stability_score'][:] = 94
        self.assertFalse(objective(*args, filling=True, height_only=np.array([True]))[2][0])

    def test_compensation_mode_survives_selection_for_final_evaluation(self):
        rows = pd.DataFrame(dict(pose_kind=['half_kneel', 'half_kneel', 'kneel'],
                                 height_only=[np.nan, True, False],
                                 compensation_seed_kind=[None, 'kneel', '']))
        modes = goal_modes(rows)
        np.testing.assert_array_equal(modes['height_only'], [False, True, False])
        np.testing.assert_array_equal(modes['seed_kinds'], ['half_kneel', 'kneel', 'kneel'])
        rows.loc[1, 'compensation_seed_kind'] = 'crouch'
        with self.assertRaisesRegex(ValueError, 'kneeling seed'):
            goal_modes(rows)

    def test_scarce_half_kneel_quota_transfers_to_double_kneel(self):
        self.assertEqual(quotas(['kneel', 'half_kneel'], [1013, 11], 512,
                                {'kneel': 1, 'half_kneel': 1}), {'kneel': 501, 'half_kneel': 11})

    def test_compensation_uses_reached_goal_observations_and_preserves_true_half_kneel(self):
        cfg, opt = Postures(), Optimization(fill_attempts=1)
        selected = annotate(postures().iloc[[3, 2]].reset_index(drop=True), cfg)
        db = MotionDatabase({}, np.array([3, 2]), FakeFrames().gather_frames(np.array([2, 3])))
        q = np.array([[selected.iloc[i][f'qpos_{j}'] for j in range(36)] for i in range(2)], dtype=np.float32)
        result = dict(valid=np.ones(2, bool), stability_score=np.array([80.,99.]), height_err=np.zeros(2),
                      held_pose_kind=np.array(['half_kneel','kneel']), held_bucket=np.array(['11_kneel']*2),
                      hold_height=np.array([.35,.35]), matching=np.ones(2,bool), height_only=np.zeros(2,bool),
                      seed_kind=np.array(['half_kneel','kneel']), z=np.ones((2,4),np.float32), final_qpos=q,
                      optimized=np.zeros(2,bool), attempted=np.ones(2,bool), cem_iterations=np.ones(2,int),
                      seed_restarts=np.zeros(2,int), stability_initial=np.array([70.,99.]), height_err_initial=np.zeros(2))
        batch = {k:v[1:].copy() for k,v in result.items()}
        batch['height_only'][:] = True
        batch['z'][:] = 2
        batch['final_obs_state'] = np.full((1,64),7.,np.float32)
        batch['final_obs_privileged_state'] = np.full((1,463),8.,np.float32)
        held = selected.iloc[1]
        report = {}
        with (patch('vocab.loco_posture_vocab.loco_posture_vocab.evaluate',return_value=batch) as evaluate_mock,
              patch('vocab.loco_posture_vocab.loco_posture_vocab.compute_metrics',return_value={k:np.array([held[k]]) for k in METRICS}),
              patch('vocab.loco_posture_vocab.loco_posture_vocab.collision_heights',return_value=np.array([.01]))):
            compensate_half_kneel(SimpleNamespace(),db,selected,np.arange(2),result,cfg,opt,100,report)
        self.assertEqual(report['half_kneel_compensation']['accepted'],1)
        self.assertEqual(selected.pose_kind.iloc[0],'half_kneel')  # nominal height slot
        self.assertEqual(result['held_pose_kind'][0],'kneel')
        self.assertTrue(result['height_only'][0])
        self.assertEqual(db.source_index[0],100)
        np.testing.assert_array_equal(db.frames['qpos'][0],q[1])
        self.assertTrue(np.all(db.frames['qvel'][0]==0))
        self.assertTrue(np.all(db.frames['next_obs']['state'][0,:29]==7))
        self.assertTrue(np.all(db.frames['next_obs']['state'][0,29:58]==0))
        self.assertTrue(np.all(db.frames['next_obs']['privileged_state'][0,-186:]==0))
        np.testing.assert_array_equal(evaluate_mock.call_args.kwargs['height_only'],[True])
        # A genuine half-kneel that already passes is never replaced by a related pose.
        result['held_pose_kind'][0]='half_kneel'
        with patch('vocab.loco_posture_vocab.loco_posture_vocab.evaluate') as evaluate_mock:
            compensate_half_kneel(SimpleNamespace(),db,selected,np.arange(2),result,cfg,opt,101,{})
        evaluate_mock.assert_not_called()
        self.assertFalse(result['height_only'][0])
        # A stable crouch cannot substitute for the missing double-kneeling donor.
        result['stability_score'][0] = 80.
        selected.loc[1, 'pose_kind'] = 'crouch'
        result['held_pose_kind'][1] = 'crouch'
        report = {}
        with patch('vocab.loco_posture_vocab.loco_posture_vocab.evaluate') as evaluate_mock:
            compensate_half_kneel(SimpleNamespace(), db, selected, np.arange(2), result, cfg, opt, 102, report)
        evaluate_mock.assert_not_called()
        self.assertEqual(report['half_kneel_compensation']['remaining'], 1)

    def test_distinct_replacements_keep_locomotion_and_initial_scores(self):
        z=np.array([[1.,1.,1.,1.],[1.,1.,1.,1.],[2.,2.,2.,2.]],np.float32)
        selected=pd.DataFrame(dict(bucket=['01_loco','13_stand','13_stand'],pose_kind=['loco','stand','stand']))
        result=dict(z=z[1:].copy(),height_only=np.zeros(2,bool),seed_kind=np.array(['stand']*2),
                    optimized=np.zeros(2,bool),valid=np.ones(2,bool),matching=np.ones(2,bool),
                    stability_score=np.array([99.,98.]),stability_initial=np.array([80.,81.]),height_err=np.zeros(2))
        db=MotionDatabase({},np.arange(3),FakeFrames().gather_frames(np.arange(3)))
        def probe(source,qpos,seeds,*args,**kwargs):
            batch={k:np.repeat(v[:1],len(seeds),axis=0) for k,v in result.items()}
            batch['z']=seeds
            return batch
        report={}
        with patch('vocab.loco_posture_vocab.loco_posture_vocab.evaluate',side_effect=probe):
            deduplicate_posture_latents(SimpleNamespace(),db,selected,np.array([1,2]),z,result,Postures(),Optimization(),report)
        np.testing.assert_array_equal(z[0],[1.]*4)
        self.assertFalse(np.array_equal(result['z'][0],z[0]))
        self.assertEqual(report['latent_duplicates']['after'],3)
        np.testing.assert_array_equal(result['stability_initial'],[80.,81.])


class ScanCacheTests(unittest.TestCase):
    def test_reuses_valid_scan_and_gathers_writable_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            expected = {"version": 1, "source": "test"}
            (folder / "manifest.json").write_text(json.dumps(expected))
            joblib.dump(FakeFrames().gather_frames(np.array([0, 1])), folder / "frames.pkl")
            joblib.dump(postures().iloc[:2], folder / "table.pkl")
            with (
                patch("vocab.loco_posture_vocab.scan_cache.identity", return_value=expected),
                patch("vocab.loco_posture_vocab.scan_cache.subprocess.run") as launch,
            ):
                frames, table = load_scan(Source(model_folder=folder, lafan="*"), folder)
            launch.assert_not_called()
            self.assertEqual(len(table), 2)
            gathered = frames.gather_frames(np.array([1, 0]))
            gathered["qpos"][0, 2] = 10
            self.assertAlmostEqual(frames.gather_frames(np.array([1]))["qpos"][0, 2], 0.7, places=6)

    def test_source_and_stride_invalidate_but_worker_count_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            motions = folder / "motions"
            motions.mkdir()
            (folder / "config.json").write_text("{}")
            motion = motions / "test.pkl"
            motion.write_bytes(b"first")
            source = Source(model_folder=folder, lafan="*", lafan_dir=motions)
            original = identity(source)
            self.assertEqual(identity(replace(source, max_workers=2, chunk=99)), original)
            self.assertNotEqual(identity(replace(source, stride=2)), original)
            motion.write_bytes(b"different bytes")
            self.assertNotEqual(identity(source), original)


if __name__ == "__main__":
    unittest.main()
