from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pyvista as pv
from spatialmath import SE3

from legged_gym.expert_complex_utils.se3_path_vis import (
    PathPlayback,
    PoseSeries,
    SE3PathVisualizer,
    classify_feasibility,
    load_se3_path_json,
    se3_de_casteljau,
    se3_interpolate,
)


def _pose_block(translations):
    return {
        "t": translations,
        "ang": [0.0] * len(translations),
        # Zero-angle axes are deliberately allowed to be zero.
        "vec": [[0.0, 0.0, 0.0]] * len(translations),
    }


def _valid_document():
    return {
        "schema_version": 1,
        "metadata": {"name": "unit-test"},
        "waypoints": _pose_block([[0, 0, 0], [1, 0, 0], [2, 1, 0]]),
        "dense_poses": {
            "time": [0.0, 0.5, 1.0, 2.0],
            **_pose_block([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 1, 0]]),
        },
        "bezier_segments": [
            {
                "degree": 3,
                "duration": 1.0,
                "control_poses": _pose_block(
                    [[0, 0, 0], [0.3, 0, 0], [0.7, 0, 0], [1, 0, 0]]
                ),
            },
            {
                "degree": 2,
                "duration": 1.0,
                "control_poses": _pose_block(
                    [[1, 0, 0], [1.5, 0.25, 0], [2, 1, 0]]
                ),
            },
        ],
    }


def _write_json(directory: Path, document):
    path = directory / "path.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class SE3PathDataTest(unittest.TestCase):
    def test_load_complete_path_and_sample_arbitrary_degree(self):
        with tempfile.TemporaryDirectory() as directory:
            data = load_se3_path_json(
                _write_json(Path(directory), _valid_document())
            )
        self.assertEqual(len(data.waypoints), 3)
        self.assertEqual([segment.degree for segment in data.bezier_segments], [3, 2])

        sampled = data.sample_bezier(0.3)
        self.assertAlmostEqual(sampled.time[0], 0.0)
        self.assertAlmostEqual(sampled.time[-1], 2.0)
        self.assertTrue(np.all(np.diff(sampled.time) > 0))
        np.testing.assert_allclose(sampled.poses[0].A, data.waypoints[0].A)
        np.testing.assert_allclose(sampled.poses[-1].A, data.waypoints[-1].A)

    def test_field_level_validation(self):
        cases = [
            (lambda doc: doc.pop("waypoints"), "missing field: waypoints"),
            (
                lambda doc: doc["dense_poses"]["ang"].pop(),
                r"dense_poses\.t, \.ang and \.vec must have equal lengths",
            ),
            (
                lambda doc: doc["dense_poses"].update(time=[0.0, 0.5, 0.4, 2.0]),
                "strictly increasing",
            ),
            (
                lambda doc: doc["bezier_segments"][0].update(degree=2),
                r"degree \+ 1",
            ),
            (
                lambda doc: (
                    doc["dense_poses"]["ang"].__setitem__(1, 0.2),
                    doc["dense_poses"]["vec"].__setitem__(1, [0, 0, 0]),
                ),
                "must be normalizable",
            ),
        ]
        for mutate, match in cases:
            with self.subTest(match=match), tempfile.TemporaryDirectory() as directory:
                document = copy.deepcopy(_valid_document())
                mutate(document)
                with self.assertRaisesRegex(ValueError, match):
                    load_se3_path_json(_write_json(Path(directory), document))

    def test_waypoint_endpoint_difference_only_warns(self):
        document = _valid_document()
        document["dense_poses"]["t"][0] = [0.01, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertWarnsRegex(UserWarning, "dense start"):
                load_se3_path_json(_write_json(Path(directory), document))

    def test_cubic_de_casteljau_matches_explicit_construction(self):
        controls = (
            SE3.Trans(0, 0, 0),
            SE3.Trans(0.4, 0.2, 0) * SE3.Rz(0.2),
            SE3.Trans(0.8, 0.1, 0.1) * SE3.Ry(-0.1),
            SE3.Trans(1, 0, 0) * SE3.Rz(0.4),
        )
        u = 0.37
        first = [se3_interpolate(controls[i], controls[i + 1], u) for i in range(3)]
        second = [se3_interpolate(first[i], first[i + 1], u) for i in range(2)]
        expected = se3_interpolate(second[0], second[1], u)
        actual = se3_de_casteljau(controls, u)
        np.testing.assert_allclose(actual.A, expected.A, atol=1e-12)
        np.testing.assert_allclose(se3_de_casteljau(controls, 0.0).A, controls[0].A)
        np.testing.assert_allclose(se3_de_casteljau(controls, 1.0).A, controls[-1].A)


class PlaybackAndFeasibilityTest(unittest.TestCase):
    def test_switch_step_and_end_stop(self):
        poses = (
            SE3.Trans(0, 0, 0), SE3.Trans(1, 0, 0), SE3.Trans(2, 0, 0)
        )
        dense = PoseSeries(np.array([0.0, 1.0, 2.0]), poses)
        bezier = PoseSeries(np.array([0.0, 2.0, 4.0]), poses)
        playback = PathPlayback(dense, bezier)

        playback.step(-1)
        self.assertEqual(playback.current_time, 0.0)
        playback.step(1)
        self.assertEqual(playback.current_time, 1.0)
        playback.switch_source()
        self.assertEqual(playback.source, "bezier")
        self.assertAlmostEqual(playback.current_time, 2.0)
        playback.step(1)
        playback.step(1)
        self.assertEqual(playback.current_time, 4.0)
        playback.playing = True
        playback.advance(1.0)
        self.assertEqual(playback.current_time, 4.0)
        self.assertFalse(playback.playing)

    def test_feasibility_levels(self):
        body_free = np.ones(4, dtype=bool)
        strong_legs = np.ones((6, 3, 2), dtype=bool)
        weak_legs = np.zeros((6, 3, 2), dtype=bool)
        weak_legs[:, 0, 0] = True

        self.assertEqual(classify_feasibility(body_free, strong_legs, 3).level, "strong")
        self.assertEqual(classify_feasibility(body_free, weak_legs, 3).level, "weak")
        self.assertEqual(
            classify_feasibility(np.array([True, False]), strong_legs, 3).level,
            "invalid",
        )

    def test_native_timer_processes_one_frame_per_event(self):
        class FakeInteractor:
            def __init__(self):
                self.callback = None

            def add_observer(self, event, callback):
                self.asserted_event = event
                self.callback = callback
                return 12

            def create_timer(self, duration, repeating=True):
                self.duration = duration
                self.repeating = repeating
                return 34

        fake_interactor = FakeInteractor()
        viewer = object.__new__(SE3PathVisualizer)
        viewer.plotter = type("FakePlotter", (), {"iren": fake_interactor})()
        viewer._timer_id = None
        viewer._timer_observer_id = None
        viewer._on_timer = Mock()

        viewer._install_timer(duration_ms=33)
        self.assertEqual(fake_interactor.asserted_event, "TimerEvent")
        self.assertEqual(fake_interactor.duration, 33)
        self.assertTrue(fake_interactor.repeating)
        self.assertEqual(viewer._timer_id, 34)
        self.assertEqual(viewer._timer_observer_id, 12)

        fake_interactor.callback(None, "TimerEvent")
        viewer._on_timer.assert_called_once_with(0)


class _MockVoxels:
    def __init__(self, center, esdf_flat=None):
        self.grid_shape = np.array([3, 3, 3], dtype=np.int32)
        self.center = center
        self.esdf_flat = esdf_flat

    def FlatIndex2GridIndex(self, flat_index):
        return np.column_stack(np.unravel_index(np.asarray(flat_index), self.grid_shape))

    def GridIndex2FlatIndex(self, grid_index):
        return np.ravel_multi_index(np.asarray(grid_index).T, self.grid_shape)


class _MockKinematic:
    @staticmethod
    def _B2R(points, _leg_index):
        return points


class _MockRobot:
    def __init__(self):
        grid = np.indices((3, 3, 3)).reshape(3, -1).T
        center = (grid - 1) * 0.04
        body_esdf = np.ones(27)
        body_esdf[13] = -1.0
        self.body_voxels = _MockVoxels(center, body_esdf)
        self.leg_voxels = _MockVoxels(center)
        self.robot_reachable_legs = np.ones((27, 2, 6), dtype=bool)
        self.to_bound_dist_flat = np.full((27, 6), 0.1)
        self.kin = _MockKinematic()


class _MockPointMap:
    def __init__(self):
        self.landing_points = np.asarray(
            [[x, y, 0.0] for x in np.linspace(0, 2, 20) for y in (-0.1, 0.1)]
        )


class _MockHexState:
    def __init__(self):
        self.robot_voxels = _MockRobot()
        self.env_pointsmap_voxels = _MockPointMap()

    def RobotFeasiCheck(self, _pose):
        landing_idx = np.arange(18)
        leg_mask = np.zeros((6, 18, 2), dtype=bool)
        for index in range(6):
            leg_mask[index, index * 3 : index * 3 + 3, 0] = True
        return np.arange(5), landing_idx, np.ones(5, dtype=bool), leg_mask


class OffscreenSceneTest(unittest.TestCase):
    def test_scene_contains_all_actor_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = load_se3_path_json(_write_json(directory, _valid_document()))
            stl = directory / "map.stl"
            pv.Box(bounds=(-0.2, 2.2, -0.3, 0.3, -0.1, 0.0)).triangulate().save(stl)
            viewer = SE3PathVisualizer(
                _MockHexState(), data, map_stl=stl, curve_dt=0.2
            )
            plotter = viewer.build_scene(off_screen=True)
            try:
                names = set(plotter.renderer.actors)
                expected = {
                    "environment", "static_landings", "active_path", "waypoints",
                    "robot_body",
                }
                self.assertTrue(expected <= names)
                self.assertTrue(all(f"robot_leg_{index}" in names for index in range(6)))
                self.assertTrue(
                    all(f"feasible_landing_{index}" in names for index in range(6))
                )

                interactor = plotter.iren.interactor
                self.assertEqual(viewer.playback.current_time, 0.0)
                interactor.SetKeyEventInformation(0, 0, "n", 0, "n")
                interactor.InvokeEvent("KeyPressEvent")
                self.assertEqual(viewer.playback.current_time, 0.5)
                interactor.SetKeyEventInformation(0, 0, " ", 0, "space")
                interactor.InvokeEvent("KeyPressEvent")
                self.assertTrue(viewer.playback.playing)
            finally:
                plotter.close()


if __name__ == "__main__":
    unittest.main()
