"""Headless checks of angular attribution and independent radial prefixes.

Use small in-memory robot/environment maps; never rebuild or write robot caches.
"""
from types import SimpleNamespace
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from spatialmath import SE3

from legged_gym.expert_complex_utils.env_robot_voxels import (
    HexState, LegVoxels, RoboVoxels, SphereVoxels, Voxels,
)
from legged_gym.expert_complex_utils.hex_utils import Kinematic


class SphereGeometryTest(unittest.TestCase):
    def test_centers_order_and_partial_last_cell(self):
        grid = SphereVoxels(np.deg2rad([[-10, 15], [-5, 10]]), np.deg2rad(10))
        np.testing.assert_array_equal(grid.grid_shape, [3, 2])
        self.assertEqual(grid.grid_size, 6)
        np.testing.assert_allclose(grid.alpha_values, np.deg2rad([-5, 5, 12.5]))
        np.testing.assert_allclose(grid.beta_values, np.deg2rad([0, 7.5]))
        np.testing.assert_allclose(grid.Pos2Sphere(grid.directions)[:, :2],
                                   grid.angle_centers.reshape(-1, 2), atol=1e-12)
        np.testing.assert_array_equal(grid.Pos2GridIndex(grid.directions),
                                      [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])

    def test_roundtrip_axes_single_and_empty(self):
        grid = SphereVoxels([[-np.pi, np.pi], [-np.pi/2, np.pi/2]], np.deg2rad(10))
        np.testing.assert_array_equal(grid.grid_shape, [36, 18])
        points = np.array([[1, 0, 0], [0, 2, 0], [0, 0, -3], [.2, -.3, .4]])
        np.testing.assert_allclose(grid.Sphere2Pos(grid.Pos2Sphere(points)), points, atol=1e-14)
        np.testing.assert_allclose(grid.Pos2Sphere([0, 2, 0]), [np.pi/2, 0, 2])
        np.testing.assert_allclose(grid.Sphere2Pos([np.pi/2, 0, 2]), [0, 2, 0], atol=1e-14)
        self.assertEqual(grid.Pos2Sphere(np.empty((0, 3))).shape, (0, 3))
        self.assertEqual(grid.Pos2Section(np.empty((0, 3))).shape, (0, 2))

    def test_boundaries_and_invalid_directions(self):
        grid = SphereVoxels(np.deg2rad([[-40, 92], [-50, 50]]), np.deg2rad(10))
        np.testing.assert_array_equal(grid.grid_shape, [14, 10])
        sphere = np.array([[-40, -50, 1], [92, 50, 1], [-40.01, 0, 1],
                           [92.01, 0, 1], [0, 50.01, 1]], dtype=float)
        sphere[:, :2] = np.deg2rad(sphere[:, :2])
        np.testing.assert_array_equal(grid.Pos2GridIndex(grid.Sphere2Pos(sphere)),
                                      [[0, 0], [13, 9], [-1, -1], [-1, -1], [-1, -1]])
        invalid = np.array([[0, 0, 0], [np.nan, 0, 1], [np.inf, 0, 1]])
        np.testing.assert_array_equal(grid.Pos2GridIndex(invalid), np.full((3, 2), -1))
        np.testing.assert_array_equal(grid.Pos2Section(invalid), np.full((3, 2), -1))
        boundary = grid.Sphere2Pos([np.deg2rad(-30), 0, 1])
        np.testing.assert_array_equal(
            grid.Pos2GridIndex(np.array([boundary * .001, boundary * 100])),
            [[1, 5], [1, 5]],
        )

    def test_parameters_and_shapes(self):
        for bounds, scale in [([[0, 0], [0, 1]], 1), ([[0, 1]], 1),
                              ([[0, np.nan], [0, 1]], 1), ([[0, 1], [0, 1]], 0),
                              ([[0, 1], [0, 91]], 1)]:
            with self.subTest(bounds=bounds, scale=scale), self.assertRaises(ValueError):
                SphereVoxels(np.deg2rad(bounds), np.deg2rad(scale))
        grid = SphereVoxels(np.deg2rad([[-10, 10], [-10, 10]]), np.deg2rad(10))
        with self.assertRaises(ValueError):
            grid.Pos2Sphere(np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            grid.CountFootholds([[1, 0, 0]], np.ones((1, 1)))

    def test_components_are_four_connected_nonperiodic_and_branch_specific(self):
        grid = SphereVoxels(np.deg2rad([[-180, 180], [-30, 30]]), np.deg2rad(60))
        grid.radius_grid[0, 0, 0] = .2
        grid.radius_grid[-1, 0, 0] = .2
        grid.radius_grid[0, 0, 1] = .2
        grid.radius_grid[1, 0, 1] = .2
        grid.LabelSections()
        np.testing.assert_array_equal(grid.section_count, [2, 1])
        self.assertNotEqual(grid.section_grid[0, 0, 0], grid.section_grid[-1, 0, 0])
        self.assertEqual(grid.section_grid[0, 0, 1], grid.section_grid[1, 0, 1])
        diagonal = SphereVoxels(np.deg2rad([[-20, 20], [-20, 20]]), np.deg2rad(20))
        diagonal.radius_grid[0, 0, 0] = .2
        diagonal.radius_grid[1, 1, 0] = .2
        diagonal.LabelSections()
        np.testing.assert_array_equal(diagonal.section_count, [2, 0])
        self.assertTrue((diagonal.section_grid[..., 1] == -1).all())

    def test_attribution_ignores_radius_counts_zero_regions_and_masks(self):
        grid = SphereVoxels(np.deg2rad([[-30, 30], [-10, 10]]), np.deg2rad(20))
        grid.radius_grid[0, 0, :] = .2
        grid.radius_grid[2, 0, :] = .2
        grid.LabelSections()
        point = grid.directions[0]
        points = np.array([point * .001, point * 100, grid.directions[1]])
        labels, counts = grid.CountFootholds(points, [[1, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(grid.Pos2Section(points[:2]), [[0, 0], [0, 0]])
        np.testing.assert_array_equal(labels, [[0, 0], [0, -1], [-1, -1]])
        np.testing.assert_array_equal(counts[0], [2, 0])
        np.testing.assert_array_equal(counts[1], [1, 0])
        labels, counts = grid.CountFootholds(np.empty((0, 3)), np.empty((0, 2)))
        self.assertEqual(labels.shape, (0, 2))
        np.testing.assert_array_equal(counts, [[0, 0], [0, 0]])

    def test_leg_angles_scale_and_left_right_frames(self):
        kin = Kinematic()
        leg = LegVoxels(kin, [[.1, .12], [-.01, .01], [-.01, .01]], .01,
                        sphere_scale=np.deg2rad(7))
        for i, grid in enumerate(leg.sphere_voxels_list):
            np.testing.assert_array_equal(grid.bounds[0], kin.joints_limits_thigh[i])
            np.testing.assert_allclose(grid.bounds[1], [-np.pi/2, np.pi/2])
            self.assertEqual(grid.scale, np.deg2rad(7))
            points = grid.directions[:3].T * .2
            np.testing.assert_allclose(kin._R2B(kin._B2R(points, i), i), points, atol=1e-14)
        with patch.object(RoboVoxels, 'ReadRobotVoxels', return_value=True):
            robot = RoboVoxels(kin, leg_xyz_range=[[.1, .12], [-.01, .01], [-.01, .01]],
                               body_xyz_range=[[-.01, .01]] * 3, sphere_scale=np.deg2rad(9))
        self.assertTrue(all(s.scale == np.deg2rad(9) for s in robot.leg_voxels.sphere_voxels_list))
        default_leg = LegVoxels(kin, [[.1, .12], [-.01, .01], [-.01, .01]], .01)
        self.assertTrue(all(s.scale == np.deg2rad(1) for s in default_leg.sphere_voxels_list))


class RadialConnectionsTest(unittest.TestCase):
    def make_state(self):
        state = HexState.__new__(HexState)
        state.kin = Kinematic()
        leg = Voxels(np.array([[-.5, .5]] * 3), .05)
        leg.joints = np.zeros((leg.grid_size, 2, 3))
        leg.ankle_pos = np.zeros((leg.grid_size, 2, 3))
        leg.ankle_collide_radi = .045
        leg.sphere_voxels_list = [
            SphereVoxels(np.deg2rad([[-20, 20], [-10, 10]]), np.deg2rad(20))
            for _ in range(6)
        ]
        state.robot_voxels = SimpleNamespace(
            kin=state.kin, leg_voxels=leg,
            robot_reachable_legs=np.ones((leg.grid_size, 2, 6), dtype=bool))
        env = Voxels(np.array([[-2., 2.]] * 3), .1)
        state.env_pointsmap_voxels = SimpleNamespace(
            voxels=env, env_esdf=np.ones(tuple(env.grid_shape)))
        return state

    def test_independent_branches_no_recovery_and_exact_terminal_radius(self):
        state = self.make_state()
        visited = []

        def collision(points, pose, leg_index, **kwargs):
            self.assertEqual(kwargs, dict(ignore_ankle=False, ignore_end=False))
            radius = np.linalg.norm(points, axis=1)
            visited.extend(radius)
            # Branch 0 fails once at .2, then would be free again at larger radii.
            return np.column_stack([~np.isclose(radius, .2), np.ones(len(points), dtype=bool)])

        with patch.object(state, '_LegEnvCollisionFree', side_effect=collision) as cf:
            table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.35, radial_step=.1)
        self.assertEqual(cf.call_count, 1)
        np.testing.assert_allclose(visited, np.repeat([.1, .2, .3, .35], 2))
        np.testing.assert_allclose(table.radius_grid[..., 0], .1)
        np.testing.assert_allclose(table.radius_grid[..., 1], .35)
        np.testing.assert_array_equal(table.section_count, [0, 1])
        self.assertTrue((table.section_grid[..., 0] == -1).all())

    def test_direction_and_branch_prefixes_are_independent_in_one_batch(self):
        state = self.make_state()
        batches = []

        def collision(points, *args, **kwargs):
            radius = np.linalg.norm(points, axis=1)
            batches.append(radius)
            negative = points[:, 1] < 0
            return np.column_stack([
                ~(negative & (radius > .15)), ~(negative & (radius > .25))])

        with patch.object(state, '_LegEnvCollisionFree', side_effect=collision):
            table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.4, radial_step=.1)
        self.assertEqual(len(batches), 1)
        np.testing.assert_allclose(batches[0], np.repeat([.1, .2, .3, .4], 2))
        np.testing.assert_allclose(table.radius_grid[:, 0, :], [[.1, .2], [.4, .4]])
        np.testing.assert_array_equal(table.section_grid[0, 0], [-1, -1])
        np.testing.assert_array_equal(table.section_grid[1, 0], [0, 0])

    def test_initial_failure_remains_invalid_after_outer_recovery(self):
        state = self.make_state()
        def collision(points, *args, **kwargs):
            valid = ~np.isclose(np.linalg.norm(points, axis=1), .1)
            return np.repeat(valid[:, None], 2, axis=1)

        with patch.object(state, '_LegEnvCollisionFree', side_effect=collision) as cf:
            table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.4, radial_step=.1)
        self.assertEqual(cf.call_count, 1)
        self.assertTrue(np.isneginf(table.radius_grid).all())
        self.assertTrue((table.section_grid == -1).all())

    def test_rebuild_clears_previous_pose_and_does_not_write_caches(self):
        state = self.make_state()
        table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.1)
        self.assertTrue(np.isfinite(table.radius_grid).all())
        tables = state.BuildRobotConnections(SE3.Tx(10), safe_radius=.1, max_radius=.2)
        self.assertIs(table, tables[0])
        for table in tables:
            self.assertTrue(np.isneginf(table.radius_grid).all())
            self.assertTrue((table.section_grid == -1).all())
            np.testing.assert_array_equal(table.section_count, [0, 0])

    def test_default_step_and_equal_radius(self):
        state = self.make_state()
        radii = []

        def collision(points, *args, **kwargs):
            radii.extend(np.linalg.norm(points, axis=1))
            return np.ones((len(points), 2), bool)

        with patch.object(state, '_LegEnvCollisionFree', side_effect=collision):
            state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.21)
        np.testing.assert_allclose(radii, np.repeat([.1, .15, .2, .21], 2))
        table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.1)
        np.testing.assert_allclose(table.radius_grid, .1)

    def test_default_band_requires_every_sample_in_three_centimeters(self):
        state = self.make_state()
        def collision(points, *args, **kwargs):
            radii = np.linalg.norm(points, axis=1)
            # Both endpoints are free; only one branch fails inside the band.
            return np.column_stack([~np.isclose(radii, .155), np.ones(len(points), bool)])

        with patch.object(state, '_LegEnvCollisionFree', side_effect=collision) as cf:
            table = state._LegConnections(SE3(), 0, radial_step=.01)
        self.assertEqual(cf.call_count, 1)
        np.testing.assert_allclose(np.linalg.norm(cf.call_args.args[0], axis=1),
                                   np.repeat([.145, .155, .165, .175], 2))
        self.assertAlmostEqual(table.required_radius, .175)
        np.testing.assert_allclose(table.radius_grid[..., 0], .145)
        np.testing.assert_allclose(table.radius_grid[..., 1], .175)
        np.testing.assert_array_equal(table.section_count, [0, 1])
        # Re-labeling must preserve the thickness criterion, not just finite radii.
        table.LabelSections()
        np.testing.assert_array_equal(table.section_count, [0, 1])

    def test_batch_prefix_matches_sequential_reference(self):
        state = self.make_state()
        rng = np.random.default_rng(42)
        samples = np.arange(6) * .01 + .1
        free = rng.random((6, 2, 2)) > .25
        # Include an always-valid branch and a failure followed by recovery.
        free[:, 0, 0] = True
        free[:, 1, 1] = [True, False, True, True, True, True]
        active = np.ones((2, 2), dtype=bool)
        expected = np.full((2, 2), -np.inf)
        for radius, sample_free in zip(samples, free):
            active &= sample_free
            expected[active] = radius
        with patch.object(state, '_LegEnvCollisionFree', return_value=free.reshape(-1, 2)):
            table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.15, radial_step=.01)
        np.testing.assert_allclose(table.radius_grid.reshape(2, 2), expected)
        np.testing.assert_array_equal(table.section_grid.reshape(2, 2) >= 0, active)

    def test_foot_and_ankle_collisions_independently_truncate(self):
        for obstacle in ('foot', 'ankle'):
            with self.subTest(obstacle=obstacle):
                state = self.make_state()
                leg = state.robot_voxels.leg_voxels
                env = state.env_pointsmap_voxels
                table = leg.sphere_voxels_list[0]
                points = .1 * table.directions
                flat = leg.Pos2FlatIndex(points)
                if obstacle == 'foot':
                    positions = state.kin._B2R(leg.center[flat].T, 0).T
                else:
                    # Put ankles away from feet so each test isolates one check.
                    leg.ankle_pos[flat] = [.0, .0, -.3]
                    positions = state.kin._B2R(leg.ankle_pos[flat, 0].T, 0).T
                index = env.voxels.Pos2GridIndex(positions)
                env.env_esdf[tuple(index.T)] = 0
                table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.2)
                self.assertTrue(np.isneginf(table.radius_grid).all())

    def test_robot_bounds_and_ankle_bounds_are_not_clipped_to_free_cells(self):
        state = self.make_state()
        table = state._LegConnections(SE3(), 0, safe_radius=1, max_radius=1)
        self.assertTrue(np.isneginf(table.radius_grid).all())
        leg = state.robot_voxels.leg_voxels
        leg.ankle_pos[:, 0] = [10, 0, 0]
        table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.1)
        self.assertTrue(np.isneginf(table.radius_grid[..., 0]).all())
        self.assertTrue(np.isfinite(table.radius_grid[..., 1]).all())

    def test_nonfinite_ik_and_ankle_are_invalid_per_branch(self):
        state = self.make_state()
        leg = state.robot_voxels.leg_voxels
        leg.joints[:, 0] = np.nan
        leg.ankle_pos[:, 1] = np.nan
        with patch.object(state, '_LegEnvCollisionFree') as collision:
            table = state._LegConnections(SE3(), 0, safe_radius=.1, max_radius=.2)
        collision.assert_not_called()
        self.assertTrue(np.isneginf(table.radius_grid).all())

    def test_manual_demo_uses_direction_counts_without_opening_a_window(self):
        from legged_gym.expert_complex_utils.test_val import visualize_connections

        state = self.make_state()
        state.env_pointsmap_voxels.landing_points = state.kin._B2R(
            (.3 * state.robot_voxels.leg_voxels.sphere_voxels_list[0].directions).T, 0).T
        state.RobotFeasiCheck = lambda pose: (
            None, np.arange(2), None, np.ones((6, 2, 2), dtype=bool))
        with patch('legged_gym.expert_complex_utils.test_val.HexState', return_value=state), \
                patch('pyvista.Plotter') as factory, contextlib.redirect_stdout(io.StringIO()):
            visualize_connections(SE3(), 0, 0, safe_radius=.1, max_radius=.1)
            factory.return_value.show.assert_called_once_with()
            self.assertEqual(factory.return_value.subplot.call_count, 3)
            factory.reset_mock()
            visualize_connections(SE3.Tx(10), 0, 0, safe_radius=.1, max_radius=.1)
            factory.return_value.show.assert_called_once_with()
            factory.return_value.add_mesh.assert_not_called()

    def test_parameter_validation(self):
        state = self.make_state()
        for kwargs in (dict(safe_radius=0), dict(max_radius=.1), dict(radial_step=0),
                       dict(radial_step=np.nan), dict(max_radius=np.inf)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                state._LegConnections(SE3(), 0, **kwargs)
        for leg_index in (-1, 6, True, 1.5):
            with self.assertRaises(ValueError):
                state._LegConnections(SE3(), leg_index)


if __name__ == '__main__':
    unittest.main()
