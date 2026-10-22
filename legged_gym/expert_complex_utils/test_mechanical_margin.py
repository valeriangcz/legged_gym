"""Headless checks of the exact joint-based margin; caches use temp dirs only."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from legged_gym.expert_complex_utils.env_robot_voxels import LegVoxels
from legged_gym.expert_complex_utils.hex_utils import Kinematic


class MechanicalMarginTest(unittest.TestCase):
    def make_voxels(self, negative=False, kin=None):
        bounds = [[-0.08, -0.04], [-0.02, 0.02], [-0.10, -0.06]] if negative else [
            [0.20, 0.24], [-0.02, 0.02], [-0.12, -0.08]]
        return LegVoxels(kin or Kinematic(), bounds, 0.01)

    def solve(self, negative=False, kin=None):
        voxels = self.make_voxels(negative, kin)
        with contextlib.redirect_stdout(io.StringIO()):
            voxels.SolveAllVoxels()
        return voxels

    def test_jacobian_matches_foot_finite_difference(self):
        kin = Kinematic()
        for q in (np.array([0.2, -0.7, 1.2]), np.array([-0.5, 0.4, -1.4])):
            numerical = np.column_stack([
                (kin.ForwardKinReturn(q + dq) - kin.ForwardKinReturn(q - dq)) / 2e-6
                for dq in np.eye(3) * 1e-6
            ])
            np.testing.assert_allclose(kin.Jacobian(q[None])[0], numerical, atol=1e-10)

    def test_joint_formula_matches_each_branch_and_torque_boundary(self):
        for negative in (False, True):
            with self.subTest(negative_x=negative):
                voxels = self.solve(negative)
                valid = np.isfinite(voxels.joints).all(axis=2)
                self.assertTrue(valid.any())
                self.assertTrue(valid.all(axis=1).any())
                self.assertEqual(voxels.mechanical_margin.shape, (voxels.grid_size,))
                self.assertEqual(voxels.mechanical_margin.dtype, np.float32)
                self.assertFalse(hasattr(voxels, 'mechanical_margin_direction'))
                for row, branch in np.argwhere(valid):
                    jac = voxels.kin.Jacobian(voxels.joints[row, branch][None].astype(float))[0]
                    norms = np.linalg.norm(jac, axis=0)
                    expected = 27 / norms.max()
                    self.assertAlmostEqual(voxels.mechanical_margin[row], expected, places=4)
                    direction = jac[:, np.argmax(norms)] / norms.max()
                    tau = jac.T @ (direction * voxels.mechanical_margin[row])
                    np.testing.assert_allclose(np.max(np.abs(tau)), 27, rtol=4e-7)
                    self.assertGreater(np.max(np.abs(tau * 1.001)), 27)

    def test_uses_solved_joints_and_repeatability(self):
        voxels = self.solve()
        arrays = (voxels.joints, voxels.reachable_legs, voxels.ankle_pos,
                  voxels.x_b3, voxels.mechanical_margin)
        before = [array.copy() for array in arrays]
        with patch.object(voxels.kin, 'Jacobian', wraps=voxels.kin.Jacobian) as jacobian:
            voxels.ComputeMechanicalMargin()
            valid = np.isfinite(voxels.joints).all(axis=2)
            rows = np.flatnonzero(valid.any(axis=1))
            branches = np.argmax(valid[rows], axis=1)
            np.testing.assert_array_equal(jacobian.call_args.args[0],
                                          voxels.joints[rows, branches].astype(np.float64))
        for expected, actual in zip(before, (*arrays[:-1], voxels.mechanical_margin)):
            np.testing.assert_array_equal(actual, expected)

    def test_center_changes_do_not_override_solved_joints(self):
        voxels = self.solve()
        expected = voxels.mechanical_margin.copy()
        voxels.center.fill(np.nan)
        voxels.ComputeMechanicalMargin()
        np.testing.assert_array_equal(voxels.mechanical_margin, expected)
        voxels.joints[0] = np.array([0.1, -0.4, 1.0])
        voxels.ComputeMechanicalMargin()
        jac = voxels.kin.Jacobian(voxels.joints[0, :1].astype(np.float64))[0]
        self.assertAlmostEqual(voxels.mechanical_margin[0], 27 / np.linalg.norm(jac, axis=0).max(), places=4)

    def test_batching_and_first_valid_branch(self):
        voxels = self.make_voxels()
        voxels.grid_size = 4101
        rng = np.random.default_rng(31)
        voxels.joints = rng.uniform(-1, 1, (4101, 2, 3))
        voxels.joints[::2, 0] = np.nan
        voxels.joints[-1] = np.nan
        with patch.object(voxels.kin, 'Jacobian', wraps=voxels.kin.Jacobian) as jacobian:
            voxels.ComputeMechanicalMargin()
            self.assertEqual([call.args[0].shape[0] for call in jacobian.call_args_list], [4096, 4])
        rows = np.arange(4100)
        branches = np.where(rows % 2 == 0, 1, 0)
        jac = voxels.kin.Jacobian(voxels.joints[rows, branches])
        expected = 27 / np.linalg.norm(jac, axis=1).max(axis=1)
        np.testing.assert_allclose(voxels.mechanical_margin[:-1], expected, rtol=1e-7)
        self.assertTrue(np.isnan(voxels.mechanical_margin[-1]))

    def test_no_solution_single_solution_and_nonfinite_joints(self):
        voxels = self.solve()
        voxels.joints[0] = np.nan
        voxels.joints[1, 0] = np.nan
        voxels.joints[2, 1] = np.inf
        voxels.ComputeMechanicalMargin()
        self.assertTrue(np.isnan(voxels.mechanical_margin[0]))
        self.assertTrue(np.isfinite(voxels.mechanical_margin[1:3]).all())
        voxels.joints.fill(np.nan)
        voxels.ComputeMechanicalMargin()
        self.assertTrue(np.isnan(voxels.mechanical_margin).all())

    def test_upper_bound_and_custom_geometry(self):
        for kin in (Kinematic(), Kinematic(l1=0.08, l2=0.14, l3=0.19)):
            voxels = self.solve(kin=kin)
            finite = np.isfinite(voxels.mechanical_margin)
            self.assertTrue(finite.any())
            self.assertTrue(np.all(voxels.mechanical_margin[finite] <= 27 / kin.l3 + 1e-5))
            valid = np.isfinite(voxels.joints).all(axis=2)
            jac = kin.Jacobian(voxels.joints[valid].astype(float))
            expected = 27 / np.max(np.linalg.norm(jac, axis=1), axis=1)
            np.testing.assert_allclose(np.broadcast_to(voxels.mechanical_margin[:, None], valid.shape)[valid],
                                       expected, rtol=4e-7)

    def test_axis_boundary_and_ankle_limited_plateau(self):
        voxels = self.make_voxels()
        # Solve synthetic positions directly, including +/-y axes and the origin.
        positions = np.array([[0, .20, -.1], [0, -.20, -.1], [0, 0, 0], [.165, .005, .055]])
        voxels.center[:4] = positions
        voxels.joints[:4], _ = voxels.kin.InverseKin2MultiBatch(voxels.center[:4])
        voxels.ComputeMechanicalMargin()
        self.assertTrue(np.isfinite(voxels.mechanical_margin[[0, 1, 3]]).all())
        self.assertTrue(np.isnan(voxels.mechanical_margin[2]))
        self.assertAlmostEqual(voxels.mechanical_margin[0], voxels.mechanical_margin[1])
        self.assertAlmostEqual(voxels.mechanical_margin[3], 27 / voxels.kin.l3, places=4)
        for row, branch in np.argwhere(np.isfinite(voxels.joints[:4]).all(axis=2)):
            jac = voxels.kin.Jacobian(voxels.joints[row, branch][None].astype(float))[0]
            self.assertAlmostEqual(voxels.mechanical_margin[row], 27 / np.linalg.norm(jac, axis=0).max(), places=4)

    def test_cache_roundtrip_and_legacy_recomputation(self):
        voxels = self.solve()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            voxels.WriteFile(directory)
            file = Path(directory) / 'leg_voxels_info.npz'
            with np.load(file) as saved:
                payload = {key: saved[key] for key in saved.files}
            self.assertNotIn('mechanical_margin_direction', payload)
            loaded = self.make_voxels()
            with patch.object(loaded, 'ComputeMechanicalMargin', side_effect=AssertionError('Valid cache')):
                self.assertTrue(loaded.BuildFromFile(directory))
            np.testing.assert_array_equal(loaded.mechanical_margin, voxels.mechanical_margin)
            variants = [
                {key: value for key, value in payload.items() if key != 'mechanical_margin'},
                dict(payload, mechanical_margin=np.zeros((voxels.grid_size, 2)),
                     mechanical_margin_direction=np.ones(voxels.joints.shape)),
                dict(payload, mechanical_margin=np.zeros((1,))),
            ]
            for legacy in variants:
                np.savez(file, **legacy)
                original_bytes = file.read_bytes()
                loaded = self.make_voxels()
                with patch.object(loaded, 'ComputeMechanicalMargin', wraps=loaded.ComputeMechanicalMargin) as compute:
                    self.assertTrue(loaded.BuildFromFile(directory))
                    compute.assert_called_once_with()
                self.assertEqual(file.read_bytes(), original_bytes)
                self.assertFalse(hasattr(loaded, 'mechanical_margin_direction'))
                np.testing.assert_array_equal(loaded.mechanical_margin, voxels.mechanical_margin)
                for key in ('joints', 'ankle_pos', 'x_b3', 'reachable_legs'):
                    np.testing.assert_array_equal(getattr(loaded, key), payload[key])
                loaded.WriteFile(directory)
                with np.load(file) as updated:
                    self.assertNotIn('mechanical_margin_direction', updated.files)
                    np.testing.assert_array_equal(updated['mechanical_margin'], voxels.mechanical_margin)
            # A leftover direction key must not force valid single-value caches to recompute.
            np.savez(file, **dict(payload, mechanical_margin_direction=np.ones(voxels.joints.shape)))
            loaded = self.make_voxels()
            with patch.object(loaded, 'ComputeMechanicalMargin', side_effect=AssertionError('Valid cache')):
                loaded.BuildFromFile(directory)
            self.assertFalse(hasattr(loaded, 'mechanical_margin_direction'))

    def test_plot_filters_leg_and_nonfinite_values_in_B_frame(self):
        voxels = self.make_voxels()
        voxels.reachable_legs = np.zeros((voxels.grid_size, 2, 6), dtype=bool)
        voxels.reachable_legs[0, 0, 2] = True
        voxels.reachable_legs[1, 1, 2] = True
        voxels.reachable_legs[2, 0, 2] = True
        voxels.reachable_legs[3, 0, 2] = True
        voxels.reachable_legs[4, 0, 0] = True
        voxels.mechanical_margin[:5] = [80, 120, np.nan, np.inf, 100]
        before = voxels.mechanical_margin.copy()
        target = 'legged_gym.expert_complex_utils.env_robot_voxels.pv'
        with patch(target + '.Plotter') as factory, patch(target + '.PolyData') as polydata:
            voxels.plot_leg_voxels(np.int64(2))
            np.testing.assert_array_equal(polydata.call_args.args[0], voxels.center[:2])
            name, values = polydata.return_value.__setitem__.call_args.args
            self.assertEqual(name, 'mechanical_margin')
            np.testing.assert_array_equal(values, [80, 120])
            plotter = factory.return_value
            kwargs = plotter.add_mesh.call_args.kwargs
            self.assertEqual(kwargs['clim'], (60, 159))
            self.assertEqual(kwargs['cmap'], 'PiYG')
            self.assertEqual(kwargs['scalars'], name)
            self.assertEqual(kwargs['point_size'], 20)
            self.assertTrue(kwargs['render_points_as_spheres'])
            bar = kwargs['scalar_bar_args']
            self.assertEqual(bar['title'], 'Mechanical margin (N)')
            self.assertEqual(bar['fmt'], '%.0f')
            self.assertTrue(bar['vertical'])
            self.assertGreaterEqual(bar['position_x'], 0.8)
            self.assertEqual(bar['title_font_size'], 22)
            self.assertEqual(bar['label_font_size'], 20)
            self.assertIn('LM', plotter.add_text.call_args.args[0])
            plotter.show_axes.assert_called_once_with()
            plotter.show_grid.assert_called_once_with(xtitle='B x (m)', ytitle='B y (m)', ztitle='B z (m)')
            plotter.show.assert_called_once_with()
        np.testing.assert_array_equal(voxels.mechanical_margin, before)

    def test_plot_rejects_invalid_index_and_uninitialized_data(self):
        voxels = self.make_voxels()
        for index in (True, np.bool_(False), 1.5, '0', None):
            with self.subTest(index=index), self.assertRaises(TypeError):
                voxels.plot_leg_voxels(index)
        for index in (-1, 6):
            with self.subTest(index=index), self.assertRaises(ValueError):
                voxels.plot_leg_voxels(index)
        with self.assertRaisesRegex(RuntimeError, 'SolveAllVoxels'):
            voxels.plot_leg_voxels(0)

    def test_plot_empty_selection_does_not_create_window(self):
        voxels = self.make_voxels()
        voxels.reachable_legs = np.zeros((voxels.grid_size, 2, 6), dtype=bool)
        with patch('legged_gym.expert_complex_utils.env_robot_voxels.pv.Plotter') as factory:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                voxels.plot_leg_voxels(5)
            factory.assert_not_called()
            self.assertIn('RM', output.getvalue())


if __name__ == '__main__':
    unittest.main()
