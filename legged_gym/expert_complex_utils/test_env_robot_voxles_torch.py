"""Numerical reference tests; CPU fixtures never construct or write map caches."""
import math
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial import cKDTree
from spatialmath import SE3
from spatialmath.base import trlog
import torch

from .env_robot_voxels import HexState, Voxels
from .hex_utils import Kinematic
from .env_robot_voxles_torch import (
    HexStateTorch, KinematicTorch, VoxelsTorch, RobotVoxelData, VoxelGridData,
    se3_exp, se3_log, se3_inverse, se3_geodesic, se3_bezier,
)


def make_state(point_count=42):
    """Real asymmetric six-leg frames, spatially varying fields and lookup data."""
    state = object.__new__(HexState)
    state.kin = Kinematic(bx=0.13, by=0.24)
    rng = np.random.default_rng(137)
    points = []
    for leg_id in range(6):
        local = np.array([0.123, 0.021, -0.143])+rng.uniform(-0.018, 0.018, (7, 3))
        points.extend(state.kin._B2R(local.T, leg_id).T)
    if point_count > len(points):
        for i in range(point_count-len(points)):
            local = np.array([0.123, 0.021, -0.143])+rng.uniform(-0.055, 0.055, 3)
            points.append(state.kin._B2R(local[:, None], i % 6)[:, 0])
    points = np.asarray(points)[:point_count].reshape(-1, 3)
    leg = Voxels(np.array([[-0.5, 0.5]]*3), 0.05)
    leg.x_b3 = np.broadcast_to([0., 0., -1.], (leg.grid_size, 2, 3)).copy()
    leg.x_b3[:, 0] = [0.4, 0., -0.9]
    leg.ankle_pos = np.repeat((leg.center*0.6)[:, None], 2, axis=1)
    leg.ankle_pos[:, 0, 2] += 0.013
    leg.ankle_collide_radi = 0.045
    boundary = (0.085+0.015*leg.center[:, 0]+0.011*leg.center[:, 1]).reshape(tuple(leg.grid_shape))
    boundary = np.stack([boundary+0.001*i for i in range(6)], -1)
    reachable = np.broadcast_to((np.linalg.norm(leg.center, axis=1) < 0.43)[:, None, None],
                                (leg.grid_size, 2, 6)).copy()
    # Different legs/branches have different table availability.
    reachable[leg.center[:, 0] < -0.3, 0, 2] = False
    body = Voxels(np.array([[-0.4, 0.4], [-0.5, 0.5], [-0.2, 0.2]]), 0.05)
    body.bounding_points = np.array([[0.213, 0.087, 0.091], [-0.171, -0.133, 0.063]])
    body.esdf_for_env = np.full(tuple(body.grid_shape), 0.1)
    state.robot_voxels = SimpleNamespace(
        leg_voxels=leg, to_bound_dist=boundary, to_bound_dist_flat=boundary.reshape(-1, 6),
        robot_reachable_legs=reachable, kin=state.kin, body_voxels=body,
    )
    env = Voxels(np.array([[-1., 1.]]*3), 0.1)
    esdf = (0.067+0.04*env.center[:, 0]+0.015*env.center[:, 1]+0.009*env.center[:, 2]).reshape(tuple(env.grid_shape))
    normals = np.broadcast_to([0., 0., 1.], points.shape).copy()
    state.env_pointsmap_voxels = SimpleNamespace(
        points=points.copy(), landing_points=points.copy(), normals=normals,
        landing_count=len(points), voxels=env, env_esdf=esdf,
        _tree=cKDTree(points) if len(points) else None,
        _landing_tree=cKDTree(points) if len(points) else None,
        BuildESDF=Mock(side_effect=AssertionError("ESDF rebuild is forbidden")),
    )
    return state


def make_map_data(state):
    """Test fixture packing; production accepts only explicit data or NPZ files."""
    robot, env = state.robot_voxels, state.env_pointsmap_voxels
    leg, body = robot.leg_voxels, robot.body_voxels
    grid = lambda value: VoxelGridData(value._bounds, value.voxel_scale)
    copy = lambda value: np.array(value, copy=True)
    return RobotVoxelData(grid(leg), grid(env.voxels), copy(state.kin.leg_base_p), copy(leg.center),
                          copy(leg.x_b3), copy(leg.ankle_pos), copy(robot.robot_reachable_legs),
                          copy(robot.to_bound_dist), copy(body.bounding_points), copy(env.points),
                          copy(env.landing_points), copy(env.normals), env.landing_count, copy(env.env_esdf),
                          leg.ankle_collide_radi, grid(body), copy(body.esdf_for_env))


class SE3TorchTest(unittest.TestCase):
    def test_exp_log_and_inverse_match_spatialmath(self):
        rng = np.random.default_rng(7)
        twists = rng.normal(size=(32, 6))*0.3
        twists[0] = 0
        twists[1] *= 1e-9
        matrices = np.asarray([SE3.Exp(twist).A for twist in twists])
        values = torch.tensor(twists, dtype=torch.float64)
        np.testing.assert_allclose(se3_exp(values).numpy(), matrices, atol=1e-12)
        result = se3_log(torch.tensor(matrices))
        np.testing.assert_allclose(result.numpy(), twists, atol=1e-12)
        identity = se3_inverse(torch.tensor(matrices)) @ torch.tensor(matrices)
        np.testing.assert_allclose(identity.numpy(), np.broadcast_to(np.eye(4), identity.shape), atol=1e-12)

    def test_near_pi_and_exact_pi_roundtrip(self):
        axis = np.array([-0.2, 0.4, 0.7]); axis /= np.linalg.norm(axis)
        poses = [SE3.Trans(0.1, -0.2, 0.3)*SE3.AngVec(angle, axis)
                 for angle in (math.pi-1e-5, math.pi-1e-10, math.pi)]
        tensor = torch.tensor(np.asarray([pose.A for pose in poses]))
        np.testing.assert_allclose(se3_exp(se3_log(tensor)).numpy(), tensor.numpy(), atol=1e-10)
        expected = trlog(poses[0].A, twist=True, check=False)
        np.testing.assert_allclose(se3_log(tensor)[0].numpy(), expected, atol=1e-6)

    def test_geodesic_and_bezier(self):
        from .se3_post_process import PostProcess
        reference = object.__new__(PostProcess)
        poses = [SE3(), SE3.Tx(0.15)*SE3.Rz(0.12), SE3.Trans(0.25, 0.04, -0.02)*SE3.Ry(0.2),
                 SE3.Trans(0.4, 0.1, 0.03)*SE3.Rz(0.3)]
        u = torch.linspace(0, 1, 17, dtype=torch.float64)
        controls = torch.tensor(np.asarray([pose.A for pose in poses]))[None]
        actual = se3_bezier(controls, u).numpy()
        expected = np.asarray([reference.Bezier(poses, float(parameter)).A for parameter in u])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        extrapolation = se3_geodesic(controls[0, 1], controls[0, 2], 2.3)
        np.testing.assert_allclose(extrapolation.numpy(), reference.Geodesic(poses[1], poses[2], 2.3).A, atol=1e-12)


class VoxelAndKinematicTorchTest(unittest.TestCase):
    def test_frame_transforms_scalar_and_broadcast_indices(self):
        kin = Kinematic(bx=0.19, by=0.27)
        snapshot = KinematicTorch(kin.leg_base_p, dtype=torch.float64)
        points = np.random.default_rng(8).normal(size=(6, 11, 3))
        tensor = torch.tensor(points)
        indices = torch.arange(6)[:, None]
        local = snapshot._R2B(tensor, indices)
        for i in range(6):
            np.testing.assert_allclose(local[i].numpy(), kin._R2B(points[i].T, i).T)
            np.testing.assert_allclose(snapshot._B2R(tensor[i], i).numpy(), kin._B2R(points[i].T, i).T)
            np.testing.assert_allclose(snapshot.RVectorToLeg(tensor[i], i).numpy(), kin.RVectorToLeg(points[i].T, i).T)
        np.testing.assert_allclose(snapshot._B2R(local, indices).numpy(), points, atol=1e-12)

    def test_sampling_edges_outside_and_channels(self):
        voxel = Voxels(np.array([[-0.2, 0.3], [-0.3, 0.4], [-0.1, 0.2]]), 0.1)
        rng = np.random.default_rng(4)
        values = rng.normal(size=tuple(voxel.grid_shape))
        points = np.r_[voxel.center[:5], rng.uniform(-0.6, 0.6, (91, 3)),
                       [[-0.2, -0.3, -0.1], [0.3, 0.4, 0.2]]].astype(np.float32)
        snapshot = VoxelsTorch(VoxelGridData(voxel._bounds, voxel.voxel_scale), dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "VoxelGridData"):
            VoxelsTorch(voxel)
        for outside in (0., -0.08, 0.05):
            actual = snapshot.TrilinearSample(torch.tensor(values), torch.tensor(points).double(), outside)
            expected = voxel.TrilinearSample(values, points, outside)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
        channels = torch.tensor(np.stack((values, values+3), -1))
        selected = torch.arange(len(points)) % 2
        actual = snapshot.TrilinearSample(channels, torch.tensor(points).double(), -0.08, selected)
        expected = np.array([voxel.TrilinearSample(values+3*int(channel), point, -0.08)
                             for point, channel in zip(points, selected)])
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
        np.testing.assert_array_equal(snapshot.Pos2FlatIndex(torch.tensor(points)).numpy(), voxel.Pos2FlatIndex(points))


class RobotScoringTorchTest(unittest.TestCase):
    def compare_reference(self, state, poses, rows, branch=1, dtype=torch.float64):
        snapshot = HexStateTorch(make_map_data(state), "cpu", dtype, pose_chunk_size=2)
        prepared = snapshot.prepare_candidates(rows, branch)
        actual = snapshot.RobotFeasiCost(torch.tensor(np.asarray([pose.A for pose in poses])), prepared).numpy()
        expected = [state.RobotFeasiCost(pose, row, branch) for pose, row in zip(poses, rows)]
        np.testing.assert_allclose(actual, expected, rtol=1e-6 if dtype == torch.float64 else 1e-4,
                                   atol=1e-6 if dtype == torch.float64 else 1e-4)
        return snapshot, actual

    def test_asymmetric_frames_both_branches_and_mixed_precision(self):
        state = make_state()
        poses = [SE3(), SE3.Trans(0.017, -0.009, 0.013)*SE3.Rz(0.12), SE3.Trans(-0.031, 0.021, 0.007)*SE3.Ry(-0.14)]
        rows = [np.arange(42), np.arange(3, 38), np.arange(25)]
        for dtype in (torch.float64, torch.float32):
            for branch in (0, 1):
                self.compare_reference(state, poses, rows, branch, dtype)

    def test_invalid_candidates_empty_and_no_score(self):
        for invalid in ("none", "xb3", "ankle", "reachable", "outside", "normal"):
            state = make_state()
            if invalid == "xb3": state.robot_voxels.leg_voxels.x_b3[:] = np.nan
            if invalid == "ankle": state.robot_voxels.leg_voxels.ankle_pos[:] = np.nan
            if invalid == "reachable": state.robot_voxels.robot_reachable_legs[:] = False
            if invalid == "outside": state.env_pointsmap_voxels.landing_points[:] = 2
            if invalid == "normal": state.env_pointsmap_voxels.normals[:] = np.nan
            with self.subTest(invalid=invalid):
                rows = [np.repeat(np.arange(5), 2), np.array([-1, 999]), np.arange(42)]
                self.compare_reference(state, [SE3()]*3, rows)
        self.compare_reference(make_state(0), [SE3()], [np.array([], dtype=np.int64)])

    def test_blocked_and_per_point_branches(self):
        state = make_state()
        pointmap = state.env_pointsmap_voxels
        pointmap.points = np.r_[pointmap.points, [[0., 0., 0.]]]
        pointmap.normals = np.r_[pointmap.normals, [[0., 0., 1.]]]
        snapshot = HexStateTorch(make_map_data(state), "cpu", torch.float64)
        indices = np.arange(43)[None]
        branches = np.ones_like(indices)
        expected = state.RobotFeasiCost(SE3(), indices[0], 1)
        actual = snapshot.RobotFeasiCost(torch.eye(4)[None], indices, branches)
        self.assertAlmostEqual(float(actual[0]), expected, places=5)
        branches[:] = 0
        actual = snapshot.RobotFeasiCost(torch.eye(4)[None], indices, branches)
        self.assertAlmostEqual(float(actual[0]), state.RobotFeasiCost(SE3(), indices[0], 0), places=5)
        branches[:, ::2] = 1
        self.assertTrue(torch.isfinite(snapshot.RobotFeasiCost(torch.eye(4)[None], indices, branches)).all())
        with self.assertRaisesRegex(ValueError, "conflicting"):
            snapshot.prepare_candidates([np.array([1, 1])], np.array([[0, 1]]))
        for invalid in (True, 2, -1, 0.5):
            with self.assertRaises(ValueError):
                snapshot.RobotFeasiCost(torch.eye(4)[None], indices, invalid)

    def test_snapshot_no_rebuild_reload_and_input_validation(self):
        state = make_state()
        with patch("numpy.load", side_effect=AssertionError("disk read")), patch("numpy.savez", side_effect=AssertionError("disk write")):
            snapshot = HexStateTorch(make_map_data(state), "cpu")
            state.env_pointsmap_voxels.BuildESDF.assert_not_called()
        previous = snapshot.env_esdf.clone()
        state.env_pointsmap_voxels.env_esdf += 0.1
        torch.testing.assert_close(snapshot.env_esdf, previous)
        snapshot.reload_from(make_map_data(state))
        torch.testing.assert_close(snapshot.env_esdf, previous+0.1)
        state.env_pointsmap_voxels.env_esdf = np.zeros((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "env_esdf shape"):
            snapshot.reload_from(make_map_data(state))
        torch.testing.assert_close(snapshot.env_esdf, previous+0.1)
        with self.assertRaisesRegex(TypeError, "RobotVoxelData"):
            HexStateTorch(make_state(), "cpu")
        state = make_state()
        state.env_pointsmap_voxels.voxels.voxel_scale = 0
        with self.assertRaises(ValueError): HexStateTorch(make_map_data(state), "cpu")
        poses = torch.eye(4)[None].repeat(2, 1, 1)
        poses[1, 0, 0] = float("nan")
        snapshot = HexStateTorch(make_map_data(make_state()), "cpu")
        cost = snapshot.RobotFeasiCost(poses, np.tile(np.arange(42), (2, 1)))
        self.assertTrue(torch.isfinite(cost[0])); self.assertTrue(torch.isinf(cost[1]))

    def test_batched_search_matches_cpu(self):
        state = make_state()
        snapshot = HexStateTorch(make_map_data(state), "cpu")
        poses = [SE3(), SE3.Tx(0.03)]
        rows = snapshot.GetRobotFeasiCostPointsBatch(np.asarray([pose.A for pose in poses]), workers=2)
        for row, pose in zip(rows, poses):
            np.testing.assert_array_equal(np.sort(row), np.sort(state.GetRobotFeasiCostPoints(pose)))
        empty = HexStateTorch(make_map_data(make_state(0)), "cpu")
        self.assertEqual(empty.GetRobotFeasiCostPointsBatch(np.eye(4)[None])[0].size, 0)

    def test_native_hard_check_matches_reference(self):
        for variant in ("normal", "body_collision", "zero_normals", "nan_normals", "nan_xb3", "nan_ankle", "blocked"):
            state = make_state()
            if variant == "body_collision": state.robot_voxels.body_voxels.esdf_for_env[:] = -0.1
            if variant == "zero_normals": state.env_pointsmap_voxels.normals[:] = 0
            if variant == "nan_normals": state.env_pointsmap_voxels.normals[:] = np.nan
            if variant == "nan_xb3": state.robot_voxels.leg_voxels.x_b3[:] = np.nan
            if variant == "nan_ankle": state.robot_voxels.leg_voxels.ankle_pos[:] = np.nan
            if variant == "blocked": state.robot_voxels.robot_reachable_legs[:] = False
            scorer = HexStateTorch(make_map_data(state), "cpu", torch.float64)
            for pose in (SE3(), SE3.Trans(0.017, -0.009, 0.013)*SE3.Rz(0.12), SE3.Tx(1.3)):
                with self.subTest(variant=variant, pose=pose.t):
                    actual = scorer.RobotFeasiCheck(pose, np.arange(42))
                    expected = state.RobotFeasiCheck(pose, np.arange(42))
                    for first, second in zip(actual, expected):
                        np.testing.assert_array_equal(first, second)
        empty = HexStateTorch(make_map_data(make_state(0)), "cpu")
        checked = empty.RobotFeasiCheck(np.eye(4))
        self.assertEqual(checked[2].shape, (0,))
        self.assertEqual(checked[3].shape, (6, 0, 2))
        data = make_map_data(make_state())
        data.body_grid = data.body_esdf = None
        with self.assertRaisesRegex(RuntimeError, "body_grid and body_esdf"):
            HexStateTorch(data, "cpu").RobotFeasiCheck(np.eye(4))

    def test_independent_cache_loading_and_validation(self):
        data = make_map_data(make_state())
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            cloud = directory/"pointmap.npz"
            # Only the fields actually needed for inference are required.
            np.savez(cloud, points=data.points, normals=data.normals,
                     landing_count=[data.landing_count], bounds=data.env_grid.bounds)
            np.savez(directory/"leg_voxels_info.npz", x_b3=data.x_b3, ankle_pos=data.ankle_pos)
            np.savez(directory/"body_voxels_info.npz", esdf_for_env=data.body_esdf, bounding_points=data.bounding_points)
            np.savez(directory/"robot_reachable_legs.npz", robot_reachable_legs=data.reachable, to_bound_dist=data.to_bound_dist)
            np.savez(directory/"env_voxels.npz", esdf=data.env_esdf)
            options = dict(leg_bounds=data.leg_grid.bounds, body_bounds=data.body_grid.bounds,
                           leg_voxel_scale=0.05, body_voxel_scale=0.05, env_voxel_scale=0.1,
                           leg_base_p=data.leg_base_p)
            expected_landing = data.points+data.normals*0.03
            data.landing_points = expected_landing
            expected = HexStateTorch(data, "cpu", torch.float64)
            with patch("numpy.load", wraps=np.load) as reader, patch("numpy.savez", side_effect=AssertionError("disk write")):
                scorer = HexStateTorch.from_files(cloud, directory, device="cpu", dtype=torch.float64, **options)
                self.assertEqual(reader.call_count, 5)
                scorer.RobotFeasiCost(torch.eye(4)[None], np.arange(42)[None])
                scorer.RobotFeasiCheck(np.eye(4))
                self.assertEqual(reader.call_count, 5)
            torch.testing.assert_close(scorer.env_esdf, expected.env_esdf)
            torch.testing.assert_close(scorer.landing_points, expected.landing_points)
            torch.testing.assert_close(scorer.RobotFeasiCost(torch.eye(4)[None], np.arange(42)[None]),
                                       expected.RobotFeasiCost(torch.eye(4)[None], np.arange(42)[None]))
            self.assertFalse(hasattr(scorer, "source"))
            # The module also loads with all legacy robot modules disabled.
            name = "legged_gym.expert_complex_utils.env_robot_voxles_torch_isolated"
            spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("env_robot_voxles_torch.py"))
            independent = importlib.util.module_from_spec(spec)
            blocked = {f"legged_gym.expert_complex_utils.{part}": None
                       for part in ("env_robot_voxels", "hex_utils", "point_map")}
            with patch.dict(sys.modules, {**blocked, name: independent}):
                spec.loader.exec_module(independent)
                independent.HexStateTorch.from_files(cloud, directory, device="cpu", **options)
            with self.assertRaisesRegex(ValueError, "env_esdf shape"):
                HexStateTorch.from_files(cloud, directory, device="cpu", **{**options, "env_voxel_scale": 0.2})
            np.savez(directory/"env_voxels.npz", wrong_field=data.env_esdf)
            with self.assertRaisesRegex(ValueError, "missing fields: esdf"):
                HexStateTorch.from_files(cloud, directory, device="cpu", **options)
            missing = directory/"missing_env.npz"
            with self.assertRaises(FileNotFoundError):
                HexStateTorch.from_files(cloud, directory, device="cpu", env_esdf_file=missing, **options)
            self.assertFalse(missing.exists())

    def test_hard_boundary_count_and_precision_diagnostics(self):
        state = make_state()
        # Isolate the workspace hard threshold: radius/normal/ankle constraints
        # are satisfied for some points, with identical lookup tables in both modes.
        state.env_pointsmap_voxels.env_esdf[:] = 0.2
        for boundary in (0.04-1e-5, 0.04+1e-5):
            state.robot_voxels.to_bound_dist[:] = boundary
            results = []
            for dtype in (torch.float32, torch.float64):
                scorer = HexStateTorch(make_map_data(state), "cpu", dtype)
                cost, counts = scorer.RobotFeasiCost(torch.eye(4)[None], np.arange(42)[None], return_details=True)
                results.append(counts["hard_feasible_count"])
                np.testing.assert_allclose(float(cost[0]), state.RobotFeasiCost(SE3(), np.arange(42)), rtol=1e-4)
            torch.testing.assert_close(results[0], results[1])
            if boundary < 0.04: self.assertEqual(int(results[0].sum()), 0)
            else: self.assertGreater(int(results[0].sum()), 0)
        # At the threshold the mixed/double difference must be observable through
        # per-leg diagnostics, rather than concealed in a total-cost tolerance.
        state.robot_voxels.to_bound_dist[:] = 0.04
        for dtype in (torch.float32, torch.float64):
            scorer = HexStateTorch(make_map_data(state), "cpu", dtype)
            _, counts = scorer.RobotFeasiCost(torch.eye(4)[None], np.arange(42)[None], return_details=True)
            self.assertEqual(counts["hard_feasible_count"].shape, (1, 6))
            self.assertTrue((counts["hard_feasible_count"] <= counts["scorable_count"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: GPU correctness/performance not measured")
    def test_cuda_matches_cpu(self):
        state = make_state()
        data = make_map_data(state)
        cpu, cuda = HexStateTorch(data, "cpu"), HexStateTorch(data, "cuda")
        indices = np.tile(np.arange(42), (5, 1))
        poses = torch.eye(4)[None].repeat(5, 1, 1)
        np.testing.assert_allclose(cuda.RobotFeasiCost(poses, indices).cpu().numpy(),
                                   cpu.RobotFeasiCost(poses, indices).numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
