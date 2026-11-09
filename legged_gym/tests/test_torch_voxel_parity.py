"""Run directly: python3 legged_gym/tests/test_torch_voxel_parity.py.

Load actual implementations without importing expert_complex_utils.__init__:
that package also imports simulator/planner dependencies unrelated to parity.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from spatialmath import SE3
import torch


UTILS = Path(__file__).resolve().parents[1] / "expert_complex_utils"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TORCH = load_module("voxel_torch_under_test", UTILS / "env_robot_voxles_torch.py")
KINEMATIC = load_module("kinematic_under_test", UTILS / "hex_utils.py").Kinematic
tree = ast.parse((UTILS / "env_robot_voxels.py").read_text())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("Voxels", "HexState")]
namespace = dict(np=np, map_coordinates=map_coordinates)
exec(compile(ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *classes],
                        type_ignores=[]), "<NumPy voxel implementation>", "exec"), namespace)
Voxels, HexState = namespace["Voxels"], namespace["HexState"]


def numpy_state(data):
    def grid(metadata):
        voxel = Voxels.__new__(Voxels)
        voxel._bounds = metadata.bounds
        voxel.voxel_scale = metadata.voxel_scale
        voxel.grid_shape = metadata.grid_shape
        return voxel
    state = HexState.__new__(HexState)
    state.kin = KINEMATIC.__new__(KINEMATIC)
    state.kin.leg_base_p = data.leg_base_p
    leg = grid(data.leg_grid)
    leg.center, leg.x_b3, leg.ankle_pos = data.centers, data.x_b3, data.ankle_pos
    leg.mechanical_margin = data.mechanical_margin
    leg.ankle_collide_radi = state.LEG_ANKLE_HARD_MARGIN
    state.robot_voxels = SimpleNamespace(
        kin=state.kin, leg_voxels=leg,
        body_voxels=SimpleNamespace(bounding_points=data.bounding_points),
        robot_reachable_legs=data.reachable, to_bound_dist=data.to_bound_dist,
    )
    state.env_pointsmap_voxels = SimpleNamespace(
        voxels=grid(data.env_grid), env_esdf=data.env_esdf,
        points=data.points, landing_points=data.landing_points, normals=data.normals,
        landing_count=data.landing_count, _tree=cKDTree(data.points),
        _landing_tree=cKDTree(data.landing_points),
    )
    return state


def fixture():
    rng = np.random.default_rng(119)
    leg = TORCH.VoxelGridData([[-0.5, 0.5]] * 3, 0.05)
    env = TORCH.VoxelGridData([[-1, 1]] * 3, 0.05)
    centers = leg.centers()
    g = len(centers)
    base = np.array([[-0.1]*3+[0.1]*3, [-0.22, 0.22, 0, -0.22, 0.22, 0], [0]*6], dtype=np.float32)
    points = rng.uniform(-0.4, 0.4, (120, 3)).astype(np.float32)
    normals = rng.normal(size=(120, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    xb3 = rng.normal(size=(g, 2, 3)).astype(np.float32)
    xb3 /= np.linalg.norm(xb3, axis=-1, keepdims=True)
    ankle = np.repeat(centers[:, None], 2, axis=1)
    ankle[:, :, 2] += 0.1
    xb3[::211, 0] = np.nan
    ankle[::307, 1] = np.nan
    margin = rng.uniform(75, 125, g).astype(np.float32)
    margin[::101], margin[::103], margin[::107] = np.nan, -np.inf, np.inf
    # Planar environment field, including both negative and positive clearances.
    field = (env.centers()[:, 2] + 0.18).reshape(tuple(env.grid_shape))
    return TORCH.RobotVoxelData(
        leg, env, base, centers, xb3, ankle,
        rng.random((g, 2, 6)) > 0.1,
        rng.uniform(0.015, 0.09, (*leg.grid_shape, 6)).astype(np.float32),
        rng.uniform(-0.15, 0.15, (257, 3)), points, points.copy(), normals, len(points), field,
        mechanical_margin=margin,
    )


class TorchVoxelParity(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.numpy = numpy_state(self.data)
        self.poses = [SE3(), SE3(0.017, -0.023, -0.035)*SE3.RPY(0.07, -0.09, 0.13),
                      SE3(0.01, 0.03, 0.08)*SE3.RPY(-0.06, 0.05, -0.11)]
        self.idx = np.arange(len(self.data.points))

    def test_constants(self):
        names = [name for name in vars(HexState) if name.startswith(("BODY_", "LEG_", "COST_"))]
        self.assertTrue(names)
        for name in names:
            self.assertEqual(getattr(HexState, name), getattr(TORCH.HexStateTorch, name), name)

    def test_cost_and_hard_check_parity(self):
        for dtype in (torch.float32, torch.float64):
            scorer = TORCH.HexStateTorch(self.data, device="cpu", dtype=dtype, pose_chunk_size=2)
            for branch in (0, 1):
                expected = [self.numpy.RobotFeasiCost(p, self.idx, branch) for p in self.poses]
                candidates = scorer.prepare_candidates([self.idx]*len(self.poses), branch)
                actual = scorer.RobotFeasiCost(np.array([p.A for p in self.poses]), candidates)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=5e-4, atol=1e-4)
                scorer.pose_chunk_size = 1
                chunked = scorer.RobotFeasiCost(np.array([p.A for p in self.poses]), candidates)
                torch.testing.assert_close(chunked, actual)
            for pose in self.poses:
                expected = self.numpy.RobotFeasiCheck(pose, self.idx)
                actual = scorer.RobotFeasiCheck(pose, self.idx)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right)

    def test_body_queries_and_outside(self):
        scorer = TORCH.HexStateTorch(self.data, device="cpu")
        for pose in [*self.poses, SE3(0.95, 0, 0), SE3(3, 0, 0)]:
            expected_points, expected_distance = self.numpy._BodyEnvironmentDistances(pose)
            world, distance = scorer._BodyEnvironmentDistances(torch.tensor(pose.A, dtype=scorer.dtype))
            np.testing.assert_allclose(world.numpy(), expected_points, atol=1e-6)
            np.testing.assert_allclose(distance.numpy(), expected_distance, atol=2e-6)
            _, _, mask, _ = scorer.RobotFeasiCheck(pose, np.zeros(0, dtype=np.int64))
            np.testing.assert_array_equal(mask, expected_distance >= self.numpy.BODY_HARD_CLEARANCE)
        out_cost = scorer.RobotFeasiCost(np.array([SE3(3, 0, 0).A]), self.idx[None])
        self.assertTrue(torch.isposinf(out_cost).all())

    def test_invalid_sampling_and_pose(self):
        for points in (np.empty((0, 3)), np.zeros(3), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                TORCH.HexStateTorch(replace(self.data, bounding_points=points), device="cpu")
        scorer = TORCH.HexStateTorch(self.data, device="cpu")
        pose = np.eye(4)
        pose[0, 0] = np.nan
        self.assertTrue(torch.isposinf(scorer.RobotFeasiCost(pose[None], self.idx[None])).all())
        with self.assertRaises(ValueError):
            scorer.RobotFeasiCheck(pose)
        with self.assertRaises(ValueError):
            scorer.RobotFeasiCost(np.eye(4)[None], self.idx[None], sol_index=2)
        self.assertEqual(tuple(scorer.RobotFeasiCost(
            np.empty((0, 4, 4)), np.empty((0, 0), dtype=np.int64),
        ).shape), (0,))

    def test_hard_thresholds_and_custom_class_constants(self):
        # One foothold belongs to every leg; isolate each threshold around a
        # passing case, including conditions previously different in Torch.
        data = replace(
            self.data, points=np.zeros((1, 3), dtype=np.float32),
            landing_points=np.zeros((1, 3), dtype=np.float32),
            normals=np.array([[0., 0., -1.]], dtype=np.float32), landing_count=1,
            x_b3=np.broadcast_to([0., 0., 1.], self.data.x_b3.shape).copy(),
            reachable=np.ones_like(self.data.reachable),
            mechanical_margin=np.full_like(self.data.mechanical_margin, 100),
            to_bound_dist=np.full_like(self.data.to_bound_dist, 0.07),
            env_esdf=np.full_like(self.data.env_esdf, 0.1),
        )
        cases = [(data, True)]
        for margin in (89.9, 90.1):
            cases.append((replace(data, mechanical_margin=np.full_like(data.mechanical_margin, margin)), margin > 90))
        for distance in (0.029, 0.031):
            cases.append((replace(data, to_bound_dist=np.full_like(data.to_bound_dist, distance)), distance > 0.03))
        for distance in (0.049, 0.051):
            cases.append((replace(data, env_esdf=np.full_like(data.env_esdf, distance)), distance > 0.05))
        for angle in (84., 86.):
            v = [np.sin(np.deg2rad(angle)), 0., np.cos(np.deg2rad(angle))]
            cases.append((replace(data, x_b3=np.broadcast_to(v, data.x_b3.shape).copy()), angle < 85))
        for item, passes in cases:
            state, scorer = numpy_state(item), TORCH.HexStateTorch(item, device="cpu")
            expected = state.RobotFeasiCheck(SE3(), np.array([0]))
            actual = scorer.RobotFeasiCheck(SE3(), np.array([0]))
            np.testing.assert_array_equal(expected[3], actual[3])
            self.assertEqual(bool(actual[3].all()), passes)
            cost, details = scorer.RobotFeasiCost(np.eye(4)[None], np.array([[0]]), return_details=True)
            np.testing.assert_array_equal(details["hard_feasible_count"].numpy(), np.full((1, 6), int(passes)))
            np.testing.assert_allclose(cost.numpy(), [state.RobotFeasiCost(SE3(), np.array([0]))], atol=1e-4)
        for angle in (14., 16.):
            plane = np.array([data.leg_base_p[1, 0], -data.leg_base_p[0, 0], 0.])
            plane /= np.linalg.norm(plane)
            normal = plane*np.sin(np.deg2rad(angle)) + np.array([0., 0., -np.cos(np.deg2rad(angle))])
            xb3 = -normal*np.array([-1, -1, 1])
            item = replace(data, normals=normal[None], x_b3=np.broadcast_to(xb3, data.x_b3.shape).copy())
            scorer, state = TORCH.HexStateTorch(item, device="cpu"), numpy_state(item)
            expected, actual = state.RobotFeasiCheck(SE3(), np.array([0])), scorer.RobotFeasiCheck(SE3(), np.array([0]))
            np.testing.assert_array_equal(expected[3], actual[3])
            self.assertEqual(bool(actual[3][0].all()), angle < 15)
        class CustomScorer(TORCH.HexStateTorch):
            BODY_SOFT_CLEARANCE = 0.2
            BODY_HARD_CLEARANCE = 0.15
            LEG_FORCE_SOFT_MARGIN = 110.
            LEG_ANKLE_COST_SCALE = 0.01
        state = numpy_state(data)
        for name in ("BODY_SOFT_CLEARANCE", "BODY_HARD_CLEARANCE", "LEG_FORCE_SOFT_MARGIN", "LEG_ANKLE_COST_SCALE"):
            setattr(state, name, getattr(CustomScorer, name))
        scorer = CustomScorer(data, device="cpu")
        for idx in (np.array([0]), np.empty(0, dtype=np.int64)):
            actual = scorer.RobotFeasiCost(np.eye(4)[None], idx[None])
            np.testing.assert_allclose(actual.numpy(), [state.RobotFeasiCost(SE3(), idx)], atol=1e-4)
        self.assertFalse(scorer.RobotFeasiCheck(SE3(), np.array([0]))[2].any())

    def test_empty_and_unscorable_candidates(self):
        for reachable in (self.data.reachable, np.zeros_like(self.data.reachable)):
            data = replace(self.data, reachable=reachable)
            scorer, state = TORCH.HexStateTorch(data, device="cpu"), numpy_state(data)
            for idx in (np.empty(0, dtype=np.int64), self.idx):
                expected = state.RobotFeasiCost(SE3(), idx)
                actual = scorer.RobotFeasiCost(np.eye(4)[None], idx[None])
                np.testing.assert_allclose(actual.numpy(), [expected], rtol=5e-4, atol=1e-4)

    def test_cached_storage_and_prepared_candidates(self):
        scorer = TORCH.HexStateTorch(self.data, device="cpu")
        names = ("env_esdf", "to_bound_dist", "x_b3", "ankle_pos", "reachable", "bounding_points")
        addresses = {name: getattr(scorer, name).data_ptr() for name in names}
        prepared = scorer.prepare_candidates([np.r_[self.idx, self.idx[:5]]])
        address = prepared.indices.data_ptr()
        first = scorer.RobotFeasiCost(np.eye(4)[None], prepared)
        for _ in range(3):
            torch.testing.assert_close(scorer.RobotFeasiCost(np.eye(4)[None], prepared), first)
        self.assertEqual(addresses, {name: getattr(scorer, name).data_ptr() for name in names})
        self.assertEqual(address, prepared.indices.data_ptr())
        poses = torch.eye(4, dtype=scorer.dtype)[None]
        with torch.profiler.profile(record_shapes=True) as profile:
            scorer.RobotFeasiCost(poses, prepared)
        cache_shapes = {tuple(getattr(scorer, name).shape) for name in names}
        for event in profile.events():
            if event.name in ("aten::copy_", "aten::clone", "aten::_to_copy", "aten::contiguous"):
                self.assertFalse(any(tuple(shape) in cache_shapes for shape in event.input_shapes),
                                 (event.name, event.input_shapes))
        # Loading creates an independent snapshot; caller mutation cannot change it.
        self.data.env_esdf[:] = -100
        self.assertGreater(float(scorer.env_esdf.max()), 0)

    def test_post_process_hard_check_consumer(self):
        # Execute the actual consumer methods without importing simulator-bound
        # package initialization or changing the post-processing implementation.
        tree = ast.parse((UTILS / "se3_post_process_torch.py").read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PostProcessTorch")
        original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in (
            "_ensure_torch", "_check_pose", "_PoseFeasibilityDiagnostics",
        )]
        ns = dict(np=np, torch=torch, HexStateTorch=TORCH.HexStateTorch)
        module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], original], type_ignores=[])
        exec(compile(module, "<actual post-process consumer>", "exec"), ns)
        post = ns["PostProcessTorch"]()
        post.torch_hex = TORCH.HexStateTorch(self.data, device="cpu")
        post.opt_cfg = SimpleNamespace(torch_precision="mixed", torch_population_chunk_size=2, torch_pose_chunk_size=2)
        result = post._check_pose(SE3())
        self.assertEqual(result[2].shape, (len(self.data.bounding_points),))
        valid, details = post._PoseFeasibilityDiagnostics(SE3(), "parity test")
        self.assertEqual(details["body_valid"], bool(result[2].all()))
        self.assertEqual(details["body_collision_count"], int(np.count_nonzero(~result[2])))
        self.assertEqual(valid, bool(result[2].all() and (result[3].any(-1).sum(-1) >= 3).all()))

    def test_interpolation_channels_and_boundary(self):
        scorer = TORCH.HexStateTorch(self.data, device="cpu")
        voxel = self.numpy.robot_voxels.leg_voxels
        rng = np.random.default_rng(32)
        points = rng.uniform(-0.55, 0.55, (300, 3)).astype(np.float32)
        points = np.vstack([points, [[-0.5]*3, [0.5]*3, [0.475]*3]])
        for channel in range(6):
            expected = voxel.TrilinearSample(self.data.to_bound_dist[..., channel], points, -0.01)
            actual = scorer.leg_voxels.TrilinearSample(
                scorer.to_bound_dist, torch.tensor(points, dtype=scorer.dtype), -0.01, channel,
            )
            np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
