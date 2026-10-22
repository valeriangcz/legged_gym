"""Population, continuity, lifecycle and optional CUDA benchmark tests."""
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from spatialmath import SE3
import torch

from .env_robot_voxles_torch import HexStateTorch
from .se3_post_process import PostProcess, TrajectorySampleGrid as ReferenceGrid
from .se3_post_process_torch import OptCfgTorch, PostProcessTorch
from .test_env_robot_voxles_torch import make_state, make_map_data


def make_post(segments=2, precision="float64", device="cpu", candidate_count=42):
    post = object.__new__(PostProcessTorch)
    post.se3_path = []
    post.se3_path_short = [SE3.Trans(0.03*i, 0.01*np.sin(i), 0.004*np.sin(i/2))*SE3.Rz(0.018*i)
                          for i in range(segments+1)]
    post.se3_ctrl_poses = []
    post.se3_segment_nums = 0
    post.se3_segment_times = []
    post.opt_poses_index = []
    post.vec_continuous_poses_index = []
    post.reference_state = make_state(candidate_count)
    cfg = post.opt_cfg = OptCfgTorch()
    cfg.torch_precision = precision
    cfg.sampling_rho_interval = 0.025
    cfg.sampling_phi_interval = 0.08
    cfg.presample_count = 7
    cfg.torch_population_chunk_size = 2
    cfg.torch_pose_chunk_size = 11
    cfg.cost_progress_interval = 0
    post.torch_hex = HexStateTorch(make_map_data(post.reference_state), device,
                                  torch.float64 if precision == "float64" else torch.float32,
                                  pose_chunk_size=cfg.torch_pose_chunk_size)
    post._ensure_torch()
    post.GetCtrlPoes()
    return post


def make_reference(post):
    """Separate NumPy object: comparisons must not dispatch to Torch adapters."""
    reference = object.__new__(PostProcess)
    reference.__dict__.update(post.__dict__)
    reference.hex_state = post.reference_state
    reference.se3_path_short = [pose.copy() for pose in post.se3_path_short]
    reference.se3_ctrl_poses = [pose.copy() for pose in post.se3_ctrl_poses]
    reference.se3_segment_times = list(post.se3_segment_times)
    reference.opt_poses_index = list(post.opt_poses_index)
    reference.vec_continuous_poses_index = list(post.vec_continuous_poses_index)
    return reference


class TrajectoryBatchTorchTest(unittest.TestCase):
    def test_updates_and_sampling_match_reference(self):
        post = make_post(3)
        rng = np.random.default_rng(3)
        for optimize_all in (True, False):
            post.opt_cfg.optimize_all = optimize_all
            post.GetCtrlPoes()
            # Exercise uneven segment durations and continuity extrapolation.
            post.se3_segment_times = [0.9, 1.7, 0.6]
            post.UpdateCtrlPoses(np.zeros((len(post.opt_poses_index), 6)))
            reference = make_reference(post)
            deltas = rng.normal(size=(4, len(post.opt_poses_index), 6))*0.006
            deltas[0] = 0
            updated = post.UpdateCtrlPosesBatch(deltas)
            grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
            reference_grid = reference._BuildTrajectorySampleGrid(reference.se3_ctrl_poses)
            np.testing.assert_allclose(grid.times, reference_grid.times, rtol=1e-10, atol=1e-12)
            sampled = post.SampleTrajectoryBatch(updated, grid)
            for i, delta in enumerate(deltas):
                expected_ctrl = reference.UpdateCtrlPoses(delta, update_original=False)
                expected = np.asarray([pose.A for pose in expected_ctrl])
                np.testing.assert_allclose(updated[i].numpy(), expected, atol=1e-10)
                _, expected_samples = reference._SampleTrajectory(
                    expected_ctrl, ReferenceGrid(grid.times, grid.segment_indices, grid.u_values))
                np.testing.assert_allclose(sampled[i].numpy(), np.asarray([pose.A for pose in expected_samples]), atol=1e-10)
            torch.testing.assert_close(updated[:, 0], updated[0, 0].expand_as(updated[:, 0]))
            torch.testing.assert_close(updated[:, -1], updated[0, -1].expand_as(updated[:, -1]))

    def test_population_cost_matches_numpy_and_chunking(self):
        post = make_post(3)
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        candidates = post.PrepareTrajectoryCandidates(grid)
        rng = np.random.default_rng(10)
        deltas = rng.normal(size=(5, len(post.opt_poses_index), 6))*0.007
        updated = post.UpdateCtrlPosesBatch(deltas)
        total, feasibility, acceleration = post.TrajectoryCostBatch(updated, candidates, grid)
        _, base_poses = post._SampleTrajectory(post.se3_ctrl_poses, grid)
        rows = [post.reference_state.GetRobotFeasiCostPoints(pose) for pose in base_poses]
        reference = make_reference(post)
        reference_grid = ReferenceGrid(grid.times, grid.segment_indices, grid.u_values)
        for i, delta in enumerate(deltas):
            controls = reference.UpdateCtrlPoses(delta, update_original=False)
            expected = reference._TrajectoryMeanCost(controls, rows, reference_grid)
            np.testing.assert_allclose([total[i], feasibility[i], acceleration[i]], expected, rtol=1e-6, atol=1e-6)
        post.opt_cfg.torch_population_chunk_size = 8
        post.opt_cfg.torch_pose_chunk_size = 64
        rechunked = post.TrajectoryCostBatch(updated, candidates, grid)
        for first, second in zip((total, feasibility, acceleration), rechunked):
            torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
        updated = updated.clone()
        updated[1, 2, 0, 0] = float("nan")
        invalid = post.TrajectoryCostBatch(updated, candidates, grid)
        self.assertTrue(torch.isinf(invalid[0][1]))
        torch.testing.assert_close(invalid[0][0], total[0])

    def test_twenty_segments_mixed_precision(self):
        post = make_post(20, "mixed", candidate_count=2400)
        self.assertEqual(len(post.se3_ctrl_poses), 80)
        self.assertEqual(len(post.opt_poses_index), 40)
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        deltas = np.random.default_rng(11).normal(size=(3, 40, 6))*0.002
        controls = post.UpdateCtrlPosesBatch(deltas)
        candidates = post.PrepareTrajectoryCandidates(grid)
        self.assertEqual(candidates.indices.shape[1], 2400)
        costs = post.TrajectoryCostBatch(controls, candidates, grid)
        self.assertEqual(costs[0].shape, (3,))
        self.assertTrue(torch.isfinite(costs[0]).all())
        self.assertEqual(controls.dtype, torch.float64)
        self.assertEqual(post.torch_hex.env_esdf.dtype, torch.float32)

    def test_constructor_from_path_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"path.json"
            path.write_text(json.dumps({"t": [[0., 0., 0.], [0.03, 0., 0.]],
                                        "ang": [0., 0.], "vec": [[0., 0., 0.], [0., 0., 0.]]}))
            cfg = OptCfgTorch()
            scorer = HexStateTorch(make_map_data(make_state()), "cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                post = PostProcessTorch(str(path), scorer, cfg)
            self.assertEqual(len(post.se3_path), 2)
            with self.assertRaisesRegex(TypeError, "HexStateTorch"):
                PostProcessTorch(str(path), make_state(), cfg)
            cfg.torch_precision = "unsupported"
            with self.assertRaises(ValueError): PostProcessTorch(str(path), scorer, cfg)
        post = make_post()
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        with self.assertRaisesRegex(ValueError, "candidate rows"):
            post.TrajectoryCostBatch(post._control_tensor()[None], [np.array([0])], grid)

    def test_independent_module_and_explicit_scorer_checker(self):
        module_name = "legged_gym.expert_complex_utils.se3_post_process_torch_isolated"
        location = Path(__file__).with_name("se3_post_process_torch.py")
        spec = importlib.util.spec_from_file_location(module_name, location)
        independent = importlib.util.module_from_spec(spec)
        # Loading must succeed even when the legacy postprocessor cannot import.
        with patch.dict(sys.modules, {"legged_gym.expert_complex_utils.se3_post_process": None,
                                      module_name: independent}):
            spec.loader.exec_module(independent)
        self.assertEqual(independent.PostProcessTorch.__bases__, (object,))
        self.assertEqual(independent.OptCfgTorch.__bases__, (object,))
        state = make_state()
        checker = Mock(return_value=(np.arange(1), np.arange(1), np.ones(1, dtype=bool),
                                     np.ones((6, 4, 2), dtype=bool)))
        scorer = HexStateTorch(make_map_data(state), "cpu", torch.float64)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"path.json"
            path.write_text(json.dumps({"t": [[0., 0., 0.], [0.03, 0., 0.]],
                                        "ang": [0., 0.], "vec": [[0., 0., 0.], [0., 0., 0.]]}))
            post = PostProcessTorch(str(path), scorer, hard_check=checker)
            self.assertIs(post.torch_hex, scorer)
            post.opt_cfg.presample_count = 7
            post.ShortCutPath()
            post.GetCtrlPoes()
            grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
            valid, _ = post._ValidateTrajectory(post.se3_ctrl_poses, grid)
            self.assertTrue(valid)
            checker.assert_called()
            post.hard_check = None
            with patch.object(scorer, "RobotFeasiCheck", wraps=scorer.RobotFeasiCheck) as native:
                post._ValidateTrajectory(post.se3_ctrl_poses, grid)
                native.assert_called()
            self.assertFalse(hasattr(post, "hex_state"))
            self.assertFalse(hasattr(scorer, "source"))

    def test_grid_stationary_and_hard_checks_match_reference(self):
        post = make_post(3)
        post.se3_path_short[1] = post.se3_path_short[0].copy()
        post.GetCtrlPoes()
        reference = make_reference(post)
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        expected_grid = reference._BuildTrajectorySampleGrid(reference.se3_ctrl_poses)
        np.testing.assert_allclose(grid.times, expected_grid.times, rtol=1e-10, atol=1e-12)
        body = np.ones(3, dtype=bool)
        legs = np.ones((6, 5, 2), dtype=bool)
        post.hard_check = post.reference_state.RobotFeasiCheck = lambda pose: (None, None, body, legs)
        actual = post._ValidateTrajectory(post.se3_ctrl_poses, grid)
        expected = reference._ValidateTrajectory(reference.se3_ctrl_poses,
                                                ReferenceGrid(grid.times, grid.segment_indices, grid.u_values))
        self.assertEqual(actual[0], expected[0])
        for name in actual[1]:
            self.assertAlmostEqual(actual[1][name], expected[1][name], places=8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: GPU trajectory benchmark not measured")
    def test_cuda_population_and_benchmark(self):
        cpu = make_post(20, "mixed", candidate_count=2400)
        cuda = make_post(20, "mixed", "cuda", candidate_count=2400)
        grid = cpu._BuildTrajectorySampleGrid(cpu.se3_ctrl_poses)
        deltas = np.random.default_rng(22).normal(size=(8, 40, 6))*0.002
        expected = cpu.TrajectoryCostBatch(cpu.UpdateCtrlPosesBatch(deltas), cpu.PrepareTrajectoryCandidates(grid), grid)
        controls = cuda.UpdateCtrlPosesBatch(deltas)
        candidates = cuda.PrepareTrajectoryCandidates(grid)
        for _ in range(2): cuda.TrajectoryCostBatch(controls, candidates, grid)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actual = cuda.TrajectoryCostBatch(controls, candidates, grid)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        for first, second in zip(expected, actual):
            np.testing.assert_allclose(first.numpy(), second.cpu().numpy(), rtol=1e-4, atol=1e-4)
        print(json.dumps({"benchmark": "synthetic_20_segment_population", "population": 8,
                          "samples": len(grid.times), "candidate_width": candidates.indices.shape[1],
                          "seconds": elapsed, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}))


class CMATorchLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        context = patch("legged_gym.expert_complex_utils.se3_post_process_torch.LEGGED_GYM_ROOT_DIR", self.directory.name)
        context.start(); self.addCleanup(context.stop)

    def post(self):
        post = make_post(1)
        post.se3_path_short = [SE3(), SE3.Tx(0.12)]
        cfg = post.opt_cfg
        cfg.max_retraction_iterations = 1
        cfg.cma_population_size = 4
        cfg.cma_max_generations = 2
        cfg.acceleration_weight = 0
        cfg.cost_abs_change_tol = cfg.cost_rel_change_tol = 0
        cfg.delt_rho_limit = 0.02
        post.torch_hex._score = lambda poses, candidates: ((poses[:, 2, 3]-0.015)/0.02)**2
        post._ValidateTrajectory = Mock(return_value=(True, {"valid": True}))
        return post

    @staticmethod
    def strategy_factory(generations, captured):
        def factory(x0, sigma, options):
            batches = []
            for heights in generations:
                rows = []
                for height in heights:
                    solution = np.zeros_like(x0); solution[2::6] = height
                    rows.append(solution)
                batches.append(rows)
            strategy = SimpleNamespace(popsize=len(batches[0]), ask=Mock(side_effect=batches),
                                       tell=Mock(), stop=lambda: {})
            captured.append(strategy)
            return strategy
        return factory

    def run_scripted(self, post, heights):
        captured = []
        with patch("cma.CMAEvolutionStrategy", side_effect=self.strategy_factory(heights, captured)):
            with contextlib.redirect_stdout(io.StringIO()):
                result = post.Optimize_CMA_ES()
        return result, captured[0]

    def test_population_order_counts_export_and_final_cost(self):
        post = self.post()
        result, strategy = self.run_scripted(post, [[0., 0.2, 0.4, 0.6], [0.1, 0.3, 0.5, 0.7]])
        self.assertTrue(result.success)
        self.assertEqual(result.nit, 2)
        self.assertEqual(result.nfev, 10)  # base + 2*4 + final
        self.assertEqual(strategy.tell.call_count, 2)
        # first generation costs retain original ask order and favor larger height
        first_costs = strategy.tell.call_args_list[0].args[1]
        self.assertTrue(np.all(np.diff(first_costs) < 0))
        self.assertIn("torch", result.output_path.name)
        document = json.loads(result.output_path.read_text())
        self.assertEqual(len(document["bezier_segments"]), 1)
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        expected = post._TrajectoryMeanCost(post.se3_ctrl_poses, post.PrepareTrajectoryCandidates(grid), grid)[0]
        self.assertAlmostEqual(result.fun, expected, places=12)
        np.testing.assert_allclose(document["bezier_segments"][0]["control_poses"]["t"],
                                   [pose.t for pose in post.se3_ctrl_poses])

    def test_invalid_generation_stops_without_tell(self):
        post = self.post()
        original = post.TrajectoryCostBatch
        call_number = 0
        def invalid_population(controls, *args, **kwargs):
            nonlocal call_number
            call_number += 1
            if len(controls) > 1:
                return tuple(torch.full((len(controls),), float("inf"), dtype=torch.float64) for _ in range(3))
            return original(controls, *args, **kwargs)
        post.TrajectoryCostBatch = invalid_population
        result, strategy = self.run_scripted(post, [[0., 0.2, 0.4, 0.6]])
        strategy.tell.assert_not_called()
        self.assertEqual(result.nfev, 6)
        self.assertIn("non-finite", result.message)

    def test_fresh_grid_validation_rolls_back_to_initial(self):
        post = self.post()
        # Accept initial; reject all modified curves. Initial controls are zero z.
        post._ValidateTrajectory = lambda controls, grid=None: (
            all(abs(pose.t[2]) < 1e-12 for pose in controls), {"checked": True})
        result, _ = self.run_scripted(post, [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
        self.assertTrue(result.rolled_back)
        np.testing.assert_allclose(result.x, 0., atol=1e-12)

    def test_final_nonfinite_cost_restores_initial(self):
        post = self.post()
        original = post.TrajectoryCostBatch
        seen_single = 0
        def fail_final(controls, *args, **kwargs):
            nonlocal seen_single
            if len(controls) == 1:
                seen_single += 1
                if seen_single == 2:
                    return tuple(torch.full((1,), float("inf"), dtype=torch.float64) for _ in range(3))
            return original(controls, *args, **kwargs)
        post.TrajectoryCostBatch = fail_final
        with self.assertRaisesRegex(RuntimeError, "restored"):
            self.run_scripted(post, [[0., 0.2, 0.4, 0.6], [0.1, 0.2, 0.3, 0.4]])
        self.assertTrue(all(abs(pose.t[2]) < 1e-12 for pose in post.se3_ctrl_poses))

    def test_real_cma_and_local_rng(self):
        post = self.post()
        post.opt_cfg.cma_population_size = 8
        post.opt_cfg.cma_max_generations = 3
        np.random.seed(991)
        before = np.random.get_state()
        with contextlib.redirect_stdout(io.StringIO()):
            result = post.Optimize_CMA_ES()
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(result.nfev, 26)
        self.assertLess(result.fun, (0.015/0.02)**2)


if __name__ == "__main__":
    unittest.main()
