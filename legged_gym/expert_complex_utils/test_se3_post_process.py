import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
from spatialmath import SE3
from spatialmath.base import trlog

from legged_gym.expert_complex_utils.env_robot_voxels import Voxels
from legged_gym.expert_complex_utils.se3_post_process import OptCfg,PostProcess,TrajectorySampleGrid


class SE3PostProcessNumericalTest(unittest.TestCase):
    @staticmethod
    def _post_process_without_files()->PostProcess:
        post = object.__new__(PostProcess)
        post.se3_path = []
        post.se3_path_short = [
            SE3(),
            SE3.Trans(0.2,0.0,0.0)*SE3.Rz(0.2),
            SE3.Trans(0.35,0.15,0.0)*SE3.Rz(0.35),
        ]
        post.se3_ctrl_poses = []
        post.se3_segment_nums = 0
        post.se3_segment_times = []
        post.opt_poses_index = []
        post.vec_continuous_poses_index = []
        post.opt_cfg = OptCfg()
        return post

    def test_time_average_uses_trapezoidal_rule(self):
        values = np.array([2.0,4.0,8.0])
        times = np.array([0.0,1.0,2.0])
        self.assertAlmostEqual(PostProcess._TimeAverage(values,times),4.5)

    def test_trilinear_sampling_reproduces_linear_field(self):
        voxels = Voxels(
            np.array([[0.0,0.04],[0.0,0.04],[0.0,0.04]]),
            voxel_scale=0.01,
        )
        field = (
            2.0*voxels.center[:,0]
            -3.0*voxels.center[:,1]
            +0.5*voxels.center[:,2]
        ).reshape(voxels.grid_shape)
        points = np.array([
            [0.015,0.015,0.015],
            [0.019,0.021,0.027],
            [0.025,0.015,0.025],
        ])
        expected = 2.0*points[:,0]-3.0*points[:,1]+0.5*points[:,2]
        np.testing.assert_allclose(
            voxels.TrilinearSample(field,points,outside_value=-1.0),
            expected,
            atol=3e-9,
        )

    def test_initial_control_poses_are_velocity_continuous(self):
        post = self._post_process_without_files()
        start = post.se3_path_short[0].A.copy()
        end = post.se3_path_short[-1].A.copy()
        post.GetCtrlPoes()
        np.testing.assert_allclose(post.se3_ctrl_poses[0].A,start,atol=1e-12)
        np.testing.assert_allclose(post.se3_ctrl_poses[-1].A,end,atol=1e-12)
        for segment_index in range(1,post.se3_segment_nums):
            junction = post.se3_ctrl_poses[4*segment_index]
            left_ctrl = post.se3_ctrl_poses[4*segment_index-2]
            right_ctrl = post.se3_ctrl_poses[4*segment_index+1]
            left_velocity = np.asarray(
                (left_ctrl.inv()*junction).log(twist=True)
            )*3.0/post.se3_segment_times[segment_index-1]
            right_velocity = np.asarray(
                (junction.inv()*right_ctrl).log(twist=True)
            )*3.0/post.se3_segment_times[segment_index]
            np.testing.assert_allclose(left_velocity,right_velocity,atol=1e-10)

    def test_geodesic_has_negligible_acceleration_cost(self):
        post = self._post_process_without_files()
        post.se3_path_short = post.se3_path_short[:2]
        post.GetCtrlPoes()
        times,poses = post._SampleTrajectory(post.se3_ctrl_poses)
        self.assertLess(post._AccelerationMeanCost(poses,times),1e-20)

    def test_near_orthogonal_rotation_skips_strict_log_check(self):
        post = self._post_process_without_files()
        nearly_rotated = SE3.Rz(0.2)
        nearly_rotated.data[0][0,0] += 1e-14

        with self.assertRaises(ValueError):
            nearly_rotated.log(twist=True)

        midpoint = post.Geodesic(SE3(),nearly_rotated,0.5)
        self.assertTrue(np.isfinite(midpoint.A).all())
        cost = post._AccelerationMeanCost(
            [SE3(),nearly_rotated,SE3.Rz(0.4)],
            np.array([0.0,1.0,2.0]),
        )
        self.assertTrue(np.isfinite(cost))

    @staticmethod
    def _linear_post(waypoints):
        post = SE3PostProcessNumericalTest._post_process_without_files()
        post.se3_path_short = waypoints
        post.GetCtrlPoes()
        return post

    def test_global_distance_sampling_does_not_reset_at_join(self):
        post = self._linear_post([SE3(),SE3.Tx(0.025),SE3.Tx(0.055)])
        times,poses = post._SampleTrajectory(post.se3_ctrl_poses)
        np.testing.assert_allclose([p.t[0] for p in poses],
                                   [0,0.01,0.02,0.025,0.03,0.04,0.05,0.055],atol=1e-12)
        self.assertEqual(np.count_nonzero(times==post.se3_segment_times[0]),1)
        self.assertTrue((np.diff(times)>0).all())
        self.assertAlmostEqual(times[-1],sum(post.se3_segment_times))

    def test_pure_rotation_and_mixed_motion_use_both_limits(self):
        for end in (SE3.Rz(0.12),SE3.Tx(0.07)*SE3.Rz(0.35)):
            with self.subTest(end=end):
                post = self._linear_post([SE3(),end])
                times,poses = post._SampleTrajectory(post.se3_ctrl_poses)
                self.assertGreater(len(poses),3)
                steps = post._SamplingDiagnostics(poses)
                self.assertLessEqual(steps['max_rho_step'],0.010001)
                self.assertLessEqual(steps['max_phi_step'],np.deg2rad(2)+1e-6)
                np.testing.assert_allclose(poses[0].A,SE3().A,atol=1e-12)
                np.testing.assert_allclose(poses[-1].A,end.A,atol=1e-12)

    def test_short_and_stationary_curves_keep_three_time_samples(self):
        for end in (SE3.Tx(0.005),SE3()):
            with self.subTest(end=end):
                post = self._linear_post([SE3(),end])
                times,poses = post._SampleTrajectory(post.se3_ctrl_poses)
                self.assertEqual(len(times),3)
                self.assertTrue((np.diff(times)>0).all())
                self.assertTrue(np.isfinite(post._AccelerationMeanCost(poses,times)))

    def test_stationary_platform_keeps_duration_and_global_distance(self):
        post = self._linear_post([SE3(),SE3.Tx(0.025),SE3.Tx(0.025),SE3.Tx(0.055)])
        # Deliberate stop: each segment is linear, with a stationary middle segment.
        post.se3_segment_times = [1.0,2.0,1.0]
        post.se3_ctrl_poses = []
        for a,b in zip(post.se3_path_short[:-1],post.se3_path_short[1:]):
            post.se3_ctrl_poses.extend([post.Geodesic(a,b,u) for u in (0,1/3,2/3,1)])
        times,poses = post._SampleTrajectory(post.se3_ctrl_poses)
        for time in (1.0,2.0,3.0):
            self.assertEqual(np.count_nonzero(times==time),1)
            np.testing.assert_allclose(poses[np.flatnonzero(times==time)[0]].t,[0.025,0,0])
        self.assertTrue(np.any(np.isclose([p.t[0] for p in poses],0.03)))
        self.assertTrue((np.diff(times)>0).all())
        self.assertTrue(np.isfinite(post._AccelerationMeanCost(poses,times)))

    def test_folded_curve_is_sampled_even_when_endpoints_coincide(self):
        post = self._linear_post([SE3(),SE3()])
        post.se3_segment_times = [1.0]
        post.se3_ctrl_poses = [SE3(),SE3.Ty(0.2),SE3.Ty(0.2),SE3()]
        times,poses = post._SampleTrajectory(post.se3_ctrl_poses)
        self.assertGreater(len(poses),25)
        self.assertGreater(max(p.t[1] for p in poses),0.14)
        self.assertTrue((np.diff(times)>0).all())

    def test_presampling_converges_for_curved_se3_bezier(self):
        post = self._linear_post([SE3(),SE3.Tx(0.3)])
        post.se3_ctrl_poses = [SE3(),SE3.Trans(0.1,0.15,0)*SE3.Rz(0.1),
                              SE3.Trans(0.2,-0.1,0)*SE3.Rx(0.2),
                              SE3.Trans(0.3,0.05,0)*SE3.Rz(0.3)]
        _,coarse = post._SampleTrajectory(post.se3_ctrl_poses)
        post.opt_cfg.presample_count = 401
        _,fine = post._SampleTrajectory(post.se3_ctrl_poses)
        self.assertEqual(len(coarse),len(fine))
        for a,b in zip(coarse,fine):
            delta = trlog((a.inv()*b).A,twist=True,check=False)
            self.assertLess(np.linalg.norm(delta[:3]),5e-5)
            self.assertLess(np.linalg.norm(delta[3:]),5e-5)
        self.assertLess(post._SamplingDiagnostics(coarse)['max_rho_step'],0.01005)

    def test_invalid_sampling_config_is_rejected(self):
        post = self._linear_post([SE3(),SE3.Tx(0.1)])
        for name,invalid_values in (
            ('presample_count',(True,2,3.5)),
            ('sampling_rho_interval',(True,0,-1,np.nan,np.inf)),
            ('sampling_phi_interval',(True,0,-1,np.nan,np.inf)),
        ):
            original = getattr(post.opt_cfg,name)
            for invalid in invalid_values:
                with self.subTest(name=name,value=invalid):
                    setattr(post.opt_cfg,name,invalid)
                    with self.assertRaises(ValueError):
                        post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
            setattr(post.opt_cfg,name,original)

    def test_cost_acceleration_and_validation_share_fixed_grid(self):
        post = self._linear_post([SE3(),SE3.Tx(0.2)])
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        initial_poses = post._SampleTrajectory(post.se3_ctrl_poses,grid)[1]
        delta = np.zeros((len(post.opt_poses_index),6))
        delta[:,2] = 0.02
        updated = post.UpdateCtrlPoses(delta,update_original=False)
        cost_poses,check_poses = [],[]
        def cost(pose,points_idx=None):
            cost_poses.append(pose.A.copy())
            return 2.0
        def check(pose):
            check_poses.append(pose.A.copy())
            return None,None,np.ones(1,dtype=bool),np.ones((6,3,2),dtype=bool)
        post.hex_state = SimpleNamespace(RobotFeasiCost=cost,RobotFeasiCheck=check)
        indices = [np.array([0]) for _ in grid.times]
        acceleration = post._AccelerationMeanCost
        with patch.object(post,'_BuildTrajectorySampleGrid',side_effect=AssertionError('grid rebuilt')):
            with patch.object(post,'_AccelerationMeanCost',wraps=acceleration) as observed:
                _,feasibility,_ = post._TrajectoryMeanCost(updated,indices,grid)
                valid,_ = post._ValidateTrajectory(updated,grid)
                self.assertIs(observed.call_args.args[1],grid.times)
        self.assertEqual(feasibility,2.0)
        self.assertTrue(valid)
        np.testing.assert_array_equal(cost_poses,check_poses)
        self.assertFalse(np.array_equal(cost_poses,[p.A for p in initial_poses]))
        with self.assertRaises(ValueError):
            grid.u_values[0] = 1.0

    def test_validation_rejects_collision_single_leg_and_speed_failures(self):
        post = self._linear_post([SE3(),SE3.Tx(0.1)])
        for body_ok,count,expected in ((True,3,True),(False,3,False),(True,2,False)):
            leg_mask = np.ones((6,3,2),dtype=bool)
            leg_mask[0,count:] = False
            post.hex_state = SimpleNamespace(RobotFeasiCheck=lambda pose:(
                None,None,np.array([body_ok]),leg_mask))
            self.assertEqual(bool(post._ValidateTrajectory(post.se3_ctrl_poses)[0]),expected)
        post.hex_state = _CMAHexState()
        post.se3_segment_times = [0.01]
        self.assertFalse(post._ValidateTrajectory(post.se3_ctrl_poses)[0])

    def test_shortcut_checks_endpoints_even_for_short_edges(self):
        post = self._post_process_without_files()
        post.se3_path = [SE3(),SE3.Tx(0.005)]
        post.hex_state = SimpleNamespace(RobotFeasiCheck=lambda pose:(
            None,None,np.array([pose.t[0]<0.004]),np.ones((6,3,2),dtype=bool)))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'cannot connect'):
                post.ShortCutPath()

    def test_json_exports_the_exact_validated_grid(self):
        from legged_gym.expert_complex_utils.se3_path_vis import load_se3_path_json
        post = self._linear_post([SE3(),SE3.Tx(0.025),SE3.Tx(0.055)])
        grid = post._BuildTrajectorySampleGrid(post.se3_ctrl_poses)
        checked = []
        def check(pose):
            checked.append(pose.A.copy())
            return None,None,np.ones(1,dtype=bool),np.ones((6,3,2),dtype=bool)
        post.hex_state = SimpleNamespace(RobotFeasiCheck=check)
        self.assertTrue(post._ValidateTrajectory(post.se3_ctrl_poses,grid)[0])
        with tempfile.TemporaryDirectory() as root:
            with contextlib.redirect_stdout(io.StringIO()):
                file = post.WriteJson(str(Path(root)/'path.json'),grid)
            path = load_se3_path_json(file)
            np.testing.assert_array_equal(path.dense.time,grid.times)
            np.testing.assert_allclose([p.A for p in path.dense.poses],checked,atol=1e-12)

    def test_final_selection_rechecks_older_snapshots_with_new_grids(self):
        post = self._linear_post([SE3(),SE3.Tx(0.1)])
        initial = post.se3_ctrl_poses.copy()
        snapshots = []
        for height in (0.005,0.008,0.01):
            delta = np.zeros((len(post.opt_poses_index),6))
            delta[:,2] = height
            snapshots.append(post.UpdateCtrlPoses(delta,update_original=False))
        post.se3_ctrl_poses = snapshots[-1]
        observed = []
        def validate(poses,grid=None):
            self.assertIsInstance(grid,TrajectorySampleGrid)
            observed.append(poses[1].t[2])
            return poses[1].t[2]<0.006,{'z':poses[1].t[2]}
        post._ValidateTrajectory = validate
        with contextlib.redirect_stdout(io.StringIO()):
            grid,diagnostics,rolled_back = post._SelectFinalTrajectory(initial,snapshots,'test')
        self.assertTrue(rolled_back)
        np.testing.assert_allclose(observed,[0.01,0.008,0.005])
        self.assertEqual(diagnostics['z'],0.005)
        self.assertEqual(post.se3_ctrl_poses[1].t[2],0.005)
        self.assertEqual(grid.times[-1],sum(post.se3_segment_times))



class _CMAHexState:
    """无需体素地图，提供高度目标和确定性的硬可行性检查。"""
    def __init__(self):
        self.cost_call_count = 0

    def GetRobotFeasiCostPoints(self,pose):
        return np.array([0],dtype=np.int64)

    def RobotFeasiCost(self,pose,points_idx=None):
        self.cost_call_count += 1
        return float(((pose.t[2]-0.015)/0.02)**2)

    def RobotFeasiCheck(self,pose):
        return None,None,np.ones(1,dtype=bool),np.ones((6,3,2),dtype=bool)


class SE3CMAOptimizationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        root_patch = patch(
            "legged_gym.expert_complex_utils.se3_post_process.LEGGED_GYM_ROOT_DIR",
            directory.name,
        )
        root_patch.start()
        self.addCleanup(root_patch.stop)

    @staticmethod
    def _post():
        post = SE3PostProcessNumericalTest._post_process_without_files()
        post.hex_state = _CMAHexState()
        cfg = post.opt_cfg
        cfg.sampling_rho_interval = 0.05
        cfg.sampling_phi_interval = 0.1
        cfg.presample_count = 21
        cfg.delt_rho_limit = 0.02
        cfg.delt_fai_limit = 0.1
        cfg.acceleration_weight = 0.0
        cfg.cost_progress_interval = 0
        cfg.cost_abs_change_tol = 0.0
        cfg.cost_rel_change_tol = 0.0
        cfg.max_retraction_iterations = 1
        cfg.cma_population_size = 8
        cfg.cma_max_generations = 5
        return post

    @staticmethod
    def _run(post):
        with contextlib.redirect_stdout(io.StringIO()):
            return post.Optimize_CMA_ES()

    @staticmethod
    def _scripted_factory(generations,instances):
        """每代四个确定性高度扰动，隔离回滚和异常处理与随机采样。"""
        def factory(x0,sigma,options):
            batches = []
            for heights in generations:
                batch = []
                for height in heights:
                    solution = np.zeros_like(x0)
                    solution[2::6] = height
                    batch.append(solution)
                batches.append(batch)
            strategy = SimpleNamespace(
                popsize=4,ask=Mock(side_effect=batches),tell=Mock(),stop=lambda: {},
            )
            instances.append(strategy)
            return strategy
        return factory

    def _scripted_post(self,generations):
        post = self._post()
        post.opt_cfg.cma_population_size = 4
        post.opt_cfg.cma_max_generations = len(generations)
        post._TrajectoryMeanCost = lambda poses,indices,grid: (
            1.0-poses[1].t[2],1.0-poses[1].t[2],0.0
        )
        instances = []
        strategy_patch = patch(
            "cma.CMAEvolutionStrategy",
            side_effect=self._scripted_factory(generations,instances),
        )
        strategy_patch.start()
        self.addCleanup(strategy_patch.stop)
        return post,instances

    def test_real_cma_improves_cost_and_exports_matching_result(self):
        post = self._post()
        post.GetCtrlPoes()
        initial = [pose.A.copy() for pose in post.se3_ctrl_poses]
        result = self._run(post)
        self.assertTrue(result.success)
        self.assertLess(result.fun,(0.015/0.02)**2)
        np.testing.assert_allclose(post.se3_ctrl_poses[0].A,initial[0],atol=1e-12)
        np.testing.assert_allclose(post.se3_ctrl_poses[-1].A,initial[-1],atol=1e-12)
        for index,delta in zip(post.opt_poses_index,result.x.reshape(-1,6)):
            np.testing.assert_allclose(
                (SE3(initial[index])*SE3.Exp(delta)).A,
                post.se3_ctrl_poses[index].A,atol=1e-12,
            )
        initial_grid = post._BuildTrajectorySampleGrid([SE3(pose) for pose in initial])
        self.assertEqual(post.hex_state.cost_call_count,
                         (result.nfev-1)*len(initial_grid.times)+result.validation["sample_count"])
        self.assertEqual(result.nfev,1+8*result.nit+1)
        self.assertTrue(result.output_path.name.startswith("optimized_cma_es_path_"))
        with result.output_path.open() as file:
            document = json.load(file)
        for segment_index,segment in enumerate(document["bezier_segments"]):
            block = segment["control_poses"]
            for offset,(t,ang,vec) in enumerate(zip(block["t"],block["ang"],block["vec"])):
                np.testing.assert_allclose(
                    (SE3.Trans(t)*SE3.AngVec(ang,vec)).A,
                    post.se3_ctrl_poses[4*segment_index+offset].A,atol=1e-12,
                )
        self.assertEqual(result.validation["sample_count"],
                         len(post._BuildTrajectorySampleGrid(post.se3_ctrl_poses).times))
        _,poses = post._SampleTrajectory(post.se3_ctrl_poses)
        indices = [post.hex_state.GetRobotFeasiCostPoints(pose) for pose in poses]
        self.assertAlmostEqual(result.fun,post._TrajectoryMeanCost(post.se3_ctrl_poses,indices,post._BuildTrajectorySampleGrid(post.se3_ctrl_poses))[0])

    def test_seed_reproduces_samples_without_changing_global_rng(self):
        post1,post2 = self._post(),self._post()
        post1.opt_cfg.cma_seed = post2.opt_cfg.cma_seed = 0
        random_state = np.random.get_state()
        result1 = self._run(post1)
        result2 = self._run(post2)
        np.testing.assert_array_equal(result1.x,result2.x)
        self.assertEqual(result1.fun,result2.fun)
        after = np.random.get_state()
        self.assertEqual(random_state[0],after[0])
        np.testing.assert_array_equal(random_state[1],after[1])
        self.assertEqual(random_state[2:],after[2:])

    def test_sampling_obeys_bounds_and_keeps_base_and_candidates_fixed(self):
        post = self._post()
        update = post.UpdateCtrlPoses
        base_snapshots = []
        candidate_collections = []
        sample_grids = []
        cost = post._TrajectoryMeanCost

        def checked_update(delta,update_original=True):
            if not update_original:
                self.assertTrue((np.abs(delta[:,:3])<=post.opt_cfg.delt_rho_limit+1e-14).all())
                self.assertTrue((np.abs(delta[:,3:])<=post.opt_cfg.delt_fai_limit+1e-14).all())
                base_snapshots.append(np.stack([pose.A for pose in post.se3_ctrl_poses]))
            return update(delta,update_original=update_original)

        def checked_cost(ctrl,indices,grid):
            candidate_collections.append(indices)
            sample_grids.append(grid)
            return cost(ctrl,indices,grid)

        post.UpdateCtrlPoses = checked_update
        post._TrajectoryMeanCost = checked_cost
        self._run(post)
        self.assertGreater(len(base_snapshots),8)
        for snapshot in base_snapshots[1:]:
            np.testing.assert_array_equal(snapshot,base_snapshots[0])
        for indices in candidate_collections[:-1]:
            self.assertIs(indices,candidate_collections[0])
        self.assertIsNot(candidate_collections[-1],candidate_collections[0])
        for grid in sample_grids[:-1]:
            self.assertIs(grid,sample_grids[0])
        self.assertIsNot(sample_grids[-1],sample_grids[0])

    def test_configured_generations_are_not_truncated_by_evaluation_count(self):
        generations = [[0.01*(i+1)]*4 for i in range(70)]
        post,instances = self._scripted_post(generations)
        post._ValidateTrajectory = lambda poses,sample_grid=None: (True,{})
        result = self._run(post)
        self.assertEqual(result.nit,70)
        self.assertEqual(result.nfev,282)  # 基准1 + 70代各4次 + 最终评估1。
        self.assertEqual(instances[0].tell.call_count,70)

    def test_both_waypoint_modes_preserve_joins_and_velocity(self):
        for optimize_all in (False,True):
            with self.subTest(optimize_all=optimize_all):
                post = self._post()
                post.opt_cfg.optimize_all = optimize_all
                start,end = post.se3_path_short[0].A.copy(),post.se3_path_short[-1].A.copy()
                self._run(post)
                np.testing.assert_allclose(post.se3_ctrl_poses[0].A,start,atol=1e-12)
                np.testing.assert_allclose(post.se3_ctrl_poses[-1].A,end,atol=1e-12)
                junction = post.se3_ctrl_poses[4]
                np.testing.assert_allclose(junction.A,post.se3_ctrl_poses[3].A,atol=1e-12)
                if not optimize_all:
                    np.testing.assert_allclose(junction.A,post.se3_path_short[1].A,atol=1e-12)
                left_velocity = np.asarray(trlog(
                    (post.se3_ctrl_poses[2].inv()*junction).A,twist=True,check=False,
                ))*3/post.se3_segment_times[0]
                right_velocity = np.asarray(trlog(
                    (junction.inv()*post.se3_ctrl_poses[5]).A,twist=True,check=False,
                ))*3/post.se3_segment_times[1]
                np.testing.assert_allclose(left_velocity,right_velocity,atol=1e-10)

    def test_rollback_keeps_last_feasible_generation_and_result(self):
        post,_ = self._scripted_post([[0.1,0.15,0.2,0.25],[0.3,0.4,0.45,0.5]])
        post._ValidateTrajectory = lambda poses,sample_grid=None: (poses[1].t[2]<=0.007,{"z":float(poses[1].t[2])})
        result = self._run(post)
        self.assertTrue(result.rolled_back)
        self.assertTrue(result.success)
        self.assertAlmostEqual(post.se3_ctrl_poses[1].t[2],0.005)
        self.assertAlmostEqual(result.x[2],0.005)
        self.assertAlmostEqual(result.fun,0.995)
        self.assertAlmostEqual(result.validation["z"],0.005)
        self.assertIn("rolled back",result.message)

    def test_no_feasible_result_restores_initial_controls_without_export(self):
        post,_ = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post.GetCtrlPoes()
        initial = np.stack([pose.A for pose in post.se3_ctrl_poses])
        post._ValidateTrajectory = lambda poses,sample_grid=None: (False,{"invalid":True})
        with self.assertRaisesRegex(RuntimeError,"no hard-feasible trajectory"):
            self._run(post)
        np.testing.assert_array_equal(np.stack([pose.A for pose in post.se3_ctrl_poses]),initial)
        self.assertEqual(list(self.root.rglob("*.json")),[])

    def test_nonfinite_candidates_keep_baseline_and_do_not_call_tell(self):
        post,instances = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post._TrajectoryMeanCost = lambda poses,indices,grid: (
            (np.nan if poses[1].t[2]>0 else 1.0),0.0,0.0
        )
        result = self._run(post)
        self.assertEqual(result.fun,1.0)
        np.testing.assert_allclose(result.x,0.0,atol=1e-12)
        self.assertEqual(result.nfev,6)
        self.assertIn("non-finite",result.message)
        instances[0].tell.assert_not_called()

    def test_mixed_finite_and_nonfinite_costs_are_passed_in_order(self):
        post,instances = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post._TrajectoryMeanCost = lambda poses,indices,grid: (
            np.nan if poses[1].t[2]>0.0045 else 1.0-poses[1].t[2],0.0,0.0
        )
        result = self._run(post)
        costs = instances[0].tell.call_args.args[1]
        np.testing.assert_allclose(costs[:3],[0.998,0.997,0.996])
        self.assertEqual(costs[3],np.inf)
        self.assertAlmostEqual(result.fun,0.996)

    def test_nonfinite_robot_cost_is_handled_by_existing_integrator(self):
        post,instances = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post._TrajectoryMeanCost = lambda ctrl,indices,grid: PostProcess._TrajectoryMeanCost(
            post,ctrl,indices,grid
        )
        post.hex_state.RobotFeasiCost = lambda pose,points_idx=None: (
            np.nan if pose.t[2]>0.0 else 1.0
        )
        result = self._run(post)
        self.assertEqual(result.fun,1.0)
        self.assertIn("non-finite",result.message)
        instances[0].tell.assert_not_called()

    def test_no_improvement_waits_for_retraction_patience(self):
        post,instances = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post.opt_cfg.max_retraction_iterations = 10
        post._TrajectoryMeanCost = lambda poses,indices,grid: (1.0,1.0,0.0)
        result = self._run(post)
        self.assertEqual(len(instances),3)
        self.assertEqual(result.retraction_iterations,3)
        self.assertEqual(result.nit,3)
        self.assertEqual(result.nfev,16)
        self.assertEqual(result.status,0)
        np.testing.assert_allclose(result.x,0.0,atol=1e-12)

    def test_cumulative_result_spans_multiple_retractions(self):
        post,instances = self._scripted_post([[0.1,0.3,0.5,0.75]])
        post.opt_cfg.max_retraction_iterations = 3
        result = self._run(post)
        self.assertEqual(len(instances),3)
        self.assertEqual(result.nfev,16)
        self.assertAlmostEqual(post.se3_ctrl_poses[1].t[2],0.045)
        self.assertAlmostEqual(result.x[2],0.045)
        self.assertGreater(result.x[2],post.opt_cfg.delt_rho_limit)
        self.assertAlmostEqual(result.fun,0.955)

    def test_nonfinite_baseline_can_recover_to_finite_candidate(self):
        post,_ = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post._TrajectoryMeanCost = lambda poses,indices,grid: (
            np.inf if poses[1].t[2]==0 else 1.0-poses[1].t[2],0.0,0.0
        )
        result = self._run(post)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.fun,0.995)

    def test_nonfinite_final_cost_restores_initial_controls_without_export(self):
        post,_ = self._scripted_post([[0.1,0.15,0.2,0.25]])
        post.GetCtrlPoes()
        initial = np.stack([pose.A for pose in post.se3_ctrl_poses])
        post._TrajectoryMeanCost = Mock(side_effect=[
            (1.0,1.0,0.0),(0.99,0.99,0.0),(0.98,0.98,0.0),
            (0.97,0.97,0.0),(0.96,0.96,0.0),(np.nan,np.nan,0.0),
        ])
        with self.assertRaisesRegex(RuntimeError,"Final CMA-ES cost is non-finite"):
            self._run(post)
        np.testing.assert_array_equal(np.stack([pose.A for pose in post.se3_ctrl_poses]),initial)
        self.assertEqual(list(self.root.rglob("*.json")),[])

    def test_missing_cma_dependency_has_install_hint(self):
        post = self._post()
        with patch.dict("sys.modules",{"cma":None}):
            with self.assertRaisesRegex(ImportError,"pip install cma"):
                self._run(post)

    def test_automatic_population_and_random_seed_are_supported(self):
        post = self._post()
        post.opt_cfg.cma_population_size = None
        post.opt_cfg.cma_seed = None
        post.opt_cfg.cma_max_generations = 1
        population_size = int(4+3*np.log(24))
        result = self._run(post)
        self.assertTrue(result.success)
        self.assertEqual(result.nit,1)
        self.assertEqual(result.nfev,2+population_size)


class SE3LBFGSSamplingTest(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        context = patch('legged_gym.expert_complex_utils.se3_post_process.LEGGED_GYM_ROOT_DIR',root.name)
        context.start()
        self.addCleanup(context.stop)

    @staticmethod
    def _post():
        post = SE3PostProcessNumericalTest._linear_post([SE3(),SE3.Tx(0.2)])
        post.hex_state = _CMAHexState()
        post.opt_cfg.max_retraction_iterations = 1
        post.opt_cfg.cost_progress_interval = 0
        return post

    def test_lbfgs_cost_validation_and_export_share_grids(self):
        from scipy.optimize import OptimizeResult
        post = self._post()
        costs,validations,exports = [],[],[]
        cost = post._TrajectoryMeanCost
        validate = post._ValidateTrajectory
        write = post.WriteJson
        def observed_cost(poses,indices,grid):
            costs.append((indices,grid))
            return cost(poses,indices,grid)
        def observed_validation(poses,grid=None):
            validations.append(grid)
            return validate(poses,grid)
        def observed_export(file,sample_grid=None):
            exports.append(sample_grid)
            return write(file,sample_grid)
        def minimize(function,x0,**kwargs):
            function(x0)
            candidate = np.array(x0,copy=True).reshape(-1,6)
            candidate[:,2] = 0.008
            return OptimizeResult(x=candidate.ravel(),fun=function(candidate.ravel()),
                                  success=True,status=0,message='scripted')
        post._TrajectoryMeanCost = observed_cost
        post._ValidateTrajectory = observed_validation
        post.WriteJson = observed_export
        with patch('legged_gym.expert_complex_utils.se3_post_process.minimize',side_effect=minimize):
            with contextlib.redirect_stdout(io.StringIO()):
                result = post.Optimize()
        for indices,grid in costs[:-1]:
            self.assertIs(indices,costs[0][0])
            self.assertIs(grid,costs[0][1])
        self.assertIs(validations[1],costs[0][1])
        self.assertIs(validations[-1],costs[-1][1])
        self.assertIs(exports[0],costs[-1][1])
        self.assertIsNot(costs[-1][1],costs[0][1])
        self.assertTrue(result.output_path.exists())
        self.assertFalse(result.rolled_back)

    def test_real_lbfgs_improves_cost_and_exports_a_valid_trajectory(self):
        post = self._post()
        post.opt_cfg.sampling_rho_interval = 0.05
        post.opt_cfg.presample_count = 21
        post.opt_cfg.acceleration_weight = 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            result = post.Optimize()
        self.assertLess(result.fun,(0.015/0.02)**2)
        self.assertEqual(result.validation['invalid_sample_count'],0)
        self.assertTrue(result.output_path.exists())

    def test_new_grid_detects_collision_missed_by_frozen_grid_and_rolls_back(self):
        from scipy.optimize import OptimizeResult
        from legged_gym.expert_complex_utils.se3_path_vis import load_se3_path_json
        post = self._post()
        initial = [pose.copy() for pose in post.se3_ctrl_poses]
        base_grid = post._BuildTrajectorySampleGrid(initial)
        delta = np.zeros((len(post.opt_poses_index),6))
        delta[:,2] = 0.008
        candidate = post.UpdateCtrlPoses(delta,update_original=False)
        frozen_poses = post._SampleTrajectory(candidate,base_grid)[1]
        fresh_poses = post._SampleTrajectory(candidate)[1]
        distances = np.linalg.norm(
            np.asarray([p.t for p in fresh_poses])[:,None,:]
            -np.asarray([p.t for p in frozen_poses])[None,:,:],axis=-1).min(axis=1)
        index = int(np.argmax(distances))
        self.assertGreater(distances[index],1e-5)
        collision_point = fresh_poses[index].t.copy()
        radius = distances[index]/4
        post.hex_state.RobotFeasiCheck = lambda pose:(
            None,None,np.array([np.linalg.norm(pose.t-collision_point)>radius]),
            np.ones((6,3,2),dtype=bool))
        self.assertTrue(post._ValidateTrajectory(candidate,base_grid)[0])
        self.assertFalse(post._ValidateTrajectory(candidate)[0])
        def minimize(function,x0,**kwargs):
            return OptimizeResult(x=delta.ravel(),fun=function(delta.ravel()),
                                  success=True,status=0,message='scripted')
        with patch('legged_gym.expert_complex_utils.se3_post_process.minimize',side_effect=minimize):
            with contextlib.redirect_stdout(io.StringIO()):
                result = post.Optimize()
        self.assertTrue(result.rolled_back)
        np.testing.assert_array_equal([p.A for p in post.se3_ctrl_poses],[p.A for p in initial])
        exported = load_se3_path_json(result.output_path)
        self.assertTrue(all(pose.t[2]==0 for pose in exported.dense.poses))
        self.assertAlmostEqual(result.fun,(0.015/0.02)**2)

if __name__=="__main__":
    unittest.main()
