import unittest

import numpy as np
from spatialmath import SE3

from legged_gym.expert_complex_utils.env_robot_voxels import Voxels
from legged_gym.expert_complex_utils.se3_post_process import OptCfg,PostProcess


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
        times,poses = post._SampleTrajectory(post.se3_ctrl_poses,21)
        self.assertLess(post._AccelerationMeanCost(poses,times),1e-20)

    def test_each_segment_has_fixed_sample_count_and_one_shared_join(self):
        post = self._post_process_without_files()
        post.GetCtrlPoes()
        samples_per_segment = 5
        times,poses = post._SampleTrajectory(
            post.se3_ctrl_poses,samples_per_segment
        )

        first_duration = post.se3_segment_times[0]
        total_duration = sum(post.se3_segment_times)
        self.assertNotAlmostEqual(
            post.se3_segment_times[0],post.se3_segment_times[1]
        )
        self.assertEqual(
            len(poses),
            post.se3_segment_nums*(samples_per_segment-1)+1,
        )
        self.assertEqual(np.count_nonzero(np.isclose(times,first_duration)),1)
        self.assertEqual(np.count_nonzero(times<=first_duration),samples_per_segment)
        self.assertEqual(np.count_nonzero(times>=first_duration),samples_per_segment)
        self.assertAlmostEqual(times[0],0.0)
        self.assertAlmostEqual(times[-1],total_duration)
        np.testing.assert_allclose(
            np.diff(times[:samples_per_segment]),
            first_duration/(samples_per_segment-1),
        )
        np.testing.assert_allclose(
            np.diff(times[samples_per_segment-1:]),
            post.se3_segment_times[1]/(samples_per_segment-1),
        )

    def test_fixed_sample_count_rejects_invalid_values(self):
        post = self._post_process_without_files()
        post.GetCtrlPoes()
        for invalid_value in (True,2,3.5):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    post._SampleTrajectory(
                        post.se3_ctrl_poses,invalid_value
                    )

    def test_feasibility_and_acceleration_use_independent_sampling(self):
        class CountingHexState:
            def __init__(self):
                self.cost_call_count = 0

            def RobotFeasiCost(
                self,pose,points_idx=None,smooth_temperature=0.05
            ):
                self.cost_call_count += 1
                return 2.0

        post = self._post_process_without_files()
        post.se3_path_short = post.se3_path_short[:2]
        post.hex_state = CountingHexState()
        post.opt_cfg.samples_per_segment = 5
        post.opt_cfg.acceleration_samples_per_segment = 21
        post.GetCtrlPoes()
        _,feasibility_poses = post._SampleTrajectory(
            post.se3_ctrl_poses,post.opt_cfg.samples_per_segment
        )
        candidate_indices = [np.array([0]) for _ in feasibility_poses]
        requested_sample_counts = []
        sample_trajectory = post._SampleTrajectory

        def recording_sample_trajectory(ctrl_poses,samples_per_segment):
            requested_sample_counts.append(samples_per_segment)
            return sample_trajectory(ctrl_poses,samples_per_segment)

        post._SampleTrajectory = recording_sample_trajectory
        _,feasibility_cost,acceleration_cost = post._TrajectoryMeanCost(
            post.se3_ctrl_poses,candidate_indices
        )
        self.assertEqual(
            post.hex_state.cost_call_count,len(feasibility_poses)
        )
        self.assertEqual(requested_sample_counts,[5,21])
        self.assertAlmostEqual(feasibility_cost,2.0)
        self.assertLess(acceleration_cost,1e-20)


if __name__=="__main__":
    unittest.main()
