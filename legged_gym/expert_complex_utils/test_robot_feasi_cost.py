import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial import cKDTree
from spatialmath import SE3

from legged_gym.expert_complex_utils.env_robot_voxels import HexState,Voxels


class RobotFeasiCostTest(unittest.TestCase):
    @staticmethod
    def _state(points):
        """Use real voxel lookup/interpolation with identical frames for six legs."""
        points = np.asarray(points,dtype=np.float64).reshape(-1,3)
        state = object.__new__(HexState)
        state.kin = SimpleNamespace(
            _R2B=lambda values,leg_index:values.copy(),
            _B2R=lambda values,leg_index:values.copy(),
            RVectorToLeg=lambda values,leg_index:values.copy(),
            InverseKin2MultiBatch=Mock(side_effect=AssertionError("online IK called")),
        )
        leg = Voxels(np.array([[-0.4,0.4],[-0.06,0.06],[-0.22,0.06]]),0.02)
        leg.x_b3 = np.broadcast_to([0.0,0.0,-1.0],(leg.grid_size,2,3)).copy()
        leg.ankle_pos = np.broadcast_to([0.15,0.0,-0.1],(leg.grid_size,2,3)).copy()
        leg.ankle_collide_radi = 0.045
        boundary_field = np.full(tuple(leg.grid_shape)+(6,),0.1)
        state.robot_voxels = SimpleNamespace(
            leg_voxels=leg,
            body_voxels=SimpleNamespace(
                bounding_points=np.zeros((0,3)),
                IsInsideRange=lambda points:np.zeros(points.shape[0],dtype=bool),
            ),
            kin=state.kin,
            robot_reachable_legs=np.ones((leg.grid_size,2,6),dtype=bool),
            to_bound_dist=boundary_field,
            to_bound_dist_flat=boundary_field.reshape(-1,6),
        )
        env = Voxels(np.array([[-1.0,1.0]]*3),0.1)
        normals = np.broadcast_to([0.0,0.0,1.0],points.shape).copy()
        state.env_pointsmap_voxels = SimpleNamespace(
            points=points.copy(),landing_points=points.copy(),normals=normals,
            landing_count=points.shape[0],voxels=env,
            env_esdf=np.full(tuple(env.grid_shape),0.1),
            _tree=cKDTree(points) if points.size else None,
            _landing_tree=cKDTree(points) if points.size else None,
        )
        return state

    @staticmethod
    def _points(count):
        return np.column_stack([
            0.12+0.02*np.arange(count),np.zeros(count),np.full(count,-0.1),
        ])

    @staticmethod
    def _cost(state,pose=None,indices=None,sol_index=1):
        if indices is None:
            indices = np.arange(state.env_pointsmap_voxels.landing_count)
        return state.RobotFeasiCost(
            SE3() if pose is None else pose,points_idx=indices,sol_index=sol_index,
        )

    @staticmethod
    def _empty_cost(min_esdf=0.1):
        # Voxels stores its bounds as float32, including these box extrema.
        x,y,z = np.array([0.4,0.06,0.22],dtype=np.float32).astype(np.float64)
        radius_max_cost = ((np.sqrt(x*x+y*y+z*z)-0.3)/0.03)**2
        return 36+radius_max_cost+(100/80)**2+25+(0.06-min(-0.06,min_esdf))**2

    def test_count_penalty_and_empty_candidate_worst_score(self):
        for count in (0,1,5,6,7):
            with self.subTest(count=count):
                state = self._state(self._points(count))
                expected = self._empty_cost() if count==0 else max(6-count,0)**2
                self.assertAlmostEqual(self._cost(state),expected,places=10)
                state.kin.InverseKin2MultiBatch.assert_not_called()

    def test_only_selected_branch_controls_scoring_and_count(self):
        state = self._state(self._points(6))
        state.robot_voxels.robot_reachable_legs[:,1,:] = False
        self.assertAlmostEqual(self._cost(state,sol_index=0),0.0)
        self.assertAlmostEqual(self._cost(state,sol_index=1),self._empty_cost())

        state.robot_voxels.robot_reachable_legs[:,1,:] = True
        state.robot_voxels.leg_voxels.x_b3[:,1] = [0.0,0.0,1.0]
        self.assertAlmostEqual(self._cost(state,sol_index=0),0.0)
        self.assertAlmostEqual(self._cost(state,sol_index=1),36+25+(100/80)**2)

        state.robot_voxels.leg_voxels.x_b3[:,1] = [0.0,0.0,-1.0]
        state.robot_voxels.leg_voxels.ankle_pos[:,1] = [2.0,0.0,0.0]
        self.assertAlmostEqual(self._cost(state,sol_index=0),0.0)
        self.assertAlmostEqual(self._cost(state,sol_index=1),36+0.12**2)

    def test_invalid_lookup_and_out_of_range_candidates_use_worst_score(self):
        for invalid_kind in ("xb3","ankle","outside"):
            with self.subTest(invalid_kind=invalid_kind):
                state = self._state(self._points(6))
                leg = state.robot_voxels.leg_voxels
                if invalid_kind=="xb3":
                    leg.x_b3[:,1] = np.nan
                elif invalid_kind=="ankle":
                    leg.ankle_pos[:,1] = np.nan
                else:
                    state.env_pointsmap_voxels.landing_points[:,0] = 2.0
                self.assertAlmostEqual(self._cost(state),self._empty_cost())

    def test_landing_offset_blocked_points_and_duplicate_indices(self):
        state = self._state(self._points(6))
        pointmap = state.env_pointsmap_voxels
        pointmap.points[:] = [2.0,0.0,0.0]
        pointmap.points = np.vstack([pointmap.points,[0.2,0.0,-0.1]])
        pointmap.normals = np.vstack([pointmap.normals,[0.0,0.0,1.0]])
        self.assertAlmostEqual(self._cost(state,indices=np.arange(7)),0.0)
        self.assertAlmostEqual(self._cost(state,indices=np.repeat(np.arange(5),2)),1.0)
        self.assertAlmostEqual(self._cost(state,indices=np.array([6])),self._empty_cost())

    def test_score_candidates_remain_when_hard_count_is_zero(self):
        state = self._state(self._points(6))
        state.robot_voxels.to_bound_dist_flat[:] = 0.03
        self.assertAlmostEqual(self._cost(state),36+(0.01/0.03)**2)

    def test_workspace_outer_radius_penalty(self):
        for radius in (0.05,0.10,0.20,0.30,0.35):
            with self.subTest(radius=radius):
                points = np.tile([radius,0.0,0.0],(6,1))
                state = self._state(points)
                leg = state.robot_voxels.leg_voxels
                # Isolate the radius formula from the hard check's center quantization.
                leg.center[leg.Pos2FlatIndex(points)] = points
                shell_cost = (max(0.0,radius-0.30)/0.03)**2
                count_cost = 36 if radius>0.30 else 0
                self.assertAlmostEqual(self._cost(state),shell_cost+count_cost)

    def test_workspace_boundary_penalty_inside_thirty_centimeters(self):
        for distance in (0.06,0.04,0.03,0.0,-0.01):
            with self.subTest(distance=distance):
                state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
                state.robot_voxels.to_bound_dist[:] = distance
                count_cost = 36 if distance<0.04 else 0
                self.assertAlmostEqual(self._cost(state),count_cost+(max(0.0,0.04-distance)/0.03)**2)

    def test_outer_radius_uses_radius_penalty_only(self):
        state = self._state(np.tile([0.35,0.0,0.0],(6,1)))
        state.robot_voxels.to_bound_dist[:] = -0.1
        self.assertAlmostEqual(self._cost(state),36+(0.05/0.03)**2)

    def test_workspace_boundary_interpolation_provides_direction_within_voxel(self):
        state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
        leg = state.robot_voxels.leg_voxels
        field = (0.01+0.1*leg.center[:,0]).reshape(leg.grid_shape)
        state.robot_voxels.to_bound_dist[:] = field[...,None]
        base = self._cost(state)
        farther_from_boundary = self._cost(state,pose=SE3.Tx(-0.001))
        nearer_boundary = self._cost(state,pose=SE3.Tx(0.001))
        self.assertAlmostEqual(base,36+(0.01/0.03)**2,delta=1e-6)
        self.assertAlmostEqual(farther_from_boundary,36+(0.0099/0.03)**2,delta=1e-6)
        self.assertAlmostEqual(nearer_boundary,36+(0.0101/0.03)**2,delta=1e-6)
        self.assertLess(farther_from_boundary,base)
        self.assertGreater(nearer_boundary,base)

    def test_hard_count_and_check_share_interpolated_boundary_distance(self):
        state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
        leg = state.robot_voxels.leg_voxels
        indices = np.arange(6)
        # The queried cell center is 0.01 m to the right of the actual point.
        field = (0.20-0.78*leg.center[:,0]).reshape(leg.grid_shape)
        state.robot_voxels.to_bound_dist[:] = field[...,None]
        points = state.env_pointsmap_voxels.landing_points
        flat = leg.Pos2FlatIndex(points)
        self.assertTrue((state.robot_voxels.to_bound_dist_flat[flat,0]<0.04).all())
        self.assertTrue((leg.TrilinearSample(field,points,outside_value=0.0)>0.04).all())
        state._LegNormFeasi = lambda points,normals,leg_index:np.ones((len(points),2),dtype=bool)
        _,_,_,mask = state.RobotFeasiCheck(SE3(),points_idx=indices)
        np.testing.assert_array_equal(mask[:,:,1].sum(axis=1),np.full(6,6))
        self.assertAlmostEqual(self._cost(state),0.0)

    def test_ankle_hard_check_uses_same_interpolation_as_cost(self):
        state = self._state(self._points(6))
        pointmap = state.env_pointsmap_voxels
        pointmap.env_esdf = (
            0.04+(0.017/0.03)*(pointmap.voxels.center[:,0]-0.35)
        ).reshape(pointmap.voxels.grid_shape)
        leg = state.robot_voxels.leg_voxels
        leg.ankle_pos[:,1] = [0.38,0.0,0.0]
        nearest_index = pointmap.voxels.Pos2GridIndex([0.38,0.0,0.0])
        nearest_distance = pointmap.env_esdf[tuple(nearest_index)]
        self.assertLess(nearest_distance,0.045)
        mask = state._LegEnvCollisionFree(pointmap.landing_points,SE3(),0,ignore_end=True)
        self.assertTrue(mask[:,1].all())
        self.assertAlmostEqual(self._cost(state),(0.06-0.057)**2,delta=1e-7)

    def test_empty_score_upper_bound_includes_boundary_penalty(self):
        state = self._state([])
        state.robot_voxels.to_bound_dist[:] = -0.4
        expected = 36+(0.44/0.03)**2+(100/80)**2+25+0.12**2
        self.assertAlmostEqual(self._cost(state),expected)

    def test_plane_score_and_ten_degree_xb3_transition(self):
        for angle in (0.0,79.0,80.0,85.0,90.0,94.0,96.0):
            with self.subTest(angle=angle):
                state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
                theta = np.deg2rad(angle)
                state.robot_voxels.leg_voxels.x_b3[:,1] = [np.sin(theta),0.0,-np.cos(theta)]
                plane_cost = 0 if angle<=80 else 12.5 if angle==85 else 25
                xb3_cost = (max(0.0,angle-80)/80)**2
                count_cost = 36 if angle>95 else 0
                self.assertAlmostEqual(self._cost(state),plane_cost+xb3_cost+count_cost)

        state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
        phi = np.deg2rad(18.0)
        normal = np.array([0.0,np.sin(phi),np.cos(phi)])
        state.env_pointsmap_voxels.normals[:] = normal
        state.robot_voxels.leg_voxels.x_b3[:,1] = -normal
        self.assertAlmostEqual(self._cost(state),(3/15)**2)

    def test_transition_is_continuous_at_eighty_and_ninety_degrees(self):
        state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
        for boundary in (80.0,90.0):
            costs = []
            for offset in (-1e-4,0.0,1e-4):
                theta = np.deg2rad(boundary+offset)
                state.robot_voxels.leg_voxels.x_b3[:,1] = [np.sin(theta),0.0,-np.cos(theta)]
                costs.append(self._cost(state))
            self.assertLess(max(costs)-min(costs),1e-6)

    def test_hard_normal_and_ankle_thresholds(self):
        for phi in (19.9,20.1):
            with self.subTest(plane=phi):
                state = self._state(np.tile([0.2,0.0,0.0],(6,1)))
                normal = np.array([0.0,np.sin(np.deg2rad(phi)),np.cos(np.deg2rad(phi))])
                state.env_pointsmap_voxels.normals[:] = normal
                state.robot_voxels.leg_voxels.x_b3[:,1] = -normal
                self.assertAlmostEqual(self._cost(state),((phi-15)/15)**2+(36 if phi>20 else 0))
        for distance in (0.0449,0.0451):
            with self.subTest(ankle_distance=distance):
                state = self._state(self._points(6))
                state.env_pointsmap_voxels.env_esdf[:] = distance
                self.assertAlmostEqual(self._cost(state),(0.06-distance)**2+(36 if distance<0.045 else 0))

    def test_ankle_score_uses_interpolated_environment_distance(self):
        for ankle_x in (0.5,0.3,-0.1):
            with self.subTest(ankle_x=ankle_x):
                state = self._state(self._points(6))
                pointmap = state.env_pointsmap_voxels
                pointmap.env_esdf = (0.1*pointmap.voxels.center[:,0]).reshape(pointmap.voxels.grid_shape)
                state.robot_voxels.leg_voxels.ankle_pos[:,1] = [ankle_x,0.0,0.0]
                distance = 0.1*ankle_x
                expected = (0.06-distance)**2+(36 if distance<0.045 else 0)
                self.assertAlmostEqual(self._cost(state),expected,places=8)

    def test_best_six_are_reselected_for_each_pose(self):
        points = np.column_stack([0.07+0.02*np.arange(7),np.zeros(7),np.zeros(7)])
        state = self._state(points)
        leg = state.robot_voxels.leg_voxels
        field = (0.03+0.1*np.abs(leg.center[:,0])).reshape(leg.grid_shape)
        state.robot_voxels.to_bound_dist[:] = field[...,None]
        indices = np.arange(7)
        # The worst point switches from index 0 to index 6 after translation.
        expected = 1+(0.001/0.03)**2/6
        self.assertAlmostEqual(self._cost(state,indices=indices),expected,delta=1e-6)
        self.assertAlmostEqual(self._cost(state,pose=SE3.Tx(0.26),indices=indices),expected,delta=1e-6)

    def test_fewer_than_six_score_candidates_use_actual_mean(self):
        state = self._state([[0.07,0.0,0.0],[0.13,0.0,0.0]])
        leg = state.robot_voxels.leg_voxels
        field = (0.02+0.2*np.abs(leg.center[:,0])).reshape(leg.grid_shape)
        state.robot_voxels.to_bound_dist[:] = field[...,None]
        self.assertAlmostEqual(self._cost(state),25+0.02,delta=1e-6)

    def test_degenerate_plane_and_invalid_normals_produce_finite_cost(self):
        state = self._state(np.tile([0.0,0.0,-0.2],(6,1)))
        self.assertAlmostEqual(self._cost(state),36+25)
        for normal in ([0.0,0.0,0.0],[np.nan,0.0,0.0]):
            state.env_pointsmap_voxels.normals[:] = normal
            self.assertAlmostEqual(self._cost(state),36+25+(100/80)**2)

    def test_body_collision_formula_is_unchanged(self):
        state = self._state(self._points(6))
        pointmap = state.env_pointsmap_voxels
        pointmap.env_esdf = (0.1*pointmap.voxels.center[:,0]).reshape(pointmap.voxels.grid_shape)
        base_cost = self._cost(state)
        state.robot_voxels.body_voxels.bounding_points = np.array([[0.5,0.0,0.0],[0.2,0.0,0.0]])
        self.assertAlmostEqual(self._cost(state)-base_cost,10*(0.0+0.5**2)/2,delta=1e-6)
        state.robot_voxels.body_voxels.bounding_points = np.array([[2.0,0.0,0.0]])
        self.assertAlmostEqual(self._cost(state)-base_cost,40.0)

    def test_empty_score_upper_bound_accounts_for_negative_esdf(self):
        state = self._state([])
        state.env_pointsmap_voxels.env_esdf[:] = -0.4
        self.assertAlmostEqual(self._cost(state),self._empty_cost(min_esdf=-0.4))

    def test_query_supplements_landing_points_before_candidates_are_fixed(self):
        state = self._state(self._points(6))
        state.env_pointsmap_voxels._tree = Mock()
        state.env_pointsmap_voxels._tree.query.return_value = (np.zeros(6),np.arange(6,12))
        indices = state.GetRobotFeasiCostPoints(SE3())
        np.testing.assert_array_equal(indices,np.arange(12))

        state = self._state([])
        np.testing.assert_array_equal(state.GetRobotFeasiCostPoints(SE3()),np.zeros(0,dtype=np.int64))

    def test_candidate_supplement_threshold_and_small_maps(self):
        for landing_count in (1,60,61,120):
            with self.subTest(landing_count=landing_count):
                state = self._state(np.tile([0.2,0.0,0.0],(landing_count,1)))
                pointmap = state.env_pointsmap_voxels
                pointmap._landing_tree = Mock(wraps=pointmap._landing_tree)
                indices = state.GetRobotFeasiCostPoints(SE3())
                np.testing.assert_array_equal(np.sort(indices),np.arange(landing_count))
                self.assertEqual(pointmap._landing_tree.query.call_count,int(landing_count<=60))

    def test_parameter_validation_is_preserved(self):
        state = self._state(self._points(6))
        for sol_index in (True,-1,2,0.5):
            with self.subTest(sol_index=sol_index):
                with self.assertRaises(ValueError):
                    self._cost(state,sol_index=sol_index)


if __name__=="__main__":
    unittest.main()
