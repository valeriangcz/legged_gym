
# Author: Mark Moll, Weihang Guo

from ompl import base as ob
from ompl import geometric as og
from legged_gym.expert_complex_utils import HexState, Kinematic
from spatialmath import SE3
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
import pyvista as pv
import time


def _Normalize(v:np.ndarray,eps:float=1e-8):
    v = np.asarray(v,dtype=np.float64)
    if not np.isfinite(v).all():
        return None
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < eps:
        return None
    return v / norm


def _SE3Distance(a:SE3,b:SE3,rot_weight:float=0.15)->float:
    rot_angle = Rotation.from_matrix(a.R.T @ b.R).magnitude()
    return float(np.linalg.norm(a.t-b.t) + rot_weight*rot_angle)


def _MakeFrameFromZ(z_axis:np.ndarray,yaw:float)->np.ndarray:
    z_axis = _Normalize(z_axis)
    ref = np.array([1.0,0.0,0.0]) if abs(float(z_axis[0])) < 0.9 else np.array([0.0,1.0,0.0])
    x_axis = _Normalize(np.cross(ref,z_axis))
    y_axis = np.cross(z_axis,x_axis)
    R = np.column_stack([x_axis,y_axis,z_axis]) @ Rotation.from_euler("z",yaw).as_matrix()
    return Rotation.from_matrix(R).as_matrix()


def _InterpolateSE3(a:SE3,b:SE3,alpha:float)->SE3:
    pos = (1.0-alpha)*a.t + alpha*b.t
    rots = Rotation.from_matrix(np.stack([a.R,b.R],axis=0))
    R = Slerp([0.0,1.0],rots)([alpha]).as_matrix()[0]
    return SE3.Rt(R,pos)

def StateToSE3(state:ob.SE3State)->SE3:
        q = state.rotation()
        quat = [q.x,q.y,q.z,q.w]
        R_mat = Rotation.from_quat(quat).as_matrix()
        return SE3.Rt(R_mat,[state.getX(),state.getY(),state.getZ()])        
    


def SE3ToState(se3:SE3,state:ob.SE3State):
    """state 是ompl通过 allocState() 构造，因此需要传入"""
    state.setXYZ(se3.x,se3.y,se3.z)
    quat = Rotation.from_matrix(se3.R).as_quat()
    q = state.rotation()
    q.x = quat[0]
    q.y = quat[1]
    q.z = quat[2]
    q.w = quat[3]



class HexStateValidChecker(ob.StateValidityChecker):
    def __init__(self, hex_state:HexState,si:ob.SpaceInformation):
        super().__init__(si)
        self.hex_state = hex_state
        self.si = si
    def isValid(self, state:ob.SE3State)->bool:
        se3 = StateToSE3(state)
    #搜索距离最近的环境点 超过一定距离或者太靠近都为无效
        _,idx = self.hex_state.env_pointsmap_voxels._tree.query(se3.t)
        env_point = self.hex_state.env_pointsmap_voxels.points[idx]
        env_distance = np.linalg.norm(env_point-se3.t)
        if env_distance<0.1 or env_distance>0.57:
            return False
        
        #N;             6,M,2
        _,_,body_cf_mask,leg_feasi_mask = self.hex_state.RobotFeasiCheck(se3)
        if (~body_cf_mask).any():
            return False
        #6
        points_num_in_legs = np.sum(leg_feasi_mask.any(axis=-1),axis=1)
        if (points_num_in_legs>=3).all():
            return True
        return False



class HexPlanning:
    def __init__(self):
        self.space = ob.SE3StateSpace()
        self._bounds = ob.RealVectorBounds(3)
        self.hex_state = HexState(Kinematic())
        for i in range(3):
            self._bounds.setLow(i,self.hex_state.env_pointsmap_voxels._bounds[i,0])
            self._bounds.setHigh(i,self.hex_state.env_pointsmap_voxels._bounds[i,1])
        self.space.setBounds(self._bounds)
        self.si = ob.SpaceInformation(self.space)
        self.checker = HexStateValidChecker(self.hex_state,self.si)       
        self.si.setStateValidityChecker(self.checker)
        self.si.setup()
        self.start:ob.SE3State = self.si.getStateSpace().allocState()    
        self.goal:ob.SE3State = self.si.getStateSpace().allocState()
        self.pdef = ob.ProblemDefinition(self.si)

    def RRTconnect(self,start_se3:SE3,goal_se3:SE3)->ob.Path:
        """使用 ompl 的RRT connect 算法搜索路径"""
        self.pdef.clearStartStates()
        self.pdef.clearGoal()
        self.pdef.clearSolutionPaths()

        SE3ToState(start_se3,self.start)
        SE3ToState(goal_se3,self.goal)
        self.pdef.setStartAndGoalStates(self.start,self.goal)
        planner = og.RRTConnect(self.si)
        planner.setProblemDefinition(self.pdef)
        planner.setup()
        self.si.printSettings()
        print(f"------problem def-----------\n{self.pdef}")
        solved = planner.solve(10.0)
        if solved:
            path = self.pdef.getSolutionPath()
            print("path find\n",path)
            return path
        return None

    def _PoseValid(self,W_T_R:SE3,
                   min_surface_distance:float=0.1,
                   max_surface_distance:float=0.57,
                   min_landing_points_per_leg:int=3):
        pointmap = self.hex_state.env_pointsmap_voxels
        env_distance,idx = pointmap._tree.query(W_T_R.t)
        if env_distance < min_surface_distance or env_distance > max_surface_distance:
            return False,env_distance,np.zeros(6,dtype=np.int32)

        _,_,body_cf_mask,leg_feasi_mask = self.hex_state.RobotFeasiCheck(W_T_R)
        if (~body_cf_mask).any():
            return False,env_distance,np.zeros(6,dtype=np.int32)
        landing_counts = np.sum(leg_feasi_mask.any(axis=-1),axis=1).astype(np.int32)
        return bool((landing_counts>=min_landing_points_per_leg).all()),env_distance,landing_counts

    def _SampleProjectedSE3(self,rng:np.random.Generator,
                            max_sample_attempts:int=200,
                            min_surface_distance:float=0.1,
                            max_surface_distance:float=0.57,
                            min_landing_points_per_leg:int=3,
                            edge_probability:float=0.5):
        pointmap = self.hex_state.env_pointsmap_voxels
        z_offsets = (0.14,0.18,0.22,0.26,0.34,0.46)
        yaw_values = np.deg2rad((0,60,120,180,240,300))

        for _ in range(max_sample_attempts):
            pos = rng.uniform(pointmap._bounds[:,0],pointmap._bounds[:,1])
            raw = SE3.Rt(Rotation.random(random_state=rng).as_matrix(),pos)
            env_distance,idx = pointmap._tree.query(pos)

            if env_distance < min_surface_distance:
                continue
            if env_distance <= max_surface_distance:
                valid,_,_ = self._PoseValid(raw,min_surface_distance,max_surface_distance,min_landing_points_per_leg)
                if valid:
                    return raw
                continue

            modes = ["edge","surface"] if rng.random() < edge_probability else ["surface","edge"]
            for mode in modes:
                if mode == "edge":
                    if getattr(pointmap,"_edge_tree",None) is None or pointmap.edge_points.shape[0] == 0:
                        continue
                    _,edge_idx = pointmap._edge_tree.query(pos)
                    anchor = pointmap.edge_points[edge_idx]
                    z_axis = _Normalize(pos-anchor)
                else:
                    anchor = pointmap.points[idx]
                    z_axis = _Normalize(pointmap.normals[idx])
                    if z_axis is not None and np.dot(z_axis,pos-anchor) < 0.0:
                        z_axis = -z_axis
                if z_axis is None:
                    continue

                candidates = []
                for z_offset in z_offsets:
                    center = anchor + z_offset*z_axis
                    for yaw in yaw_values:
                        q = SE3.Rt(_MakeFrameFromZ(z_axis,yaw),center)
                        valid,_,counts = self._PoseValid(
                            q,min_surface_distance,max_surface_distance,min_landing_points_per_leg
                        )
                        if valid:
                            candidates.append((int(counts.min()),int(counts.sum()),-np.linalg.norm(center-pos),q))
                if len(candidates) > 0:
                    candidates.sort(reverse=True,key=lambda item:(item[0],item[1],item[2]))
                    return candidates[0][3]
        return None

    def RRTconnectProjected(self,start_se3:SE3,goal_se3:SE3,
                            solve_time:float=10.0,
                            extend_range:float|None=None,
                            edge_check_resolution:float|None=None,
                            rot_weight:float=0.15,
                            seed:int|None=None):
        """自写RRT-Connect，采样阶段使用表面/棱投影。"""
        pointmap = self.hex_state.env_pointsmap_voxels
        voxel_scale = float(getattr(pointmap.voxels,"voxel_scale",0.04))
        extend_range = 2.0*voxel_scale if extend_range is None else float(extend_range)
        edge_check_resolution = 0.5*voxel_scale if edge_check_resolution is None else float(edge_check_resolution)
        rng = np.random.default_rng(seed)

        if not self._PoseValid(start_se3)[0]:
            print("projected RRTConnect failed: start pose is invalid")
            return None
        if not self._PoseValid(goal_se3)[0]:
            print("projected RRTConnect failed: goal pose is invalid")
            return None

        def edge_valid(a:SE3,b:SE3)->bool:
            check_count = max(1,int(np.ceil(_SE3Distance(a,b,rot_weight)/edge_check_resolution)))
            for i in range(1,check_count+1):
                if not self._PoseValid(_InterpolateSE3(a,b,i/check_count))[0]:
                    return False
            return True

        def nearest(tree,target):
            distances = [_SE3Distance(node["pose"],target,rot_weight) for node in tree]
            return int(np.argmin(distances))

        def grow(tree,target):
            near_idx = nearest(tree,target)
            near = tree[near_idx]["pose"]
            dist = _SE3Distance(near,target,rot_weight)
            if dist < 1e-9:
                return "reached",near_idx
            new_pose = target if dist <= extend_range else _InterpolateSE3(near,target,extend_range/dist)
            if not edge_valid(near,new_pose):
                return "trapped",near_idx
            tree.append({"pose":new_pose,"parent":near_idx})
            return ("reached" if dist <= extend_range else "advanced"),len(tree)-1

        def connect(tree,target):
            last_idx = None
            while True:
                status,idx = grow(tree,target)
                if status == "trapped":
                    return status,last_idx
                last_idx = idx
                if status == "reached":
                    return status,idx

        def trace(tree,idx):
            path = []
            while idx is not None and idx >= 0:
                path.append(tree[idx]["pose"])
                idx = tree[idx]["parent"]
            path.reverse()
            return path

        trees = [
            [{"pose":start_se3,"parent":-1}],
            [{"pose":goal_se3,"parent":-1}],
        ]
        start_time = time.monotonic()
        active = 0
        sample_count = 0

        while time.monotonic()-start_time < solve_time:
            sample = self._SampleProjectedSE3(rng)
            sample_count += 1
            if sample is None:
                continue

            status,new_idx = grow(trees[active],sample)
            if status != "trapped":
                other = 1-active
                q_new = trees[active][new_idx]["pose"]
                connect_status,connect_idx = connect(trees[other],q_new)
                if connect_status == "reached":
                    if active == 0:
                        start_path = trace(trees[0],new_idx)
                        goal_path = trace(trees[1],connect_idx)
                    else:
                        start_path = trace(trees[0],connect_idx)
                        goal_path = trace(trees[1],new_idx)
                    path = start_path + list(reversed(goal_path[:-1]))
                    print(
                        f"projected RRTConnect path find: states={len(path)}, "
                        f"samples={sample_count}, start_tree={len(trees[0])}, goal_tree={len(trees[1])}"
                    )
                    return path
            active = 1-active

        print(
            f"projected RRTConnect failed: timeout, samples={sample_count}, "
            f"start_tree={len(trees[0])}, goal_tree={len(trees[1])}"
        )
        return None

    def VisRoboPath(self,
                    path,
                    env_stride:int=1,
                    path_stride:int=1,
                    leg_voxel_stride:int=16,
                    show_leg_workspace:bool=True,
                    show_feasible_landing:bool=True):
        """使用pyvista可视化机器人的SE3路径"""
        #将path 转为list[SE3]
        def _PathToSE3List(path):
            if isinstance(path,SE3):
                return [path]
            if isinstance(path,(list,tuple)):
                se3_path = []
                for item in path:
                    if isinstance(item,SE3):
                        se3_path.append(item)
                    else:
                        se3_path.append(StateToSE3(item))
                return se3_path
            if hasattr(path,"getStates"):
                return [StateToSE3(state) for state in path.getStates()]
            if hasattr(path,"states"):
                return [StateToSE3(state) for state in path.states]
            if hasattr(path,"getStateCount") and hasattr(path,"getState"):
                return [StateToSE3(path.getState(i)) for i in range(path.getStateCount())]
            raise TypeError("path must be an SE3, a sequence of SE3/OMPL states, or an OMPL PathGeometric")

        def _TransformRPointsToW(W_T_R:SE3,points_R:np.ndarray)->np.ndarray:
            if points_R.shape[0] == 0:
                return points_R.reshape(0,3)
            return (W_T_R*points_R.T).T

        def _AddPointCloud(plotter:pv.Plotter,points:np.ndarray,
                           color:str,point_size:float,
                           opacity:float=1.0,label:str|None=None):
            if points.shape[0] == 0:
                return
            plotter.add_mesh(
                pv.PolyData(points),
                render_points_as_spheres=True,
                point_size=point_size,
                color=color,
                opacity=opacity,
                label=label,
            )

        se3_path = _PathToSE3List(path)
        if len(se3_path) == 0:
            raise ValueError("path is empty")

        env_stride = max(1,int(env_stride))
        path_stride = max(1,int(path_stride))
        leg_voxel_stride = max(1,int(leg_voxel_stride))
        color_list = ["red","green","blue","yellow","magenta","cyan"]
        leg_names = self.hex_state.robot_voxels.leg_voxels._leg_names

        sampled_path = se3_path[::path_stride]
        if sampled_path[-1] is not se3_path[-1]:
            sampled_path.append(se3_path[-1])

        plotter = pv.Plotter()
        plotter.show_axes()

        #可视化env point map
        env_points = self.hex_state.env_pointsmap_voxels.points[::env_stride]
        landing_points = self.hex_state.env_pointsmap_voxels.landing_points[::env_stride]
        _AddPointCloud(plotter,env_points,"lightgray",3,0.16,"env points")
        _AddPointCloud(plotter,landing_points,"black",5,0.20,"landing points")

        path_points = np.vstack([se3.t.reshape(1,3) for se3 in se3_path])
        plotter.add_mesh(
            pv.lines_from_points(path_points),
            color="navy",
            line_width=5,
            label="body path",
        )
        _AddPointCloud(plotter,path_points,"navy",8,1.0,"path samples")

        body_occ_mask = self.hex_state.robot_voxels.body_voxels.esdf_flat <= 0.0
        body_points_R = self.hex_state.robot_voxels.body_voxels.center[body_occ_mask]

        #沿着path可视化 robovoxels，采用降采样绘制身体包络和腿部工作空间。
        for pose_index,W_T_R in enumerate(sampled_path):
            body_points_W = _TransformRPointsToW(W_T_R,body_points_R)
            _AddPointCloud(
                plotter,
                body_points_W,
                "orange",
                6,
                0.18 if pose_index not in (0,len(sampled_path)-1) else 0.45,
                "body voxels" if pose_index == 0 else None,
            )

            if show_leg_workspace:
                for leg_index in range(6):
                    reachable = self.hex_state.robot_voxels.robot_reachable_legs[...,leg_index].any(axis=1)
                    leg_points_B = self.hex_state.robot_voxels.leg_voxels.center[reachable][::leg_voxel_stride]
                    leg_points_R = self.hex_state.robot_voxels.kin._B2R(leg_points_B.T,leg_index).T
                    leg_points_W = _TransformRPointsToW(W_T_R,leg_points_R)
                    _AddPointCloud(
                        plotter,
                        leg_points_W,
                        color_list[leg_index],
                        3,
                        0.06,
                        f"{leg_names[leg_index]} workspace" if pose_index == 0 else None,
                    )

            #同时将此位姿下每条腿的可选落脚点使用彩色标注
            if show_feasible_landing:
                _,landing_idx,_,leg_point_feasi_mask = self.hex_state.RobotFeasiCheck(W_T_R)
                near_landing_points_W = self.hex_state.env_pointsmap_voxels.landing_points[landing_idx]
                for leg_index in range(6):
                    feasible_mask = leg_point_feasi_mask[leg_index].any(axis=1)
                    _AddPointCloud(
                        plotter,
                        near_landing_points_W[feasible_mask],
                        color_list[leg_index],
                        10,
                        0.70,
                        f"{leg_names[leg_index]} feasible landing" if pose_index == 0 else None,
                    )

        #显示
        plotter.add_legend()
        plotter.show()


if __name__ == '__main__':
    hex_planning = HexPlanning()
    # path = hex_planning.RRTconnect(SE3(0.37,0.7,0.9),SE3(1.38,4.2,0.15))
    # path = hex_planning.RRTconnect(SE3(0.37,0.7,0.9),SE3(0.38,4.2,0.15))
    path = hex_planning.RRTconnectProjected(SE3(0.37,0.7,0.9),SE3(0.38,4.2,0.15))
    print("path=",path)
    # path = hex_planning.RRTconnect(SE3(1.38, 4.2, 0.15), SE3(1.0, 3.2, 0.15))
    hex_planning.VisRoboPath(path)
