#机器人的体素：身体体素与ESDF，腿部工作空间体素与对应的逆运动学解，还有末端关节位置
#环境的体素：环境的ESDF，环境点云筛选后的结果：落脚点与降采样环境点

from __future__ import annotations
from typing import Tuple,Union
from math import degrees
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from .hex_utils import Kinematic
from .point_map import PointMap
from spatialmath import SE3
import pyvista as pv
import os
from legged_gym import EXPERT_COMPLEX_DIR, LEGGED_GYM_ROOT_DIR
import trimesh


class Voxels:
    """保存体素共同特征：边界 scale 中心位置 索引转换 体素数量"""
    def __init__(self,
                 bounds:np.ndarray,
                 voxel_scale:float,
                 ):
        """构建规则体素网格。

        输入:
        - bounds: (3, 2)，每一行是 x/y/z 轴的 [min, max]。
        - voxel_scale: float，体素边长。

        生成:
        - grid_shape: (3,)，x/y/z 三个方向的体素数量。
        - grid_size: int，总体素数量。
        - grid_index: (grid_size, 3)，每个体素的三维网格索引。
        - center: (grid_size, 3)，每个体素中心点坐标。
        """
        
        self._bounds = bounds
        self.voxel_scale = voxel_scale
        if self._bounds.shape != (3, 2):
            raise ValueError("xyz_range must have shape (3,2)")
        if np.any(self._bounds[:, 1] <= self._bounds[:, 0]):
            raise ValueError("each axis range must satisfy max > min")
        if self.voxel_scale <= 0.0:
            raise ValueError("voxel_scale must be positive")        
        self._bounds = np.asarray(bounds, dtype=np.float32)
        self.grid_shape = self._compute_grid_shape()
        self.grid_size = int(np.prod(self.grid_shape))
        self.grid_index = np.zeros((self.grid_size, 3), dtype=np.int32)
        self.center = np.zeros((self.grid_size, 3), dtype=np.float32)

        nx, ny, nz = self.grid_shape
        ix, iy, iz = np.meshgrid(np.arange(nx),np.arange(ny),np.arange(nz),indexing='ij')
        self.grid_index = np.stack([ix,iy,iz],axis=-1).reshape(-1,3)
        # 3 + (grid_size,3)
        self.center = (self._bounds[None,:,0] + (self.grid_index+0.5)*self.voxel_scale).astype(np.float32)

    def _as_points(self, arr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """把单点或批量点统一成 (N, 3)。

        输入:
        - arr: (3,) 或 (N, 3)。

        返回:
        - points: (N, 3)。
        - squeeze: bool，输入是否为单点 (3,)。
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape == (3,):
            return arr[None, :], True
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr, False
        raise ValueError("input must have shape (3,) or (N, 3)")

    def IsInsideRange(self, pos: np.ndarray) -> np.ndarray | bool:
        """判断点是否位于体素地图边界内。

        输入:
        - pos: (3,) 或 (N, 3)，世界/局部坐标点。

        返回:
        - bool，输入为 (3,) 时返回单个布尔值。
        - inside: (N,)，输入为 (N, 3) 时返回每个点的判断结果。
        """
        pos, squeeze = self._as_points(pos)
        inside = ((pos >= self._bounds[None, :, 0]) & (pos < self._bounds[None, :, 1])).all(axis=1)
        return bool(inside[0]) if squeeze else inside

    def _compute_grid_shape(self) -> np.ndarray:
        """根据边界和体素边长计算网格尺寸。

        返回:
        - grid_shape: (3,)，x/y/z 三个方向的体素数量，dtype=int32。
        """
        axis_len = self._bounds[:, 1] - self._bounds[:, 0]
        return np.ceil(axis_len / self.voxel_scale).astype(np.int32)

    def Pos2GridIndex(self, pos: np.ndarray) -> np.ndarray:
        """把坐标点转换成三维网格索引。

        输入:
        - pos: (3,) 或 (N, 3)，世界/局部坐标点。

        返回:
        - grid_index: (3,)，输入为 (3,) 时返回单个点的 [ix, iy, iz]。
        - grid_index: (N, 3)，输入为 (N, 3) 时返回每个点的 [ix, iy, iz]。
        """
        pos, squeeze = self._as_points(pos)
        grid_index = np.floor((pos - self._bounds[None, :, 0]) / self.voxel_scale).astype(np.int32)
        grid_index = np.clip(grid_index, 0, self.grid_shape[None, :] - 1)
        return grid_index[0] if squeeze else grid_index
    
    def Pos2FlatIndex(self,pos:np.ndarray) -> int | np.ndarray:
        """把坐标点转换成一维网格索引。

        输入:
        - pos: (3,) 或 (N, 3)，世界/局部坐标点。

        返回:
        - int，输入为 (3,) 时返回单个 flat index。
        - flat_index: (N,)，输入为 (N, 3) 时返回每个点的一维索引。
        """
        pos, squeeze = self._as_points(pos)
        grid_index = np.floor((pos - self._bounds[None, :, 0]) / self.voxel_scale).astype(np.int32)
        grid_index = np.clip(grid_index, 0, self.grid_shape[None, :] - 1)
        return self.GridIndex2FlatIndex(grid_index[0] if squeeze else grid_index)
    
            

    def GridIndex2FlatIndex(self, grid_index: np.ndarray) -> int | np.ndarray:
        """把三维网格索引转换成一维索引。

        输入:
        - grid_index: (3,) 或 (N, 3)，每行是 [ix, iy, iz]。

        返回:
        - int，输入为 (3,) 时返回单个 flat index。
        - flat_index: (N,)，输入为 (N, 3) 时返回每个网格的一维索引。

        说明:
        - flat index 按 nx, ny, nz 的 C-order 展平顺序排列。
        """
        grid_index = np.asarray(grid_index, dtype=np.int64)
        squeeze = grid_index.shape == (3,)
        if squeeze:
            grid_index = grid_index[None, :]
        elif grid_index.ndim != 2 or grid_index.shape[1] != 3:
            raise ValueError("grid_index must have shape (3,) or (N, 3)")
        ix, iy, iz = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]
        _, ny, nz = self.grid_shape
        flat_index = (ix * ny + iy) * nz + iz
        return int(flat_index[0]) if squeeze else flat_index

    def FlatIndex2GridIndex(self, flat_index: int | np.ndarray) -> np.ndarray:
        """把一维索引转换成三维网格索引。

        输入:
        - flat_index: int、标量 ndarray、或 (N,) 整数数组。

        返回:
        - grid_index: (3,)，输入为单个索引时返回 [ix, iy, iz]。
        - grid_index: (N, 3)，输入为 (N,) 时返回每个索引对应的 [ix, iy, iz]。
        """
        nx, ny, nz = self.grid_shape
        flat_index = np.asarray(flat_index, dtype=np.int64)
        squeeze = flat_index.ndim == 0
        flat_index = flat_index.reshape(-1)
        if ((flat_index < 0) | (flat_index >= nx * ny * nz)).any():
            raise IndexError("flat_index out of range")
        ix = flat_index // int(ny * nz)
        rem = flat_index % int(ny * nz)
        iy = rem // int(nz)
        iz = rem % int(nz)
        grid_index = np.stack([ix, iy, iz], axis=1).astype(np.int32)
        return grid_index[0] if squeeze else grid_index

    def GridCenter(self, grid_index: np.ndarray) -> np.ndarray:
        """查询三维网格索引对应的体素中心坐标。

        输入:
        - grid_index: (3,) 或 (N, 3)，每行是 [ix, iy, iz]。

        返回:
        - center: (3,)，输入为 (3,) 时返回单个体素中心。
        - center: (N, 3)，输入为 (N, 3) 时返回每个体素中心。
        """
        grid_index, squeeze = self._as_points(grid_index)
        grid_index = np.clip(grid_index,0,self.grid_shape[None,:]-1)
        center = self._bounds[None, :, 0] + (grid_index + 0.5) * self.voxel_scale
        return center[0] if squeeze else center

    def FlatIndex2Center(self, flat_index: int) -> np.ndarray:
        """查询一维索引对应的体素中心坐标。

        输入:
        - flat_index: int、标量 ndarray、或 (N,) 整数数组。

        返回:
        - center: (3,)，输入为单个索引时返回单个体素中心。
        - center: (N, 3)，输入为 (N,) 时返回每个体素中心。
        """
        return self.GridCenter(self.FlatIndex2GridIndex(flat_index))

    def QueryCenter(self, pos: np.ndarray) -> np.ndarray:
        """查询坐标点所在体素的中心坐标。

        输入:
        - pos: (3,) 或 (N, 3)，世界/局部坐标点。

        返回:
        - center: (3,)，输入为 (3,) 时返回所在体素中心。
        - center: (N, 3)，输入为 (N, 3) 时返回每个点所在体素中心。
        """
        return self.GridCenter(self.Pos2GridIndex(pos))

    def TrilinearSample(
        self,
        values:np.ndarray,
        pos:np.ndarray,
        outside_value:float,
    )->np.ndarray|float:
        """在体素中心定义的标量场上进行三线性插值。

        输入:
        - values: 形状必须为 ``self.grid_shape`` 的三维标量场。数组索引
          ``values[i, j, k]`` 对应第 ``(i, j, k)`` 个体素中心的值。
        - pos: 要采样的坐标点，可为单点 ``(3,)`` 或多个点 ``(N, 3)``。
        - outside_value: 采样位置在体素网格外时使用的常量值。

        返回:
        - pos 为 ``(3,)`` 时，返回单个 ``float`` 插值结果。
        - pos 为 ``(N, 3)`` 时，返回形状为 ``(N,)`` 的 ``ndarray``。

        第一个体素中心位于 ``self._bounds[:, 0] + 0.5 *
        self.voxel_scale``，而非网格边界。例如，体素尺寸为 0.1、x 方向
        最小边界为 0 时，x=0.05 和 x=0.15 分别是索引 0、1 的体素中心；
        在 x=0.10 采样会得到两者的线性平均值。

        示例::

            values = np.zeros(self.grid_shape, dtype=np.float32)
            values[0, 0, 0] = 1.0
            values[1, 0, 0] = 3.0

            # 当 voxel_scale 为 0.1，且 x 最小边界为 0 时，结果为 2.0。
            value = self.TrilinearSample(
                values, np.array([0.10, 0.05, 0.05]), outside_value=-1.0
            )

            # 一次查询多个点，返回形状为 (2,) 的数组。
            samples = self.TrilinearSample(
                values,
                np.array([[0.05, 0.05, 0.05], [0.10, 0.05, 0.05]]),
                outside_value=-1.0,
            )
        """
        points,squeeze = self._as_points(pos)
        values = np.asarray(values)
        if values.shape != tuple(self.grid_shape):
            raise ValueError(
                f"values shape {values.shape} does not match voxel grid "
                f"{tuple(self.grid_shape)}"
            )
        # 第一个体素中心对应连续网格坐标0，而不是体素边界。
        grid_pos = (
            (points-self._bounds[None,:,0])/self.voxel_scale-0.5
        ).T
        sampled = map_coordinates(
            values.astype(np.float64,copy=False),
            grid_pos,
            order=1,
            mode="grid-constant",
            cval=float(outside_value),
            prefilter=False,
        )
        return float(sampled[0]) if squeeze else sampled

class LegVoxels(Voxels):
    """腿部体素独有的数据 关节逆运动学解，关节位置，末端关节x朝向，可行腿"""
    def __init__(
        self,
        kinematic: Kinematic,
        xyz_range=[[-0.14, 0.3], [-0.3, 0.3], [-0.27, 0.2]],
        voxel_scale: float = 0.005
    ):
        self.kin = kinematic
        super().__init__(np.array(xyz_range),voxel_scale)
        #采用批量numpy代理
        max_sol = 2
        self.solution_num = np.zeros((self.grid_size,), dtype=np.uint8)
        self.reachable_legs = None #grid_size 解的数量 6个腿 solve all中有定义
        self.joints = np.full((self.grid_size, max_sol, 3), np.nan, dtype=np.float32)
        self.x_b3 = np.full((self.grid_size, max_sol, 3), np.nan, dtype=np.float32)
        # self.knee_pos = np.full((self.grid_size, max_sol, 3), np.nan, dtype=np.float32)
        self.ankle_pos = np.full((self.grid_size, max_sol, 3),np.nan, dtype=np.float32)
        self.ankle_collide_radi = 0.04 #ankle关节碰撞圆的半径

        #[LB:[0,1,2,3],LF:[3,4,5,6],[],[]]
        self._leg_names=["LB","LF","LM","RB","RF","RM"]
        self.leg_feasi_voxel_index=[]

        #计算由身腿部的体素转换到R系下共同组成的机器人整体最外侧的包络边界
        self._bounds_R = np.zeros((3,2),dtype=np.float32)
        #3,2,6
        leg_bounds_B = np.asarray(xyz_range, dtype=np.float32)  # 3,2
        xs = leg_bounds_B[0]
        ys = leg_bounds_B[1]
        zs = leg_bounds_B[2]
        corners_B = np.array(
            [[x, y, z] for x in xs for y in ys for z in zs],
            dtype=np.float32,
        ) #8,3
        all_points_R = []
        for leg_id in range(6):
            corners_R = self.kin._B2R(corners_B.T, leg_id).T #8,3
            all_points_R.append(corners_R)

        all_points_R = np.vstack(all_points_R)
        self._bounds_R[:, 0] = all_points_R.min(axis=0)
        self._bounds_R[:, 1] = all_points_R.max(axis=0)

    def IsInsideRangeR(self, pos):
        pos, squeeze = self._as_points(pos)
        inside = ((pos >= self._bounds_R[None, :, 0]) & (pos < self._bounds_R[None, :, 1])).all(axis=1)
        return bool(inside[0]) if squeeze else inside
            
    def SolveAllVoxels(self):
        """这里只对角度范围内的进行逆解算，没有碰撞检查"""
        joints, _ = self.kin.InverseKin2MultiBatch(self.center) #grid_size,2,3
        self.joints = joints.astype(np.float32)
        valid_mask = ~(np.isnan(self.joints).any(axis=2)) #grid_size,2
        valid_joints = self.joints[valid_mask] # valid_size*2, 3
        self.x_b3[valid_mask] = self.kin.Get_X_B3(valid_joints).astype(np.float32)

        valid_pos = np.zeros_like(valid_joints)
        # self.kin.ForwardKinKnee(valid_joints,valid_pos)
        # self.knee_pos[valid_mask]=valid_pos    
        valid_pos.fill(0)
        self.kin.ForwardKinAnkle(valid_joints,valid_pos)
        self.ankle_pos[valid_mask]=valid_pos

        self.solution_num = valid_mask.sum(axis=1).astype(np.uint8)

        #grid_size,2,6
        thigh_joints = self.joints[...,0] #grid_size,2
        rest_joints = self.joints[...,1:3] #grid_size,2,2
        #grid_size,2,1  >=  1,1,6   -> grid_size,2,6
        thigh_valid = (thigh_joints[...,None] >= self.kin.joints_limits_thigh[None,None,:,0]) \
                      & (thigh_joints[...,None] <= self.kin.joints_limits_thigh[None,None,:,1])
        #grid_size,2,2 >= 1,1,2   -> grid_size, 2
        rest_valid = (rest_joints>=self.kin.joints_limits_rest[None,None,:,0]).all(axis=2) \
                     & (rest_joints<=self.kin.joints_limits_rest[None,None,:,1]).all(axis=2)
        # 这里排除需要穿过奇异点才能到达的工作空间
        #grid_size
        # not_cross_singular = np.cos(np.arctan2(self.center[:,1],self.center[:,0]))>np.cos(np.deg2rad(95.0))
        not_cross_singular = self.center[:,0]>=0

        # grid_size,2,6 
        self.reachable_legs = thigh_valid&rest_valid[...,None]&not_cross_singular[:,None,None]


        #2两个解中 只要有一个有解在工作空间内，就可以
        leg_valid = self.reachable_legs.any(axis=1)
        #若有两个解存在的情况下,把两个解一个在工作空间，另一个不在工作空间的情况列出来
        max_sol_index = np.where(self.solution_num==2)[0] # two_sol_size
        # two_sol_size, 6
        different_mask = self.reachable_legs[max_sol_index,0,:] != self.reachable_legs[max_sol_index,1,:]
        if different_mask.any():
            sol_index, leg_index = np.where(different_mask)
            print("Solving legs: one solution in this leg, but the other not")
            print(f"grid index={self.grid_index[max_sol_index[sol_index]]} leg index={leg_index}\n \
                    center\n {self.center[max_sol_index[sol_index]]}\n joints\n {self.joints[max_sol_index[sol_index]]}")

        for i in range(6):
            self.leg_feasi_voxel_index.append(np.where(leg_valid[:,i])[0])


    def WriteFile(self,dir_path:str):
        """dir_path代表文件夹路径"""

        np.savez(os.path.join(dir_path,"leg_voxels_info.npz"),
                 joints=self.joints,
                #  knee_pos=self.knee_pos,
                 ankle_pos=self.ankle_pos,
                 x_b3=self.x_b3,
                 reachable_legs=self.reachable_legs,
                 )
        print("--------->save leg voxels info to "+os.path.join(dir_path,"leg_voxels_info.npz"))

    def BuildFromFile(self,dir_path:str):
        """dir_path代表文件夹路径"""
        file = os.path.join(dir_path,"leg_voxels_info.npz")

        if os.path.exists(file):
            with np.load(file) as leg_voxels_info:
                self.joints = leg_voxels_info["joints"]
                # self.knee_pos = leg_voxels_info["knee_pos"]
                self.ankle_pos = leg_voxels_info["ankle_pos"]
                self.x_b3 = leg_voxels_info["x_b3"]
                self.reachable_legs = leg_voxels_info["reachable_legs"]
            for i in range(6):
                self.leg_feasi_voxel_index.append(np.where(self.reachable_legs[...,i].any(axis=1)))
            print("-------->build leg voxels from file")
            return True
        return False

class BodyVoxels(Voxels):
    """身体体素的独有数据 身体作为障碍物的ESDF地图"""
    def __init__(self,
                xyz_range=[[-0.4, 0.4], [-0.52, 0.52], [-0.27, 0.2]],
                voxel_scale: float = 0.005):
        super().__init__(np.array(xyz_range),voxel_scale)
        #构建身体-腿部碰撞部分,从STL文件中读取，构建封闭几何体
        # 这个STL文件用于检查腿部和身体的碰撞，只把身体和knee关节包络其中，供ankle和end检查，knee通过角度限制确保了不会碰撞
        body_mesh_for_leg:trimesh.Trimesh= trimesh.load_mesh(LEGGED_GYM_ROOT_DIR+"/resources/robots/hex_v4/body_leg_collision.STL") 
        #这个STL文件把身体在水平方向扩张一直到ankle关节，用于和环境检查是否碰撞，可以为ankle摆动预留空间
        body_mesh_for_env:trimesh.Trimesh= trimesh.load_mesh(LEGGED_GYM_ROOT_DIR+"/resources/robots/hex_v4/body_extend_collision.STL")
        if isinstance(body_mesh_for_leg,trimesh.Scene):
            body_mesh_for_leg = trimesh.util.concatenate(tuple(body_mesh_for_leg.geometry.values()))
        if isinstance(body_mesh_for_env,trimesh.Scene):
            body_mesh_for_env = trimesh.util.concatenate(tuple(body_mesh_for_env.geometry.values()))
        self.body_mesh_for_leg = body_mesh_for_leg
        self.body_mesh_for_env = body_mesh_for_env

    def BuildBodyAllVoxels(self):
        """
        构建身体的Voxl, 构建两个Voxl，一个用于腿部碰撞检查判断，一个用于质心规划阶段，避免身体靠近障碍物太近，采用ESDF保存
        """
        #body-leg区域
        inside_center_mask = self.body_mesh_for_leg.contains(self.center)
        body_occ = inside_center_mask.reshape(self.grid_shape)
        dist_out = distance_transform_edt(~body_occ,sampling=self.voxel_scale)
        dist_in = distance_transform_edt(body_occ,sampling=self.voxel_scale)
        self.esdf_for_leg:np.ndarray = (dist_out - dist_in).astype(np.float32)

        #body-环境障碍物区域
        inside_center_mask = self.body_mesh_for_env.contains(self.center)
        body_occ = inside_center_mask.reshape(self.grid_shape)
        dist_out = distance_transform_edt(~body_occ,sampling=self.voxel_scale)
        dist_in = distance_transform_edt(body_occ,sampling=self.voxel_scale)
        self.esdf_for_env:np.ndarray = (dist_out - dist_in).astype(np.float32)


        #构建平铺开的esdf
        self.esdf_flat_for_leg = self.esdf_for_leg.ravel()
        self.esdf_flat_for_env = self.esdf_for_env.ravel()

        # plotter = pv.Plotter()
        # body_cloud = pv.PolyData(self.center[inside_center_mask])
        # plotter.add_mesh(body_cloud,render_points_as_spheres=True,point_size=30)
        # plotter.show()

    def WriteFile(self,dir_path:str):
        """dir_path代表文件夹路径"""
        np.savez(os.path.join(dir_path,"body_voxels_info.npz"), 
                 esdf_for_leg=self.esdf_for_leg,
                 esdf_for_env=self.esdf_for_env
                 )
        
        print(f"--------->save body voxels info to "+os.path.join(dir_path,"body_voxels_info.npz"))        

    def BuildFromFile(self,dir_path:str):
        """dir_path代表文件夹路径"""
        file = os.path.join(dir_path,"body_voxels_info.npz")
        if os.path.exists(file):
            with np.load(file) as body_voxels_info:
                self.esdf_for_leg = body_voxels_info["esdf_for_leg"]
                self.esdf_flat_for_leg = self.esdf_for_leg.ravel()
                self.esdf_for_env = body_voxels_info["esdf_for_env"]
                self.esdf_flat_for_env = self.esdf_for_env.ravel()                
            print("---------->Build body voxels from file")
            return True
        else:
            return False

class EnvPointsVoxels(PointMap):
    """
    继承PointMap 在内部构造一个Voxels变量
    环境体素独有的数据，环境ESDF，环境点云的位置，法向量，密度，粗糙度等
    不需要再单独保存，点云部分包括了最大的计算量，这里只需计算esdf，速度很快"""
    def __init__(self,cloud_file:str,voxel_scale=0.04):
        super().__init__(cloud_file)
        """@input cloud_file 点云的原始文件位置
        """
        self.voxels = Voxels(self._bounds,voxel_scale)
        #构建ESDF地图
        ix,iy,iz = self.voxels.Pos2GridIndex(self.points).T
        env_occ = np.zeros(self.voxels.grid_shape,dtype=bool)
        env_occ[ix,iy,iz] = True
        dist_out = distance_transform_edt(~env_occ,sampling=voxel_scale)
        dist_in  = distance_transform_edt(env_occ,sampling=voxel_scale)
        self.env_esdf:np.ndarray = (dist_out - dist_in).astype(np.float32)

        self.env_esdf_flat = self.env_esdf.ravel()

    def __enter__(self)->EnvPointsVoxels:
        """方便 with self.hex_state.env_pointsmap_voxels as pointmap:
        此时能推断出 pointmap的类型"""
        return self

class RoboVoxels:
    """包含了机器人腿部和身体的体素"""
    def __init__(self,
                 kinematic:Kinematic,
                 leg_xyz_range=[[-0.14, 0.38], [-0.38, 0.38], [-0.27, 0.25]], #在腿部B坐标系下
                 body_xyz_range=[[-0.4, 0.4], [-0.5, 0.5], [-0.2, 0.2]],#载身体R坐标系下
                 voxel_scale = 0.01 #腿和身体采用同样大小的体素
                 ):
        self.kin = kinematic
        self.leg_voxels = LegVoxels(kinematic,leg_xyz_range,voxel_scale)
        self.body_voxels = BodyVoxels(body_xyz_range,voxel_scale)
        #考虑身体体素后，腿部部分空间可能不能达到，因此在这里定义新的reachable_legs属性
        # leg_grid_size, 2, 6
        self.robot_reachable_legs=np.zeros((self.leg_voxels.grid_size,2,6),dtype=np.bool_) 
        #grid_size * 6
        self.to_bound_dist_flat = np.zeros((self.leg_voxels.grid_size,6),dtype=np.float32) #距离工作空间边界的距离
        #ix * iy * iz * 6
        self.to_bound_dist = np.zeros((tuple(self.leg_voxels.grid_shape)+(6,)),dtype=np.float32)

        if not self.ReadRobotVoxels():
            print("Rebuild leg voxels and body voxles")
            self.RebuildRobotVoxels()
            self.WriteRobotVoxels()
            self.PlotRobotVoxels()

    def _LegBodyFree(self,leg_end_pos:np.ndarray,leg_index:int)->np.ndarray:
        """检查腿部ankle关节，末端位置是否与身体碰撞 这里不考虑knee关节 因为规划阶段身体膨胀区域已经将其包括在里面了 
        此外 在腿部规划阶段 thigh关节角度的限制限制了knee关节不会发生碰撞
        @input 在腿部基坐标系B下 leg_end_pos:batch_size,3
        @output free_mask batch_size,2 两个解中可行的解"""
        #直接把leg_eng_pos转化为R系下
        R_end = self.kin._B2R(leg_end_pos.T,leg_index).T #batch_size,3
        #划分到腿部的center，然后查表找到对应的index
        leg_flat_index = self.leg_voxels.Pos2FlatIndex(leg_end_pos)
        #获取到index对应的末端和knee和ankle的位置，转换到R系下
        # R_knee = self.leg_voxels._B2R(self.leg_voxels.knee_pos[leg_flat_index].reshape(-1,3).T,leg_index).T#batch_size,2,3
        R_ankle = self.kin._B2R(self.leg_voxels.ankle_pos[leg_flat_index].reshape(-1,3).T,leg_index).T

        #转换到身体voxels里面，
        # ix,iy,iz = self.body_voxels.Pos2GridIndex(R_end).T
        # end_free_mask = self.body_voxels.esdf[ix,iy,iz]>=0.03#batch_size
        #使用三维线性插值直接计算R_end的ESDF, 如果超出了身体的范围，设置ESDF为0.05 因为此时末端肯定不会和身体发生碰撞
        end_free_mask = self.body_voxels.TrilinearSample(self.body_voxels.esdf_for_leg,R_end,0.05)>=0.03
        ankle_free_mask = self.body_voxels.TrilinearSample(self.body_voxels.esdf_for_leg,
                                                           np.nan_to_num(R_ankle),outside_value=0.05) \
                                                           >= self.leg_voxels.ankle_collide_radi
        ankle_free_mask = ankle_free_mask.reshape(-1,2)
        #考虑到knee中存在nan的情况，设置为0，检查结果为碰撞
        # ix,iy,iz = self.body_voxels.Pos2GridIndex( np.nan_to_num(R_knee,nan=0.0) ).T
        # knee_free_mask = (self.body_voxels.esdf[ix,iy,iz]>=0.04).reshape(-1,2)#batch_size,2
        
        # ix,iy,iz = self.body_voxels.Pos2GridIndex( np.nan_to_num(R_ankle,nan=0.0) ).T
        # ankle_free_mask = (self.body_voxels.esdf[ix,iy,iz]>=self.leg_voxels.ankle_collide_radi).reshape(-1,2)#batch_size,2
        #以上mask只考虑了碰撞，为了加入关节角度的限制，还需要加入对应腿部原有的reachable_mask
        collision_free_mask = ankle_free_mask  & end_free_mask[:,None]#batch_size,2
        # collision_free_mask = ankle_free_mask & knee_free_mask & end_free_mask[:,None]#batch_size,2
        reachable_mask = self.leg_voxels.reachable_legs[leg_flat_index][...,leg_index]#batch_size,2
        #leg原有的工作空间只考虑了角度限制范围，没有考虑能否支撑和接触，这里绘制了一个stl文件，用来进一步限制腿部活动范围
        #这里的STL文件是一个截面旋转200度构成的，这里只需要考虑是否在内部，超过thigh关节限制的原有的reachable_mask会筛除

        return collision_free_mask&reachable_mask

    def RebuildRobotVoxels(self):
        """对腿部构建好的voxel与身体进行碰撞检查，将计算好的结果放入这个类定义的robot_reachable_legs中
        计算距离边界的最小距离"""
        # leg_mesh:trimesh.Trimesh= trimesh.load_mesh(LEGGED_GYM_ROOT_DIR+"/resources/robots/hex_v4/working_range.STL")
        # inside_STL = leg_mesh.contains(self.leg_voxels.center) #leg_grid_size

        self.leg_voxels.SolveAllVoxels()
        self.body_voxels.BuildBodyAllVoxels()
        for i in range(6):
            feasible_mask = self._LegBodyFree(self.leg_voxels.center,leg_index=i) #N,2
            self.robot_reachable_legs[...,i] = feasible_mask
            occ_map = feasible_mask.any(axis=1).reshape(self.leg_voxels.grid_shape)
            dist_inside = distance_transform_edt(occ_map,sampling=self.leg_voxels.voxel_scale)
            dist_outside = distance_transform_edt(~occ_map,sampling=self.leg_voxels.voxel_scale)
            self.to_bound_dist[...,i] = dist_inside - dist_outside
        self.to_bound_dist_flat = self.to_bound_dist.reshape(-1,6)
        print("to bound dist share memeory")
        print(np.shares_memory(self.to_bound_dist,self.to_bound_dist_flat))

        #N,2,6
        # self.robot_reachable_legs &= inside_STL[:,None,None]

    def WriteRobotVoxels(self)->bool:
        """把腿部的体素和身体的体素与ESDF保存在文件中"""
        self.body_voxels.WriteFile(f"{EXPERT_COMPLEX_DIR}/voxels_info")
        self.leg_voxels.WriteFile(f"{EXPERT_COMPLEX_DIR}/voxels_info")
        robot_info_file = os.path.join(f"{EXPERT_COMPLEX_DIR}/voxels_info/robot_reachable_legs.npz")
        np.savez(robot_info_file,
                 robot_reachable_legs=self.robot_reachable_legs,
                 to_bound_dist = self.to_bound_dist)
        print("-------->save robot_reachable legs to "+robot_info_file)

    def ReadRobotVoxels(self):
        """直接从文件中加载腿部和身体的体素"""
        robo_info_file = os.path.join(f"{EXPERT_COMPLEX_DIR}/voxels_info/robot_reachable_legs.npz")
        if os.path.exists(robo_info_file):
            with np.load(robo_info_file) as robo_info:
                self.robot_reachable_legs = robo_info["robot_reachable_legs"]
                self.to_bound_dist = robo_info["to_bound_dist"]
                self.to_bound_dist_flat = self.to_bound_dist.reshape(-1,6)
                print("to bound dist share memeory")
                print(np.shares_memory(self.to_bound_dist,self.to_bound_dist_flat))

            return self.leg_voxels.BuildFromFile(f"{EXPERT_COMPLEX_DIR}/voxels_info") &\
                   self.body_voxels.BuildFromFile(f"{EXPERT_COMPLEX_DIR}/voxels_info")
        else:
            print("robot reachable leg info does not exit")
            return False

    def PlotRobotVoxels(self):
        #绘制身体voxels
        plotter = pv.Plotter()
        plotter.show_axes()
        flat_index = np.where(self.body_voxels.esdf_flat_for_leg<=0)
        plotter.add_mesh(pv.PolyData(self.body_voxels.center[flat_index]),render_points_as_spheres=True,point_size=20,color='red')

        flat_index = np.where(self.body_voxels.esdf_flat_for_env<=0)
        plotter.add_mesh(pv.PolyData(self.body_voxels.center[flat_index]),render_points_as_spheres=True,point_size=20,color='green',opacity=0.2)        

        #绘制身体STL
        # body_mesh_for_leg = self.body_voxels.body_mesh_for_leg
        # faces = np.column_stack([
        #     np.full(body_mesh_for_leg.faces.shape[0], 3),
        #     body_mesh_for_leg.faces,
        # ]).ravel()
        # plotter.add_mesh(
        #     pv.PolyData(body_mesh_for_leg.vertices, faces),
        #     color="white",
        #     opacity=0.35,
        #     show_edges=True,
        # )

        # points = pv.PolyData(np.array([[0,0,-0.09],[0.23,0.513,0]]))
        # plotter.add_mesh(points,render_points_as_spheres=True,point_size=50,color='red')
        color_list=["red","green","blue","yellow","gray","cyan"]
        for i in range(6):
            # if i==1 or i==5:
            # reachable = (self.leg_voxels.reachable_legs[...,i]).any(axis=1) #leg_grid_size
            # cloud = pv.PolyData(self.kin._B2R( self.leg_voxels.center[reachable].T ,i).T)
            # plotter.add_mesh(cloud,render_points_as_spheres=True,point_size=20,color=color_list[i],opacity=0.2)

            #此处绘制的是腿部原始可行的工作空间 没有检查和身体的碰撞
            # reachable = (self.leg_voxels.reachable_legs[...,i]).any(axis=1)
            # cloud = pv.PolyData(self.kin._B2R( self.leg_voxels.center[reachable].T ,i).T)
            # plotter.add_mesh(cloud,render_points_as_spheres=True,point_size=20,color=color_list[i],opacity=0.4)

            #此处绘制的是腿部和身体esdf_for_leg检查后的工作空间
            reachable = (self.robot_reachable_legs[...,i]).any(axis=1) #leg_grid_size
            cloud = pv.PolyData(self.kin._B2R( self.leg_voxels.center[reachable].T ,i).T)
            plotter.add_mesh(cloud,render_points_as_spheres=True,point_size=20,color=color_list[i],opacity=0.2)
            # stance_inside = (self.to_bound_dist_flat[...,i]>=0.04)
            # cloud = pv.PolyData(self.kin._B2R(self.leg_voxels.center[stance_inside].T,i).T)
            # plotter.add_mesh(cloud,render_points_as_spheres=True,point_size=20,color=color_list[i])
        plotter.show()

class HexState:
    """构造机器人和环境的体素和点云环境，
    提供:可行性检查，轨迹规划等方法
    1. 当前身体位姿可行性检查与对应腿部可选的落脚点
    2. 点是否在对应的腿部空间内
    3. 给定起点和方向，获得在腿部工作空间内，从起点沿方向最远的点
    4. 整体轨迹规划和摆动腿部的轨迹规划方法"""
    def __init__(self,kinematic:Kinematic):
        self.kin = kinematic
        self.robot_voxels = RoboVoxels(kinematic=self.kin)
        self.env_pointsmap_voxels = EnvPointsVoxels(
        # LEGGED_GYM_ROOT_DIR+"/resources/environments/sutructure1/complex_surface_point_map.npz")
        LEGGED_GYM_ROOT_DIR+"/resources/environments/structure2/regular_dodecagon_point_map.npz")

    def _LegNormFeasi(self,R_points:np.ndarray,R_normals:np.ndarray,leg_index:Union[np.ndarray|int]):
        """检查这一点是否法向量可行, 先检查这一点法向量与腿部连杆所在平面的夹角是否在范围内，再选出两个解中哪一个解的x_b3可行

        @input R_points(N,3)R系下的点 R_normals(N,3)R系下对应的法向量 leg_index:N 或者 int
        @output feasible_mask(N,2) 对应的点的两个解的可行性
        """
        pn_x = self.kin.leg_base_p[1,leg_index] - R_points[:,1]
        pn_y = R_points[:,0] - self.kin.leg_base_p[0,leg_index]
        pn = np.column_stack([pn_x,pn_y,np.zeros_like(pn_y)]) #N,3
        eps = 1e-5
        pn = pn/np.maximum(np.linalg.norm(pn,axis=1,keepdims=True),eps)
        R_normals = R_normals/np.maximum(np.linalg.norm(R_normals,axis=1,keepdims=True),eps)
        dot_res = np.sum(R_normals*pn,axis=1)
        #在连杆所在平面的角度锥之内 N,1
        inside_plane_cone = (np.abs(dot_res) <= np.cos(np.deg2rad(75.0)))[:,None]

        #选择这一点对应的解的分支，采用x_b3的方向判断
        B_points = self.kin._R2B(R_points.T,leg_index) #3,N
        B_normals = R_normals.copy() #N,3
        if np.isscalar(leg_index):
            if leg_index<3:
                B_normals[:,:2] *= -1
        else:
            left_mask = (leg_index<3)
            B_normals[left_mask,:2] *= -1

        flat_index = self.robot_voxels.leg_voxels.Pos2FlatIndex(B_points.T)
        x_b3 = self.robot_voxels.leg_voxels.x_b3[flat_index] # N,2,3
        #N,2 因为x_b3方向是靠近指向平面的，平面法向量方向是原理朝向身体外侧的，因此小于0的才算在同一侧
        # along_with_xb3:np.ndarray = np.einsum("njk,nk->nj",x_b3,B_normals)<=0

        #现在要求x_b3与R_normals夹角不能超过95°
        along_with_xb3:np.ndarray = np.einsum("njk,nk->nj",x_b3,-B_normals)>=np.cos(np.deg2rad(95.0))


        
        #找出两个解同时成立和同时不成立的 排除nan的部分
        # all_mask = along_with_xb3.all(axis=1) #N
        # none_mask = (~along_with_xb3.any(axis=1)) #N
        # nan_mask = np.isnan(np.einsum("njk,nk->nj",x_b3,B_normals)).all(axis=1) #N,2
        # print("two solutios all satisfy")
        # print(f"joints\n{self.robot_voxels.leg_voxels.joints[flat_index[all_mask]]}")
        # print(f"x_b3\n {self.robot_voxels.leg_voxels.x_b3[flat_index[all_mask]]}")
        # print("two solutions all not satisfy")
        # none_mask = (~nan_mask)&(none_mask)
        # print(f"joints={self.robot_voxels.leg_voxels.joints[flat_index[none_mask]]}")
        # print(f"x_b3\n {self.robot_voxels.leg_voxels.x_b3[flat_index[none_mask]]}")
        

        return inside_plane_cone&along_with_xb3

    def _LegEnvCollisionFree(self,B_points:np.ndarray,W_T_R:SE3,leg_index:int,ignore_ankle=False,ignore_end=False):
        """给定腿部的点和当前身体位姿和腿部索引 综合 有解&ankle无碰&end无碰 返回检查结果

        @input B_points N,3 W_T_R SE3 leg_index:int ignore_ankle ignore_end 设置True无视碰撞结果

        @output feasible_mask N,2"""
        #足端可达可行性
        flat_index = self.robot_voxels.leg_voxels.Pos2FlatIndex(B_points)
        #grid_size,2,6 -> batch_size,2 针对第i个腿的两个解
        #要使用RobotVoxels中重建的
        reachable_mask = self.robot_voxels.robot_reachable_legs[flat_index,:,leg_index]
        #拿到ankle关节和末端位置，转换到世界坐标系下，判断与环境的ESDF
        #grid_size,2,3 -> batch_size,2,3
        if not ignore_ankle:
            ankle_pos =np.nan_to_num( self.robot_voxels.leg_voxels.ankle_pos[flat_index].reshape(-1,3), nan=0.0)
            ankle_pos = (W_T_R*self.robot_voxels.kin._B2R(ankle_pos.T,leg_index)).T #batch_size*2,3
            ix,iy,iz = self.env_pointsmap_voxels.voxels.Pos2GridIndex(ankle_pos).T
            ankle_free_mask = (self.env_pointsmap_voxels.env_esdf[ix,iy,iz]>=
                                self.robot_voxels.leg_voxels.ankle_collide_radi).reshape(-1,2)
        else:
            ankle_free_mask = np.ones_like(reachable_mask,dtype=bool)
        if not ignore_end:
            end_pos = self.robot_voxels.leg_voxels.center[flat_index]
            end_pos = (W_T_R*self.robot_voxels.kin._B2R(end_pos.T,leg_index)).T #batch_size,3
            ix,iy,iz = self.env_pointsmap_voxels.voxels.Pos2GridIndex(end_pos).T
            end_free_mask = (self.env_pointsmap_voxels.env_esdf[ix,iy,iz]>=0.007)[:,None] #batch_size,1  
        else:
            end_free_mask = np.ones_like(reachable_mask,dtype=bool)

        return reachable_mask&ankle_free_mask&end_free_mask
    
    def _LegLegCollisionFree(self,checking_points:np.ndarray,stance_points:np.ndarray,stance_sol_index:np.ndarray,
                             checking_index:int,stance_index:Union[int|np.ndarray])->np.ndarray:
        """
        计算一条腿末端位于不同点时，与其他指定的腿之间的碰撞情况，
        为了简化计算，这里只考虑腿部末端与末端 ankle关节之间的碰撞 不做交叉检查

        @input 
        checking_points(N,3) R系下 待检查的腿所有可能的点;
        stance_points(M,3) R系下 已经确定位置腿部的点; stanc_sol_index(M,) 上一个点对应的逆解的索引
        checking_index(int) 待检查的腿索引 stance_index(M,) 其余已经确定位置腿的索引

        @output free_mask(N,2)
        """
        #1 检查末端位置 
        #N,1,3 - 1,M,3 ->N,M,3 -> N,M
        distance = np.linalg.norm(checking_points[:,None,:]-stance_points[None,...],axis=-1)
        end_free_mask = (distance>=0.05).all(axis=1)[:,None] #(N,1)
        #2 检查ankle关节碰撞
        flat_index = self.robot_voxels.leg_voxels.Pos2FlatIndex(
            self.kin._R2B(checking_points.T,checking_index).T
        )
        ankle_pos_flat = self.robot_voxels.leg_voxels.ankle_pos[flat_index].reshape(-1,3) #N*2, 3
        checking_ankle_pos = (self.kin._B2R(ankle_pos_flat.T,checking_index).T)#N*2,3


        flat_index = self.robot_voxels.leg_voxels.Pos2FlatIndex(
            self.kin._R2B(stance_points.T,stance_index).T
        )
        ankle_pos_flat = self.robot_voxels.leg_voxels.ankle_pos[flat_index,stance_sol_index] #M, 3
        stance_ankle_pos = self.kin._B2R(ankle_pos_flat.T,stance_index).T#M,3
        
        #N*2,1,3 - 1,M,3 - -> N*2,M,3 -> N*2,M
        distance = np.linalg.norm(checking_ankle_pos[:,None,:]-stance_ankle_pos[None,...],axis=-1)
        ankle_free_mask = (distance>=self.robot_voxels.leg_voxels.ankle_collide_radi).all(axis=1) #N*2
        ankle_free_mask = ankle_free_mask.reshape(-1,2)

        return end_free_mask&ankle_free_mask

    def RobotFeasiCheck(self,W_T_R:SE3,points_idx:Union[None|np.ndarray]=None)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        输入当前机器人位姿，获得当前身体是否与环境碰撞，若不碰撞，每个腿部工作空间都有哪些点
        @input W_T_R:SE3; 
        
        points_idx(N,3 | None)设置为None时 机器人自动从环境中搜索附近点 输入array时 代表点云中要查询的点的索引
        计算就只围绕这些点开展

        @output 
        inside_idx point_map中在身体附近的点的索引
        landing_idx point_map中经过延长后在身体附近的落脚点的索引
        下面两个mask分别针对以上两个索引
        身体的body_collisionfree_mask: point_num True代表不碰撞的可行点
        腿部的leg_point_feasi_mask: 6,landing_num,2 2的维度分别代表逆运动学解可能的两个分支
        """
        #选取身体附近所有点
        if points_idx is None:
            _,idx = self.env_pointsmap_voxels._tree.query(W_T_R.t,k=2400)
        else:
            idx = points_idx
            
        near_env_points = (W_T_R.inv()*self.env_pointsmap_voxels.points[idx].T).T
        #取出在身体ESDF范围内的点，范围外的不考虑
        
        inside_body_mask = self.robot_voxels.body_voxels.IsInsideRange(near_env_points)
        body_cf_mask = np.ones_like(inside_body_mask)
        if inside_body_mask.any():
            #使用三维插值
            inrange_collisionfree_mask= self.robot_voxels.body_voxels.TrilinearSample(
                                            self.robot_voxels.body_voxels.esdf_for_env,
                                            near_env_points[inside_body_mask],
                                            outside_value=0.05) >= 0.005
            # ix,iy,iz = self.robot_voxels.body_voxels.Pos2GridIndex(near_env_points[inside_body_mask]).T
            # inrange_collisionfree_mask = self.robot_voxels.body_voxels.esdf[ix,iy,iz]>=0.005
            body_cf_mask[inside_body_mask] = inrange_collisionfree_mask

        #选出附近点中可以被选为可行落脚点的,这里暂时不考虑是否在腿部工作空间内，在循环时才会筛选
        possible_landing_mask = idx<self.env_pointsmap_voxels.landing_count
        if np.sum(possible_landing_mask)<=60 and points_idx is None:
            _,possible_landing_idx = self.env_pointsmap_voxels._landing_tree.query(W_T_R.t,k=100)
        else:
            possible_landing_idx = idx[possible_landing_mask]
        near_landing_points = self.env_pointsmap_voxels.landing_points[possible_landing_idx].T
        near_landing_normals = self.env_pointsmap_voxels.normals[possible_landing_idx].T
        near_landing_points = (W_T_R.inv() * near_landing_points).T #landing_num,3
        near_landing_normals = (W_T_R.R.T @ near_landing_normals).T #landing_num,3

        #构造一个(6,landing_idx,2)的mask 标记每个点的两个解在每个腿内是否可行 (可行: 在可行空间内，且与环境无碰)
        leg_point_feasi_mask = np.zeros((6,possible_landing_idx.shape[0],2),dtype=bool)

        #检查在有效工作空间内
        #找出腿部工作空间的点，判断当前配置下是否存在与环境无碰的可行解
        #创建临时变量，用于debug，检查点
        for i in range(6):
            B_points = self.robot_voxels.kin._R2B(near_landing_points.T,i).T
            #筛选出在腿部的体素边界内的 这里只是选出在腿部体素定义范围内的 不代表腿部在这里一定有解
            inside_leg_mask = self.robot_voxels.leg_voxels.IsInsideRange(B_points)
            B_points = B_points[inside_leg_mask] #N,3
            #筛选在当前腿部可行工作范围内 不仅要求有可行解 同时不能距离腿部基坐标的距离超过30cm
            _flat_index = self.robot_voxels.leg_voxels.Pos2FlatIndex(B_points)
            # inside_leg_range_mask = self.robot_voxels.robot_reachable_legs[_flat_index,:,i] #N,2
            # inside_leg_range_mask = self.robot_voxels.to_bound_dist_flat[_flat_index,i]>=0.042 #N
            inside_leg_range_mask = self.robot_voxels.to_bound_dist_flat[_flat_index,i]>=0.0 #N
            # N,2 & N,1 -> N,2
            inside_leg_range_mask &= (np.linalg.norm(self.robot_voxels.leg_voxels.center[_flat_index],axis=1)<=0.3)

            #法向量可行性 N,2
            norm_feasi_mask = self._LegNormFeasi(near_landing_points[inside_leg_mask],
                                                 near_landing_normals[inside_leg_mask],
                                                 leg_index=i)
            #碰撞可行性，目前先忽略足端的碰撞情况检查,env_esdf分辨率不够，可能会引起误判
            collision_free_mask = self._LegEnvCollisionFree(B_points,W_T_R,i,ignore_end=True)

            # leg_free_mask = np.zeros_like(inside_leg_mask)
            leg_free_mask = np.zeros((inside_leg_mask.shape[0],2),dtype=bool)
            leg_free_mask[inside_leg_mask] = collision_free_mask&norm_feasi_mask&inside_leg_range_mask[:,None]
            #测试取消腿部法向量可行性检查
            # leg_free_mask[inside_leg_mask] = collision_free_mask&inside_leg_range_mask
            leg_point_feasi_mask[i,...]=leg_free_mask
            #检查通过碰撞和法向量筛选后的点

        # print(leg_point_feasi_mask.sum())
        return idx,possible_landing_idx,body_cf_mask, leg_point_feasi_mask

    def GetRobotFeasiCostPoints(self,W_T_R:SE3)->np.ndarray:
        """返回一次代价计算使用的环境候选点，可供优化器固定候选集合。"""
        query_num = min(2400,self.env_pointsmap_voxels.points.shape[0])
        if query_num==0:
            return np.zeros(0,dtype=np.int64)
        _,idx = self.env_pointsmap_voxels._tree.query(W_T_R.t,k=query_num)
        idx = np.asarray(idx,dtype=np.int64).reshape(-1)
        # Optimize会固定points_idx，因此RobotFeasiCost内部基于points_idx is None
        # 的landing fallback不会触发。这里直接把最近landing点并入固定集合，
        # 避免出现腿部代价很高但完全没有落脚点梯度的情况。
        landing_count = self.env_pointsmap_voxels.landing_points.shape[0]
        landing_query_num = min(100,landing_count)
        if landing_query_num>0:
            _,landing_idx = self.env_pointsmap_voxels._landing_tree.query(
                W_T_R.t,k=landing_query_num
            )
            idx = np.unique(np.concatenate([
                idx,np.asarray(landing_idx,dtype=np.int64).reshape(-1)
            ]))
        return idx

    def RobotFeasiCost(
        self,
        W_T_R:SE3,
        points_idx:Union[None|np.ndarray]=None,
        smooth_temperature:float=0.05,
    )->float:
        """
        计算单个机器人位姿的身体碰撞与腿部落脚可行性代价。

        腿部只评估落在 LegVoxels bounds 内的落脚点。工作空间和足端
        距离是分支无关项；法向、IK 有效性和 ankle 碰撞是分支相关项。
        每个候选点选择两个 IK 分支中得分较高者，每条腿再选择得分最高的
        若干候选点计算代价。返回值越小代表位姿越可行。
        """
        body_safe_margin = 0.01
        body_distance_scale = 0.005
        body_top_k = 20
        body_weight = 10.0

        workspace_safe_margin = 0.042
        max_leg_radius = 0.30
        distance_score_scale = 0.01
        normal_score_scale = 0.05
        candidate_top_k = 6  # 可根据实际候选点密度改为 10
        missing_candidate_score = -10.0
        leg_weight = 1.0

        if smooth_temperature<=0.0:
            raise ValueError("smooth_temperature must be positive")

        def _Softplus(value:np.ndarray)->np.ndarray:
            return np.logaddexp(0.0,value)

        def _SmoothMin(left:np.ndarray,right:np.ndarray)->np.ndarray:
            left = np.asarray(left,dtype=np.float64)
            right = np.asarray(right,dtype=np.float64)
            return -smooth_temperature*np.logaddexp(
                -left/smooth_temperature,-right/smooth_temperature
            )

        def _SmoothMax(values:np.ndarray,axis:int)->np.ndarray:
            values = np.asarray(values,dtype=np.float64)
            max_value = np.max(values,axis=axis,keepdims=True)
            result = max_value+smooth_temperature*np.log(
                np.sum(
                    np.exp((values-max_value)/smooth_temperature),
                    axis=axis,
                    keepdims=True,
                )
            )
            return np.squeeze(result,axis=axis)

        def _TopKMean(values:np.ndarray,k:int)->float:
            values = np.asarray(values,dtype=np.float64).reshape(-1)
            if values.size == 0:
                return 0.0
            k = min(int(k),values.size)
            return float(np.mean(np.partition(values,values.size-k)[-k:]))

        # 查询当前身体附近的环境点。
        if points_idx is None:
            idx = self.GetRobotFeasiCostPoints(W_T_R)
        else:
            idx = points_idx
        idx = np.asarray(idx,dtype=np.int64).reshape(-1)

        near_env_points = (W_T_R.inv()*self.env_pointsmap_voxels.points[idx].T).T

        # 身体ESDF采用三线性插值。范围外给一个足够安全的距离，避免体素
        # floor查询造成大面积零梯度，也避免候选点进出bounds时突然跳变。
        if near_env_points.shape[0]>0:
            body_voxels = self.robot_voxels.body_voxels
            body_distance = body_voxels.TrilinearSample(
                body_voxels.esdf_for_env,
                near_env_points,
                outside_value=body_safe_margin+10.0*body_distance_scale,
            )
            body_violation = _Softplus(
                (body_safe_margin-body_distance)/body_distance_scale
            )**2
            body_cost = _TopKMean(body_violation,body_top_k)
        else:
            body_cost = 0.0

        # 取得当前位姿附近的落脚点候选。
        possible_landing_mask = idx<self.env_pointsmap_voxels.landing_count
        landing_point_num = self.env_pointsmap_voxels.landing_points.shape[0]
        if landing_point_num==0:
            possible_landing_idx = np.zeros(0,dtype=np.int64)
        elif np.sum(possible_landing_mask)<=60 and points_idx is None:
            landing_query_num = min(
                100,landing_point_num
            )
            _,possible_landing_idx = self.env_pointsmap_voxels._landing_tree.query(
                W_T_R.t,k=landing_query_num
            )
        else:
            possible_landing_idx = idx[possible_landing_mask]
        possible_landing_idx = np.asarray(possible_landing_idx,dtype=np.int64).reshape(-1)

        near_landing_points = self.env_pointsmap_voxels.landing_points[possible_landing_idx].T
        near_landing_normals = self.env_pointsmap_voxels.normals[possible_landing_idx].T
        near_landing_points = (W_T_R.inv() * near_landing_points).T #landing_num,3
        near_landing_normals = (W_T_R.R.T @ near_landing_normals).T #landing_num,3

        leg_costs = np.zeros(6,dtype=np.float64)
        cos_plane_limit = np.cos(np.deg2rad(75.0))
        cos_branch_limit = np.cos(np.deg2rad(95.0))
        leg_voxels = self.robot_voxels.leg_voxels
        env_voxels = self.env_pointsmap_voxels.voxels

        def _OutsideWorkspaceScores(
            all_B_points:np.ndarray,
            inside_leg_mask:np.ndarray,
            leg_index:int,
            select_num:int,
        )->np.ndarray:
            """按扩展to_bound_dist选择最接近腿工作空间的范围外环境点。"""
            outside_points = all_B_points[~inside_leg_mask]
            if select_num<=0 or outside_points.shape[0]==0:
                return np.zeros(0,dtype=np.float64)
            center_min = leg_voxels._bounds[:,0]+0.5*leg_voxels.voxel_scale
            center_max = (
                leg_voxels._bounds[:,0]
                +(leg_voxels.grid_shape-0.5)*leg_voxels.voxel_scale
            )
            boundary_points = np.clip(outside_points,center_min,center_max)
            boundary_distance = leg_voxels.TrilinearSample(
                self.robot_voxels.to_bound_dist[...,leg_index],
                boundary_points,
                outside_value=-10.0*distance_score_scale,
            )
            # 将体素范围内的有符号距离向外延拓；越接近可行域，值越大。
            extended_distance = boundary_distance-np.linalg.norm(
                outside_points-boundary_points,axis=1
            )
            radial_margin = max_leg_radius-np.linalg.norm(outside_points,axis=1)
            outside_scores = _SmoothMin(
                (extended_distance-workspace_safe_margin)/distance_score_scale,
                radial_margin/distance_score_scale,
            )
            select_num = min(select_num,outside_scores.size)
            selected_index = np.argpartition(
                extended_distance,extended_distance.size-select_num
            )[-select_num:]
            return outside_scores[selected_index]

        for leg_index in range(6):
            all_B_points = self.kin._R2B(near_landing_points.T,leg_index).T
            inside_leg_mask = leg_voxels.IsInsideRange(all_B_points)
            B_points = all_B_points[inside_leg_mask]
            R_points = near_landing_points[inside_leg_mask]
            R_normals = near_landing_normals[inside_leg_mask]

            if B_points.shape[0]==0:
                selected_scores = np.full(
                    candidate_top_k,missing_candidate_score,dtype=np.float64
                )
                outside_scores = _OutsideWorkspaceScores(
                    all_B_points,inside_leg_mask,leg_index,select_num=3
                )
                selected_scores[:outside_scores.size] = outside_scores
                leg_costs[leg_index] = float(np.mean(_Softplus(-selected_scores)**2))
                continue

            flat_index = leg_voxels.Pos2FlatIndex(B_points)

            # 分支无关项：工作空间有符号距离和足端最大半径。
            workspace_distance = leg_voxels.TrilinearSample(
                self.robot_voxels.to_bound_dist[...,leg_index],
                B_points,
                outside_value=-10.0*distance_score_scale,
            )
            workspace_margin = workspace_distance-workspace_safe_margin
            radial_margin = max_leg_radius-np.linalg.norm(B_points,axis=1)
            workspace_feasible_count = int(np.sum(
                (workspace_margin>=0.0)&(radial_margin>=0.0)
            ))
            common_score = _SmoothMin(
                workspace_margin/distance_score_scale,
                radial_margin/distance_score_scale,
            )

            # 腿平面法向约束是公共项，x_b3 朝向是分支项。
            pn = np.column_stack([
                self.kin.leg_base_p[1,leg_index]-R_points[:,1],
                R_points[:,0]-self.kin.leg_base_p[0,leg_index],
                np.zeros(R_points.shape[0],dtype=R_points.dtype),
            ])
            pn /= np.maximum(np.linalg.norm(pn,axis=1,keepdims=True),1e-5)
            normalized_normals = R_normals/np.maximum(
                np.linalg.norm(R_normals,axis=1,keepdims=True),1e-5
            )
            plane_margin = cos_plane_limit-np.abs(
                np.sum(normalized_normals*pn,axis=1)
            )

            B_normals = normalized_normals.copy()
            if leg_index<3:
                B_normals[:,:2] *= -1
            x_b3 = leg_voxels.x_b3[flat_index]
            finite_x_b3 = np.isfinite(x_b3).all(axis=2)
            branch_normal_margin = (
                np.einsum(
                    "njk,nk->nj",np.nan_to_num(x_b3,nan=0.0),-B_normals
                )-cos_branch_limit
            )
            normal_score = _SmoothMin(
                plane_margin[:,None]/normal_score_scale,
                branch_normal_margin/normal_score_scale,
            )

            # ankle 与环境的碰撞距离依赖 IK 分支，足端本身允许接触环境。
            ankle_B = leg_voxels.ankle_pos[flat_index]
            finite_ankle = np.isfinite(ankle_B).all(axis=2)
            ankle_R = self.kin._B2R(
                np.nan_to_num(ankle_B,nan=0.0).reshape(-1,3).T,
                leg_index,
            ).T
            ankle_W = (W_T_R*ankle_R.T).T
            ankle_distance = env_voxels.TrilinearSample(
                self.env_pointsmap_voxels.env_esdf,
                ankle_W,
                outside_value=-distance_score_scale,
            )
            collision_score = (
                ankle_distance.reshape(-1,2)-leg_voxels.ankle_collide_radi
            )/distance_score_scale

            branch_valid = (
                self.robot_voxels.robot_reachable_legs[flat_index,:,leg_index]
                & finite_ankle
                & finite_x_b3
            )
            branch_score = _SmoothMin(normal_score,collision_score)
            score_per_branch = _SmoothMin(common_score[:,None],branch_score)
            score_per_branch[~branch_valid] = missing_candidate_score

            # 若当前体素没有有效分支，保留工作空间负分数作为优化方向。
            has_valid_branch = branch_valid.any(axis=1)
            candidate_scores = common_score.astype(np.float64,copy=True)
            candidate_scores[has_valid_branch] = _SmoothMax(
                score_per_branch[has_valid_branch],axis=1
            )
            # bounds内但超过工作空间的点已经包含在candidate_scores中；若真正
            # 的工作空间点仍少于3个，再从bounds外按扩展to_bound_dist补充。
            outside_scores = _OutsideWorkspaceScores(
                all_B_points,
                inside_leg_mask,
                leg_index,
                select_num=max(0,3-workspace_feasible_count),
            )
            if outside_scores.size>0:
                candidate_scores = np.concatenate([
                    candidate_scores,outside_scores
                ])

            select_num = min(candidate_top_k,candidate_scores.size)
            selected_scores = np.full(
                candidate_top_k,missing_candidate_score,dtype=np.float64
            )
            if select_num>0:
                top_scores = np.partition(
                    candidate_scores,candidate_scores.size-select_num
                )[-select_num:]
                selected_scores[:select_num] = top_scores

            # 分数越高代价越低；候选不足时由缺失分数补齐。
            leg_costs[leg_index] = float(np.mean(_Softplus(-selected_scores)**2))

        worst_leg_cost = float(_SmoothMax(leg_costs,axis=0))
        leg_cost = 0.5*float(np.mean(leg_costs))+0.5*worst_leg_cost
        return float(body_weight*body_cost+leg_weight*leg_cost)


    def VisualizeRobotFeasiCheck(self,
                                 W_T_R:SE3,
                                 check_result:tuple|None=None,
                                 env_stride:int=1,
                                 leg_voxel_stride:int=8):
        """
        可视化RobotFeasiCheck的检查结果。

        显示内容:
        1. 环境降采样点和landing_point。
        2. 当前位姿下的机器人身体体素与六条腿工作空间体素。
        3. KD搜索到的身体附近点，并区分与身体碰撞/无碰。
        4. 六条腿各自的landing_point可行/不可行结果。
        """

        def _TransformRPointsToW(W_T_R:SE3,points_R:np.ndarray)->np.ndarray:
            """把R系下的(N,3)点转换到W系下。"""
            if points_R.shape[0] == 0:
                return points_R.reshape(0,3)
            return (W_T_R*points_R.T).T
        
        def _AddPointCloud(plotter:pv.Plotter,points:np.ndarray,
                           color:str,point_size:float,
                           opacity:float=1.0,label:str|None=None):
            """向plotter添加非空点云。"""
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
        
        if check_result is None:
            check_result = self.RobotFeasiCheck(W_T_R)
        idx,landing_idx,collisionfree_mask,leg_point_feasi_mask = check_result

        env_stride = max(1,int(env_stride))
        leg_voxel_stride = max(1,int(leg_voxel_stride))
        color_list = ["red","green","blue","yellow","magenta","cyan"]
        leg_names = self.robot_voxels.leg_voxels._leg_names

        plotter = pv.Plotter()
        plotter.show_axes()

        env_points = self.env_pointsmap_voxels.points[::env_stride]
        landing_points = self.env_pointsmap_voxels.landing_points[::env_stride]
        _AddPointCloud(plotter,env_points,"lightgray",3,0.18,"env points")
        _AddPointCloud(plotter,landing_points,"black",5,0.22,"landing points")

        body_occ_mask = self.robot_voxels.body_voxels.esdf_flat <= 0.0
        body_points_W = _TransformRPointsToW(
            W_T_R,
            self.robot_voxels.body_voxels.center[body_occ_mask],
        )
        _AddPointCloud(plotter,body_points_W,"orange",9,0.65,"body voxels")

        for i in range(6):
            reachable = self.robot_voxels.robot_reachable_legs[...,i].any(axis=1)
            leg_points_B = self.robot_voxels.leg_voxels.center[reachable][::leg_voxel_stride]
            leg_points_R = self.robot_voxels.kin._B2R(leg_points_B.T,i).T
            leg_points_W = _TransformRPointsToW(W_T_R,leg_points_R)
            _AddPointCloud(
                plotter,
                leg_points_W,
                color_list[i],
                4,
                0.14,
                f"{leg_names[i]} workspace",
            )

        near_env_points_W = self.env_pointsmap_voxels.points[idx]
        _AddPointCloud(
            plotter,
            near_env_points_W[collisionfree_mask],
            "deepskyblue",
            11,
            0.85,
            "near body free",
        )
        _AddPointCloud(
            plotter,
            near_env_points_W[~collisionfree_mask],
            "darkred",
            15,
            1.0,
            "near body collision",
        )

        near_landing_points_W = self.env_pointsmap_voxels.landing_points[landing_idx]
        for i in range(6):
            feasible_mask = leg_point_feasi_mask[i].any(axis=1)
            infeasible_mask = ~feasible_mask
            _AddPointCloud(
                plotter,
                near_landing_points_W[infeasible_mask],
                "dimgray",
                5,
                0.12,
                None,
            )
            _AddPointCloud(
                plotter,
                near_landing_points_W[feasible_mask],
                color_list[i],
                14,
                0.95,
                f"{leg_names[i]} feasible landing",
            )

        plotter.add_legend()
        plotter.show()

    def PointsFeasiCheck(self,W_T_R:SE3,R_points:np.ndarray,R_normals:np.ndarray,leg_indices:np.ndarray,sol_indices:np.ndarray)->bool:
        """
        为支撑轨迹可行性检查设计 输入R系下的点points 与对应的leg_indices 点与腿部索引一一对应

        @input R_points(N,3)R系下支撑足端位置; R_normals(N,3)R系下支撑足端的支撑点法向量;leg_indices(N,)腿部对应的索引; sol_indices(N,)当前腿部对应的逆解的分支

        @output bool
        """
        #提取法向量
        # W_points = W_T_R*(R_points.T).T # N,3
        # normals = self.env_pointsmap_voxels.GetNorm(W_points) #N,3
        # R_normals = (W_T_R.R.T @ normals.T).T #N,3
        #检查法向量可行性

        norm_feasi_mask = self._LegNormFeasi(R_points,R_normals,leg_indices)[np.arange(R_points.shape[0]),sol_indices] #leg_size,2
        if not norm_feasi_mask.all():
            return False
        #计算点所在的的体素索引
        B_points = self.robot_voxels.kin._R2B(R_points.T,leg_indices).T #N,3
        #检查体素内可行性
        if not self.robot_voxels.leg_voxels.IsInsideRange(B_points).all():
            return False
        flat_index = self.robot_voxels.leg_voxels.Pos2FlatIndex(B_points)

        # reachable_mask = self.robot_voxels.robot_reachable_legs[flat_index,sol_indices,leg_indices]#N
        # reachable_mask = self.robot_voxels.to_bound_dist_flat[flat_index,leg_indices] >= 0.04
        reachable_mask = self.robot_voxels.to_bound_dist_flat[flat_index,leg_indices] >= 0.0
        reachable_mask &= np.linalg.norm( self.robot_voxels.leg_voxels.center[flat_index],axis=1 ) <= 0.32

        ankle_pos = np.nan_to_num(self.robot_voxels.leg_voxels.ankle_pos[flat_index,sol_indices],nan=0.0)
        #这里需要满足ankle pos一直在环境的ESDF内，假设这一点可以满足
        ankle_pos = ( W_T_R * self.robot_voxels.kin._B2R(ankle_pos.T,leg_indices) ).T 
        ix,iy,iz = self.env_pointsmap_voxels.voxels.Pos2GridIndex(ankle_pos).T
        ankle_free_mask = (self.env_pointsmap_voxels.env_esdf[ix,iy,iz]>=
                           self.robot_voxels.leg_voxels.ankle_collide_radi)
        return (reachable_mask&ankle_free_mask).all()
        # return (inside_mask&ankle_free_mask).all()

    def FarestPoints(self,point:np.ndarray,direction:np.ndarray,W_T_R:SE3,leg_index:int,distance=0.08,
                     ignore_ankle=False, ignore_end=False)->np.ndarray:
        """
        输入点和方向以及对应腿部的索引，输出在工作空间内沿这个方向能达到最远的距离

        @input 在B系下 point:(3,) direction (3,) direction需要先进行归一化

        @output 在B系下 point(3,)  dist :距离原来点的距离
        """
        delt = self.robot_voxels.leg_voxels.voxel_scale
        # delt = 0.01
        nums = int(distance/delt)
        #可达性条件 这里不需要检查是否超过腿部工作空间 就算超过也会自动被归入最远的grid中
        candidate_points = point[None,:]+direction[None,:]*np.arange(nums)[:,None]*delt #nums,3
        free_mask = self._LegEnvCollisionFree(candidate_points,W_T_R,leg_index,ignore_ankle,ignore_end).any(axis=1)#nums
        if (~free_mask).all():
            # print(f"free_mask={free_mask}")
            # print(f"candidate points\n {candidate_points}")
            # print(f"W_T_R\n {W_T_R}")
            # print(f"leg_index={leg_index}, ignore_ankle={ignore_ankle}, ignore_end={ignore_end}")
            print("Farest points get the original point, point may not inside the body")
            print(f"point: {point}, direction:{direction}")            
            return point, 0

        farest_index = np.max(np.where(free_mask)[0])
        if farest_index == 0:
            print("Farest points get the original point, point may not inside the body")
            print(f"point: {point}, direction:{direction}")
        return candidate_points[farest_index], distance*(farest_index+1)/nums
