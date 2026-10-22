# 定义表面点地图：读取 SLAM 点云或 STL mesh 表面，得到点/法向/平面质量，
# 筛选吸盘可落脚点，再把可落脚和非落脚表面分别降采样成在线查询地图。
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from legged_gym import LEGGED_GYM_ROOT_DIR

@dataclass
class LandingQuality:
    """原始表面点的平面误差和几何可落脚分类。"""

    plane_error: np.ndarray
    blocked_mask: np.ndarray
    valid_mask: np.ndarray


class PointMap:
    """从 SLAM 点云或 STL mesh 构建可落脚点地图。 输入cloud_file类型为.npz会直接加载计算结果，如果是STL或者ply会进行处理

    设计原则：
    1. 同一半径邻域的协方差同时决定原始点的法向和平面误差。
    2. 可落脚点与其余表面点分别降采样，后者仍参与碰撞地图构建。
    3. 每个 voxel 保留最靠近体素中心的原始点及其法向。

    坐标约定：
    - self._raw_points / self.points 均为 (N, 3)，与 PLY 文件坐标一致。
    - GetNorm 输入支持 (3,), (N, 3), (3, N)，输出形状与输入点集合对应。

    注：
    点云处理降采样后为 self.points。
    self.points[:self.landing_count] 是可落脚点，后续点是非落脚表面点。
    self.landing_points 是可落脚点沿法向量向外偏移后的查询点。
    """

    def __init__(
        self,
        target_file: str,
        voxel_size: float = 0.02, #降采样后的点的平均间隔
        suction_diameter: float = 0.04, #吸盘直径
        normal_radius: float = 0.04, #
        plane_error_threshold: float = 0.03,
        edge_buffer_distance: Optional[float] = None,
        edge_normal_angle_threshold: float = np.deg2rad(35.0),
        boundary_angle_gap_threshold: float = np.deg2rad(120.0),
        source_scale: Optional[float] = None,
        surface_sample_spacing: float = 0.01, #对STL文件表面的采样间距，建议取比voxel_size更细一些
        sensor_pose_file: Optional[str] = None,
        # suction_to_foot_offset:float = 0.0385, #吸盘末端到足端的距离，用于偏移landing_point 适用于hex_v4
        suction_to_foot_offset:float = 0.03, #吸盘末端到足端的距离，用于偏移landing_point 适用于hex_magnetic
        verbose: bool = False,
    ):
        self.target_file = target_file
        self.file_type = self._InferFileType(target_file)
        self.original_file = None if self.file_type == "saved" else target_file
        self.saved_file = target_file if self.file_type == "saved" else self._DefaultSavedFile(target_file)
        self.voxel_size = float(voxel_size)
        self.suction_radius = float(suction_diameter) * 0.5
        self.normal_radius = float(normal_radius)
        self.plane_error_threshold = float(plane_error_threshold)
        self.edge_buffer_distance = (
            self.suction_radius if edge_buffer_distance is None else float(edge_buffer_distance)
        )
        self.edge_normal_angle_threshold = float(edge_normal_angle_threshold)
        self.boundary_angle_gap_threshold = float(boundary_angle_gap_threshold)
        self.source_scale = source_scale
        self.surface_sample_spacing = float(surface_sample_spacing)
        self.sensor_pose_file = sensor_pose_file
        self.normal_reference_points: Optional[np.ndarray] = None
        self.verbose = bool(verbose)

        self.mesh_is_watertight: Optional[bool] = None
        self.mesh_min_thickness: Optional[float] = None
        self.mesh_area: Optional[float] = None
        self.mesh_normal_check_offset: Optional[float] = None
        self._source_mesh = None
        self._bounds = np.zeros((3, 2), dtype=np.float32)

        self._raw_points: Optional[np.ndarray] = None
        self._raw_tree: Optional[cKDTree] = None
        self._raw_normals: Optional[np.ndarray] = None
        self._mesh_sample_normals: Optional[np.ndarray] = None
        self._raw_plane_error: Optional[np.ndarray] = None
        self._raw_blocked_mask: Optional[np.ndarray] = None
        self._quality: Optional[LandingQuality] = None
        self._raw_edge_index:Optional[np.ndarray] = None

        # 在线查询使用的降采样后表面点，包含可落脚点和其余表面点。
        # points[:landing_count] 是可落脚点，后续点是非落脚表面点。
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.normals = np.zeros((0, 3), dtype=np.float32)
        self.landing_count = 0
        self._tree: Optional[cKDTree] = None
        # 维护一个属性，保存楞边
        self.edge_points = np.zeros((0,3),dtype=np.float32)
        self._edge_tree :Optional[cKDTree] = None
        #将可行落脚点沿着normals偏移h高度得到的点（h由吸盘底面与foot关节距离决定）
        self.landing_points = np.zeros((0, 3), dtype=np.float32) 
        self.suction_to_foot_offset = suction_to_foot_offset
        self._landing_tree: Optional[cKDTree] = None

        if self.file_type == "saved":
            self._BuildFromSavedFile(self.target_file)
        else:
            if self.file_type == "cloud":
                self.normal_reference_points = self._LoadNormalReferencePoints(sensor_pose_file)
            self._BuildFromRawFile(self.target_file)
            self._WriteSavedFile(self.saved_file)

    def _InferFileType(self, target_file: str) -> str:
        if os.path.isdir(target_file):
            raise ValueError("PointMap expects a file path, not a directory.")
        ext = os.path.splitext(target_file)[1].lower()
        if ext == ".npz":
            return "saved"
        if ext == ".stl":
            return "stl"
        if ext in (".ply", ".pcd", ".xyz", ".xyzn", ".pts"):
            return "cloud"
        raise ValueError(f"unsupported PointMap file suffix: {ext}")

    def _DefaultSavedFile(self, target_file: str) -> str:
        root, _ = os.path.splitext(target_file)
        return root + "_point_map.npz"

    def _LoadNormalReferencePoints(self, sensor_pose_file: Optional[str]) -> Optional[np.ndarray]:
        if sensor_pose_file is None:
            return None
        ext = os.path.splitext(sensor_pose_file)[1].lower()
        if ext == ".npy":
            arr = np.load(sensor_pose_file).astype(np.float32)
        elif ext == ".npz":
            data = np.load(sensor_pose_file)
            for key in ("normal_reference_points", "sensor_positions", "poses"):
                if key in data:
                    arr = data[key].astype(np.float32)
                    break
            else:
                arr = data[list(data.keys())[0]].astype(np.float32)
        else:
            arr = np.loadtxt(sensor_pose_file, delimiter="," if ext == ".csv" else None).astype(np.float32)

        if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
            arr = arr[:, :3, 3]
        elif arr.ndim == 2 and arr.shape == (4, 4):
            arr = arr[:3, 3][None, :]
        elif arr.ndim == 2 and arr.shape[1] >= 3:
            arr = arr[:, :3]
        elif arr.shape == (3,):
            arr = arr[None, :]
        else:
            raise ValueError("sensor_pose_file must contain (N,3), (N,>=3), (N,4,4), (4,4), or (3,) data")

        finite_mask = np.isfinite(arr).all(axis=1)
        arr = arr[finite_mask]
        if arr.shape[0] == 0:
            raise ValueError("sensor_pose_file has no finite reference points")
        return arr.astype(np.float32)

    def _BuildFromRawFile(self, target_file: str):
        """从原始点云或 STL 文件重建地图。"""
        self.file_type = self._InferFileType(target_file)
        if self.file_type == "saved":
            raise ValueError("_BuildFromRawFile expects a raw cloud or STL file, not .npz")
        self.target_file = target_file
        self.original_file = target_file
        self.saved_file = self._DefaultSavedFile(target_file)
        if self.file_type == "stl":
            self._LoadRawStlFile(target_file)
        elif self.file_type == "cloud":
            self._LoadRawCloudFile(target_file)
        else:
            raise ValueError(f"unsupported raw file type: {self.file_type}")
        self._FinalizeRawMapArrays()
        self._Build()

    def _FinalizeRawMapArrays(self):
        if self._raw_points is None:
            raise RuntimeError("raw points have not been loaded")
        finite_mask = np.isfinite(self._raw_points).all(axis=1)
        if self._raw_normals is not None:
            finite_mask &= np.isfinite(self._raw_normals).all(axis=1)
        if self._mesh_sample_normals is not None:
            finite_mask &= np.isfinite(self._mesh_sample_normals).all(axis=1)
        self._raw_points = self._raw_points[finite_mask].astype(np.float32)
        if self._raw_normals is not None:
            self._raw_normals = self._NormalizeRows(self._raw_normals[finite_mask].astype(np.float32))
        if self._mesh_sample_normals is not None:
            self._mesh_sample_normals = self._NormalizeRows(
                self._mesh_sample_normals[finite_mask].astype(np.float32)
            )
        if self._raw_plane_error is not None:
            self._raw_plane_error = self._raw_plane_error[finite_mask].astype(np.float32)
        if self._raw_points.shape[0] == 0:
            raise ValueError(f"source has no finite surface points: {self.target_file}")
        self._raw_tree = cKDTree(self._raw_points)
        self.points = self._raw_points
        self.normals = np.zeros_like(self.points) if self._raw_normals is None else self._raw_normals
        if self._raw_normals is None:
            self.normals[:, 2] = 1.0
        self.landing_count = self.points.shape[0]
        #self._RebuildQueryTrees()
        self._UpdateBounds()

    def _UpdateBounds(self):
        if self._raw_points is None or self._raw_points.shape[0] == 0:
            self._bounds = np.zeros((3, 2), dtype=np.float32)
        else:
            self._bounds = np.column_stack([self._raw_points.min(axis=0), self._raw_points.max(axis=0)]).astype(np.float32)
        #为了避免机器人在边界位置超出point map边缘，将最低和最高边界扩展机器人最外侧包络面 _bounds 3,2
        self._bounds[:,0] -= 0.54
        self._bounds[:,1] += 0.54
        print(self._bounds)

    def _NormalizeRows(self, values: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norm, 1e-12)

    def _LoadRawCloudFile(self, target_file: str):
        self._raw_points = self._ReadCloud(target_file)
        if self.source_scale is None:
            extent = np.ptp(self._raw_points, axis=0)
            self.source_scale = 0.001 if float(np.max(extent)) > 50.0 else 1.0
        self._raw_points = (self._raw_points * float(self.source_scale)).astype(np.float32)
        if self.normal_reference_points is not None:
            self.normal_reference_points = (self.normal_reference_points * float(self.source_scale)).astype(np.float32)

    def _LoadRawStlFile(self, target_file: str):
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("PointMap requires trimesh to read and sample STL files.") from exc
        if self.verbose:
            print("Sampling from STL file and calculating the norm~")
        mesh = trimesh.load_mesh(target_file, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) != 1:
                raise ValueError("PointMap only supports one STL mesh; scene contains multiple geometries.")
            mesh = next(iter(mesh.geometry.values()))
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise ValueError(f"failed to read valid STL mesh from {target_file}")

        if self.source_scale is None:
            extent = np.ptp(np.asarray(mesh.vertices), axis=0)
            self.source_scale = 0.001 if float(np.max(extent)) > 50.0 else 1.0
        if float(self.source_scale) != 1.0:
            mesh.apply_scale(float(self.source_scale))

        self._RepairMeshForSampling(mesh)
        self.mesh_is_watertight = bool(mesh.is_watertight)
        if not self.mesh_is_watertight:
            raise ValueError("STL mesh must be watertight to orient normals outward reliably.")
        self.mesh_area = float(mesh.area)
        self.mesh_min_thickness = self._ComputeMeshMinThickness(mesh)
        self._source_mesh = mesh
        
        points, normals = self._SampleMeshSurface(mesh, self.surface_sample_spacing)
        normals = self._OrientMeshNormalsOutward(mesh, points, normals)
        self._raw_points = points.astype(np.float32)
        # 网格法向只用来确定 PCA 法向的正负，平面误差由采样邻域计算。
        self._mesh_sample_normals = normals.astype(np.float32)

    def _RepairMeshForSampling(self, mesh):
        try:
            mesh.update_faces(mesh.unique_faces())
        except Exception:
            pass
        try:
            mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
        except Exception:
            pass
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            mesh.merge_vertices()
        except Exception:
            pass
        try:
            mesh.fix_normals()
        except Exception:
            pass

    def _SampleMeshSurface(self, mesh, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("PointMap requires trimesh to sample STL surface.") from exc

        spacing = max(float(spacing), 1e-4)
        #面积/voxel区域面积=采样数量
        target_count = max(int(np.ceil(float(mesh.area) / (spacing * spacing))), 1)
        points = None
        face_idx = None
        try:
            points, face_idx = trimesh.sample.sample_surface_even(
                mesh,
                count=target_count,
                radius=spacing * 0.5,
            )
        except Exception as exc:
            warnings.warn(f"sample_surface_even failed, fallback to random surface sampling: {exc}")

        if points is None or len(points) < max(1, int(target_count * 0.3)):
            oversample = target_count * 3
            points, face_idx = trimesh.sample.sample_surface(mesh, count=oversample)
            points, face_idx = self._VoxelUniqueSurfaceSamples(points, face_idx, spacing)

        normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
        return np.asarray(points, dtype=np.float32), self._NormalizeRows(normals)

    def _VoxelUniqueSurfaceSamples(
        self,
        points: np.ndarray,
        face_idx: np.ndarray,
        spacing: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        min_bound = points.min(axis=0)
        voxel_idx = np.floor((points - min_bound[None, :]) / spacing).astype(np.int64)
        picked = {}
        for i, key_arr in enumerate(voxel_idx):
            key = tuple(int(v) for v in key_arr)
            if key not in picked:
                picked[key] = i
        idx = np.fromiter(picked.values(), dtype=np.int64)
        return points[idx], face_idx[idx]

    def _OrientMeshNormalsOutward(
        self,
        mesh,
        points: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        normals = self._NormalizeRows(normals.astype(np.float32))
        if self.mesh_min_thickness is not None and np.isfinite(self.mesh_min_thickness):
            eps = min(self.surface_sample_spacing * 0.05, float(self.mesh_min_thickness) * 0.25)
        else:
            eps = self.surface_sample_spacing * 0.05
        eps = max(float(eps), 1e-4)
        self.mesh_normal_check_offset = eps
        try:
            plus_inside = mesh.contains(points + normals * eps)
            minus_inside = mesh.contains(points - normals * eps)
            flip = plus_inside & (~minus_inside)
            ambiguous = plus_inside == minus_inside
            if np.any(ambiguous):
                warnings.warn(
                    f"{int(np.count_nonzero(ambiguous))} STL sample normals have ambiguous inside/outside check; "
                    "keep their STL face normal direction."
                )
            out = normals.copy()
            out[flip] *= -1.0
            return out
        except Exception as exc:
            warnings.warn(f"failed to validate mesh normal orientation; use STL face normals: {exc}")
            return normals

    def _ComputeMeshMinThickness(self, mesh) -> float:
        if not bool(mesh.is_watertight):
            return float("nan")
        sample_count = min(20000, max(int(mesh.faces.shape[0] * 4), 1))
        if sample_count <= 0:
            return float("nan")
        try:
            import trimesh
            points, face_idx = trimesh.sample.sample_surface(mesh, count=sample_count)
            normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
            eps = max(float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0))) * 1e-6, 1e-6)
            origins = points - normals * eps
            directions = -normals
            locations, ray_index, _ = mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=True,
            )
            if len(ray_index) == 0:
                return float("nan")
            dist = np.linalg.norm(locations - origins[ray_index], axis=1)
            valid = dist > eps * 10.0
            if not np.any(valid):
                return float("nan")
            return float(np.min(dist[valid]))
        except Exception as exc:
            warnings.warn(f"failed to compute mesh minimum thickness: {exc}")
            return float("nan")

    def _Build(self):
        """构建完整在线地图：法向/平面误差 -> 吸盘落脚质量 -> voxel 代表点。"""
        if self._raw_points is None or self._raw_tree is None:
            raise RuntimeError("raw source has not been loaded; use _BuildFromRawFile first")
        if self._raw_normals is None or self._raw_plane_error is None:
            if self.verbose:
                print("cal norm and plane error")
            self.NormCal()
        if self.verbose:
            print("detect edge")
        self.DetectRawBlockedArea()
        if self.verbose:
            print("filter")
        self.FilterLandingCandidates()
        if self.verbose:
            print("down sample")
        self.DownSample()

    def _WriteSavedFile(self, saved_file: Optional[str] = None):
        """把处理后的点云地图保存为 npz 文件。"""
        if self._raw_points is None:
            raise RuntimeError("raw surface points are unavailable; rebuild the map from its source file")
        saved_file = self.saved_file if saved_file is None else saved_file
        saved_dir = os.path.dirname(saved_file)
        os.makedirs(saved_dir or ".", exist_ok=True)
        if self._quality is None:
            self.FilterLandingCandidates()
        if self._tree is None or self.points.shape[0] == 0:
            self.DownSample()
        np.savez_compressed(
            saved_file,
            map_version=np.array([3], dtype=np.int32),
            # file_type=np.array([self.file_type]),
            original_file=np.array(["" if self.original_file is None else self.original_file]),
            source_scale=np.array([np.nan if self.source_scale is None else float(self.source_scale)], dtype=np.float32),
            # surface_sample_spacing=np.array([self.surface_sample_spacing], dtype=np.float32),
            # mesh_is_watertight=np.array([False if self.mesh_is_watertight is None else bool(self.mesh_is_watertight)], dtype=np.bool_),
            # mesh_min_thickness=np.array([np.nan if self.mesh_min_thickness is None else float(self.mesh_min_thickness)], dtype=np.float32),
            # mesh_normal_check_offset=np.array([np.nan if self.mesh_normal_check_offset is None else float(self.mesh_normal_check_offset)], dtype=np.float32),
            # mesh_area=np.array([np.nan if self.mesh_area is None else float(self.mesh_area)], dtype=np.float32),
            bounds=self._bounds,
            raw_points=self._raw_points,
            # voxel_size=np.array([self.voxel_size], dtype=np.float32),
            # suction_radius=np.array([self.suction_radius], dtype=np.float32),
            # normal_radius=np.array([self.normal_radius], dtype=np.float32),
            points=self.points,
            normals=self.normals,
            landing_count=np.array([self.landing_count], dtype=np.int64),
            edge_points = self.edge_points
        )
        self.saved_file = saved_file
        if self.verbose:
            print(f"save point map to {saved_file}")

    def _BuildFromSavedFile(self, saved_file: str):
        """从 npz 保存文件读取处理结果。"""
        if not os.path.exists(saved_file):
            raise FileNotFoundError(saved_file)
        has_bounds = False
        with np.load(saved_file) as data:
            version = int(data["map_version"][0]) if "map_version" in data else 0
            if version < 2:
                warnings.warn("point map uses the old filtering rules; rebuild it from the raw source")
            if "raw_points" in data:
                self._raw_points = data["raw_points"].astype(np.float32)
            else:
                warnings.warn("point map has no raw_points; rebuild it from the raw source to display original samples")
            # if "file_type" in data:
            #     self.file_type = str(data["file_type"][0])
            if "original_file" in data:
                original_file = str(data["original_file"][0])
                self.original_file = None if original_file == "" else original_file
            if "source_scale" in data and np.isfinite(data["source_scale"][0]):
                self.source_scale = float(data["source_scale"][0])
            # if "surface_sample_spacing" in data:
            #     self.surface_sample_spacing = float(data["surface_sample_spacing"][0])
            # if "mesh_is_watertight" in data:
            #     self.mesh_is_watertight = bool(data["mesh_is_watertight"][0])
            # if "mesh_min_thickness" in data:
            #     self.mesh_min_thickness = float(data["mesh_min_thickness"][0])
            # if "mesh_normal_check_offset" in data:
            #     self.mesh_normal_check_offset = float(data["mesh_normal_check_offset"][0])
            # if "mesh_area" in data:
            #     self.mesh_area = float(data["mesh_area"][0])
            if "bounds" in data:
                self._bounds = data["bounds"].astype(np.float32)
                has_bounds = True
            self.points = data["points"].astype(np.float32)
            self.normals = data["normals"].astype(np.float32)
            self.landing_count = int(data["landing_count"][0])
            self.edge_points = data["edge_points"].astype(np.float32)
        self._raw_normals = None
        self._mesh_sample_normals = None
        self._raw_plane_error = None
        self._raw_blocked_mask = None
        self._quality = None
        self._raw_tree = None
        self._RebuildQueryTrees()
        if not has_bounds:
            self._bounds = np.column_stack([self.points.min(axis=0), self.points.max(axis=0)]).astype(np.float32)
        self.target_file = saved_file
        self.saved_file = saved_file
        if self.verbose:
            print(f"load point map from {saved_file}")

    def _RebuildQueryTrees(self):
        self._tree = cKDTree(self.points) if self.points.shape[0] > 0 else None
        # 沿 normal 方向对 landing_points 延伸 suction2footend 距离，也就是从吸盘底部到 foot 关节位置。
        self.landing_points = (
            self.points[:self.landing_count]
            + self.normals[:self.landing_count] * self.suction_to_foot_offset
        )
        self._landing_tree = (
            cKDTree(self.landing_points)
            if self.landing_points.shape[0] > 0 else None
        )
        self._edge_tree = (
            cKDTree(self.edge_points)
            if self.edge_points.shape[0]>0 else None
        )

    def _ReadCloud(self, cloud_file: str) -> np.ndarray:
        """读取 PLY 点云。忽略文件自带法向，只使用点重新计算 PCA 法向。"""
        cloud = o3d.io.read_point_cloud(cloud_file)
        cloud = cloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
        points = np.asarray(cloud.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"failed to read valid point cloud from {cloud_file}")
        return points

    def _AsPointsNx3(self, points: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """返回 (N,3), is_single, was_3xN。"""
        points = np.asarray(points, dtype=np.float32)
        if points.shape == (3,):
            return points[None, :], True, False
        if points.ndim == 2 and points.shape[1] == 3:
            return points, False, False
        if points.ndim == 2 and points.shape[0] == 3:
            return points.T, False, True
        raise ValueError("points must have shape (3,), (N,3), or (3,N)")

    # def KDsearch(self, point: np.ndarray, k: int = 500) -> np.ndarray:
    #     """查询离 point 最近的 k 个在线表面点，返回 self.points 的索引。"""
    #     query, is_single, _ = self._AsPointsNx3(point)
    #     if not is_single:
    #         query = query[:1]
    #     if self._tree is None or self.points.shape[0] == 0:
    #         return np.zeros(0, dtype=np.int64)
    #     k = min(int(k), self.points.shape[0])
    #     _, idx = self._tree.query(query[0], k=k)
    #     return np.atleast_1d(idx).astype(np.int64)
    

    def GetBlockedMask(self, points: np.ndarray) -> np.ndarray:
        """查询点最近在线表面点是否为 blocked。"""
        if self._tree is None or self.points.shape[0] == 0:
            query, is_single, _ = self._AsPointsNx3(points)
            out = np.ones(query.shape[0], dtype=np.bool_)
            return bool(out[0]) if is_single else out
        query, is_single, _ = self._AsPointsNx3(points)
        _, idx = self._tree.query(query, k=1)
        out = np.asarray(idx) >= self.landing_count
        return bool(out[0]) if is_single else out

    def NormCal(self, radius: Optional[float] = None, min_neighbors: int = 12):
        """每点查询一次半径邻域，用其协方差同时求 PCA 法向和平面误差。"""
        radius = self.normal_radius if radius is None else float(radius)
        if radius <= 0.0:
            raise ValueError("normal radius must be positive")
        points = self._raw_points
        normals = np.zeros_like(points)
        plane_error = np.full(points.shape[0], np.inf, dtype=np.float32)

        for start in range(0, len(points), 2048):
            neighborhoods = self._raw_tree.query_ball_point(points[start:start + 2048], r=radius)
            for i, idx in enumerate(neighborhoods, start):
                if len(idx) < max(3, int(min_neighbors)):
                    continue
                neigh = points[idx].astype(np.float64)
                centered = neigh - neigh.mean(axis=0)
                covariance = centered.T @ centered / len(idx)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                normals[i] = eigenvectors[:, 0]
                plane_error[i] = np.sqrt(max(float(eigenvalues[0]), 0.0))

        invalid = ~np.isfinite(plane_error)
        if np.any(invalid):
            if self._mesh_sample_normals is not None:
                normals[invalid] = self._mesh_sample_normals[invalid]
            elif np.any(~invalid):
                valid_tree = cKDTree(points[~invalid])
                _, nearest = valid_tree.query(points[invalid])
                normals[invalid] = normals[~invalid][nearest]
            else:
                normals[invalid, 2] = 1.0

        if self._mesh_sample_normals is not None:
            flip = np.einsum("ij,ij->i", normals, self._mesh_sample_normals) < 0.0
            normals[flip] *= -1.0
        else:
            normals = self._OrientRawCloudNormals(points, normals)
        self._raw_normals = normals
        self._raw_plane_error = plane_error

    def _OrientRawCloudNormals(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        if self.normal_reference_points is not None:
            if self.normal_reference_points.shape[0] == 1:
                return self._OrientNormalsTowardsCameraLocation(
                    points,
                    normals,
                    self.normal_reference_points[0],
                )
            return self._OrientNormalsWithSensorPositions(points, normals)
        return normals

    def _OrientNormalsTowardsCameraLocation(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        camera_location: np.ndarray,
    ) -> np.ndarray:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        cloud.orient_normals_towards_camera_location(np.asarray(camera_location, dtype=np.float64).reshape(3))
        return np.asarray(cloud.normals, dtype=np.float32)

    def _OrientNormalsWithSensorPositions(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        sensor_tree = cKDTree(self.normal_reference_points)
        _, sensor_idx = sensor_tree.query(points, k=1)
        ref = self.normal_reference_points[np.asarray(sensor_idx)]
        to_sensor = ref - points
        dot = np.einsum("ij,ij->i", normals, to_sensor)
        flip = dot < 0.0
        normals[flip] *= -1.0
        return normals
        # out = normals.copy()
        # out[flip] *= -1.0
        # return out

    def _OrientNormalsWithReference(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        reference_point: Optional[np.ndarray],
        towards_reference: bool = True,
    ) -> np.ndarray:
        """按参考点统一法向正负。

        PCA 法向只有轴方向，没有正负。对无传感器点云，边缘检测使用 abs(dot)
        可避免正负影响；但摆动腿沿法向靠近表面时必须给出方向。若提供
        reference_point，则把法向翻到指向/背向参考点。
        """
        if reference_point is None:
            return normals
        ref = np.asarray(reference_point, dtype=np.float32).reshape(1, 3)
        to_ref = ref - points
        dot = np.einsum("ij,ij->i", normals, to_ref)
        flip = dot < 0.0 if towards_reference else dot > 0.0
        out = normals.copy()
        out[flip] *= -1.0
        return out

    def DetectRawBlockedArea(self) -> np.ndarray:
        """检测尖锐边和开放边界，并排除其吸附半径范围内的表面点。"""
        if self._raw_normals is None:
            self.NormCal()

        points = self._raw_points
        normals = self._raw_normals
        normal_edge = np.zeros(points.shape[0], dtype=np.bool_)
        boundary_gap = np.zeros(points.shape[0], dtype=np.bool_)
        valid_normal = np.isfinite(self._raw_plane_error)
        angle_limit = np.cos(self.edge_normal_angle_threshold)
        for start in range(0, len(points), 2048):
            neighborhoods = self._raw_tree.query_ball_point(
                points[start:start + 2048], r=self.normal_radius
            )
            for i, neighbors in enumerate(neighborhoods, start):
                if not valid_normal[i]:
                    boundary_gap[i] = True
                    continue
                neigh_idx = np.asarray(neighbors, dtype=np.int64)
                neigh_idx = neigh_idx[neigh_idx != i]
                comparison_idx = neigh_idx[valid_normal[neigh_idx]]
                if comparison_idx.size:
                    cos_angle = np.abs(np.clip(normals[comparison_idx] @ normals[i], -1.0, 1.0))
                    normal_edge[i] = np.min(cos_angle) < angle_limit
                vec = points[neigh_idx] - points[i]
                tangent0 = self._AnyTangent(normals[i])
                tangent1 = np.cross(normals[i], tangent0)
                tangent1 /= max(np.linalg.norm(tangent1), 1e-12)
                x = vec @ tangent0
                y = vec @ tangent1
                valid = (x * x + y * y) > 1e-12
                if np.count_nonzero(valid) < 3:
                    boundary_gap[i] = True
                    continue
                angle = np.sort(np.arctan2(y[valid], x[valid]))
                gaps = np.diff(np.r_[angle, angle[0] + 2.0 * np.pi])
                boundary_gap[i] = np.max(gaps) > self.boundary_angle_gap_threshold

        feature_mask = boundary_gap | normal_edge
        self._raw_edge_index = np.flatnonzero(feature_mask)
        self.edge_points = points[self._raw_edge_index]
        self._edge_tree = cKDTree(self.edge_points) if self.edge_points.size else None
        blocked_mask = feature_mask.copy()
        if self._edge_tree is not None and self.edge_buffer_distance > 0.0:
            distance, _ = self._edge_tree.query(points)
            blocked_mask |= distance < self.edge_buffer_distance

        self._raw_blocked_mask = blocked_mask
        return blocked_mask

    def FilterLandingCandidates(self) -> LandingQuality:
        """按照边界距离和平面误差筛选几何可落脚点。"""
        if self._raw_normals is None:
            self.NormCal()
        if self._raw_blocked_mask is None:
            self.DetectRawBlockedArea()

        plane_error = self._raw_plane_error
        blocked_mask = self._raw_blocked_mask
        valid_mask = (~blocked_mask) & (plane_error <= self.plane_error_threshold)

        self._quality = LandingQuality(
            plane_error=plane_error,
            blocked_mask=blocked_mask,
            valid_mask=valid_mask,
        )
        return self._quality

    def _AnyTangent(self, normal: np.ndarray) -> np.ndarray:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(normal @ ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tangent = np.cross(normal, ref)
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        return tangent.astype(np.float32)

    def DownSample(self):
        """对可落脚点和其余表面点分别采样，保留各格最靠近中心的原始点。"""
        if self._quality is None:
            self.FilterLandingCandidates()

        valid_idx = np.where(self._quality.valid_mask)[0]
        if valid_idx.size == 0:
            warnings.warn("no valid landing point after filtering; all surface points remain in the collision map")

        picked_idx = self._VoxelPickRawIndices(valid_idx)
        nonlanding_idx = np.where(~self._quality.valid_mask)[0]
        picked_nonlanding_idx = self._VoxelPickRawIndices(nonlanding_idx)

        combined_idx = np.concatenate([picked_idx, picked_nonlanding_idx]).astype(np.int64)
        self.points = self._raw_points[combined_idx]
        self.normals = self._raw_normals[combined_idx]
        self.landing_count = picked_idx.shape[0]
        self._RebuildQueryTrees()

    def _VoxelPickRawIndices(self, raw_idx: np.ndarray) -> np.ndarray:
        if raw_idx.size == 0:
            return np.zeros(0, dtype=np.int64)

        points = self._raw_points[raw_idx].astype(np.float64)
        origin = self._bounds[:, 0].astype(np.float64)
        voxel_index = np.floor((points - origin) / self.voxel_size).astype(np.int64)
        center = origin + (voxel_index + 0.5) * self.voxel_size
        distance2 = np.sum((points - center) ** 2, axis=1)
        _, group = np.unique(voxel_index, axis=0, return_inverse=True)
        order = np.lexsort((raw_idx, distance2, group))
        _, first = np.unique(group[order], return_index=True)
        return raw_idx[order[first]]

    def GetNorm(
        self,
        points: np.ndarray,
        reference_point: Optional[np.ndarray] = None,
        towards_reference: bool = True,
    ) -> np.ndarray:
        """查询点最近 landing 点，返回对应的法向量"""
        query, is_single, was_3xN = self._AsPointsNx3(points)
        if self._landing_tree is None or self.landing_points.shape[0] == 0:
            out = np.full(query.shape[0], -1, dtype=np.int64)
        else:
            _, landing_idx = self._landing_tree.query(query, k=1)
            out = np.asarray(landing_idx, dtype=np.int64)
        if is_single:
            return self.normals[out[0]]
            # return int(out[0])
        return self.normals[out]

    def _PlotSourceMesh(self, plotter, pv, opacity: float = 0.12):
        if self._InferFileType(self.original_file) == 'stl' or 'cloud':
            mesh = pv.read(self.original_file)
        else:
            return

        if self.source_scale is not None and float(self.source_scale) != 1.0:
            mesh.points *= float(self.source_scale)

        if mesh.n_cells > 0:
            # triangle mesh
            plotter.add_mesh(
                mesh,
                color="lightgray",
                opacity=opacity,
                show_edges=True,
                edge_color="silver",
            )
        else:
            # pure point cloud
            plotter.add_mesh(
                mesh,
                color="lightgray",
                opacity=opacity,
                point_size=5,
                render_points_as_spheres=True,
            )

    def _SourceInfoText(self) -> str:
        lines = [f"file type: {self.file_type}"]
        if self.original_file is not None:
            lines.append(f"raw: {os.path.basename(self.original_file)}")
        lines.append(f"saved: {os.path.basename(self.saved_file)}")
        if self.file_type == "stl":
            lines.append(f"watertight: {self.mesh_is_watertight}")
            if self.mesh_min_thickness is not None and np.isfinite(self.mesh_min_thickness):
                lines.append(f"min thickness: {self.mesh_min_thickness:.4f} m")
            else:
                lines.append("min thickness: unavailable")
            if self.mesh_normal_check_offset is not None and np.isfinite(self.mesh_normal_check_offset):
                lines.append(f"normal check offset: {self.mesh_normal_check_offset:.6f} m")
            lines.append(f"sample spacing: {self.surface_sample_spacing:.3f} m")
        else:
            if self.normal_reference_points is not None:
                lines.append(f"normal: towards nearest reference ({self.normal_reference_points.shape[0]})")
            else:
                lines.append("normal: raw PCA sign")
        return "\n".join(lines)

    def PlotProcessingStages(
        self,
        point_size: float = 3.0,
        show_normals: bool = True,
        normal_length: float = 0.05,
        normal_max_count: int = 1000,
        show: bool = True,
        off_screen: bool = False,
        screenshot_path: Optional[str] = None,
    ):
        """用 PyVista 显示点云处理阶段。

        布局为左侧三行、右侧一个大图：
        1. 原始点云
        2. 降采样后的可落脚点
        3. 降采样后的非落脚表面点及边界特征点
        4. 右侧叠加以上三部分；黑色箭头为 PCA 计算后的下采样点法向。
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError("PlotProcessingStages requires pyvista") from exc

        if off_screen or screenshot_path is not None:
            try:
                pv.start_xvfb()
            except Exception:
                pass

        raw_points = (
            self._raw_points
            if self._raw_points is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        downsampled_points = self.points[:self.landing_count]
        downsampled_nonlanding_points = self.points[self.landing_count:]

        if screenshot_path is not None:
            off_screen = True
        plotter = pv.Plotter(shape="2|2", off_screen=off_screen, window_size=(1600, 1000))
        plotter.set_background("white")

        def add_cloud(points: np.ndarray, color: str, size: float, opacity: float = 1.0):
            if points.size == 0:
                return
            plotter.add_mesh(
                pv.PolyData(points),
                color=color,
                point_size=size,
                opacity=opacity,
                render_points_as_spheres=True,
            )

        # 左侧三行：单独看每个阶段。
        plotter.subplot(0)
        # self._PlotSourceMesh(plotter, pv, opacity=0.08)
        add_cloud(raw_points, "lightgray", point_size+2, 1.0)
        # plotter.add_text(f"Raw surface samples\nN={raw_points.shape[0]}\n{self._SourceInfoText()}", font_size=10, color="black")
        plotter.show_axes()

        plotter.subplot(1)
        self._PlotSourceMesh(plotter, pv, opacity=0.06)
        add_cloud(downsampled_points, "royalblue", point_size, 1.0)
        # plotter.add_text(
        #     f"Voxel downsample landing points \nlanding={downsampled_points.shape[0]}",
        #     font_size=10,
        #     color="black",
        # )
        plotter.show_axes()

        plotter.subplot(2)
        self._PlotSourceMesh(plotter, pv, opacity=0.06)
        add_cloud(downsampled_nonlanding_points, "black", point_size + 1.0, 0.75)
        add_cloud(self.edge_points, "orange", point_size + 1.0, 0.8)
        # plotter.add_text(
        #     f"Voxel nonlanding points\nN={downsampled_nonlanding_points.shape[0]}"
        #     f"\nedge/boundary={self.edge_points.shape[0]}",
        #     font_size=10,
        #     color="black",
        # )
        plotter.show_axes()

        # 右侧大图：三类点叠加。
        plotter.subplot(3)
        self._PlotSourceMesh(plotter, pv, opacity=0.08)
        add_cloud(raw_points, "lightgray", max(point_size +2.5, 1.0), 0.18)
        add_cloud(self.landing_points, "tomato", point_size + 3, 1.0)
        add_cloud(downsampled_nonlanding_points, "black", point_size + 2, 1.0)
        normal_text = ""
        if (
            show_normals
            and downsampled_points.shape[0] > 0
        ):
            arrow_count = min(int(normal_max_count), downsampled_points.shape[0])
            if arrow_count > 0:
                sample_idx = np.linspace(0, downsampled_points.shape[0] - 1, arrow_count, dtype=np.int64)
                arrow_points = downsampled_points[sample_idx]
                landing_normals = self.normals[:self.landing_count]
                arrow_dirs = landing_normals[sample_idx]
                arrow_dirs = arrow_dirs / np.maximum(np.linalg.norm(arrow_dirs, axis=1, keepdims=True), 1e-8)
                plotter.add_arrows(
                    arrow_points,
                    arrow_dirs,
                    mag=float(normal_length)*2.0,
                    color="black",
                )
                normal_text = f"\ncomputed normals: black arrows ({arrow_count})"
        # plotter.add_text(
        #     "Overlay\nraw: gray\nlanding: red\nnonlanding: black"
        #     + normal_text + "\n" + self._SourceInfoText(),
        #     font_size=10,
        #     color="black",
        # )
        plotter.show_axes()

        plotter.link_views()
        plotter.view_isometric()

        if show or screenshot_path is not None:
            plotter.show(screenshot=screenshot_path)
        return plotter


if __name__=='__main__':
    point_map = PointMap(
        # "/home/sharpa/valerian_ws/legged_gym/resources/environments/sutructure1/complex_surface.STL",
        # "/home/sharpa/valerian_ws/legged_gym/resources/environments/sutructure1/complex_surface_point_map.npz",
        # LEGGED_GYM_ROOT_DIR+"/resources/environments/sutructure1/complex_surface.STL",
        # LEGGED_GYM_ROOT_DIR+"/resources/environments/structure2/complex_surface4.STL",
        LEGGED_GYM_ROOT_DIR+"/resources/environments/structure2/complex_surface4_point_map.npz",
        # "/home/val/BIH_ws/legged_gym/resources/environments/sutructure1/complex_surface_point_map.npz",
        verbose=True,
    )
    print(f"source_scale={point_map.source_scale}")
    # print(f"raw_points={point_map._raw_points.shape[0]}")
    # print(f"blocked_raw={int(point_map._quality.blocked_mask.sum())}")
    # print(f"valid_raw={int(point_map._quality.valid_mask.sum())}")
    print(f"map_points={point_map.points.shape[0]}")
    print(f"landing_points={point_map.landing_count}")
    print(f"nonlanding_points={point_map.points.shape[0] - point_map.landing_count}")
    point_map.PlotProcessingStages()
