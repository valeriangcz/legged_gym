"""Independent cached-map loading, GPU robot checks and batched SE(3) math.

Use ``HexStateTorch.from_files(point_map_file, voxel_dir)`` without constructing
any legacy HexState/Kinematic/Voxels classes. Missing caches are not rebuilt.
RobotVoxelData is the explicit in-memory data interface. No legacy robot object
is accepted, retained or used as a runtime dependency.
Points/vectors use (..., 3), unlike the legacy kinematic API's (3, N).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial import cKDTree
import torch


def resolve_device(device="auto"):
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.warn("CUDA unavailable; Torch trajectory scoring uses CPU.", RuntimeWarning)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def se3_inverse(transform):
    """Rigid inverse, retaining arbitrary leading batch dimensions."""
    out = torch.zeros_like(transform)
    rotation = transform[..., :3, :3].transpose(-1, -2)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = -(rotation @ transform[..., :3, 3, None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def se3_exp(twist):
    """Exponential for twists ordered [rho_x, rho_y, rho_z, phi_x, ...]."""
    rho, phi = twist[..., :3], twist[..., 3:]
    theta2 = (phi * phi).sum(-1)
    safe2 = theta2.clamp_min(torch.finfo(twist.dtype).eps ** 2)
    theta = safe2.sqrt()
    small = theta2 < 1e-8
    a = torch.where(small, 1-theta2/6+theta2**2/120, theta.sin()/theta)
    b = torch.where(small, 0.5-theta2/24+theta2**2/720, (1-theta.cos())/safe2)
    c = torch.where(small, 1/6-theta2/120+theta2**2/5040, (theta-theta.sin())/(safe2*theta))
    skew = _skew(phi)
    skew2 = skew @ skew
    identity = torch.eye(3, device=twist.device, dtype=twist.dtype)
    rotation = identity + a[..., None, None]*skew + b[..., None, None]*skew2
    jacobian = identity + b[..., None, None]*skew + c[..., None, None]*skew2
    out = torch.zeros(twist.shape[:-1]+(4, 4), device=twist.device, dtype=twist.dtype)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = (jacobian @ rho[..., None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def se3_log(transform):
    """Principal SE(3) logarithm, stable at zero and near pi rotations.

    Select the best-conditioned matrix-to-quaternion formula on device. At
    exactly pi the axis sign is inherently ambiguous; exp(log(T)) still equals T.
    """
    r = transform[..., :3, :3]
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    q2 = torch.stack((1+r00+r11+r22, 1+r00-r11-r22,
                      1-r00+r11-r22, 1-r00-r11+r22), -1).clamp_min(0)
    xy, xz, yz = r[..., 0, 1]+r[..., 1, 0], r[..., 0, 2]+r[..., 2, 0], r[..., 1, 2]+r[..., 2, 1]
    wx, wy, wz = r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]
    candidates = torch.stack((
        torch.stack((q2[..., 0], wx, wy, wz), -1),
        torch.stack((wx, q2[..., 1], xy, xz), -1),
        torch.stack((wy, xy, q2[..., 2], yz), -1),
        torch.stack((wz, xz, yz, q2[..., 3]), -1),
    ), -2) / (2*q2.sqrt().clamp_min(1e-12)[..., :, None])
    best = q2.argmax(-1)
    quat = candidates.gather(-2, best[..., None, None].expand(best.shape+(1, 4))).squeeze(-2)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    sin_half = torch.linalg.vector_norm(quat[..., 1:], dim=-1)
    angle = 2*torch.atan2(sin_half, quat[..., 0])
    factor = torch.where(sin_half > 1e-8, angle/sin_half.clamp_min(1e-12),
                         2+sin_half**2/3+3*sin_half**4/20)
    phi = quat[..., 1:] * factor[..., None]
    theta2 = (phi*phi).sum(-1)
    safe2 = theta2.clamp_min(torch.finfo(transform.dtype).eps**2)
    theta = safe2.sqrt()
    half = theta/2
    coefficient = torch.where(
        theta2 < 1e-8, 1/12+theta2/720+theta2**2/30240,
        (1-half*half.cos()/half.sin().clamp_min(1e-12))/safe2,
    )
    skew = _skew(phi)
    jacobian_inv = (torch.eye(3, device=r.device, dtype=r.dtype)-skew/2
                    + coefficient[..., None, None]*(skew @ skew))
    rho = (jacobian_inv @ transform[..., :3, 3, None]).squeeze(-1)
    return torch.cat((rho, phi), -1)


def se3_geodesic(first, second, u):
    u = torch.as_tensor(u, device=first.device, dtype=first.dtype)
    return first @ se3_exp(u[..., None]*se3_log(se3_inverse(first) @ second))


def se3_bezier(controls, u):
    """Cubic de Casteljau curve; controls (..., 4, 4, 4), u broadcastable."""
    a, b, c, d = controls.unbind(-3)
    ab, bc, cd = (se3_geodesic(a, b, u), se3_geodesic(b, c, u), se3_geodesic(c, d, u))
    return se3_geodesic(se3_geodesic(ab, bc, u), se3_geodesic(bc, cd, u), u)


class KinematicTorch:
    """Six-leg frame transforms constructed directly from leg-root positions."""
    def __init__(self, leg_base_p, device="cpu", dtype=torch.float32):
        base = np.asarray(leg_base_p)
        if base.shape != (3, 6) or not np.isfinite(base).all():
            raise ValueError("leg_base_p must be finite with shape (3,6)")
        # The legacy API transforms xy only, retaining z without translation.
        if np.any(base[2] != 0):
            raise ValueError("leg frame convention requires zero leg_base_p z")
        self.leg_base_p = torch.tensor(base.T.copy(), device=device, dtype=dtype)
        signs = [[-1, -1, 1]]*3 + [[1, 1, 1]]*3
        self.signs = torch.tensor(signs, device=device, dtype=dtype)
        self.device, self.dtype = self.leg_base_p.device, dtype

    def _R2B(self, points, leg_indices):
        return (points-self.leg_base_p[leg_indices])*self.signs[leg_indices]

    def _B2R(self, points, leg_indices):
        return points*self.signs[leg_indices]+self.leg_base_p[leg_indices]

    def RVectorToLeg(self, vectors, leg_indices):
        return vectors*self.signs[leg_indices]

    LegVectorToR = RVectorToLeg


class VoxelsTorch:
    """Voxel metadata plus eight-neighbor sampling with constant padding."""
    def __init__(self, voxels, device="cpu", dtype=torch.float32):
        if not isinstance(voxels, VoxelGridData):
            raise TypeError("voxels must be VoxelGridData")
        bounds = np.asarray(voxels.bounds)
        shape = np.asarray(voxels.grid_shape)
        scale = float(voxels.voxel_scale)
        if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 1] <= bounds[:, 0]) or not math.isfinite(scale) or scale <= 0
                or shape.shape != (3,) or np.any(shape <= 0)
                or not np.issubdtype(shape.dtype, np.integer)):
            raise ValueError("invalid voxel bounds, shape or scale")
        expected = np.ceil((bounds[:, 1]-bounds[:, 0])/scale).astype(np.int64)
        if not np.array_equal(expected, shape):
            raise ValueError("voxel grid_shape does not match bounds and scale")
        self._bounds = torch.tensor(bounds.copy(), device=device, dtype=dtype)
        self.grid_shape = tuple(int(v) for v in shape)
        self.shape_tensor = torch.tensor(shape.copy(), device=device, dtype=torch.long)
        self.grid_size = math.prod(self.grid_shape)
        self.voxel_scale = scale
        self.device, self.dtype = self._bounds.device, dtype

    def IsInsideRange(self, pos):
        return ((pos >= self._bounds[:, 0]) & (pos < self._bounds[:, 1])).all(-1)

    def Pos2GridIndex(self, pos):
        index = torch.floor((pos-self._bounds[:, 0])/self.voxel_scale).long()
        return torch.minimum(index.clamp_min(0), self.shape_tensor-1)

    def GridIndex2FlatIndex(self, index):
        _, ny, nz = self.grid_shape
        return (index[..., 0]*ny+index[..., 1])*nz+index[..., 2]

    def Pos2FlatIndex(self, pos):
        return self.GridIndex2FlatIndex(self.Pos2GridIndex(pos))

    def TrilinearSample(self, values, pos, outside_value, channel=None):
        """Sample scalar field or one selected channel of an xyz-channel field."""
        if tuple(values.shape[:3]) != self.grid_shape:
            raise ValueError("values shape does not match voxel grid")
        if (values.ndim != 3 and not (values.ndim == 4 and channel is not None)):
            raise ValueError("values must be xyz scalar field, or xyz-channel with channel indices")
        if pos.shape[-1] != 3:
            raise ValueError("pos must have last dimension 3")
        grid_pos = (pos-self._bounds[:, 0])/self.voxel_scale-0.5
        lower = torch.floor(grid_pos).long()
        fraction = grid_pos-lower.to(grid_pos.dtype)
        result = torch.zeros_like(pos[..., 0])
        for corner in product((0, 1), repeat=3):
            offset = lower.new_tensor(corner)
            index = lower+offset
            inside = ((index >= 0) & (index < self.shape_tensor)).all(-1)
            safe = torch.minimum(index.clamp_min(0), self.shape_tensor-1)
            ix, iy, iz = safe.unbind(-1)
            sample = values[ix, iy, iz] if channel is None else values[ix, iy, iz, channel]
            sample = torch.where(inside, sample, float(outside_value))
            weight = torch.where(offset.bool(), fraction, 1-fraction).prod(-1)
            result = result+weight*sample
        return result


@dataclass(frozen=True)
class PreparedCandidates:
    """Device-resident, deduplicated candidate rows. Prepare once per retraction."""
    indices: torch.Tensor
    branches: torch.Tensor
    valid: torch.Tensor

    def select(self, rows):
        return PreparedCandidates(self.indices[rows], self.branches[rows], self.valid[rows])


@dataclass
class VoxelGridData:
    """Grid metadata, independent of the legacy voxel implementation."""
    bounds: np.ndarray
    voxel_scale: float

    def __post_init__(self):
        self.bounds = np.array(self.bounds, dtype=np.float32, copy=True)
        self.voxel_scale = float(self.voxel_scale)
        if (self.bounds.shape != (3, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 1] <= self.bounds[:, 0])
                or not math.isfinite(self.voxel_scale) or self.voxel_scale <= 0):
            raise ValueError("invalid voxel bounds or scale")

    @property
    def grid_shape(self):
        return np.ceil((self.bounds[:, 1]-self.bounds[:, 0])/self.voxel_scale).astype(np.int64)

    def centers(self):
        index = np.stack(np.meshgrid(*(np.arange(n) for n in self.grid_shape), indexing="ij"), -1).reshape(-1, 3)
        return (self.bounds[:, 0]+(index+0.5)*self.voxel_scale).astype(np.float32)


@dataclass
class RobotVoxelData:
    """Arrays and metadata only: no robot, map-builder or legacy class handles.

    Cache files currently omit grid scales/robot bounds; from_files therefore
    exposes those parameters explicitly, defaulting to the existing robot.
    Body ESDF is optional for soft scoring but required for native hard checks.
    """
    leg_grid: VoxelGridData
    env_grid: VoxelGridData
    leg_base_p: np.ndarray
    centers: np.ndarray
    x_b3: np.ndarray
    ankle_pos: np.ndarray
    reachable: np.ndarray
    to_bound_dist: np.ndarray
    bounding_points: np.ndarray
    points: np.ndarray
    landing_points: np.ndarray
    normals: np.ndarray
    landing_count: int
    env_esdf: np.ndarray
    ankle_radius: float = 0.045
    body_grid: VoxelGridData | None = None
    body_esdf: np.ndarray | None = None

    @classmethod
    def from_files(cls, point_map_file, voxel_dir=None, *, env_esdf_file=None,
                   leg_bounds=((-0.14, 0.38), (-0.38, 0.38), (-0.27, 0.25)),
                   body_bounds=((-0.4, 0.4), (-0.5, 0.5), (-0.2, 0.2)),
                   leg_voxel_scale=0.01, body_voxel_scale=0.01, env_voxel_scale=0.04,
                   leg_base_p=None, bx=0.1, by=0.22, ankle_radius=0.045,
                   suction_to_foot_offset=0.03):
        """Read existing NPZ files only; never generate ESDF/IK or write caches."""
        if voxel_dir is None:
            voxel_dir = Path(__file__).parent/"voxels_info"
        directory = Path(voxel_dir)

        def read(path, required, optional=()):
            with np.load(path, allow_pickle=False) as file:
                missing = set(required)-set(file.files)
                if missing:
                    raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
                return {key: np.array(file[key], copy=True)
                        for key in (*required, *optional) if key in file.files}

        cloud = read(point_map_file, ("points", "normals", "landing_count"), ("bounds",))
        leg = read(directory/"leg_voxels_info.npz", ("x_b3", "ankle_pos"))
        body = read(directory/"body_voxels_info.npz", ("esdf_for_env", "bounding_points"))
        robot = read(directory/"robot_reachable_legs.npz", ("robot_reachable_legs", "to_bound_dist"))
        env = read(env_esdf_file or directory/"env_voxels.npz", ("esdf",))
        points = cloud["points"].astype(np.float32)
        normals = cloud["normals"].astype(np.float32)
        saved_count = np.asarray(cloud["landing_count"])
        if saved_count.size != 1 or not np.issubdtype(saved_count.dtype, np.integer):
            raise ValueError("landing_count must contain one integer")
        count = int(saved_count.reshape(-1)[0])
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or not 0 <= count <= len(points):
            raise ValueError("invalid saved points/normals/landing_count")
        if "bounds" in cloud:
            bounds = cloud["bounds"]
        elif len(points):
            bounds = np.column_stack((points.min(0), points.max(0)))
        else:
            raise ValueError("an empty point map requires saved bounds")
        if leg_base_p is None:
            leg_base_p = np.array([[-bx]*3+[bx]*3, [-by, by, 0, -by, by, 0], [0]*6], dtype=np.float32)
        if not math.isfinite(suction_to_foot_offset):
            raise ValueError("suction_to_foot_offset must be finite")
        leg_grid = VoxelGridData(leg_bounds, leg_voxel_scale)
        landing = points[:count]+normals[:count]*suction_to_foot_offset
        return cls(leg_grid, VoxelGridData(bounds, env_voxel_scale), leg_base_p, leg_grid.centers(),
                   leg["x_b3"], leg["ankle_pos"], robot["robot_reachable_legs"], robot["to_bound_dist"],
                   body["bounding_points"], points, landing, normals, count, env["esdf"], ankle_radius,
                   VoxelGridData(body_bounds, body_voxel_scale), body["esdf_for_env"])


class HexStateTorch:
    """Independent map snapshot, KD-trees and inference/checks on the device.

    ``dtype`` controls voxel scoring (float32 by default, float64 for reference).
    Geometry in PostProcessTorch remains float64 in both modes.
    ``RobotFeasiCost`` also accepts PreparedCandidates to skip input validation
    and deduplication in the optimizer's hot path.
    """
    def __init__(self, map_data, device="auto", dtype=torch.float32, pose_chunk_size=64):
        self.device = resolve_device(device)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if isinstance(pose_chunk_size, bool) or not isinstance(pose_chunk_size, int) or pose_chunk_size <= 0:
            raise ValueError("pose_chunk_size must be a positive integer")
        self.dtype, self.pose_chunk_size = dtype, pose_chunk_size
        self.reload_from(map_data)

    @classmethod
    def from_files(cls, point_map_file, voxel_dir=None, *, device="auto", dtype=torch.float32,
                   pose_chunk_size=64, **map_options):
        data = RobotVoxelData.from_files(point_map_file, voxel_dir, **map_options)
        return cls(data, device, dtype, pose_chunk_size)

    def reload_from(self, map_data):
        """Explicitly replace all cached tensors; no disk IO or map rebuilding."""
        if not isinstance(map_data, RobotVoxelData):
            raise TypeError("map_data must be RobotVoxelData; use HexStateTorch.from_files to load caches")
        replacement = object.__new__(type(self))
        replacement.__dict__.update(self.__dict__)
        replacement._load_snapshot(map_data)
        self.__dict__.update(replacement.__dict__)

    def _load_snapshot(self, data):
        self.leg_voxels = VoxelsTorch(data.leg_grid, self.device, self.dtype)
        self.env_voxels = VoxelsTorch(data.env_grid, self.device, self.dtype)
        self.kin = KinematicTorch(data.leg_base_p, self.device, self.dtype)
        self.map_data = data
        self.landing_count = int(data.landing_count)

        def snapshot(name, values, shape, finite=True, dtype=None):
            array = np.asarray(values)
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} does not match {shape}")
            if finite and not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            return torch.tensor(array.copy(), device=self.device, dtype=dtype or self.dtype)

        if self.landing_count < 0 or self.landing_count > len(data.points):
            raise ValueError("invalid landing_count")
        g = self.leg_voxels.grid_size
        self.env_esdf = snapshot("env_esdf", data.env_esdf, self.env_voxels.grid_shape)
        self.to_bound_dist = snapshot("to_bound_dist", data.to_bound_dist, self.leg_voxels.grid_shape+(6,))
        self.x_b3 = snapshot("x_b3", data.x_b3, (g, 2, 3), finite=False)
        self.ankle_pos = snapshot("ankle_pos", data.ankle_pos, (g, 2, 3), finite=False)
        self.reachable = snapshot("robot_reachable_legs", data.reachable,
                                  (g, 2, 6), dtype=torch.bool)
        self.centers = snapshot("leg centers", data.centers, (g, 3))
        self.landing_points = snapshot("landing_points", data.landing_points, (self.landing_count, 3))
        self.normals = snapshot("normals", data.normals[:self.landing_count],
                                (self.landing_count, 3), finite=False)
        bounding = np.asarray(data.bounding_points).reshape(-1, 3)
        self.bounding_points = snapshot("bounding_points", bounding, bounding.shape)
        self.ankle_radius = float(data.ankle_radius)
        if not math.isfinite(self.ankle_radius) or self.ankle_radius < 0:
            raise ValueError("invalid ankle_collide_radi")
        self._points_cpu = np.array(data.points, dtype=np.float64, copy=True)
        if self._points_cpu.ndim != 2 or self._points_cpu.shape[1] != 3 or not np.isfinite(self._points_cpu).all():
            raise ValueError("points must be finite (N,3)")
        self.points = snapshot("points", data.points, self._points_cpu.shape)
        landing_cpu = np.array(data.landing_points, dtype=np.float64, copy=True)
        self._tree = cKDTree(self._points_cpu) if len(self._points_cpu) else None
        self._landing_tree = cKDTree(landing_cpu) if len(landing_cpu) else None
        self.body_voxels = None
        self.body_esdf = None
        if data.body_grid is not None and data.body_esdf is not None:
            self.body_voxels = VoxelsTorch(data.body_grid, self.device, self.dtype)
            self.body_esdf = snapshot("body_esdf", data.body_esdf, self.body_voxels.grid_shape)
        bounds = np.asarray(data.leg_grid.bounds, dtype=np.float64)
        max_radius = np.linalg.norm(np.max(np.abs(bounds), axis=1))
        radius_max = (max(0., max_radius-0.28)/0.03)**2
        min_boundary = min(0., float(np.min(data.to_bound_dist)))
        boundary_max = (max(0., 0.05-min_boundary)/0.03)**2
        min_ankle = min(-0.06, float(np.min(data.env_esdf)))
        self.empty_leg_cost = 36+max(radius_max, boundary_max)+(100/80)**2+25+(0.06-min_ankle)**2
        self.leg_ids = torch.arange(6, device=self.device)[None, :, None]

    def GetRobotFeasiCostPointsBatch(self, W_T_R, workers=-1):
        """CPU setup-only search; returns ragged rows, matching legacy queries."""
        matrices = np.asarray(W_T_R, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
            raise ValueError("W_T_R must be finite (B,4,4) CPU matrices")
        batch = len(matrices)
        k = min(2400, len(self._points_cpu))
        if k == 0 or batch == 0:
            return [np.zeros(0, dtype=np.int64) for _ in range(batch)]
        _, idx = self._tree.query(matrices[:, :3, 3], k=k, workers=workers)
        idx = np.asarray(idx, dtype=np.int64).reshape(batch, k)
        need = np.count_nonzero(idx < self.landing_count, axis=1) <= 60
        supplement = {}
        if self.landing_count and need.any():
            rows = np.flatnonzero(need)
            _, extra = self._landing_tree.query(matrices[rows, :3, 3],
                                                    k=min(100, self.landing_count), workers=workers)
            extra = np.asarray(extra, dtype=np.int64).reshape(len(rows), -1)
            supplement = dict(zip(rows, extra))
        return [np.unique(np.concatenate((row, supplement[i]))) if i in supplement else row
                for i, row in enumerate(idx)]

    @torch.inference_mode()
    def RobotFeasiCheck(self, W_T_R, points_idx=None):
        """Native hard check, returning CPU indices and masks for the planner.

        Both IK branches are tested. Body clearance uses environment points in
        the robot body ESDF (5 mm), workspace uses 4 cm/30 cm limits, normals use
        the 70-degree plane cone and 95-degree xb3 rule, and ankle uses env ESDF.
        End-point collision remains excluded, matching the planner's hard rules.
        """
        if self.body_voxels is None or self.body_esdf is None:
            raise RuntimeError("RobotFeasiCheck requires body_grid and body_esdf in RobotVoxelData")
        matrix = W_T_R.A if hasattr(W_T_R, "A") else W_T_R
        pose = torch.as_tensor(matrix, device=self.device, dtype=self.dtype)
        if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all()):
            raise ValueError("W_T_R must be a finite (4,4) pose")
        rotation, translation = pose[:3, :3], pose[:3, 3]
        if points_idx is None:
            k = min(2400, len(self._points_cpu))
            if k:
                _, idx = self._tree.query(translation.cpu().numpy(), k=k)
                idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            else:
                idx = np.zeros(0, dtype=np.int64)
        else:
            idx = np.asarray(points_idx)
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError("points_idx must be a one-dimensional integer array")
            if np.any(idx < 0) or np.any(idx >= len(self._points_cpu)):
                raise IndexError("points_idx out of range")
        tensor_idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        robot_env_points = (self.points[tensor_idx]-translation) @ rotation
        inside_body = self.body_voxels.IsInsideRange(robot_env_points)
        body_distance = self.body_voxels.TrilinearSample(self.body_esdf, robot_env_points, 0.05)
        body_free = ~inside_body | (body_distance >= 0.005)
        landing_idx = idx[idx < self.landing_count]
        if points_idx is None and len(landing_idx) <= 60 and self.landing_count:
            _, landing_idx = self._landing_tree.query(translation.cpu().numpy(), k=min(100, self.landing_count))
            landing_idx = np.asarray(landing_idx, dtype=np.int64).reshape(-1)
        landing_tensor = torch.as_tensor(landing_idx, device=self.device, dtype=torch.long)
        robot_points = (self.landing_points[landing_tensor]-translation) @ rotation
        robot_normals = self.normals[landing_tensor] @ rotation
        leg_ids = torch.arange(6, device=self.device)[:, None]
        leg_points = self.kin._R2B(robot_points[None], leg_ids)
        leg_normals = self.kin.RVectorToLeg(robot_normals[None], leg_ids)
        flat = self.leg_voxels.Pos2FlatIndex(leg_points)
        boundary = self.leg_voxels.TrilinearSample(self.to_bound_dist, leg_points, 0., leg_ids)
        workspace = (self.leg_voxels.IsInsideRange(leg_points) & (boundary >= 0.04)
                     & (torch.linalg.vector_norm(self.centers[flat], dim=-1) <= 0.3))
        # Preserve hard-check normalization, including degenerate/zero vectors.
        normalized = leg_normals/torch.linalg.vector_norm(leg_normals, dim=-1, keepdim=True).clamp_min(1e-5)
        plane = torch.stack((-leg_points[..., 1], leg_points[..., 0], torch.zeros_like(boundary)), -1)
        plane = plane/torch.linalg.vector_norm(plane, dim=-1, keepdim=True).clamp_min(1e-5)
        plane_ok = (normalized*plane).sum(-1).abs() <= math.cos(math.radians(70))
        xb3 = self.x_b3[flat]
        xb3_ok = (xb3 * -normalized[:, :, None]).sum(-1) >= math.cos(math.radians(95))
        ankle = torch.nan_to_num(self.ankle_pos[flat], nan=0.)
        robot_ankle = self.kin._B2R(ankle, leg_ids[:, :, None])
        world_ankle = robot_ankle @ rotation.T+translation
        ankle_distance = self.env_voxels.TrilinearSample(self.env_esdf, world_ankle, -0.06)
        feasible = (workspace[:, :, None] & plane_ok[:, :, None] & xb3_ok
                    & self.reachable[flat, :, leg_ids] & (ankle_distance >= self.ankle_radius))
        return idx, landing_idx, body_free.cpu().numpy(), feasible.cpu().numpy()

    def prepare_candidates(self, rows, sol_index=1):
        """Setup-only packing of ragged CPU rows, including branch validation."""
        rows = [np.asarray(row) for row in rows]
        width = max((row.size for row in rows), default=0)
        packed = np.full((len(rows), width), -1, dtype=np.int64)
        for i, row in enumerate(rows):
            if row.ndim != 1 or (row.size and not np.issubdtype(row.dtype, np.integer)):
                raise ValueError("candidate rows must be one-dimensional integer indices")
            packed[i, :row.size] = row
        return self._canonicalize(packed, sol_index)

    def _canonicalize(self, points_idx, sol_index):
        indices = torch.as_tensor(points_idx, device=self.device)
        if indices.ndim != 2 or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("points_idx must be an integer (B,N) array")
        indices = indices.long()
        branches = torch.as_tensor(sol_index, device=self.device)
        if branches.dtype == torch.bool or (branches.ndim and branches.dtype.is_floating_point):
            raise ValueError("sol_index must contain integer 0 or 1")
        if branches.ndim not in (0, 2) or (branches.ndim and branches.shape != indices.shape):
            raise ValueError("sol_index must be scalar or match (B,N)")
        if not bool(((branches == 0) | (branches == 1)).all()):
            raise ValueError("sol_index must be 0 or 1")
        branches = branches.long().expand_as(indices)
        valid = (indices >= 0) & (indices < self.landing_count)
        indices = torch.where(valid, indices, self.landing_count)
        indices, order = indices.sort(dim=-1, stable=True)
        branches = branches.gather(-1, order)
        valid = indices < self.landing_count
        if indices.shape[1]:
            repeated = (indices[:, 1:] == indices[:, :-1]) & valid[:, 1:]
            if bool((repeated & (branches[:, 1:] != branches[:, :-1])).any()):
                raise ValueError("duplicate landing point has conflicting IK branches")
            valid = valid & torch.cat((torch.ones_like(valid[:, :1]), ~repeated), -1)
        safe = indices.clamp(min=0, max=max(self.landing_count-1, 0))
        return PreparedCandidates(safe, branches, valid)

    @torch.inference_mode()
    def RobotFeasiCost(self, W_T_R, points_idx, sol_index=1, *, return_details=False):
        """Score a batch; optionally return (cost, per-leg count diagnostics).

        ``return_details`` is useful when comparing mixed/double precision near
        hard thresholds. The CMA hot path requests only costs.
        """
        poses = torch.as_tensor(W_T_R, device=self.device, dtype=self.dtype)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("W_T_R must have shape (B,4,4)")
        candidates = points_idx if isinstance(points_idx, PreparedCandidates) else self._canonicalize(points_idx, sol_index)
        if candidates.indices.shape[0] != len(poses) or candidates.indices.device != self.device:
            raise ValueError("candidate batch/device does not match poses")
        if not len(poses):
            empty = poses.new_empty(0)
            details = {"hard_feasible_count": torch.empty((0, 6), device=self.device, dtype=torch.long),
                       "scorable_count": torch.empty((0, 6), device=self.device, dtype=torch.long)}
            return (empty, details) if return_details else empty
        chunks = [self._score(poses[start:start+self.pose_chunk_size],
                             candidates.select(slice(start, start+self.pose_chunk_size)), return_details)
                  for start in range(0, len(poses), self.pose_chunk_size)]
        if not return_details:
            return torch.cat(chunks)
        costs, counts = zip(*chunks)
        return torch.cat(costs), {key: torch.cat([part[key] for part in counts]) for key in counts[0]}

    def _score(self, poses, candidates, return_details=False):
        # All data-dependent masking and all six legs remain on device.
        finite_pose = torch.isfinite(poses).all(dim=-1).all(dim=-1)
        poses = torch.where(finite_pose[:, None, None], poses,
                            torch.eye(4, device=self.device, dtype=self.dtype))
        rotation, translation = poses[:, :3, :3], poses[:, :3, 3]
        body_cost = poses.new_zeros(len(poses))
        if len(self.bounding_points):
            world_body = self.bounding_points[None] @ rotation.transpose(-1, -2)+translation[:, None]
            distance = self.env_voxels.TrilinearSample(self.env_esdf, world_body, -0.08)
            body_cost = ((0.08-distance).clamp_min(0)/0.08).square().mean(-1)
        if not self.landing_count or not candidates.indices.shape[1]:
            cost = 10*body_cost+self.empty_leg_cost
            cost = torch.where(finite_pose, cost, float("inf"))
            zero = torch.zeros((len(poses), 6), device=self.device, dtype=torch.long)
            return (cost, {"hard_feasible_count": zero, "scorable_count": zero}) if return_details else cost
        world_points = self.landing_points[candidates.indices]
        world_normals = self.normals[candidates.indices]
        robot_points = (world_points-translation[:, None]) @ rotation
        robot_normals = world_normals @ rotation
        leg_points = self.kin._R2B(robot_points[:, None], self.leg_ids)
        leg_normals = self.kin.RVectorToLeg(robot_normals[:, None], self.leg_ids)
        flat = self.leg_voxels.Pos2FlatIndex(leg_points)
        branches = candidates.branches[:, None]
        xb3, ankle = self.x_b3[flat, branches], self.ankle_pos[flat, branches]
        score_mask = (candidates.valid[:, None] & self.leg_voxels.IsInsideRange(leg_points)
                      & self.reachable[flat, branches, self.leg_ids]
                      & torch.isfinite(xb3).all(-1) & torch.isfinite(ankle).all(-1))
        xb3, ankle = torch.nan_to_num(xb3), torch.nan_to_num(ankle)
        radius = torch.linalg.vector_norm(leg_points, dim=-1)
        workspace_cost = ((radius-0.28).clamp_min(0)/0.03).square()
        boundary = self.leg_voxels.TrilinearSample(self.to_bound_dist, leg_points, 0., self.leg_ids)
        boundary_cost = ((0.05-boundary).clamp_min(0)/0.03).square()
        eps = 1e-8
        normal_norm = torch.linalg.vector_norm(leg_normals, dim=-1)
        normal_valid = torch.isfinite(leg_normals).all(-1) & (normal_norm > eps)
        normalized = torch.where(normal_valid[..., None],
                                 torch.nan_to_num(leg_normals)/normal_norm.clamp_min(eps)[..., None], 0.)
        xb3_norm = torch.linalg.vector_norm(xb3, dim=-1)
        dot = ((xb3/xb3_norm.clamp_min(eps)[..., None]) * -normalized).sum(-1)
        xb3_angle = torch.where(normal_valid & (xb3_norm > eps), dot.clamp(-1, 1).acos(), math.pi)
        xb3_limit = math.radians(80)
        xb3_cost = ((xb3_angle-xb3_limit).clamp_min(0)/xb3_limit).square()
        # cross(z_axis, point) = [-y, x, 0].
        plane = torch.stack((-leg_points[..., 1], leg_points[..., 0], torch.zeros_like(radius)), -1)
        plane_norm = torch.linalg.vector_norm(plane, dim=-1)
        plane_dot = (normalized*(plane/plane_norm.clamp_min(eps)[..., None])).sum(-1)
        plane_angle = torch.where(normal_valid & (plane_norm > eps), plane_dot.clamp(-1, 1).asin().abs(), math.pi/2)
        plane_cost = ((plane_angle-math.radians(15)).clamp_min(0)/math.radians(15)).square()
        robot_ankle = self.kin._B2R(ankle, self.leg_ids)
        world_ankle = torch.einsum("blni,bji->blnj", robot_ankle, rotation)+translation[:, None, None]
        ankle_distance = self.env_voxels.TrilinearSample(self.env_esdf, world_ankle, -0.06)
        ankle_cost = (0.06-ankle_distance).clamp_min(0).square()
        transition = ((xb3_angle-xb3_limit)/math.radians(10)).clamp(0, 1)
        smooth = transition.square()*(3-2*transition)
        plane_cost = (1-smooth)*plane_cost+smooth*25
        hard = (score_mask & (boundary >= 0.04)
                & (torch.linalg.vector_norm(self.centers[flat], dim=-1) <= 0.3)
                & (xb3_angle <= math.radians(95)) & (plane_angle <= math.radians(20))
                & (ankle_distance >= self.ankle_radius))
        count_cost = (6-hard.sum(-1)).clamp_min(0).to(self.dtype).square()
        point_cost = workspace_cost+boundary_cost+xb3_cost+plane_cost+ankle_cost
        point_cost = torch.where(score_mask, point_cost, float("inf"))
        best = point_cost.topk(min(6, point_cost.shape[-1]), dim=-1, largest=False).values
        best_valid = torch.isfinite(best)
        mean_best = torch.where(best_valid, best, 0.).sum(-1)/best_valid.sum(-1).clamp_min(1)
        leg_cost = torch.where(score_mask.any(-1), count_cost+mean_best, self.empty_leg_cost)
        cost = 10*body_cost+leg_cost.mean(-1)
        cost = torch.where(finite_pose, cost, float("inf"))
        if return_details:
            return cost, {"hard_feasible_count": hard.sum(-1)*finite_pose[:, None],
                          "scorable_count": score_mask.sum(-1)*finite_pose[:, None]}
        return cost
