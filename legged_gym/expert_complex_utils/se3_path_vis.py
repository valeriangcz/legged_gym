from __future__ import annotations


"""Interactive PyVista viewer for an already optimized SE(3) path.

This module deliberately does not import or invoke :class:`PostProcess`.  It
only loads the versioned JSON format documented by ``build_arg_parser`` and
visualizes the path against the precomputed environment/robot voxel data.
"""

"""
codex摘要
严格加载并校验 version 1 JSON schema。
Dense 路径和任意阶 SE(3) Bezier 重建。
半透明 STL、间隔落脚点、管状路径、路标、标签及 RGB 坐标轴。
身体和六腿可达体素外壳。
每帧调用 RobotFeasiCheck()，显示身体碰撞、六腿落脚点数量和强/弱/不可行状态。
可行落脚点按六腿分别着色。
自动播放、单步、回退、重置、路径切换、相机跟随、速度调节。
Dense/Bezier 切换保持归一化进度。
自动播放使用实际时间，终点自动停止
python legged_gym/expert_complex_utils/se3_path_vis.py \
    --path /path/to/optimized_path.json
python legged_gym/expert_complex_utils/se3_path_vis.py \
    --path /path/to/optimized_path.json \
    --source bezier \
    --curve-dt 0.05
"""


import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import HexState, Kinematic


GREEN = "#25a244"
YELLOW = "#e9c46a"
RED = "#d62828"
BODY_GRAY = "#718096"
PATH_BLUE = "#2166d1"
WAYPOINT_ORANGE = "#f28e2b"
LEG_COLORS = (
    "#e63946", "#2a9d4b", "#277da1", "#e9c46a", "#9b5de5", "#00b4d8"
)
DEFAULT_MAP_STL = str(
    Path(LEGGED_GYM_ROOT_DIR)
    / "resources/environments/sutructure1/complex_surface.STL"
)


@dataclass(frozen=True)
class PoseSeries:
    """Timestamped SE(3) samples."""

    time: np.ndarray
    poses: tuple[SE3, ...]

    def __post_init__(self) -> None:
        times = np.asarray(self.time, dtype=np.float64)
        if times.ndim != 1 or len(times) != len(self.poses):
            raise ValueError("PoseSeries.time and poses must have equal lengths")
        if not len(times):
            raise ValueError("PoseSeries must not be empty")
        object.__setattr__(self, "time", times)

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    def pose_at(self, query_time: float) -> SE3:
        """Interpolate by timestamp using the SE(3) geodesic."""
        t = float(np.clip(query_time, self.time[0], self.time[-1]))
        hi = int(np.searchsorted(self.time, t, side="right"))
        if hi == 0:
            return self.poses[0].copy()
        if hi >= len(self.time):
            return self.poses[-1].copy()
        lo = hi - 1
        span = float(self.time[hi] - self.time[lo])
        u = 0.0 if span <= 0.0 else (t - float(self.time[lo])) / span
        return se3_interpolate(self.poses[lo], self.poses[hi], u)


@dataclass(frozen=True)
class BezierSegment:
    degree: int
    duration: float
    control_poses: tuple[SE3, ...]


@dataclass(frozen=True)
class SE3PathData:
    metadata: Mapping[str, Any]
    waypoints: tuple[SE3, ...]
    dense: PoseSeries
    bezier_segments: tuple[BezierSegment, ...]

    def sample_bezier(self, dt: float) -> PoseSeries:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("curve dt must be finite and positive")
        times: list[float] = []
        poses: list[SE3] = []
        elapsed = 0.0
        for segment_index, segment in enumerate(self.bezier_segments):
            intervals = max(1, int(math.ceil(segment.duration / dt)))
            for local_index in range(intervals + 1):
                if segment_index and local_index == 0:
                    continue
                u = local_index / intervals
                times.append(elapsed + u * segment.duration)
                poses.append(se3_de_casteljau(segment.control_poses, u))
            elapsed += segment.duration
        return PoseSeries(np.asarray(times, dtype=np.float64), tuple(poses))


def se3_interpolate(first: SE3, second: SE3, u: float) -> SE3:
    """Geodesic interpolation consistent with the optimizer's convention."""
    u = float(u)
    return first * SE3.Exp(u * (first.inv() * second).log())


def se3_de_casteljau(control_poses: Sequence[SE3], u: float) -> SE3:
    """Evaluate an arbitrary-degree SE(3) Bezier curve."""
    if not control_poses:
        raise ValueError("control_poses must not be empty")
    if not np.isfinite(u):
        raise ValueError("Bezier parameter u must be finite")
    if u <= 0.0:
        return control_poses[0].copy()
    if u >= 1.0:
        return control_poses[-1].copy()
    work = [pose.copy() for pose in control_poses]
    for level in range(1, len(work)):
        for index in range(len(work) - level):
            work[index] = se3_interpolate(work[index], work[index + 1], u)
    return work[0]


def _array_field(
    block: Mapping[str, Any], field: str, path: str, *, ndim: int
) -> np.ndarray:
    if field not in block:
        raise ValueError(f"missing field: {path}.{field}")
    try:
        value = np.asarray(block[field], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}.{field} must be numeric") from exc
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{path}.{field} must be a finite {ndim}-D array")
    return value


def _parse_pose_block(value: Any, path: str) -> tuple[SE3, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    translations = _array_field(value, "t", path, ndim=2)
    angles = _array_field(value, "ang", path, ndim=1)
    axes = _array_field(value, "vec", path, ndim=2)
    if translations.shape[1:] != (3,):
        raise ValueError(f"{path}.t must have shape (N, 3)")
    if axes.shape[1:] != (3,):
        raise ValueError(f"{path}.vec must have shape (N, 3)")
    count = len(translations)
    if len(angles) != count or len(axes) != count:
        raise ValueError(f"{path}.t, .ang and .vec must have equal lengths")
    if count == 0:
        raise ValueError(f"{path} must contain at least one pose")

    poses: list[SE3] = []
    for index, (translation, angle, axis) in enumerate(
        zip(translations, angles, axes)
    ):
        if abs(float(angle)) <= 1e-12:
            rotation = SE3()
        else:
            norm = float(np.linalg.norm(axis))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError(
                    f"{path}.vec[{index}] must be normalizable when angle is nonzero"
                )
            rotation = SE3.AngleAxis(float(angle), axis / norm)
        pose = SE3.Trans(translation) * rotation
        if not np.isfinite(pose.A).all():
            raise ValueError(f"{path}[{index}] does not define a finite SE(3) pose")
        poses.append(pose)
    return tuple(poses)


def _pose_error(first: SE3, second: SE3) -> tuple[float, float]:
    delta = first.inv() * second
    angle = float(Rotation.from_matrix(delta.R).magnitude())
    return float(np.linalg.norm(delta.t)), angle


def _warn_if_mismatch(first: SE3, second: SE3, description: str) -> None:
    translation_error, rotation_error = _pose_error(first, second)
    if translation_error > 1e-4 or rotation_error > 1e-4:
        warnings.warn(
            f"{description} differs by {translation_error:.3g} m and "
            f"{rotation_error:.3g} rad",
            UserWarning,
            stacklevel=2,
        )


def load_se3_path_json(path: str | Path) -> SE3PathData:
    """Load and validate the version-1 visualization path schema."""
    json_path = Path(path)
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            root = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {json_path}: {exc}") from exc
    if not isinstance(root, Mapping):
        raise ValueError("root must be an object")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    metadata = root.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")

    if "waypoints" not in root:
        raise ValueError("missing field: waypoints")
    waypoints = _parse_pose_block(root["waypoints"], "waypoints")
    if len(waypoints) < 2:
        raise ValueError("waypoints must contain at least two poses")

    if "dense_poses" not in root:
        raise ValueError("missing field: dense_poses")
    dense_root = root["dense_poses"]
    if not isinstance(dense_root, Mapping):
        raise ValueError("dense_poses must be an object")
    dense_time = _array_field(dense_root, "time", "dense_poses", ndim=1)
    dense_poses = _parse_pose_block(dense_root, "dense_poses")
    if len(dense_time) != len(dense_poses):
        raise ValueError("dense_poses.time and pose arrays must have equal lengths")
    if len(dense_time) < 2:
        raise ValueError("dense_poses must contain at least two poses")
    if abs(float(dense_time[0])) > 1e-12:
        raise ValueError("dense_poses.time[0] must be 0")
    if np.any(np.diff(dense_time) <= 0.0):
        raise ValueError("dense_poses.time must be strictly increasing")
    dense = PoseSeries(dense_time, dense_poses)

    if "bezier_segments" not in root:
        raise ValueError("missing field: bezier_segments")
    raw_segments = root["bezier_segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("bezier_segments must be a non-empty array")
    segments: list[BezierSegment] = []
    for index, raw in enumerate(raw_segments):
        field = f"bezier_segments[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field} must be an object")
        degree = raw.get("degree")
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError(f"{field}.degree must be a positive integer")
        duration = raw.get("duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not np.isfinite(duration)
            or duration <= 0.0
        ):
            raise ValueError(f"{field}.duration must be finite and positive")
        if "control_poses" not in raw:
            raise ValueError(f"missing field: {field}.control_poses")
        controls = _parse_pose_block(raw["control_poses"], f"{field}.control_poses")
        if len(controls) != degree + 1:
            raise ValueError(
                f"{field}.control_poses must contain degree + 1 poses"
            )
        segments.append(BezierSegment(degree, float(duration), controls))

    if len(waypoints) != len(segments) + 1:
        raise ValueError("waypoints count must equal bezier_segments count + 1")
    for index, segment in enumerate(segments):
        _warn_if_mismatch(
            segment.control_poses[0], waypoints[index],
            f"bezier_segments[{index}] start and waypoints[{index}]",
        )
        _warn_if_mismatch(
            segment.control_poses[-1], waypoints[index + 1],
            f"bezier_segments[{index}] end and waypoints[{index + 1}]",
        )
        if index:
            _warn_if_mismatch(
                segments[index - 1].control_poses[-1], segment.control_poses[0],
                f"bezier segment join {index - 1}/{index}",
            )
    _warn_if_mismatch(dense.poses[0], waypoints[0], "dense start and first waypoint")
    _warn_if_mismatch(dense.poses[-1], waypoints[-1], "dense end and last waypoint")
    return SE3PathData(metadata, waypoints, dense, tuple(segments))


class PathPlayback:
    """GUI-independent timeline shared by keyboard and timer callbacks."""

    def __init__(
        self, dense: PoseSeries, bezier: PoseSeries, source: str = "dense", speed: float = 1.0
    ) -> None:
        if source not in {"dense", "bezier"}:
            raise ValueError("source must be 'dense' or 'bezier'")
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be finite and positive")
        self.series = {"dense": dense, "bezier": bezier}
        self.source = source
        self.speed = float(speed)
        self.current_time = 0.0
        self.playing = False

    @property
    def active(self) -> PoseSeries:
        return self.series[self.source]

    @property
    def progress(self) -> float:
        return 0.0 if self.active.duration <= 0 else self.current_time / self.active.duration

    @property
    def pose(self) -> SE3:
        return self.active.pose_at(self.current_time)

    def reset(self) -> None:
        self.current_time = 0.0
        self.playing = False

    def step(self, direction: int) -> None:
        times = self.active.time
        if direction > 0:
            index = int(np.searchsorted(times, self.current_time + 1e-12, side="right"))
            self.current_time = float(times[min(index, len(times) - 1)])
        elif direction < 0:
            index = int(np.searchsorted(times, self.current_time - 1e-12, side="left")) - 1
            self.current_time = float(times[max(index, 0)])
        self.playing = False

    def advance(self, wall_seconds: float) -> None:
        if not self.playing:
            return
        self.current_time += max(0.0, float(wall_seconds)) * self.speed
        if self.current_time >= self.active.duration:
            self.current_time = self.active.duration
            self.playing = False

    def switch_source(self) -> None:
        ratio = float(np.clip(self.progress, 0.0, 1.0))
        self.source = "bezier" if self.source == "dense" else "dense"
        self.current_time = ratio * self.active.duration


@dataclass(frozen=True)
class FeasibilityStatus:
    body_valid: bool
    leg_counts: tuple[int, ...]
    level: str


def classify_feasibility(
    body_collision_free: np.ndarray,
    leg_feasible: np.ndarray,
    min_landing_per_leg: int = 3,
) -> FeasibilityStatus:
    """Collapse the planner masks into strong/weak/invalid display state."""
    body_valid = bool(np.asarray(body_collision_free, dtype=bool).all())
    mask = np.asarray(leg_feasible, dtype=bool)
    if mask.ndim != 3 or mask.shape[0] != 6:
        raise ValueError("leg_feasible must have shape (6, landing_count, branches)")
    counts = tuple(int(value) for value in mask.any(axis=-1).sum(axis=1))
    if body_valid and all(value >= min_landing_per_leg for value in counts):
        level = "strong"
    elif body_valid and all(value >= 1 for value in counts):
        level = "weak"
    else:
        level = "invalid"
    return FeasibilityStatus(body_valid, counts, level)


def _surface_flat_indices(voxels: Any, mask_flat: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_flat, dtype=bool).reshape(-1)
    candidate = np.flatnonzero(mask)
    if candidate.size == 0:
        return candidate
    grid = voxels.FlatIndex2GridIndex(candidate)
    offsets = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.int32,
    )
    boundary = np.zeros(candidate.size, dtype=bool)
    shape = np.asarray(voxels.grid_shape, dtype=np.int32)
    for offset in offsets:
        neighbor = grid + offset
        inside = ((neighbor >= 0) & (neighbor < shape)).all(axis=1)
        boundary |= ~inside
        if inside.any():
            neighbor_flat = voxels.GridIndex2FlatIndex(neighbor[inside])
            boundary[inside] |= ~mask[neighbor_flat]
    return candidate[boundary]


class SE3PathVisualizer:
    """Interactive PyVista scene for a loaded :class:`SE3PathData`."""

    def __init__(
        self,
        hex_state: HexState,
        path_data: SE3PathData,
        map_stl: str | Path = DEFAULT_MAP_STL,
        source: str = "dense",
        landing_stride: int = 10,
        curve_dt: float = 0.05,
        speed: float = 1.0,
        min_landing_per_leg: int = 3,
    ) -> None:
        if landing_stride < 1:
            raise ValueError("landing_stride must be at least 1")
        if min_landing_per_leg < 1:
            raise ValueError("min_landing_per_leg must be at least 1")
        self.hex_state = hex_state
        self.pointmap = hex_state.env_pointsmap_voxels
        self.path_data = path_data
        self.map_stl = Path(map_stl)
        if not self.map_stl.is_file():
            raise FileNotFoundError(f"map STL not found: {self.map_stl}")
        self.landing_stride = int(landing_stride)
        self.min_landing_per_leg = int(min_landing_per_leg)
        self.playback = PathPlayback(
            path_data.dense, path_data.sample_bezier(curve_dt), source, speed
        )
        self.plotter: pv.Plotter | None = None
        self.follow_camera = False
        self.show_feasible_landings = True
        self._last_wall_time = time.monotonic()
        self._last_robot_position: np.ndarray | None = None
        self._timer_id: int | None = None
        self._timer_observer_id: int | None = None
        self._path_actor: Any = None
        self._status_actor: Any = None
        self._body_mesh: pv.PolyData | None = None
        self._body_actor: Any = None
        self._leg_meshes: list[pv.PolyData] = []
        self._leg_actors: list[Any] = []
        self._landing_meshes: list[pv.PolyData] = []
        self._landing_actors: list[Any] = []
        self._body_surface_R, self._leg_surfaces_R = self._robot_surfaces_R()

    def _robot_surfaces_R(self) -> tuple[np.ndarray, list[np.ndarray]]:
        robot = self.hex_state.robot_voxels
        body_voxels = robot.body_voxels
        body_surface = _surface_flat_indices(body_voxels, body_voxels.esdf_flat_for_env <= 0.0)
        body_points = np.asarray(body_voxels.center[body_surface], dtype=np.float64)
        within_radius = np.linalg.norm(robot.leg_voxels.center, axis=1) <= 0.3
        legs: list[np.ndarray] = []
        for leg_index in range(6):
            reachable = robot.robot_reachable_legs[..., leg_index].any(axis=1)
            reachable &= within_radius
            reachable &= robot.to_bound_dist_flat[:, leg_index] >= 0.042
            surface = _surface_flat_indices(robot.leg_voxels, reachable)
            points_B = robot.leg_voxels.center[surface]
            points_R = robot.kin._B2R(points_B.T, leg_index).T
            legs.append(np.asarray(points_R, dtype=np.float64))
        return body_points, legs

    @staticmethod
    def _world_points(pose: SE3, points_R: np.ndarray) -> np.ndarray:
        if points_R.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray((pose * points_R.T).T, dtype=np.float64)

    def build_scene(self, off_screen: bool = False) -> pv.Plotter:
        """Build all static and mutable actors without entering the GUI loop."""
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=(1440, 900))
        self.plotter.set_background("#f7fafc")
        environment = pv.read(str(self.map_stl))
        self.plotter.add_mesh(
            environment, color="#a8bed0", opacity=0.18, smooth_shading=True,
            name="environment",
        )
        landing = np.asarray(self.pointmap.landing_points[:: self.landing_stride])
        self.plotter.add_mesh(
            pv.PolyData(landing), color="#45a049", point_size=6,
            render_points_as_spheres=True, name="static_landings",
        )
        self._replace_path_actor()
        self._add_waypoints()
        self._add_robot_actors()
        self._status_actor = self.plotter.add_text(
            "", position="upper_left", font_size=10, color="#1a202c", name="status"
        )
        self.plotter.add_text(
            "Space play/pause | ←/→ step | C source | F camera | L landings | H help",
            position="lower_left", font_size=9, color="#4a5568", name="key_hint",
        )
        self._register_keys()
        self._update_frame()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        return self.plotter

    def _replace_path_actor(self) -> None:
        assert self.plotter is not None
        if self._path_actor is not None:
            self.plotter.remove_actor(self._path_actor, render=False)
        points = np.asarray([pose.t for pose in self.playback.active.poses])
        curve = pv.lines_from_points(points, close=False)
        self._path_actor = self.plotter.add_mesh(
            curve, color=PATH_BLUE, line_width=7, render_lines_as_tubes=True,
            name="active_path",
        )

    def _add_waypoints(self) -> None:
        assert self.plotter is not None
        positions = np.asarray([pose.t for pose in self.path_data.waypoints])
        self.plotter.add_mesh(
            pv.PolyData(positions), color=WAYPOINT_ORANGE, point_size=15,
            render_points_as_spheres=True, name="waypoints",
        )
        self.plotter.add_point_labels(
            positions, [f"W{i}" for i in range(len(positions))], font_size=12,
            text_color="#b45309", point_size=0, shape=None, always_visible=True,
            name="waypoint_labels",
        )
        axes = (("#d62828", 0), ("#2a9d4b", 1), ("#2166d1", 2))
        for waypoint_index, pose in enumerate(self.path_data.waypoints):
            for color, axis in axes:
                arrow = pv.Arrow(pose.t, pose.R[:, axis], scale=0.14)
                self.plotter.add_mesh(
                    arrow, color=color, name=f"waypoint_{waypoint_index}_axis_{axis}"
                )

    @staticmethod
    def _mutable_cloud(points: np.ndarray) -> pv.PolyData:
        # VTK cannot render a truly empty point set reliably. Hidden one-point
        # clouds are used until the first feasibility result is available.
        return pv.PolyData(points if len(points) else np.zeros((1, 3)))

    def _add_robot_actors(self) -> None:
        assert self.plotter is not None
        pose = self.playback.pose
        self._body_mesh = self._mutable_cloud(self._world_points(pose, self._body_surface_R))
        self._body_actor = self.plotter.add_mesh(
            self._body_mesh, color=BODY_GRAY, point_size=5, render_points_as_spheres=True,
            name="robot_body",
        )
        for index, points_R in enumerate(self._leg_surfaces_R):
            mesh = self._mutable_cloud(self._world_points(pose, points_R))
            actor = self.plotter.add_mesh(
                mesh, color=LEG_COLORS[index], opacity=0.52, point_size=4,
                render_points_as_spheres=True, name=f"robot_leg_{index}",
            )
            landing_mesh = self._mutable_cloud(np.zeros((0, 3)))
            landing_actor = self.plotter.add_mesh(
                landing_mesh, color=LEG_COLORS[index], point_size=13,
                render_points_as_spheres=True, name=f"feasible_landing_{index}",
            )
            landing_actor.SetVisibility(False)
            self._leg_meshes.append(mesh)
            self._leg_actors.append(actor)
            self._landing_meshes.append(landing_mesh)
            self._landing_actors.append(landing_actor)

    @staticmethod
    def _set_actor_color(actor: Any, color: str) -> None:
        actor.prop.color = color

    def _query_feasibility(
        self, pose: SE3
    ) -> tuple[FeasibilityStatus, np.ndarray, np.ndarray, str | None]:
        try:
            _, landing_idx, body_mask, leg_mask = self.hex_state.RobotFeasiCheck(pose)
            status = classify_feasibility(
                body_mask, leg_mask, self.min_landing_per_leg
            )
            return status, np.asarray(landing_idx, dtype=np.int64), np.asarray(leg_mask), None
        except Exception as exc:  # Keep visualization responsive on malformed map data.
            status = FeasibilityStatus(False, (0, 0, 0, 0, 0, 0), "invalid")
            empty = np.zeros((6, 0, 2), dtype=bool)
            return status, np.zeros(0, dtype=np.int64), empty, str(exc)

    def _update_frame(self) -> None:
        if self.plotter is None or self._body_mesh is None:
            return
        pose = self.playback.pose
        status, landing_idx, leg_mask, error = self._query_feasibility(pose)
        self._body_mesh.points = self._world_points(pose, self._body_surface_R)
        body_color = RED if not status.body_valid else (GREEN if status.level == "strong" else YELLOW)
        self._set_actor_color(self._body_actor, body_color)
        for index, (mesh, actor, points_R) in enumerate(
            zip(self._leg_meshes, self._leg_actors, self._leg_surfaces_R)
        ):
            mesh.points = self._world_points(pose, points_R)
            count = status.leg_counts[index]
            color = GREEN if count >= self.min_landing_per_leg else (YELLOW if count else RED)
            self._set_actor_color(actor, color)

            feasible = leg_mask[index].any(axis=-1) if leg_mask.shape[1] else np.zeros(0, bool)
            points = self.pointmap.landing_points[landing_idx[feasible]] if feasible.any() else np.zeros((0, 3))
            if len(points):
                self._landing_meshes[index].points = points
                self._landing_actors[index].SetVisibility(self.show_feasible_landings)
            else:
                self._landing_actors[index].SetVisibility(False)

        rpy = Rotation.from_matrix(pose.R).as_euler("xyz", degrees=True)
        state = status.level.upper()
        error_line = f"\nerror: {error}" if error else ""
        status_text = (
            f"{self.playback.source.upper()} | {state} | "
            f"t={self.playback.current_time:.2f}/{self.playback.active.duration:.2f}s "
            f"({100.0 * self.playback.progress:.1f}%) | speed={self.playback.speed:.2f}x\n"
            f"xyz=({pose.t[0]:.3f}, {pose.t[1]:.3f}, {pose.t[2]:.3f}) m | "
            f"rpy=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) deg\n"
            f"body={'free' if status.body_valid else 'collision'} | legs={list(status.leg_counts)}"
            f"{error_line}"
        )
        if hasattr(self._status_actor, "SetInput"):
            self._status_actor.SetInput(status_text)
        elif hasattr(self._status_actor, "SetText"):
            self._status_actor.SetText(0, status_text)

        if self.follow_camera:
            current = np.asarray(pose.t)
            if self._last_robot_position is not None:
                delta = current - self._last_robot_position
                camera = self.plotter.camera
                camera.position = tuple(np.asarray(camera.position) + delta)
                camera.focal_point = tuple(np.asarray(camera.focal_point) + delta)
            self._last_robot_position = current.copy()
        else:
            self._last_robot_position = np.asarray(pose.t).copy()
        self.plotter.render()

    def _register_keys(self) -> None:
        assert self.plotter is not None
        callbacks = {
            "space": self._toggle_play,
            "Right": lambda: self._step(1),
            "n": lambda: self._step(1),
            "Left": lambda: self._step(-1),
            "b": lambda: self._step(-1),
            "r": self._reset,
            "c": self._switch_source,
            "f": self._toggle_follow,
            "l": self._toggle_landings,
            "plus": lambda: self._change_speed(1.25),
            "equal": lambda: self._change_speed(1.25),
            "minus": lambda: self._change_speed(0.8),
            "h": self.print_help,
            "q": self._close,
            "Escape": self._close,
        }
        for key, callback in callbacks.items():
            self.plotter.add_key_event(key, callback)

    def _toggle_play(self) -> None:
        if self.playback.current_time >= self.playback.active.duration:
            self.playback.current_time = 0.0
        self.playback.playing = not self.playback.playing
        self._last_wall_time = time.monotonic()
        self._update_frame()

    def _step(self, direction: int) -> None:
        self.playback.step(direction)
        self._update_frame()

    def _reset(self) -> None:
        self.playback.reset()
        self._update_frame()

    def _switch_source(self) -> None:
        self.playback.switch_source()
        self._replace_path_actor()
        self._update_frame()

    def _toggle_follow(self) -> None:
        self.follow_camera = not self.follow_camera
        self._last_robot_position = np.asarray(self.playback.pose.t).copy()
        self._update_frame()

    def _toggle_landings(self) -> None:
        self.show_feasible_landings = not self.show_feasible_landings
        self._update_frame()

    def _change_speed(self, factor: float) -> None:
        self.playback.speed = float(np.clip(self.playback.speed * factor, 0.05, 20.0))
        self._update_frame()

    def _close(self) -> None:
        if self.plotter is not None:
            self.plotter.close()

    def _on_timer(self, _: int) -> None:
        now = time.monotonic()
        elapsed = now - self._last_wall_time
        self._last_wall_time = now
        if self.playback.playing:
            self.playback.advance(elapsed)
            self._update_frame()

    def _on_vtk_timer(self, _interactor: Any, _event: str) -> None:
        """Handle exactly one animation frame for each VTK TimerEvent.

        PyVista 0.44.2 ``add_timer_event`` runs all ``max_steps`` in a
        blocking ``while`` loop on the first event.  Registering the native
        repeating timer avoids starving window redraw and keyboard events.
        """
        self._on_timer(0)

    def _install_timer(self, duration_ms: int = 33) -> None:
        assert self.plotter is not None
        interactor = self.plotter.iren
        self._timer_observer_id = interactor.add_observer(
            "TimerEvent", self._on_vtk_timer
        )
        self._timer_id = interactor.create_timer(duration_ms, repeating=True)

    def _remove_timer(self) -> None:
        if self.plotter is None:
            return
        interactor = self.plotter.iren
        if interactor is None:
            return
        if self._timer_id is not None:
            try:
                interactor.destroy_timer(self._timer_id)
            except (AttributeError, RuntimeError):
                pass
            self._timer_id = None
        if self._timer_observer_id is not None:
            try:
                interactor.remove_observer(self._timer_observer_id)
            except (AttributeError, RuntimeError):
                pass
            self._timer_observer_id = None

    @staticmethod
    def print_help() -> None:
        print(
            "\nSE3 path viewer keys:\n"
            "  Space       auto play / pause\n"
            "  Right / N   next sample\n"
            "  Left / B    previous sample\n"
            "  R           reset to start\n"
            "  C           switch Dense / Bezier\n"
            "  F           fixed / follow camera\n"
            "  L           show / hide feasible landings\n"
            "  + / -       playback speed\n"
            "  H           print this help\n"
            "  Esc / Q     quit\n"
        )

    def run(self) -> None:
        plotter = self.build_scene(off_screen=False)
        self.print_help()
        self._last_wall_time = time.monotonic()
        self._install_timer(duration_ms=33)
        try:
            plotter.show(title="SE3 Path Feasibility Viewer")
        finally:
            self._remove_timer()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="version-1 SE3 path JSON")
    parser.add_argument("--source", choices=("dense", "bezier"), default="dense")
    parser.add_argument("--map-stl", default=DEFAULT_MAP_STL)
    parser.add_argument("--landing-stride", type=int, default=10)
    parser.add_argument("--curve-dt", type=float, default=0.05)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min-landing-per-leg", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path_data = load_se3_path_json(args.path)
    viewer = SE3PathVisualizer(
        HexState(Kinematic()),
        path_data,
        map_stl=args.map_stl,
        source=args.source,
        landing_stride=args.landing_stride,
        curve_dt=args.curve_dt,
        speed=args.speed,
        min_landing_per_leg=args.min_landing_per_leg,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
