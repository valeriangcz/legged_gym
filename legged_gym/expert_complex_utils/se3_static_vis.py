"""多路径静态查看器；修改下方配置后运行 python -m
legged_gym.expert_complex_utils.se3_static_vis。

POSE_SOURCE 与 POSE_INDICES 的外层按路径对应，内层每个来源对应一组索引。
例如 (["teleop"], ["waypoints", "dense_poses"]) 与 ([[0, 5, 10]], [[2], [100, 200]])。
索引组 [] 表示跳过该来源；兼容单个整数，例如 ([3], [2, 100])。
([], []) 表示两条路径都不显示机器人；仍绘制路径及路标。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from spatialmath import SE3

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import HexState, Kinematic
from legged_gym.expert_complex_utils.se3_animation_vis import (
    DEFAULT_MAP_STL, GREEN, YELLOW, RED, LEG_COLORS,
    SE3PathVisualizer, _parse_pose_block, classify_feasibility,
)


# ======================== 用户配置 ========================
PATH_DIR = Path(LEGGED_GYM_ROOT_DIR) / "legged_gym/expert_complex_utils/SE3_path"
PATH_FILE = [
    PATH_DIR / "feasibility_teleop_demo_20261105_220311.json",
    # PATH_DIR / "optimized_cma_es_torch_path_20261105_183455_665425.json"
    # PATH_DIR / "shortcut_initial_se3_path_20261105_173348.json",
]
# POSE_SOURCE = (["teleop"], ["waypoints", "dense_poses"])
# POSE_SOURCE = (["teleop"], ["waypoints"])
POSE_SOURCE = (["teleop"],)
# POSE_INDICES = ([[]],)  # 0-based；每条路径的两个列表均为空时不显示机器人
# POSE_INDICES = ([[0,10,40,62,78,100]],)  # 0-based；每个来源对应索引列表，[] 跳过该来源
POSE_INDICES = ([[0,1,2,3,4]],)  # 0-based；每个来源对应索引列表，[] 跳过该来源

# 以下样式支持单值（全部路径共用）或列表（按路径对应）；RGB 颜色用元组。
PATH_POINT_SIZE = [8.0]
PATH_FRAME_SIZE = [0.20]
PATH_LINE_COLOR = ["yellow"]
PATH_LINE_WIDTH = [20.0]

SHOW_MAP = True
SHOW_ENV_POINTS = True
SHOW_ENV_LANDINGS = True
SHOW_ENV_NONLANDINGS = True
SHOW_PATH = False
SHOW_AXES = False  # 显示各路径全部关键点坐标系，与机器人位姿选择无关
SHOW_LABELS = False
MAP_STL = DEFAULT_MAP_STL  # 应与 HexState 所加载的环境一致
MAP_OPACITY = 0.08
ENV_POINT_SIZE = 5.5
ENV_OPACITY = 0.18
ENV_NONLANDING_POINT_SIZE = 5.0
ENV_NONLANDING_OPACITY = 1.0
ENV_LANDING_POINT_SIZE = 6.0
ENV_LANDING_OPACITY = 1.0
ENV_LANDING_STRIDE = 1  # 与 keyboard_pose_labeler 一致，不降采样背景落脚点
BODY_POINT_SIZE = 20.0
BODY_OPACITY = 1.0
LEG_POINT_SIZE = 20.0
LEG_OPACITY = 0.4
FEASIBLE_LANDING_POINT_SIZE = 5.0
FEASIBLE_LANDING_OPACITY = 1.0
MIN_LANDING_PER_LEG = 6  # 身体无碰撞时：各腿均 >=6 为绿色；某腿 1~5 为黄色；某腿为0 为红色
WINDOW_SIZE = (1280, 820)
# 固定初始视角：(相机位置, 观察目标, 屏幕向上方向)，均为世界坐标。
# 调整前两项控制观察方向和距离；设为 None 则自动适配场景。
CAMERA_POSITION = (
    (2.88, 12.07, 7.29),
    (1.5, 2.5, 0.5),
    (-0.068, -0.058, 0.811),
)
# ==========================================================


@dataclass(frozen=True)
class StaticPath:
    file: Path
    sources: dict[str, tuple[SE3, ...]]
    marker_source: str
    line_poses: tuple[SE3, ...]
    selections: tuple[tuple[str, int], ...]
    point_size: float
    frame_size: float
    line_color: Any
    line_width: float


def _per_path(value: Any, count: int, name: str) -> list:
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(f"{name}: expected {count} entries, got {len(value)}")
        return value
    return [value] * count


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name}: expected a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected a positive finite number")
    return result


def load_paths() -> tuple[StaticPath, ...]:
    """读取文件开头配置并校验，尚不加载地图或机器人数据。"""
    if not isinstance(PATH_FILE, (list, tuple)) or not PATH_FILE:
        raise ValueError("PATH_FILE: expected a nonempty list of paths")
    count = len(PATH_FILE)
    for name, value in (("POSE_SOURCE", POSE_SOURCE), ("POSE_INDICES", POSE_INDICES)):
        if not isinstance(value, tuple) or len(value) != count:
            raise ValueError(f"{name}: expected a tuple containing {count} lists")
    styles = {
        name: _per_path(value, count, name) for name, value in (
            ("PATH_POINT_SIZE", PATH_POINT_SIZE), ("PATH_FRAME_SIZE", PATH_FRAME_SIZE),
            ("PATH_LINE_COLOR", PATH_LINE_COLOR), ("PATH_LINE_WIDTH", PATH_LINE_WIDTH),
        )
    }
    paths = []
    for number, filename in enumerate(PATH_FILE):
        file = Path(filename).expanduser()
        context = f"path[{number}] ({file})"
        try:
            with file.open(encoding="utf-8") as stream:
                root = json.load(stream)
            if not isinstance(root, dict):
                raise ValueError("JSON root must be an object")
            if "waypoints" in root or "dense_poses" in root:
                sources = {key: _parse_pose_block(root.get(key), key)
                           for key in ("waypoints", "dense_poses")}
                marker_source, line_source = "waypoints", "dense_poses"
            else:
                sources = {"teleop": _parse_pose_block(root, "teleop")}
                marker_source = line_source = "teleop"
            source_list, index_list = POSE_SOURCE[number], POSE_INDICES[number]
            if not isinstance(source_list, list) or not isinstance(index_list, list):
                raise ValueError("POSE_SOURCE and POSE_INDICES entries must be lists")
            if len(source_list) != len(index_list):
                raise ValueError("POSE_SOURCE and POSE_INDICES lists must have equal lengths")
            selections = []
            for source, indices in zip(source_list, index_list):
                if not isinstance(source, str) or source not in ("teleop", "waypoints", "dense_poses"):
                    raise ValueError(f"POSE_SOURCE: unknown source {source!r}")
                indices = indices if isinstance(indices, list) else [indices]
                if not indices:
                    continue  # 空组允许为当前文件不具备的来源保留配置位置
                if source not in sources:
                    raise ValueError(f"POSE_SOURCE: {source!r}; available: {list(sources)}")
                for index in indices:
                    if (isinstance(index, bool) or not isinstance(index, int)
                            or not 0 <= index < len(sources[source])):
                        raise ValueError(f"POSE_INDICES: {source}[{index!r}]; valid range "
                                         f"0..{len(sources[source]) - 1}")
                    if (source, index) not in selections:
                        selections.append((source, index))
            sizes = {key: _positive(styles[key][number], key) for key in (
                "PATH_POINT_SIZE", "PATH_FRAME_SIZE", "PATH_LINE_WIDTH")}
            color = styles["PATH_LINE_COLOR"][number]
            pv.Color(color)  # 提前报告颜色错误，避免部分场景构建后才失败
            paths.append(StaticPath(
                file, sources, marker_source, sources[line_source], tuple(selections),
                sizes["PATH_POINT_SIZE"], sizes["PATH_FRAME_SIZE"], color,
                sizes["PATH_LINE_WIDTH"],
            ))
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"{context}: {exc}") from exc
    return tuple(paths)


class SE3StaticVisualizer:
    """共享一个环境的静态场景；build_scene 可用于离屏检查。"""

    def __init__(self, paths: tuple[StaticPath, ...], hex_state: HexState) -> None:
        self.paths = paths
        self.hex_state = hex_state
        self.pointmap = hex_state.env_pointsmap_voxels
        self.plotter: pv.Plotter | None = None
        self._camera_text_actor: Any = None
        self.status_lines: list[str] = []
        self._body_surface_R = np.empty((0, 3))
        self._leg_surfaces_R: list[np.ndarray] = []
        if any(path.selections for path in paths):
            # 此几何方法仅依赖 self.hex_state，与动画查看器保持相同筛选规则。
            self._body_surface_R, self._leg_surfaces_R = (
                SE3PathVisualizer._robot_surfaces_R(self)
            )

    def _cloud(self, name: str, points: np.ndarray, **style: Any) -> None:
        if len(points):
            self.plotter.add_mesh(pv.PolyData(points), name=name,
                                  render_points_as_spheres=True,
                                  reset_camera=False, render=False, **style)

    def _frame(self, name: str, pose: SE3, size: float) -> None:
        if SHOW_AXES:
            self.plotter.add_mesh(SE3PathVisualizer._coordinate_frame(size, pose),
                                  name=name, scalars="rgb", rgb=True)

    def _labels(self, name: str, poses: list[SE3], labels: list[str]) -> None:
        if SHOW_LABELS and poses:
            self.plotter.add_point_labels(
                np.asarray([pose.t for pose in poses]), labels, name=name,
                font_size=14, text_color="black", point_size=0,
                shape=None, always_visible=True,
            )

    def update_camera_position(self, *_: Any) -> None:
        """在窗口显示当前相机配置；也可直接作为相机 ModifiedEvent 回调。"""
        if self.plotter is None:
            return
        vectors = [tuple(float(value) for value in vector)
                   for vector in self.plotter.camera_position]
        lines = ["CAMERA_POSITION = ("]
        lines.extend("    (" + ", ".join(f"{value:.6f}" for value in vector) + "),"
                     for vector in vectors)
        lines.append(")")
        text = "\n".join(lines)
        if self._camera_text_actor is None:
            self._camera_text_actor = self.plotter.add_text(
                text, position="upper_right", font_size=9, color="black",
                name="camera_position", render=False,
            )
        else:
            self._camera_text_actor.SetText(3, text)
        # 相机交互会触发重绘；此处不主动 render，避免回调递归。

    def _add_environment(self) -> None:
        if SHOW_MAP:
            self.plotter.add_mesh(pv.read(str(MAP_STL)), name="environment",
                                  color="lightgray", opacity=MAP_OPACITY,
                                  show_edges=True, edge_color="silver",
                                  reset_camera=False, render=False)
        if SHOW_ENV_POINTS:
            raw = getattr(self.pointmap, "_raw_points", None)
            self._cloud("environment_points", np.asarray(
                np.empty((0, 3)) if raw is None else raw),
                color="lightgray", point_size=ENV_POINT_SIZE, opacity=ENV_OPACITY)
        if SHOW_ENV_LANDINGS:
            self._cloud("environment_landings",
                        np.asarray(self.pointmap.landing_points[::ENV_LANDING_STRIDE]),
                        color="tomato", point_size=ENV_LANDING_POINT_SIZE,
                        opacity=ENV_LANDING_OPACITY)
        if SHOW_ENV_NONLANDINGS:
            self._cloud("environment_nonlandings",
                        np.asarray(self.pointmap.points[self.pointmap.landing_count:]),
                        color="black", point_size=ENV_NONLANDING_POINT_SIZE,
                        opacity=ENV_NONLANDING_OPACITY)

    def _add_robot(self, prefix: str, pose: SE3) -> None:
        color = "gray"
        try:
            _, landing_idx, body_mask, leg_mask = self.hex_state.RobotFeasiCheck(pose)
            status = classify_feasibility(body_mask, leg_mask, MIN_LANDING_PER_LEG)
            landing_idx = np.asarray(landing_idx, dtype=np.int64)
            leg_mask = np.asarray(leg_mask, dtype=bool)
            # 提前计算全部落脚点，若数据有误则不绘制部分结果。
            landings = [self.pointmap.landing_points[landing_idx[mask.any(axis=-1)]]
                        for mask in leg_mask]
            color = {"strong": GREEN, "weak": YELLOW, "invalid": RED}[status.level]
            message = (f"{prefix}: {status.level}; body="
                       f"{'free' if status.body_valid else 'collision'}; "
                       f"legs={list(status.leg_counts)}")
        except Exception as exc:
            landings = []
            message = f"{prefix}: ERROR {type(exc).__name__}: {exc}"
        self.status_lines.append(message)
        print(message)
        world = SE3PathVisualizer._world_points
        self._cloud(f"{prefix}_body", world(pose, self._body_surface_R),
                    color=color, point_size=BODY_POINT_SIZE, opacity=BODY_OPACITY)
        for leg, points in enumerate(self._leg_surfaces_R):
            self._cloud(f"{prefix}_leg_{leg}", world(pose, points),
                        color=LEG_COLORS[leg], point_size=LEG_POINT_SIZE, opacity=LEG_OPACITY)
        for leg, points in enumerate(landings):
            self._cloud(f"{prefix}_landing_{leg}", points,
                        color=LEG_COLORS[leg], point_size=FEASIBLE_LANDING_POINT_SIZE,
                        opacity=FEASIBLE_LANDING_OPACITY)

    def build_scene(self, off_screen: bool = False) -> pv.Plotter:
        """构建并返回场景；调用者负责 show() 或 close()。"""
        _validate_scene_config()
        self.status_lines = []
        self._camera_text_actor = None
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=WINDOW_SIZE)
        self.plotter.set_background("white")
        try:
            self.plotter.enable_custom_trackball_style(
                left="rotate", control_left="pan", shift_left="spin",
                right="pan", control_right="pan", shift_right="pan",
            )
            self._add_environment()
            for number, path in enumerate(self.paths):
                prefix = f"path_{number}"
                markers = path.sources[path.marker_source]
                if SHOW_PATH:
                    points = np.asarray([pose.t for pose in path.line_poses])
                    if len(points) > 1:
                        self.plotter.add_mesh(pv.lines_from_points(points, close=False),
                                              name=f"{prefix}_line", color=path.line_color,
                                              line_width=path.line_width)
                    self._cloud(f"{prefix}_points", points, color=path.line_color,
                                point_size=path.point_size)
                    self._cloud(f"{prefix}_markers", np.asarray([p.t for p in markers]),
                                color=path.line_color, point_size=path.point_size)
                for index, pose in enumerate(markers):
                    self._frame(f"{prefix}_{path.marker_source}_{index}_frame",
                                pose, path.frame_size)
                marker_labels = [f"P{number}/{path.marker_source}/{i}"
                                 for i in range(len(markers))]
                self._labels(f"{prefix}_marker_labels", list(markers), marker_labels)
                selected_poses, selected_labels = [], []
                for source, index in path.selections:
                    pose = path.sources[source][index]
                    name = f"{prefix}_{source}_{index}"
                    self._add_robot(name, pose)
                    if source != path.marker_source:
                        self._frame(f"{name}_frame", pose, path.frame_size)
                    if source != path.marker_source:
                        selected_poses.append(pose)
                        selected_labels.append(f"P{number}/{source}/{index}")
                self._labels(f"{prefix}_selected_labels", selected_poses, selected_labels)
            if self.status_lines:
                self.plotter.add_text("\n".join(self.status_lines), position="upper_left",
                                      font_size=9, color="black", name="status")
            if CAMERA_POSITION is None:
                self.plotter.reset_camera(render=False)
            else:
                # 最后设置，避免添加图层或自动适配场景覆盖固定视角。
                self.plotter.camera_position = CAMERA_POSITION
                self.plotter.reset_camera_clipping_range()
        except Exception:
            self.plotter.close()
            raise
        return self.plotter


def _validate_scene_config() -> None:
    if CAMERA_POSITION is not None:
        camera = np.asarray(CAMERA_POSITION, dtype=float)
        if camera.shape != (3, 3) or not np.isfinite(camera).all():
            raise ValueError("CAMERA_POSITION: expected three finite xyz vectors or None")
        direction = camera[1] - camera[0]
        if (np.linalg.norm(direction) <= 1e-12 or np.linalg.norm(camera[2]) <= 1e-12
                or np.linalg.norm(np.cross(direction, camera[2]))
                <= 1e-12 * np.linalg.norm(direction) * np.linalg.norm(camera[2])):
            raise ValueError("CAMERA_POSITION: position and target must differ; "
                             "view-up must be nonzero and not parallel to viewing direction")
    for name, value in (("ENV_LANDING_STRIDE", ENV_LANDING_STRIDE),
                        ("MIN_LANDING_PER_LEG", MIN_LANDING_PER_LEG)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name}: expected a positive integer")
    for name in ("MAP_OPACITY", "ENV_OPACITY", "ENV_LANDING_OPACITY", "ENV_NONLANDING_OPACITY", "BODY_OPACITY",
                 "LEG_OPACITY", "FEASIBLE_LANDING_OPACITY"):
        value = globals()[name]
        if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name}: expected a number in [0, 1]")
    for name in ("ENV_POINT_SIZE", "ENV_LANDING_POINT_SIZE", "ENV_NONLANDING_POINT_SIZE", "BODY_POINT_SIZE",
                 "LEG_POINT_SIZE", "FEASIBLE_LANDING_POINT_SIZE"):
        _positive(globals()[name], name)


def main() -> None:
    paths = load_paths()
    _validate_scene_config()
    viewer = SE3StaticVisualizer(paths, HexState(Kinematic()))
    plotter = viewer.build_scene()
    viewer.update_camera_position()
    camera = plotter.camera
    camera_observer = camera.AddObserver("ModifiedEvent", viewer.update_camera_position)
    try:
        plotter.show()
    finally:
        camera.RemoveObserver(camera_observer)
        plotter.close()


if __name__ == "__main__":
    main()
