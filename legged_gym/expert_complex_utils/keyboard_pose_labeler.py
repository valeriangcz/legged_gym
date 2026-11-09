from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import time

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import HexState, Kinematic


# 固定初始视角：(相机位置, 观察目标, 屏幕向上方向)，均为世界坐标。
# 调整前两项控制观察方向和距离；设为 None 则自动适配场景。
CAMERA_POSITION = (
    (2.88, 12.07, 7.29),
    (1.5, 2.5, 0.5),
    (-0.068, -0.058, 0.811),
)


GREEN = [0.1, 0.8, 0.1]
YELLOW = [0.9, 0.7, 0.1]
RED = [0.9, 0.1, 0.1]
GRAY = [0.5, 0.5, 0.5]
# RGB in [0, 1]: origin sphere of Ctrl+Space protected poses.
PROTECTED_POSE_ORIGIN_COLOR = [0.85, 0.1, 0.85]
BLUE = [0.1, 0.25, 0.9]
QUERY_GREEN = [0.0, 0.9, 0.3]
LEG_COLORS = [
    [0.9, 0.1, 0.1],
    [0.1, 0.65, 0.2],
    [0.1, 0.25, 0.9],
    [0.9, 0.75, 0.05],
    [0.75, 0.15, 0.9],
    [0.0, 0.75, 0.85],
]


@dataclass
class PoseTeleopState:
    """保存键盘遥控过程中的可变状态。"""

    # W_T_R: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    W_T_R: SE3 = field(default_factory=lambda: SE3())
    translation_step: float = 0.01
    rotation_step: float = np.deg2rad(2.0)
    saved_poses: dict = field(default_factory=lambda: {"t":[],"ang":[],"vec":[],"W_T_R":[]})
    protected_pose_indices: list[int] = field(default_factory=list)
    last_eval: dict | None = None
    label_mode: bool = False
    initial_W_T_R: SE3 | None = None


class KeyboardPoseLabeler:
    """PyVista 键盘遥控位姿标注器。

    启动后先进入 INIT 模式，用户只移动机器人并观察有效性颜色；
    确认初始位姿后进入 LABEL 模式，才允许保存和导出轨迹。
    """

    def __init__(
        self,
        hex_state: HexState,
        init_pose: SE3 | None = None,
        landing_stride: int = 1,
        output_dir: str | None = None,
    ):
        """绑定 HexState 和可视化参数，并提前创建 PyVista 几何对象。"""
        self.hex_state = hex_state
        self.pointmap = hex_state.env_pointsmap_voxels
        self.state = PoseTeleopState()
        if init_pose is not None:
            # self.state.W_T_R = np.asarray(init_pose, dtype=np.float64).copy()
            self.state.W_T_R = init_pose.copy()

        self.landing_stride = max(1, int(landing_stride))
        self.output_dir = output_dir or os.path.join(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path")

        # 手动修改此路径以切换标注时显示的环境 STL；它必须与 pointmap 使用相同坐标系和单位。
        self.environment_stl_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "resources/environments/structure2/complex_surface4.STL",
        )

        self.vis: pv.Plotter | None = None
        self._cloud_actors = {}
        self.body_surface_points_R = self._body_surface_points_R()
        self.leg_surface_points_R = self._leg_surface_points_R()
        self.body_surface_cloud = self._make_point_cloud(np.zeros((0, 3)))
        self.leg_surface_clouds = [
            self._make_point_cloud(np.zeros((0, 3)))
            for i in range(6)
        ]
        self.frame_mesh: pv.PolyData | None = None
        self.saved_frame_meshes: list[pv.PolyData] = []
        self.env_mesh = self._load_environment_mesh()
        self.query_highlight_cloud = self._make_point_cloud(np.zeros((0, 3)))
        self.feasible_landing_clouds = [
            self._make_point_cloud(np.zeros((0, 3)))
            for i in range(6)
        ]
        self.landing_cloud = self._make_point_cloud(
            self.pointmap.landing_points[:: self.landing_stride],
        )
        raw_points = getattr(self.pointmap, "_raw_points", None)
        self.raw_cloud = self._make_point_cloud(
            raw_points if raw_points is not None else np.zeros((0, 3)),
        )
        self.nonlanding_cloud = self._make_point_cloud(
            self.pointmap.points[self.pointmap.landing_count:],
        )
        self.trajectory_line = pv.PolyData()
        self.trajectory_line_added = False

    def run(self):
        """创建窗口、注册按键回调，并进入 PyVista 事件循环。"""
        self._build_scene()
        self._print_help()
        self._evaluate_and_render("init")
        self.vis.show(title="Keyboard Pose Labeler")

    def _build_scene(self, off_screen=False):
        """创建渲染场景；离屏模式用于检查绘制和事件适配。"""
        self.vis = pv.Plotter(window_size=(1280, 820), off_screen=off_screen)
        self._cloud_actors.clear()
        self.vis.set_background("white")
        self.vis.enable_custom_trackball_style(
            left="rotate", control_left="pan", shift_left="spin",
            right="pan", control_right="pan", shift_right="pan",
        )

        # 静态环境网格和点云只添加一次；动态几何在按键回调中原地更新。
        self.vis.add_mesh(
            self.env_mesh, name="environment", color="lightgray", opacity=0.08,
            show_edges=True, edge_color="silver", reset_camera=False, render=False,
        )
        self._show_cloud("raw_points", self.raw_cloud, "lightgray", 5.5, 0.18)
        self._show_cloud("landing_points", self.landing_cloud, "tomato", 6.0)
        self._show_cloud("nonlanding_points", self.nonlanding_cloud, "black", 5.0)
        self._show_cloud("query_highlight", self.query_highlight_cloud, QUERY_GREEN)
        for i, cloud in enumerate(self.feasible_landing_clouds):
            self._show_cloud(f"feasible_landing_{i}", cloud, LEG_COLORS[i])
        self._update_robot_surface_geometry()
        self._show_cloud("robot_body", self.body_surface_cloud, GRAY, 20.0)
        for i, cloud in enumerate(self.leg_surface_clouds):
            self._show_cloud(f"robot_leg_{i}", cloud, LEG_COLORS[i], 20.0, 0.4)
        self._replace_frame_mesh()
        self._register_callbacks()
        if CAMERA_POSITION is None:
            self.vis.reset_camera(render=False)
        else:
            self.vis.camera_position = CAMERA_POSITION
            self.vis.reset_camera_clipping_range()
        return self.vis

    def _surface_flat_indices(self, voxels, mask_flat: np.ndarray) -> np.ndarray:
        """返回体素 mask 的 6 邻域外侧 flat index。"""
        mask_flat = np.asarray(mask_flat, dtype=bool).reshape(-1)
        candidate_flat = np.flatnonzero(mask_flat)
        if candidate_flat.size == 0:
            return candidate_flat

        grid_index = voxels.FlatIndex2GridIndex(candidate_flat)
        offsets = np.asarray(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.int32,
        )
        boundary = np.zeros(candidate_flat.shape[0], dtype=bool)
        grid_shape = np.asarray(voxels.grid_shape, dtype=np.int32)
        for offset in offsets:
            neighbor_grid = grid_index + offset[None, :]
            inside = ((neighbor_grid >= 0) & (neighbor_grid < grid_shape[None, :])).all(axis=1)
            boundary |= ~inside
            if inside.any():
                neighbor_flat = voxels.GridIndex2FlatIndex(neighbor_grid[inside])
                boundary[inside] |= ~mask_flat[neighbor_flat]
        return candidate_flat[boundary]

    def _body_surface_points_R(self) -> np.ndarray:
        """从 body_voxels 的占用体素中提取身体外侧点云，坐标在 R 系。"""
        body_voxels = self.hex_state.robot_voxels.body_voxels
        body_occ = np.asarray(body_voxels.esdf_flat_for_env <= 0.0, dtype=bool)
        surface_flat = self._surface_flat_indices(body_voxels, body_occ)
        return np.asarray(body_voxels.center[surface_flat], dtype=np.float64)

    def _leg_surface_points_R(self) -> list[np.ndarray]:
        """从 robot_reachable_legs 中提取每条腿可行体素外侧点云，转换到 R 系。"""
        robot_voxels = self.hex_state.robot_voxels
        leg_voxels = robot_voxels.leg_voxels
        #根据实际的距离修改robot_reachable_legs的值，距离中心太远或者距离边界太近
        #grid_size
        within_force_mask = robot_voxels.leg_voxels.mechanical_margin >= self.hex_state.LEG_FORCE_HARD_MARGIN
        
        #grid_size
        # within_mask = np.linalg.norm(robot_voxels.leg_voxels.center,axis=1)<=0.3
        # within_mask = np.linalg.norm(robot_voxels.leg_voxels.center,axis=1)>=0.14
        #grid_size,6
        # inside_mask = robot_voxels.to_bound_dist_flat >= 0.0
        inside_mask = robot_voxels.to_bound_dist_flat >= self.hex_state.LEG_BOUNDARY_HARD_MARGIN
        leg_points_R = []
        for leg_index in range(6):
            reachable = robot_voxels.robot_reachable_legs[..., leg_index].any(axis=1)
            # reachable = reachable&within_mask&inside_mask[:,leg_index]
            reachable = reachable&within_force_mask&inside_mask[:,leg_index]
            # reachable = reachable&within_force_mask&within_mask

            surface_flat = self._surface_flat_indices(leg_voxels, reachable)
            points_B = leg_voxels.center[surface_flat]
            if points_B.shape[0] == 0:
                leg_points_R.append(np.zeros((0, 3), dtype=np.float64))
                continue
            points_R = robot_voxels.kin._B2R(points_B.T, leg_index).T
            leg_points_R.append(np.asarray(points_R, dtype=np.float64))
        return leg_points_R

    def _make_point_cloud(self, points: np.ndarray) -> pv.PolyData:
        """把 numpy 点集转换成具有顶点单元的 PyVista 点云。"""
        return pv.PolyData(np.asarray(points, dtype=np.float64))

    def _set_cloud_points(self, cloud: pv.PolyData, points: np.ndarray):
        """点数变化时同步重建顶点单元，保留 mapper 使用的数据对象。"""
        cloud.copy_from(self._make_point_cloud(points))

    def _show_cloud(self, name, cloud, color, point_size=5.0, opacity=1.0):
        """添加或更新点云 actor，空点集移除显示且不改变相机。"""
        assert self.vis is not None
        if cloud.n_points == 0:
            self.vis.remove_actor(name, reset_camera=False, render=False)
            self._cloud_actors.pop(name, None)
            return
        actor = self._cloud_actors.get(name)
        if actor is None:
            actor = self.vis.add_mesh(
                cloud, name=name, color=color, point_size=point_size, opacity=opacity,
                render_points_as_spheres=True, reset_camera=False, render=False,
            )
            self._cloud_actors[name] = actor
        else:
            actor.prop.color = color

    @staticmethod
    def _coordinate_frame(size: float, pose: SE3, origin_color=None) -> pv.PolyData:
        """创建与原窗口同尺寸的 RGB 箭头坐标系，并放到世界坐标系。"""
        frame = pv.Sphere(radius=0.1*size)
        color = GRAY if origin_color is None else origin_color
        frame["rgb"] = np.tile(np.asarray(color)*255, (frame.n_points, 1)).astype(np.uint8)
        for direction, color in zip(np.eye(3), np.eye(3)):
            arrow = pv.Arrow(
                direction=direction, scale=size, tip_length=0.2,
                tip_radius=0.06, shaft_radius=0.035,
            )
            arrow["rgb"] = np.tile(color*255, (arrow.n_points, 1)).astype(np.uint8)
            frame = frame.merge(arrow, merge_points=False)
        return frame.transform(pose.A, inplace=True)

    def _load_environment_mesh(self) -> pv.PolyData:
        """读取手动指定的环境 STL，并创建用于显示的浅灰色网格。"""
        if not os.path.isfile(self.environment_stl_path):
            raise FileNotFoundError(
                f"environment STL file does not exist: {self.environment_stl_path}"
            )

        mesh = pv.read(self.environment_stl_path)
        if mesh.n_points == 0 or mesh.n_cells == 0:
            raise ValueError(
                f"failed to load a valid triangle mesh from STL: {self.environment_stl_path}"
            )
        return mesh

    def _transform_points_R_to_W(self, points_R: np.ndarray) -> np.ndarray:
        """把 R 系点云批量转换到 W 系。"""
        if points_R.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray((self.state.W_T_R * points_R.T).T, dtype=np.float64)

    def _update_robot_surface_geometry(self):
        """按当前身体位姿更新身体和腿部外侧点云。"""
        self._set_cloud_points(
            self.body_surface_cloud, self._transform_points_R_to_W(self.body_surface_points_R)
        )
        for leg_index,(cloud,points_R) in enumerate(zip(self.leg_surface_clouds,self.leg_surface_points_R)):
            self._set_cloud_points(cloud, self._transform_points_R_to_W(points_R))


    def _query_robot_feasi_env_indices(self) -> np.ndarray:
        """查询 RobotFeasiCheck 用于筛选腿部候选的附近环境点索引。"""
        if self.pointmap.points.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        k = min(4800, self.pointmap.points.shape[0])
        _, idx = self.pointmap._tree.query(self.state.W_T_R.t, k=k)
        return np.asarray(idx, dtype=np.int64).reshape(-1)

    def _update_query_highlight_geometry(self, idx: np.ndarray):
        """在地图上高亮腿部候选查询的环境点；身体净空独立查询全部身体采样点。"""
        assert self.vis is not None
        points = self.pointmap.points[np.asarray(idx, dtype=np.int64)]
        self._set_cloud_points(self.query_highlight_cloud, points)
        self._show_cloud("query_highlight", self.query_highlight_cloud, QUERY_GREEN,point_size=15.0)

    def _update_feasible_landing_geometry(self, landing_idx: np.ndarray, leg_mask: np.ndarray):
        """按每条腿当前可行 mask 高亮可行落脚点。"""
        assert self.vis is not None
        landing_idx = np.asarray(landing_idx, dtype=np.int64).reshape(-1)
        leg_mask = np.asarray(leg_mask, dtype=bool)
        landing_points = self.pointmap.landing_points[landing_idx]
        for leg_index, cloud in enumerate(self.feasible_landing_clouds):
            feasible = leg_mask[leg_index].any(axis=-1)
            self._set_cloud_points(cloud, landing_points[feasible])
            self._show_cloud(f"feasible_landing_{leg_index}", cloud, LEG_COLORS[leg_index])

    def _clear_feasible_landing_geometry(self):
        """清空每条腿的可行落脚点显示。"""
        assert self.vis is not None
        for leg_index, cloud in enumerate(self.feasible_landing_clouds):
            self._set_cloud_points(cloud, np.zeros((0, 3), dtype=np.float64))
            self._show_cloud(f"feasible_landing_{leg_index}", cloud, LEG_COLORS[leg_index])

    def _register_callbacks(self):
        """注册键盘控制：移动、旋转、确认、保存、删除、导出和退出。"""
        assert self.vis is not None
        self.vis.iren.clear_key_event_callbacks()
        # VTK 的 OnChar 默认把 q/e 用于退出、w/s 用于显示模式；只保留原标注器回调。
        self.vis.iren.style.AddObserver("CharEvent", lambda *_: None)

        def register_key(key, callback):
            self.vis.add_key_event(key, callback)
            if len(key) == 1 and key.isalpha():
                self.vis.add_key_event(key.lower(), callback)

        # 所有运动都用局部坐标系增量，集中在一个映射表中避免写成一批重复函数。
        motion_keys = {
            ord("W"): ("translate", np.array([0.0, 1.0, 0.0])),
            ord("S"): ("translate", np.array([0.0, -1.0, 0.0])),
            ord("A"): ("translate", np.array([-1.0, 0.0, 0.0])),
            ord("D"): ("translate", np.array([1.0, 0.0, 0.0])),
            ord("Q"): ("translate", np.array([0.0, 0.0, 1.0])),
            ord("E"): ("translate", np.array([0.0, 0.0, -1.0])),
            ord("U"): ("rotate", np.array([1.0, 0.0, 0.0])),
            ord("O"): ("rotate", np.array([-1.0, 0.0, 0.0])),
            ord("I"): ("rotate", np.array([0.0, 1.0, 0.0])),
            ord("K"): ("rotate", np.array([0.0, -1.0, 0.0])),
            ord("J"): ("rotate", np.array([0.0, 0.0, 1.0])),
            ord("L"): ("rotate", np.array([0.0, 0.0, -1.0])),
        }
        for key, (kind, axis) in motion_keys.items():
            register_key(
                chr(key),
                lambda kind=kind, axis=axis: self._on_motion(kind, axis),
            )

        register_key("Z", lambda: self._scale_steps(0.5))
        register_key("X", lambda: self._scale_steps(2.0))
        register_key("M", lambda: self._confirm_or_export(export_when_label=False))
        register_key("Return", lambda: self._confirm_or_export(export_when_label=True))  # Enter
        register_key("space", self._on_space)  # Ctrl+Space saves a protected pose.
        register_key("P", lambda: self._save_pose())
        register_key("BackSpace", lambda: self._delete_last_pose())  # Backspace
        register_key("B", lambda: self._delete_last_pose())
        register_key("T", lambda: self._export_trajectory())
        register_key("Escape", lambda: self._close_window())  # Esc

    def _on_motion(self, kind: str, axis: np.ndarray) -> bool:
        """应用一次局部平移或旋转，并刷新几何和有效性颜色。"""
        delta = SE3()
        if kind == "translate":
            t = axis * self.state.translation_step
            delta = SE3.Trans(t)
        else:
            delta = SE3.AngleAxis(self.state.rotation_step, axis)

        #这里的norm只会把SO3部分归一化，确保是满足约束的，不会修改t的值
        self.state.W_T_R = (self.state.W_T_R * delta).norm()
        self._update_pose_geometry()
        self._evaluate_and_render("move")
        return False

    def _update_pose_geometry(self):
        """更新机器人外侧点云，并重建当前坐标轴。"""
        assert self.vis is not None
        self._update_robot_surface_geometry()
        for i, cloud in enumerate(self.leg_surface_clouds):
            self._show_cloud(f"robot_leg_{i}", cloud, LEG_COLORS[i], 20.0, 0.4)
        self._replace_frame_mesh()

    def _replace_frame_mesh(self):
        """重建小坐标系，显示当前机器人 body 坐标方向。"""
        assert self.vis is not None
        self.frame_mesh = self._coordinate_frame(0.25, self.state.W_T_R)
        self.vis.add_mesh(
            self.frame_mesh, name="current_frame", scalars="rgb", rgb=True,
            reset_camera=False, render=False,
        )

    def _evaluate_and_render(self, reason: str):
        """调用 RobotFeasiCheck，按 strong/weak/invalid 给身体着色并打印状态。"""
        assert self.vis is not None
        try:
            query_idx = self._query_robot_feasi_env_indices()
            self._update_query_highlight_geometry(query_idx)
            idx, landing_idx, body_cf_mask, leg_mask = self.hex_state.RobotFeasiCheck(
                self.state.W_T_R,
                points_idx=query_idx,
            )
            self._update_feasible_landing_geometry(landing_idx, leg_mask)
            body_valid = bool(np.asarray(body_cf_mask, dtype=bool).all())
            leg_counts = np.sum(np.asarray(leg_mask).any(axis=-1), axis=1).astype(np.int32)
            weak_valid = bool(body_valid and (leg_counts >= 1).all())
            strong_valid = bool(body_valid and (leg_counts >= 6).all())
            score = -100.0 if not body_valid else float(leg_counts.min() + 0.05 * leg_counts.sum())
            self.state.last_eval = {
                "body_valid": body_valid,
                "weak_valid": weak_valid,
                "strong_valid": strong_valid,
                "leg_counts": leg_counts.astype(int).tolist(),
                "min_leg_count": int(leg_counts.min()) if leg_counts.size else 0,
                "total_leg_count": int(leg_counts.sum()),
                "score": score,
                "near_env_count": int(np.asarray(idx).size),
                "near_landing_count": int(np.asarray(landing_idx).size),
            }
        except Exception as exc:
            self._clear_feasible_landing_geometry()
            self.state.last_eval = {
                "body_valid": False,
                "weak_valid": False,
                "strong_valid": False,
                "leg_counts": [0, 0, 0, 0, 0, 0],
                "min_leg_count": 0,
                "total_leg_count": 0,
                "score": -100.0,
                "error": str(exc),
            }

        # 颜色只表达当前弱/强有效性，详细数字保留在终端输出和导出 JSON 中。
        if self.state.last_eval["strong_valid"]:
            color = GREEN
            valid_text = "strong"
        elif self.state.last_eval["weak_valid"]:
            color = YELLOW
            valid_text = "weak"
        else:
            color = RED
            valid_text = "invalid"
        self._show_cloud("robot_body", self.body_surface_cloud, color, 20.0)
        self.vis.render()
        self._print_status(reason, valid_text)

    def _print_status(self, reason: str, valid_text: str):
        """打印一行紧凑状态，避免把大 mask 刷到终端。"""
        rpy_deg = Rotation.from_matrix(self.state.W_T_R.R).as_euler("xyz", degrees=True)
        pos = self.state.W_T_R.t
        mode = "LABEL" if self.state.label_mode else "INIT"
        eval_info = self.state.last_eval or {}
        print(
            f"[{mode}:{reason}] "
            f"pose: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}, "
            f"rpy=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}), "
            f"valid={valid_text}, legs={eval_info.get('leg_counts')}, "
            f"score={float(eval_info.get('score', -100.0)):.2f}"
        )
        if "error" in eval_info:
            print(f"  evaluate error: {eval_info['error']}")

    def _scale_steps(self, factor: float) -> bool:
        """同步调整平移和旋转步长。"""
        self.state.translation_step = float(np.clip(self.state.translation_step * factor, 0.002, 0.5))
        self.state.rotation_step = float(np.clip(self.state.rotation_step * factor, np.deg2rad(0.5), np.deg2rad(45.0)))
        print(
            "step: "
            f"translation={self.state.translation_step:.3f} m, "
            f"rotation={np.rad2deg(self.state.rotation_step):.1f} deg"
        )
        return False

    def _confirm_or_export(self, export_when_label: bool) -> bool:
        """INIT 模式确认初始位姿；LABEL 模式按需导出当前轨迹。"""
        if not self.state.label_mode:
            self.state.label_mode = True
            self.state.initial_W_T_R = self.state.W_T_R.copy()
            self.state.saved_poses = {"t":[],"ang":[],"vec":[],"W_T_R":[]}
            self.state.protected_pose_indices.clear()
            self._append_current_pose()
            self._update_trajectory_geometry()
            print("Initial pose confirmed. LABEL mode started; saved pose #1.")
            return False
        if export_when_label:
            return self._export_trajectory()
        print("Already in LABEL mode. Use Space/P to save poses, Enter/T to export JSON.")
        return False

    def _on_space(self) -> bool:
        """Read the modifier from VTK; PyVista registers only the space keysym."""
        assert self.vis is not None
        return self._save_pose(protected=bool(self.vis.iren.interactor.GetControlKey()))

    def _save_pose(self, protected: bool = False) -> bool:
        """在 LABEL 模式保存当前位姿和有效性摘要。"""
        if not self.state.label_mode:
            print("Still in INIT mode. Press Enter or M to confirm the initial pose before saving.")
            return False

        self._append_current_pose()
        if protected:
            index = len(self.state.saved_poses["t"])-1
            self.state.protected_pose_indices.append(index)
            print(f"Protected pose index={index} (0-based; preserved during shortcut).")
        self._update_trajectory_geometry()
        print("Saved pose=",len(self.state.saved_poses["t"]) )
        return False

    def _append_current_pose(self):
        """把当前位姿加入已保存轨迹。"""
        # rpy = Rotation.from_matrix(self.state.W_T_R.R).as_euler("xyz", degrees=False)
        #这里把SE3保存为t和AngVec，直接保存4*4矩阵容易出现精度不够，无法满足转移矩阵约束
        self.state.saved_poses["t"].append(self.state.W_T_R.t.tolist())
        ang,vec = self.state.W_T_R.angvec()
        self.state.saved_poses["ang"].append(ang)
        self.state.saved_poses["vec"].append(vec.tolist())
        self.state.saved_poses["W_T_R"].append(self.state.W_T_R)

    def _delete_last_pose(self) -> bool:
        """删除最近保存的一个轨迹点。"""
        if not self.state.saved_poses["t"]:
            print("No saved pose to delete.")
            return False
        index = len(self.state.saved_poses["t"])-1
        if index in self.state.protected_pose_indices:
            self.state.protected_pose_indices.remove(index)
        for key in self.state.saved_poses.keys():
            self.state.saved_poses[key].pop()
        self._update_trajectory_geometry()
        print("Deleted last pose. Remaining saved poses=",len(self.state.saved_poses["t"]))
        return False

    def _update_trajectory_geometry(self):
        """更新保存轨迹坐标系和折线；少于两个点时不显示线段。"""
        assert self.vis is not None
        self._rebuild_saved_frame_meshes()
        positions = np.asarray(self.state.saved_poses["t"],dtype=np.float64)
        if len(positions) < 2:
            if self.trajectory_line_added:
                self.vis.remove_actor("trajectory", reset_camera=False, render=False)
                self.trajectory_line_added = False
            self.vis.render()
            return

        lines = np.asarray(
            [[i, i + 1] for i in range(len(positions) - 1)],
            dtype=np.int32,
        )
        cells = np.column_stack((np.full(len(lines), 2), lines)).ravel()
        self.trajectory_line = pv.PolyData(positions, lines=cells)
        self.vis.add_mesh(
            self.trajectory_line, name="trajectory", color=BLUE, line_width=10.0,
            reset_camera=False, render=False
        )
        self.trajectory_line_added = True
        self.vis.render()

    def _rebuild_saved_frame_meshes(self):
        """用坐标系显示每个已保存位姿。"""
        assert self.vis is not None
        for i in range(len(self.saved_frame_meshes)):
            self.vis.remove_actor(f"saved_frame_{i}", reset_camera=False, render=False)
        self.saved_frame_meshes.clear()
        protected = set(self.state.protected_pose_indices)
        for i, W_T_R in enumerate(self.state.saved_poses["W_T_R"]):
            origin_color = PROTECTED_POSE_ORIGIN_COLOR if i in protected else GRAY
            frame = self._coordinate_frame(0.2, W_T_R, origin_color=origin_color)
            self.vis.add_mesh(
                frame, name=f"saved_frame_{i}", scalars="rgb", rgb=True,
                reset_camera=False, render=False,
            )
            self.saved_frame_meshes.append(frame)

    def _export_trajectory(self) -> bool:
        """导出当前 LABEL 轨迹为 JSON 文件。"""
        if not self.state.label_mode:
            print("Still in INIT mode. Press Enter or M to confirm the initial pose before exporting.")
            return False
        os.makedirs(self.output_dir, exist_ok=True)
        created_at = datetime.now().isoformat(timespec="seconds")
        file_path = os.path.join(
            self.output_dir,
            f"teleop_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        # JSON 只保存紧凑摘要，避免把 RobotFeasiCheck 的大 mask 写进示教数据。
        payload = {
            "metadata": {
                "tool": "keyboard_pose_labeler",
                "created_at": created_at,
                "translation_step": self.state.translation_step,
                "rotation_step_deg": float(np.rad2deg(self.state.rotation_step)),
                "initial_W_T_R": (
                    self.state.initial_W_T_R.A.tolist()
                    if self.state.initial_W_T_R is not None
                    else None
                ),
            },
            "t":self.state.saved_poses["t"],
            "ang":self.state.saved_poses["ang"],
            "vec":self.state.saved_poses["vec"],
            "W_T_R":[W_T_R.A.tolist() for W_T_R in self.state.saved_poses["W_T_R"]],
            # "poses": self.state.saved_poses,
            "protected_pose_indices": list(self.state.protected_pose_indices),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Exported {} poses to {}".format(len(self.state.saved_poses["t"]),file_path))
        return False
    def _close_window(self) -> bool:
        """关闭 PyVista 窗口。"""
        assert self.vis is not None
        self.vis.close()
        return False

    def _print_help(self):
        """打印一次控制说明。"""
        print(
            "\nKeyboard Pose Labeler controls\n"
            "  INIT:  WSAD move in body XY, Q/E move body +Z/-Z, J/L yaw, I/K pitch, U/O roll\n"
            "         Enter or M confirms the initial pose and starts LABEL mode\n"
            "  LABEL: Space or P saves pose, Backspace or B deletes last pose, Enter or T exports JSON\n"
            "         Ctrl+Space saves a pose protected from shortcut (0-based JSON indices)\n"
            "  Both:  Z/X decreases/increases step, Esc closes the window\n"
        )


if __name__ == "__main__":
    hex_state = HexState(Kinematic())
    initial_se3 = SE3.Trans([0.387,2.311,0.090])*SE3.Rz(3.14)
    app = KeyboardPoseLabeler(hex_state,init_pose=initial_se3)
    app.run()
