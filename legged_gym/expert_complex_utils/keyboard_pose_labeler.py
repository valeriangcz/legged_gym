from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import HexState, Kinematic


GREEN = [0.1, 0.8, 0.1]
YELLOW = [0.9, 0.7, 0.1]
RED = [0.9, 0.1, 0.1]
GRAY = [0.5, 0.5, 0.5]
BLUE = [0.1, 0.25, 0.9]
QUERY_ORANGE = [1.0, 0.35, 0.05]
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
    saved_poses = {"t":[],"ang":[],"vec":[],"W_T_R":[]}
    last_eval: dict | None = None
    label_mode: bool = False
    initial_W_T_R: SE3 | None = None


class KeyboardPoseLabeler:
    """Open3D 键盘遥控位姿标注器。

    启动后先进入 INIT 模式，用户只移动机器人并观察有效性颜色；
    确认初始位姿后进入 LABEL 模式，才允许保存和导出轨迹。
    """

    def __init__(
        self,
        hex_state: HexState,
        init_pose: SE3 | None = None,
        env_stride: int = 4,
        landing_stride: int = 2,
        output_dir: str | None = None,
    ):
        """绑定 HexState 和可视化参数，并提前创建 Open3D 几何对象。"""
        self.hex_state = hex_state
        self.pointmap = hex_state.env_pointsmap_voxels
        self.state = PoseTeleopState()
        if init_pose is not None:
            # self.state.W_T_R = np.asarray(init_pose, dtype=np.float64).copy()
            self.state.W_T_R = init_pose.copy()

        self.env_stride = max(1, int(env_stride))
        self.landing_stride = max(1, int(landing_stride))
        self.output_dir = output_dir or os.path.join(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path")

        self.vis: o3d.visualization.VisualizerWithKeyCallback | None = None
        self.body_surface_points_R = self._body_surface_points_R()
        self.leg_surface_points_R = self._leg_surface_points_R()
        self.body_surface_cloud = self._make_point_cloud(np.zeros((0, 3)), GRAY)
        self.leg_surface_clouds = [
            self._make_point_cloud(np.zeros((0, 3)), LEG_COLORS[i])
            for i in range(6)
        ]
        self.frame_mesh: o3d.geometry.TriangleMesh | None = None
        self.saved_frame_meshes: list[o3d.geometry.TriangleMesh] = []
        self.env_cloud = self._make_point_cloud(
            self.pointmap.points[:: self.env_stride],
            [0.72, 0.72, 0.72],
        )
        self.query_highlight_cloud = self._make_point_cloud(
            np.zeros((0, 3)),
            QUERY_ORANGE,
        )
        self.feasible_landing_clouds = [
            self._make_point_cloud(np.zeros((0, 3)), LEG_COLORS[i])
            for i in range(6)
        ]
        self.landing_cloud = self._make_point_cloud(
            self.pointmap.landing_points[:: self.landing_stride],
            [0.02, 0.02, 0.02],
        )
        self.trajectory_line = o3d.geometry.LineSet()
        self.trajectory_line_added = False

    def run(self):
        """创建窗口、注册按键回调，并进入 Open3D 事件循环。"""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Keyboard Pose Labeler", width=1280, height=820)

        # 静态点云只添加一次；动态几何在按键回调中原地更新。
        self.vis.add_geometry(self.env_cloud)
        self.vis.add_geometry(self.landing_cloud)
        self.vis.add_geometry(self.query_highlight_cloud)
        for cloud in self.feasible_landing_clouds:
            self.vis.add_geometry(cloud)
        self._update_robot_surface_geometry()
        self.vis.add_geometry(self.body_surface_cloud)
        for cloud in self.leg_surface_clouds:
            self.vis.add_geometry(cloud)
        self._replace_frame_mesh()

        render_option = self.vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])
        render_option.point_size = 5.0
        render_option.line_width = 3.0

        self._register_callbacks()
        self._print_help()
        self._evaluate_and_render("init")
        self.vis.run()
        self.vis.destroy_window()

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
        within_mask = np.linalg.norm(robot_voxels.leg_voxels.center,axis=1)<=0.3
        #grid_size,6
        # inside_mask = robot_voxels.to_bound_dist_flat >= 0.0
        inside_mask = robot_voxels.to_bound_dist_flat >= 0.042
        leg_points_R = []
        for leg_index in range(6):
            reachable = robot_voxels.robot_reachable_legs[..., leg_index].any(axis=1)
            reachable = reachable&within_mask&inside_mask[:,leg_index]

            surface_flat = self._surface_flat_indices(leg_voxels, reachable)
            points_B = leg_voxels.center[surface_flat]
            if points_B.shape[0] == 0:
                leg_points_R.append(np.zeros((0, 3), dtype=np.float64))
                continue
            points_R = robot_voxels.kin._B2R(points_B.T, leg_index).T
            leg_points_R.append(np.asarray(points_R, dtype=np.float64))
        return leg_points_R

    def _make_point_cloud(self, points: np.ndarray, color: list[float]) -> o3d.geometry.PointCloud:
        """把 numpy 点集转换成带统一颜色的 Open3D 点云。"""
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        cloud.paint_uniform_color(color)
        return cloud

    def _transform_points_R_to_W(self, points_R: np.ndarray) -> np.ndarray:
        """把 R 系点云批量转换到 W 系。"""
        if points_R.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray((self.state.W_T_R * points_R.T).T, dtype=np.float64)

    def _update_robot_surface_geometry(self):
        """按当前身体位姿更新身体和腿部外侧点云。"""
        self.body_surface_cloud.points = o3d.utility.Vector3dVector(
            self._transform_points_R_to_W(self.body_surface_points_R)
        )
        for leg_index,(cloud,points_R) in enumerate(zip(self.leg_surface_clouds,self.leg_surface_points_R)):
            cloud.points = o3d.utility.Vector3dVector(self._transform_points_R_to_W(points_R))
            cloud.paint_uniform_color(LEG_COLORS[leg_index])


    def _query_robot_feasi_env_indices(self) -> np.ndarray:
        """查询 RobotFeasiCheck 使用的身体附近环境点索引。"""
        if self.pointmap.points.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        k = min(2400, self.pointmap.points.shape[0])
        _, idx = self.pointmap._tree.query(self.state.W_T_R.t, k=k)
        return np.asarray(idx, dtype=np.int64).reshape(-1)

    def _update_query_highlight_geometry(self, idx: np.ndarray):
        """在地图上高亮当前参与 RobotFeasiCheck 的环境点。"""
        assert self.vis is not None
        points = self.pointmap.points[np.asarray(idx, dtype=np.int64)]
        self.query_highlight_cloud.points = o3d.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64)
        )
        self.query_highlight_cloud.paint_uniform_color(QUERY_ORANGE)
        self.vis.update_geometry(self.query_highlight_cloud)

    def _update_feasible_landing_geometry(self, landing_idx: np.ndarray, leg_mask: np.ndarray):
        """按每条腿当前可行 mask 高亮可行落脚点。"""
        assert self.vis is not None
        landing_idx = np.asarray(landing_idx, dtype=np.int64).reshape(-1)
        leg_mask = np.asarray(leg_mask, dtype=bool)
        landing_points = self.pointmap.landing_points[landing_idx]
        for leg_index, cloud in enumerate(self.feasible_landing_clouds):
            feasible = leg_mask[leg_index].any(axis=-1)
            cloud.points = o3d.utility.Vector3dVector(
                np.asarray(landing_points[feasible], dtype=np.float64)
            )
            cloud.paint_uniform_color(LEG_COLORS[leg_index])
            self.vis.update_geometry(cloud)

    def _clear_feasible_landing_geometry(self):
        """清空每条腿的可行落脚点显示。"""
        assert self.vis is not None
        for cloud in self.feasible_landing_clouds:
            cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.vis.update_geometry(cloud)

    def _register_callbacks(self):
        """注册键盘控制：移动、旋转、确认、保存、删除、导出和退出。"""
        assert self.vis is not None

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
            self.vis.register_key_callback(
                key,
                lambda vis, kind=kind, axis=axis: self._on_motion(kind, axis),
            )

        self.vis.register_key_callback(ord("Z"), lambda vis: self._scale_steps(0.5))
        self.vis.register_key_callback(ord("X"), lambda vis: self._scale_steps(2.0))
        self.vis.register_key_callback(ord("M"), lambda vis: self._confirm_or_export(export_when_label=False))
        self.vis.register_key_callback(257, lambda vis: self._confirm_or_export(export_when_label=True))  # Enter
        self.vis.register_key_callback(32, lambda vis: self._save_pose())  # Space
        self.vis.register_key_callback(ord("P"), lambda vis: self._save_pose())
        self.vis.register_key_callback(259, lambda vis: self._delete_last_pose())  # Backspace
        self.vis.register_key_callback(ord("B"), lambda vis: self._delete_last_pose())
        self.vis.register_key_callback(ord("T"), lambda vis: self._export_trajectory())
        self.vis.register_key_callback(256, lambda vis: self._close_window())  # Esc

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
        self.vis.update_geometry(self.body_surface_cloud)
        for cloud in self.leg_surface_clouds:
            self.vis.update_geometry(cloud)
        self._replace_frame_mesh()

    def _replace_frame_mesh(self):
        """重建小坐标系，显示当前机器人 body 坐标方向。"""
        assert self.vis is not None
        if self.frame_mesh is not None:
            self.vis.remove_geometry(self.frame_mesh, reset_bounding_box=False)
        self.frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        self.frame_mesh.transform(self.state.W_T_R.A)
        self.vis.add_geometry(self.frame_mesh, reset_bounding_box=False)

    def _evaluate_and_render(self, reason: str):
        """调用 RobotFeasiCheck，按 strong/weak/invalid 给身体着色并打印状态。"""
        assert self.vis is not None
        try:
            query_idx = self._query_robot_feasi_env_indices()
            # self._update_query_highlight_geometry(query_idx)
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
        self.body_surface_cloud.paint_uniform_color(color)
        self.vis.update_geometry(self.body_surface_cloud)
        # self.vis.poll_events()
        self.vis.update_renderer()
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
            self._append_current_pose()
            self._update_trajectory_geometry()
            print("Initial pose confirmed. LABEL mode started; saved pose #1.")
            return False
        if export_when_label:
            return self._export_trajectory()
        print("Already in LABEL mode. Use Space/P to save poses, Enter/T to export JSON.")
        return False

    def _save_pose(self) -> bool:
        """在 LABEL 模式保存当前位姿和有效性摘要。"""
        if not self.state.label_mode:
            print("Still in INIT mode. Press Enter or M to confirm the initial pose before saving.")
            return False

        self._append_current_pose()
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
        if not self.state.saved_poses:
            print("No saved pose to delete.")
            return False
        for key in self.state.saved_poses.keys():
            self.state.saved_poses[key].pop()
        self._update_trajectory_geometry()
        print("Deleted last pose. Remaining saved poses=",len(self.state.saved_poses["t"]))
        return False

    def _update_trajectory_geometry(self):
        """更新保存轨迹坐标系和 LineSet；少于两个点时不显示线段。"""
        assert self.vis is not None
        self._rebuild_saved_frame_meshes()
        positions = np.asarray(self.state.saved_poses["t"],dtype=np.float64)
        if len(positions) < 2:
            if self.trajectory_line_added:
                self.vis.remove_geometry(self.trajectory_line, reset_bounding_box=False)
                self.trajectory_line_added = False
            # self.vis.poll_events()
            self.vis.update_renderer()
            return

        lines = np.asarray(
            [[i, i + 1] for i in range(len(positions) - 1)],
            dtype=np.int32,
        )
        colors = np.tile(np.asarray(BLUE, dtype=np.float64), (len(lines), 1))
        self.trajectory_line.points = o3d.utility.Vector3dVector(positions)
        self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)
        self.trajectory_line.colors = o3d.utility.Vector3dVector(colors)
        if not self.trajectory_line_added:
            self.vis.add_geometry(self.trajectory_line, reset_bounding_box=False)
            self.trajectory_line_added = True
        self.vis.update_geometry(self.trajectory_line)
        # self.vis.poll_events()
        self.vis.update_renderer()

    def _rebuild_saved_frame_meshes(self):
        """用坐标系显示每个已保存位姿。"""
        assert self.vis is not None
        for mesh in self.saved_frame_meshes:
            self.vis.remove_geometry(mesh, reset_bounding_box=False)
        self.saved_frame_meshes.clear()
        for W_T_R in self.state.saved_poses["W_T_R"]:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.16)
            frame.transform(np.asarray(W_T_R.A, dtype=np.float64))
            self.vis.add_geometry(frame, reset_bounding_box=False)
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
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Exported {} poses to {}".format(len(self.state.saved_poses["t"]),file_path))
        return False
    def _close_window(self) -> bool:
        """关闭 Open3D 窗口。"""
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
            "  Both:  Z/X decreases/increases step, Esc closes the window\n"
        )


if __name__ == "__main__":
    hex_state = HexState(Kinematic())
    initial_se3 = SE3.Trans([1.2,4.2,0.24])*SE3.Rz(3.14)
    app = KeyboardPoseLabeler(hex_state,init_pose=initial_se3)
    app.run()
