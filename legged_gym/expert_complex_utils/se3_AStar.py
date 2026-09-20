from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from legged_gym.expert_complex_utils import HexState, Kinematic


GridNode = Tuple[int, int, int, int]


@dataclass
class AStarConfig:
    """Parameters for the SE3 A* planner.

    The planner searches a 4D grid: body position plus yaw around the robot
    body z axis. Roll and pitch are projected from the average normal of the
    feasible landing points near the robot center.
    """

    grid_resolution: Optional[float] = None
    goal_tolerance: Optional[float] = None
    yaw_resolution: Optional[float] = None
    yaw_goal_tolerance: Optional[float] = None
    yaw_weight: float = 0.15
    max_expansions: int = 50000
    time_limit: float = 60.0
    diagonal_motion: bool = True
    coupled_yaw_motion: bool = False
    min_surface_distance: float = 0.1
    max_surface_distance: float = 0.57
    min_landing_points_per_leg: int = 3
    verbose: bool = True


@dataclass
class PoseValidityInfo:
    surface_distance: float
    landing_points_per_leg: np.ndarray
    avg_normal: np.ndarray
    pose: SE3


@dataclass
class AStarResult:
    solved: bool
    path: Optional[list[SE3]]
    reason: str
    expanded: int
    generated: int
    cost: float
    runtime: float


class HexAStarPlanning:
    """A* planner for the hex robot body pose.

    The planner intentionally keeps graph search logic local and reuses the
    existing HexState feasibility checks for robot/environment validity.
    """

    def __init__(self, config: Optional[AStarConfig] = None):
        self.config = AStarConfig() if config is None else config
        self.hex_state = HexState(Kinematic())
        self.env_map = self.hex_state.env_pointsmap_voxels
        self.bounds = np.asarray(self.env_map._bounds, dtype=np.float32).copy()

        if self.config.grid_resolution is None:
            self.grid_resolution = float(self.env_map.voxels.voxel_scale)
        else:
            self.grid_resolution = float(self.config.grid_resolution)
        if self.grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive")

        if self.config.goal_tolerance is None:
            self.goal_tolerance = self.grid_resolution
        else:
            self.goal_tolerance = float(self.config.goal_tolerance)
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance must be non-negative")

        if self.config.yaw_resolution is None:
            requested_yaw_resolution = np.deg2rad(30.0)
        else:
            requested_yaw_resolution = float(self.config.yaw_resolution)
        if requested_yaw_resolution <= 0.0:
            raise ValueError("yaw_resolution must be positive")
        self.yaw_bins = max(1, int(np.ceil(2.0 * np.pi / requested_yaw_resolution)))
        self.yaw_resolution = float(2.0 * np.pi / self.yaw_bins)

        if self.config.yaw_goal_tolerance is None:
            self.yaw_goal_tolerance = 0.5 * self.yaw_resolution
        else:
            self.yaw_goal_tolerance = float(self.config.yaw_goal_tolerance)
        if self.yaw_goal_tolerance < 0.0:
            raise ValueError("yaw_goal_tolerance must be non-negative")
        if self.config.yaw_weight < 0.0:
            raise ValueError("yaw_weight must be non-negative")

        self.grid_shape = np.ceil(
            (self.bounds[:, 1] - self.bounds[:, 0]) / self.grid_resolution
        ).astype(np.int32)
        if (self.grid_shape <= 0).any():
            raise ValueError("environment bounds do not define a positive grid")

        self._neighbor_offsets = self._build_neighbor_offsets(
            diagonal_motion=self.config.diagonal_motion,
            coupled_yaw_motion=self.config.coupled_yaw_motion,
        )
        self._node_feasi_cache: Dict[GridNode, Tuple[bool, Optional[PoseValidityInfo]]] = {}
        self.last_result: Optional[AStarResult] = None

    def AStar(self, start_se3: SE3, goal_se3: SE3) -> Optional[list[SE3]]:
        """Convenience wrapper returning only the SE3 path."""
        result = self.Search(start_se3, goal_se3)
        return result.path

    def Search(self, start_se3: SE3, goal_se3: SE3) -> AStarResult:
        """Run A* and return the path plus planner statistics."""
        start_time = time.monotonic()
        start_node = self._pose_to_node(start_se3)
        goal_node = self._pose_to_node(goal_se3)
        goal_yaw = self._pose_yaw(goal_se3)

        start_valid, start_info = self._check_pose_validity(start_se3)
        if not start_valid:
            return self._finish(False, None, "start pose is invalid", 0, 0, np.inf, start_time)

        goal_valid, goal_info = self._check_pose_validity(goal_se3)
        if not goal_valid:
            return self._finish(False, None, "goal pose is invalid", 0, 0, np.inf, start_time)

        self._node_feasi_cache[start_node] = (start_valid, start_info)
        self._node_feasi_cache[goal_node] = (goal_valid, goal_info)

        if not self._is_valid_node(start_node):
            return self._finish(False, None, "start grid node is invalid", 0, 0, np.inf, start_time)

        open_heap: list[tuple[float, float, int, GridNode]] = []
        counter = itertools.count()
        start_h = self._heuristic(start_node, goal_node)
        heapq.heappush(open_heap, (start_h, start_h, next(counter), start_node))

        came_from: Dict[GridNode, Optional[GridNode]] = {start_node: None}
        g_score: Dict[GridNode, float] = {start_node: 0.0}
        closed: set[GridNode] = set()
        expanded = 0
        generated = 1

        while open_heap:
            if time.monotonic() - start_time > self.config.time_limit:
                return self._finish(
                    False,
                    None,
                    "time limit reached",
                    expanded,
                    generated,
                    np.inf,
                    start_time,
                )

            _, _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expanded += 1

            if self._reached_goal(current, goal_node, goal_se3, goal_yaw):
                nodes = self._reconstruct_nodes(came_from, current)
                path = self._nodes_to_path(nodes, start_se3, goal_se3)
                if path is None:
                    return self._finish(
                        False,
                        None,
                        "found grid path, but path reconstruction failed",
                        expanded,
                        generated,
                        np.inf,
                        start_time,
                    )
                return self._finish(
                    True,
                    path,
                    "path found",
                    expanded,
                    generated,
                    g_score[current],
                    start_time,
                )

            if expanded >= self.config.max_expansions:
                return self._finish(
                    False,
                    None,
                    "max expansions reached",
                    expanded,
                    generated,
                    np.inf,
                    start_time,
                )

            for neighbor in self._iter_neighbors(current):
                if neighbor in closed:
                    continue

                tentative_g = g_score[current] + self._transition_cost(current, neighbor)
                if tentative_g >= g_score.get(neighbor, np.inf):
                    continue
                if not self._is_valid_node(neighbor):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = self._heuristic(neighbor, goal_node)
                heapq.heappush(open_heap, (tentative_g + h, h, next(counter), neighbor))
                generated += 1

        return self._finish(
            False,
            None,
            "open set exhausted",
            expanded,
            generated,
            np.inf,
            start_time,
        )

    def _finish(
        self,
        solved: bool,
        path: Optional[list[SE3]],
        reason: str,
        expanded: int,
        generated: int,
        cost: float,
        start_time: float,
    ) -> AStarResult:
        result = AStarResult(
            solved=solved,
            path=path,
            reason=reason,
            expanded=expanded,
            generated=generated,
            cost=float(cost),
            runtime=time.monotonic() - start_time,
        )
        self.last_result = result
        if self.config.verbose:
            print(
                f"A* {reason}: solved={solved}, expanded={expanded}, "
                f"generated={generated}, cost={cost:.3f}, runtime={result.runtime:.3f}s"
            )
        return result

    def _check_pose_validity(self, W_T_R: SE3) -> tuple[bool, Optional[PoseValidityInfo]]:
        valid, surface_distance, landing_counts, _, _ = self._evaluate_pose(W_T_R)
        normal = self._normalize(np.asarray(W_T_R.R[:, 2], dtype=np.float64))
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return valid, PoseValidityInfo(surface_distance, landing_counts, normal, W_T_R)

    def _evaluate_pose(
        self,
        W_T_R: SE3,
    ) -> tuple[bool, float, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        pos = np.asarray(W_T_R.t, dtype=np.float32)
        empty_counts = np.zeros(6, dtype=np.int32)
        if not self.env_map.voxels.IsInsideRange(pos):
            return False, np.inf, empty_counts, None, None

        _, idx = self.env_map._tree.query(pos)
        env_point = self.env_map.points[idx]
        surface_distance = float(np.linalg.norm(env_point - pos))
        if (
            surface_distance <= self.config.min_surface_distance
            or surface_distance >= self.config.max_surface_distance
        ):
            return False, surface_distance, empty_counts, None, None

        _, landing_idx, body_cf_mask, leg_feasi_mask = self.hex_state.RobotFeasiCheck(W_T_R)
        if (~body_cf_mask).any():
            return False, surface_distance, empty_counts, None, None

        landing_counts = np.sum(leg_feasi_mask.any(axis=-1), axis=1).astype(np.int32)
        valid = bool((landing_counts >= self.config.min_landing_points_per_leg).all())
        return valid, surface_distance, landing_counts, landing_idx, leg_feasi_mask

    def _is_valid_node(self, node: GridNode) -> bool:
        return self._node_validity(node)[0]

    def _node_validity(self, node: GridNode) -> tuple[bool, Optional[PoseValidityInfo]]:
        cached = self._node_feasi_cache.get(node)
        if cached is None:
            cached = self._project_pose_from_position_yaw(
                self._node_center(node),
                self._node_yaw(node),
            )
            self._node_feasi_cache[node] = cached
        return cached

    def _project_pose_from_position_yaw(
        self,
        pos: np.ndarray,
        yaw: float,
    ) -> tuple[bool, Optional[PoseValidityInfo]]:
        pos = np.asarray(pos, dtype=np.float32)
        seed_normal = self._surface_normal_at_position(pos)
        seed_pose = SE3.Rt(self._frame_from_normal_yaw(seed_normal, yaw), pos)
        seed_valid, seed_distance, seed_counts, landing_idx, leg_feasi_mask = self._evaluate_pose(seed_pose)
        if landing_idx is None or leg_feasi_mask is None:
            return False, PoseValidityInfo(seed_distance, seed_counts, seed_normal, seed_pose)

        # The body z axis is projected once from the seed pose's feasible landing normals.
        avg_normal = self._average_feasible_landing_normal(
            landing_idx,
            leg_feasi_mask,
            seed_normal,
        )
        if avg_normal is None:
            return False, PoseValidityInfo(seed_distance, seed_counts, seed_normal, seed_pose)
        if np.dot(seed_normal, avg_normal) > 1.0 - 1e-8:
            return seed_valid, PoseValidityInfo(seed_distance, seed_counts, avg_normal, seed_pose)

        pose = SE3.Rt(self._frame_from_normal_yaw(avg_normal, yaw), pos)
        valid, surface_distance, landing_counts, _, _ = self._evaluate_pose(pose)
        return valid, PoseValidityInfo(surface_distance, landing_counts, avg_normal, pose)

    def _nodes_to_path(
        self,
        nodes: list[GridNode],
        start_se3: SE3,
        goal_se3: SE3,
    ) -> Optional[list[SE3]]:
        path: list[SE3] = []
        for node in nodes:
            valid, info = self._node_validity(node)
            if not valid or info is None:
                return None
            path.append(info.pose)

        if len(path) == 0:
            return None

        if not self._poses_close(start_se3, path[0]):
            path.insert(0, start_se3)
        else:
            path[0] = start_se3

        if not self._poses_close(path[-1], goal_se3):
            path.append(goal_se3)
        else:
            path[-1] = goal_se3

        return path

    def _reconstruct_nodes(
        self,
        came_from: Dict[GridNode, Optional[GridNode]],
        current: GridNode,
    ) -> list[GridNode]:
        nodes = [current]
        while came_from[current] is not None:
            current = came_from[current]
            nodes.append(current)
        nodes.reverse()
        return nodes

    def _pose_to_node(self, pose: SE3) -> GridNode:
        pos_node = self._pos_to_node(pose.t)
        return (*pos_node, self._yaw_to_idx(self._pose_yaw(pose)))

    def _pos_to_node(self, pos: np.ndarray) -> Tuple[int, int, int]:
        pos = np.asarray(pos, dtype=np.float32)
        node = np.floor((pos - self.bounds[:, 0]) / self.grid_resolution).astype(np.int32)
        node = np.clip(node, 0, self.grid_shape - 1)
        return tuple(int(v) for v in node)

    def _node_center(self, node: GridNode) -> np.ndarray:
        return self.bounds[:, 0] + (np.asarray(node[:3], dtype=np.float32) + 0.5) * self.grid_resolution

    def _node_yaw(self, node: GridNode) -> float:
        return self._idx_to_yaw(node[3])

    def _iter_neighbors(self, node: GridNode) -> Iterable[GridNode]:
        base_pos = np.asarray(node[:3], dtype=np.int32)
        yaw_idx = int(node[3])
        yielded: set[GridNode] = set()
        for offset in self._neighbor_offsets:
            neighbor_pos = base_pos + offset[:3]
            if not ((neighbor_pos >= 0) & (neighbor_pos < self.grid_shape)).all():
                continue
            neighbor = (
                int(neighbor_pos[0]),
                int(neighbor_pos[1]),
                int(neighbor_pos[2]),
                int((yaw_idx + int(offset[3])) % self.yaw_bins),
            )
            if neighbor == node or neighbor in yielded:
                continue
            yielded.add(neighbor)
            yield neighbor

    def _build_neighbor_offsets(
        self,
        diagonal_motion: bool,
        coupled_yaw_motion: bool,
    ) -> list[np.ndarray]:
        position_offsets = []
        if diagonal_motion:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        position_offsets.append(np.array([dx, dy, dz], dtype=np.int32))
        else:
            position_offsets = [
                np.array([1, 0, 0], dtype=np.int32),
                np.array([-1, 0, 0], dtype=np.int32),
                np.array([0, 1, 0], dtype=np.int32),
                np.array([0, -1, 0], dtype=np.int32),
                np.array([0, 0, 1], dtype=np.int32),
                np.array([0, 0, -1], dtype=np.int32),
            ]

        offsets = []
        yaw_offsets = (-1, 0, 1) if coupled_yaw_motion else (0,)
        for pos_offset in position_offsets:
            for dyaw in yaw_offsets:
                offsets.append(
                    np.array([pos_offset[0], pos_offset[1], pos_offset[2], dyaw], dtype=np.int32)
                )
        offsets.extend(
            [
                np.array([0, 0, 0, -1], dtype=np.int32),
                np.array([0, 0, 0, 1], dtype=np.int32),
            ]
        )
        offsets.sort(
            key=lambda item: float(
                np.linalg.norm(item[:3]) * self.grid_resolution
                + abs(int(item[3])) * self.config.yaw_weight * self.yaw_resolution
            )
        )
        return offsets

    def _transition_cost(self, from_node: GridNode, to_node: GridNode) -> float:
        pos_delta = np.asarray(to_node[:3], dtype=np.float32) - np.asarray(
            from_node[:3],
            dtype=np.float32,
        )
        yaw_delta = abs(self._shortest_yaw_delta(self._node_yaw(from_node), self._node_yaw(to_node)))
        return float(
            np.linalg.norm(pos_delta) * self.grid_resolution
            + self.config.yaw_weight * yaw_delta
        )

    def _heuristic(self, node: GridNode, goal_node: GridNode) -> float:
        pos_delta = np.asarray(node[:3], dtype=np.float32) - np.asarray(
            goal_node[:3],
            dtype=np.float32,
        )
        pos_dist = float(np.linalg.norm(pos_delta) * self.grid_resolution)
        yaw_dist = abs(self._shortest_yaw_delta(self._node_yaw(node), self._node_yaw(goal_node)))
        return float(pos_dist + self.config.yaw_weight * yaw_dist)

    def _reached_goal(
        self,
        node: GridNode,
        goal_node: GridNode,
        goal_se3: SE3,
        goal_yaw: float,
    ) -> bool:
        if node == goal_node:
            return True
        pos_reached = self._heuristic_position_only(node, goal_se3) <= self.goal_tolerance
        yaw_reached = abs(self._shortest_yaw_delta(self._node_yaw(node), goal_yaw)) <= self.yaw_goal_tolerance
        return pos_reached and yaw_reached

    def _heuristic_position_only(self, node: GridNode, goal_se3: SE3) -> float:
        return float(np.linalg.norm(self._node_center(node) - np.asarray(goal_se3.t)))

    def _surface_normal_at_position(self, pos: np.ndarray) -> np.ndarray:
        _, idx = self.env_map._tree.query(np.asarray(pos, dtype=np.float32))
        normal = self._normalize(self.env_map.normals[idx])
        if normal is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        to_pos = np.asarray(pos, dtype=np.float64) - np.asarray(self.env_map.points[idx], dtype=np.float64)
        if np.dot(normal, to_pos) < 0.0:
            normal = -normal
        return normal

    def _average_feasible_landing_normal(
        self,
        landing_idx: np.ndarray,
        leg_feasi_mask: np.ndarray,
        reference_normal: np.ndarray,
    ) -> Optional[np.ndarray]:
        feasible_point_mask = leg_feasi_mask.any(axis=(0, 2))
        if not feasible_point_mask.any():
            return None
        normals = np.asarray(self.env_map.normals[landing_idx[feasible_point_mask]], dtype=np.float64)
        avg_normal = self._normalize(normals.mean(axis=0))
        if avg_normal is None:
            return None
        if np.dot(avg_normal, reference_normal) < 0.0:
            avg_normal = -avg_normal
        return avg_normal

    def _frame_from_normal_yaw(self, normal: np.ndarray, yaw: float) -> np.ndarray:
        z_axis = self._normalize(normal)
        if z_axis is None:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_ref, y_ref = self._tangent_reference_axes(z_axis)
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x_axis = c * x_ref + s * y_ref
        y_axis = -s * x_ref + c * y_ref
        return np.column_stack([x_axis, y_axis, z_axis])

    def _pose_yaw(self, pose: SE3) -> float:
        R = np.asarray(pose.R, dtype=np.float64)
        z_axis = self._normalize(R[:, 2])
        if z_axis is None:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_ref, y_ref = self._tangent_reference_axes(z_axis)
        x_axis = R[:, 0] - np.dot(R[:, 0], z_axis) * z_axis
        x_axis = self._normalize(x_axis)
        if x_axis is None:
            x_axis = x_ref
        return self._wrap_angle(float(np.arctan2(np.dot(x_axis, y_ref), np.dot(x_axis, x_ref))))

    def _tangent_reference_axes(self, z_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, z_axis))) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = ref - np.dot(ref, z_axis) * z_axis
        x_axis = self._normalize(x_axis)
        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return x_axis, y_axis

    def _yaw_to_idx(self, yaw: float) -> int:
        yaw_positive = self._wrap_angle(yaw) % (2.0 * np.pi)
        return int(np.round(yaw_positive / self.yaw_resolution)) % self.yaw_bins

    def _idx_to_yaw(self, yaw_idx: int) -> float:
        return self._wrap_angle(float(int(yaw_idx) % self.yaw_bins) * self.yaw_resolution)

    @staticmethod
    def _shortest_yaw_delta(start_yaw: float, goal_yaw: float) -> float:
        return HexAStarPlanning._wrap_angle(goal_yaw - start_yaw)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        v = np.asarray(v, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if (not np.isfinite(v).all()) or (not np.isfinite(norm)) or norm < eps:
            return None
        return v / norm

    @staticmethod
    def _poses_close(a: SE3, b: SE3, pos_tol: float = 1e-6, rot_tol: float = 1e-6) -> bool:
        if np.linalg.norm(np.asarray(a.t) - np.asarray(b.t)) > pos_tol:
            return False
        return Rotation.from_matrix(a.R.T @ b.R).magnitude() <= rot_tol


if __name__ == "__main__":
    planner = HexAStarPlanning()
    path = planner.AStar(SE3(1.38, 4.2, 0.15), SE3(0.37, 0.7, 0.9))
    # path = planner.AStar(SE3(1.3, 4.2, 0.15), SE3(1.3, 3.2, 0.15))
    if path is None:
        print("A* failed")
    else:
        print(f"A* path length: {len(path)}")
