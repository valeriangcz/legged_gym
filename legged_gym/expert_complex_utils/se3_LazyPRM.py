from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from spatialmath import SE3

from legged_gym.expert_complex_utils import HexState, Kinematic


UNKNOWN = 0
VALID = 1
INVALID = -1
MODE_NAMES = ("raw", "surface", "edge")


@dataclass
class LazyPRMConfig:
    sample_count: int = 3000
    max_sample_attempts: int = 200
    min_surface_distance: float = 0.1
    max_surface_distance: float = 0.57
    min_landing_points_per_leg: int = 3
    node_merge_radius: float = 0.04
    k_neighbors: int = 16
    connection_radius: float = 0.35
    query_neighbors: int = 32
    edge_check_resolution: float | None = None
    rot_weight: float = 0.15
    edge_probability: float = 0.8
    seed: int | None = None
    progress_interval: int = 10


def _Normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if (not np.isfinite(v).all()) or (not np.isfinite(n)) or n < eps:
        return None
    return v / n


def _FrameFromZ(z_axis, yaw):
    z_axis = _Normalize(z_axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(z_axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = _Normalize(np.cross(ref, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis]) @ Rotation.from_euler("z", yaw).as_matrix()
    return Rotation.from_matrix(R).as_matrix()


def _SE3Distance(a: SE3, b: SE3, rot_weight: float) -> float:
    return float(np.linalg.norm(a.t - b.t) + rot_weight * Rotation.from_matrix(a.R.T @ b.R).magnitude())


def _InterpolateSE3(a: SE3, b: SE3, alpha: float) -> SE3:
    pos = (1.0 - alpha) * a.t + alpha * b.t
    rots = Rotation.from_matrix(np.stack([a.R, b.R], axis=0))
    R = Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]
    return SE3.Rt(R, pos)


class HexLazyPRMPlanning:
    """LazyPRM：先高效构建可行SE3节点图，查询时再懒检查边。"""

    def __init__(self, config: LazyPRMConfig | None = None, load_if_exists: bool = False):
        self.config = LazyPRMConfig() if config is None else config
        self.hex_state = HexState(Kinematic())
        self.pointmap = self.hex_state.env_pointsmap_voxels
        root, _ = os.path.splitext(getattr(self.pointmap, "saved_file", self.pointmap.target_file))
        self.roadmap_file = root + "_lazy_prm.npz"
        self.nodes: list[dict] = []
        self.edges: dict[tuple[int, int], int] = {}
        self.last_path: list[SE3] | None = None
        if load_if_exists and os.path.exists(self.roadmap_file):
            self.LoadRoadmap()

    def _PoseValid(self, W_T_R: SE3):
        dist, _ = self.pointmap._tree.query(W_T_R.t)
        if dist < self.config.min_surface_distance or dist > self.config.max_surface_distance:
            return False, np.zeros(6, dtype=np.int32)
        _, _, body_cf_mask, leg_mask = self.hex_state.RobotFeasiCheck(W_T_R)
        if (~body_cf_mask).any():
            return False, np.zeros(6, dtype=np.int32)
        counts = np.sum(leg_mask.any(axis=-1), axis=1).astype(np.int32)
        return bool((counts >= self.config.min_landing_points_per_leg).all()), counts

    def _SampleNode(self, rng: np.random.Generator):
        """采样逻辑：太近直接丢，合法距离只验raw，太远才投影到面/棱。"""
        # z_offsets = (0.14, 0.18, 0.22, 0.26, 0.34, 0.46)
        z_offsets = (0.14, 0.18, 0.22, 0.26)
        yaw_values = np.deg2rad((0, 60, 120, 180, 240, 300))

        for _ in range(self.config.max_sample_attempts):
            pos = rng.uniform(self.pointmap._bounds[:, 0], self.pointmap._bounds[:, 1])
            raw = SE3.Rt(Rotation.random(random_state=rng).as_matrix(), pos)
            dist, idx = self.pointmap._tree.query(pos)

            if dist < self.config.min_surface_distance:
                continue
            if dist <= self.config.max_surface_distance:
                valid, counts = self._PoseValid(raw)
                if valid:
                    return {"pose": raw, "counts": counts, "mode": 0, "proj_dist": 0.0}
                continue

            modes = (2, 1) if rng.random() < self.config.edge_probability else (1, 2)
            best = None
            for mode in modes:
                if mode == 2:
                    if getattr(self.pointmap, "_edge_tree", None) is None or self.pointmap.edge_points.shape[0] == 0:
                        continue
                    _, edge_idx = self.pointmap._edge_tree.query(pos)
                    anchor = self.pointmap.edge_points[edge_idx]
                    z_axis = _Normalize(pos - anchor)
                else:
                    anchor = self.pointmap.points[idx]
                    z_axis = _Normalize(self.pointmap.normals[idx])
                    if z_axis is not None and np.dot(z_axis, pos - anchor) < 0.0:
                        z_axis = -z_axis
                if z_axis is None:
                    continue

                for z_offset in z_offsets:
                    center = anchor + z_offset * z_axis
                    for yaw in yaw_values:
                        pose = SE3.Rt(_FrameFromZ(z_axis, yaw), center)
                        valid, counts = self._PoseValid(pose)
                        if not valid:
                            continue
                        item = (
                            int(counts.min()),
                            int(counts.sum()),
                            -float(np.linalg.norm(center - pos)),
                            {"pose": pose, "counts": counts, "mode": mode, "proj_dist": float(np.linalg.norm(center - pos))},
                        )
                        if best is None or item[:3] > best[:3]:
                            best = item
            if best is not None:
                return best[3]
        return None

    def BuildRoadmap(self, sample_count: int | None = None, save: bool = True):
        """构建一次roadmap：节点先去重/择优，边只建立候选关系，不做碰撞检查。"""
        self.nodes = []
        self.edges = {}
        sample_count = self.config.sample_count if sample_count is None else int(sample_count)
        rng = np.random.default_rng(self.config.seed)
        start_time = time.monotonic()
        accepted = replaced = failed = 0
        batches = 0
        max_batches = max(sample_count * 50, sample_count + 100)
        progress_interval = max(1, int(self.config.progress_interval))
        next_progress = progress_interval

        while len(self.nodes) < sample_count and batches < max_batches:
            batches += 1
            node = self._SampleNode(rng)
            if node is None:
                failed += 1
                continue

            pos = node["pose"].t
            score = (int(node["counts"].min()), int(node["counts"].sum()))
            near_idx = [
                i for i, old in enumerate(self.nodes)
                if np.linalg.norm(old["pose"].t - pos) <= self.config.node_merge_radius
            ]
            if not near_idx:
                self.nodes.append(node)
                accepted += 1
                if len(self.nodes) >= next_progress:
                    print(
                        f"LazyPRM build progress: nodes={len(self.nodes)}/{sample_count}, "
                        f"accepted={accepted}, replaced={replaced}, failed_batches={failed}, "
                        f"batches={batches}, runtime={time.monotonic()-start_time:.2f}s"
                    )
                    next_progress += progress_interval
                continue

            # 同一个小区域只保留落脚点数量更好的节点，避免PRM图重复堆点。
            best_i = max(near_idx, key=lambda i: (int(self.nodes[i]["counts"].min()), int(self.nodes[i]["counts"].sum())))
            old_score = (int(self.nodes[best_i]["counts"].min()), int(self.nodes[best_i]["counts"].sum()))
            if score > old_score:
                self.nodes[best_i] = node
                replaced += 1

        self._RebuildEdges()
        if save:
            self.SaveRoadmap()
        print(
            f"LazyPRM build: nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"accepted={accepted}, replaced={replaced}, failed_batches={failed}, "
            f"runtime={time.monotonic()-start_time:.2f}s"
        )

    def _RebuildEdges(self):
        if len(self.nodes) == 0:
            return
        pts = np.vstack([n["pose"].t for n in self.nodes])
        tree = cKDTree(pts)
        for i, p in enumerate(pts):
            _, idx = tree.query(p, k=min(self.config.k_neighbors + 1, len(self.nodes)))
            for j in np.atleast_1d(idx):
                j = int(j)
                if j <= i:
                    continue
                if np.linalg.norm(pts[i] - pts[j]) <= self.config.connection_radius:
                    self.edges[(i, j)] = UNKNOWN

    def SaveRoadmap(self, file_path: str | None = None):
        file_path = self.roadmap_file if file_path is None else file_path
        pos = np.vstack([n["pose"].t for n in self.nodes]).astype(np.float32)
        quat = np.vstack([Rotation.from_matrix(n["pose"].R).as_quat() for n in self.nodes]).astype(np.float32)
        counts = np.vstack([n["counts"] for n in self.nodes]).astype(np.int32)
        modes = np.asarray([n["mode"] for n in self.nodes], dtype=np.int8)
        proj_dist = np.asarray([n["proj_dist"] for n in self.nodes], dtype=np.float32)
        edge_items = list(self.edges.items())
        edge_i = np.asarray([k[0] for k, _ in edge_items], dtype=np.int32)
        edge_j = np.asarray([k[1] for k, _ in edge_items], dtype=np.int32)
        edge_status = np.asarray([v for _, v in edge_items], dtype=np.int8)
        np.savez_compressed(
            file_path,
            positions=pos,
            quaternions=quat,
            landing_counts=counts,
            modes=modes,
            proj_dist=proj_dist,
            edge_i=edge_i,
            edge_j=edge_j,
            edge_status=edge_status,
            config_json=np.array([json.dumps(asdict(self.config))]),
        )
        print(f"save LazyPRM roadmap to {file_path}")

    def LoadRoadmap(self, file_path: str | None = None):
        file_path = self.roadmap_file if file_path is None else file_path
        with np.load(file_path) as data:
            pos = data["positions"]
            quat = data["quaternions"]
            counts = data["landing_counts"]
            modes = data["modes"]
            proj_dist = data["proj_dist"]
            self.nodes = [
                {
                    "pose": SE3.Rt(Rotation.from_quat(quat[i]).as_matrix(), pos[i]),
                    "counts": counts[i].astype(np.int32),
                    "mode": int(modes[i]),
                    "proj_dist": float(proj_dist[i]),
                }
                for i in range(pos.shape[0])
            ]
            self.edges = {
                (int(i), int(j)): int(s)
                for i, j, s in zip(data["edge_i"], data["edge_j"], data["edge_status"])
            }
        print(f"load LazyPRM roadmap from {file_path}: nodes={len(self.nodes)}, edges={len(self.edges)}")

    def LazyPRM(self, start_se3: SE3, goal_se3: SE3, max_rounds: int = 200):
        """查询时才检查候选路径上的边；坏边删除，好边缓存。"""
        start_ok, start_counts = self._PoseValid(start_se3)
        goal_ok, goal_counts = self._PoseValid(goal_se3)
        if not start_ok or not goal_ok:
            print(f"LazyPRM failed: start_valid={start_ok}, goal_valid={goal_ok}")
            return None
        if len(self.nodes) == 0:
            self.BuildRoadmap()

        nodes = self.nodes + [
            {"pose": start_se3, "counts": start_counts, "mode": 0, "proj_dist": 0.0},
            {"pose": goal_se3, "counts": goal_counts, "mode": 0, "proj_dist": 0.0},
        ]
        start_idx, goal_idx = len(nodes) - 2, len(nodes) - 1
        local_edges = dict(self.edges)
        pts = np.vstack([n["pose"].t for n in nodes])
        tree = cKDTree(pts[:len(self.nodes)])

        for qi in (start_idx, goal_idx):
            _, idx = tree.query(pts[qi], k=min(self.config.query_neighbors, len(self.nodes)))
            for j in np.atleast_1d(idx):
                j = int(j)
                if np.linalg.norm(pts[qi] - pts[j]) <= self.config.connection_radius:
                    local_edges[tuple(sorted((qi, j)))] = UNKNOWN
        if np.linalg.norm(pts[start_idx] - pts[goal_idx]) <= self.config.connection_radius:
            local_edges[(start_idx, goal_idx)] = UNKNOWN

        def edge_valid(a: SE3, b: SE3):
            res = self.config.edge_check_resolution
            if res is None:
                res = 0.5 * float(getattr(self.pointmap.voxels, "voxel_scale", 0.04))
            count = max(1, int(np.ceil(_SE3Distance(a, b, self.config.rot_weight) / float(res))))
            for i in range(1, count + 1):
                if not self._PoseValid(_InterpolateSE3(a, b, i / count))[0]:
                    return False
            return True

        def shortest_path():
            adj = [[] for _ in nodes]
            for (i, j), status in local_edges.items():
                if status == INVALID:
                    continue
                cost = _SE3Distance(nodes[i]["pose"], nodes[j]["pose"], self.config.rot_weight)
                adj[i].append((j, cost))
                adj[j].append((i, cost))
            heap = [(0.0, start_idx)]
            parent = {start_idx: -1}
            dist = {start_idx: 0.0}
            while heap:
                d, i = heapq.heappop(heap)
                if i == goal_idx:
                    path = []
                    while i >= 0:
                        path.append(i)
                        i = parent[i]
                    return path[::-1]
                if d > dist[i]:
                    continue
                for j, cost in adj[i]:
                    nd = d + cost
                    if nd < dist.get(j, np.inf):
                        dist[j] = nd
                        parent[j] = i
                        heapq.heappush(heap, (nd, j))
            return None

        for round_idx in range(max_rounds):
            path_idx = shortest_path()
            if path_idx is None:
                print(f"LazyPRM failed: graph disconnected after {round_idx} lazy rounds")
                return None
            changed = False
            for a, b in zip(path_idx[:-1], path_idx[1:]):
                key = tuple(sorted((a, b)))
                if local_edges[key] != UNKNOWN:
                    continue
                if edge_valid(nodes[a]["pose"], nodes[b]["pose"]):
                    local_edges[key] = VALID
                    if a < len(self.nodes) and b < len(self.nodes):
                        self.edges[key] = VALID
                else:
                    local_edges[key] = INVALID
                    if a < len(self.nodes) and b < len(self.nodes):
                        self.edges[key] = INVALID
                changed = True
                break
            if not changed:
                self.last_path = [nodes[i]["pose"] for i in path_idx]
                print(f"LazyPRM path found: states={len(self.last_path)}, lazy_rounds={round_idx}")
                return self.last_path
        print("LazyPRM failed: max lazy rounds reached")
        return None

    def VisPRMMap(self, path: list[SE3] | None = None, env_stride: int = 4, edge_stride: int = 1, show_invalid: bool = False):
        import pyvista as pv

        path = self.last_path if path is None else path
        plotter = pv.Plotter()
        plotter.show_axes()

        def add_points(points, color, size, opacity, label=None):
            if points is not None and points.shape[0] > 0:
                plotter.add_mesh(pv.PolyData(points), render_points_as_spheres=True, point_size=size, color=color, opacity=opacity, label=label)

        def add_edges(status, color, width, label):
            pairs = [(i, j) for (i, j), s in self.edges.items() if s == status]
            if len(pairs) == 0:
                return
            pairs = pairs[::max(1, int(edge_stride))]
            pts = np.vstack([self.nodes[i]["pose"].t for pair in pairs for i in pair])
            lines = np.asarray([[2, 2 * k, 2 * k + 1] for k in range(len(pairs))], dtype=np.int64)
            mesh = pv.PolyData(pts)
            mesh.lines = lines
            plotter.add_mesh(mesh, color=color, line_width=width, opacity=0.45, label=label)

        add_points(self.pointmap.points[::max(1, env_stride)], "lightgray", 2, 0.08, "env")
        add_points(self.pointmap.landing_points[::max(1, env_stride)], "black", 4, 0.12, "landing")
        add_points(getattr(self.pointmap, "edge_points", np.zeros((0, 3)))[::max(1, env_stride)], "red", 4, 0.35, "edge points")

        if len(self.nodes) > 0:
            pts = np.vstack([n["pose"].t for n in self.nodes])
            modes = np.asarray([n["mode"] for n in self.nodes])
            for mode, color in ((0, "cyan"), (1, "orange"), (2, "magenta")):
                add_points(pts[modes == mode], color, 7, 0.8, MODE_NAMES[mode])
            add_edges(UNKNOWN, "gray", 1, "unknown edges")
            add_edges(VALID, "green", 3, "valid edges")
            if show_invalid:
                add_edges(INVALID, "red", 2, "invalid edges")

        if path is not None and len(path) > 0:
            path_points = np.vstack([q.t.reshape(1, 3) for q in path])
            plotter.add_mesh(pv.lines_from_points(path_points), color="blue", line_width=6, label="path")
            add_points(path_points, "blue", 10, 1.0, "path states")

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    planner = HexLazyPRMPlanning(LazyPRMConfig(sample_count=1000, seed=0))
    planner.BuildRoadmap()
    planner.VisPRMMap()
