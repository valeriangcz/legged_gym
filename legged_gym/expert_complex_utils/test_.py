"""测试 _SwingTrajCal 轨迹生成，并在 IsaacGym 中验证阻尼雅可比关节控制。"""

import os
import sys
from argparse import ArgumentParser
import matplotlib

SHOW_CASES = "--show-cases" in sys.argv
if not SHOW_CASES:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from legged_gym.expert_complex_utils import ExpertComplex


BASE_POS = [0.38, 2.2, 1.5]
TARGET_SE3 = SE3(*BASE_POS)
LEG_INDEX = 5
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "swing_traj_debug.png")
TRACKING_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "swing_traj_tracking.png")
SPLINE_WORKSPACE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "swing_spline_workspace.png")


def _build_cases(category="base"):
    """集中定义所有摆动轨迹测试案例，category=None 时返回全部。"""
    cases = [
        {
            "name": "same branch, normal",
            "category": "base",
            "start_point": [0.08, 0.0, -0.12],
            "landing_point": [0.25, 0.08, 0.06],
            "start_branch": 1,
            "landing_branch": 1,
        },
        {
            "name": "same branch, pass singular",
            "category": "base",
            "start_point": [-0.11, 0.04, -0.17],
            "landing_point": [0.16, 0.05, -0.08],
            "start_branch": 1,
            "landing_branch": 1,
        },
        {
            "name": "change branch, no singular",
            "category": "base",
            "start_point": [0.14, -0.04, -0.16],
            "landing_point": [0.25, 0.06, -0.06],
            "start_branch": 1,
            "landing_branch": 0,
        },
        {
            "name": "singular then branch",
            "category": "base",
            "start_point": [-0.12, 0.05, -0.17],
            "landing_point": [0.22, -0.07, -0.07],
            "start_branch": 1,
            "landing_branch": 0,
        },
        {
            "name": "branch then singular",
            "category": "base",
            "start_point": [0.22, 0.05, -0.07],
            "landing_point": [-0.10, -0.07, -0.17],
            "start_branch": 0,
            "landing_branch": 1,
        },
        {
            "name": "stress singular z init",
            "category": "stress",
            "start_point": [-0.11, 0.04, -0.18],
            "landing_point": [0.16, 0.05, -0.07],
            "start_branch": 1,
            "landing_branch": 1,
            "singular_z_init": -0.16,
        },
        {
            "name": "stress branch q init",
            "category": "stress",
            "start_point": [0.14, -0.04, -0.16],
            "landing_point": [0.25, 0.06, -0.06],
            "start_branch": 1,
            "landing_branch": 0,
            "branch_q_init": [-0.5, 0.7],
        },
        {
            "name": "stress singular+branch init",
            "category": "stress",
            "start_point": [-0.12, 0.05, -0.18],
            "landing_point": [0.22, -0.07, -0.07],
            "start_branch": 1,
            "landing_branch": 0,
            "singular_z_init": -0.16,
            "branch_q_init": [0.65, -0.6],
        },
        {
            "name": "diag branch safe front",
            "category": "diagnostic",
            "start_point": [0.2141, -0.0088, -0.1542],
            "landing_point": [0.2454, -0.0357, -0.0584],
            "start_branch": 1,
            "landing_branch": 0,
            "branch_q_init": [0.45, -0.30],
        },
        {
            "name": "diag branch safe side",
            "category": "diagnostic",
            "start_point": [0.2404, 0.0396, -0.0555],
            "landing_point": [0.2102, 0.0174, -0.1485],
            "start_branch": 1,
            "landing_branch": 0,
            "branch_q_init": [-0.45, 0.25],
        },
        {
            "name": "diag short branch safe",
            "category": "diagnostic",
            "start_point": [0.2126, -0.0085, -0.1619],
            "landing_point": [0.2122, 0.0177, -0.0733],
            "start_branch": 1,
            "landing_branch": 0,
            "branch_q_init": [0.35, -0.20],
        },
        {
            "name": "diag singular then branch safe",
            "category": "diagnostic",
            "start_point": [-0.1523, 0.0030, -0.1789],
            "landing_point": [0.2430, 0.0327, -0.0659],
            "start_branch": 1,
            "landing_branch": 0,
            "singular_z_init": -0.13,
            "branch_q_init": [0.40, -0.25],
        },
        {
            "name": "diag branch then singular safe",
            "category": "diagnostic",
            "start_point": [0.2409, -0.0379, -0.0611],
            "landing_point": [-0.1317, 0.0047, -0.1749],
            "start_branch": 0,
            "landing_branch": 1,
            "singular_z_init": -0.13,
            "branch_q_init": [-0.40, 0.25],
        },
    ]
    if category is None:
        return cases
    return [case for case in cases if case.get("category") == category]


def _run_case(case):
    """调用 _SwingTrajCal 生成单个测试案例的足端摆动轨迹。"""
    start_point = np.asarray(case["start_point"], dtype=np.float32)
    landing_point = np.asarray(case["landing_point"], dtype=np.float32)
    start_normal = np.asarray(case.get("start_normal", [0.0, 0.0, 1.0]), dtype=np.float32)
    landing_normal = np.asarray(case.get("landing_normal", [0.0, 0.0, 1.0]), dtype=np.float32)
    expert = ExpertComplex()
    expert.gaits.fill(True)
    expert.gaits[LEG_INDEX] = False
    expert.last_gaits[:] = expert.gaits[:]
    expert.B_e_cur.fill(0.0)
    expert.B_e_cur[:, LEG_INDEX] = start_point
    expert.B_support_n.fill(0.0)
    expert.B_landing_n.fill(0.0)
    expert.B_support_n[2, :] = 1.0
    expert.B_landing_n[2, :] = 1.0
    expert.q3_branches.fill(int(case["start_branch"]))
    expert.B_e_traj = [expert.B_e_cur[:, i, None].copy() for i in range(6)]
    expert.B_e_traj_len.fill(1)
    expert.q_traj_index.fill(0)
    debug_inputs = {
        "leg_index": LEG_INDEX,
        "start_point": start_point,
        "landing_point": landing_point,
        "start_normal": start_normal,
        "landing_normal": landing_normal,
        "start_q3_branch": int(case["start_branch"]),
        "landing_q3_branch": int(case["landing_branch"]),
    }
    if "singular_z_init" in case:
        debug_inputs["singular_z_init"] = float(case["singular_z_init"])
    if "branch_q_init" in case:
        debug_inputs["branch_q_init"] = np.asarray(case["branch_q_init"], dtype=np.float32)
    expert._SwingTrajCal(
        TARGET_SE3,
        debug_mode=True,
        debug_inputs=debug_inputs,
    )
    info = _build_debug_info(expert, case)
    return expert, info


def _build_debug_info(expert, case):
    """读取 _SwingTrajCal 的debug关键点，供画图和控制测试使用。"""
    B_traj = expert.B_e_traj[LEG_INDEX]
    start_point = np.asarray(case["start_point"], dtype=np.float32)
    landing_point = np.asarray(case["landing_point"], dtype=np.float32)
    change_branch = int(case["start_branch"]) != int(case["landing_branch"])
    raw_info = getattr(expert, "_last_swing_debug_info", {})
    case_type = raw_info.get("opt_case_type")
    opt_optimized = raw_info.get("opt_optimized_values")
    optimized_spline_points = raw_info.get("optimized_spline_points")
    swing_q_debug_info = raw_info.get("swing_q_debug_info", {})
    down2terminal_points = None
    if optimized_spline_points is not None:
        spline_len = np.asarray(optimized_spline_points).shape[1]
        if B_traj.shape[1] >= spline_len:
            down2terminal_points = B_traj[:, spline_len - 1:]
    info = {
        "traj_len": int(B_traj.shape[1]),
        "change_branch": raw_info.get("change_branch", change_branch),
        "pass_singular": raw_info.get("pass_singular", start_point[0] * landing_point[0] <= 0.0),
        "singular_index": -1,
        "branch_index": -1,
        "singular_point": None,
        "branch_point": None,
        "opt_case_type": case_type,
        "opt_initial_values": raw_info.get("opt_initial_values"),
        "opt_optimized_values": opt_optimized,
        "opt_delta_values": raw_info.get("opt_delta_values"),
        "opt_success": raw_info.get("opt_success"),
        "opt_fun": raw_info.get("opt_fun"),
        "initial_spline_points": raw_info.get("initial_spline_points"),
        "initial_spline_branch_indices": raw_info.get("initial_spline_branch_indices"),
        "initial_spline_control_points": raw_info.get("initial_spline_control_points"),
        "optimized_spline_points": optimized_spline_points,
        "down2terminal_points": down2terminal_points,
        "optimized_spline_branch_indices": raw_info.get("optimized_spline_branch_indices"),
        "optimized_spline_control_points": raw_info.get("optimized_spline_control_points"),
        "swing_q_debug_info": swing_q_debug_info,
    }
    if opt_optimized is not None and case_type in ("singular", "singular_branch", "branch_singular"):
        singular_z_idx = {"singular": 7, "singular_branch": 4, "branch_singular": 18}[case_type]
        singular_point = np.array([0.0, 0.0, opt_optimized[singular_z_idx]], dtype=np.float32)
        info["singular_point"] = singular_point
        info["singular_index"] = int(np.argmin(np.linalg.norm(B_traj - singular_point[:, None], axis=0)))
    if opt_optimized is not None and case_type in ("branch", "singular_branch", "branch_singular"):
        branch_q_slice = {"branch": slice(7, 9), "singular_branch": slice(14, 16), "branch_singular": slice(7, 9)}[case_type]
        branch_q = opt_optimized[branch_q_slice]
        branch_point = expert.kin.ForwardKinReturn(np.array([branch_q[0], branch_q[1], 0.0], dtype=np.float32))
        info["branch_point"] = branch_point
        info["branch_index"] = int(np.argmin(np.linalg.norm(B_traj - branch_point[:, None], axis=0)))
    return info


def _debug_mask(info, name, length):
    """读取关节轨迹IK调试mask，并保证长度和正式足端轨迹一致。"""
    mask = info.get("swing_q_debug_info", {}).get(name)
    if mask is None:
        return np.zeros(length, dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape[0] != length:
        fixed = np.zeros(length, dtype=bool)
        fixed[:min(length, mask.shape[0])] = mask[:min(length, mask.shape[0])]
        return fixed
    return mask


def _branch_ik_fail_masks(expert, B_points, branch_indices):
    """按给定分支检查每个点的IK状态。"""
    if B_points is None or branch_indices is None:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    B_points = np.asarray(B_points, dtype=np.float32)
    branch_indices = np.asarray(branch_indices, dtype=np.int32).reshape(-1)
    if branch_indices.shape[0] != B_points.shape[1]:
        length = min(branch_indices.shape[0], B_points.shape[1])
        B_points = B_points[:, :length]
        branch_indices = branch_indices[:length]
    _, solution_mask = expert.kin.InverseKin2MultiBatch(B_points.T)
    selected_valid = solution_mask[np.arange(B_points.shape[1]), branch_indices]
    any_valid = solution_mask.any(axis=1)
    return (~selected_valid) & any_valid, ~any_valid


def _scatter_masked_points(ax, B_points, mask, color, marker, label, point_size):
    """按mask绘制一类调试点。"""
    if B_points is None:
        return
    B_points = np.asarray(B_points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape[0] != B_points.shape[1]:
        fixed = np.zeros(B_points.shape[1], dtype=bool)
        fixed[:min(mask.shape[0], B_points.shape[1])] = mask[:min(mask.shape[0], B_points.shape[1])]
        mask = fixed
    if not mask.any():
        return
    p = B_points[:, mask]
    ax.scatter(
        p[0],
        p[1],
        p[2],
        color=color,
        marker=marker,
        s=point_size,
        depthshade=False,
        label=label,
    )


def _scatter_optimized_ik_debug_points(ax, B_traj, info, point_size=24):
    """在优化后足端轨迹上标出该分支无解、所有分支无解和过渡区内无解。"""
    if B_traj is None:
        return
    B_traj = np.asarray(B_traj, dtype=np.float32)
    masks = [
        ("ik_selected_branch_fail_mask", "tab:pink", "x", "optimized: no IK in this branch"),
        ("ik_all_branch_fail_mask", "red", "X", "optimized: no IK in all branch"),
        ("ik_transition_fail_mask", "black", "o", "optimized: no IK in transition"),
    ]
    for mask_name, color, marker, label in masks:
        mask = _debug_mask(info, mask_name, B_traj.shape[1])
        _scatter_masked_points(ax, B_traj, mask, color, marker, label, point_size)


def _set_axes_equal(ax, points):
    """让3D图坐标轴等比例显示。"""
    points = np.asarray(points, dtype=np.float32)
    points = points[:, np.isfinite(points).all(axis=0)]
    if points.shape[1] == 0:
        return
    mins = points.min(axis=1)
    maxs = points.max(axis=1)
    center = 0.5 * (mins + maxs)
    radius = max(0.55 * max(maxs - mins), 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _merge_intervals(intervals, n_points):
    """合并重叠的关节空间过渡区间。"""
    clipped = []
    for start, end, label in intervals:
        start = max(0, int(start))
        end = min(n_points - 1, int(end))
        if end > start:
            clipped.append([start, end, {label}])
    if not clipped:
        return []

    clipped.sort(key=lambda x: x[0])
    merged = [clipped[0]]
    for start, end, labels in clipped[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
            last[2].update(labels)
        else:
            merged.append([start, end, labels])
    return merged


def _mask_to_intervals(mask, label):
    """把距离阈值生成的布尔mask转换为连续区间。"""
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        return []
    intervals = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx != prev + 1:
            intervals.append((start, prev, label))
            start = idx
        prev = idx
    intervals.append((start, prev, label))
    return intervals


def _transition_intervals(info, B_traj, singular_radius=0.04, branch_radius=0.05):
    """根据足端到奇异轴/分支点的距离生成关节空间过渡区间。"""
    intervals = []
    if info["singular_index"] >= 0:
        singular_mask = np.linalg.norm(B_traj[:2], axis=0) <= singular_radius
        intervals.extend(_mask_to_intervals(singular_mask, "singular"))
    if info["branch_point"] is not None:
        branch_point = np.asarray(info["branch_point"], dtype=np.float32).reshape(3, 1)
        branch_mask = np.linalg.norm(B_traj - branch_point, axis=0) <= branch_radius
        intervals.extend(_mask_to_intervals(branch_mask, "branch"))
    return _merge_intervals(intervals, B_traj.shape[1])


def _build_control_targets(expert, info):
    """读取 ExpertComplex 内部生成的正式摆动关节轨迹。"""
    B_traj = expert.B_e_traj[LEG_INDEX]
    q_traj = expert.q_traj[LEG_INDEX]
    print(f"q_traj.shape={q_traj.shape} B_traj.shape={B_traj.shape}")
    if q_traj is None:
        raise RuntimeError("ExpertComplex did not generate q_traj for swing leg")
    B_ctrl = np.zeros_like(B_traj, dtype=np.float32)
    expert.kin.ForwardKin(q_traj.T, B_ctrl.T)
    intervals = _transition_intervals(info, B_traj)
    transition_used = np.zeros(B_traj.shape[1], dtype=bool)
    for start, end, _ in intervals:
        transition_used[start:end + 1] = True
    return B_ctrl, q_traj, transition_used


def _plot_offline(results):
    """保存足端轨迹和阻尼雅可比关节轨迹。"""
    fig = plt.figure(figsize=(13.0, 4.0 * len(results)))
    for row, (case, expert, info, B_ctrl, q_traj, transition_used) in enumerate(results):
        B_traj = expert.B_e_traj[LEG_INDEX]
        t = np.arange(q_traj.shape[1]) * expert.dt
        ax_q = fig.add_subplot(len(results), 2, row * 2 + 1)
        ax_q.plot(t, q_traj[0], label="thigh/q1")
        ax_q.plot(t, q_traj[1], label="knee/q2")
        ax_q.plot(t, q_traj[2], label="ankle/q3")
        if info["singular_index"] >= 0:
            ax_q.axvline(info["singular_index"] * expert.dt, color="tab:red", linestyle="--", label="singular")
        if info["branch_index"] >= 0:
            ax_q.axvline(info["branch_index"] * expert.dt, color="tab:purple", linestyle="--", label="branch")
        _shade_transition_regions(ax_q, transition_used, expert.dt)
        ax_q.set_title(f"{case['name']}  N={info['traj_len']}  joint_trans={int(transition_used.sum())}")
        ax_q.set_xlabel("time / s")
        ax_q.set_ylabel("joint / rad")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(fontsize=16)

        ax_3d = fig.add_subplot(len(results), 2, row * 2 + 2, projection="3d")
        ax_3d.plot(B_traj[0], B_traj[1], B_traj[2], color="tab:blue", label="planned foot")
        ax_3d.plot(B_ctrl[0], B_ctrl[1], B_ctrl[2], color="tab:green", alpha=0.75, label="control foot")
        if transition_used.any():
            p = B_ctrl[:, transition_used]
            ax_3d.scatter(p[0], p[1], p[2], color="black", s=7, label="joint transition")
        if info["singular_index"] >= 0:
            p = B_traj[:, info["singular_index"]]
            ax_3d.scatter([p[0]], [p[1]], [p[2]], color="tab:red", s=40, label="singular")
        if info["branch_index"] >= 0:
            p = B_traj[:, info["branch_index"]]
            ax_3d.scatter([p[0]], [p[1]], [p[2]], color="tab:purple", s=40, label="branch")
        _scatter_optimized_ik_debug_points(ax_3d, B_traj, info, point_size=28)
        _set_axes_equal(ax_3d, np.column_stack([B_traj, B_ctrl]))
        ax_3d.set_xlabel("Bx")
        ax_3d.set_ylabel("By")
        ax_3d.set_zlabel("Bz")
        ax_3d.view_init(elev=22, azim=-58)
        ax_3d.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(f"saved trajectory figure: {OUTPUT_PATH}")


def _workspace_cloud(expert, max_points=8000):
    """提取当前腿的可行体素中心点，用半透明散点表示工作空间。"""
    leg_voxels = expert.hex_state.robot_voxels.leg_voxels
    reachable = expert.hex_state.robot_voxels.robot_reachable_legs[:, :, LEG_INDEX].any(axis=1)
    reachable &= expert.hex_state.robot_voxels.to_bound_dist_flat[:, LEG_INDEX] >= 0.0
    centers = leg_voxels.center[reachable]
    if centers.shape[0] > max_points:
        stride = int(np.ceil(centers.shape[0] / max_points))
        centers = centers[::stride]
    return centers.T.astype(np.float32)


def _plot_control_points(ax, control_points, color, label):
    """绘制每段B样条控制点和控制多边形。"""
    if not control_points:
        return []
    plotted = []
    for seg_idx, cps in enumerate(control_points):
        cps = np.asarray(cps, dtype=np.float32)
        cur_label = label if seg_idx == 0 else None
        ax.plot(
            cps[0],
            cps[1],
            cps[2],
            color=color,
            linestyle=":",
            linewidth=1.1,
            marker="o",
            markersize=3.5,
            alpha=0.9,
            label=cur_label,
        )
        plotted.append(cps)
    return plotted


def _plot_region_sphere(ax, center, color, radius=0.03):
    """画半透明球，表示该区域使用初末IK关节插值。"""
    center = np.asarray(center, dtype=np.float32).reshape(3)
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.18, linewidth=0, shade=False)


def _draw_spline_workspace_case(ax, case, expert, info, workspace=None):
    """在指定坐标轴上绘制一个case的B样条、控制点和工作空间。"""
    if workspace is None:
        workspace = _workspace_cloud(expert)
    ax.scatter(
        workspace[0],
        workspace[1],
        workspace[2],
        color="0.55",
        s=30,
        alpha=0.1,
        depthshade=False,
        label="reachable voxels",
    )

    initial = info.get("initial_spline_points")
    optimized = info.get("optimized_spline_points")
    down2terminal = info.get("down2terminal_points")
    all_points = [workspace]
    if initial is not None:
        initial = np.asarray(initial, dtype=np.float32)
        all_points.append(initial)
        ax.plot(initial[0], initial[1], initial[2], color="tab:orange", linewidth=1.6, label="initial spline")
        all_points.extend(_plot_control_points(
            ax, info.get("initial_spline_control_points"), "tab:orange", "initial ctrl"
        ))
        branch_fail, all_fail = _branch_ik_fail_masks(
            expert,
            initial,
            info.get("initial_spline_branch_indices"),
        )
        _scatter_masked_points(
            ax, initial, branch_fail, "tab:pink", "x", "initial: no IK in this branch", 16
        )
        _scatter_masked_points(
            ax, initial, all_fail, "red", "X", "initial: no IK in all branch", 20
        )
    if optimized is not None:
        optimized = np.asarray(optimized, dtype=np.float32)
        all_points.append(optimized)
        ax.plot(optimized[0], optimized[1], optimized[2], color="tab:blue", linewidth=1.8, label="optimized spline")
        if down2terminal is not None:
            down2terminal = np.asarray(down2terminal, dtype=np.float32)
            all_points.append(down2terminal)
            ax.plot(
                down2terminal[0],
                down2terminal[1],
                down2terminal[2],
                color="tab:cyan",
                linewidth=1.8,
                linestyle="--",
                label="down2terminal",
            )
        all_points.extend(_plot_control_points(
            ax, info.get("optimized_spline_control_points"), "tab:blue", "optimized ctrl"
        ))

    B_traj = expert.B_e_traj[LEG_INDEX]
    _scatter_optimized_ik_debug_points(ax, B_traj, info, point_size=30)
    all_points.append(B_traj)

    start = np.asarray(case["start_point"], dtype=np.float32)
    landing = np.asarray(case["landing_point"], dtype=np.float32)
    ax.scatter([start[0]], [start[1]], [start[2]], color="black", s=35, label="start")
    ax.scatter([landing[0]], [landing[1]], [landing[2]], color="tab:green", s=35, label="landing")
    all_points.extend([start.reshape(3, 1), landing.reshape(3, 1)])
    if info.get("singular_point") is not None:
        p = np.asarray(info["singular_point"], dtype=np.float32)
        _plot_region_sphere(ax, p, "tab:red")
        ax.scatter([p[0]], [p[1]], [p[2]], color="tab:red", s=40, label="singular")
        all_points.append(p.reshape(3, 1))
    if info.get("branch_point") is not None:
        p = np.asarray(info["branch_point"], dtype=np.float32)
        _plot_region_sphere(ax, p, "tab:purple")
        ax.scatter([p[0]], [p[1]], [p[2]], color="tab:purple", s=40, label="branch")
        all_points.append(p.reshape(3, 1))

    _set_axes_equal(ax, np.hstack(all_points))
    ax.set_title(case["name"])
    ax.set_xlabel("Bx")
    ax.set_ylabel("By")
    ax.set_zlabel("Bz")
    ax.view_init(elev=23, azim=-58)
    ax.legend(fontsize=16, loc="upper left")


def _plot_spline_workspace(results, show_cases=False):
    """保存工作空间图，并可逐个弹窗查看每个case。"""
    if len(results) == 0:
        return
    fig = plt.figure(figsize=(7.5, 4.7 * len(results)))
    for row, (case, expert, info) in enumerate(results):
        ax = fig.add_subplot(len(results), 1, row + 1, projection="3d")
        _draw_spline_workspace_case(ax, case, expert, info)
    fig.tight_layout()
    fig.savefig(SPLINE_WORKSPACE_OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(f"saved spline workspace figure: {SPLINE_WORKSPACE_OUTPUT_PATH}")

    if show_cases:
        for case, expert, info in results:
            fig = plt.figure(figsize=(8.5, 7.0))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            _draw_spline_workspace_case(ax, case, expert, info)
            fig.tight_layout()
            plt.show(block=True)
            plt.close(fig)


def _shade_transition_regions(ax, transition_used, dt):
    """在关节曲线图中标出关节空间过渡区间。"""
    indices = np.nonzero(transition_used)[0]
    if indices.size == 0:
        return
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx != prev + 1:
            ax.axvspan(start * dt, prev * dt, color="black", alpha=0.08)
            start = idx
        prev = idx
    ax.axvspan(start * dt, prev * dt, color="black", alpha=0.08)


def _qa_to_action(env, q_des, adhesions):
    """将 6x4 关节期望和吸附状态转换为 HexClimb action。"""
    import torch

    actions = torch.zeros((env.num_envs, 30), dtype=torch.float32, device=env.device)
    actions[:, 24:30] = torch.tensor(adhesions, dtype=torch.float32, device=env.device)[None, :]
    q_des_tensor = torch.tensor(q_des.reshape(1, 24), dtype=torch.float32, device=env.device)
    actions[:, 0:24] = (q_des_tensor - env.default_dof_pos[:, env.dof_drive_indices]) / env.cfg.control.action_scale
    return actions


def _initial_q_for_case(env, expert, case):
    """根据起点和指定IK分支生成仿真初始关节角。"""
    q_init = env.default_dof_pos[0, env.dof_drive_indices].detach().cpu().numpy().reshape(6, 4).copy()
    joints, mask = expert.kin.InverseKin2MultiBatch(np.asarray(case["start_point"], dtype=np.float32)[None, :])
    branch = int(case["start_branch"])
    if not mask[0, branch]:
        raise ValueError(f"{case['name']} start point has no IK solution in branch {branch}")
    q_init[LEG_INDEX, 0:3] = joints[0, branch]
    return q_init


def _set_sim_state(env, q_init):
    """直接写入 IsaacGym 初始状态。"""
    from isaacgym import gymtorch
    import torch

    with torch.no_grad():
        env.dof_pos[:] = env.default_dof_pos[:]
        env.dof_vel.zero_()
        env.dof_pos[:, env.dof_drive_indices] = torch.tensor(q_init.reshape(1, 24), dtype=torch.float32, device=env.device)
        env.dof_pos_des.zero_()
        env.dof_pos_des[:, env.dof_drive_indices] = env.dof_pos[:, env.dof_drive_indices]
        env.root_states[:, 0:3] = env.base_init_state[0:3]
        env.root_states[:, 3:7] = env.base_init_state[3:7]
        env.root_states[:, 7:13] = 0.0
        env.rb_forces.zero_()
        env.last_contacts.zero_()
        env.episode_length_buf.zero_()
        env.reset_buf.zero_()

    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)


def _make_hex_climb_env(headless=True, sim_device="cuda:0"):
    """创建单环境 HexClimb。"""
    from types import SimpleNamespace

    from isaacgym import gymapi
    from legged_gym.envs import HexClimb, HexClimbCfg
    from legged_gym.utils.helpers import class_to_dict, parse_sim_params

    env_cfg = HexClimbCfg()
    env_cfg.env.num_envs = 1
    env_cfg.init_state.pos = BASE_POS
    env_cfg.init_state.buffer_time = 1e6
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False
    env_cfg.terrain.measure_heights = False

    args = SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=sim_device,
        device=sim_device,
        sim_device_id=0,
        graphics_device_id=0,
        headless=headless,
        use_gpu=sim_device.startswith("cuda"),
        use_gpu_pipeline=sim_device.startswith("cuda"),
        subscenes=0,
        num_threads=0,
    )
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    return HexClimb(env_cfg, sim_params, args.physics_engine, args.sim_device, args.headless)


def _current_motor_q(env):
    return env.dof_pos[0, env.dof_motor_drive_indices].detach().cpu().numpy().reshape(6, 3)


def _draw_points(env, points, color=(1.0, 1.0, 0.0), radius=0.01):
    """在 viewer 中画点，headless 时自动跳过。"""
    if getattr(env, "viewer", None) is None:
        return
    from isaacgym import gymapi, gymutil

    geometry = gymutil.WireframeSphereGeometry(radius, 4, 4, color=color)
    for point in np.asarray(points, dtype=np.float32).reshape(-1, 3):
        pose = gymapi.Transform(gymapi.Vec3(x=float(point[0]), y=float(point[1]), z=float(point[2])), r=None)
        gymutil.draw_lines(geometry, env.gym, env.viewer, env.envs[0], pose)


def _draw_tracking_preview(env, expert, info, B_ctrl):
    """每条新轨迹绘制前先清理旧点，再绘制轨迹和关键点。"""
    if getattr(env, "viewer", None) is None:
        return
    env.gym.clear_lines(env.viewer)
    stride = max(1, B_ctrl.shape[1] // 250)
    W_traj = (TARGET_SE3 * expert.kin._B2R(B_ctrl[:, ::stride], LEG_INDEX)).T
    _draw_points(env, W_traj, color=(1.0, 1.0, 0.0), radius=0.008)
    if info["singular_index"] >= 0:
        p = (TARGET_SE3 * expert.kin._B2R(B_ctrl[:, info["singular_index"], None], LEG_INDEX)).T
        _draw_points(env, p, color=(1.0, 0.0, 0.0), radius=0.018)
    if info["branch_index"] >= 0:
        p = (TARGET_SE3 * expert.kin._B2R(B_ctrl[:, info["branch_index"], None], LEG_INDEX)).T
        _draw_points(env, p, color=(0.8, 0.0, 1.0), radius=0.018)


def _settle_initial_state(env, actions, settle_threshold, settle_stable_steps, settle_max_steps):
    """保持初始动作，直到被测腿关节误差连续稳定。"""
    stable = 0
    target_q = env.dof_pos[0, env.dof_motor_drive_indices].detach().cpu().numpy().reshape(6, 3)[LEG_INDEX]
    last_error = float("inf")
    for step in range(settle_max_steps):
        env.step(actions)
        q_cur = _current_motor_q(env)[LEG_INDEX]
        last_error = float(np.linalg.norm(q_cur - target_q))
        if last_error < settle_threshold:
            stable += 1
            if stable >= settle_stable_steps:
                return step + 1, last_error
        else:
            stable = 0
    return settle_max_steps, last_error


def _run_isaacgym_tracking(cases, offline_results, max_steps, headless, sim_device, settle_threshold,
                           settle_stable_steps, settle_max_steps):
    """在 IsaacGym 中验证阻尼雅可比关节控制跟踪效果。"""
    env = _make_hex_climb_env(headless=headless, sim_device=sim_device)
    tracking_results = []
    try:
        for case, expert, info, B_ctrl, q_traj, _ in offline_results:
            q_init = _initial_q_for_case(env, expert, case)
            _set_sim_state(env, q_init)
            adhesions = np.ones(6, dtype=np.float32)
            adhesions[LEG_INDEX] = 0.0
            q_des = q_init.copy()
            actions = _qa_to_action(env, q_des, adhesions)
            _draw_tracking_preview(env, expert, info, B_ctrl)
            settle_steps, settle_error = _settle_initial_state(
                env, actions, settle_threshold, settle_stable_steps, settle_max_steps
            )
            print(f"{case['name']}: settled in {settle_steps} steps, q_error={settle_error:.5f}")

            q_errors = []
            foot_errors = []
            run_steps = min(max_steps, q_traj.shape[1])
            for step_idx in range(run_steps):
                q_cur = _current_motor_q(env)
                B_cur_all = np.zeros((6, 3), dtype=np.float32)
                expert.kin.ForwardKin(q_cur, B_cur_all)
                q_des[LEG_INDEX, 0:3] = q_traj[:, step_idx]
                q_errors.append(float(np.linalg.norm(q_cur[LEG_INDEX] - q_traj[:, step_idx])))
                foot_errors.append(float(np.linalg.norm(B_cur_all[LEG_INDEX] - B_ctrl[:, step_idx])))
                actions = _qa_to_action(env, q_des, adhesions)
                _, _, _, reset_buf, _ = env.step(actions)
                if bool(reset_buf[0].item()):
                    print(f"{case['name']}: reset at step {step_idx}")
                    break

            q_errors = np.asarray(q_errors, dtype=np.float32)
            foot_errors = np.asarray(foot_errors, dtype=np.float32)
            tracking_results.append((case, q_errors, foot_errors))
            print(
                f"{case['name']}: steps={len(q_errors)}, "
                f"q_mean={q_errors.mean():.5f}, q_max={q_errors.max():.5f}, "
                f"foot_mean={foot_errors.mean():.5f}, foot_max={foot_errors.max():.5f}"
            )
    finally:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    _plot_tracking(tracking_results)


def _plot_tracking(results):
    """保存 IsaacGym 控制跟踪误差。"""
    fig, axes = plt.subplots(len(results), 1, figsize=(9.5, 2.8 * len(results)), squeeze=False)
    for row, (case, q_errors, foot_errors) in enumerate(results):
        ax = axes[row, 0]
        ax.plot(q_errors, label="joint error")
        ax.plot(foot_errors, label="foot error")
        ax.set_title(case["name"])
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(TRACKING_OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(f"saved tracking figure: {TRACKING_OUTPUT_PATH}")


def _print_optimization_variables(info):
    """打印所有优化变量的初值、优化后值和变化量。"""
    case_type = info.get("opt_case_type")
    init_values = info.get("opt_initial_values")
    opt_values = info.get("opt_optimized_values")
    if case_type is None or init_values is None or opt_values is None:
        print("  优化变量: None")
        return
    layouts = {
        "normal": [
            ("raise_dist", slice(0, 1)), ("ctrl1", slice(1, 4)),
            ("ctrl2", slice(4, 7)), ("down_dist", slice(7, 8)),
        ],
        "branch": [
            ("raise_dist", slice(0, 1)), ("ctrl1", slice(1, 4)), ("ctrl2", slice(4, 7)),
            ("branch_q", slice(7, 9)), ("ctrl4", slice(9, 12)), ("down_dist", slice(12, 13)),
        ],
        "singular": [
            ("raise_dist", slice(0, 1)), ("ctrl1", slice(1, 4)), ("ctrl2", slice(4, 7)),
            ("singular_z", slice(7, 8)), ("ctrl4", slice(8, 11)), ("down_dist", slice(11, 12)),
        ],
        "singular_branch": [
            ("raise_dist", slice(0, 1)), ("ctrl1", slice(1, 4)), ("singular_z", slice(4, 5)),
            ("ctrl3", slice(5, 8)), ("ctrl4", slice(8, 11)), ("ctrl5", slice(11, 14)),
            ("branch_q", slice(14, 16)), ("ctrl7", slice(16, 19)), ("down_dist", slice(19, 20)),
        ],
        "branch_singular": [
            ("raise_dist", slice(0, 1)), ("ctrl1", slice(1, 4)), ("ctrl2", slice(4, 7)),
            ("branch_q", slice(7, 9)), ("ctrl4", slice(9, 12)), ("ctrl5", slice(12, 15)),
            ("ctrl6", slice(15, 18)), ("singular_z", slice(18, 19)), ("down_dist", slice(19, 20)),
        ],
    }
    delta_flat = opt_values - init_values
    print(
        f"  优化结果: type={case_type}, success={info.get('opt_success')}, "
        f"fun={info.get('opt_fun')}, "
        f"|dx|={np.linalg.norm(delta_flat):.6f}, "
        f"max|dx|={np.max(np.abs(delta_flat)):.6f}"
    )
    print("  全部优化变量对比:")
    for name, slc in layouts.get(case_type, []):
        delta = opt_values[slc] - init_values[slc]
        print(
            f"    {name}: init={np.array2string(init_values[slc], precision=5, suppress_small=True)} "
            f"opt={np.array2string(opt_values[slc], precision=5, suppress_small=True)} "
            f"delta={np.array2string(delta, precision=5, suppress_small=True)}"
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--isaacgym", action="store_true", help="运行 IsaacGym 混合足端/关节空间控制验证")
    parser.add_argument("--show-cases", action="store_true", help="逐个弹窗显示每个case的B样条工作空间图")
    parser.add_argument("--case-category", default="base", choices=["base", "stress", "diagnostic", "all"], help="选择运行哪一组集中定义的case")
    parser.add_argument("--headless", action="store_true", help="IsaacGym 无窗口运行")
    parser.add_argument("--sim-device", default="cuda:0", help="IsaacGym 仿真设备")
    parser.add_argument("--max-tracking-steps", type=int, default=5000, help="每个案例最多仿真步数")
    parser.add_argument("--settle-threshold", type=float, default=0.01, help="初始稳定阶段关节误差阈值")
    parser.add_argument("--settle-stable-steps", type=int, default=20, help="连续达标多少步后开始跟踪")
    parser.add_argument("--settle-max-steps", type=int, default=500, help="初始稳定最多等待步数")
    args = parser.parse_args()
    cases = _build_cases(None if args.case_category == "all" else args.case_category)

    offline_results = []
    spline_workspace_results = []
    for case in cases:
        expert, info = _run_case(case)
        B_ctrl, q_traj, transition_used = _build_control_targets(expert, info)
        offline_results.append((case, expert, info, B_ctrl, q_traj, transition_used))
        spline_workspace_results.append((case, expert, info))
        print(
            f"{case['name']}: len={info['traj_len']}, "
            f"singular_idx={info['singular_index']}, branch_idx={info['branch_index']}, "
            f"joint_transition={int(transition_used.sum())}"
        )
        _print_optimization_variables(info)
    _plot_offline(offline_results)
    _plot_spline_workspace(spline_workspace_results, show_cases=args.show_cases)

    if args.isaacgym:
        _run_isaacgym_tracking(
            cases,
            offline_results,
            max_steps=args.max_tracking_steps,
            headless=args.headless,
            sim_device=args.sim_device,
            settle_threshold=args.settle_threshold,
            settle_stable_steps=args.settle_stable_steps,
            settle_max_steps=args.settle_max_steps,
        )


if __name__ == "__main__":
    main()
