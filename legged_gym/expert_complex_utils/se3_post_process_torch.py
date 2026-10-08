"""Population-parallel CMA-ES with CPU search/validation and Torch inference.

Usage::

    cfg = OptCfgTorch()
    scorer = HexStateTorch.from_files(point_map_file, device="cuda")
    post = PostProcessTorch(path_file, scorer, cfg)
    post.ShortCutPath()
    result = post.Optimize_CMA_ES()

Use ``dtype=torch.float64`` when creating the scorer together with
``cfg.torch_precision = "float64"`` for reference scoring. The default
``"mixed"`` uses double SE(3)/integration and single voxel scoring. This module
is independent of the NumPy PostProcess and HexState implementations. The second
constructor argument must be a HexStateTorch scorer, including its own native
hard check. An optional hard_check(SE3) callback can override that check. No
legacy robot object or compatibility adapter is accepted or retained here.
Set device and scoring dtype when constructing HexStateTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from math import radians
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import OptimizeResult
from spatialmath import SE3
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from .env_robot_voxles_torch import (
    HexStateTorch, PreparedCandidates, se3_exp, se3_log,
    se3_inverse, se3_geodesic, se3_bezier,
)


class OptCfgTorch:
    optimize_all = True
    delt_rho_limit = 0.04
    delt_fai_limit = 0.2
    v_max = 0.1
    omega_max = 0.2
    max_retraction_iterations = 10
    cost_abs_change_tol = 1e-4
    cost_rel_change_tol = 1e-3
    sampling_rho_interval = 0.01
    sampling_phi_interval = radians(2)
    presample_count = 201
    acceleration_weight = 0.1
    speed_limit_tolerance = 1.0
    cost_progress_interval = 10
    cma_sigma0 = 0.3
    cma_population_size = None
    cma_max_generations = 20
    cma_seed = 42
    cma_retraction_patience = 3
    torch_precision = "mixed"       # "mixed" or "float64"
    torch_population_chunk_size = 8
    torch_pose_chunk_size = 64
    torch_query_workers = -1


@dataclass(frozen=True)
class TrajectorySampleGrid:
    """A fixed retraction grid, owned by the Torch implementation."""
    times: np.ndarray
    segment_indices: np.ndarray
    u_values: np.ndarray

    def __post_init__(self):
        for name, dtype in (("times", np.float64), ("segment_indices", np.int64), ("u_values", np.float64)):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


class PostProcessTorch:
    def __init__(self, se3_path_file, torch_hex, opt_cfg=None, *, hard_check=None):
        if not isinstance(torch_hex, HexStateTorch):
            raise TypeError("torch_hex must be HexStateTorch; construct it with from_files or RobotVoxelData")
        self.se3_path = []
        self.se3_path_short = []
        self.se3_ctrl_poses = []
        self.se3_segment_nums = 0
        self.se3_segment_times = []
        self.opt_poses_index = []
        self.vec_continuous_poses_index = []
        self.opt_cfg = opt_cfg or OptCfgTorch()
        self.hard_check = hard_check
        self.torch_hex = torch_hex
        if opt_cfg is None:
            self.opt_cfg.torch_precision = "float64" if torch_hex.dtype == torch.float64 else "mixed"
            self.opt_cfg.torch_pose_chunk_size = torch_hex.pose_chunk_size
        self.LoadJson(se3_path_file)
        self._ensure_torch()

    def LoadJson(self, json_file):
        with open(json_file, encoding="utf-8") as stream:
            data = json.load(stream)
        translations, angles, axes = (np.asarray(data[key], dtype=np.float64) for key in ("t", "ang", "vec"))
        if (translations.ndim != 2 or translations.shape[1] != 3 or angles.shape != (len(translations),)
                or axes.shape != translations.shape or not np.isfinite(translations).all()
                or not np.isfinite(angles).all() or not np.isfinite(axes).all()):
            raise ValueError("path JSON requires finite t:(N,3), ang:(N,), vec:(N,3)")
        self.se3_path = [SE3.Trans(t)*SE3.AngVec(float(angle), axis)
                         for t, angle, axis in zip(translations, angles, axes)]

    def _check_pose(self, pose):
        checker = getattr(self, "hard_check", None)
        if checker is None:
            checker = self._ensure_torch().RobotFeasiCheck
        if not callable(checker):
            raise RuntimeError("hard_check must be callable")
        return checker(pose)

    def ShortCutPath(self):
        if len(self.se3_path) < 2:
            raise ValueError("path shortening requires at least two poses")
        self._CheckSamplingConfig()
        self.se3_path_short = [self.se3_path[0]]
        start = 0
        while start < len(self.se3_path)-1:
            first = self.se3_path[start]
            for end in range(len(self.se3_path)-1, start, -1):
                second = self.se3_path[end]
                # Match the front-end's SLERP + linear translation shortcut.
                u = np.linspace(0., 1., self.opt_cfg.presample_count)
                dense = [first.interp(second, float(value)) for value in u]
                parameters = self._DistanceSampleTimes(u, dense, [0., 1.])
                feasible = True
                for value in parameters:
                    _, _, body, legs = self._check_pose(first.interp(second, float(value)))
                    if not body.all() or (legs.any(axis=-1).sum(axis=-1) < 3).any():
                        feasible = False
                        break
                if feasible:
                    self.se3_path_short.append(second)
                    start = end
                    break
            else:
                raise RuntimeError("adjacent transforms cannot connect to each other")

    def GetCtrlPoes(self):
        if len(self.se3_path_short) < 2:
            raise ValueError("SE3 Bezier optimization requires at least two way poses")
        if self.opt_cfg.v_max <= 0 or self.opt_cfg.omega_max <= 0:
            raise ValueError("v_max and omega_max must be positive")
        self.se3_segment_nums = len(self.se3_path_short)-1
        waypoints = self._control_tensor(self.se3_path_short)
        first, last = waypoints[:-1], waypoints[1:]
        controls = torch.stack((first, se3_geodesic(first, last, 1/3),
                                se3_geodesic(first, last, 2/3), last), 1)
        self.se3_ctrl_poses = self._pose_list(controls.reshape(-1, 4, 4))
        self.se3_segment_times = [max(np.linalg.norm(b.t-a.t)/(self.opt_cfg.v_max*0.3),
                                      a.angdist(b)/(self.opt_cfg.omega_max*0.3), 1e-6)
                                  for a, b in zip(self.se3_path_short[:-1], self.se3_path_short[1:])]
        self.opt_poses_index = [1, 2]
        self.vec_continuous_poses_index = []
        for segment in range(1, self.se3_segment_nums):
            if self.opt_cfg.optimize_all:
                self.opt_poses_index.append(4*segment-1)
            self.vec_continuous_poses_index.append(4*segment+1)
            self.opt_poses_index.append(4*segment+2)
        self.UpdateCtrlPoses(np.zeros((len(self.opt_poses_index), 6)))

    @staticmethod
    def _pose_list(matrices):
        return [SE3(matrix, check=False) for matrix in matrices.detach().cpu().numpy()]

    def UpdateCtrlPoses(self, se3_delta, update_original=True):
        delta = np.asarray(se3_delta, dtype=np.float64)
        if delta.shape != (len(self.opt_poses_index), 6) or not np.isfinite(delta).all():
            raise ValueError("se3_delta must be finite (number_of_free_controls,6)")
        updated = self._pose_list(self.UpdateCtrlPosesBatch(delta[None])[0])
        if not update_original:
            return updated
        self.se3_ctrl_poses = updated
        self.se3_opt_poses = [updated[i] for i in self.opt_poses_index]

    def _CheckSamplingConfig(self):
        for name in ("sampling_rho_interval", "sampling_phi_interval"):
            value = getattr(self.opt_cfg, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        count = self.opt_cfg.presample_count
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 3:
            raise ValueError("presample_count must be an integer of at least 3")

    def _SegmentTimes(self, controls):
        if self.se3_segment_nums <= 0 or len(controls) != 4*self.se3_segment_nums:
            raise ValueError("ctrl_poses does not match segment count")
        durations = np.asarray(self.se3_segment_times, dtype=np.float64)
        if durations.shape != (self.se3_segment_nums,) or not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("Bezier segment times must be finite and positive")
        if not all(np.isfinite(pose.A).all() for pose in controls):
            raise ValueError("integration input contains non-finite values")
        return durations

    def _DistanceSampleTimes(self, dense_times, dense_poses, mandatory_times):
        self._CheckSamplingConfig()
        dense_times = np.asarray(dense_times, dtype=np.float64)
        matrices = dense_poses if isinstance(dense_poses, torch.Tensor) else self._control_tensor(dense_poses)
        delta = se3_log(se3_inverse(matrices[:-1]) @ matrices[1:])
        dq = torch.maximum(torch.linalg.vector_norm(delta[:, :3], dim=-1)/self.opt_cfg.sampling_rho_interval,
                           torch.linalg.vector_norm(delta[:, 3:], dim=-1)/self.opt_cfg.sampling_phi_interval).cpu().numpy()
        if not np.isfinite(dq).all():
            raise ValueError("integration input contains non-finite values")
        dq[dq <= 1e-12] = 0.
        cumulative = np.r_[0., np.cumsum(dq)]
        targets = np.arange(0., cumulative[-1], 1.)
        targets = targets[targets < cumulative[-1]-1e-10]
        times = list(mandatory_times)
        if targets.size:
            left = np.searchsorted(cumulative, targets, side="right")-1
            fraction = (targets-cumulative[left])/dq[left]
            times.extend(dense_times[left]+fraction*np.diff(dense_times)[left])
        changes = np.diff(np.r_[False, dq == 0., False].astype(np.int8))
        for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            times.extend((dense_times[first], (dense_times[first]+dense_times[last])/2, dense_times[last]))
        mandatory = np.asarray(mandatory_times, dtype=np.float64)
        tolerance = 32*np.finfo(np.float64).eps*max(1., float(dense_times[-1]))
        extras = [time for time in times if np.min(np.abs(mandatory-time)) > tolerance]
        result = np.sort(np.r_[mandatory, extras])
        result = result[np.r_[True, np.diff(result) > tolerance]]
        if len(result) < 3:
            result = np.sort(np.r_[result, (dense_times[0]+dense_times[-1])/2])
        return result

    @torch.inference_mode()
    def _BuildTrajectorySampleGrid(self, controls):
        self._CheckSamplingConfig()
        durations = self._SegmentTimes(controls)
        dense_times, indices, parameters = [], [], []
        elapsed = 0.
        for segment, duration in enumerate(durations):
            u = np.linspace(0., 1., self.opt_cfg.presample_count)
            if segment:
                u = u[1:]
            dense_times.extend(elapsed+u*duration)
            indices.extend([segment]*len(u))
            parameters.extend(u)
            elapsed += duration
        base = self._control_tensor(controls)
        index = torch.tensor(indices, device=base.device)
        u = torch.tensor(parameters, device=base.device, dtype=base.dtype)
        poses = self._sample_batch(base[None], index, u)[0]
        ends = np.cumsum(durations)
        times = self._DistanceSampleTimes(dense_times, poses, np.r_[0., ends])
        segments = np.minimum(np.searchsorted(ends, times, side="right"), len(durations)-1)
        starts = np.r_[0., ends[:-1]]
        parameters = np.clip((times-starts[segments])/durations[segments], 0., 1.)
        return TrajectorySampleGrid(times, segments, parameters)

    def _SampleTrajectory(self, controls, sample_grid=None):
        self._SegmentTimes(controls)
        grid = sample_grid if sample_grid is not None else self._BuildTrajectorySampleGrid(controls)
        poses = self.SampleTrajectoryBatch(self._control_tensor(controls)[None], grid)[0]
        return grid.times, self._pose_list(poses)

    @staticmethod
    def _TimeAverage(values, times):
        values, times = np.asarray(values), np.asarray(times)
        if values.ndim != 1 or times.ndim != 1 or values.size != times.size or len(times) < 2:
            raise ValueError("values and times must be one-dimensional equal-sized arrays of at least two samples")
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError("integration input contains non-finite values")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        return float(np.trapz(values, times)/(times[-1]-times[0]))

    def _AccelerationMeanCost(self, poses, times):
        if len(poses) < 3:
            return 0.
        matrices = self._control_tensor(poses)[None]
        timestamps = torch.as_tensor(np.array(times, copy=True), device=matrices.device, dtype=matrices.dtype)
        return float(self._acceleration_batch(matrices, timestamps)[0].cpu())

    def _SamplingDiagnostics(self, poses):
        matrices = self._control_tensor(poses)
        delta = se3_log(se3_inverse(matrices[:-1]) @ matrices[1:])
        maxima = torch.stack((torch.linalg.vector_norm(delta[:, :3], dim=-1).max(),
                              torch.linalg.vector_norm(delta[:, 3:], dim=-1).max())).cpu().numpy()
        return {"sample_count": len(poses), "max_rho_step": float(maxima[0]), "max_phi_step": float(maxima[1])}

    def _ValidateTrajectory(self, controls, sample_grid=None):
        times, poses = self._SampleTrajectory(controls, sample_grid)
        invalid_body = invalid_legs = invalid_samples = 0
        for pose in poses:
            _, _, body, legs = self._check_pose(pose)
            body_failed = not body.all()
            legs_failed = (legs.any(axis=-1).sum(axis=-1) < 3).any()
            invalid_body += int(body_failed)
            invalid_legs += int(legs_failed)
            invalid_samples += int(body_failed or legs_failed)
        linear_speed = max(np.linalg.norm(b.t-a.t)/dt for a, b, dt in zip(poses[:-1], poses[1:], np.diff(times)))
        angular_speed = max(a.angdist(b)/dt for a, b, dt in zip(poses[:-1], poses[1:], np.diff(times)))
        factor = 1+self.opt_cfg.speed_limit_tolerance
        speed_valid = linear_speed <= factor*self.opt_cfg.v_max and angular_speed <= factor*self.opt_cfg.omega_max
        diagnostics = {"invalid_sample_count": invalid_samples, "invalid_body_count": invalid_body,
                       "invalid_leg_count": invalid_legs, **self._SamplingDiagnostics(poses),
                       "max_linear_speed": float(linear_speed), "max_angular_speed": float(angular_speed),
                       "speed_valid": bool(speed_valid)}
        return invalid_samples == 0 and bool(speed_valid), diagnostics

    def _SelectFinalTrajectory(self,initial_ctrl_poses,valid_snapshots,optimizer_name):
        """用各候选自己的新网格复验，从最近快照向前回滚。"""
        candidates = [self.se3_ctrl_poses]+list(reversed(valid_snapshots))+[initial_ctrl_poses]
        seen = set()
        diagnostics = None
        for candidate in candidates:
            key = np.asarray([pose.A for pose in candidate]).tobytes()
            if key in seen:
                continue
            seen.add(key)
            grid = self._BuildTrajectorySampleGrid(candidate)
            valid,diagnostics = self._ValidateTrajectory(candidate,grid)
            if valid:
                rolled_back = candidate is not candidates[0]
                self.se3_ctrl_poses = [pose.copy() for pose in candidate]
                self.se3_opt_poses = [self.se3_ctrl_poses[i] for i in self.opt_poses_index]
                if rolled_back:
                    print(f"{optimizer_name}: rolled back to a trajectory passing fresh-grid validation")
                return grid,diagnostics,rolled_back
        self.se3_ctrl_poses = [pose.copy() for pose in initial_ctrl_poses]
        self.se3_opt_poses = [self.se3_ctrl_poses[i] for i in self.opt_poses_index]
        raise RuntimeError(
            f"{optimizer_name} produced no hard-feasible trajectory; restored the initial "
            f"controls. Fresh-grid validation={diagnostics}"
        )

    def WriteJson(self,json_file:str,sample_grid:TrajectorySampleGrid|None=None)->Path:
        """
        将优化后的分段三次 SE(3) Bezier 路径写入可视化 JSON 格式。

        输出格式与 ``SE3_path/example_se3_path.json`` 一致，包含路标点、
        按距离重采样并保存真实时间戳的稠密轨迹，以及每段的控制点和持续时间。
        所有位姿均使用 ``t``（平移）、``ang``（轴角角度）和 ``vec``（单位轴）保存。
        """
        if self.se3_segment_nums<=0 or len(self.se3_ctrl_poses)!=4*self.se3_segment_nums:
            raise RuntimeError("No valid Bezier control poses are available to write")
        if len(self.se3_segment_times)!=self.se3_segment_nums:
            raise RuntimeError("Bezier segment times do not match the control poses")

        def PoseBlock(poses:Sequence[SE3])->dict:
            translations = []
            angles = []
            axes = []
            for pose in poses:
                translation = np.asarray(pose.t,dtype=np.float64).reshape(3)
                angle,axis = pose.angvec()
                angle = float(angle)
                axis = np.asarray(axis,dtype=np.float64).reshape(3)
                if not np.isfinite(translation).all() or not np.isfinite(angle) or not np.isfinite(axis).all():
                    raise ValueError("Cannot write a non-finite SE(3) pose")
                if abs(angle)<=1e-12:
                    angle = 0.0
                    axis = np.zeros(3,dtype=np.float64)
                else:
                    axis_norm = float(np.linalg.norm(axis))
                    if axis_norm<=1e-12:
                        raise ValueError("Nonzero axis-angle rotation has a zero axis")
                    axis = axis/axis_norm
                translations.append(translation.tolist())
                angles.append(angle)
                axes.append(axis.tolist())
            return {"t":translations,"ang":angles,"vec":axes}

        segment_times = np.asarray(self.se3_segment_times,dtype=np.float64)
        if not np.isfinite(segment_times).all() or np.any(segment_times<=0.0):
            raise ValueError("Bezier segment times must be finite and positive")
        dense_times,dense_poses = self._SampleTrajectory(self.se3_ctrl_poses,sample_grid)

        waypoints = [self.se3_ctrl_poses[0]]
        waypoints.extend(
            self.se3_ctrl_poses[4*segment_index+3]
            for segment_index in range(self.se3_segment_nums)
        )
        bezier_segments = []
        for segment_index,duration in enumerate(segment_times):
            bezier_segments.append({
                "degree":3,
                "duration":float(duration),
                "control_poses":PoseBlock(
                    self.se3_ctrl_poses[4*segment_index:4*segment_index+4]
                ),
            })

        document = {
            "schema_version":1,
            "metadata":{
                "name":"Optimized SE(3) Bezier path",
                "description":"Path exported by PostProcessTorch.WriteJson.",
                "time_unit":"s",
                "translation_unit":"m",
                "rotation_unit":"rad",
            },
            "waypoints":PoseBlock(waypoints),
            "dense_poses":{"time":dense_times.tolist(),**PoseBlock(dense_poses)},
            "bezier_segments":bezier_segments,
        }
        output_path = Path(json_file)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        with output_path.open("w",encoding="utf-8") as file:
            json.dump(document,file,indent=2,ensure_ascii=False)
            file.write("\n")
        print(f"----->Finish writing optimized SE3 path to {output_path}<-------")
        return output_path

    def _ensure_torch(self):
        cfg = self.opt_cfg
        if cfg.torch_precision not in ("mixed", "float64"):
            raise ValueError("torch_precision must be 'mixed' or 'float64'")
        for name in ("torch_population_chunk_size", "torch_pose_chunk_size"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        dtype = torch.float32 if cfg.torch_precision == "mixed" else torch.float64
        if not isinstance(getattr(self, "torch_hex", None), HexStateTorch):
            raise TypeError("PostProcessTorch requires an initialized HexStateTorch")
        if self.torch_hex.dtype != dtype:
            raise ValueError("torch_precision does not match the HexStateTorch scoring dtype")
        self.torch_hex.pose_chunk_size = cfg.torch_pose_chunk_size
        return self.torch_hex

    def ReloadTorchMaps(self, map_data=None):
        """Upload new RobotVoxelData and rebuild local trees between retractions."""
        scorer = self._ensure_torch()
        scorer.reload_from(scorer.map_data if map_data is None else map_data)

    def _control_tensor(self, controls=None):
        scorer = self._ensure_torch()
        controls = self.se3_ctrl_poses if controls is None else controls
        if isinstance(controls, torch.Tensor):
            return controls.to(device=scorer.device, dtype=torch.float64)
        return torch.as_tensor(np.asarray([pose.A for pose in controls]),
                               device=scorer.device, dtype=torch.float64)

    @torch.inference_mode()
    def UpdateCtrlPosesBatch(self, deltas, base_controls=None):
        """Right perturbations (P,K,6) -> controls (P,4*segments,4,4)."""
        base = self._control_tensor(base_controls)
        deltas = torch.as_tensor(deltas, device=base.device, dtype=torch.float64)
        if deltas.ndim != 3 or deltas.shape[1:] != (len(self.opt_poses_index), 6):
            raise ValueError("deltas must have shape (P, number_of_free_controls, 6)")
        if base.shape != (4*self.se3_segment_nums, 4, 4):
            raise ValueError("base_controls does not match segment count")
        updated = base.unsqueeze(0).expand(len(deltas), -1, -1, -1).clone()
        updated[:, self.opt_poses_index] = base[self.opt_poses_index] @ se3_exp(deltas)
        if self.opt_cfg.optimize_all:
            starts = torch.arange(1, self.se3_segment_nums, device=base.device)*4
            updated[:, starts] = updated[:, starts-1]
        # Current constraints have no chain dependence: all C_i1 depend on the
        # previous segment's independently optimized C_(i-1)2 and endpoint.
        if self.vec_continuous_poses_index:
            indices = torch.tensor(self.vec_continuous_poses_index, device=base.device)
            segments = indices//4
            durations = torch.as_tensor(self.se3_segment_times, device=base.device, dtype=base.dtype)
            parameters = 1+durations[segments]/durations[segments-1]
            updated[:, indices] = se3_geodesic(updated[:, indices-3], updated[:, indices-2], parameters)
        return updated

    def _grid_tensors(self, grid):
        if not isinstance(grid, TrajectorySampleGrid):
            raise TypeError("sample_grid must be a TrajectorySampleGrid")
        durations = self._SegmentTimes(self.se3_ctrl_poses)
        times, indices, u = grid.times, grid.segment_indices, grid.u_values
        if (times.ndim != 1 or indices.shape != times.shape or u.shape != times.shape
                or len(times) < 3 or not np.isfinite(times).all() or not np.isfinite(u).all()
                or np.any(np.diff(times) <= 0) or np.any(indices < 0)
                or np.any(indices >= len(durations)) or np.any(u < 0) or np.any(u > 1)):
            raise ValueError("invalid trajectory sample grid")
        starts = np.r_[0., np.cumsum(durations)[:-1]]
        if (times[0] != 0 or not np.isclose(times[-1], sum(durations))
                or not np.allclose(times, starts[indices]+u*durations[indices], rtol=0, atol=1e-12)):
            raise ValueError("sample grid does not match segment times")
        device = self._ensure_torch().device
        return (torch.tensor(times.copy(), device=device, dtype=torch.float64),
                torch.tensor(indices.copy(), device=device),
                torch.tensor(u.copy(), device=device, dtype=torch.float64))

    def _sample_batch(self, controls, indices, parameters):
        controls = controls.reshape(len(controls), self.se3_segment_nums, 4, 4, 4)
        return se3_bezier(controls[:, indices], parameters)

    @torch.inference_mode()
    def SampleTrajectoryBatch(self, controls, sample_grid):
        """Evaluate controls (P,C,4,4) at the CPU reference's fixed grid."""
        scorer = self._ensure_torch()
        controls = torch.as_tensor(controls, device=scorer.device, dtype=torch.float64)
        if controls.ndim != 4 or controls.shape[1:] != (4*self.se3_segment_nums, 4, 4):
            raise ValueError("controls must have shape (P,4*segments,4,4)")
        _, indices, parameters = self._grid_tensors(sample_grid)
        return self._sample_batch(controls, indices, parameters)

    @staticmethod
    def _time_average_batch(values, times):
        return ((values[..., 1:]+values[..., :-1])*0.5*(times[1:]-times[:-1])).sum(-1)/(times[-1]-times[0])

    def _acceleration_batch(self, poses, times):
        if poses.shape[1] < 3:
            return poses.new_zeros(len(poses))
        middle_inv = se3_inverse(poses[:, 1:-1])
        prev_dt, next_dt = times[1:-1]-times[:-2], times[2:]-times[1:-1]
        previous = -se3_log(middle_inv @ poses[:, :-2])/prev_dt[None, :, None]
        following = se3_log(middle_inv @ poses[:, 2:])/next_dt[None, :, None]
        acceleration = 2*(following-previous)/(prev_dt+next_dt)[None, :, None]
        inside = ((acceleration[..., :3].square().sum(-1)/self.opt_cfg.v_max**2)
                  +(acceleration[..., 3:].square().sum(-1)/self.opt_cfg.omega_max**2))
        values = torch.cat((inside[:, :1], inside, inside[:, -1:]), -1)
        return self._time_average_batch(values, times)

    def PrepareTrajectoryCandidates(self, sample_grid):
        """CPU search and candidate upload once per retraction."""
        scorer = self._ensure_torch()
        _, poses = self._SampleTrajectory(self.se3_ctrl_poses, sample_grid)
        rows = scorer.GetRobotFeasiCostPointsBatch(
            np.asarray([pose.A for pose in poses]), workers=self.opt_cfg.torch_query_workers)
        return scorer.prepare_candidates(rows)

    @torch.inference_mode()
    def TrajectoryCostBatch(self, controls, candidates, sample_grid, *, _grid=None):
        """Return device tensors (total, feasibility, acceleration), each (P,).

        Only geometry is expanded across populations. Static candidate rows and
        map fields remain shared; candidate rows are gathered for each pose chunk.
        Invalid candidates produce inf without contaminating other trajectories.
        """
        scorer = self._ensure_torch()
        controls = torch.as_tensor(controls, device=scorer.device, dtype=torch.float64)
        if controls.ndim != 4 or controls.shape[1:] != (4*self.se3_segment_nums, 4, 4):
            raise ValueError("controls must have shape (P,4*segments,4,4)")
        if not isinstance(candidates, PreparedCandidates):
            candidates = scorer.prepare_candidates(candidates)
        times, indices, parameters = self._grid_tensors(sample_grid) if _grid is None else _grid
        if len(candidates.indices) != len(times) or candidates.indices.device != scorer.device:
            raise ValueError("candidate rows/device do not match trajectory samples")
        result = [[], [], []]
        for start in range(0, len(controls), self.opt_cfg.torch_population_chunk_size):
            chunk = controls[start:start+self.opt_cfg.torch_population_chunk_size]
            valid = torch.isfinite(chunk).reshape(len(chunk), -1).all(-1)
            safe = torch.where(valid[:, None, None, None], chunk,
                               torch.eye(4, device=chunk.device, dtype=chunk.dtype))
            poses = self._sample_batch(safe, indices, parameters)
            flattened = poses.reshape(-1, 4, 4).to(scorer.dtype)
            scores = []
            for offset in range(0, len(flattened), scorer.pose_chunk_size):
                stop = min(offset+scorer.pose_chunk_size, len(flattened))
                rows = torch.arange(offset, stop, device=scorer.device) % len(times)
                scores.append(scorer._score(flattened[offset:stop], candidates.select(rows)))
            feasibility = self._time_average_batch(torch.cat(scores).reshape(len(chunk), -1).double(), times)
            acceleration = self._acceleration_batch(poses, times)
            total = feasibility+self.opt_cfg.acceleration_weight*acceleration
            for target, values in zip(result, (total, feasibility, acceleration)):
                target.append(torch.where(valid & torch.isfinite(values), values, float("inf")))
        return tuple(torch.cat(values) if values else times.new_empty(0) for values in result)

    def _TrajectoryMeanCost(self, ctrl_poses, candidate_indices, sample_grid):
        """Single-trajectory compatibility adapter, including final reevaluation."""
        controls = self._control_tensor(ctrl_poses)[None]
        parts = self.TrajectoryCostBatch(controls, candidate_indices, sample_grid)
        return tuple(torch.stack(parts, -1).cpu().numpy()[0])

    def Optimize_CMA_ES(self):
        """Batch score each ask() population; preserve the original CPU lifecycle."""
        try:
            import cma
        except ImportError as exc:
            raise ImportError("Optimize_CMA_ES requires pycma; install it with `pip install cma`.") from exc
        scorer = self._ensure_torch()
        cfg = self.opt_cfg
        if cfg.max_retraction_iterations < 1 or cfg.cma_max_generations < 1:
            raise ValueError("retraction and CMA generation budgets must be positive")
        self.GetCtrlPoes()
        dimension = 6*len(self.opt_poses_index)
        population_size = int(4+3*np.log(dimension)) if cfg.cma_population_size is None else int(cfg.cma_population_size)
        scales = np.tile([cfg.delt_rho_limit]*3+[cfg.delt_fai_limit]*3, len(self.opt_poses_index))
        device_scales = torch.tensor(scales, device=scorer.device, dtype=torch.float64)
        initial = self.se3_ctrl_poses.copy()
        initial_valid, _ = self._ValidateTrajectory(initial)
        snapshots = [initial.copy()] if initial_valid else []
        evaluations = generations = stagnation = 0
        stop_message, local_stop = "Retraction budget reached", ""
        for iteration in range(cfg.max_retraction_iterations):
            grid = self._BuildTrajectorySampleGrid(self.se3_ctrl_poses)
            grid_tensors = self._grid_tensors(grid)
            candidates = self.PrepareTrajectoryCandidates(grid)
            base_controls = self._control_tensor()
            base_parts = self.TrajectoryCostBatch(base_controls[None], candidates, grid, _grid=grid_tensors)
            base_cost = float(base_parts[0].cpu()[0])
            evaluations += 1
            best_cost, best_controls = base_cost, self.se3_ctrl_poses.copy()
            rng = np.random.default_rng(None if cfg.cma_seed is None else int(cfg.cma_seed)+iteration)
            strategy = cma.CMAEvolutionStrategy(np.zeros(dimension), cfg.cma_sigma0, {
                "bounds": [-1., 1.], "popsize": population_size, "maxiter": int(cfg.cma_max_generations),
                "seed": np.nan, "randn": lambda *shape: rng.standard_normal(shape),
                "verbose": -9, "verb_log": 0, "signals_filename": "",
            })
            print(f"Torch CMA-ES retraction {iteration+1}: variables={dimension}, "
                  f"population={strategy.popsize}, samples={len(grid.times)}, "
                  f"device={scorer.device}, precision={cfg.torch_precision}, base cost={base_cost}", flush=True)
            local_stop = "Generation budget reached"
            for generation in range(cfg.cma_max_generations):
                termination = strategy.stop()
                if termination:
                    local_stop = f"CMA-ES stopped: {termination}"
                    break
                solutions = strategy.ask()
                deltas = torch.as_tensor(np.asarray(solutions), device=scorer.device, dtype=torch.float64)*device_scales
                # Control tensors are small (P*C*16); costly scoring is chunked.
                controls = self.UpdateCtrlPosesBatch(deltas.reshape(-1, len(self.opt_poses_index), 6), base_controls)
                parts = self.TrajectoryCostBatch(controls, candidates, grid, _grid=grid_tensors)
                costs = parts[0].cpu().numpy()
                evaluations += len(solutions)
                generations += 1
                if not np.isfinite(costs).any():
                    local_stop = "All population costs were non-finite"
                    print(f"  {local_stop}", flush=True)
                    break
                strategy.tell(solutions, costs.tolist())
                winner = int(np.argmin(costs))
                validation = None
                if costs[winner] < best_cost:
                    best_cost = float(costs[winner])
                    matrices = controls[winner].cpu().numpy()
                    best_controls = [SE3(matrix, check=False) for matrix in matrices]
                    valid, validation = self._ValidateTrajectory(best_controls, grid)
                    if valid:
                        snapshots.append(best_controls.copy())
                print(f"  generation {generation+1}: evaluations={evaluations}, "
                      f"best cost={best_cost}, validation={validation}", flush=True)
            self.se3_ctrl_poses = best_controls
            self.se3_opt_poses = [best_controls[i] for i in self.opt_poses_index]
            threshold = max(cfg.cost_abs_change_tol, cfg.cost_rel_change_tol*max(abs(base_cost), abs(best_cost)))
            improved = base_cost-best_cost > threshold if np.isfinite(base_cost) and np.isfinite(best_cost) else np.isfinite(best_cost)
            stagnation = 0 if improved else stagnation+1
            print(f"Torch CMA-ES retraction {iteration+1}: cost={best_cost}, "
                  f"stagnation={stagnation}, stop={local_stop}", flush=True)
            if stagnation >= cfg.cma_retraction_patience:
                stop_message = "Retraction improvement tolerance reached"
                break
        final_grid, diagnostics, rolled_back = self._SelectFinalTrajectory(initial, snapshots, "Torch CMA-ES")
        final_candidates = self.PrepareTrajectoryCandidates(final_grid)
        final_parts = self.TrajectoryCostBatch(self._control_tensor()[None], final_candidates, final_grid)
        final_cost, feasibility, acceleration = torch.stack(final_parts, -1).cpu().numpy()[0]
        evaluations += 1
        if not np.isfinite(final_cost):
            self.se3_ctrl_poses = initial
            self.se3_opt_poses = [initial[i] for i in self.opt_poses_index]
            raise RuntimeError("Final CMA-ES cost is non-finite; restored the initial controls")
        initial_matrices = self._control_tensor(initial)[self.opt_poses_index]
        final_matrices = self._control_tensor()[self.opt_poses_index]
        cumulative = se3_log(se3_inverse(initial_matrices) @ final_matrices).reshape(-1).cpu().numpy()
        message = f"{stop_message}; {local_stop}"
        if rolled_back:
            message += "; rolled back to the last hard-feasible controls"
        output = Path(LEGGED_GYM_ROOT_DIR)/"legged_gym/expert_complex_utils/SE3_path"
        filename = output/f"optimized_cma_es_torch_path_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        output_path = self.WriteJson(str(filename), sample_grid=final_grid)
        return OptimizeResult(x=cumulative, fun=float(final_cost), success=True,
                              status=0 if stagnation >= cfg.cma_retraction_patience else 1,
                              message=message, nit=generations, nfev=evaluations,
                              retraction_iterations=iteration+1, rolled_back=rolled_back,
                              validation=diagnostics, feasibility_cost=float(feasibility),
                              acceleration_cost=float(acceleration), output_path=output_path,
                              torch_device=str(scorer.device), torch_precision=cfg.torch_precision)
