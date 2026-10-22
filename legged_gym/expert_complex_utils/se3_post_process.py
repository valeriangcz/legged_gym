from __future__ import annotations

#在得到前端的SE3轨迹后，在这里进行加载，进行shortcut，沿着shortcut路径进行可视化，然后使用分段样条曲线优化SE3轨迹
import json,os
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from spatialmath import SE3
from spatialmath.base import trlog
from math import radians
from pathlib import Path
from typing import List,Sequence
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import HexState,Kinematic
from scipy.optimize import minimize, OptimizeResult
class OptCfg:
    # optimize_all = False #False/True 路标点是否可以被优化
    optimize_all = True #False/True 路标点是否可以被优化
    #李代数更新范围限制，值并不代表真实的位移距离，可以作为反馈
    delt_rho_limit=0.04 #平移对应范围限制
    delt_fai_limit=0.2 #旋转对应范围限制
    v_max = 0.1
    omega_max = 0.2
    max_retraction_iterations = 10
    cost_abs_change_tol = 1e-4
    cost_rel_change_tol = 1e-3
    # 预采样只用于建立距离表；正式采样按平移/旋转李代数间隔生成。
    sampling_rho_interval = 0.01
    sampling_phi_interval = radians(2)
    presample_count = 201
    fd_rho_step = 0.001
    fd_fai_step = 0.002
    acceleration_weight = 0.1
    speed_limit_tolerance = 1.0
    cost_progress_interval = 10
    # 每次retraction只做少量L-BFGS迭代，再在新的切空间重新线性化。
    optimizer_max_iterations = 2
    optimizer_max_function_evaluations = 64
    # CMA-ES每次retraction的采样配置和代数上限。
    cma_sigma0 = 0.3
    cma_population_size = None
    cma_max_generations = 20
    cma_seed = 42
    cma_retraction_patience = 3


    

@dataclass(frozen=True)
class TrajectorySampleGrid:
    """一轮优化共用的采样参数，不缓存会随控制点变化的位姿。"""
    times: np.ndarray
    segment_indices: np.ndarray
    u_values: np.ndarray

    def __post_init__(self):
        for name,dtype in (("times",np.float64),("segment_indices",np.int64),
                           ("u_values",np.float64)):
            values = np.array(getattr(self,name),dtype=dtype,copy=True)
            values.setflags(write=False)
            object.__setattr__(self,name,values)


class PostProcess:
    def __init__(self,se3_path_file:str,hex_state:HexState,opt_cfg:OptCfg):
        self.se3_path:List[SE3] = []
        self.se3_path_short:List[SE3] = []
        self.se3_ctrl_poses:List[SE3] = []
        self.se3_segment_nums = 0
        self.se3_segment_times = []
        # self.se3_opt_poses:List=None #待优化的SE3位姿，为了便于批量操作，就放在一个里面了
        self.opt_poses_index:List = [] #待优化变量在se3_ctrl_poses中的索引
        self.vec_continuous_poses_index = [] #根据速度连续性条件得到的控制点在se3_ctrl_poses中的索引
        self.opt_cfg = opt_cfg
        self.hex_state = hex_state

        self.LoadJson(se3_path_file)


    def LoadJson(self,json_file:str):
        with open(json_file,"r") as file:
            raw_data = json.load(file)
        t_array = np.array(raw_data["t"])
        ang_array = np.array(raw_data["ang"])
        vec_array = np.array(raw_data["vec"])

        self.se3_path.clear()
        for t,ang,vec in zip(t_array,ang_array,vec_array):
            self.se3_path.append(SE3.Trans(t)*SE3.AngVec(ang,vec))
        print(f"----->Finish Loading SE3 path\n length={t_array.shape[0]}\n source= {json_file}<-------")    

    def ShortCutPath(self):
        """
        把前端输入的轨迹进行裁剪，减少路标点数量，尽可能使用直线连接起点终点
        从起点开始，找到距离最远的可行点，然后删除中间的点，然后沿着下一个点继续寻找
        """
        start_index = 0
        short_path_index = [0]
        self.se3_path_short=[self.se3_path[0]]
        while start_index<len(self.se3_path)-1:
            T1 = self.se3_path[start_index]
            #只要有任意一点能连接，就认为成功，要是计算到相邻点发现也无法连接，那就需要报错
            connectin_feasi = False
            for end_index in range(len(self.se3_path)-1,start_index,-1):
                print(f"checking path between index {start_index} and {end_index}")
                T2 = self.se3_path[end_index]
                self._CheckSamplingConfig()
                dense_u = np.linspace(0.0,1.0,self.opt_cfg.presample_count)
                dense_poses = [T1.interp(T2,float(u)) for u in dense_u]
                u_samples = self._DistanceSampleTimes(dense_u,dense_poses,[0.0,1.0])
                interp_feasi = True
                for u in u_samples:
                    T = T1.interp(T2,float(u))
                    _,_,body_cf_mask,leg_mask = self.hex_state.RobotFeasiCheck(T)
                    landing_counts = leg_mask.any(axis=-1).sum(axis=-1)
                    if body_cf_mask.all() and ((landing_counts>=3).all()):
                        continue
                    else:
                        interp_feasi = False
                        break
                # print("interp_feasi=",interp_feasi)
                if interp_feasi:
                    short_path_index.append(end_index)
                    self.se3_path_short.append(self.se3_path[end_index])
                    start_index = end_index
                    connectin_feasi = True
                    break
            if not connectin_feasi:
                print(f"start_index={start_index}, end_index={end_index}\n T1:\n{T1.A}\n, T2:\n{T2.A}")
                raise RuntimeError("adjecent transform cannot connect to each other")
        print("Shorten path have key points num=",len(short_path_index))
        print(short_path_index)

    def GetCtrlPoes(self):
        """
        根据ShortCutPath得到的结果计算分段贝塞尔曲线的控制点
        同时根据优化配置参数决定可以优化位姿在self.ctrl_poses中的索引
        """
        if len(self.se3_path_short)<2:
            raise ValueError("SE3 Bezier optimization requires at least two way poses")
        if self.opt_cfg.v_max<=0.0 or self.opt_cfg.omega_max<=0.0:
            raise ValueError("v_max and omega_max must be positive")
        self.se3_ctrl_poses.clear()
        self.se3_segment_times.clear()
        self.vec_continuous_poses_index.clear()
        for i in range(len(self.se3_path_short)-1):
            T1 = self.se3_path_short[i]
            T2 = self.se3_path_short[i+1]
            self.se3_ctrl_poses.extend([T1,self.Geodesic(T1,T2,1/3),self.Geodesic(T1,T2,2/3),T2])
            times1 = np.linalg.norm(T1.t-T2.t)/(self.opt_cfg.v_max*0.3)
            times2 = T1.angdist(T2)/(self.opt_cfg.omega_max*0.3)
            self.se3_segment_times.append(max(times1,times2,1e-6))
        self.se3_segment_nums = len(self.se3_path_short)-1
        #T0, c0, c1, T1, f(c1), c2, T2, f(c2), c3, T3, ..... Tn-1, f(cn-1), cn, Tn
        #0 , 1 , 2 , 3 , 4    , 5 , 6 , 7    , 8 , 9 ,
        #T0, T1, T2,....,Tn为经过shortcut之后的路标点，c0,c1,c2,...,cn为路标点之间的控制点，f(ci)w是根据速度连续性条件得到的控制点
        self.opt_poses_index=[1,2]
        if self.opt_cfg.optimize_all:
            # 版本1 路标点也参与优化。每个内部路标点在相邻两段中各保存一次：
            # [..., C(i-1)2, Ti], [Ti, Ci1, Ci2, ...]。只优化前一段末尾的 Ti，
            # UpdateCtrlPoses 会将其同步到后一段开头，避免两个副本发生分离。
            # 起点 T0 和终点 Tn 保持固定；每段的第二个控制点仍参与优化。
            for segment_index in range(1,self.se3_segment_nums):
                self.opt_poses_index.append(4*segment_index-1)
                self.vec_continuous_poses_index.append(4*segment_index+1)
                self.opt_poses_index.append(4*segment_index+2)
        else:
            # 版本2 路标点固定。每段在 se3_ctrl_poses 中独立保存4个位姿：
            # [T0,C01,C02,T1], [T1,C11,C12,T2], ...
            # 第二段起的第一个控制点由速度连续性推导，第二个控制点参与优化。
            for segment_index in range(1,self.se3_segment_nums):
                self.vec_continuous_poses_index.append(4*segment_index+1)
                self.opt_poses_index.append(4*segment_index+2)
        self.se3_opt_poses = [self.se3_ctrl_poses[i] for i in self.opt_poses_index]
        #为了满足速度连续性约束，更新一次，se3_delta为0
        se3_delta = np.zeros((len(self.opt_poses_index),6),dtype=np.float64)
        self.UpdateCtrlPoses(se3_delta,update_original=True)

    def UpdateCtrlPoses(self,se3_delta:np.ndarray,update_original=True)->List[SE3]|None:
        """
        根据输入的李代数se3_delta(N,6) 对现有控制点进行右扰动更新self.ctrl_poses
        update_original为True 则对self.se3_ctrl_poses进行更新不返回
        update_original为False 则不对self.se3_ctrl_poses进行更新 返回一个扰动后的新位姿序列
        """
        se3_delta = np.asarray(se3_delta,dtype=np.float64)
        if se3_delta.ndim!=2 or se3_delta.shape[1]!=6:
            raise ValueError("se3_delta must have shape (N,6)")
        if not np.isfinite(se3_delta).all():
            raise ValueError("se3_delta contains non-finite values")
        if se3_delta.shape[0] != len(self.opt_poses_index):
            raise RuntimeError(f"The first dim of se3_delta={se3_delta.shape[0]} not equal to self.opt_poses_index={len(self.opt_poses_index)}")
        
        if update_original:
            #先对优化变量进行更新
            for i,delta in enumerate(se3_delta):
                opt_index = self.opt_poses_index[i]
                self.se3_ctrl_poses[opt_index] = self.se3_ctrl_poses[opt_index]*SE3.Exp(delta)
            if self.opt_cfg.optimize_all:
                # 内部路标点在相邻段中有两个副本；使后一段起点与前一段终点一致。
                for segment_index in range(1,self.se3_segment_nums):
                    self.se3_ctrl_poses[4*segment_index] = self.se3_ctrl_poses[4*segment_index-1]
            #在根据速度连续性约束更新其余控制点
            if len(self.vec_continuous_poses_index)>0:
                for index in self.vec_continuous_poses_index:
                    segment_index = index//4
                    self.se3_ctrl_poses[index] = self.VecContinuous(
                        self.se3_ctrl_poses[index-3],
                        self.se3_ctrl_poses[index-2],
                        self.se3_segment_times[segment_index-1],
                        self.se3_segment_times[segment_index],
                    )
            self.se3_opt_poses = [
                self.se3_ctrl_poses[i] for i in self.opt_poses_index
            ]
            return None
        else:
            updated_se3_ctrl_poses = self.se3_ctrl_poses.copy()
            for i,delta in enumerate(se3_delta):
                opt_index = self.opt_poses_index[i]
                updated_se3_ctrl_poses[opt_index] = self.se3_ctrl_poses[opt_index]*SE3.Exp(delta)
            if self.opt_cfg.optimize_all:
                # 与原地更新分支相同：同步相邻段中重复保存的内部路标点。
                for segment_index in range(1,self.se3_segment_nums):
                    updated_se3_ctrl_poses[4*segment_index] = updated_se3_ctrl_poses[4*segment_index-1]
            #在根据速度连续性约束更新其余控制点
            if len(self.vec_continuous_poses_index)>0:
                for index in self.vec_continuous_poses_index:
                    segment_index = index//4
                    updated_se3_ctrl_poses[index] = self.VecContinuous(
                        updated_se3_ctrl_poses[index-3],
                        updated_se3_ctrl_poses[index-2],
                        self.se3_segment_times[segment_index-1],
                        self.se3_segment_times[segment_index],
                    )
            return updated_se3_ctrl_poses


    def _CheckSamplingConfig(self):
        for name in ("sampling_rho_interval","sampling_phi_interval"):
            value = getattr(self.opt_cfg,name)
            if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or value<=0.0:
                raise ValueError(f"{name} must be finite and positive")
        count = self.opt_cfg.presample_count
        if (
            isinstance(count,(bool,np.bool_))
            or not isinstance(count,(int,np.integer)) or count<3
        ):
            raise ValueError("presample_count must be an integer of at least 3")

    def _SegmentTimes(self,ctrl_poses:Sequence[SE3])->np.ndarray:
        if self.se3_segment_nums<=0 or len(ctrl_poses)!=4*self.se3_segment_nums:
            raise ValueError("ctrl_poses does not match segment count")
        durations = np.asarray(self.se3_segment_times,dtype=np.float64)
        if (durations.shape!=(self.se3_segment_nums,)
            or not np.isfinite(durations).all() or np.any(durations<=0.0)):
            raise ValueError("Bezier segment times must be finite and positive")
        if not all(np.isfinite(pose.A).all() for pose in ctrl_poses):
            raise ValueError("integration input contains non-finite values")
        return durations

    def _DistanceSampleTimes(self,dense_times,dense_poses,mandatory_times)->np.ndarray:
        """累计归一化 body twist 距离，再在非静止区间线性反查时间。"""
        self._CheckSamplingConfig()
        dense_times = np.asarray(dense_times,dtype=np.float64)
        delta = np.asarray([
            trlog((first.inv()*second).A,twist=True,check=False)
            for first,second in zip(dense_poses[:-1],dense_poses[1:])
        ],dtype=np.float64)
        if not np.isfinite(delta).all():
            raise ValueError("integration input contains non-finite values")
        dq = np.maximum(
            np.linalg.norm(delta[:,:3],axis=1)/self.opt_cfg.sampling_rho_interval,
            np.linalg.norm(delta[:,3:],axis=1)/self.opt_cfg.sampling_phi_interval,
        )
        # 数值静止区间形成平台；不让零分母参与反查。
        dq[dq<=1e-12] = 0.0
        q = np.r_[0.0,np.cumsum(dq)]
        targets = np.arange(0.0,q[-1],1.0)
        targets = targets[targets<q[-1]-1e-10]
        times = list(mandatory_times)
        if targets.size:
            left = np.searchsorted(q,targets,side="right")-1
            fraction = (targets-q[left])/dq[left]
            times.extend(dense_times[left]+fraction*np.diff(dense_times)[left])
        # 连续平台只保留起止与中点，保留静止时长及其两端的速度变化。
        changes = np.diff(np.r_[False,dq==0.0,False].astype(np.int8))
        for first,last in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            times.extend([dense_times[first],
                          (dense_times[first]+dense_times[last])/2.0,dense_times[last]])
        mandatory = np.asarray(mandatory_times,dtype=np.float64)
        tolerance = 32*np.finfo(np.float64).eps*max(1.0,float(dense_times[-1]))
        # 优先保留准确的端点/连接时间，避免浮点误差产生极短时间间隔。
        extras = [t for t in times if np.min(np.abs(mandatory-t))>tolerance]
        result = np.sort(np.r_[mandatory,extras])
        result = result[np.r_[True,np.diff(result)>tolerance]]
        if result.size<3:
            result = np.sort(np.r_[result,(dense_times[0]+dense_times[-1])/2.0])
        return result

    def _BuildTrajectorySampleGrid(self,ctrl_poses:Sequence[SE3])->TrajectorySampleGrid:
        """每段均匀预采样后，在整条轨迹上连续累计距离，不在连接处归零。"""
        self._CheckSamplingConfig()
        durations = self._SegmentTimes(ctrl_poses)
        dense_times,dense_poses = [],[]
        elapsed = 0.0
        for segment_index,duration in enumerate(durations):
            u_samples = np.linspace(0.0,1.0,self.opt_cfg.presample_count)
            if segment_index>0:
                u_samples = u_samples[1:]
            segment_ctrl = ctrl_poses[4*segment_index:4*segment_index+4]
            for u in u_samples:
                dense_times.append(elapsed+float(u)*duration)
                dense_poses.append(self.Bezier(segment_ctrl,float(u)))
            elapsed += duration
        ends = np.cumsum(durations)
        times = self._DistanceSampleTimes(dense_times,dense_poses,np.r_[0.0,ends])
        segments = np.minimum(np.searchsorted(ends,times,side="right"),len(durations)-1)
        starts = np.r_[0.0,ends[:-1]]
        u_values = np.clip((times-starts[segments])/durations[segments],0.0,1.0)
        return TrajectorySampleGrid(times,segments,u_values)

    def _SampleTrajectory(
        self,ctrl_poses:Sequence[SE3],sample_grid:TrajectorySampleGrid|None=None,
    )->tuple[np.ndarray,List[SE3]]:
        """建表或复用固定参数网格，在当前控制点上重新求值。"""
        durations = self._SegmentTimes(ctrl_poses)
        if sample_grid is None:
            sample_grid = self._BuildTrajectorySampleGrid(ctrl_poses)
        if not isinstance(sample_grid,TrajectorySampleGrid):
            raise TypeError("sample_grid must be a TrajectorySampleGrid")
        times,indices,u = sample_grid.times,sample_grid.segment_indices,sample_grid.u_values
        if (times.ndim!=1 or indices.shape!=times.shape or u.shape!=times.shape
            or times.size<3 or not np.isfinite(times).all() or not np.isfinite(u).all()
            or np.any(np.diff(times)<=0.0) or np.any(indices<0)
            or np.any(indices>=len(durations)) or np.any(u<0.0) or np.any(u>1.0)):
            raise ValueError("invalid trajectory sample grid")
        starts = np.r_[0.0,np.cumsum(durations)[:-1]]
        if (times[0]!=0.0 or not np.isclose(times[-1],sum(durations))
            or not np.allclose(times,starts[indices]+u*durations[indices],rtol=0,atol=1e-12)):
            raise ValueError("sample grid does not match segment times")
        poses = [self.Bezier(ctrl_poses[4*i:4*i+4],float(parameter))
                 for i,parameter in zip(indices,u)]
        return times,poses

    @staticmethod
    def _SamplingDiagnostics(poses:Sequence[SE3])->dict:
        delta = np.asarray([trlog((a.inv()*b).A,twist=True,check=False)
                            for a,b in zip(poses[:-1],poses[1:])])
        return {
            "sample_count":len(poses),
            "max_rho_step":float(np.max(np.linalg.norm(delta[:,:3],axis=1),initial=0.0)),
            "max_phi_step":float(np.max(np.linalg.norm(delta[:,3:],axis=1),initial=0.0)),
        }

    @staticmethod
    def _TimeAverage(values:np.ndarray,times:np.ndarray)->float:
        """使用梯形公式计算时间平均值。"""
        values = np.asarray(values,dtype=np.float64)
        times = np.asarray(times,dtype=np.float64)
        if values.ndim!=1 or times.ndim!=1 or values.size!=times.size:
            raise ValueError("values and times must be one-dimensional and equal-sized")
        if values.size<2 or times[-1]<=times[0]:
            raise ValueError("at least two increasing time samples are required")
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError("integration input contains non-finite values")
        if np.any(np.diff(times)<=0.0):
            raise ValueError("times must be strictly increasing")
        return float(np.trapz(values,times)/(times[-1]-times[0]))

    def _AccelerationMeanCost(
        self,
        sample_poses:Sequence[SE3],
        sample_times:np.ndarray,
    )->float:
        """计算局部body twist加速度平方的时间平均值。"""
        if len(sample_poses)<3:
            return 0.0
        acceleration_cost = np.zeros(len(sample_poses),dtype=np.float64)
        for i in range(1,len(sample_poses)-1):
            dt_prev = sample_times[i]-sample_times[i-1]
            dt_next = sample_times[i+1]-sample_times[i]
            # 两个速度都表达在当前位姿的局部坐标系中。
            relative_prev = sample_poses[i].inv()*sample_poses[i-1]
            velocity_prev = -np.asarray(
                trlog(relative_prev.A,twist=True,check=False),
                dtype=np.float64,
            )/dt_prev
            relative_next = sample_poses[i].inv()*sample_poses[i+1]
            velocity_next = np.asarray(
                trlog(relative_next.A,twist=True,check=False),
                dtype=np.float64,
            )/dt_next
            acceleration = 2.0*(velocity_next-velocity_prev)/(dt_prev+dt_next)
            acceleration_cost[i] = (
                np.dot(acceleration[:3],acceleration[:3])/(self.opt_cfg.v_max**2)
                +np.dot(acceleration[3:],acceleration[3:])/(self.opt_cfg.omega_max**2)
            )
        # 端点没有中心差分，用最近的内部值延拓，使梯形积分覆盖完整时长。
        acceleration_cost[0] = acceleration_cost[1]
        acceleration_cost[-1] = acceleration_cost[-2]
        return self._TimeAverage(acceleration_cost,sample_times)

    def _TrajectoryMeanCost(
        self,
        ctrl_poses:Sequence[SE3],
        candidate_indices:Sequence[np.ndarray],
        sample_grid:TrajectorySampleGrid,
    )->tuple[float,float,float]:
        sample_times,sample_poses = self._SampleTrajectory(ctrl_poses,sample_grid)
        if len(candidate_indices)!=len(sample_poses):
            raise ValueError("candidate_indices does not match trajectory samples")
        feasibility_values = np.asarray([
            self.hex_state.RobotFeasiCost(
                pose,
                points_idx=indices,
            )
            for pose,indices in zip(sample_poses,candidate_indices)
        ],dtype=np.float64)
        feasibility_cost = self._TimeAverage(feasibility_values,sample_times)
        acceleration_cost = self._AccelerationMeanCost(
            sample_poses,sample_times
        )
        total_cost = (
            feasibility_cost+self.opt_cfg.acceleration_weight*acceleration_cost
        )
        return float(total_cost),feasibility_cost,acceleration_cost

    def _ValidateTrajectory(
        self,
        ctrl_poses:Sequence[SE3],
        sample_grid:TrajectorySampleGrid|None=None,
    )->tuple[bool,dict]:
        """以硬判定复核碰撞、落脚候选数和采样速度。"""
        sample_times,sample_poses = self._SampleTrajectory(ctrl_poses,sample_grid)
        invalid_sample_count = 0
        invalid_body_count = 0
        invalid_leg_count = 0
        for pose in sample_poses:
            _,_,body_mask,leg_mask = self.hex_state.RobotFeasiCheck(pose)
            landing_counts = leg_mask.any(axis=-1).sum(axis=-1)
            if not body_mask.all():
                invalid_body_count += 1
            if (landing_counts<3).any():
                invalid_leg_count += 1
            if (not body_mask.all()) or ((landing_counts<3).any()):
                invalid_sample_count += 1
        dt = np.diff(sample_times)
        linear_speed = np.asarray([
            np.linalg.norm(next_pose.t-pose.t)/delta_t
            for pose,next_pose,delta_t in zip(sample_poses[:-1],sample_poses[1:],dt)
        ])
        angular_speed = np.asarray([
            pose.angdist(next_pose)/delta_t
            for pose,next_pose,delta_t in zip(sample_poses[:-1],sample_poses[1:],dt)
        ])
        max_linear_speed = float(np.max(linear_speed,initial=0.0))
        max_angular_speed = float(np.max(angular_speed,initial=0.0))
        speed_factor = 1.0+self.opt_cfg.speed_limit_tolerance
        speed_valid = (
            max_linear_speed<=speed_factor*self.opt_cfg.v_max
            and max_angular_speed<=speed_factor*self.opt_cfg.omega_max
        )
        diagnostics = {
            "invalid_sample_count":invalid_sample_count,
            "invalid_leg_count":invalid_leg_count,
            "invalid_body_count":invalid_body_count,
            **self._SamplingDiagnostics(sample_poses),
            "max_linear_speed":max_linear_speed,
            "max_angular_speed":max_angular_speed,
            "speed_valid":speed_valid,
        }
        return invalid_sample_count==0 and speed_valid,diagnostics

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

    def Optimize(self):
        """
        使用分段SE(3) Bezier曲线优化可行性和加速度平滑性。

        每次retraction内固定环境候选点；只有代价非增且通过硬校验的结果
        才记录为可回滚结果。
        """
        self.GetCtrlPoes()
        if self.opt_cfg.acceleration_weight<0.0:
            raise ValueError("acceleration_weight must be non-negative")
        if self.opt_cfg.fd_rho_step<=0.0 or self.opt_cfg.fd_fai_step<=0.0:
            raise ValueError("finite-difference steps must be positive")
        opt_poses_num = len(self.opt_poses_index)
        initial_variables = np.zeros(opt_poses_num*6,dtype=np.float64)
        single_bound = (
            [(-self.opt_cfg.delt_rho_limit,self.opt_cfg.delt_rho_limit)]*3
            +[(-self.opt_cfg.delt_fai_limit,self.opt_cfg.delt_fai_limit)]*3
        )
        bounds = single_bound*opt_poses_num
        finite_diff_step = np.tile(
            [self.opt_cfg.fd_rho_step]*3+[self.opt_cfg.fd_fai_step]*3,
            opt_poses_num,
        )

        initial_ctrl_poses = self.se3_ctrl_poses.copy()
        initial_valid,_ = self._ValidateTrajectory(initial_ctrl_poses)
        valid_snapshots = [initial_ctrl_poses.copy()] if initial_valid else []
        last_result:OptimizeResult|None = None
        for iteration in range(self.opt_cfg.max_retraction_iterations):
            sample_grid = self._BuildTrajectorySampleGrid(self.se3_ctrl_poses)
            _,base_sample_poses = self._SampleTrajectory(self.se3_ctrl_poses,sample_grid)
            # 此列表在本次minimize期间保持不变，到下一次retraction才更新。
            candidate_indices = [
                self.hex_state.GetRobotFeasiCostPoints(pose)
                for pose in base_sample_poses
            ]
            cost_evaluation_count = 0
            print(
                f"Retraction {iteration+1}: {len(base_sample_poses)} feasibility "
                f"samples, {opt_poses_num*6} optimization variables, "
                f"sampling={self._SamplingDiagnostics(base_sample_poses)}",
                flush=True,
            )

            def CostFunc(variables:np.ndarray)->float:
                nonlocal cost_evaluation_count
                variables = np.asarray(variables,dtype=np.float64)
                if variables.shape!=(opt_poses_num*6,) or not np.isfinite(variables).all():
                    return np.inf
                updated_ctrl_poses = self.UpdateCtrlPoses(
                    variables.reshape(-1,6),update_original=False
                )
                total_cost,_,_ = self._TrajectoryMeanCost(
                    updated_ctrl_poses,candidate_indices,sample_grid
                )
                cost_evaluation_count += 1
                if (
                    self.opt_cfg.cost_progress_interval>0
                    and cost_evaluation_count%self.opt_cfg.cost_progress_interval==0
                ):
                    print(
                        f"  cost evaluations={cost_evaluation_count}, "
                        f"latest cost={total_cost}",
                        flush=True,
                    )
                return total_cost

            base_cost,base_feasibility,base_acceleration = self._TrajectoryMeanCost(
                self.se3_ctrl_poses,candidate_indices,sample_grid
            )
            if iteration==0:
                print(
                    f"Initial time-mean cost={base_cost}, feasibility="
                    f"{base_feasibility}, acceleration={base_acceleration}",
                    flush=True,
                )
            res:OptimizeResult = minimize(
                CostFunc,
                initial_variables,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "eps":finite_diff_step,
                    "maxiter":self.opt_cfg.optimizer_max_iterations,
                    # "maxfun":self.opt_cfg.optimizer_max_function_evaluations,
                    "maxfun":(len(self.se3_opt_poses)+1)*3*2+20
                },
            )
            last_result = res
            candidate_cost = float(res.fun)
            if (
                not np.isfinite(candidate_cost)
                or not np.isfinite(res.x).all()
            ):
                print(f"Retraction {iteration+1} rejected: non-finite optimizer result")
                break
            change_threshold = max(
                self.opt_cfg.cost_abs_change_tol,
                self.opt_cfg.cost_rel_change_tol*max(abs(base_cost),abs(candidate_cost)),
            )
            if candidate_cost>base_cost:
                if candidate_cost<=base_cost+change_threshold:
                    print(
                        f"Retraction {iteration+1} stopped: cost change is within "
                        f"numerical tolerance ({base_cost} -> {candidate_cost})"
                    )
                else:
                    print(
                        f"Retraction {iteration+1} rejected: cost increased from "
                        f"{base_cost} to {candidate_cost}"
                    )
                break

            candidate_ctrl_poses = self.UpdateCtrlPoses(
                res.x.reshape(-1,6),update_original=False
            )
            candidate_valid,diagnostics = self._ValidateTrajectory(candidate_ctrl_poses,sample_grid)
            #优化过程可能出现不可行的结果，先取用，可能在后续优化中会变成可行
            self.se3_ctrl_poses = candidate_ctrl_poses
            self.se3_opt_poses = [
                self.se3_ctrl_poses[index] for index in self.opt_poses_index
            ]

            if candidate_valid:
                valid_snapshots.append(candidate_ctrl_poses.copy())
                print(f"iterations {iteration} success")

            cost_change = abs(base_cost-candidate_cost)
            print(
                f"Retraction {iteration+1}: cost={candidate_cost}, "
                f"change={cost_change}, hard_valid={candidate_valid}, "
                f"validation={diagnostics}"
            )
            if cost_change<=change_threshold:
                print(f"Retraction converged after {iteration+1} iterations")
                break
            if not res.success:
                if res.status==1:
                    # maxiter/maxfun是每次retraction的主动预算；已接受当前下降
                    # 步后，在新的切空间继续下一次retraction。
                    print(f"L-BFGS-B local budget reached: {res.message}")
                else:
                    print(f"L-BFGS-B stopped: {res.message}")
                    break

        final_grid,final_diagnostics,rolled_back = self._SelectFinalTrajectory(
            initial_ctrl_poses,valid_snapshots,"Optimization"
        )
        _,final_poses = self._SampleTrajectory(self.se3_ctrl_poses,final_grid)
        final_indices = [self.hex_state.GetRobotFeasiCostPoints(pose) for pose in final_poses]
        final_cost,feasibility_cost,acceleration_cost = self._TrajectoryMeanCost(
            self.se3_ctrl_poses,final_indices,final_grid
        )
        if last_result is None:
            last_result = OptimizeResult(x=initial_variables,success=True,message="No local iterations")
        last_result.fun = final_cost
        last_result.validation = final_diagnostics
        last_result.rolled_back = rolled_back
        last_result.feasibility_cost = feasibility_cost
        last_result.acceleration_cost = acceleration_cost
        print(f"Final time-mean cost={final_cost}, validation={final_diagnostics}")
        print("Optimize get optimized SE3 poses")
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path")

        json_file = os.path.join(out_dir,
                                 f"optimized_se3_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        last_result.output_path = self.WriteJson(json_file,sample_grid=final_grid)
        return last_result
    
    def Optimize_CMA_ES(self)->OptimizeResult:
        """串行CMA-ES优化；调用前需准备好se3_path_short。

        在每次retraction的固定切空间中运行多代有界采样，使用实际评估
        过的最佳候选更新轨迹，并保存通过当前硬校验的结果供最终回滚。
        返回的x是最终自由控制点相对初始控制点的累计右扰动（物理单位），
        不受单次retraction的边界限制。success表示最终轨迹通过硬校验，
        而非CMA-ES是否耗尽预算；nfev包含基准和最终重新评估的软代价，
        不包含硬校验。原有Optimize不依赖cma。
        """
        try:
            import cma
        except ImportError as exc:
            raise ImportError(
                "Optimize_CMA_ES requires pycma; install it with `pip install cma`."
            ) from exc

        cfg = self.opt_cfg

        self.GetCtrlPoes()
        dimension = 6*len(self.opt_poses_index)
        population_size = (
            int(4+3*np.log(dimension)) if cfg.cma_population_size is None
            else int(cfg.cma_population_size)
        )
        scales = np.tile(
            [cfg.delt_rho_limit]*3+[cfg.delt_fai_limit]*3,
            len(self.opt_poses_index),
        )
        initial_ctrl_poses = self.se3_ctrl_poses.copy()
        initial_valid,_ = self._ValidateTrajectory(initial_ctrl_poses)
        valid_snapshots = [initial_ctrl_poses.copy()] if initial_valid else []
        total_evaluations = 0
        total_generations = 0
        stagnation_count = 0
        stop_message = "Retraction budget reached"
        local_stop_message = ""

        def TrajectoryCost(ctrl_poses,indices,grid):
            try:
                return self._TrajectoryMeanCost(ctrl_poses,indices,grid)
            except ValueError as exc:
                # 现有积分器拒绝NaN/Inf；仅把这类明确的数值失败视为无效
                # 候选，形状、配置等其他错误仍向调用者报告。
                if str(exc)!="integration input contains non-finite values":
                    raise
                return np.inf,np.inf,np.inf

        for iteration in range(cfg.max_retraction_iterations):
            sample_grid = self._BuildTrajectorySampleGrid(self.se3_ctrl_poses)
            _,base_sample_poses = self._SampleTrajectory(self.se3_ctrl_poses,sample_grid)
            candidate_indices = [
                self.hex_state.GetRobotFeasiCostPoints(pose)
                for pose in base_sample_poses
            ]
            def Evaluate(ctrl_poses:Sequence[SE3])->float:
                nonlocal total_evaluations
                cost,_,_ = TrajectoryCost(ctrl_poses,candidate_indices,sample_grid)
                total_evaluations += 1
                if (
                    cfg.cost_progress_interval>0
                    and total_evaluations%cfg.cost_progress_interval==0
                ):
                    print(
                        f"  CMA-ES cost evaluations={total_evaluations}, "
                        f"latest cost={cost}",flush=True,
                    )
                return float(cost) if np.isfinite(cost) else np.inf

            base_cost = Evaluate(self.se3_ctrl_poses)
            best_cost = base_cost
            best_ctrl_poses = self.se3_ctrl_poses.copy()
            # 使用局部随机数生成器，避免pycma重置调用者的NumPy随机状态。
            seed = None if cfg.cma_seed is None else int(cfg.cma_seed)+iteration
            rng = np.random.default_rng(seed)
            es = cma.CMAEvolutionStrategy(
                np.zeros(dimension),cfg.cma_sigma0,
                {
                    "bounds":[-1.0,1.0],
                    "popsize":population_size,
                    "maxiter":int(cfg.cma_max_generations),
                    "seed":np.nan,
                    "randn":lambda *shape: rng.standard_normal(shape),
                    "verbose":-9,
                    "verb_log":0,
                    "signals_filename":"",
                },
            )
            print(
                f"CMA-ES retraction {iteration+1}: {dimension} variables, "
                f"population={es.popsize}, baseline cost={base_cost}, "
                f"sampling={self._SamplingDiagnostics(base_sample_poses)}",flush=True,
            )
            local_stop_message = "Generation budget reached"
            for generation in range(cfg.cma_max_generations):
                termination = es.stop()
                if termination:
                    local_stop_message = f"CMA-ES stopped: {termination}"
                    break
                solutions = es.ask()
                costs = []
                generation_best_cost = best_cost
                generation_best_ctrl_poses = None
                for solution in solutions:
                    delta = np.asarray(solution,dtype=np.float64)*scales
                    ctrl_poses = self.UpdateCtrlPoses(
                        delta.reshape(-1,6),update_original=False
                    )
                    cost = Evaluate(ctrl_poses)
                    costs.append(cost)
                    if cost<generation_best_cost:
                        generation_best_cost = cost
                        generation_best_ctrl_poses = ctrl_poses
                total_generations += 1
                if not np.isfinite(costs).any():
                    local_stop_message = "All population costs were non-finite"
                    print(f"  {local_stop_message}",flush=True)
                    break
                es.tell(solutions,costs)
                validation = None
                if generation_best_ctrl_poses is not None:
                    best_cost = generation_best_cost
                    best_ctrl_poses = generation_best_ctrl_poses
                    valid,validation = self._ValidateTrajectory(best_ctrl_poses,sample_grid)
                    if valid:
                        valid_snapshots.append(best_ctrl_poses.copy())
                print(
                    f"  generation {generation+1}: evaluations={total_evaluations}, "
                    f"best cost={best_cost}, validation={validation}",flush=True,
                )

            # 采样期间self中的基准保持不变；仅在本轮结束后执行retraction。
            self.se3_ctrl_poses = best_ctrl_poses
            self.se3_opt_poses = [
                best_ctrl_poses[index] for index in self.opt_poses_index
            ]
            if np.isfinite(base_cost) and np.isfinite(best_cost):
                threshold = max(
                    cfg.cost_abs_change_tol,
                    cfg.cost_rel_change_tol*max(abs(base_cost),abs(best_cost)),
                )
                significant_improvement = base_cost-best_cost>threshold
            else:
                significant_improvement = np.isfinite(best_cost)
            stagnation_count = 0 if significant_improvement else stagnation_count+1
            print(
                f"CMA-ES retraction {iteration+1}: cost={best_cost}, "
                f"stagnation={stagnation_count}, stop={local_stop_message}",flush=True,
            )
            if stagnation_count>=cfg.cma_retraction_patience:
                stop_message = "Retraction improvement tolerance reached"
                break

        final_grid,final_diagnostics,rolled_back = self._SelectFinalTrajectory(
            initial_ctrl_poses,valid_snapshots,"CMA-ES"
        )

        # 回滚和候选点更新后重新评估，返回值与实际导出的轨迹一致。
        _,final_sample_poses = self._SampleTrajectory(
            self.se3_ctrl_poses,final_grid
        )
        final_indices = [
            self.hex_state.GetRobotFeasiCostPoints(pose) for pose in final_sample_poses
        ]
        final_cost,feasibility_cost,acceleration_cost = TrajectoryCost(
            self.se3_ctrl_poses,final_indices,final_grid
        )
        total_evaluations += 1
        if not np.isfinite(final_cost):
            self.se3_ctrl_poses = initial_ctrl_poses
            self.se3_opt_poses = [
                initial_ctrl_poses[index] for index in self.opt_poses_index
            ]
            raise RuntimeError("Final CMA-ES cost is non-finite; restored the initial controls")
        cumulative_delta = np.concatenate([
            np.asarray(trlog(
                (initial_ctrl_poses[index].inv()*self.se3_ctrl_poses[index]).A,
                twist=True,check=False,
            ),dtype=np.float64)
            for index in self.opt_poses_index
        ])
        message = f"{stop_message}; {local_stop_message}"
        if rolled_back:
            message += "; rolled back to the last hard-feasible controls"
        out_dir = Path(LEGGED_GYM_ROOT_DIR)/"legged_gym/expert_complex_utils/SE3_path"
        json_file = out_dir/f"optimized_cma_es_path_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        output_path = self.WriteJson(str(json_file),sample_grid=final_grid)
        print(f"CMA-ES finished: cost={final_cost}, validation={final_diagnostics}",flush=True)
        return OptimizeResult(
            x=cumulative_delta,fun=float(final_cost),success=True,
            status=0 if stagnation_count>=cfg.cma_retraction_patience else 1,
            message=message,nit=total_generations,nfev=total_evaluations,
            retraction_iterations=iteration+1,rolled_back=rolled_back,
            validation=final_diagnostics,feasibility_cost=float(feasibility_cost),
            acceleration_cost=float(acceleration_cost),output_path=output_path,
        )

    def Geodesic(self,T1:SE3,T2:SE3,u:float)->SE3:
        """
        SE3的螺旋测底线差值 u=[0,1]从0到1变化时 从T1变化到T2
        """
        return T1 * SE3.Exp(u*trlog((T1.inv()*T2).A,check=False))
    def VecContinuous(self,ctrl1:SE3,T1:SE3,times1,times2)->SE3:
        """
        控制点排序为 ctrl1 T1 ctrl2 T1连接了两段 时间分别为times1和times2
        根据速度连续性条件返回ctrl2
        """
        #都是分段贝塞尔，速度连续性计算时间相除相同的
        times2_div_times1 = times2/times1
        return self.Geodesic(ctrl1,T1,1.0+times2_div_times1)

    def Bezier(self,ctrl_poses:List[SE3],u:float):
        """
        递推计算B赛尔曲线上的位姿值
        """
        T0 = ctrl_poses[0]
        T1 = ctrl_poses[1]
        T2 = ctrl_poses[2]
        T3 = ctrl_poses[3]
        T01=self.Geodesic(T0,T1,u)
        T12=self.Geodesic(T1,T2,u)
        T23=self.Geodesic(T2,T3,u)
        T012=self.Geodesic(T01,T12,u)
        T123=self.Geodesic(T12,T23,u)
        return self.Geodesic(T012,T123,u)

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
                "description":"Path exported by PostProcess.WriteJson.",
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

            

if __name__ == "__main__":
    hex_state = HexState(Kinematic())
    # se3_initial_file = LEGGED_GYM_ROOT_DIR+"/legged_gym/expert_complex_utils/SE3_path/teleop_demo_20260707_221719.json"
    # se3_initial_file = LEGGED_GYM_ROOT_DIR+"/legged_gym/expert_complex_utils/SE3_path/teleop_demo_20260708_220712.json"
    se3_initial_file = LEGGED_GYM_ROOT_DIR+"/legged_gym/expert_complex_utils/SE3_path/teleop_demo_20260929_105051.json"
    post = PostProcess(se3_initial_file,hex_state,OptCfg())
    
    post.ShortCutPath()
    post.Optimize()
    # post.Optimize_CMA_ES()

    # post.se3_path_short.clear()
    # for se3 in post.se3_path:
    #     post.se3_path_short.append(se3)
    # post.GetCtrlPoes()
    # out_dir = os.path.join(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path")
    # json_file = os.path.join(out_dir,
    #                              f"initial_se3_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    # post.WriteJson(json_file)
    # axises= np.random.random((2,3))
    # axises = axises/np.linalg.norm(axises,axis=1,keepdims=True)
    # T1 = SE3.AngleAxis(1.3,axises[0])*SE3(np.random.random((3,)))
    # T2 = SE3.AngleAxis(0.4,axises[1])*SE3(np.random.random((3,)))
    # # print(T1.A)
    # # print(T2.A)
    # print(T1.log(twist=True))
