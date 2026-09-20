from __future__ import annotations

#在得到前端的SE3轨迹后，在这里进行加载，进行shortcut，沿着shortcut路径进行可视化，然后使用分段样条曲线优化SE3轨迹
import json,os
import numpy as np
from datetime import datetime
from spatialmath import SE3
from math import radians,ceil
from pathlib import Path
from typing import List,Sequence
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import HexState,Kinematic
from scipy.optimize import minimize, OptimizeResult
class OptCfg:
    # optimize_all = False #False/True 路标点是否可以被优化
    optimize_all = True #False/True 路标点是否可以被优化
    #李代数更新范围限制，值并不代表真实的位移距离，可以作为反馈
    delt_rho_limit=0.02 #平移对应范围限制
    delt_fai_limit=0.1 #旋转对应范围限制
    v_max = 0.1
    omega_max = 0.2
    max_retraction_iterations = 10
    cost_abs_change_tol = 1e-4
    cost_rel_change_tol = 1e-3
    # 每段贝塞尔曲线固定采样：困难区域保留的路标段更多，因而自然获得更密采样。
    # RobotFeasiCost很昂贵，使用较少点数；加速度和硬校验单独密采样。
    samples_per_segment = 5
    acceleration_samples_per_segment = 21
    validation_samples_per_segment = 41
    fd_rho_step = 0.001
    fd_fai_step = 0.002
    smooth_aggregation_temperature = 0.05
    acceleration_weight = 0.1
    speed_limit_tolerance = 1.0
    cost_progress_interval = 10
    # 每次retraction只做少量L-BFGS迭代，再在新的切空间重新线性化。
    optimizer_max_iterations = 2
    optimizer_max_function_evaluations = 64


    

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
        self._check_length_interval = 0.05
        self._check_rotate_interval = radians(10)

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
                len_dist_num = ceil(np.linalg.norm(T1.t-T2.t)/self._check_length_interval)
                rot_dist_num = ceil(T1.angdist(T2)/self._check_rotate_interval)
                check_num = max(len_dist_num,rot_dist_num)
                # print("check_num=",check_num)
                interp_feasi = True
                for i in range(check_num-1):
                    T = T1.interp(T2,(i+1)/check_num)
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
            times1 = np.linalg.norm(T1.t-T2.t)/self.opt_cfg.v_max
            times2 = T1.angdist(T2)/self.opt_cfg.omega_max
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


    def _SampleTrajectory(
        self,
        ctrl_poses:Sequence[SE3],
        samples_per_segment:int,
    )->tuple[np.ndarray,List[SE3]]:
        """每段固定采样点数，同时保留真实时间戳；连接点只保留一次。"""
        if (
            isinstance(samples_per_segment,(bool,np.bool_))
            or not isinstance(samples_per_segment,(int,np.integer))
            or samples_per_segment<3
        ):
            raise ValueError("samples_per_segment must be an integer of at least 3")
        if len(ctrl_poses)!=4*self.se3_segment_nums:
            raise ValueError("ctrl_poses does not match segment count")
        sample_times:List[float] = []
        sample_poses:List[SE3] = []
        elapsed = 0.0
        for segment_index,segment_time in enumerate(self.se3_segment_times):
            u_samples = np.linspace(0.0,1.0,samples_per_segment)
            if segment_index>0:
                u_samples = u_samples[1:]
            segment_ctrl = ctrl_poses[4*segment_index:4*segment_index+4]
            for u in u_samples:
                sample_times.append(elapsed+float(u)*segment_time)
                sample_poses.append(self.Bezier(segment_ctrl,float(u)))
            elapsed += segment_time
        return np.asarray(sample_times,dtype=np.float64),sample_poses

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
            velocity_prev = -np.asarray(
                (sample_poses[i].inv()*sample_poses[i-1]).log(twist=True),
                dtype=np.float64,
            )/dt_prev
            velocity_next = np.asarray(
                (sample_poses[i].inv()*sample_poses[i+1]).log(twist=True),
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
    )->tuple[float,float,float]:
        sample_times,sample_poses = self._SampleTrajectory(
            ctrl_poses,self.opt_cfg.samples_per_segment
        )
        if len(candidate_indices)!=len(sample_poses):
            raise ValueError("candidate_indices does not match trajectory samples")
        feasibility_values = np.asarray([
            self.hex_state.RobotFeasiCost(
                pose,
                points_idx=indices,
                smooth_temperature=self.opt_cfg.smooth_aggregation_temperature,
            )
            for pose,indices in zip(sample_poses,candidate_indices)
        ],dtype=np.float64)
        feasibility_cost = self._TimeAverage(feasibility_values,sample_times)
        if (
            self.opt_cfg.acceleration_samples_per_segment
            ==self.opt_cfg.samples_per_segment
        ):
            acceleration_times = sample_times
            acceleration_poses = sample_poses
        else:
            acceleration_times,acceleration_poses = self._SampleTrajectory(
                ctrl_poses,self.opt_cfg.acceleration_samples_per_segment
            )
        acceleration_cost = self._AccelerationMeanCost(
            acceleration_poses,acceleration_times
        )
        total_cost = (
            feasibility_cost+self.opt_cfg.acceleration_weight*acceleration_cost
        )
        return float(total_cost),feasibility_cost,acceleration_cost

    def _ValidateTrajectory(
        self,
        ctrl_poses:Sequence[SE3],
    )->tuple[bool,dict]:
        """以硬判定复核碰撞、落脚候选数和采样速度。"""
        sample_times,sample_poses = self._SampleTrajectory(
            ctrl_poses,self.opt_cfg.validation_samples_per_segment
        )
        invalid_sample_count = 0
        for pose in sample_poses:
            _,_,body_mask,leg_mask = self.hex_state.RobotFeasiCheck(pose)
            landing_counts = leg_mask.any(axis=-1).sum(axis=-1)
            if not (body_mask.all() and (landing_counts>=3).all()):
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
            "sample_count":len(sample_poses),
            "max_linear_speed":max_linear_speed,
            "max_angular_speed":max_angular_speed,
            "speed_valid":speed_valid,
        }
        return invalid_sample_count==0 and speed_valid,diagnostics

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
        initial_valid,initial_diagnostics = self._ValidateTrajectory(initial_ctrl_poses)
        best_valid_ctrl_poses = initial_ctrl_poses.copy() if initial_valid else None
        last_result:OptimizeResult|None = None
        last_feasi_iteration = -1
        for iteration in range(self.opt_cfg.max_retraction_iterations):
            _,base_sample_poses = self._SampleTrajectory(
                self.se3_ctrl_poses,self.opt_cfg.samples_per_segment
            )
            # 此列表在本次minimize期间保持不变，到下一次retraction才更新。
            candidate_indices = [
                self.hex_state.GetRobotFeasiCostPoints(pose)
                for pose in base_sample_poses
            ]
            cost_evaluation_count = 0
            print(
                f"Retraction {iteration+1}: {len(base_sample_poses)} feasibility "
                f"samples, {opt_poses_num*6} optimization variables",
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
                    updated_ctrl_poses,candidate_indices
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
                self.se3_ctrl_poses,candidate_indices
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
            candidate_valid,diagnostics = self._ValidateTrajectory(candidate_ctrl_poses)
            #优化过程可能出现不可行的结果，先取用，可能在后续优化中会变成可行
            self.se3_ctrl_poses = candidate_ctrl_poses
            self.se3_opt_poses = [
                self.se3_ctrl_poses[index] for index in self.opt_poses_index
            ]

            if candidate_valid:
                best_valid_ctrl_poses = candidate_ctrl_poses.copy()
                last_feasi_iteration = iteration
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

        final_valid,final_diagnostics = self._ValidateTrajectory(self.se3_ctrl_poses)
        if not final_valid:
            if best_valid_ctrl_poses is None:
                self.se3_ctrl_poses = initial_ctrl_poses
                self.se3_opt_poses = [
                    self.se3_ctrl_poses[index] for index in self.opt_poses_index
                ]
                raise RuntimeError(
                    "Optimization produced no hard-feasible trajectory; restored "
                    f"the initial controls. Initial validation={initial_diagnostics}, "
                    f"final validation={final_diagnostics}"
                )
            self.se3_ctrl_poses = best_valid_ctrl_poses
            self.se3_opt_poses = [
                self.se3_ctrl_poses[index] for index in self.opt_poses_index
            ]
            print(
                f"Final candidate failed hard validation; rolled back to the {last_feasi_iteration+1} "
                "th times iterations hard-feasible controls"
            )
        print("Optimize get optimized SE3 poses")
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path")

        json_file = os.path.join(out_dir,
                                 f"optimized_se3_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        self.WriteJson(json_file,dense_sample_dt=0.1)
        return last_result
    
    def Geodesic(self,T1:SE3,T2:SE3,u:float)->SE3:
        """
        SE3的螺旋测底线差值 u=[0,1]从0到1变化时 从T1变化到T2
        """
        return T1 * SE3.Exp( u*(T1.inv()*T2).log() )
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

    def WriteJson(self,json_file:str,dense_sample_dt:float=0.5)->Path:
        """
        将优化后的分段三次 SE(3) Bezier 路径写入可视化 JSON 格式。

        输出格式与 ``SE3_path/example_se3_path.json`` 一致，包含路标点、
        按全局真实时间均匀采样的稠密轨迹，以及每段的控制点和持续时间。
        所有位姿均使用 ``t``（平移）、``ang``（轴角角度）和 ``vec``（单位轴）保存。
        """
        if not np.isfinite(dense_sample_dt) or dense_sample_dt<=0.0:
            raise ValueError("dense_sample_dt must be finite and positive")
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
        segment_end_times = np.cumsum(segment_times)
        total_time = float(segment_end_times[-1])

        # 除最终不足一个间隔的尾段外，稠密轨迹的时间点严格等间隔，并总会包含终点。
        dense_times = np.arange(0.0,total_time,dense_sample_dt,dtype=np.float64)
        if len(dense_times)==0 or dense_times[0]!=0.0:
            dense_times = np.insert(dense_times,0,0.0)
        if total_time-dense_times[-1]>1e-12:
            dense_times = np.append(dense_times,total_time)
        else:
            dense_times[-1] = total_time
        dense_poses = []
        for sample_time in dense_times:
            segment_index = min(
                int(np.searchsorted(segment_end_times,sample_time,side="right")),
                self.se3_segment_nums-1,
            )
            segment_start_time = 0.0 if segment_index==0 else segment_end_times[segment_index-1]
            u = (sample_time-segment_start_time)/segment_times[segment_index]
            dense_poses.append(self.Bezier(
                self.se3_ctrl_poses[4*segment_index:4*segment_index+4],
                float(np.clip(u,0.0,1.0)),
            ))

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
    se3_initial_file = LEGGED_GYM_ROOT_DIR+"/legged_gym/expert_complex_utils/SE3_path/teleop_demo_20260808_113346.json"
    post = PostProcess(se3_initial_file,hex_state,OptCfg())
    
    post.ShortCutPath()
    # post.se3_path_short.clear()
    # for se3 in post.se3_path:
    #     post.se3_path_short.append(se3)
    post.GetCtrlPoes()
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path")
    json_file = os.path.join(out_dir,
                                 f"initial_se3_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    post.WriteJson(json_file,dense_sample_dt=0.02)
    # post.Optimize()
    # axises= np.random.random((2,3))
    # axises = axises/np.linalg.norm(axises,axis=1,keepdims=True)
    # T1 = SE3.AngleAxis(1.3,axises[0])*SE3(np.random.random((3,)))
    # T2 = SE3.AngleAxis(0.4,axises[1])*SE3(np.random.random((3,)))
    # # print(T1.A)
    # # print(T2.A)
    # print(T1.log(twist=True))
