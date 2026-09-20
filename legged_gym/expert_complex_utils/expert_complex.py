#六足在复杂环境中的控制算法，终极统一架构控制程序
#输入局部环境的点云，参考位姿，输出每条腿关节的期望位置和吸盘吸附状态
from typing import List, Union,Tuple
from math import *
import numpy as np
import warnings
from spatialmath import SE3
from .hex_utils import Kinematic
from .env_robot_voxels import HexState,EnvPointsVoxels
from scipy.interpolate import BSpline
from scipy.optimize import minimize

def se3_vee(Xi:np.ndarray):
    rho = Xi[:3,3]
    theta =np.array([Xi[2,1],Xi[0,2],Xi[1,0]])
    return np.hstack((rho,theta))
    # return np.hstack((theta,rho))

class ExpertComplex:
    """
    需要调用 LoadSE3(path:List[SE3])才能完成初始化
    """
    def __init__(self,singularity_threshold=0.02):
        # self.body_offset_x = 0.1
        # self.body_offset_y = 0.22
        # self.body_shape=np.zeros((3,6))
        # self.body_shape[0,0:3] = -self.body_offset_x
        # self.body_shape[0,3:6] = self.body_offset_x
        # self.body_shape[1,[0,3]] = -self.body_offset_y
        # self.body_shape[1,[1,4]] = self.body_offset_y

        self.singularity_threshold = singularity_threshold
        self.singularity_legs = np.zeros((6,),dtype=np.bool_) #进入奇异点，设置为1
        self.hold_q = np.zeros((6,),dtype=np.float32) #记录第一次进入奇异点值对应的真实关节角
        self.kin = Kinematic(singularity_threshold=self.singularity_threshold)
        # self.chs = RobotConvexHulls(singularity_threshold=self.singularity_threshold)
        self.hex_state = HexState(kinematic=self.kin)


        self.leg_names=["LB","LF","LM","RB","RF","RM"]
        self.gaits = np.zeros((6,),dtype=np.bool_) #1表示stance 0表示swing
        self.last_gaits = self.gaits.copy()
        self.dt = 0.01 # s
        self.v_max = 0.02 # m/s
        self.w_max = 0.06 # rad/s
        #stance 状态下最多移动的距离或者角度
        self._stance_maxmove_t = 0.1 #m
        self._stance_maxmove_w = 0.2 #rad
        self.path_se3 = None #外界输入的参考轨迹
        self.follow_path_index = 0
        self.interp_path_se3 = [] #根据当前位置插值计算的轨迹，与当前stance轨迹长度相同，与索引对应
        self.set_init = False
        #三角步态分组
        self.groups = [[0,1,5],[2,3,4]]
        # self.groups = [[2,3,4],[0,1,5]]
        self.stance_group_index = 0
        self.gaits[self.groups[self.stance_group_index]]=1
        self.last_gaits = self.gaits.copy()
        self._swing_t = 3.0
        #关节位置
        self.q_des = np.zeros((6,4),dtype=np.float32)
        self.q_traj_index = np.array([0]*6) #每条腿当前执行到的足端轨迹索引
        self.q_traj = [None for _ in range(6)] #摆动腿可选的关节空间轨迹，支撑腿保持None
        #足端位置
        # self.B_e_init= np.zeros((3,6)) #
        self.B_e_cur = np.zeros((3,6))
        #逆运动学解代表的分支 0对应q3为正，1对应q3为负 代表期望分支 在摆动阶段，当前在一个分支 期望在另一个分支
        self.q3_branches = np.ones((6),dtype=np.int32)#
        self.B_e_traj=[] #[np.ndarray(3,N),np.ndarray(3,M),...] 按照leg_names顺序
        self.B_e_traj_len = np.zeros(6,dtype=np.int32)
        self.landing_point = np.zeros((3,6)) #保存在B系下的表达
        self.W_landing_points = np.zeros((3,6)) #保存在W系下的表达
        self.B_landing_n = np.zeros((3,6)) #保存在B系下的表达 保存在B系下的表达 当前支撑/落脚点的法向量
        self.B_support_n = np.zeros((3,6))
        self.B_landing_n[2,:] = 1.0 #初始化为身体z轴方向
        self.B_support_n[2,:] = 1.0
        #接触力状态
        self.contact_count = np.zeros(6,dtype=np.int32)
        #吸附状态
        self.adhesions = np.zeros(6,dtype=np.bool_)
        self.max_adhesions_force = 300.0
        # 准静态支撑前馈：在机身 R 系中分配接触力，再映射到各腿局部 Jacobian。
        self.total_mass = 14.530042
        self.R_body_com = np.array([0.0014417,0.043319,0.0041358],dtype=np.float64)
        self.world_gravity = np.array([0.0,0.0,-9.81],dtype=np.float64)
        self.motor_torque_limits = np.full((6,3),27.0,dtype=np.float64)
        self.tau_ff = np.zeros((6,3),dtype=np.float32)
        self.static_contact_forces_R = np.zeros((6,3),dtype=np.float32)
        self._tau_ff_stance_mask = np.zeros(6,dtype=np.bool_)
        self._tau_ff_transition_start = np.zeros((6,3),dtype=np.float32)
        self._tau_ff_transition_steps = max(1,int(round(0.1/self.dt)))
        self._tau_ff_transition_remaining = 0
        #位于两个凸包相交部分的最高z值的奇异点，B系坐标为[0,0,-0.075]
        self.B_singular = np.array([0,0,-0.075],dtype=np.float32)
        #下面的解是经过验证的可以解出一个可行解的
        self.q_singular = self.kin.InverseKin2Multi(np.array([0,0,-0.08])).squeeze(0)
        
        #为了兼容numpy 不同版本，有些支持trapezoid，有些支持trapz
        if hasattr(np,"trapezoid"):
            self.trapezoid = np.trapezoid
        elif hasattr(np,"trapz"):
            self.trapezoid = np.trapz
        else:
            raise AttributeError("Current NumPy version provides neither 'trapezoid' nor 'trapz'; ")

    def SetInit(self,q_init:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """
        q_init 6,4，按照leg_names排列的
        @output q_des 6,4 角度期望值, adhesions 6吸附状态
        """
        #设置末端位置初始值和最开始的轨迹
        # self.kin.ForwardKin(q_init,self.B_e_init.T)
        self.q_des = q_init.copy()
        #设置腿部步态初始值
        self.stance_group_index = 0
        self.gaits.fill(0)
        self.gaits[self.groups[self.stance_group_index]] = 1
        self.adhesions.fill(0)
        self.adhesions[self.gaits] = 1
        self.last_gaits[:] = self.gaits[:]
        #清空腿末端轨迹
        self.B_e_traj.clear()
        self.q_traj = [None for _ in range(6)]
        self.B_e_traj_len.fill(0)
        self.q_traj_index.fill(0)
        self.B_landing_n.fill(0.0)
        self.B_support_n.fill(0.0)
        self.B_landing_n[2,:] = 1.0
        self.B_support_n[2,:] = 1.0
        self.tau_ff.fill(0.0)
        self.static_contact_forces_R.fill(0.0)
        self._tau_ff_stance_mask.fill(False)
        self._tau_ff_transition_start.fill(0.0)
        self._tau_ff_transition_remaining = 0
        self.set_init = True

        return self.q_des, self.adhesions

    def LoadSE3(self,path_se3:List[SE3]):
        self.path_se3 = path_se3

    def SetInitBySE3(self):
        """
        根据加载的se3的初值对每个脚设置初始位置
        """
        if len(self.path_se3)==0:
            raise RuntimeError("Use self.LoadSE3 before SetInitBySE3!!!")
        self.SetInit(np.zeros((6,4)))
        _,landing_idx,body_free,leg_feasi = self.hex_state.RobotFeasiCheck(self.path_se3[0])
        if (~body_free).any():
            raise RuntimeWarning("The body collide with env at path_se3[0] please check the path")
        #对每个支撑腿，选择一个最中心的落脚点
        point_map = self.hex_state.env_pointsmap_voxels
        B_init_points=[]
        B_init_norms = []
        for i in range(6):
            if self.gaits[i]:
                # 6,landing_num,2
                lp_points_idx = landing_idx[leg_feasi[i,:,self.q3_branches[i]]]
                #R2B W_T_R.inv() * w_points
                #N,3
                B_points = self.kin._R2B(self.path_se3[0].inv()*point_map.landing_points[lp_points_idx].T,i).T
                #N,3
                B_norms = (self.path_se3[0].R.T @ point_map.normals[lp_points_idx].T).T
                if i<3:
                    B_norms[:,:2] = -1*B_norms[:,:2]
                #在腿部的voxels下点在体素的索引
                leg_flat_idx=self.hex_state.robot_voxels.leg_voxels.Pos2FlatIndex(B_points)
                best_in_leg_idx = np.argmax(self.hex_state.robot_voxels.to_bound_dist_flat[leg_flat_idx,i])
                #为了避免初始状态就碰撞，设置距离落脚点原理平面2mm的位置
                self.B_landing_n[:,i] = B_norms[best_in_leg_idx]
                B_init_points.append(B_points[best_in_leg_idx]+B_norms[best_in_leg_idx]*0.002)
            else:
                B_init_points.append(np.array([0.18,0.0,0.0]))
        #N,3
        B_init_points = np.column_stack(B_init_points)
        print(B_init_points.shape)
        joints,mask = self.kin.InverseKin2MultiBatch(B_init_points.T)
        if mask[np.arange(6),self.q3_branches].all():
            self.q_des[:,:3] = joints[np.arange(6),self.q3_branches,:]
            self.q_des[:,3] = self._GetFootAngle(self.q_des[:,:3],B_init_points,real_robot=False)
        else:
            print("current pos\n",B_init_points)
            print("current branch\n",self.q3_branches)
            raise ValueError("Current pos and branch can not be solved")
        return self.q_des,self.adhesions
        





    def RequestSingleStep(self, cur_se3:SE3,q_cur:np.ndarray,q_torque:np.ndarray,adhesion_force:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        @input cur_se3当前的se3状态，q_cur6，3 电机驱动的关节角位置，q_torque 关节电机的扭矩 6，3
        adhesion_force 6 吸附力大小
        @output q_des (6,4), tau_ff (6,3), adhesions (6,)
        """
        if self.path_se3 == None:
            raise ValueError("path_se3 is None, please use LoadSE3(path_se3) first")
        if not self.set_init:
            raise RuntimeError("Please set init state of robot first by SetInit(q_init)")
        self.kin.ForwardKin(q_cur,self.B_e_cur.T)
        self.GaitPlanning(cur_se3,q_cur,q_torque,adhesion_force) #更新步态，更新B_e_des_traj
        self.GetJointAngles(q_cur)# 根据当前角度和期望执行的B_e_des计算目标角度
        self._ComputeQuasiStaticTauFF(cur_se3,q_cur)
        return self.q_des, self.tau_ff, self.adhesions

    @staticmethod
    def _Skew(vector:np.ndarray)->np.ndarray:
        """返回叉乘矩阵，使 ``_Skew(r) @ f == r x f``。"""
        x,y,z = np.asarray(vector,dtype=np.float64).reshape(3)
        return np.array([
            [0.0,-z,y],
            [z,0.0,-x],
            [-y,x,0.0],
        ],dtype=np.float64)

    def _ComputeQuasiStaticTauFF(self,cur_se3:SE3,q_cur:np.ndarray)->np.ndarray:
        """计算并平滑当前 stance 腿的准静态关节前馈扭矩。

        ``B_e_cur`` 和 ``Jacobian`` 都在各腿局部系；接触力分配在机身 R
        系完成。点坐标通过 ``_B2R`` 转换以包含腿根平移，但力仅经
        ``RVectorToLeg`` 做轴旋转，绝不使用点坐标变换。
        """
        q_cur = np.asarray(q_cur,dtype=np.float64)
        if q_cur.shape != (6,3) or not np.isfinite(q_cur).all():
            raise ValueError("q_cur must be a finite array with shape (6, 3)")

        target_tau = np.zeros((6,3),dtype=np.float64)
        forces_R = np.zeros((6,3),dtype=np.float64)
        stance_indices = np.flatnonzero(self.gaits)
        if stance_indices.size >= 3:
            # _B2R 是点坐标变换：这里必须保留腿根平移以形成正确力臂。
            R_foot = self.kin._B2R(
                self.B_e_cur[:,stance_indices],stance_indices
            )
            r_R = R_foot-self.R_body_com[:,None]
            count = stance_indices.size
            A = np.zeros((6,3*count),dtype=np.float64)
            for column in range(count):
                A[:3,3*column:3*column+3] = np.eye(3)
                A[3:,3*column:3*column+3] = self._Skew(r_R[:,column])

            R_WR = np.asarray(cur_se3.R,dtype=np.float64).reshape(3,3)
            gravity_R = R_WR.T@self.world_gravity
            b = np.concatenate((-self.total_mass*gravity_R,np.zeros(3)))
            if np.linalg.matrix_rank(A) == 6:
                force_stack = A.T@np.linalg.pinv(A@A.T,rcond=1e-9)@b
                residual = A@force_stack-b
                if np.isfinite(force_stack).all() and np.linalg.norm(residual) <= 1e-6*max(1.0,np.linalg.norm(b)):
                    force_R = force_stack.reshape(count,3)
                    forces_R[stance_indices] = force_R
                    # 力是向量：只做 R<->腿局部系旋转，不加入腿根平移。
                    force_leg = self.kin.RVectorToLeg(
                        force_R.T,stance_indices
                    ).T
                    jacobian = self.kin.Jacobian(q_cur[stance_indices])
                    target_tau[stance_indices] = -np.einsum(
                        "nai,na->ni",jacobian,force_leg
                    )

        target_tau = np.clip(
            target_tau,-self.motor_torque_limits,self.motor_torque_limits
        ).astype(np.float32)
        self.static_contact_forces_R[:] = forces_R.astype(np.float32)

        if not np.array_equal(self.gaits,self._tau_ff_stance_mask):
            self._tau_ff_transition_start[:] = self.tau_ff
            self._tau_ff_transition_remaining = self._tau_ff_transition_steps
            self._tau_ff_stance_mask[:] = self.gaits
        if self._tau_ff_transition_remaining > 0:
            completed = (
                self._tau_ff_transition_steps-self._tau_ff_transition_remaining+1
            )
            blend = completed/self._tau_ff_transition_steps
            self.tau_ff[:] = (
                (1.0-blend)*self._tau_ff_transition_start+blend*target_tau
            )
            self._tau_ff_transition_remaining -= 1
        else:
            self.tau_ff[:] = target_tau
        self.tau_ff[:] = np.clip(
            self.tau_ff,-self.motor_torque_limits,self.motor_torque_limits
        )
        return self.tau_ff

    def GaitPlanning(self,cur_se3:SE3,q_cur:np.ndarray,q_torque:np.ndarray,adhesion_force:np.ndarray):
        #步态的判断和切换
        self.adhesions[self.gaits]=True
        self.adhesions[~self.gaits]=False

        #判断swing 和 stance是否均为空，
        if len(self.B_e_traj) == 0:
            # self.B_e_traj=[self.B_e_init[:,i,None].copy() for i in range(6)]
            self.B_e_traj=[self.B_e_cur[:,i,None].copy() for i in range(6)]
            #均为空说明刚初始化，直接计算对应的stance和swing轨迹
            #计算SE3轨迹插值，同时计算可行支撑轨迹与可行长度，返回执行完支撑轨迹时的机器人理想位姿
            self._SE3Interp_and_StanceCal(cur_se3)
            #假设摆动轨迹2s内完成 支撑轨迹耗时大于等于2s 就采用2s位置的se3 否则采用最后的se3
            print("len(self.interp_path_se3)=",len(self.interp_path_se3))
            # target_se3 = self.interp_path_se3[-1] if len(self.interp_path_se3)*self.dt<3.0 else self.interp_path_se3[int(3.0/self.dt)]
            self._SwingTrajCal(self.interp_path_se3[-1])

        #判断swing当前轨迹索引，并进行接触检测
        swing_dones = self._ContactDetection(q_cur,q_torque) #6维，stance腿也为False
        #判断当前已经执行的轨迹长度
        execute_done_mask = self.q_traj_index == self.B_e_traj_len -1 
        swing_continue = ~swing_dones & ~self.gaits & ~execute_done_mask
        #已经接触的swing要设置为stance 同时清空对应的轨迹
        self.gaits[swing_dones] = True
        self._ClearTraj(swing_dones)
        #超过一半轨迹或者swing_done就设置为吸附，这一项不需要筛除stance
        set_adhesion = (self.q_traj_index >= self.B_e_traj_len/2.0) | swing_dones
        self.adhesions[set_adhesion] = True

        #判断stance是否到达极值
        stance_done = (execute_done_mask & self.gaits).any()


        #stance未到达极值 swing部分已经接触 同时gait要与上次不一样
        if not stance_done and swing_dones.any() and (self.gaits != self.last_gaits).any():
            #接触部分设置为stance，重新计算stance可行轨迹，
            # self._SE3Follow_and_StanceCal(cur_se3)
            stance_done = True
            print("se3 follow cal")
            #测试使用，认为此时已经stance_done
            # self._SE3Follow_StanceCal_Batch(cur_se3)

        #全部为stance状态，就可以开始交替了
        # if self.gaits.all() and stance_done:
        if self.gaits.all():

            #判断之前由swing变成stance组的吸附状态 group_index是吸附的那组腿
            swing_groud_index =(self.stance_group_index+1)%2
            if (adhesion_force[self.groups[swing_groud_index]]>0.8*self.max_adhesions_force).all():
            #成功
                #设置之前那一组释放
                self.adhesions[self.groups[self.stance_group_index]] = False
                #判断旧的stance释放是否成功
                if (adhesion_force[self.groups[self.stance_group_index]]<0.05*self.max_adhesions_force).all():
                #成功
                    if stance_done:
                        #切换swing和stance轨迹
                        self.stance_group_index = swing_groud_index
                        self.gaits.fill(0)
                        self.gaits[self.groups[self.stance_group_index]]=1
                        #计算步态切换之后的swing和stance轨迹，此时需要重新计算距离path最近的点，然后计算插值轨迹
                        self._SE3Interp_and_StanceCal(cur_se3)
                        print("len(self.interp_path_se3)=",len(self.interp_path_se3))
                        # target_se3 = self.interp_path_se3[-1] if len(self.interp_path_se3)*self.dt<3.0 else self.interp_path_se3[int(3.0/self.dt)]
                        self._SwingTrajCal(self.interp_path_se3[-1])
                #失败
                    #轨迹不动，继续执行释放
            #失败
                #轨迹不动，继续执行吸附

        #因为上面对swing和stace轨迹重新计算了，为了避免计算后轨迹长度只有1，而此时还没有stance done,q_traj_index会越过索引，因此增加判断
        stance_done = (self.B_e_traj_len[self.gaits] == 1).all() | stance_done
        swing_dones = (self.B_e_traj_len == 1) & (~self.gaits)
        swing_continue = ~swing_dones & swing_continue

        #对正在stance的腿执继续执行轨迹
        if not stance_done:
            self.q_traj_index[self.gaits]+=1
            self.follow_path_index += 1
            #测试阶段，每次没有stance结束就计算未来两个时间步的轨迹
        
        #对正在swing的腿继续执行轨迹
        if swing_continue.any():
            self.q_traj_index[swing_continue]+=1
        self.last_gaits = self.gaits.copy()


        #以上的保持不动是轨迹索引不动，执行是轨迹索引加一
        # for i in range(6):
        #     if not self.gaits[i] and adhesion_force[i] !=0:
        #         print(f"leg {self.leg_names[i]}, adhesion_force={adhesion_force[i]}")
        # print(f"adhesion={self.adhesions},contact_force={contact_force}")
        # if swing_dones.any():
            # print(f"contact force={contact_force}, gaits={self.gaits}")
        #     print(f"swing_dones={swing_dones}, q_traj_index={self.q_traj_index}, B_e_traj_len={self.B_e_traj_len}")
                #计算stance的足端在世界坐标系下的位置
        # stance_index = np.nonzero(self.gaits)[0]
        # print(f"stance W_e = {cur_se3 * self.chs._B2R(self.B_e_cur[:,stance_index],stance_index)}")

    def GetJointAngles(self,q_cur:np.ndarray,real_robot=False):
        #q_cur 6,3
        B_e_des = np.zeros((6,3))
        for i in range(6):
            B_e_des[i,:] = self.B_e_traj[i][:,self.q_traj_index[i]] #3,
            # print(f"B_e_traj[{i}].shape={self.B_e_traj[i].shape}, index={self.q_traj_index[i]}")

        #使用迟滞的奇异点判断，
        self._Singularity_Hysteresis(B_e_des,q_cur)        
        damp_inv_jac = self.kin.DampInvJac(q_cur,self.singularity_legs)
        delt_q = 65*(damp_inv_jac @ ((B_e_des-self.B_e_cur.T)[...,None])).squeeze(-1)
        self.q_des[:,0:3] = q_cur + delt_q*self.dt
        self.q_des[self.singularity_legs,0] = self.hold_q[self.singularity_legs]

        #观察奇异点和摆动保持情况
        # if self.singularity_legs.any():
            # print(f"leg {self.singularity_legs.nonzero()[0]} in singularity, hold_q is {self.hold_q[self.singularity_legs]}")
        
        # self.q_des[:,0:3] = q_cur.copy()
        # self.kin.InverseKin2(B_e_des,self.q_des)

        # 摆动腿如果已经规划出关节空间轨迹，则直接使用该轨迹，避免奇异点/分支切换点附近的IK跳变。
        for i in range(6):
            if self.q_traj[i] is not None:
                q_idx = min(int(self.q_traj_index[i]),self.q_traj[i].shape[1]-1)
                self.q_des[i,0:3] = self.q_traj[i][:,q_idx]
        #对于摆动腿部，只有当轨迹离开地面，才会控制吸盘角度
        self.q_des[:,3] = self._GetFootAngle(self.q_des,B_e_des.T,real_robot)

        # 根据运动学模型中的范围裁剪前三个关节：
        # - thigh 的范围因腿而异，shape 为 (6, 2)；
        # - knee / ankle 的范围由所有腿共用，shape 为 (2, 2)。
        # q_des 的第 4 列为 foot；它在 _GetFootAngle 中已被限制在 [-pi/2, pi/2]。
        self.q_des[:, 0] = np.clip(
            self.q_des[:, 0],
            self.kin.joints_limits_thigh[:, 0],
            self.kin.joints_limits_thigh[:, 1],
        )
        self.q_des[:, 1:3] = np.clip(
            self.q_des[:, 1:3],
            self.kin.joints_limits_rest[None, :, 0],
            self.kin.joints_limits_rest[None, :, 1],
        )

    #使用迟滞区间判断腿部是否进入r_xy奇异区间
    def _Singularity_Hysteresis(self,B_e:np.ndarray,q_cur:np.ndarray):
        """
        @input B_e: 6,3, q_cur: 6,3
        修改self.singularity_legs,进入奇异区间为1，离开为0，记录刚进入奇异区间时的关节0角度在hold_q中
        """
        r_xy = np.linalg.norm(B_e[:,:2],axis=1)
        #形成x 到 x+0.005的奇异点缓冲区间，避免来回跳变
        enter_sing_mask = r_xy<self.singularity_threshold
        exit_sing_mask = r_xy>self.singularity_threshold+0.005
            #对第一次进入奇异值区间的关节0的位置进行记录
        first_enter_index =  np.where((~self.singularity_legs)&enter_sing_mask)[0]
        if len(first_enter_index)>0:
            self.hold_q[first_enter_index] = q_cur[first_enter_index,0]
        self.singularity_legs[enter_sing_mask] = True
        self.singularity_legs[exit_sing_mask] = False            

    def _GetFootAngle(self,q:np.ndarray,B_e:np.ndarray,real_robot)->np.ndarray:
        """
        @input q: 6,3, 当前/期望的关节角度, B_e 3,6 在B系下足端当前/期望的位置:
        @output foot_angle:6
        调用前需保证已经计算好了落脚点的法向量
        """
        """仿真中不进行区分"""
        """实物中区分支撑和摆动状态"""
        #以下返回测试用
        # return -(q[:,1]+q[:,2])-np.pi/2.0
        
        foot_angle = np.zeros((6))
        if real_robot:
            #对于支撑腿而言，舵机关节只需要保持被动跟随，不需要主动施加扭矩
            foot_angle[self.gaits]=-999
            #对于摆动腿而言，需要根据当前摆动腿的期望关节角度计算出末端关节的朝向，然后计算舵机应该旋转的角度
            # 以下计算在B系下进行
            active_index = np.nonzero(~self.gaits)[0]
        else:
            active_index = np.arange(6)

        if active_index.size > 0:
            #为了保证摆动过程中足端位置实时与落脚点法向量一致，需要调用
            # landing_points = self.chs._R2B(self.landing_point[:,active_index],active_index) #3,N
            B_e = B_e[:,active_index]
            # choose_support_mask = self.q_traj_index[active_index]<self.B_e_traj_len[active_index]/6.0
            allign_norms = self.B_landing_n[:,active_index].copy()
            # allign_norms[:,choose_support_mask] = self.B_support_n[:,active_index[choose_support_mask]].copy()
        

            # left_mask = active_index<3
            # allign_norms[0:2,left_mask] *= -1
            
            #1）计算落脚点法向量在p平面的投影（p是腿部基坐标系z轴和B_e组成的平面） parallel_norm
            n_p = np.vstack([-B_e[1,:],
                             B_e[0,:],
                             np.zeros(len(active_index))])
            n_p = n_p/np.maximum(np.linalg.norm(n_p,axis=0),0.00001) #平面法向量
            vertical_part = n_p*(np.sum(allign_norms*n_p,axis=0)[None,:])
            #按照这个顺序减出来的，是从landing_norm向量末端指向垂直分量的，代表了吸盘的x轴需要平行的方向
            parallel_part = vertical_part - allign_norms
            parallel_norm = parallel_part/np.maximum(np.linalg.norm(parallel_part,axis=0),0.00001)
            
            #2) 计算末端关节的x轴在B系下的法向量 x_b3
            x_b3 = self.kin.Get_X_B3(q[active_index,0:3]).T
            #3）计算两者点乘，小于0意味着超出旋转范围，需要警告
            cos_q4 = np.clip(np.sum(parallel_norm*x_b3,axis=0), -1, 1)
            if (cos_q4<0).any():
                warnings.warn("Norm of landing points may exceed the range of suction cup")
                # print(f"x_b3={x_b3}, B_norm={allign_norms}")
            #4）x_b3 cross parallel_norm 与末端关节z_b3轴是否在一个方向，在的话旋转的角度为正，否则为负
            rotate_vector = (np.cross(x_b3.T,parallel_norm.T).T)
            z_b3 = self.kin.Get_Z_B3(q[active_index,0:3]).T
            rotate_direction = np.sign( np.sum(rotate_vector*z_b3,axis=0) )
            foot_angle[active_index] = np.clip(np.arccos(cos_q4) * rotate_direction,-pi/2,pi/2)
        # print(f"foot angle={foot_angle}")
        return foot_angle

    def _WrapAngleError(self,target:np.ndarray,current:np.ndarray)->np.ndarray:
        """计算关节角最短方向误差。"""
        return (target-current+np.pi)%(2.0*np.pi)-np.pi

    def _QuinticInterp(self,p0,v0,a0,p1,v1,a1,T,t,wrap_angle=False):
        """通用五次多项式插值。

        @input p0/v0/a0/p1/v1/a1: (dim,)
        @input T: 总时长
        @input t: 标量或(N,)采样时间
        @input wrap_angle: True时按角度最短路径处理p1-p0
        @output 标量t返回(dim,)，数组t返回(dim,N)
        """
        T = max(float(T),1e-6)
        t_arr = np.asarray(t,dtype=np.float64)
        scalar = t_arr.ndim == 0
        t_arr = np.clip(t_arr.reshape(-1),0.0,T)
        p0 = np.asarray(p0,dtype=np.float64).reshape(-1)
        v0 = np.asarray(v0,dtype=np.float64).reshape(-1)
        a0 = np.asarray(a0,dtype=np.float64).reshape(-1)
        p1 = np.asarray(p1,dtype=np.float64).reshape(-1)
        v1 = np.asarray(v1,dtype=np.float64).reshape(-1)
        a1 = np.asarray(a1,dtype=np.float64).reshape(-1)
        if wrap_angle:
            p1 = p0 + self._WrapAngleError(p1,p0)

        c0 = p0
        c1 = v0
        c2 = 0.5*a0
        A = np.array([
            [T**3,    T**4,     T**5],
            [3*T**2,  4*T**3,   5*T**4],
            [6*T,    12*T**2,  20*T**3],
        ], dtype=np.float64)
        b = np.vstack([
            p1 - (c0 + c1*T + c2*T**2),
            v1 - (c1 + 2*c2*T),
            a1 - (2*c2),
        ])
        c3,c4,c5 = np.linalg.solve(A,b)
        traj = (
            c0[:,None]
            + c1[:,None]*t_arr[None,:]
            + c2[:,None]*t_arr[None,:]**2
            + c3[:,None]*t_arr[None,:]**3
            + c4[:,None]*t_arr[None,:]**4
            + c5[:,None]*t_arr[None,:]**5
        )
        if scalar:
            return traj[:,0].astype(np.float32)
        return traj.astype(np.float32)

    def _SolveIKAtBranch(self,B_point:np.ndarray,branch:int)->Union[np.ndarray,None]:
        """按指定分支求IK，失败时返回None。"""
        joints,mask = self.kin.InverseKin2MultiBatch(np.asarray(B_point,dtype=np.float32)[None,:])
        if mask[0,int(branch)]:
            return joints[0,int(branch)].astype(np.float32)
        return None

    def _SplineKinematicsAt(self,splines:List[BSpline],Times:np.ndarray,segment_id:int,u:float)->Tuple[np.ndarray,np.ndarray]:
        """由B样条直接计算足端速度和加速度。"""
        if segment_id < 0:
            return np.zeros(3,dtype=np.float32),np.zeros(3,dtype=np.float32)
        T = max(float(Times[int(segment_id)]),1e-6)
        u = float(np.clip(u,0.0,1.0))
        B_dot = splines[int(segment_id)].derivative(1)(u)/T
        B_ddot = splines[int(segment_id)].derivative(2)(u)/(T**2)
        return np.asarray(B_dot,dtype=np.float32),np.asarray(B_ddot,dtype=np.float32)

    def _JointKinematicsFromFoot(self,q:np.ndarray,B_dot:np.ndarray,B_ddot:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """用阻尼雅可比把足端速度/加速度映射到关节速度/加速度。"""
        in_singular = np.linalg.norm(self.kin.ForwardKinReturn(q)[:2]) < self.singularity_threshold + 0.01
        damp_inv = self.kin.DampInvJac(q[None,:],np.array([in_singular]))[0]
        q_dot = damp_inv @ B_dot
        jdot_qdot = self.kin.JacobianDotQdot(q[None,:],q_dot[None,:])[0] #q_dot^T H q_dot
        q_ddot = damp_inv @ (B_ddot-jdot_qdot)
        if in_singular:
            q_dot[0] = 0.0
            q_ddot[0] = 0.0
        return q_dot.astype(np.float32),q_ddot.astype(np.float32)

    def _BuildSwingQTraj(self,leg_index:int,B_traj:np.ndarray,splines:List[BSpline],Times:np.ndarray,
                         branch_indices:np.ndarray,segment_ids:np.ndarray,local_us:np.ndarray,
                         debug_info:bool=False)->Tuple[np.ndarray,dict]:
        """构造摆动腿关节轨迹；关键点附近使用五次Hermite插值，其余位置使用解析IK。"""
        N = B_traj.shape[1]
        transition_mask = np.zeros(N,dtype=np.bool_)
        intervals = []
        transition_radius = 0.03
        for spline_id in range(len(splines)-1):
            key_point = np.asarray(splines[spline_id](1.0),dtype=np.float32)
            interval_indices = np.where(np.linalg.norm(B_traj-key_point[:,None],axis=0)<=transition_radius)[0]
            if interval_indices.size > 1:
                start = int(interval_indices[0])
                end = int(interval_indices[-1])
                intervals.append((start,end))
                transition_mask[start:end+1] = True
                print("transition start=",start)
                print("transition end=",end)

        branch_indices = np.asarray(branch_indices,dtype=np.int32)
        joints_all,solution_mask = self.kin.InverseKin2MultiBatch(B_traj.T)
        selected_valid = solution_mask[np.arange(N),branch_indices]
        q_selected = joints_all[np.arange(N),branch_indices,:].astype(np.float32)
        q_traj = np.zeros((3,N),dtype=np.float32)

        valid_indices = np.nonzero(selected_valid)[0]
        if valid_indices.size > 0:
            last_q = q_selected[valid_indices[0]].copy()
        else:
            last_q = np.zeros(3,dtype=np.float32)

        ik_fail_count = 0
        for i in range(N):
            if selected_valid[i]:
                q_i = q_selected[i]
            else:
                if not transition_mask[i]:
                    ik_fail_count += 1
                # 指定branch无解析解时不再切换到另一branch，统一用上一帧关节角通过阻尼雅可比逼近期望足端。
                B_cur = self.kin.ForwardKinReturn(last_q)
                in_singular = np.linalg.norm(B_traj[:2,i]) < self.singularity_threshold + 0.01
                damp_inv = self.kin.DampInvJac(last_q[None,:],np.array([in_singular]))[0]
                delta_q = 45.0*self.dt*(damp_inv @ (B_traj[:,i]-B_cur))
                if in_singular:
                    delta_q[0] = 0.0
                q_i = (last_q+delta_q).astype(np.float32)
            q_traj[:,i] = q_i
            last_q = q_i

        for start,end in intervals:
            q0 = q_traj[:,start].copy()
            q1 = q_traj[:,end].copy()
            seg_id = segment_ids[start]
            B_dot0 = splines[seg_id].derivative(1)(local_us[start])/Times[seg_id]
            B_ddot0 = splines[seg_id].derivative(2)(local_us[start])/Times[seg_id]**2
            seg_id = segment_ids[end]
            B_dot1 = splines[seg_id].derivative(1)(local_us[end])/Times[seg_id]
            B_ddot1 = splines[seg_id].derivative(2)(local_us[end])/Times[seg_id]**2
            qd0,qdd0 = self._JointKinematicsFromFoot(q0,B_dot0,B_ddot0)
            qd1,qdd1 = self._JointKinematicsFromFoot(q1,B_dot1,B_ddot1)
            duration = max((end-start)*self.dt,self.dt)
            t = np.arange(end-start+1,dtype=np.float32)*self.dt
            q_interval = self._QuinticInterp(q0,qd0,qdd0,q1,qd1,qdd1,duration,t,wrap_angle=True).T
            q_traj[:,start:end+1] = q_interval.astype(np.float32).T

        q_debug_info = {}
        if debug_info:
            no_selected_solution = ~selected_valid
            other_valid = solution_mask[np.arange(N),1-branch_indices]
            selected_branch_fail_mask = no_selected_solution & other_valid & (~transition_mask)
            all_branch_fail_mask = no_selected_solution & (~solution_mask.any(axis=1)) & (~transition_mask)
            transition_fail_mask = no_selected_solution & transition_mask

            if selected_branch_fail_mask.any():
                warnings.warn(
                    f"leg {leg_index} swing q_traj has {int(selected_branch_fail_mask.sum())} "
                    "non-transition points without solution in the selected branch; used damped IK fallback"
                )
            if all_branch_fail_mask.any():
                warnings.warn(
                    f"leg {leg_index} swing q_traj has {int(all_branch_fail_mask.sum())} "
                    "non-transition points without any IK branch solution; used damped IK fallback"
                )
            if transition_fail_mask.any():
                warnings.warn(
                    f"leg {leg_index} swing q_traj has {int(transition_fail_mask.sum())} "
                    "transition points without solution in the selected branch"
                )
            if ik_fail_count > 0:
                warnings.warn(f"leg {leg_index} swing q_traj IK fallback at {ik_fail_count} non-transition points")

            q_debug_info = {
                "ik_selected_branch_fail_mask": selected_branch_fail_mask.copy(),
                "ik_all_branch_fail_mask": all_branch_fail_mask.copy(),
                "ik_transition_fail_mask": transition_fail_mask.copy(),
                "ik_transition_mask": transition_mask.copy(),
                "ik_selected_valid_mask": selected_valid.copy(),
                "ik_solution_mask": solution_mask.copy(),
                "ik_branch_indices": branch_indices.copy(),
            }
        q_limits = np.vstack([
            self.kin.joints_limits_thigh[leg_index],
            self.kin.joints_limits_rest,
        ])
        q_traj = np.clip(q_traj,q_limits[:,0,None],q_limits[:,1,None]).astype(np.float32)
        return q_traj,q_debug_info

    def _GetCloestIndex(self,cur_se3:SE3)->int:
        if self.path_se3 is None or len(self.path_se3) == 0:
            raise ValueError("path_se3 is empty, please LoadSE3(path_se3) first")

        # cache path translations to avoid repeated extraction overhead
        if (not hasattr(self, "_path_positions")) or len(self._path_positions) != len(self.path_se3):
            self._path_positions = np.array([pose.t for pose in self.path_se3])

        cur_pos = np.asarray(cur_se3.t)
        diff = self._path_positions - cur_pos
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        index = np.argmin(dist_sq)
        index = np.minimum(index+1,len(self.path_se3)-1) #为了避免机器人往后退，尽量选择前面的目标作为轨迹
        return index
    
    def _ContactDetection(self,q_cur,q_torque)->np.ndarray:
        #先估算接触力,这是在B系下的足端对外界的作用力大小，由于电磁铁的z轴是指向连接他螺栓的，
        # 也就是z为负代表足端对环境产生压力，此时电磁铁可能接触到了吸附面
        # 当z为正的时候，代表电磁铁悬空，需要腿部把电磁铁拉着向上
        # (6,3) * (6,3) -> 6
        contact_force = np.sum(self.kin.CF_Estimate(q_torque,q_cur,self.singularity_legs) * self.B_landing_n.T,axis=1)
        self.contact_count[(contact_force <=-7)&(~self.gaits)&(self.q_traj_index>self.B_e_traj_len/3.0)] += 1
        # print(f"contac_force={np.round(contact_force,2)} count={self.contact_count} gaits={self.gaits}")
        #接触力  & 执行超过一半轨迹（避免stance刚切换为swing时就判定接触导致摆动结束）
        # print(f" contact_force={contact_force} gaits={self.gaits}")
        # contact_mask =  (self.contact_count>=10) & (~self.gaits) & (self.q_traj_index>self.B_e_traj_len/4.0)
        contact_mask = self.contact_count>=10
        self.contact_count[contact_mask] = 0
        self.B_support_n[:,contact_mask] = self.B_landing_n[:,contact_mask]
        return contact_mask
    
    #循环一步一步计算支撑轨迹的算法
    def _SE3Interp_and_StanceCal(self,cur_se3:SE3):
        #从当前位姿（不一定在路径上）先插值到路径上，再沿着路径计算
        self._ClearTraj(self.gaits)
        self.follow_path_index = 0
        self.interp_path_se3=[cur_se3]
        stance_index = self.gaits.nonzero()[0]
        path_se3,_ = self._Start2PathSE3Interp(cur_se3) #path_se3包括了固定时间内获取到的SE3轨迹
        R_e = self.kin._B2R(self.B_e_cur[:,stance_index],stance_index)
        # R_normals = self.B_landing_n[:,stance_index].copy().T #N,3
        R_normals = self.B_support_n[:,stance_index].copy().T #N,3
        R_normals[stance_index<3,:2] *= -1.0
        #这里是原来保存了一个 array 为了循环方便放入 改成 list[array]
        for i,index in enumerate(stance_index):
            self.B_e_traj[index] = [self.B_e_traj[index]]
        #第一个点是起点，clearTraj时已经放入了，这里从第二个se3开始
        for se3 in path_se3[1:]:
            #cur_se3 W_T_R se3: W_T_Rt
            #B_e_init B_e
            #R_e_init R_e
            #Rt_e = (Rt_T_W*W_T_R)*R_e
            Rt_T_R = se3.inv() * cur_se3
            # Rt_e = Rt_T_R*self.chs._B2R(self.B_e_init[:,stance_index],stance_index)
            Rt_e = Rt_T_R*R_e 
            if self.hex_state.PointsFeasiCheck(se3,Rt_e.T,R_normals,stance_index,self.q3_branches[stance_index]):
                self.interp_path_se3.append(se3)
                Bt_e = self.kin._R2B(Rt_e,stance_index)
                for i,index in enumerate(stance_index):
                    self.B_e_traj[index].append(Bt_e[:,i])
            else:
                break

        for i,index in enumerate(stance_index):
            self.B_e_traj[index] = np.column_stack(self.B_e_traj[index])
        self.B_e_traj_len[self.gaits] = len(self.interp_path_se3)
    
    def _SE3Follow_and_StanceCal(self,cur_se3:SE3):
        #这是加入已经计算好了支撑轨迹，正在执行中加入了新的支撑腿，只需要沿着原来插值好的继续计算就行
        #重要假设：新加入的stance腿在旧的stance到达极限之前依旧可行
        # self.interp_path_se3
        stance_index = self.gaits.nonzero()[0]
        self._ClearTraj(self.gaits)
        for i,index in enumerate(stance_index):
            self.B_e_traj[index] = [self.B_e_traj[index]]

        # cur_follow_index = self.q_traj_index[self.gaits][0] #q_traj_index
        R_e = self.kin._B2R(self.B_e_cur[:,stance_index],stance_index)
        # R_normals = self.B_landing_n[:,stance_index].copy().T #N,3
        R_normals = self.B_support_n[:,stance_index].copy().T #N,3
        R_normals[stance_index<3,:2] *= -1.0
        for i in range(self.follow_path_index,len(self.interp_path_se3)):
            se3 = self.interp_path_se3[i]
            Rt_T_R = se3.inv() * cur_se3
            Rt_e = Rt_T_R * R_e
            if self.hex_state.PointsFeasiCheck(se3,Rt_e.T,R_normals,stance_index,self.q3_branches[stance_index]):
                Bt_e = self.kin._R2B(Rt_e,stance_index)
                for i,index in enumerate(stance_index):
                    self.B_e_traj[index].append(Bt_e[:,i])
            else:
                break
        for i,index in enumerate(stance_index):
            self.B_e_traj[index] = np.column_stack(self.B_e_traj[index])
        self.B_e_traj_len[self.gaits] = self.B_e_traj[stance_index[0]].shape[1]

    def _TimeDist(self,T1:SE3,T2:SE3,ceil_return=True):
        #分别计算直线距离和角度距离在最大速度下需要多少个dt，返回需要最多dt的比例
        td_dist = (np.linalg.norm(T1.t-T2.t)/self.v_max) / self.dt
        td_angle = (T1.angdist(T2)/self.w_max) / self.dt
        # td_angle = (np.arccos( (np.trace(T1.R.T @ T2.R)-1)/2.0 ) /self.w_max) /self.dt
        if ceil_return:
            return ceil (max(td_dist,td_angle)) #返回整数倍数
        else:
            return max(td_dist,td_angle) #返回实际倍数

    def _Start2PathSE3Interp(self,start_se3:SE3,max_t=None,max_w=None)->Tuple[List[SE3],List[SE3]]:
        """以最大移动距离或角度为限制 按照最大移动旋转速度计算从当前位姿到路径se3的插值路径
        @input start_se3(SE3)起始插值的位姿; 
        max_t(float,m)最大容许移动距离,设置为None时选取self._stance_maxmove_t
        max_w(float,rad)最大容许移动角度,设置为None时选取self._stance_maxmove_w

        @output interp_path_se3(经过严格等速插值得到的) path_se3(从原始轨迹中获取待插值的点)
        """
        if max_t is None:
            max_t = self._stance_maxmove_t
        if max_w is None:
            max_w = self._stance_maxmove_w

        start_index = self._GetCloestIndex(start_se3)
        interp_max_num = max( ceil(max_t/(self.v_max*self.dt)) , ceil(max_w/(self.w_max*self.dt)) )
        interp_num = 0
        interp_start_se3 = start_se3
        #path_se3是间隔较大的SE3轨迹点
        path_se3 = [interp_start_se3]
        t_key=[0]
        if (np.linalg.norm(start_se3.t-self.path_se3[-1].t)<=0.01) & (start_se3.angdist(self.path_se3[-1])<=radians(5.0)):
            print(">>>>>>>>>> In _Start2PathSE3Interp, the start pose reach the destination, cancle calculation")
            return path_se3, path_se3
        #先获取当前轨迹的时刻与对应的SE3，t_key=[0,23.45dt, 50.67dt, 113,5dt, ...] path_se3=[se3_1, se3_2, se3_3, se3_4, ...]
        for i in range(start_index,len(self.path_se3)):
            if interp_num >= interp_max_num:
                break
            nums_dt = self._TimeDist(interp_start_se3,self.path_se3[i],ceil_return=False)
            if nums_dt<1e-4:
                continue
            if interp_num + nums_dt >= interp_max_num:
                cur_ratio = (interp_max_num - interp_num)/nums_dt
                t_key.append(interp_max_num)
                path_se3.append(interp_start_se3.interp(self.path_se3[i],cur_ratio))
                break
            interp_num += nums_dt
            t_key.append(interp_num)
            path_se3.append(self.path_se3[i])
            interp_start_se3 = self.path_se3[i]
        if len(path_se3)<2:
            return [],path_se3
        #采用五次连续查询
        # _tau = np.linspace(0,1.0,ceil(t_key[-1])+1)
        # _s = 10*_tau**3 - 15*_tau**4 +6*_tau**5
        # t_query = _s*ceil(t_key[-1])
        # interp_path_se3 = []
        # seg_idx = 0
        # for t in t_query:
        #     while seg_idx<len(path_se3)-2 and t>t_key[seg_idx+1]:
        #         seg_idx += 1
        #     t0=t_key[seg_idx]
        #     t1=t_key[seg_idx+1]
        #     ratio=np.clip((t-t0)/(t1-t0),0,1)
        #     interp_path_se3.append(path_se3[seg_idx].interp(path_se3[seg_idx+1],ratio))            
        #interp_path_se3是等速间隔的位姿点
        interp_path_se3=[]
        seg_idx = 0
        #再使用t_query=[0,dt,2dt,3dt,...]查询对应的SE3位姿，线性插值后放入interp_path_se3中
        for tq in np.arange( 0, ceil(t_key[-1])+1 ):
            while seg_idx<len(path_se3)-2 and tq>t_key[seg_idx+1]:
                seg_idx += 1
            t0=t_key[seg_idx]
            t1=t_key[seg_idx+1]
            ratio=np.clip((tq-t0)/(t1-t0),0,1)
            interp_path_se3.append(path_se3[seg_idx].interp(path_se3[seg_idx+1],ratio))
        return interp_path_se3, path_se3

    #足端位置空间中的轨迹优化
    def _SwingTrajCal(self,target_se3:SE3,debug_mode=False,debug_inputs=None):
        """
        输入target_se3，按照足端空间中的分段B样条逻辑计算摆动腿轨迹。
        目前设定2s内完成摆动轨迹
        """
        if len(self.B_e_traj) != 6:
            self.B_e_traj = [[self.B_e_cur[:,i].copy()] for i in range(6)]
        self._ClearTraj(~self.gaits)
        swing_index = (~self.gaits).nonzero()[0]
        stance_index = self.gaits.nonzero()[0]

        def _Bezier4(c0,c1,c2,c3)->BSpline:
            knot = np.array([0,0,0,0,1,1,1,1],dtype=np.float32)
            return BSpline(knot,np.vstack([c0,c1,c2,c3]),3)

        def _Bspline5(c0,c1,c2,c3,c4)->BSpline:
            knot = np.array([0,0,0,0,0.5,1,1,1,1],dtype=np.float32)
            return BSpline(knot,np.vstack([c0,c1,c2,c3,c4]),3)
        
        def _Bspline6(c0,c1,c2,c3,c4,c5)->BSpline:
            knot = np.array([0,0,0,0,1/3,2/3,1,1,1,1])
            return BSpline(knot,np.vstack([c0,c1,c2,c3,c4,c5]),3)
        
        def _Bspline7(c0,c1,c2,c3,c4,c5,c6)->BSpline:
            knot = np.array([0,0,0,0,1/4,2/4,3/4,1,1,1,1])
            return BSpline(knot,np.vstack([c0,c1,c2,c3,c4,c5,c6]),3)        

        def _TimeByPolyline(points:List[np.ndarray])->float:
            length = 0.0
            for i in range(len(points)-1):
                length += np.linalg.norm(points[i+1]-points[i])
            return max(length/(self.v_max*2.0),self.dt)

        def _SampleSplines(splines:List[BSpline],Times:np.ndarray,dt:float,segment_branches:np.ndarray,cal_acc=True,return_meta=False):
            B_points = []
            acc_u_scaled = []
            sol_indices = []
            segment_ids = []
            local_us = []
            smooth_cost = 0.0
            for idx,b_spline in enumerate(splines):
                u = np.linspace(0,1,ceil(Times[idx]/dt)+1)
                if idx > 0:
                    u = u[1:]
                B_points.append(b_spline(u).T)
                sol_indices.append(np.full(u.shape,segment_branches[idx],dtype=np.int32))
                segment_ids.append(np.full(u.shape,idx,dtype=np.int32))
                local_us.append(u.astype(np.float32))
                if cal_acc:
                    #后面积分会用到acc**2/T**3 这里提前计算好acc/T**1.5 方便后续计算
                    acc_i = b_spline.derivative(2)(u).T/(Times[idx]**1.5)
                    acc_u_scaled.append(acc_i)
                    smooth_cost += self.trapezoid(np.sum(acc_i**2,axis=0),u)
            if return_meta:
                meta = (np.hstack(segment_ids),np.hstack(local_us))
                if cal_acc:
                    return np.hstack(B_points),np.hstack(acc_u_scaled),np.hstack(sol_indices),meta,smooth_cost
                return np.hstack(B_points),None,np.hstack(sol_indices),meta
            if cal_acc:
                return np.hstack(B_points),np.hstack(acc_u_scaled),np.hstack(sol_indices),smooth_cost
            return np.hstack(B_points),None,np.hstack(sol_indices)

        def _SplineControlPoints(splines:List[BSpline])->List[np.ndarray]:
            """提取每段B样条控制点，便于debug可视化。"""
            return [np.asarray(spline.c,dtype=np.float32).T.copy() for spline in splines]

        def _PlanOneLeg(leg_index:int,start_point:np.ndarray,landing_point:np.ndarray,
                        landing_q3_branch:int,debug_info=False,debug_cfg=None):
            start_point = np.asarray(start_point,dtype=np.float32)
            landing_point = np.asarray(landing_point,dtype=np.float32)
            start_q3_branch = int(self.q3_branches[leg_index])
            start_normal = self.B_support_n[:,leg_index]
            landing_normal = self.B_landing_n[:,leg_index]
            raise_point,raise_dist_init = self.hex_state.FarestPoints(start_point,start_normal,target_se3,leg_index,distance=0.06)
            down_point, down_dist_init = self.hex_state.FarestPoints(landing_point,landing_normal,target_se3,leg_index,distance=0.06)
            terminal_point,_ = self.hex_state.FarestPoints(landing_point,-landing_normal,target_se3,leg_index,ignore_end=True)
            raise_dist_min = min(0.04,float(raise_dist_init))
            down_dist_min = min(0.04,float(down_dist_init))
            self.landing_point[:,leg_index] = landing_point
            change_branch = start_q3_branch != landing_q3_branch
            pass_singular = start_point[0]*landing_point[0] <= 0

            if debug_info:
                self._last_swing_debug_info = {
                    "leg_index": leg_index,
                    "start_point": start_point.copy(),
                    "landing_point": landing_point.copy(),
                    "start_normal": start_normal.copy(),
                    "landing_normal": landing_normal.copy(),
                    "start_q3_branch": start_q3_branch,
                    "landing_q3_branch": landing_q3_branch,
                    "change_branch": bool(change_branch),
                    "pass_singular": bool(pass_singular),
                }
            debug_cfg = debug_cfg if isinstance(debug_cfg,dict) else {}

            # 不同关键点使用不同优化变量：普通点优化xyz，奇异点只优化z，切换分支点优化(q1,q2)后用FK生成xyz。
            if change_branch:
                theta = atan2(landing_point[1]+start_point[1],landing_point[0]+start_point[0])
                if theta > pi/2.0:
                    theta = theta-pi
                elif theta < -pi/2.0:
                    theta = theta+pi
                branch_q = np.array([theta,0.0],dtype=np.float32)
                if "branch_q_init" in debug_cfg:
                    branch_q = np.asarray(debug_cfg["branch_q_init"],dtype=np.float32).reshape(2,)

            singular_init = np.array([debug_cfg.get("singular_z_init",self.B_singular[2])],dtype=np.float32)
            if change_branch:
                branch0 = self.kin.ForwardKinReturn(np.array([branch_q[0],branch_q[1],0.0],dtype=np.float32)).astype(np.float32)
            if pass_singular or (change_branch and start_point[0] < 0):
                singular0 = np.array([0.0,0.0,singular_init[0]],dtype=np.float32)
            
            if (change_branch and start_point[0]<0):
                case_type = "singular_branch"
                # 变量顺序: raise_dist, ctrl1, singular_z, ctrl3, ctrl4, ctrl5, branch_q, ctrl7, down_dist
                initial_conditions = [
                    np.array([raise_dist_init],dtype=np.float32),   #0
                    0.5*(raise_point+singular0),                    #1:4
                    singular_init,                                  #4
                    2/4*singular0+2/4*branch0,                      #5:8
                    1/4*singular0 + 3/4*branch0,                    #8:11
                    branch0.copy(),                                 #11:14
                    branch_q,                                       #14:16
                    0.5*(branch0.copy()+down_point),                #16:19
                    np.array([down_dist_init],dtype=np.float32),    #19
                ]
                bounds = [(raise_dist_min,0.1)] + [(None,None)]*3 + [(-0.27,-0.075)] + [(None,None)]*14 + [(down_dist_min,0.1)]
                Times = np.array([
                    _TimeByPolyline([start_point,raise_point,singular0]),
                    _TimeByPolyline([singular0,branch0]),
                    _TimeByPolyline([branch0,down_point,landing_point]),
                ],dtype=np.float32)
                branch_schema = np.array([start_q3_branch,start_q3_branch,landing_q3_branch],dtype=np.int32)
            elif change_branch and pass_singular and start_point[0] > 0:
                case_type = "branch_singular"
                # 变量顺序: raise_dist, ctrl1, ctrl2, branch_q, ctrl4, ctrl5, ctrl6, singular_z, down_dist
                initial_conditions = [
                    np.array([raise_dist_init],dtype=np.float32),   #0
                    0.5*(branch0.copy()+start_point),               #1:4
                    branch0.copy(),                                 #4:7
                    branch_q,                                       #7:9
                    (3.0/4.0)*branch0+(1.0/4.0)*singular0,          #9:12
                    (2.0/4.0)*branch0+(2.0/4.0)*singular0,          #12:15
                    (1/4)*branch0 + (3/4)*singular0,                #15:18
                    singular_init,                                  #18
                    np.array([down_dist_init],dtype=np.float32),    #19
                ]
                bounds = [(raise_dist_min,0.1)] + [(None,None)]*17 + [(-0.27,-0.075),(down_dist_min,0.1)]
                Times = np.array([
                    _TimeByPolyline([start_point,raise_point,branch0]),
                    _TimeByPolyline([branch0,singular0]),
                    _TimeByPolyline([singular0,down_point,landing_point]),
                ],dtype=np.float32)
                branch_schema = np.array([start_q3_branch,landing_q3_branch,landing_q3_branch],dtype=np.int32)
            elif pass_singular:
                case_type = "singular"
                # 变量顺序: raise_dist, ctrl1, ctrl2, singular_z, ctrl4, down_dist
                initial_conditions = [
                    np.array([raise_dist_init],dtype=np.float32),   #0
                    2/3*raise_point+1/3*singular0,                  #1:4
                    1/3*raise_point+2/3*singular0,                  #4:7
                    singular_init,                                  #7
                    1/3*singular0+2/3*down_point,                   #8:11
                    np.array([down_dist_init],dtype=np.float32),    #11
                ]
                bounds = [(raise_dist_min,0.1)] + [(None,None)]*6 + [(-0.27,-0.075)] +[(None,None)]*3+[(down_dist_min,0.1)]
                Times = np.array([
                    _TimeByPolyline([start_point,raise_point,singular0]),
                    _TimeByPolyline([singular0,down_point,landing_point]),
                ],dtype=np.float32)
                branch_schema = np.array([start_q3_branch,start_q3_branch],dtype=np.int32)
            elif change_branch:
                case_type = "branch"
                # 变量顺序: raise_dist, ctrl1, ctrl2,branch_q, ctrl4,down_dist
                initial_conditions = [
                    np.array([raise_dist_init],dtype=np.float32),   #0
                    0.5*(raise_point+branch0),                      #1:4
                    branch0.copy(),                                 #4:7
                    branch_q,                                       #7:9
                    0.5*(branch0+down_point),                       #9:12
                    np.array([down_dist_init],dtype=np.float32),    #12
                ]
                bounds = [(raise_dist_min,0.1)] + [(None,None)]*11 + [(down_dist_min,0.1)]
                Times = np.array([
                    _TimeByPolyline([start_point,raise_point,branch0]),
                    _TimeByPolyline([branch0,down_point,landing_point]),
                ],dtype=np.float32)
                branch_schema = np.array([start_q3_branch,landing_q3_branch],dtype=np.int32)
            else:
                case_type = "normal"
                # 变量顺序: raise_dist, ctrl1, ctrl2, down_dist
                initial_conditions = [
                    np.array([raise_dist_init],dtype=np.float32),   #0
                    2/3*raise_point+1/3*down_point,                 #1:4
                    1/3*raise_point+2/3*down_point,                 #4:7
                    np.array([down_dist_init],dtype=np.float32),    #7
                ]
                bounds = [(raise_dist_min,0.1)] + [(None,None)]*6 + [(down_dist_min,0.1)]
                Times = np.array([_TimeByPolyline([start_point,raise_point,down_point,landing_point])],dtype=np.float32)
                branch_schema = np.array([start_q3_branch],dtype=np.int32)

            initial_conditions = np.hstack(initial_conditions).astype(np.float32)

            def _BuildSplines(variables:np.ndarray):
                raise_ctrl = start_point + variables[0]*start_normal
                if case_type == "singular_branch":
                    ctrl1 = variables[1:4]
                    singular = np.array([0.0,0.0,variables[4]],dtype=np.float32)
                    ctrl3 = variables[5:8]
                    ctrl4 = variables[8:11]
                    ctrl5 = variables[11:14]
                    branch = self.kin.ForwardKinReturn(np.array([variables[14],variables[15],0.0],dtype=np.float32))
                    ctrl7 = variables[16:19]
                    down_ctrl = landing_point + variables[19]*landing_normal
                    ctrl2 = singular + 1/3*(Times[1]/Times[0])*(singular-ctrl1)
                    ctrl6 = branch + 3/2*(Times[2]/Times[1])*(branch-ctrl5)
                    return [
                        _Bezier4(start_point,raise_ctrl,ctrl1,singular),
                        _Bspline6(singular,ctrl2,ctrl3,ctrl4,ctrl5,branch),
                        _Bspline6(branch,ctrl6,ctrl7,down_ctrl,landing_point,landing_point), #确保末端速度为0
                    ],Times,branch_schema
                if case_type == "branch_singular":
                    ctrl1 = variables[1:4]
                    ctrl2 = variables[4:7]
                    branch = self.kin.ForwardKinReturn(np.array([variables[7],variables[8],0.0],dtype=np.float32))
                    ctrl4 = variables[9:12]
                    ctrl5 = variables[12:15]
                    ctrl6 = variables[15:18]
                    singular = np.array([0.0,0.0,variables[18]],dtype=np.float32)
                    down_ctrl = landing_point + variables[19]*landing_normal
                    ctrl3 = branch + 2/3*(Times[1]/Times[0])*(branch-ctrl2)
                    ctrl7 = singular + 3.0*(Times[2]/Times[1])*(singular-ctrl6)
                    return [
                        _Bspline5(start_point,raise_ctrl,ctrl1,ctrl2,branch),
                        _Bspline6(branch,ctrl3,ctrl4,ctrl5,ctrl6,singular),
                        _Bspline5(singular,ctrl7,down_ctrl,landing_point,landing_point),
                    ],Times,branch_schema
                if case_type == "singular":
                    ctrl1 = variables[1:4]
                    ctrl2 = variables[4:7]
                    singular = np.array([0.0,0.0,variables[7]],dtype=np.float32)
                    ctrl4 = variables[8:11]
                    down_ctrl = landing_point + variables[11]*landing_normal
                    ctrl3 = singular + (Times[1]/Times[0])*(singular-ctrl2)
                    return [
                        _Bspline5(start_point,raise_ctrl,ctrl1,ctrl2,singular),
                        _Bspline6(singular,ctrl3,ctrl4,down_ctrl,landing_point,landing_point),
                    ],Times,branch_schema
                if case_type == "branch":
                    ctrl1 = variables[1:4]
                    ctrl2 = variables[4:7]
                    branch = self.kin.ForwardKinReturn(np.array([variables[7],variables[8],0.0],dtype=np.float32))
                    ctrl3 = branch + (Times[1]/Times[0])*(branch-ctrl2)
                    ctrl4 = variables[9:12]
                    down_ctrl = landing_point + variables[12]*landing_normal
                    return [
                        _Bspline5(start_point,raise_ctrl,ctrl1,ctrl2,branch),
                        _Bspline6(branch,ctrl3,ctrl4,down_ctrl,landing_point,landing_point),
                    ],Times,branch_schema
                ctrl1 = variables[1:4]
                ctrl2 = variables[4:7]
                down_ctrl = landing_point + variables[7]*landing_normal
                return [
                    _Bspline7(start_point,raise_ctrl,ctrl1,ctrl2,down_ctrl,landing_point,landing_point),
                ],Times,branch_schema


            def cost_func(variables:np.ndarray):
                splines,Times,segment_branches = _BuildSplines(variables)
                B_points,ddot,sol_indices,smooth_cost = _SampleSplines(splines,Times,self.dt,segment_branches,cal_acc=True)
                return self._Cal_cost(B_points,ddot,None,leg_index,target_se3,sol_indices,smooth_cost=smooth_cost)

            optimized_c_points = minimize(cost_func,initial_conditions,method='L-BFGS-B',
                                          bounds=bounds,
                                          options={"maxiter":50,"ftol":1e-4,"gtol":1e-4})
            splines,Times,segment_branches = _BuildSplines(optimized_c_points.x)
            #根据 stance 的长度,stance结束时，swing也到达landing point
            self._swing_t = np.clip(len(self.interp_path_se3)*self.dt,a_min=2.5,a_max=None)
            Times = Times*(self._swing_t/np.sum(Times))
            
            spline_points,_,spline_branch_indices,spline_meta = _SampleSplines(
                splines,Times,self.dt,segment_branches,cal_acc=False,return_meta=True
            )
            spline_segment_ids,spline_local_us = spline_meta

            down2terminal_points = self._QuinticPolyInterp(
                landing_point,
                splines[-1].derivative(1)(1)/Times[-1],
                # splines[-1].derivative(2)(1)/(Times[-1]**2), #初始加速度加入后曲线会变扭曲无法保证直线
                0,
                terminal_point,
                np.zeros((3,)),
                np.zeros((3,)),
                np.linalg.norm(landing_point-terminal_point)/(0.5*self.v_max),
                self.dt,
            )
            B_traj = np.hstack([spline_points[:,:-1],down2terminal_points])
            B_traj_branch_indices = np.hstack([
                spline_branch_indices[:-1],
                np.full(down2terminal_points.shape[1],landing_q3_branch,dtype=np.int32),
            ])
            B_traj_segment_ids = np.hstack([
                spline_segment_ids[:-1],
                np.full(down2terminal_points.shape[1],-1,dtype=np.int32),
            ])
            B_traj_local_us = np.hstack([
                spline_local_us[:-1],
                np.zeros(down2terminal_points.shape[1],dtype=np.float32),
            ])
            self.B_e_traj[leg_index] = B_traj
            self.B_e_traj_len[leg_index] = B_traj.shape[1]
            #_BuildSwingQTraj是把所有轨迹点计算逆运动学求解
            self.q_traj[leg_index],swing_q_debug_info = self._BuildSwingQTraj(
                leg_index,B_traj,splines,Times,B_traj_branch_indices,B_traj_segment_ids,B_traj_local_us,
                debug_info=debug_info
            )
            self.q_traj_index[leg_index] = 0
            self.q3_branches[leg_index] = landing_q3_branch

            if debug_info:
                initial_debug_splines,initial_debug_times,initial_debug_branches = _BuildSplines(initial_conditions)
                initial_debug_points,_,initial_debug_branch_indices = _SampleSplines(
                    initial_debug_splines,initial_debug_times,self.dt,initial_debug_branches,cal_acc=False
                )
                optimized_debug_control_points = _SplineControlPoints(splines)
                initial_debug_control_points = _SplineControlPoints(initial_debug_splines)                
                self._last_swing_debug_info["initial_spline_points"] = initial_debug_points.astype(np.float32)
                self._last_swing_debug_info["initial_spline_branch_indices"] = initial_debug_branch_indices.astype(np.int32)
                self._last_swing_debug_info["initial_spline_control_points"] = initial_debug_control_points
                self._last_swing_debug_info["optimized_spline_points"] = spline_points.astype(np.float32)
                self._last_swing_debug_info["optimized_spline_branch_indices"] = spline_branch_indices.astype(np.int32)
                self._last_swing_debug_info["optimized_spline_control_points"] = optimized_debug_control_points
                self._last_swing_debug_info["opt_success"] = bool(optimized_c_points.success)
                self._last_swing_debug_info["opt_fun"] = float(optimized_c_points.fun)
                optimized_values = np.asarray(optimized_c_points.x,dtype=np.float32).copy()
                self._last_swing_debug_info["opt_case_type"] = case_type
                self._last_swing_debug_info["opt_initial_values"] = initial_conditions.copy()
                self._last_swing_debug_info["opt_optimized_values"] = optimized_values
                self._last_swing_debug_info["opt_delta_values"] = optimized_values-initial_conditions
                self._last_swing_debug_info["swing_q_debug_info"] = swing_q_debug_info

        if debug_mode:
            debug_inputs = {} if debug_inputs is None else debug_inputs
            if swing_index.size == 0 and "leg_index" not in debug_inputs:
                raise ValueError("debug_inputs must provide leg_index when there is no swing leg")
            leg_index = int(debug_inputs.get("leg_index",swing_index[0]))
            start_point = np.asarray(debug_inputs.get("start_point",self.B_e_cur[:,leg_index]),dtype=np.float32)
            landing_point = np.asarray(debug_inputs.get("landing_point",np.array([0.25,-0.1,0.12],dtype=np.float32)),dtype=np.float32)
            default_normal = np.array([0.0,0.0,1.0],dtype=np.float32)
            start_normal = np.asarray(debug_inputs.get("start_normal",default_normal),dtype=np.float32)
            landing_normal = np.asarray(debug_inputs.get("landing_normal",default_normal),dtype=np.float32)
            start_q3_branch = int(debug_inputs.get("start_q3_branch",self.q3_branches[leg_index]))
            landing_q3_branch = int(debug_inputs.get("landing_q3_branch",(start_q3_branch+1)%2))
            self.q3_branches[leg_index] = start_q3_branch
            self.B_e_cur[:,leg_index] = start_point
            self.B_support_n[:,leg_index] = start_normal
            self.B_landing_n[:,leg_index] = landing_normal
            _PlanOneLeg(
                leg_index,start_point,landing_point,landing_q3_branch,
                debug_info=True,debug_cfg=debug_inputs
            )
            return

        _,lp_idx,body_cf_masks,lp_mask = self.hex_state.RobotFeasiCheck(target_se3)
        if (~body_cf_masks).any():
            print(">>>>>>>>>> In Swing Traj Cal, body collisions occurs sum(body_collision_mask)=",sum(~body_cf_masks))
        future_path_se3,_ = self._Start2PathSE3Interp(target_se3)
        if not len(future_path_se3)>1:
            for leg_index in swing_index:
                self.B_e_traj[leg_index] = np.column_stack(self.B_e_traj[leg_index])
                self.B_e_traj_len[leg_index] = self.B_e_traj[leg_index].shape[1]
            return
        stance_points = np.column_stack([self.B_e_traj[leg_index][:,-1] for leg_index in stance_index])
        stance_points = self.kin._B2R(stance_points,stance_index).T

        for i in range(len(future_path_se3)-1,-1,-20):
            _,future_lp_idx,_,future_lp_mask = self.hex_state.RobotFeasiCheck(future_path_se3[i],lp_idx)
            future_lp_mask = future_lp_mask&lp_mask
            swing_has_lps_mask = future_lp_mask.any(axis=-1)
            if not (np.sum(swing_has_lps_mask[swing_index],axis=1)>=1).all():
                continue
            elif i<20:
                print(">>>>>>> In Swing Traj Cal, there are less than one possible landing points in future pose")
                print(f"future pose={future_path_se3[i]}\n landing points\n{self.hex_state.env_pointsmap_voxels.landing_points[future_lp_idx]}")
                
            for leg_index in swing_index:
                pointmap = self.hex_state.env_pointsmap_voxels
                possible_points = (target_se3.inv() * (pointmap.landing_points[future_lp_idx[swing_has_lps_mask[leg_index]]].T)).T
                mask = self.hex_state._LegLegCollisionFree(
                    possible_points,stance_points,self.q3_branches[stance_index],leg_index,stance_index
                )
                future_lp_mask[leg_index,swing_has_lps_mask[leg_index]] &= mask
            if (np.sum(future_lp_mask[swing_index,...].any(axis=-1),axis=1)>=3).all():
                break
            elif i < 20:
                print(">>>>>>> In Swing Traj Cal, there are less than three possible landing points in future pose")
                print(f"future pose={future_path_se3[i]}\n landing points\n{self.hex_state.env_pointsmap_voxels.landing_points[future_lp_idx]}")

        for leg_index in swing_index:
            same_branches_mask = future_lp_mask[leg_index,:,self.q3_branches[leg_index]]
            if same_branches_mask.any():
                chosen_lp_idx = future_lp_idx[same_branches_mask]
                landing_q3_branch = int(self.q3_branches[leg_index])
            else:
                landing_q3_branch = int((self.q3_branches[leg_index]+1)%2)
                diff_branches_mask = future_lp_mask[leg_index,:,landing_q3_branch]
                chosen_lp_idx = future_lp_idx[diff_branches_mask]
            if chosen_lp_idx.size == 0:
                print("leg_index={}, q3_branches={}".format(leg_index,self.q3_branches[leg_index]))
                print("future_lp_mask true sum =",future_lp_mask.sum())
                raise RuntimeError("chosen lp idex get empty value")
                continue
            #3,N
            R_points = target_se3.inv() * self.hex_state.env_pointsmap_voxels.landing_points[chosen_lp_idx].T
            #N,3
            B_points = self.kin._R2B(R_points,leg_index).T
            flat_index = self.hex_state.robot_voxels.leg_voxels.Pos2FlatIndex(B_points)
            landing_idx = np.argmax(self.hex_state.robot_voxels.to_bound_dist_flat[flat_index,leg_index])
            R_normal = target_se3.R.T @ self.hex_state.env_pointsmap_voxels.normals[chosen_lp_idx[landing_idx]].T
            landing_normal = R_normal.copy()
            if leg_index < 3:
                landing_normal[:2] *= -1
            self.B_landing_n[:,leg_index] = landing_normal
            #放入世界系下的目标落脚点
            self.W_landing_points[:,leg_index] = (target_se3 * R_points[:,landing_idx]).squeeze(1)
            _PlanOneLeg(
                leg_index,
                self.B_e_cur[:,leg_index],
                B_points[landing_idx],
                landing_q3_branch,
            )
    
    def _Cal_cost(self,B_points:np.ndarray,acc_u_scaled:np.ndarray,u:np.ndarray,leg_index:int,target_se3:SE3,
                  sol_indices,smooth_cost=None)->float:
        """计算代价函数
        @input B_points(3,N) B系下的点; acc_u_scaled(3,N) B系下经过时间缩放后的加速度; u(N)对应的自变量; 
        leg_index 腿部索引; target_se3 期望代价下对应的位姿"""
        B_points = np.asarray(B_points,dtype=np.float32)
        acc_u_scaled = np.asarray(acc_u_scaled,dtype=np.float32)
        if B_points.ndim != 2 or B_points.shape[0] != 3:
            raise ValueError("B_points must have shape (3,N)")
        if acc_u_scaled.ndim != 2 or acc_u_scaled.shape[0] != 3:
            raise ValueError("acc must have shape (3,N)")
        if u is None:
            u = np.linspace(0,1,B_points.shape[1])
        else:
            u = np.asarray(u,dtype=np.float32).reshape(-1)
            if u.shape[0] != B_points.shape[1]:
                u = np.linspace(0,1,B_points.shape[1])
        if sol_indices is None:
            sol_indices = np.full(B_points.shape[1],self.q3_branches[leg_index],dtype=np.int32)
        else:
            sol_indices = np.asarray(sol_indices,dtype=np.int32).reshape(-1)
            if sol_indices.shape[0] != B_points.shape[1]:
                raise ValueError("sol_indices must have shape (N,)")
        #计算平滑代价函数 加速度沿着u的积分；多段样条时优先使用调用方逐段积分后的值。
        if smooth_cost is None:
            smooth_cost = self.trapezoid(np.sum(acc_u_scaled**2,axis=0),u)
        R_points = self.kin._B2R(B_points,leg_index)
        W_points = target_se3*R_points                
        #计算身体碰撞与环境碰撞ESDF代价
        env_esdf = self._QueryESDFTrilinear(W_points.T,
                                            self.hex_state.env_pointsmap_voxels.voxels,
                                            self.hex_state.env_pointsmap_voxels.env_esdf)
        env_collision_cost = np.square(np.clip(0.04-env_esdf,a_min=0,a_max=None)).sum()

        #足部末端点与身体碰撞的惩罚等价于worksapce far cost 这里就不重复添加
        # body_esdf = self._QueryESDFTrilinear(R_points.T,
        #                                      self.hex_state.robot_voxels.body_voxels,
        #                                      self.hex_state.robot_voxels.body_voxels.esdf)
        # body_collision_cost = np.square(np.clip(0.04-body_esdf,a_min=0,a_max=None)).sum()

        to_bound_esdf = self._QueryESDFTrilinear(B_points.T,
                                                 self.hex_state.robot_voxels.leg_voxels,
                                                 self.hex_state.robot_voxels.to_bound_dist[...,leg_index])
        worksapce_far_cost = np.square(np.clip(0.0-to_bound_esdf,a_min=0.0,a_max=None)).sum()
        # ix,iy,iz = self.hex_state.robot_voxels.body_voxels.Pos2GridIndex(R_points.T).T
        # body_collision_cost = np.square(np.clip(0.04-self.hex_state.robot_voxels.body_voxels.esdf[ix,iy,iz],a_min=0,a_max=None)).sum()
        #计算靠近工作空间中心代价
        # workspace_far_cost = np.square(np.clip(0.1-self.hex_state.robot_voxels.to_bound_dist_flat[flat_index,leg_index],a_min=0,a_max=None)).sum()
        
        #计算ankle与环境的碰撞代价；使用规划阶段确定的IK分支 ankle关节位置是查表得出，因此梯度不是很光滑 暂时取消
        # flat_index = self.hex_state.robot_voxels.leg_voxels.Pos2FlatIndex(B_points.T)
        # ankle_pos_B = self.hex_state.robot_voxels.leg_voxels.ankle_pos[flat_index,sol_indices] #N,3
        # valid_ankle_mask = np.isfinite(ankle_pos_B).all(axis=1)
        # ankle_pos_B[~valid_ankle_mask]  = 0.0
        # # ankle_pos_B = np.nan_to_num(ankle_pos_B,nan=0.0)
        # ankle_pos_R = self.kin._B2R(ankle_pos_B.T,leg_index)
        # ankle_pos_W = (target_se3*ankle_pos_R).T
        # ankle_esdf = self._QueryESDFTrilinear(ankle_pos_W,
        #                                       self.hex_state.env_pointsmap_voxels.voxels,
        #                                       self.hex_state.env_pointsmap_voxels.env_esdf)
        # ankle_radius = self.hex_state.robot_voxels.leg_voxels.ankle_collide_radi
        # ankle_collision = np.square(np.clip(ankle_radius-ankle_esdf,a_min=0,a_max=None))
        # ankle_collision[~valid_ankle_mask] = 0.0
        # ankle_collision_cost = ankle_collision.sum()

        #轨迹长度代价 长度代价效果很差导致曲线扭曲 光滑代价能达到同样的效果 这里取消
        # length_cost = np.linalg.norm(np.diff(B_points,axis=1),axis=0).sum()
        cost = smooth_cost + worksapce_far_cost  + 5.0*env_collision_cost #+ankle_collision_cost
        # cost = smooth_cost  + 5.0*env_collision_cost #+ankle_collision_cost

        return cost                

    def _QueryESDFTrilinear(self,points:np.ndarray,voxels,esdf:np.ndarray)->np.ndarray:
        """在体素中心网格上对ESDF做三线性插值。"""
        points = np.asarray(points,dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N,3)")

        grid_f = (points-voxels._bounds[None,:,0])/voxels.voxel_scale - 0.5
        i0_raw = np.floor(grid_f).astype(np.int32)
        w = grid_f - i0_raw

        i0 = np.clip(i0_raw,0,voxels.grid_shape[None,:]-1)
        i1 = np.clip(i0+1,0,voxels.grid_shape[None,:]-1)
        w = np.clip(w,0.0,1.0)

        ix0,iy0,iz0 = i0[:,0],i0[:,1],i0[:,2]
        ix1,iy1,iz1 = i1[:,0],i1[:,1],i1[:,2]
        wx,wy,wz = w[:,0],w[:,1],w[:,2]

        c000 = esdf[ix0,iy0,iz0]
        c100 = esdf[ix1,iy0,iz0]
        c010 = esdf[ix0,iy1,iz0]
        c110 = esdf[ix1,iy1,iz0]
        c001 = esdf[ix0,iy0,iz1]
        c101 = esdf[ix1,iy0,iz1]
        c011 = esdf[ix0,iy1,iz1]
        c111 = esdf[ix1,iy1,iz1]

        c00 = c000*(1.0-wx) + c100*wx
        c10 = c010*(1.0-wx) + c110*wx
        c01 = c001*(1.0-wx) + c101*wx
        c11 = c011*(1.0-wx) + c111*wx
        c0 = c00*(1.0-wy) + c10*wy
        c1 = c01*(1.0-wy) + c11*wy
        return c0*(1.0-wz) + c1*wz

    def _QuinticPolyInterp(self,p0,v0,a0,p1,v1,a1,T,dt):
        T = max(T,dt)
        t = np.arange(0,T,dt)
        if t[-1]<T:
            t = np.hstack([t,T])
        return self._QuinticInterp(p0,v0,a0,p1,v1,a1,T,t,wrap_angle=False)

    def _ClearTraj(self,mask:np.ndarray):
        """
        输入6维mask，清理对应轨迹索引，轨迹长度，轨迹初始值
        """
        for i in range(6):
            if mask[i]:
                # self.B_e_init[:,i] = self.B_e_traj[i][:,self.q_traj_index[i]].copy()
                # self.B_e_traj[i]=self.B_e_init[:,i,None].copy()
                self.B_e_traj[i]=self.B_e_cur[:,i,None].copy() # array(3,1)
                self.q_traj[i] = None
        self.q_traj_index[mask] = 0
        self.B_e_traj_len[mask] = 1

    def _InterpConstSpeed(self, key_points: np.ndarray, speed: float) -> np.ndarray:
        """
        @description:
        对折线关键点做等速插值。给定关键点(3,N)和速度speed(m/s)，按照控制周期self.dt
        在弧长上均匀采样，返回(3,M)轨迹点（包含起点和终点）。

        @input:
        key_points: (3,N) 折线关键点，按顺序连接
        speed: 标量，插值速度(m/s)

        @output:
        traj: (3,M) 等速采样后的轨迹点
        """
        key_points = np.asarray(key_points, dtype=np.float32)
        if key_points.ndim != 2 or key_points.shape[0] != 3:
            raise ValueError("key_points must have shape (3, N)")
        if key_points.shape[1] == 0:
            raise ValueError("key_points must contain at least one point")
        if key_points.shape[1] == 1:
            return key_points.copy()
        if speed <= 0:
            raise ValueError("speed must be positive")

        # Compute cumulative arc length of the polyline.
        seg_vec = np.diff(key_points, axis=1)
        seg_len = np.linalg.norm(seg_vec, axis=0)
        total_len = float(np.sum(seg_len))
        if total_len < 1e-12:
            return key_points[:, :1].copy()

        arc = np.hstack(([0.0], np.cumsum(seg_len)))

        # Sample uniformly in arc length with spacing ~ speed * dt.
        ds = max(speed * self.dt, 1e-6)
        sample_num = max(2, int(np.ceil(total_len / ds)) + 1)
        arc_query = np.linspace(0.0, total_len, sample_num)

        traj = np.empty((3, sample_num), dtype=np.float32)
        traj[0, :] = np.interp(arc_query, arc, key_points[0, :])
        traj[1, :] = np.interp(arc_query, arc, key_points[1, :])
        traj[2, :] = np.interp(arc_query, arc, key_points[2, :])
        return traj
        

if __name__ == '__main__':
    hex_climb = ExpertComplex()
