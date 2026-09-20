#机器人实用工具
#工作空间与凸包
#正逆运动学与雅可比矩阵计算
import numpy as np
from numpy import cos, sin, arccos, arctan2, arcsin, sqrt,pi
from math import radians
from typing import List, Tuple, Union

class Kinematic:
    def __init__(self,l1=0.072,l2=0.13,l3=0.17,
                 bx=0.1,by=0.22,
                 singularity_threshold=0.02):
        self.l1=l1
        self.l2=l2
        self.l3=l3
        self.singularity_threshold = singularity_threshold
        # 解析IK中使用的全局限位，q1取所有腿thigh限位的并集，q2/q3为所有腿共用限位。
        # 这里只在Kinematic类内部使用 外部不进行调用
        self._joints_limits=np.array([[-np.pi/2.0,np.pi/2.0],
                                     [-np.pi*2.0/3.0, np.pi*25.0/36.0],
                                     [-np.pi*31.0/36.0, np.pi*7.0/9.0]],dtype=np.float32)
        # 这里保存每条腿thigh关节的
        self.joints_limits_thigh = np.zeros((6,2),dtype=np.float32)
        self.joints_limits_rest = np.array(
            [[radians(-120),radians(125)],[radians(-155),radians(140)]],
            dtype=np.float32,
        )
        #机器人身体的参数，腿部基坐标在R系下基座的位置
        self._bx = bx
        self._by = by
        self._leg_names=["LB","LF","LM","RB","RF","RM"]
        for i,name in enumerate(self._leg_names):
            if name=="LF" or name =="RB":
                self.joints_limits_thigh[i]=[radians(-92),radians(45)]
            elif name=="RF" or name=="LB":
                self.joints_limits_thigh[i]=[radians(-45),radians(92)]
            else:
                self.joints_limits_thigh[i]=[radians(-45),radians(45)]
        self.leg_base_p = np.zeros((3,6),dtype=np.float32)
        for i,name in enumerate(self._leg_names):
            #保存腿部基坐标原点在身体坐标系下的位置
            bx = self._bx
            by = self._by
            if 'M' in name:
                by = 0
            if 'L' in name:
                bx = -self._bx
            if 'B' in name:
                by = -self._by
            self.leg_base_p[:,i]=[bx,by,0.0]        
        
    def ForwardKinReturn(self,joints)->np.ndarray:
        """joints:(3,)
        return pos:(3,)"""
        q1=joints[0]
        q2=joints[1]
        q3=joints[2]
        return np.array([(self.l1 + self.l2*cos(q2) + self.l3*cos(q2+q3))*cos(q1),
                          (self.l1 + self.l2*cos(q2) + self.l3*cos(q2+q3))*sin(q1),
                          self.l2*sin(q2) + self.l3*sin(q2+q3)])

    def ForwardKin(self,joints,pos:np.ndarray):
        """joints:[batch_size,3],pos:[batch_size,3]"""
        q1=joints[:,0]
        q2=joints[:,1]
        q3=joints[:,2]
        pos[:,0]=(self.l1 + self.l2*cos(q2) + self.l3*cos(q2+q3))*cos(q1)
        pos[:,1]=(self.l1 + self.l2*cos(q2) + self.l3*cos(q2+q3))*sin(q1)
        pos[:,2]=self.l2*sin(q2) + self.l3*sin(q2+q3)
    def ForwardKinKnee(self,joints,pos:np.ndarray):
        """joints:[batch_size,1],pos:[batch_size,3]"""
        q1 = joints[:,0]
        pos[:,0]=self.l1*cos(q1)
        pos[:,1]=self.l1*sin(q1)
        pos[:,2]=0.0

    def ForwardKinAnkle(self,joints,pos:np.ndarray):
        """joints:[batch_size,2],pos:[batch_size,3]"""
        q1=joints[:,0]
        q2=joints[:,1]        
        pos[:,0]=(self.l1 + self.l2*cos(q2) )*cos(q1)
        pos[:,1]=(self.l1 + self.l2*cos(q2) )*sin(q1)
        pos[:,2]=self.l2*sin(q2)


    def Get_X_B3(self,joints:np.ndarray)->np.ndarray:
        """joints:[batch_size,3] directions:[batch_size,3]
        获取连杆3的x轴B系下的方向向量"""
        q1=joints[:,0]
        q2=joints[:,1]
        q3=joints[:,2]
        return np.column_stack([cos(q2+q3)*cos(q1),cos(q2+q3)*sin(q1),sin(q2+q3)])
    
    def Get_Z_B3(self,joints:np.ndarray)->np.ndarray:
        """joints:[batch_size,3] directions:[batch_size,3]
        获取连杆3的z轴在B系下的方向向量"""
        q1=joints[:,0]
        return np.column_stack([sin(q1),-cos(q1),np.zeros_like(q1)])

    def Jacobian(self,joints:np.ndarray)->np.ndarray:
        """joints:[batch_size,3],parallel_num:batch_size return [batch_size,3,3]"""
        Jac=np.zeros((joints.shape[0],3,3))
        q1=joints[:,0]
        q2=joints[:,1]
        q3=joints[:,2]
        Jac[:,0,0]=-(self.l1 + self.l2*cos(q2)+self.l3*cos(q2+q3) )*sin(q1)
        Jac[:,1,0]=(self.l1 + self.l2*cos(q2)+self.l3*cos(q2+q3) )*cos(q1)
        Jac[:,2,0]=0

        Jac[:,0,1]=-(self.l2*sin(q2)+self.l3*sin(q2+q3))*cos(q1)
        Jac[:,1,1]=-(self.l2*sin(q2)+self.l3*sin(q2+q3))*sin(q1)
        Jac[:,2,1]=self.l2*cos(q2)+self.l3*cos(q2+q3)

        Jac[:,0,2]=-self.l3*cos(q1)*sin(q2+q3)
        Jac[:,1,2]=-self.l3*sin(q1)*sin(q2+q3)
        Jac[:,2,2]=self.l3*cos(q2+q3)
        return Jac

    def LegVectorToR(self,leg_vectors:np.ndarray,leg_indices:Union[np.ndarray,int])->np.ndarray:
        """将腿局部系中的方向/力转换到机身 R 系。

        这是纯旋转（左腿为绕 z 轴旋转 pi，右腿为单位旋转），不含
        ``_B2R`` 中针对点坐标的腿根平移。输入为 shape ``(3, N)``。
        """
        vectors = np.asarray(leg_vectors).copy()
        if vectors.ndim != 2 or vectors.shape[0] != 3:
            raise ValueError("leg_vectors must have shape (3, N)")
        if np.isscalar(leg_indices):
            if int(leg_indices) < 3:
                vectors[:2, :] *= -1.0
            return vectors
        indices = np.asarray(leg_indices,dtype=int).ravel()
        if vectors.shape[1] != indices.size:
            raise ValueError("leg_indices must match the vector count")
        vectors[:2,indices<3] *= -1.0
        return vectors

    def RVectorToLeg(self,R_vectors:np.ndarray,leg_indices:Union[np.ndarray,int])->np.ndarray:
        """将机身 R 系中的方向/力转换到对应腿局部系。

        当前左右腿坐标约定的旋转矩阵是自逆的，故该变换与
        :meth:`LegVectorToR` 数值相同；仍保留独立接口以避免将点变换
        ``_R2B`` 错用于方向或力。
        """
        return self.LegVectorToR(R_vectors,leg_indices)

    def Hessian(self,joints:np.ndarray)->np.ndarray:
        """计算足端位置对关节角的二阶导数。

        @input joints: (batch_size,3)
        @output Hess: (batch_size,3,3,3)
            Hess[:,xyz,i,j] = d^2 p_xyz / (dq_i dq_j)
        """
        joints = np.asarray(joints)
        Hess = np.zeros((joints.shape[0],3,3,3),dtype=joints.dtype)
        q1 = joints[:,0]
        q2 = joints[:,1]
        q3 = joints[:,2]
        c1 = cos(q1)
        s1 = sin(q1)
        s2 = sin(q2)
        c2 = cos(q2)
        s23 = sin(q2+q3)
        c23 = cos(q2+q3)
        r = self.l1 + self.l2*c2 + self.l3*c23
        A = self.l2*s2 + self.l3*s23
        B = self.l2*c2 + self.l3*c23
        C = self.l3*s23
        D = self.l3*c23

        # x方向二阶导数
        Hess[:,0,0,0] = -r*c1
        Hess[:,0,0,1] = A*s1
        Hess[:,0,1,0] = Hess[:,0,0,1]
        Hess[:,0,0,2] = C*s1
        Hess[:,0,2,0] = Hess[:,0,0,2]
        Hess[:,0,1,1] = -B*c1
        Hess[:,0,1,2] = -D*c1
        Hess[:,0,2,1] = Hess[:,0,1,2]
        Hess[:,0,2,2] = -D*c1

        # y方向二阶导数
        Hess[:,1,0,0] = -r*s1
        Hess[:,1,0,1] = -A*c1
        Hess[:,1,1,0] = Hess[:,1,0,1]
        Hess[:,1,0,2] = -C*c1
        Hess[:,1,2,0] = Hess[:,1,0,2]
        Hess[:,1,1,1] = -B*s1
        Hess[:,1,1,2] = -D*s1
        Hess[:,1,2,1] = Hess[:,1,1,2]
        Hess[:,1,2,2] = -D*s1

        # z方向二阶导数
        Hess[:,2,1,1] = -A
        Hess[:,2,1,2] = -C
        Hess[:,2,2,1] = Hess[:,2,1,2]
        Hess[:,2,2,2] = -C
        return Hess

    def JacobianDotQdot(self,joints:np.ndarray,joints_vel:np.ndarray)->np.ndarray:
        """计算 Jdot(q,qdot) @ qdot，也就是足端加速度中的二阶速度项。"""
        Hess = self.Hessian(joints)
        joints_vel = np.asarray(joints_vel)
        return np.einsum("bpij,bi,bj->bp",Hess,joints_vel,joints_vel)

    def DampInvJac(self,joints:np.ndarray,sing_hyst=None):
        """joints:[batch_size,3],sing_hyst外部输入的奇异值信号，通常是上一级有迟滞处理，如果没有输入就采用自己的
        这里的奇异处理，只能针对在z轴附近的，不能处理其他奇异值情况
            return [batch_size,3,3]"""
        Jac = self.Jacobian(joints)
        JJT=Jac@Jac.transpose(0,2,1)+np.eye(3)*0.0001
        #J的违逆 = J.T @ (J@J.T+I*lambda)^-1
        damp_inv_jac = np.linalg.solve(JJT, Jac).transpose(0,2,1) #本质在干->Jac.transpose(0,2,1)@np.linalg.inv(JJT)
        if sing_hyst is None:
            singularity_mask = self._Singularity_Z(joints)
        else:
            singularity_mask = sing_hyst

        if singularity_mask.any():
            damp_inv_jac[singularity_mask,0,:] = 0
        return damp_inv_jac
        #奇异解处理
        # singularity_mask = self._Singularity_Z(joints)
        # Jac = self.Jacobian(joints)
        # JJT = np.zeros_like(Jac)
        # if singularity_mask.any():
        #     Jac32 = Jac[singularity_mask][:,1:3]
        #     JJT[singularity_mask] = Jac32@Jac32.transpose(0,2,1) + np.eye(3)*0.00001

        # if ~singularity_mask.any():
        #     Jac33 = Jac[~singularity_mask]
        #     JJT[~singularity_mask] = Jac33@Jac33.transpose(0,2,1) = np.eye(3)*0.00001
        # damp_inv_jac = np.linalg.solve(JJT,Jac).transpose(0,2,1)

    def InverseKin1(self,pos:np.ndarray,joints_cur:np.ndarray):
        """use iterative jacobian to get desired joints, joints_cur is modified to the desired joints"""
        pos_cur = np.zeros_like(pos)
        self.ForwardKin(joints_cur,pos_cur)
        for _ in range(1000):
            diff_norm=np.linalg.norm(pos-pos_cur,axis=1)
            indices = np.where(diff_norm < 0.005)[0]
            if indices.size == pos.shape[0]:
                # print("Inverse kinematics success, joints_cur\n",joints_cur)
                #every joints is solved
                return

            damp_inv_jacs=self.DampInvJac(joints_cur)
            # print(joints_cur,pos,pos_cur,damp_inv_jac)
            
            joints_cur += ( 0.01*(damp_inv_jacs@((pos-pos_cur)[...,None])).squeeze(-1) )
            self.ForwardKin(joints_cur,pos_cur)
        diff_norm=np.linalg.norm(pos-pos_cur,axis=1)
        indices=np.where(diff_norm>0.005)[0]
        if indices.size > 0:
            print("IK failed,indices\n:{}, pos_cur[indices]:\n{}".format(indices,pos_cur[indices]))
    
    def InverseKin2(self,pos:np.ndarray,joints_cur:np.ndarray):
        """use analytical solution to get desired joints, find the nearest solution to joints_cur
           pos:[batch_size,3], joints_cur:[batch_size,3], this will modify joints_cur"""
        x=pos[:,0]
        y=pos[:,1]
        z=pos[:,2]
        q1=arctan2(y,x)
        #将q1的范围限制在±pi/2之间，超过就±pi
        bigger_mask = q1>pi/2.0
        q1[bigger_mask] = q1[bigger_mask]-pi
        smaller_mask= q1<-pi/2.0
        q1[smaller_mask] = pi + q1[smaller_mask]

        # q1_possible=torch.stack([q1,q1+pi,q1-pi],dim=0)
        # joints_cur[:,0]=self._SelectCloestAngles(q1_possible,joints_cur[:,0])
        # min_index=torch.abs(q1_possible-joints_cur[:,0]).argmin(dim=0)
        # q1=torch.gather(q1_possible,dim=0,index=min_index.unsqueeze(0)).squeeze(0)

        cos_q3 = ( (cos(q1)*x+sin(q1)*y-self.l1)**2 + z**2-self.l3**2-self.l2**2 )/( 2*self.l2*self.l3 ) 
        q3=arccos( np.clip(cos_q3,-1,1))
        q3_possible=np.stack([q3,-q3],axis=0)
        q3=self._SelectCloestAngles(q3_possible,joints_cur[:,2])
        # min_index=torch.abs(q3_possible-joints_cur[:,2]).argmin(dim=0)
        # q3=torch.gather(q3_possible,dim=0,index=min_index.unsqueeze(0)).squeeze(0)
        # q3 = q3 if abs(q3-joints_cur[:,2])<abs(-q3-joints_cur[:,2]) else -q3

        q2_back=arctan2( (self.l3*sin(q3)),( self.l3*cos(q3)+self.l2) )
        q2_front=arcsin( z/sqrt( (self.l3*cos(q3)+self.l2)**2+(self.l3*sin(q3))**2 ) )
        # q2_b_possible=torch.stack([q2_back,q2_back+pi,q2_back-pi],dim=0)
        q2_f_possible=np.stack([q2_front,-q2_front+pi,-q2_front-pi],axis=0) #3*batch_size?????????
        # q2_possible=(q2_f_possible.unsqueeze(0)-q2_b_possible.unsqueeze(1)).reshape(-1,pos.shape[0]) #3*3*batch_size
        q2_possible = q2_f_possible-q2_back

        # print("q2_f_possible\n",q2_f_possible)
        # print("q2_b_possible\n",q2_b_possible)
        q2=self._SelectCloestAngles(q2_possible,joints_cur[:,1])
        # min_index=torch.abs(q2_possible-joints_cur[:,1]).argmin(dim=0)
        # q2=torch.gather(q2_possible,dim=0,index=min_index.unsqueeze(0)).squeeze(0)
        joints_cur[:,0]=q1
        joints_cur[:,1]=q2
        joints_cur[:,2]=q3
        # print("q1={}\n; q2={}\n, q3={}\n".format(q1,q2_possible,q3_possible))
    
    def InverseKin2Multi(self,pos:np.ndarray)->np.ndarray:
        """
        计算当前位置的所有可能关节角度
        @input pos: (3,) 当前位置
        @output joints (N,3) 代表当前位置对应的N种关节角度
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(3,)
        x, y, z = pos

        q1 = arctan2(y, x)
        #将q1的范围限制在±pi/2之间，超过就±pi，可以避免足端在y轴左侧，第一关节在y轴右侧
        if q1>pi/2.0:
            q1 = q1 - pi
        elif q1<-pi/2.0:
            q1 = pi + q1
        
        r = cos(q1) * x + sin(q1) * y - self.l1
        cos_q3 = (r * r + z * z - self.l3 ** 2 - self.l2 ** 2) / (2.0 * self.l2 * self.l3)
        if cos_q3 < -1.0 - 1e-9 or cos_q3 > 1.0 + 1e-9:
            return np.zeros((0, 3), dtype=np.float64)

        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3_abs = arccos(cos_q3)
        q3_candidates = np.array([q3_abs, -q3_abs], dtype=np.float64)

        solutions: List[np.ndarray] = []
        for q3 in q3_candidates:
            if q3>=self._joints_limits[2,0] and q3<= self._joints_limits[2,1]:
                q2_back = arctan2(self.l3 * sin(q3), self.l3 * cos(q3) + self.l2)
                front_denom = sqrt((self.l3 * cos(q3) + self.l2) ** 2 + (self.l3 * sin(q3)) ** 2)
                if front_denom < 1e-12:
                    continue

                sin_q2_front = np.clip(z / front_denom, -1.0, 1.0)
                q2_front = arcsin(sin_q2_front)
                q2_candidates = np.array(
                    [
                        q2_front - q2_back,
                        -q2_front + pi - q2_back,
                        -q2_front - pi - q2_back,
                    ],
                    dtype=np.float64,
                )

                for q2 in q2_candidates:
                    if q2>= self._joints_limits[1,0] and q2 <= self._joints_limits[1,1]:
                        solutions.append(np.array([q1, q2, q3], dtype=np.float64))

        if not solutions:
            return np.zeros((0, 3), dtype=np.float64)

        joints = np.vstack(solutions)
        pos_fk = np.zeros_like(joints)
        self.ForwardKin(joints, pos_fk)
        valid_mask = np.linalg.norm(pos_fk - pos[None, :], axis=1) < 1e-6
        joints = joints[valid_mask]
        if joints.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # Wrap to principal branch so equivalent solutions can be deduplicated.
        joints = (joints + pi) % (2.0 * pi) - pi
        joints = np.unique(np.round(joints, decimals=10), axis=0)
        return joints

    def InverseKin2MultiBatch(self,pos:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """
        批量计算多个足端位置的所有可能关节角度。

        @input:
        pos: (batch_size,3)

        @output:
        joints_batch: (batch_size,2,3)
            每个位置最多保留两个解，这两个分支分布代表肘上肘下区，也就是q3的正负决定的，不存在的解以 nan 填充
        solution_mask: (batch_size,2)
            标记两个槽位中哪些解有效
        """
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (batch_size,3)")

        batch_size = pos.shape[0]
        joints_batch = np.full((batch_size, 2, 3), np.nan, dtype=np.float64)
        solution_mask = np.zeros((batch_size, 2), dtype=bool)

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        q1 = arctan2(y, x)
        # 将第一关节折叠到主工作范围内，避免左右等价角度同时出现。
        bigger_mask = q1 > pi / 2.0
        q1[bigger_mask] = q1[bigger_mask]-pi
        smaller_mask = q1 < -pi / 2.0
        q1[smaller_mask] = pi + q1[smaller_mask]

        r = cos(q1) * x + sin(q1) * y - self.l1
        cos_q3 = (r * r + z * z - self.l3 ** 2 - self.l2 ** 2) / (2.0 * self.l2 * self.l3)
        reach_mask = (cos_q3 >= -1.0 - 1e-9) & (cos_q3 <= 1.0 + 1e-9)
        if not reach_mask.any():
            return joints_batch, solution_mask

        q1_valid = q1[reach_mask]
        z_valid = z[reach_mask]
        pos_valid = pos[reach_mask]
        cos_q3_valid = np.clip(cos_q3[reach_mask], -1.0, 1.0)
        q3_abs = arccos(cos_q3_valid)

        # 两个主分支对应 q3 的正负号; 两个解分别对应了肘上和肘下区
        q3_branches = np.stack([q3_abs, -q3_abs], axis=1)  # (M,2)
        valid_indices = np.nonzero(reach_mask)[0]

        for branch_idx in range(2):
            q3 = q3_branches[:, branch_idx]
            q3_limit_mask = (q3 >= self._joints_limits[2, 0]) & (q3 <= self._joints_limits[2, 1])
            if not q3_limit_mask.any():
                continue

            active_idx = valid_indices[q3_limit_mask]
            q1_branch = q1_valid[q3_limit_mask]
            q3_branch = q3[q3_limit_mask]
            z_branch = z_valid[q3_limit_mask]
            pos_branch = pos_valid[q3_limit_mask]

            q2_back = arctan2(self.l3 * sin(q3_branch), self.l3 * cos(q3_branch) + self.l2)
            front_denom = sqrt((self.l3 * cos(q3_branch) + self.l2) ** 2 + (self.l3 * sin(q3_branch)) ** 2)
            valid_denom_mask = front_denom >= 1e-12
            if not valid_denom_mask.any():
                continue

            active_idx = active_idx[valid_denom_mask]
            q1_branch = q1_branch[valid_denom_mask]
            q3_branch = q3_branch[valid_denom_mask]
            z_branch = z_branch[valid_denom_mask]
            pos_branch = pos_branch[valid_denom_mask]
            q2_back = q2_back[valid_denom_mask]
            front_denom = front_denom[valid_denom_mask]

            sin_q2_front = np.clip(z_branch / front_denom, -1.0, 1.0)
            q2_front = arcsin(sin_q2_front)
            q2_candidates = np.stack(
                [
                    q2_front - q2_back,
                    -q2_front + pi - q2_back,
                    -q2_front - pi - q2_back,
                ],
                axis=1,
            )  # (M_valid,3)

            # 每个 q3 分支最多对应 3 个 q2 表达式。把 (M,3) 候选一次性展开成
            # (M*3,3) 关节数组，批量 FK 回代筛选，避免对每个点逐行调用 ForwardKin。
            q2_limit_mask = (
                (q2_candidates >= self._joints_limits[1, 0])
                & (q2_candidates <= self._joints_limits[1, 1])
            )
            if not q2_limit_mask.any():
                continue

            candidate_joints = np.stack(
                [
                    np.repeat(q1_branch, 3),
                    q2_candidates.reshape(-1),
                    np.repeat(q3_branch, 3),
                ],
                axis=1,
            )
            candidate_pos = np.zeros_like(candidate_joints)
            self.ForwardKin(candidate_joints, candidate_pos)

            # q2 限位和 FK 回代都通过才认为候选有效。fk_match reshape 回 (M,3)，
            # 后续按候选列填入；现实数据中 <1e-6 时每个 q3 分支只会留下一个 q2。
            fk_match = (
                np.linalg.norm(
                    candidate_pos.reshape(-1, 3, 3) - pos_branch[:, None, :],
                    axis=2,
                ) < 1e-6
            )
            valid_candidates = q2_limit_mask & fk_match
            if not valid_candidates.any():
                continue

            candidate_joints = ((candidate_joints + pi) % (2.0 * pi) - pi).reshape(-1, 3, 3)
            multi_solution_mask = valid_candidates.sum(axis=1) > 1
            if multi_solution_mask.any():
                for row_idx in np.nonzero(multi_solution_mask)[0]:
                    print("one q3 branch get two possible q2")
                    print(
                        f"q_valid\n {candidate_joints[row_idx,valid_candidates[row_idx]]} "
                        f"pos={pos[active_idx[row_idx]]} "
                    )

            # 按 q2 表达式顺序选第一个有效候选。这里最多循环 3 次，不再随点数增长。
            filled = np.zeros(active_idx.shape[0], dtype=bool)
            for q2_candidate_idx in range(3):
                fill_mask = valid_candidates[:, q2_candidate_idx] & (~filled)
                if not fill_mask.any():
                    continue
                joints_batch[active_idx[fill_mask], branch_idx] = candidate_joints[fill_mask, q2_candidate_idx]
                solution_mask[active_idx[fill_mask], branch_idx] = True
                filled[fill_mask] = True

        return joints_batch, solution_mask

    def CF_Estimate(self,torq:np.ndarray,q_cur:np.ndarray,
                    sing_hyst=None)->np.ndarray:
        """
        @description: 根据关节扭矩和当前关节角，估算足端对环境的等效作用力。

        这里求解的是 tau ~= J.T @ force 的阻尼最小二乘近似。
        因此返回的 force 更适合理解为“机器人足端对环境/接触面的作用力”。
        如果需要“环境对机器人足端的反作用力”，应在调用处取 -force。
        
        @input:
        输入扭矩 torq [batch_size,3], 当前关节角度q_cur [batch_size,3],
        singularity_hysteresis迟滞区间的奇异点

        @output:
        返回足端对环境的等效作用力 force [batch_size,3]
        """
        #筛选出在奇异点Z轴附近(5cm以内)的
        if sing_hyst is None:
            singularity_mask = self._Singularity_Z(q_cur)
        else:
            singularity_mask = sing_hyst

        Jac = self.Jacobian(q_cur)

        # force = np.zeros_like(torq)
        # if singularity_mask.any():
        #     torq2 = torq[singularity_mask][:,1:3] #带有奇异性的只取后两个分量 M,2
        #     Jac32 = Jac[singularity_mask][...,1:3]
        #     F2 = (Jac32 @ np.linalg.inv(Jac32.transpose(0,2,1)@ Jac32)) @ torq2[...,None]
        #     force[singularity_mask]=F2.squeeze(-1)
        # if (~singularity_mask).any():
        #     torq3 = torq[~singularity_mask] # N,3
        #     Jac33 = Jac[~singularity_mask]
        #     F3 = np.linalg.solve(Jac33.transpose(0,2,1),torq3[...,None])
        #     force[~singularity_mask] = F3.squeeze(-1)

        # 不做奇异/非奇异分支求解，统一用阻尼最小二乘。
        # 近似反解 tau = J.T @ F:
        # F = J @ inv(J.T @ J + lambda*I) @ tau
        damp_inv_invert_jac = Jac@np.linalg.inv(Jac.transpose(0,2,1)@Jac+np.eye(3)*0.00001)
        if singularity_mask.any():
            damp_inv_invert_jac[singularity_mask,:,0] = 0
        force = (damp_inv_invert_jac@torq[...,None]).squeeze(-1)
        return force
    
    def _Singularity_Z(self,joints:np.ndarray)->np.ndarray:
        """
        @description: 判断当前角度是否在奇异值附近
        """
        q2 = joints[:,1]
        q3 = joints[:,2]
        r_xy = abs(self.l1 + self.l2*cos(q2) + self.l3*cos(q2+q3))     
        return r_xy < self.singularity_threshold
    
    def _SelectCloestAngles(self,possible_angles:np.ndarray,current_angles:np.ndarray):
        """possible_angles:[possible_num,batch_size],current_angles:[batch_size]
           return selected_angles:[batch_size]"""
        min_index=np.abs(possible_angles-current_angles).argmin(axis=0)
        selected_angles=possible_angles[min_index,np.arange(current_angles.shape[0])]
        # same as below, gather is suitable for large possible_num
        # selected_angles=torch.gather(possible_angles,dim=0,index=min_index.unsqueeze(0)).squeeze(0)
        return selected_angles

    def _B2R(self,B_point:np.ndarray,leg_indices:Union[np.ndarray,int])->np.ndarray:
        """输入3*N个腿部基坐标系B下的点,返回3*N个身体坐标系R下的点
        1、leg_indices 是N个对应的腿部索引，共N个腿
        1、leg_indices 只有一个scalar，是所有的点都是这一条腿"""
        R_point = B_point.copy()
        # ---------- 情况 1：scalar ----------
        if np.isscalar(leg_indices):
            leg = int(leg_indices)
            if leg < 3:  # 左腿规则
                R_point[0:2, :] = self.leg_base_p[0:2, leg:leg+1] - B_point[0:2, :]
            else:        # 右腿规则
                R_point[0:2, :] = self.leg_base_p[0:2, leg:leg+1] + B_point[0:2, :]
            return R_point

        # ---------- 情况 2：array ----------
        idx = np.asarray(leg_indices, dtype=int).ravel()
        cols = np.arange(idx.size)
        left_cols = cols[idx<3]
        right_cols = cols[idx>=3]
        if left_cols.size:
            R_point[0:2, left_cols] = self.leg_base_p[0:2, idx[left_cols]] - B_point[0:2, left_cols]
        if right_cols.size:
            R_point[0:2, right_cols] = self.leg_base_p[0:2, idx[right_cols]] + B_point[0:2, right_cols]
        return R_point

    def _R2B(self,R_point:np.ndarray,leg_indices:Union[np.ndarray,int])->np.ndarray:
        """输入3*N个R系下的点，返回3*N个B系下的点
        1、leg_indices 是N个对应的腿部索引，共N个腿
        1、leg_indices 只有一个scalar，是所有的点都是这一条腿"""
        B_point = R_point.copy()
        # ---------- 情况 1：scalar -> broadcast ----------
        if np.isscalar(leg_indices):
            leg = int(leg_indices)
            if leg < 3:  # 左腿规则：B = leg_base_p - R
                B_point[0:2, :] = self.leg_base_p[0:2, leg:leg+1] - R_point[0:2, :]
            else:        # 右腿规则：B = R - leg_base_p
                B_point[0:2, :] = R_point[0:2, :] - self.leg_base_p[0:2, leg:leg+1]
            return B_point

        # ---------- 情况 2：array -> 按列索引 ----------
        idx = np.asarray(leg_indices, dtype=int).ravel()
        cols = np.arange(idx.size)
        left_cols = cols[idx<3]
        right_cols = cols[idx>=3]
        if left_cols.size:
            B_point[0:2, left_cols] = self.leg_base_p[0:2, idx[left_cols]] - R_point[0:2, left_cols]
        if right_cols.size:
            B_point[0:2, right_cols] = R_point[0:2, right_cols] - self.leg_base_p[0:2, idx[right_cols]]
        return B_point



if __name__ == '__main__':
    kin=Kinematic(0.072,0.13,0.17)
    """验证多解逆运动学"""
    # joints_array = np.random.random_sample((10,3))*(kin.joints_limits[:,1]-kin.joints_limits[:,0])+kin.joints_limits[:,0]
    # pos = np.zeros_like(joints_array)
    # kin.ForwardKin(joints_array,pos)
    # kin.InverseKin2MultiBatch(pos)
    
    print(kin.DampInvJac(np.array([[0,0,0]])))

    # for joints in joints_array:
    #     pos = np.zeros_like(joints)
    #     kin.ForwardKin(joints[None,:],pos[None,:])
    #     possible_joints = kin.InverseKin2Multi(pos)
    #     print(f"pos={pos}\n,possible joints={possible_joints}, input joints={joints}\n")

    # # joint_des = np.array([[0,0.5,2.1]])
    # # pos_des = np.zeros_like(joint_des)
    # # kin.ForwardKin(joint_des,pos_des)
    # # print(pos_des)


    # joint_cur = np.array([[0.2,-0.95,-1.9],[0,-pi,0]])
    # # torque = np.random.rand(2,3)*10
    # torque = np.array([[0,1,1],[0,-2,0]])
    # jac = kin.Jacobian(joint_cur)

    # print(kin.CF_Estimate(torque,joint_cur))
    
    # print(kin.DampInvJac(joint_cur))

    # kin.InverseKin2(pos_des,joint_cur)
    # kin.ForwardKin(joint_cur,pos_des)
    # print(joint_cur)
    # print(pos_des)

    # chs.PointsInPoly(np.random.random((3,6)),np.arange(6))

    # chs.FarestPoints(np.array([[0.3,0,0.1],[0.3,0.1,0.06]]).T,np.array([[0,0,1],[0,1,0]]).T,5)
    # point,ch_index = chs.FarestPoint(np.array([0.3,0,0.1])[:,None],np.array([0,0,1])[:,None],5)
    # print(np.vstack([point,point,point,point]).T)
    # print(f"point={point},ch_index={ch_index}")

    # print(chs.body_coeff_abcd)
    # print(np.linalg.norm(chs.body_coeff_abcd[:,0:3],axis=1))
    # print(chs.body_v.points)
    # points = chs.PushOutBody(np.array([[0,0,0],[0.05,0,0]]).T)
    # print(points)
    # # point = np.array([[0.0,0,-0.075]]).T
    # point = np.array([[0.0,(0.35+0.37)/2,(-0.075+0.2)/2]]).T
    # res = chs.body_coeff_abcd[:,0:3] @ point + chs.body_coeff_abcd[:,3,None]
    # zero_index = np.where(res==0)[0]
    pass
    # points = np.random.rand(3,6)
    # points = np.array([[0.2,0.0,0.05],[0.32,0,0.1]]).T
    # leg_ids = np.array([5,5])
    # # norms = np.random.rand(3,6)
    # norms = np.array([[0,0,1],[0,0.2,0.98]]).T

    # chs.PointInsidePoly(points,norms,leg_ids)
