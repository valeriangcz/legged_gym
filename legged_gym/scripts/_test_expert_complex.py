from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.expert_complex_utils import ExpertComplex
from legged_gym.envs import HexClimb, HexClimbCfg
from legged_gym.utils import JoyStick, get_args
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict, parse_sim_params
import json
import torch
import numpy as np
from spatialmath import SE3,UnitQuaternion,SO3
from pathlib import Path
import time
import os
from typing import List
from isaacgym import gymutil, gymapi

def StepExpert(q_des,tau_ff,adhesions):
    """将单个 ExpertComplex 输出转换为 HexClimb 的 batch 接口。"""
    return env.step_q_tao(
        torch.as_tensor(q_des,dtype=torch.float32,device=device).reshape(1,24),
        torch.as_tensor(tau_ff,dtype=torch.float32,device=device).reshape(1,18),
        torch.as_tensor(adhesions,dtype=torch.float32,device=device).reshape(1,6),
    )
def DrawPoints(points:np.ndarray,color=(1,1,0),point_size=0.01):
    """points: N,3"""
    global env
    for point in points:
        point_geometry = gymutil.WireframeSphereGeometry(point_size,4,4,color=color)
        point_pos = gymapi.Transform(gymapi.Vec3(x=point[0],y=point[1],z=point[2]),r=None)
        gymutil.draw_lines(point_geometry,env.gym,env.viewer,env.envs[0],point_pos)
    pass

def LoadFromBezierFile(file:Path)->List[SE3]:
    #读取优化/样条曲线轨迹
    path_se3=[]
    with file.open("r",encoding="utf-8") as stream:
        path_raw = json.load(stream)
        if "dense_poses" not in path_raw:
            raise ValueError("missing field: dense_poses")
        dense_poses=path_raw["dense_poses"]
        for t,ang,vec in zip(dense_poses["t"],dense_poses["ang"],dense_poses["vec"]):
            path_se3.append(SE3.Trans(t)*SE3.AngleAxis(ang,vec))
    return path_se3
def LoadFromTeletopFile(file:Path)->List[SE3]:
    #从人类遥操作数据中读取
    path_se3=[]
    with file.open("r",encoding="utf-8") as stream:
        path_raw = json.load(stream)
        if "t" not in path_raw and "ang" not in path_raw and "vec" not in path_raw:
            raise ValueError("missing field: ang vec t")
        for t,ang,vec in zip(path_raw["t"],path_raw["ang"],path_raw["vec"]):
            path_se3.append(SE3.Trans(t)*SE3.AngleAxis(ang,vec))   
    return path_se3
#初始化点云
#初始化机器人轨迹
path_se3 = []
for i in range(100):
    path_se3.append(SE3(0.345,1.24,0.24+i*0.02))
#从.json文件中读取
# path_file = Path(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path/initial_se3_path_20260808_113501.json")
# LoadFromBezierFile(path_file)
# path_file = Path(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path/teleop_demo_20260801_120504.json")
path_file = Path(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path/teleop_demo_20260918_174113.json")
path_se3 = LoadFromTeletopFile(path_file)
#初始化攀爬专家
expert_complex = ExpertComplex()
expert_complex.LoadSE3(path_se3)
# q_des,adhesions = expert_complex.SetInit(default_pos)
q_init,adhesions = expert_complex.SetInitBySE3()
tau_ff = np.zeros((6,3),dtype=np.float32)

#初始化hex_climb环境
env_cfg = HexClimbCfg()
# env_cfg.terrain.mesh_type='plane'
# env_cfg.terrain.stl_files = os.path.join(LEGGED_GYM_ROOT_DIR,"resources/environments/sutructure1/complex_surface.STL")
env_cfg.terrain.stl_files = os.path.join(LEGGED_GYM_ROOT_DIR,"resources/environments/structure2/regular_dodecagon.STL")
env_cfg.env.num_envs = 1
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.push_robots = False
# env_cfg.init_state.pos=[0.6, 4.1, 0.2]
env_cfg.init_state.pos=path_se3[0].t.tolist()
q_wxyz = path_se3[0].UnitQuaternion().data[0]
# q_wxyz = SO3.Rz(np.pi).UnitQuaternion().data[0]
env_cfg.init_state.rot = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
#把expert complex计算得到的初值作为reset的时候的初值
q_init_flatten = q_init.flatten()
q_index = 0
for leg_name in expert_complex.leg_names:
    for joint_name in ["thigh","knee","ankle","foot"]:
        key = 'j_'+leg_name.lower()+"_"+joint_name
        if key in env_cfg.init_state.default_joint_angles.keys():
            env_cfg.init_state.default_joint_angles[key] = q_init_flatten[q_index]
            q_index += 1
args = get_args()
env_cfg,_ = update_cfg_from_args(env_cfg,None,args)
sim_params = {"sim":class_to_dict(env_cfg.sim)}
sim_params = parse_sim_params(args,sim_params)

env = HexClimb(env_cfg,sim_params,args.physics_engine,args.sim_device,args.headless)
device = args.sim_device
env.reset()
# default_pos = env.default_dof_pos[0].to("cpu").numpy().reshape(6,-1)[:,0:4]

print("adhesions=",adhesions)
for _ in range(10000):
    StepExpert(q_init,tau_ff,adhesions)
    time.sleep(0.1)

#在环境中绘制原始点云地图
# DrawPoints(point_map.points)
#在环境中绘制参考轨迹
points = []
for se3 in path_se3:
    points.append(se3.t)
DrawPoints(np.vstack(points),color=(0,1,0))
# DrawPoints(np.array([[0.95046371, 0.94990469, 1.48896125],[2.20247165, 2.81674108, 2.51181516],[0.03845775, 0.03841578, 0.03841838]]).T,color=(0,0,1))
groups_drawed = False
last_stance_group_index = expert_complex.stance_group_index

#记录扭矩信息
torques = []
counts = 0
#ros发布消息
while not env.gym.query_viewer_has_closed(env.viewer):
    _,_,_reward,_reset,_extra = StepExpert(q_des,tau_ff,adhesions)
    if counts >= 100*30:
        np.savez("/home/val/BIH_ws/legged_gym/logs/torques_data.npz",np.array(torques))
        print("save torques_data to file torques_data.npz")
        break
    counts += 1

    if _reset:
        print("reset robot and SetInit")
        # q_des,adhesions = expert_complex.SetInit(default_pos)
        q_des,adhesions = expert_complex.SetInitBySE3()
        tau_ff.fill(0.0)
        #先按照默认关节开始仿真一段时间等待稳定
        for _ in range(100):
            StepExpert(q_des,tau_ff,adhesions)
    else:
        _base_quat = env.base_quat[0].to("cpu").numpy()
        _base_pos = env.base_pos[0].to("cpu").numpy()
        # print("_base_pos=",_base_pos)
        # np.set_printoptions(precision=2, suppress=True)
        # print(f"dof_pos = {env.dof_pos[0].numpy()}")
        cur_se3 = SE3.Rt(UnitQuaternion(s=_base_quat[3],v=_base_quat[:3]).R,_base_pos)
        q_cur = env.dof_pos[0,env.dof_motor_drive_indices].to("cpu").numpy().reshape(6,3)
        q_torque = env.torques[0,env.dof_motor_drive_indices].to("cpu").numpy().reshape(6,3)
        # adhesion_force = abs(env.rb_forces[0,env.feet_indices,2].to("cpu").numpy())
        adhesion_force = abs(env.rb_forces[0,env.magnetic_indices,2].reshape(6,3).sum(dim=1).to("cpu").numpy())
        contact_force = abs(env.contact_forces[0,env.feet_indices,2].to("cpu").numpy())
        # print(f"real contact force=",contact_force)
        # print("torques\n",q_torque)
        torques.append(q_torque)
        
        #请求专家轨迹
        q_des,tau_ff,adhesions = expert_complex.RequestSingleStep(
            cur_se3,q_cur,q_torque,adhesion_force
        )
        #可视化专家轨迹
        if not groups_drawed:
            #将swing 和 stance 轨迹绘制出来
            if (expert_complex.B_e_traj_len > 1).any():
                for i in range(6):
                    points = expert_complex.interp_path_se3[-1]*expert_complex.kin._B2R(expert_complex.B_e_traj[i],i)
                    DrawPoints(points.T)
                    landing_points = expert_complex.W_landing_points.T
                    DrawPoints(landing_points,color=(1,0,0),point_size=0.03)
            
            #将质心轨迹绘制出来
            points = []
            for se3 in expert_complex.interp_path_se3:
                points.append(se3.t)
            points = np.vstack(points)
            DrawPoints(points,color=(0,1,1))
            groups_drawed = True
        if last_stance_group_index != expert_complex.stance_group_index:
            groups_drawed = False
            last_stance_group_index = expert_complex.stance_group_index 
