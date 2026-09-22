from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hex_v4.hex_climb_config import HexClimbCfg, HexClimbCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.actuator import Actuator
from legged_gym.envs.hex_v4.expert import ExpertClimb
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch, get_axis_params, quat_rotate_inverse,torch_rand_float
import trimesh
import torch
import numpy as np
import os
from typing import Optional

class HexClimb(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg:HexClimbCfg = cfg
        self.debug_viz=False
        self.foot_traj_vis=False
        self.actuator = Actuator(self.cfg,self.device)
        self.expert = ExpertClimb(self.cfg,self.device,self.cfg.env.num_envs)
        self.obs_scales = self.cfg.normalization.obs_scales
        #设置仿真属性，用于调整重力
        self.sim_params:gymapi.SimParams
        # rb_prop = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
        # print("------------------>rb props<----------------\n",rb_prop[0].mass)
        print("----------->using action scale={}<-----------".format(cfg.control.action_scale))

    def step(self, actions):
        """
        actions中前24位为关节期望位置，后6位为电磁铁吸附开关共30维度 num_envs * 30
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        return self._step_with_commands()

    def step_q_tao(self,q_des:torch.Tensor,tau_ff:torch.Tensor,adhesions:torch.Tensor):
        """以绝对关节位置和主电机前馈扭矩推进一个控制步。

        Args:
            q_des: ``(num_envs, 24)``，六腿四个位置关节的绝对目标角度。
            tau_ff: ``(num_envs, 18)``，六腿前三个主电机的前馈扭矩。
            adhesions: ``(num_envs, 6)``，吸附开关，非零表示吸附。
        """
        q_des = torch.as_tensor(q_des,dtype=torch.float32,device=self.device)
        tau_ff = torch.as_tensor(tau_ff,dtype=torch.float32,device=self.device)
        adhesions = torch.as_tensor(adhesions,dtype=torch.float32,device=self.device)
        expected_q_shape = (self.num_envs,len(self.dof_drive_indices))
        expected_tau_shape = (self.num_envs,len(self.dof_motor_drive_indices))
        expected_adhesion_shape = (self.num_envs,6)
        if q_des.shape != expected_q_shape:
            raise ValueError(f"q_des must have shape {expected_q_shape}, got {tuple(q_des.shape)}")
        if tau_ff.shape != expected_tau_shape:
            raise ValueError(f"tau_ff must have shape {expected_tau_shape}, got {tuple(tau_ff.shape)}")
        if adhesions.shape != expected_adhesion_shape:
            raise ValueError(
                f"adhesions must have shape {expected_adhesion_shape}, got {tuple(adhesions.shape)}"
            )
        if not (torch.isfinite(q_des).all() and torch.isfinite(tau_ff).all() and torch.isfinite(adhesions).all()):
            raise ValueError("q_des, tau_ff, and adhesions must be finite")

        # self.actions 仍用于观测中的 last_actions 以及吸附逻辑；位置控制本身
        # 则由 q_des 直接驱动，不经过 action clip 或 action scale。
        self.actions = torch.zeros_like(self.actions)
        self.actions[:,:24] = (
            q_des-self.default_dof_pos[:,self.dof_drive_indices]
        )/self.cfg.control.action_scale
        self.actions[:,24:30] = adhesions
        return self._step_with_commands(q_des=q_des,tau_ff=tau_ff)

    def _step_with_commands(self,q_des:Optional[torch.Tensor]=None,
                            tau_ff:Optional[torch.Tensor]=None):
        """执行共享的物理步进；q_des 为 None 时沿用缩放 action 接口。"""
        # step physics and render each frame
        self.render()

        #每0.02s计算一次吸附力
        self.rb_forces = self._compute_adhesions(self.actions)

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions,q_des=q_des,tau_ff=tau_ff)
            # np.set_printoptions(precision=2, suppress=True)
            # print(f"torques \n {self.torques[0].numpy().reshape(6,7)}")
            
            self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self.torques))
            self.gym.apply_rigid_body_force_at_pos_tensors(self.sim,
                                                gymtorch.unwrap_tensor(self.rb_forces),
                                                gymtorch.unwrap_tensor(self.pos_rb_forces),
                                                gymapi.LOCAL_SPACE)
            self.gym.simulate(self.sim)
            if self.device=='cpu':
                self.gym.fetch_results(self.sim,True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # print("contact force=\n",self.contact_forces[:,self.feet_indices,2])
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf,-clip_obs,clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,-clip_obs,clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def post_physics_step(self):
        #增加了加速度的计算，重写
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.episode_length_buf+=1
        self.common_step_counter+=1
        # print("contact force=",self.contact_forces[:,self.feet_indices,2])
        # print("rb forces=",self.rb_forces[:,self.feet_indices,2])

        self.base_quat[:]=self.root_states[:,3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat,self.root_states[:,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,self.root_states[:,10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat,self.gravity_vec)

        root_acc = ((self.root_states[:,7:10] -  self.last_root_vel[:,:3])/self.dt)/9.81 - self.gravity_vec
        self.base_lin_acc = quat_rotate_inverse(self.base_quat,root_acc)

        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:]=self.actions[:]
        self.last_dof_vel[:]=self.dof_vel[:]
        self.last_root_vel[:]=self.root_states[:,7:13]


    def check_termination(self):
        """
        由于刚开始的时候，机器人是悬浮在空中，因此刚开始的几个时间步是没有足端接触地面的，因此只对episode_length_buf大于1s的进行reset

        结束条件：身体发生碰撞 | 所有足端与平面都不接触了(F<10N,为了避免估计误差) | 到规定时间"""
        reset_collide = torch.any(torch.norm(self.contact_forces[:,self.termination_contact_indices,:],dim=-1)>1.0, dim=1)
        reset_contact = torch.all(torch.norm(self.contact_forces[:,self.feet_indices,:],dim=-1)<10.0, dim=1)
        initial_condition = self.episode_length_buf > self.cfg.init_state.buffer_time/self.dt
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        
        self.reset_buf = (reset_collide | reset_contact | self.time_out_buf) & initial_condition

        #测试用
        if self.reset_buf.any():
            print(f"reset because reset_collide={reset_collide}, reset_contact={reset_contact}, timeout={self.time_out_buf}")
        # self.reset_buf = reset_collide | self.time_out_buf
    
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_lin_acc * self.obs_scales.lin_acc,
            self.last_actions,
            (self.dof_pos[:,self.dof_drive_indices]-self.default_dof_pos[:,self.dof_drive_indices]) * self.obs_scales.dof_pos,
            self.dof_vel[:,self.dof_motor_drive_indices] * self.obs_scales.dof_vel,
            self.commands * self.commands_scale,
            self.contact_forces[:,self.feet_indices,2] * self.obs_scales.contact_force
        ),dim=1)

        if self.cfg.noise.add_noise:
            self.obs_buf += torch.rand_like(self.noise_scale_vec) * self.noise_scale_vec
    
    def get_expert_actions(self,action_scaled=True):
        """
        action_scaled=True返回的是经过减去默认值和缩放后的action

        action_scaled=False返回的是关节的期望值，没有经过任何处理
        """
        #获取专家动作 
        command = torch.stack([self.reset_buf.clone(),
                               self.commands[:,0],
                               self.commands[:,1],
                               torch.zeros_like(self.commands[:,0]),
                               self.commands[:,2]],dim=1)
        q_cur = self.dof_pos[:,self.dof_motor_drive_indices].clone()
        q_dot_cur = self.dof_vel[:,self.dof_motor_drive_indices].clone()
        adhesion_force = self.rb_forces[:,self.feet_indices,2].clone().abs()
        contact_force = self.contact_forces[:,self.feet_indices,2].clone().abs()
        
        # 动作中包含吸附力
        self.expert_actions[:,24:30], expert_dofs = self.expert.ProcessCommand(command,q_cur,q_dot_cur,adhesion_force,contact_force)
        
        #从动作中取消吸附力
        # adhesion_force[self.adhesions]=self.cfg.control.suction_force_max
        # self.adhesions, expert_dofs = self.expert.ProcessCommand(command,q_cur,q_dot_cur,adhesion_force,contact_force)

        # print("expert dof pos.shape=",expert_dofs.shape)
        # print("adhesion_force=",adhesion_force)
        if action_scaled:
            self.expert_actions[:,:24] = (expert_dofs-self.default_dof_pos[:,self.dof_drive_indices])/self.cfg.control.action_scale
        else:
            self.expert_actions[:,:24] = expert_dofs
        return self.expert_actions.detach()

    """以下create_sim是用于面向复杂结构中攀爬，创建复杂地形展开的"""
    # def create_sim(self):
    #     self.up_axis_idx = 2
    #     self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
    #     #创造STL文件创建的地形
    #     #先创建一个地面
    #     # self._create_ground_plane()
    #     #使用trimesh读取STL文件 读取文件对应的边和三角面片 然后使用这些加载到isaacgym中
    #     file = self.cfg.terrain.stl_files
    #     # file = os.path.join(LEGGED_GYM_ROOT_DIR,"resources/environments/sutructure1/complex_surface.STL")
    #     # file = "/home/val/BIH_ws/legged_gym/resources/environments/sutructure1/complex_surface.STL"
    #     print(f"file={file}")
    #     _mesh = trimesh.load(file,force="mesh")
    #     vertices = np.asarray(_mesh.vertices,dtype=np.float32)
    #     triangles = np.asarray(_mesh.faces,dtype=np.uint32)
    #     tm_params = gymapi.TriangleMeshParams()
    #     tm_params.nb_vertices = vertices.shape[0]
    #     tm_params.nb_triangles = triangles.shape[0]

    #     tm_params.transform.p.x = 0
    #     tm_params.transform.p.y = 0
    #     tm_params.transform.p.z = 0

    #     tm_params.static_friction = self.cfg.terrain.static_friction
    #     tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction

    #     self.gym.add_triangle_mesh(self.sim,vertices.flatten(order='C'),triangles.flatten(order='C'),tm_params)


    #     self._create_envs()

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        #创建质心位置变量
        self.base_pos = self.root_states[:,0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        #额外初始化张量
        self.expert_actions = torch.zeros_like(self.actions)
        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        self.rb_forces = torch.zeros_like(self.contact_forces) #用于给足端施加吸力
        self.pos_rb_forces = torch.zeros_like(self.contact_forces) #指定给足施加吸附力的位置
        env_indices = torch.arange(self.num_envs,dtype=torch.long,device=self.device).unsqueeze(1)
        self.pos_rb_forces[env_indices,self.magnetic_indices[:,0].unsqueeze(0),:] = torch.tensor([0,0.02,-0.009],device=self.device,dtype=torch.float)
        self.pos_rb_forces[env_indices,self.magnetic_indices[:,1].unsqueeze(1),:] = torch.tensor([0.02 * 3**0.5,-0.01,0],device=self.device,dtype=torch.float)
        self.pos_rb_forces[env_indices,self.magnetic_indices[:,2].unsqueeze(1),:] = torch.tensor([-0.02 * 3**0.5,-0.01,0],device=self.device,dtype=torch.float)
        self.dof_pos_des = torch.zeros_like(self.dof_pos) #这里是关节期望的角度，包含了被动关节，其期望值一直为0
        # self.adhesions = torch.zeros(self.num_envs,6,dtype=torch.bool,device=self.device,requires_grad=False)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        #为了方便期望关节位置的计算，还需要计算一个没有被动关节的默认关节角度
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _create_envs(self):
        #这里要修改机器人的初始位姿，增加了电机舵机对应关节驱动的索引
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        print("begin loading asset")
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        print("end loading asset")
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        print("------>dof props<------------\n",dof_props_asset.dtype.names)
        print(dof_props_asset)
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print("dof names\n",self.dof_names)
        print(f"dof_names length={len(self.dof_names)}")
        print(f"body_names={body_names}")
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # pos = self.env_origins[i].clone()
            #给初始位置增加随机性，暂时注释掉
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            # start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # print("rigid shape friction\n")
            # for i,s in enumerate(rigid_shape_props):
            #     print(f"i={i}; s.frictions={s.friction}")
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        #额外增加主动驱动自由度的索引
        dof_drive_names=[]
        dof_motor_drive_names=[]
        for name in self.cfg.asset.dof_drive:
            dof_drive_names.extend([s for s in self.dof_names if name in s])
        for name in self.cfg.asset.dof_motor_drive:
            dof_motor_drive_names.extend([s for s in self.dof_names if name in s])
        # print(f"dof_drive_name={dof_drive_names}")
        self.dof_drive_indices=torch.zeros(len(dof_drive_names),dtype=torch.long,device=self.device,requires_grad=False)
        self.dof_motor_drive_indices=torch.zeros(len(dof_motor_drive_names),dtype=torch.long,device=self.device,requires_grad=False)

        for i, name in enumerate(dof_drive_names):
            self.dof_drive_indices[i]=self.gym.find_actor_dof_handle(self.envs[0],self.actor_handles[0],name)
        for i, name in enumerate(dof_motor_drive_names):
            self.dof_motor_drive_indices[i]=self.gym.find_actor_dof_handle(self.envs[0],self.actor_handles[0],name)

        self.dof_drive_indices=torch.sort(self.dof_drive_indices)[0] #排序，保证自由度的顺序是一条腿一条腿来的，不排序就是按照所有的thigh，所有的knee，..这样
        self.dof_motor_drive_indices=torch.sort(self.dof_motor_drive_indices)[0]

        #额外增加用于模拟电磁铁的三个关节的索引 suck toe empty
        magnetic_names = []
        for name in ["suck","toe","empty"]:
            magnetic_names.extend([s for s in body_names if name in s])
        self.magnetic_indices = torch.zeros(len(magnetic_names),dtype=torch.long,device=self.device,requires_grad=False)
        for i,name in enumerate(magnetic_names):
            self.magnetic_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],self.actor_handles[0],name)
        self.magnetic_indices = torch.sort(self.magnetic_indices)[0]
        self.magnetic_indices = self.magnetic_indices.view(-1,3)
        print("magnetic_indices\n",self.magnetic_indices)

    def _get_noise_scale_vec(self):
        noise_scalse = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        if self.privileged_obs_buf is None:
            noise_vec = torch.zeros_like(self.obs_buf[0])
        else:
            noise_vec = torch.zeros_like(self.privileged_obs_buf[0])

        noise_vec[:3] = noise_scalse.ang_vel * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scalse.lin_acc * self.obs_scales.lin_acc
        noise_vec[6:36] = 0.0
        noise_vec[36:60] = noise_scalse.dof_pos * self.obs_scales.dof_pos
        noise_vec[60:78] = noise_scalse.dof_vel * self.obs_scales.dof_vel
        noise_vec[78:81] = 0.0
        noise_vec[81:87] = noise_scalse.contact_force * self.obs_scales.contact_force

        # noise_vec[:3] = noise_scalse.ang_vel * self.obs_scales.ang_vel
        # noise_vec[3:6] = noise_scalse.lin_acc * self.obs_scales.lin_acc
        # noise_vec[6:30] = 0.0
        # noise_vec[30:54] = noise_scalse.dof_pos * self.obs_scales.dof_pos
        # noise_vec[54:72] = noise_scalse.dof_vel * self.obs_scales.dof_vel
        # noise_vec[72:75] = 0.0

        # noise_vec[84:87] = 0.0
        noise_vec = (noise_level*noise_vec).unsqueeze(0)
        return noise_vec

    def _compute_torques(self,actions:torch.Tensor,q_des:Optional[torch.Tensor]=None,
                         tau_ff:Optional[torch.Tensor]=None)->torch.Tensor:
        #这里的actions有30个维度，前6*4控制关节角度，需要计算的pos_err还包含了被动关节，这些关节的期望值都设为0
        # des_pos=torch.zeros_like(self.dof_pos).reshape(self.num_envs,6,7)
        if q_des is None:
            q_des = (
                actions[:,:24]*self.cfg.control.action_scale
                +self.default_dof_pos[:,self.dof_drive_indices]
            )
        self.dof_pos_des[:,self.dof_drive_indices]= q_des
        pos_err = self.dof_pos_des-self.dof_pos
        vel_err = -self.dof_vel
        torques = self.actuator.get_torques(pos_err,vel_err)
        if tau_ff is not None:
            torques = torques.clone()
            torques[:,self.dof_motor_drive_indices] += tau_ff
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        # print("torque_limits=",self.torque_limits)
        return torques

    def _compute_adhesions(self, actions:torch.Tensor)->torch.Tensor:
        contacts = torch.norm(self.contact_forces[:,self.feet_indices,:],dim=-1) > 1.0
        contacts_filt = contacts | self.last_contacts
        self.last_contacts = contacts

        #actions的后六位是吸附力的大小
        adhesions = actions[:,24:30].clone()

        adhesions[actions[:,24:30] > 0.8] = -5.0
        adhesions[actions[:,24:30] <= 0.8] = 1.0
        # #选择与接触面有接触的，contact_force的norm大于1.0

        # #为了使机器人初始化后能顺利先贴附到墙面上，在仿真开始的0.5s内不考虑接触情况，直接给adhesion的足端施加最大的吸附力
        max_force_mask = self.episode_length_buf < self.cfg.init_state.buffer_time/self.dt
        contacts_filt[max_force_mask] = True 
        # #有重新设置的环境&这个脚被设置为吸附
        adhesions[max_force_mask.unsqueeze(1) & (adhesions==-5.0)] = -1000 #设置一个较大的值，这样就可以实现一次就设置为最大的吸附力

        # #adhesions吸附时为-5.0，意味着吸附时的吸力变化为设置中的5倍，同时给吸附和释放都增加±5N的不确定性
        # self.rb_forces[:,self.feet_indices,2] += adhesions * (self.cfg.control.suction_force_delt) #+ 10*(torch.rand_like(adhesions)-0.5)
        # #如果此时没有接触，那么直接设置rb_forces为0
        # self.rb_forces[:,self.feet_indices,2] *= contacts_filt.float()

        # self.rb_forces = torch.clip(self.rb_forces,-self.cfg.control.suction_force_max,0.0)
        magnetic_indices = self.magnetic_indices.reshape(-1)
        adhesions_repeated = torch.repeat_interleave(adhesions,3,dim=1) # N,6 -> N,18
        contacts_filt_repeated = torch.repeat_interleave(contacts_filt,3,dim=1) #N,6 -> N,18
        self.rb_forces[:,magnetic_indices,2] += adhesions_repeated*(self.cfg.control.suction_force_delt/3.0)
        self.rb_forces[:,magnetic_indices,2] *= contacts_filt_repeated.float()
        self.rb_forces = torch.clip(self.rb_forces,-self.cfg.control.suction_force_max/3.0,0.0)
        print("robot contact force norm=",torch.norm(self.contact_forces[:,self.feet_indices,:],dim=-1))
        # print("rb_forces \n ",self.rb_forces[:,magnetic_indices,2].reshape(6,3))
        return self.rb_forces
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_root_vel[env_ids] = 0.0
        if len(env_ids) !=0:
            self.get_expert_actions()

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _resample_commands(self,env_ids):
        #速度变小了，设置为0的速度条件变低，设置为0.05m/s
        v_max=[]
        for i, key in enumerate(['lin_vel_x','lin_vel_y','ang_vel_yaw']):
            self.commands[env_ids,i]=torch_rand_float(self.command_ranges[key][0],self.command_ranges[key][1],(len(env_ids),1),device=self.device).squeeze(1)
            v_max.append(self.command_ranges[key][1])
        v_max = torch.tensor(v_max,device=self.device,requires_grad=False).unsqueeze(0) # 1*3
        cmd = self.commands[env_ids, :3]
        cmd = torch.where(cmd.abs()<0.1*v_max, 0.0, (torch.where(cmd.abs()>0.9*v_max, cmd.sign()*v_max, cmd)) )
        # cmd *= (torch.norm(cmd,dim=1)>0.1).unsqueeze(1)
        self.commands[env_ids,:3] = cmd

    def _reset_dofs(self,env_ids):
        self.dof_pos[env_ids,:]=self.default_dof_pos
        self.dof_vel[env_ids,:]=0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))
    # def _reset_root_states(self, env_ids):
    #     # base position
    #     if self.custom_origins:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #         self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
    #     else:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self.root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reward_action_rate(self):
        #只惩罚驱动关节的动作频率
        return torch.sum(torch.square(self.last_actions[:,:24]-self.actions[:,:24]),dim=1)

    def _reward_stand_still(self):
        return torch.sum(torch.abs(self.dof_pos-self.default_dof_pos),dim=1) * (torch.norm(self.commands[:,:3],dim=1)<=0.05)
    
