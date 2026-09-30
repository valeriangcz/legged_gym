from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hex_v4.hex_climb_config import HexClimbCfg, HexClimbCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.actuator import Actuator
from legged_gym.envs.hex_v4.expert import ExpertClimb
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import to_torch, quat_apply, quat_rotate_inverse,torch_rand_float
import trimesh
import torch
import numpy as np
import math
import os
from typing import Callable, Optional

class HexClimb(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg:HexClimbCfg = cfg
        self.debug_viz=False
        self.foot_traj_vis=False
        self.actuator = Actuator(self.cfg,self.device)
        self.expert = ExpertClimb(self.cfg,self.device,self.cfg.env.num_envs)
        self.expert.configure_mass_model(
            self.body_names, self.actor_body_masses, self.actor_body_com_local,
        )
        # Keep the command that was used for the current control step separate
        # from ``self.torques``. The latter also contains position-control
        # torque and is clipped to the actuator limits.
        self.recording_feedforward_torque = torch.zeros(
            self.num_envs, len(self.dof_motor_drive_indices),
            dtype=torch.float32, device=self.device,
        )
        gravity_visualization_length = float(
            getattr(self.cfg.env, "gravity_visualization_length", 0.25)
        )
        self.gravity_axes_geometry = gymutil.AxesGeometry(gravity_visualization_length)
        self.gravity_tip_geometry = gymutil.WireframeSphereGeometry(
            0.012, 6, 6, None, color=(0.1, 0.8, 1.0)
        )
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

    def step_q_tao(self,q_des:torch.Tensor,tau_ff:torch.Tensor,adhesions:torch.Tensor,
                   physics_step_callback:Optional[Callable[["HexClimb"],None]]=None):
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
        return self._step_with_commands(q_des=q_des,tau_ff=tau_ff,
                                        physics_step_callback=physics_step_callback)

    def _step_with_commands(self,q_des:Optional[torch.Tensor]=None,
                            tau_ff:Optional[torch.Tensor]=None,
                            physics_step_callback:Optional[Callable[["HexClimb"],None]]=None):
        """执行共享的物理步进；q_des 为 None 时沿用缩放 action 接口。"""
        # A callback runs once per physical step below. Snapshot the command
        # once per control step so every sample gets the exact feedforward
        # torque supplied for this control interval. Ordinary policy actions
        # have no feedforward component.
        if tau_ff is None:
            self.recording_feedforward_torque.zero_()
        else:
            self.recording_feedforward_torque.copy_(tau_ff)

        # step physics and render each frame
        self.render()

        #每0.01s计算一次吸附力
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
            if physics_step_callback is not None:
                self._refresh_recording_tensors()
                physics_step_callback(self)
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

        world_lin_acc = (self.root_states[:,7:10] - self.last_root_vel[:,:3])/self.dt
        if self.gravity_norm > 0.0:
            # Keep the IMU-style observation expressed in units of the
            # configured gravity magnitude rather than assuming earth gravity.
            root_acc = world_lin_acc/self.gravity_norm - self.gravity_vec
        else:
            # A gravity-normalized acceleration is undefined in zero gravity.
            # Preserve the physical world-frame acceleration instead.
            root_acc = world_lin_acc - self.gravity_vec
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

        if self.viewer is not None and getattr(self.cfg.env, "visualize_gravity", False):
            self._draw_gravity_vectors()


    def check_termination(self):
        """
        由于刚开始的时候，机器人是悬浮在空中，因此刚开始的几个时间步是没有足端接触地面的，因此只对episode_length_buf大于1s的进行reset

        结束条件：身体发生碰撞 | 所有足端与平面都不接触了(F<10N,为了避免估计误差) | 到规定时间"""
        # reset_collide = torch.any(torch.norm(self.contact_forces[:,self.termination_contact_indices,:],dim=-1)>1.0, dim=1)
        reset_collide = False
        reset_contact = torch.all(torch.norm(self.contact_forces[:,self.feet_indices,:],dim=-1)<1.0, dim=1)
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
    
    def get_expert_commands(self):
        """Return absolute expert commands, including main-motor feedforward.

        Returns:
            q_des: ``(num_envs, 24)`` absolute position targets.
            tau_ff: ``(num_envs, 18)`` feedforward torque for main motors.
            adhesions: ``(num_envs, 6)`` adsorption switches.
        """
        command = torch.stack([self.reset_buf.clone(),
                               self.commands[:,0],
                               self.commands[:,1],
                               torch.zeros_like(self.commands[:,0]),
                               self.commands[:,2]],dim=1)
        q_cur = self.dof_pos[:,self.dof_motor_drive_indices].clone()
        q_dot_cur = self.dof_vel[:,self.dof_motor_drive_indices].clone()
        q_drive_cur = self.dof_pos[:,self.dof_drive_indices].clone()
        adhesion_force = self.rb_forces[:,self.feet_indices,2].clone().abs()
        contact_force = self.contact_forces[:,self.feet_indices,2].clone().abs()
        gravity_R = self.projected_gravity*self.gravity_norm
        adhesions, q_des, tau_ff = self.expert.ProcessCommand(
            command,q_cur,q_dot_cur,q_drive_cur,adhesion_force,contact_force,gravity_R
        )
        return q_des.detach(), tau_ff.detach(), adhesions.detach()

    def get_expert_actions(self,action_scaled=True):
        """
        action_scaled=True返回的是经过减去默认值和缩放后的action

        action_scaled=False返回的是关节的期望值，没有经过任何处理
        """
        expert_dofs, _, adhesions = self.get_expert_commands()
        # 动作中包含吸附力；前馈扭矩通过 get_expert_commands 单独提供，
        # 以保持策略和已有专家数据的 30 维动作接口不变。
        self.expert_actions[:,24:30] = adhesions

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
        if self.recording_sensors_enabled:
            rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
            force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.recording_sensors_enabled:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        #创建质心位置变量
        self.base_pos = self.root_states[:,0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        if self.recording_sensors_enabled:
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
                self.num_envs, self.num_bodies, 13
            )
            self.force_sensor_forces = gymtorch.wrap_tensor(force_sensor_tensor).view(
                self.num_envs, self.record_joint_sensor_count, 6
            )
            self.dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_dof
            )

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        gravity_world = torch.as_tensor(
            self.cfg.sim.gravity, dtype=torch.float, device=self.device
        ).reshape(-1)
        if gravity_world.numel() != 3:
            raise ValueError(
                "cfg.sim.gravity must contain exactly three world-frame components"
            )
        if not torch.isfinite(gravity_world).all():
            raise ValueError("cfg.sim.gravity must contain only finite values")

        self.gravity_norm = torch.linalg.vector_norm(gravity_world)
        if self.gravity_norm > 0.0:
            gravity_unit = gravity_world/self.gravity_norm
        else:
            # The direction is undefined in zero gravity; use a zero vector so
            # projected gravity and expert gravity feedforward remain finite.
            gravity_unit = torch.zeros_like(gravity_world)
        self.gravity_vec = gravity_unit.unsqueeze(0).repeat((self.num_envs, 1))
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
        # Both advanced indices must broadcast to (num_envs, 6).  Keep the
        # magnetic-point dimension in axis 1 for every one of the three cup
        # force locations; ``unsqueeze(1)`` here would instead produce the
        # incompatible shapes (num_envs, 1) and (6, 1).
        magnetic_force_positions = torch.tensor(
            [[0.0, 0.03, -0.009],
             [0.03 * 3**0.5, -0.015, -0.009],
             [-0.03 * 3**0.5, -0.015, -0.009]],
            device=self.device, dtype=torch.float,
        )
        for point_index, position in enumerate(magnetic_force_positions):
            self.pos_rb_forces[
                env_indices, self.magnetic_indices[:,point_index].unsqueeze(0), :
            ] = position
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

    def _refresh_recording_tensors(self):
        """Refresh all tensors consumed by a physics-step recording callback."""
        if not self.recording_sensors_enabled:
            return
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def _draw_gravity_vectors(self):
        """Draw a world-frame gravity direction marker from every base origin.

        ``gravity_vec`` is already the configured unit gravity direction in
        world axes.  The visual length is intentionally independent of the
        gravity magnitude so Earth, Mars and tilted-gravity cases remain easy
        to compare in the viewer.  The built-in AxesGeometry local +Z axis is
        aligned to gravity; the built-in WireframeSphereGeometry marks its tip.
        """
        if self.gravity_norm <= 0.0:
            return
        arrow_length = float(getattr(self.cfg.env, "gravity_visualization_length", 0.25))
        if arrow_length <= 0.0:
            return

        self.gym.clear_lines(self.viewer)
        origins = self.root_states[:,:3].detach().cpu().numpy()
        directions = self.gravity_vec.detach().cpu().numpy()

        for env_id, (origin, direction) in enumerate(zip(origins, directions)):
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0.0:
                continue
            direction = direction/direction_norm
            end = origin + arrow_length*direction
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            rotation_axis = np.cross(z_axis, direction)
            axis_norm = np.linalg.norm(rotation_axis)
            dot_product = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            if axis_norm < 1e-6:
                # +Z and gravity are parallel (identity) or anti-parallel
                # (180 degrees about +X).
                orientation = (gymapi.Quat()
                               if dot_product >= 0.0
                               else gymapi.Quat(1.0, 0.0, 0.0, 0.0))
            else:
                rotation_axis /= axis_norm
                orientation = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(*rotation_axis), math.acos(dot_product)
                )

            axes_pose = gymapi.Transform()
            axes_pose.p = gymapi.Vec3(*origin)
            axes_pose.r = orientation
            gymutil.draw_lines(
                self.gravity_axes_geometry, self.gym, self.viewer,
                self.envs[env_id], axes_pose
            )
            tip_pose = gymapi.Transform()
            tip_pose.p = gymapi.Vec3(*end)
            gymutil.draw_lines(
                self.gravity_tip_geometry, self.gym, self.viewer,
                self.envs[env_id], tip_pose
            )

    @staticmethod
    def _rotate_inverse_batched(quat:torch.Tensor, vector:torch.Tensor)->torch.Tensor:
        return quat_rotate_inverse(
            quat.reshape(-1,4), vector.reshape(-1,3)
        ).reshape_as(vector)

    def get_recording_telemetry(self):
        """Return recording tensors after a physics step.

        This method is intentionally unavailable outside the explicit recording
        mode so normal training does not acquire extra Isaac Gym tensors.
        Wrenches are force-first: ``[Fx, Fy, Fz, Mx, My, Mz]``.
        """
        if not self.recording_sensors_enabled:
            raise RuntimeError(
                "recording sensors are disabled; set cfg.env.enable_recording_sensors=True before creation"
            )

        body_state = self.rigid_body_states
        parent_state = body_state[:,self.record_joint_parent_indices,:]
        child_state = body_state[:,self.record_joint_child_indices,:]
        raw_wrench_world = self.force_sensor_forces
        raw_force_world = raw_wrench_world[...,:3]
        raw_moment_world = raw_wrench_world[...,3:]

        # Move each sensor wrench from its child-link origin to the parent-link
        # origin, then rotate it into the parent link's current local axes.
        arm_world = child_state[...,:3]-parent_state[...,:3]
        moment_at_parent_world = raw_moment_world + torch.cross(
            arm_world, raw_force_world, dim=-1
        )
        force_parent = self._rotate_inverse_batched(
            parent_state[...,3:7], raw_force_world
        )
        moment_parent = self._rotate_inverse_batched(
            parent_state[...,3:7], moment_at_parent_world
        )
        # Isaac Gym reports the force on the sensor's child side.  The public
        # signal is the equal-and-opposite load of that downstream assembly on
        # its parent link.
        joint_wrench_parent = -torch.cat((force_parent,moment_parent),dim=-1)

        magnetic_flat = self.magnetic_indices.reshape(-1)
        magnetic_state = body_state[:,magnetic_flat,:].view(self.num_envs,6,3,13)
        magnetic_force_local = self.rb_forces[:,magnetic_flat,:].view(
            self.num_envs,6,3,3
        )
        magnetic_pos_local = self.pos_rb_forces[:,magnetic_flat,:].view(
            self.num_envs,6,3,3
        )
        magnetic_quat = magnetic_state[...,3:7]
        magnetic_force_world = quat_apply(
            magnetic_quat.reshape(-1,4), magnetic_force_local.reshape(-1,3)
        ).view(self.num_envs,6,3,3)
        magnetic_point_world = magnetic_state[...,:3] + quat_apply(
            magnetic_quat.reshape(-1,4), magnetic_pos_local.reshape(-1,3)
        ).view(self.num_envs,6,3,3)
        cup_state = body_state[:,self.record_cup_reference_indices,:]
        cup_position_world = cup_state[...,:3]
        adhesion_force_world = magnetic_force_world.sum(dim=2)
        adhesion_moment_world = torch.cross(
            magnetic_point_world-cup_position_world.unsqueeze(2),
            magnetic_force_world, dim=-1
        ).sum(dim=2)
        surface_force_world = self.contact_forces[:,magnetic_flat,:].view(
            self.num_envs,6,3,3
        ).sum(dim=2)

        body_quat = body_state[...,3:7]
        local_com = self.record_body_com_local.unsqueeze(0).expand(
            self.num_envs,-1,-1
        )
        com_offset_world = quat_apply(
            body_quat.reshape(-1,4), local_com.reshape(-1,3)
        ).view(self.num_envs,self.num_bodies,3)
        body_com_velocity = body_state[...,7:10] + torch.cross(
            body_state[...,10:13], com_offset_world, dim=-1
        )
        masses = self.record_body_masses.unsqueeze(-1)
        com_velocity_world = (body_com_velocity*masses).sum(dim=1)/masses.sum(dim=1)
        com_velocity_body = quat_rotate_inverse(
            self.root_states[:,3:7], com_velocity_world
        )

        return {
            "joint_position": self.dof_pos,
            "joint_velocity": self.dof_vel,
            "actuator_torque": self.torques,
            "feedforward_torque": self.recording_feedforward_torque,
            "dof_generalized_force": self.dof_force,
            "joint_sensor_wrench_world_raw": raw_wrench_world,
            "joint_sensor_origin_world": child_state[...,:3],
            "joint_parent_pose_world": parent_state[...,:7],
            "joint_wrench_parent": joint_wrench_parent,
            "com_velocity_world": com_velocity_world,
            "com_velocity_body": com_velocity_body,
            "desired_velocity_body": self.commands[:,:3],
            "adhesion_command_wrench_world": torch.cat(
                (adhesion_force_world,adhesion_moment_world), dim=-1
            ),
            "surface_force_world": surface_force_world,
            "cup_interface_wrench_parent": joint_wrench_parent.view(
                self.num_envs,6,5,6
            )[:,:,-1,:],
            "cup_reference_pose_world": cup_state[...,:7],
        }

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
        self.body_names = body_names
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

        self.recording_sensors_enabled = bool(
            getattr(self.cfg.env, "enable_recording_sensors", False)
        )
        self._record_mass_scaling_active = (
            getattr(self.cfg.env, "record_mass_scales", None) is not None
        )
        configured_mass_scales = getattr(self.cfg.env, "record_mass_scales", None)
        if self._record_mass_scaling_active:
            if len(configured_mass_scales) != self.num_envs:
                raise ValueError(
                    "record_mass_scales must contain one value for every environment"
                )
            if any((not np.isfinite(scale)) or scale <= 0.0
                   for scale in configured_mass_scales):
                raise ValueError("record_mass_scales values must be finite and positive")

        self.record_joint_sensor_specs = []
        if self.recording_sensors_enabled:
            # The sensor frame is placed at the child-link origin.  During
            # recording the wrench is shifted to, and expressed in, the parent
            # link frame before it is exposed to the caller.
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = False
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = True
            sensor_pose = gymapi.Transform()
            asset_leg_order = ["rf", "rm", "rb", "lf", "lm", "lb"]
            for leg in asset_leg_order:
                joint_links = [
                    ("thigh", "body", f"l_{leg}_thigh"),
                    ("knee", f"l_{leg}_thigh", f"l_{leg}_knee"),
                    ("ankle", f"l_{leg}_knee", f"l_{leg}_ankle"),
                    ("foot", f"l_{leg}_ankle", f"l_{leg}_foot"),
                    # This interface carries ball1, ball2, suck, toe and
                    # empty as one downstream cup assembly.
                    ("ball", f"l_{leg}_foot", f"l_{leg}_ball1"),
                ]
                for joint, parent_name, child_name in joint_links:
                    if child_name not in body_names or parent_name not in body_names:
                        raise RuntimeError(
                            f"recording sensor body is missing: {parent_name} -> {child_name}"
                        )
                    self.gym.create_asset_force_sensor(
                        robot_asset, body_names.index(child_name), sensor_pose, sensor_props
                    )
                    self.record_joint_sensor_specs.append({
                        "label": f"{leg}_{joint}",
                        "parent": parent_name,
                        "child": child_name,
                    })
        self.record_joint_sensor_count = len(self.record_joint_sensor_specs)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        actor_body_masses = []
        actor_body_coms = None
        fixed_friction = getattr(self.cfg.env, "record_fixed_friction", None)
        if fixed_friction is not None:
            if not np.isfinite(fixed_friction) or fixed_friction < 0.0:
                raise ValueError("record_fixed_friction must be finite and non-negative")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # pos = self.env_origins[i].clone()
            #给初始位置增加随机性，暂时注释掉
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            # start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            if fixed_friction is not None:
                for shape_prop in rigid_shape_props:
                    shape_prop.friction = float(fixed_friction)
            # print("rigid shape friction\n")
            # for i,s in enumerate(rigid_shape_props):
            #     print(f"i={i}; s.frictions={s.friction}")
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            if self._record_mass_scaling_active:
                for body_prop in body_props:
                    body_prop.mass *= float(configured_mass_scales[i])
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # Store final actor properties for ExpertClimb even outside recording:
            # they include domain randomization and any recording mass scale.
            actor_body_masses.append([body_prop.mass for body_prop in body_props])
            if actor_body_coms is None:
                actor_body_coms = [
                    [body_prop.com.x, body_prop.com.y, body_prop.com.z]
                    for body_prop in body_props
                ]
            if self.recording_sensors_enabled:
                self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.actor_body_masses = torch.tensor(
            actor_body_masses, dtype=torch.float32, device=self.device
        )
        self.actor_body_com_local = torch.tensor(
            actor_body_coms, dtype=torch.float32, device=self.device
        )
        self.actor_total_masses = self.actor_body_masses.sum(dim=1)
        if self.recording_sensors_enabled or self._record_mass_scaling_active:
            # Keep the recording interface stable while sharing the exact same
            # final PhysX properties with the expert mass model.
            self.record_body_masses = self.actor_body_masses
            self.record_body_com_local = self.actor_body_com_local
            self.record_total_masses = self.actor_total_masses

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

        # Three bodies form one cup: [suck, toe, empty].  Use the same leg
        # order as ExpertClimb and the asset body list, rather than relying on
        # a global sort of names/handles.  This keeps adhesion commands, cup
        # contact forces and the five-joint sensor block aligned by leg.
        self.magnetic_leg_order = ["lb", "lf", "lm", "rb", "rf", "rm"]
        magnetic_body_groups = [
            [f"l_{leg}_{part}" for part in ("suck", "toe", "empty")]
            for leg in self.magnetic_leg_order
        ]
        missing_magnetic_bodies = [
            name for group in magnetic_body_groups for name in group
            if name not in body_names
        ]
        if missing_magnetic_bodies:
            raise RuntimeError(
                f"magnetic bodies are missing from the asset: {missing_magnetic_bodies}"
            )
        self.magnetic_indices = torch.tensor(
            [[self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
              for name in group]
             for group in magnetic_body_groups],
            dtype=torch.long, device=self.device, requires_grad=False,
        )
        print("magnetic_indices\n",self.magnetic_indices)

        if self.recording_sensors_enabled:
            self.record_joint_parent_indices = torch.tensor(
                [body_names.index(spec["parent"])
                 for spec in self.record_joint_sensor_specs],
                dtype=torch.long, device=self.device
            )
            self.record_joint_child_indices = torch.tensor(
                [body_names.index(spec["child"])
                 for spec in self.record_joint_sensor_specs],
                dtype=torch.long, device=self.device
            )
            # Every magnetic group is explicitly [suck, toe, empty]; toe is
            # the cup reference frame.
            self.record_cup_reference_indices = self.magnetic_indices[:,1].clone()

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
        if self.debug_viz:
            print("robot contact force norm=",torch.norm(self.contact_forces[:,self.feet_indices,:],dim=-1))
        # print("rb_forces \n ",self.rb_forces[0,magnetic_indices,2].reshape(6,3))
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
    
