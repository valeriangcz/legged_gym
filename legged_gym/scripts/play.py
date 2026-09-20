# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # _,_,_ = env.reset_separate()
    _,_ = env.reset()
    obs = env.get_observations()
    # obs,obs_vgf,obs_terrain = env.get_observations_separated()
    # load policy
    train_cfg.runner.resume = True
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg) #其中包含env.reset()
    #使用服务器上挂载的logs-art目录
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root='logs-art') #其中包含env.reset()
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs-art', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    #设置变量，用于统计v g f估计的平均误差
    v_ave_err = 0
    g_ave_err = 0
    f_ave_err = 0

    for i in range(10*int(env.max_episode_length)):
        with torch.inference_mode():
            actions = policy(obs)
            # actions,obs_vgf_estimate,obs_terrain_lstm_estimates,obs_terrain_lstm = ppo_runner.alg.act_inference(obs,obs_terrain)
            #计算估计误差
            # v_err = torch.norm( (obs_vgf_estimate[:,:3]-obs_vgf[:,:3])/env.obs_scales.lin_vel, dim=1 ).mean()
            # g_err = torch.norm( (obs_vgf_estimate[:,3:6]-obs_vgf[:,3:6])/env.obs_scales.gravity, dim=1).mean()
            # f_err = torch.norm( (obs_vgf_estimate[:,6:]-obs_vgf[:,6:])/env.obs_scales.contact_force, dim=1).mean()
            
            # v_ave_err+=v_err.item()
            # g_ave_err+=g_err.item()
            # f_ave_err+=f_err.item()

            #打印估计值和输出值
            # if i%100==0:
            #     print(f"real v={obs_vgf[0,:3]/env.obs_scales.lin_vel}, est v={obs_vgf_estimate[0,:3]/env.obs_scales.lin_vel};")
            #     print(f"real g={obs_vgf_estimate[0,3:6]/env.obs_scales.gravity}, est g={obs_vgf[0,3:6]/env.obs_scales.gravity}")
            #     print(f"real f={obs_vgf_estimate[0,6:]/env.obs_scales.contact_force}, est f={obs_vgf[0,6:]/env.obs_scales.contact_force}")
                # print(f"real terrain lstm\n{obs_terrain_lstm}\n estimates terrain lstm\n{obs_terrain_lstm_estimates}\n")
            
            # obs,obs_vgf,obs_terrain,rew,dones,infos=env.step_separate(actions)
            # ppo_runner.alg.process_inference_env_step(dones)

            obs, _, rews, dones, infos = env.step(actions.detach())


            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            elif i==stop_state_log:
                logger.plot_states()
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()
                #打印平均误差
                # print(f"v err={v_ave_err/(i+1)}, g err={g_ave_err/(i+1)}, f err={f_ave_err/(i+1)}")


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
