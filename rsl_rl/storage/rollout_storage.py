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

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories, split_and_pad_obs_hist

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # print("\033[91m =============>mini_bacth_generator num_mini_batches={}, num_epochs={} \033[0m ".format(num_mini_batches, num_epochs))
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx] 
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
        

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])

                # traj_end_flags = dones.clone()
                # traj_end_flags[-1] = True
                # trajectories_batch_size = torch.sum(traj_end_flags[:, start:stop])


                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
                first_traj = last_traj


class RolloutStorageExpert(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            #为expert action添加
            self.exeprt_actions = None
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)
        self.expert_actions = torch.zeros(num_transitions_per_env,num_envs,*actions_shape,device=self.device)
    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.expert_actions[self.step].copy_(transition.exeprt_actions)
        self.step += 1
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # print("\033[91m =============>mini_bacth_generator num_mini_batches={}, num_epochs={} \033[0m ".format(num_mini_batches, num_epochs))
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        #
        expert_actions = self.expert_actions.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx] 
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                #
                expert_actions_batch = expert_actions[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, expert_actions_batch ,(None, None), None

#每一个环境都维护了一个历史观测， 并且每个环境done的长度和历史长度相同
#obs obs_vgf 拼接归属于 obs  需要管理历史，生成时序序列，用于将来的obs_terrain_lant的预测
#obs obs_vgf obs_terrain_lant 拼接归属于 obs_priv
# TODO 迁移到机器人的计算机上，只需要把关于obs_hist部分进行维护就可以
class RolloutStorageMemory():
    class Transition:
        def __init__(self):
            self.obs = None
            self.obs_terrain = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

            #为expert action添加
            self.expert_actions = None
        
        def clear(self):
            self.__init__()

    #多加一项
    def __init__(self, num_envs, num_transitions_per_env, num_hist,
                 obs_shape, obs_terrain_shape, actions_shape, device='cpu'):
        self.device = device

        self.obs_shape = obs_shape
        self.obs_terrain_shape = obs_terrain_shape
        self.actions_shape = actions_shape
        self.num_hist = num_hist

        # Core
        self.obs = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        #额外增加一项维护历史观测， 表示一批数据保存 num_transitions_per_env 个数据， 
        # 为了让每个数据都有足够的历史观测，所以obs_his的长度要在这个基础上增加 (num_hist-1)个
        self.obs_hist = torch.zeros(num_transitions_per_env+num_hist-1, num_envs, *obs_shape, device=device)

        #地形不需要维护历史观测
        self.obs_terrain = torch.zeros(num_transitions_per_env, num_envs, *obs_terrain_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 为了分割历史观测的有效部分，所以done的长度和obs_hist长度相同
        self.dones = torch.zeros(num_transitions_per_env+num_hist-1, num_envs, 1, dtype=torch.bool,device=device) 

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # For EGPO
        self.expert_actions = torch.zeros(num_transitions_per_env,num_envs,*actions_shape,device=self.device)


        self.step = 0

        #历史是从最后一位往前填充，所以有效的索引起点最开始在最后一位之后
        self.valid_hist_index = num_transitions_per_env+num_hist-1 #

    def update_obs_hist(self,observations):
        """
        维护观测历史
        把最新的状态放入观测历史的最后一位
        把done往前移动一位，最新的done无论是False 或者True对结果并不影响，这个在step后再更新
        但是如果不移动的话，done与obs_hist就对不上号了
        """

        self.obs_hist[:-1, ...]=self.obs_hist[1:, ...].clone()
        self.obs_hist[-1].copy_(observations)
        self.dones[:-1,...] = self.dones[1:,...].clone()

        #当历史被填充满时，有效观测的起始索引一直为0
        if self.valid_hist_index >0:
            self.valid_hist_index -= 1


    def update_dones_hist(self,dones):
        #在训练阶段无需调用，在测试阶段调用，确保观测历史和结束标志正常被维护
        # self.dones[:-1,...]=self.dones[1:,...].clone()
        self.dones[-1].copy_(dones.view(-1,1))

    def add_transitions(self, transition:Transition):
        #因为修改了done的维护逻辑，因此需要重写这里的函数
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step].copy_(transition.obs)
        self.obs_terrain[self.step].copy_(transition.obs_terrain)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        # self.dones[self.step].copy_(transition.dones.view(-1, 1)) # 原来的done维护逻辑, 这里注释
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        #
        self.expert_actions[self.step].copy_(transition.expert_actions)
        self.step += 1

        # self.dones[:-1,...] = self.dones[1:,...].clone()
        self.dones[-1].copy_(transition.dones.view(-1, 1))
        #只是为了测试dones在不同位置
        # self.dones[-1] = torch.rand(self.num_envs,1)<0.5
        # print("in add transition dones\n",self.dones[:,0,:].squeeze(-1))

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def clear(self):
        self.step = 0
    
    def get_current_obs_hist(self):
        #返回 obs_hist 维度为 N * H * D 返回mask维度为 N*H

        dones = self.dones.transpose(0,1).squeeze(-1).clone() #N * T+H-1

        # print("get current obs hist dones.shape=",dones.shape)
        #历史没有被填充满时，有效索引前一位要设置为True标记，
        # valid_hist_index如果为0， 也就是把-1设置为True后，下一步还是会设置为 False
        dones[:,self.valid_hist_index-1]=True
        #设置最后一位为False，不对max的逻辑造成影响
        dones[:,-1] = False


        dones_cumsum = torch.cumsum(dones, dim=-1)
        max_value, last_done_index = torch.max(dones_cumsum,dim=-1)
        
        col_index = torch.arange(self.dones.shape[0], device=self.device).unsqueeze(0)
        
        mask = last_done_index.unsqueeze(-1)<col_index
        mask[max_value==0]=True

        # print("mask.shape=",mask.shape)

        mask = mask[:,-self.num_hist:]
        obs_hist = self.obs_hist.transpose(0,1)[:,-self.num_hist:]
        # obs_hist = obs_hist.flatten(0,1)
        # mask = mask.flatten(0,1)
        return obs_hist, mask


    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        #增加历史观测数据处理，按顺序生成mini_batch
        #mini_batch的生成是拼接env0:[t1,t2,t3,...,tt] env1:[] env2:[]... t1表示了t1时刻所有的相关数据
        #因此，每一个mini_batch需要包含num_envs//num_mini_batches个环境的数据
        num_envs_per_mini_batch = self.num_envs // num_mini_batches
        obs = self.obs.transpose(0, 1).flatten(0, 1)
        obs_terrain = self.obs_terrain.transpose(0, 1).flatten(0, 1) 
        # critic_obs = critic_obs.transpose(0, 1).flatten(0, 1)
        actions = self.actions.transpose(0, 1).flatten(0, 1)
        values = self.values.transpose(0, 1).flatten(0, 1)
        returns = self.returns.transpose(0, 1).flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.transpose(0, 1).flatten(0, 1)
        advantages = self.advantages.transpose(0, 1).flatten(0, 1)
        old_mu = self.mu.transpose(0, 1).flatten(0, 1)
        old_sigma = self.sigma.transpose(0, 1).flatten(0, 1)
        expert_actions = self.expert_actions.transpose(0, 1).flatten(0, 1)

        dones = self.dones.transpose(0, 1) # num_envs * num_transitions_per_env+env_hist-1 * 1 
        # print("mini start dones\n",dones[0].squeeze(-1))
        #历史还未被填充满，有效索引前一位设置为True进行区分，valid_hist_index=0时，相当于设置最后一位为True，不会造成影响
        dones[:,self.valid_hist_index-1]=True
        obs_hist = self.obs_hist.transpose(0, 1) #num_envs * (num_transtions_per_env+num_hist-1) * D

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i*num_envs_per_mini_batch*self.num_transitions_per_env
                end = (i+1)*num_envs_per_mini_batch*self.num_transitions_per_env

                env_start = i*num_envs_per_mini_batch
                env_end = (i+1)*num_envs_per_mini_batch
                #根据环境进行分割 num_envs_per_mini_batch*T  * D
                obs_batch = obs[start:end]
                actions_batch = actions[start:end]
                target_values_batch = values[start:end]
                returns_batch = returns[start:end]
                old_actions_log_prob_batch = old_actions_log_prob[start:end]
                advantages_batch = advantages[start:end]
                old_mu_batch = old_mu[start:end]
                old_sigma_batch = old_sigma[start:end]
                expert_actions_batch = expert_actions[start:end]
                obs_terrain_batch = obs_terrain[start:end] 
                #obs_hist 维度为 
                obs_hist_batch = obs_hist[env_start:env_end] # num_envs_per_mini_batch * (num_transtions_per_env+num_hist-1) * D
                dones_batch = dones[env_start:env_end] #num_envs_per_mini_batch * (num_transtions_per_env+num_hist-1) * 1
                
                #num_envs_per_mini_batch * num_transitions_per_env * num_hist * D,  num_envs_per_mini_batch * num_transitions_per_env * num_hist
                # print("before split_and_pad_obs_hist")
                # print(f"obs_hist_batch.shape={obs_hist_batch.shape}, dones_batch.shape={dones_batch.shape}")
                obs_hist_padded_batch, masks_batch = split_and_pad_obs_hist(obs_hist_batch,
                                                               dones_batch,
                                                               self.num_hist)
                dones_batch = dones_batch[:,self.num_hist-1:]##这里的done只摘取当前batch部分，不包括历史部分
                #对第一个环境检测其dones和mask
                # if epoch ==0 and i ==0:
                #     print("dones\n",dones[0].squeeze(-1))
                #     print("mask\n",masks_batch[0])
                #     print("\n")


                #进行平展操作
                obs_hist_padded_batch = obs_hist_padded_batch.flatten(0, 1)
                masks_batch = masks_batch.flatten(0, 1)
                dones_batch = dones_batch.flatten(0, 1)
                
                yield obs_batch,obs_terrain_batch ,actions_batch, target_values_batch, advantages_batch, returns_batch,\
                      old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, expert_actions_batch,\
                      obs_hist_padded_batch, masks_batch, dones_batch
