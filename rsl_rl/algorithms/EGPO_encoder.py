#Expert Guided Policy Optimization算法，还有编码器的训练和加载部分

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticEncoder
from rsl_rl.storage import RolloutStorageMemory

import math

class EGPOEncoder:
    actor_critic: ActorCriticEncoder
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule='fixed',
                 desired_kl=0.01,
                 device='cpu',
                 expert_interface_iter=200
                 ):
        
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        #EGPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorageMemory.Transition()

        #EGPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.expert_interface_iter=expert_interface_iter

    def init_storage(self, num_envs, num_transitions_per_env ,actor_obs_shape, critic_obs_shape, action_shape):
            #先写在这里面，后面放到cfg参数中
            # num_hist = 10
            num_hist = 20
            self.storage = RolloutStorageMemory(num_envs, num_transitions_per_env,num_hist, 
                                                actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def encode_obs(self,obs,obs_vgf,obs_terrain):
        self.transition.obs_terrain = obs_terrain.detach()


        #TODO 先采用真值拼接，等estimator完成训练后再修改
        # obs_vgf_estimates = self.actor_critic.actor_obs_priv_estimator(obs)
        obs_splice = torch.cat([obs,obs_vgf],dim=-1)
        self.storage.update_obs_hist(obs_splice)
        # obs_hist, mask = self.storage.get_current_obs_hist()
        # obs_terrain_lstm_latent = self.actor_critic.LSTM_encode(obs_hist, mask)
        #对观测部分进行编码或者估算
        obs_terrain_latent = self.actor_critic.encode_terrain(obs_terrain)
        #TODO 这里先使真值，等lstm encoder训练完后再替代

        #测试，观察历史观测的划分正确与否
        #针对
        # print("Dones \n",self.storage.dones[:,0].squeeze(-1).to(torch.int32).tolist())
        # print("mask\n",mask[0].to(torch.int32).tolist())
        # print("\n")

        return obs_splice, obs_terrain_latent

    def act(self, obs_splice,obs_terrain_latent, expert_actions, it):
        
        self.transition.obs = obs_splice.detach()
        self.transition.expert_actions = expert_actions.detach()
        agent_actions = self.actor_critic.act(obs_splice,obs_terrain_latent).detach() #

        # self.transition.obs_terrain_latent = obs_terrain_latent.detach()
        #TODO 需要使用专家混合插值计算动作

        alpha_t=1.0-min(float(it)/self.expert_interface_iter,1.0)
        self.transition.actions = alpha_t*expert_actions.detach() + (1-alpha_t)*agent_actions.detach()



        self.transition.values = self.actor_critic.evaluate(obs_splice,obs_terrain_latent).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        return self.transition.actions
    
    def act_inference(self,obs,obs_terrain):
        # obs_splice, obs_terrain_latent = self.encode_obs(obs,obs_vgf,obs_terrain)
        obs_vgf_estimates = self.actor_critic.actor_obs_priv_estimator(obs)
        obs_splice = torch.cat([obs,obs_vgf_estimates],dim=-1)
        self.storage.update_obs_hist(obs_splice)
        obs_hist, mask = self.storage.get_current_obs_hist()
        # obs_terrain_lstm_latent_estimates = self.actor_critic.LSTM_encode(obs_hist, mask)
        obs_terrain_lstm_latent_estimates = self.actor_critic.TCN_encode(obs_hist, mask)

        obs_terrain_lstm_latent = self.actor_critic.encode_terrain(obs_terrain)

        return self.actor_critic.act_inference(obs_splice,obs_terrain_lstm_latent),\
               obs_vgf_estimates,\
               obs_terrain_lstm_latent_estimates,\
               obs_terrain_lstm_latent

    def process_inference_env_step(self,dones):
        self.storage.update_dones_hist(dones)
        self.transition.clear()


    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # TODO: test the following troch.squeeze()
        # if 'time_outs' in infos:
        #     print("\033[91m ========================= \033[0m")
        #     print("infos['time_outs'].shape=",infos['time_outs'].shape)
        #     print("self.transition.values.shape=",self.transition.values.shape)
        #     print("direct multiply=",self.transition.values*infos['time_outs'])
        #     print("use unsqueeze(1)",self.transition.values*infos['time_outs'].unsqueeze(1))
        #     pass

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset()
    
    def compute_returns(self, last_obs_splice, last_obs_terrain_latent):
        last_values= self.actor_critic.evaluate(last_obs_splice, last_obs_terrain_latent).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self,it,tot_iter):
        #这里更新要计算lstm encoder的误差
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_lstm_encoder_loss = 0
        mean_bc_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch,obs_terrain_batch ,actions_batch, target_values_batch, advantages_batch, returns_batch,\
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, expert_actions_batch,\
            obs_hist_padded_batch, masks_batch, dones_batch in generator:
                
                obs_terrain_latent_batch = self.actor_critic.encode_terrain(obs_terrain_batch)

                self.actor_critic.act(obs_batch,obs_terrain_latent_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(obs_batch, obs_terrain_latent_batch.detach())
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # estimator loss
                
                estimator_loss = nn.MSELoss()(self.actor_critic.actor_obs_priv_estimator(obs_batch[...,:67]), obs_batch[...,67:79])

                # lstm encoder loss
                # print("obs_hist_padded_batch.shape=",obs_hist_padded_batch.shape)
                # print("masks_batch.shape=",masks_batch.shape)
                # print("mask_batch[0]\n",masks_batch[0])
                # exit(0)

                # lstm_encoder_loss = nn.MSELoss()(self.actor_critic.LSTM_encode(obs_hist_padded_batch,masks_batch),obs_terrain_latent_batch.detach())
                lstm_encoder_loss = nn.MSELoss()(self.actor_critic.TCN_encode(obs_hist_padded_batch,masks_batch),obs_terrain_latent_batch.detach())


                # Behaviour cloning loss
                alpha =1.0-min(float(it)/self.expert_interface_iter,1.0)
                BC_loss = -self.actor_critic.get_actions_log_prob(expert_actions_batch).mean(dim=-1)*alpha*2.0


                #Total loss
                #给探索因子加上一个时间衰减项
                alpha = math.exp(-(it/tot_iter)/0.2)
                loss = surrogate_loss + self.value_loss_coef * value_loss - alpha*self.entropy_coef * entropy_batch.mean() + estimator_loss + lstm_encoder_loss + BC_loss


                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                step=True
                for name, param in self.actor_critic.named_parameters():
                    if 'actor' in name:
                        if param.grad is not None:  # 确认这个参数确实有梯度
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"NaN or Inf detected in gradient of {name}")    
                                step=False  
                if step:              
                    self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimator_loss += estimator_loss.item()
                mean_lstm_encoder_loss += lstm_encoder_loss.item()
                mean_bc_loss += BC_loss.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_lstm_encoder_loss /= num_updates
        mean_bc_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_bc_loss,mean_estimator_loss, mean_lstm_encoder_loss

    # def act