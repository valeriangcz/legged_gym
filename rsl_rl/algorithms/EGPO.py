from rsl_rl.algorithms import PPO
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.distributions import Normal
from rsl_rl.storage import RolloutStorageExpert

#需要修改act方法，接受参考的expert_action
class EGPO(PPO):
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
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 expert_interface_iter=200,
                 expert_exit_iter = 100,
                 BC_loss_coef = 2.0
                 ):
        super().__init__(actor_critic,
                         num_learning_epochs,
                         num_mini_batches,
                         clip_param,
                         gamma,
                         lam,
                         value_loss_coef,
                         entropy_coef,
                         learning_rate,
                         max_grad_norm,
                         use_clipped_value_loss,
                         schedule,
                         desired_kl,
                         device)
        # self.tic1=time.time()
        # self.tic2=time.time()
        self.expert_interface_iter=expert_interface_iter #专家干预的迭代次数
        self.expert_exit_iter = expert_exit_iter #专家退出干预的迭代次数
        self.BC_loss_coef = BC_loss_coef#模仿学习的权重系数
        self.transition = RolloutStorageExpert.Transition()
        print("---------->expert interface iter={}<--------------".format(expert_interface_iter))
        print("---------->expert exit iter={}<--------------".format(expert_exit_iter))

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageExpert(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

        
    def act(self,obs,critic_obs,expert_action,it):
        
        agent_actions=self.actor_critic.act(obs).detach()
        self.transition.exeprt_actions = expert_action.detach()

        # self.transition.actions = expert_action.detach()
        # self.transition.actions = agent_actions.detach()

        alpha_t=1.0-min(float(it)/self.expert_interface_iter,1.0)
        # alpha_t = math.exp(-(it/200.0)/0.2)#测试专家动作一下非线性衰减
        # if it<=200:
        if it<=self.expert_exit_iter:
        # if it<100:
            # if torch.rand(1).item()<alpha_t:
            #     self.transition.actions = expert_action.detach()
            # else:
            #     self.transition.actions = agent_actions.detach()
            self.transition.actions = alpha_t*expert_action.detach() + (1-alpha_t)*agent_actions.detach()
            # self.transition.actions = expert_action.detach()*0.85 + agent_actions.detach()*0.15 #使用固定权重专家
        else:
            self.transition.actions = agent_actions.detach()
        # #计算专家的策略下mix_action 的 log_prob
        # if it<300:
        #     alpha_t=(300-it)/300.0
        # elif it<600:
        #     alpha_t=(it-300)/300.0
        # #构造专家的分布
        # mixed_norm=expert_action.detach() * (1-alpha_t) + self.actor_critic.action_mean * alpha_t
        # mixed_std = torch.ones_like(expert_action)*0.01 * (1-alpha_t) + self.actor_critic.action_std * alpha_t
        # mixed_dist = Normal(mixed_norm, mixed_std)
        # self.transition.actions = mixed_dist.sample()


        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.actions_log_prob = mixed_dist.log_prob(self.transition.actions).sum(dim=-1).detach()
        # self.transition.action_mean = mixed_norm
        # self.transition.action_sigma = mixed_std

        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # if time.time()-self.tic1>3.0:
            # print("expert action=",expert_action[0])
            # print("agent action=",agent_actions[0])
            # print("(agent_action-expert_action).mean()=",torch.norm(agent_actions-expert_action,dim=1).mean())
            # print("iteration num=",it)
            # print("action log prob.shape=",self.transition.actions_log_prob.shape) #nums_env*18
            # print("==========>agent action log prob=",torch.mean(self.actor_critic.get_actions_log_prob(agent_actions),dim=0))
            # min_log_prob = torch.min(self.actor_critic.get_actions_log_prob(expert_action))
            # max_log_prob = torch.max(self.actor_critic.get_actions_log_prob(expert_action))

            # print("============>expert action log prob mean()=",torch.mean(self.actor_critic.get_actions_log_prob(expert_action),dim=0))
            # print("=============>expert action log prob min()=",min_log_prob,"\n=============>expert action log prob max()=",max_log_prob)
            # print("============>mixed action log prob=",torch.mean(self.actor_critic.get_actions_log_prob(self.transition.actions),dim=0))
            # self.tic1 = time.time()        
        return self.transition.actions
    def update(self,it,tot_iter):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_BC_loss = 0.0
        mean_bc_mse_loss = 0.0
        mean_ratio = 0.0
        mean_advantage = 0.0
        mean_cos_similarity = 0.0



        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch,expert_actions_batch,hid_states_batch, masks_batch in generator:
            # print("actions batch.shape=",actions_batch.shape)
            # agent_actions_batch=self.actor_critic.act_inference(obs_batch)
            self.actor_critic.act(obs_batch,masks=masks_batch,hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy


            #BC
            bc_loss_fn = torch.nn.MSELoss()
            bc_mse_loss = bc_loss_fn(expert_actions_batch,mu_batch.detach()).item() #只是计算均值的MSE损失，不做反向传播
            # BC_loss_fn = torch.nn.MSELoss()
            # BC_loss = BC_loss_fn(expert_actions_batch,mu_batch)
            # BC_loss = -self.actor_critic.get_actions_log_prob(expert_actions_batch).mean(dim=-1)
            
            if it<self.expert_exit_iter:
                BC_loss = -self.actor_critic.get_actions_log_prob(expert_actions_batch).mean(dim=-1)
            else:
                BC_loss = torch.tensor([0],device=self.device,requires_grad=False)
            
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

            #clip ratio
            ratio = torch.clamp(ratio, 0.1,10.0)

            # #监测ratio异常
            # abnorm_mask = (ratio>100.0) | (ratio<0.001)
            # if abnorm_mask.any():
            #     print("==============>Detecting abnormal ratio! corresponding variables<============")
            #     print("abnorm number=",abnorm_mask.sum().item())
            #     print("abnorm ratio\n",ratio[abnorm_mask])
            #     print("old_actions_log_prob_batch\n",torch.squeeze(old_actions_log_prob_batch[abnorm_mask]))
            #     print("current_log_prob_batch\n",actions_log_prob_batch[abnorm_mask])


            #     print("expert action batch")
            #     print(actions_batch[abnorm_mask])

            #     print("current agent policy network")
            #     print("mean\n",mu_batch[abnorm_mask])
            #     print("std\n",sigma_batch[abnorm_mask])
            #     print("log_prob\n",self.actor_critic.distribution.log_prob(actions_batch)[abnorm_mask])

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() * alpha

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            #给探索因子加上一个时间衰减项,专家退出前，适当提高这一项权重，避免方差下降太快导致失去探索性
            beta=math.exp( -(it/tot_iter) / 0.5)
            if it <= self.expert_exit_iter:
            # if it < 100: #200 是经验峰值
                loss=self.value_loss_coef * value_loss + self.BC_loss_coef*BC_loss + self.entropy_coef * entropy_batch.mean()
            else:
                loss=surrogate_loss + self.value_loss_coef * value_loss - beta * self.entropy_coef * entropy_batch.mean()
            # alpha =1.0-min(float(it)/self.expert_interface_iter,1.0)

            # 原来计算梯度的方式
            # loss=(1.0-alpha)*surrogate_loss + \
            #     self.value_loss_coef * value_loss \
            #         - beta*self.entropy_coef * entropy_batch.mean()*(1-alpha)\
            #             + self.BC_loss_coef*BC_loss*alpha

            policy_param=[p for p in self.actor_critic.actor.parameters() if p.requires_grad]
            # #为了统计方法，分开计算
            grad_sur = torch.autograd.grad(
                surrogate_loss,policy_param,retain_graph=True,create_graph=False,allow_unused=True)
            grad_bc = torch.autograd.grad(
                BC_loss*self.BC_loss_coef,
                policy_param,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            sur_vec = self._flatten_grad(grad_sur)
            bc_vec= self._flatten_grad(grad_bc)
            sur_norm = torch.norm(sur_vec).item()
            bc_norm = torch.norm(bc_vec).item()
            cos_similarity=torch.nn.functional.cosine_similarity(sur_vec,bc_vec,dim=0).item()

            # all_obs=torch.cat([all_obs,obs_batch],dim=0) if all_obs is not None else obs_batch
            # all_actions=torch.cat([all_actions, actions_batch],dim=0) if all_actions is not None else actions_batch
            # all_expert_actions=torch.cat([all_expert_actions,expert_actions_batch],dim=0) if all_expert_actions is not None else expert_actions_batch


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
            #因为一部分动作是从专家采样来的，因此插值后偶尔出现梯度nan的情况，只需要不在这个时候step就可以
            if step: 
                self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.abs().item()
            mean_BC_loss += BC_loss.item()
            mean_ratio += ratio.mean().item()
            mean_advantage += advantages_batch.abs().mean().item()

            mean_bc_mse_loss += bc_mse_loss

            mean_cos_similarity += cos_similarity
            # for name, param in self.actor_critic.named_parameters():
            #     if 'actor' in name:
            #         nan_param_mask = torch.isnan(param)
            #         if nan_param_mask.any():
            #             print(f"find nan in actor {name} ")
            #             print(f"actor {name} shape {param.shape}, has nan param number {nan_param_mask.sum()}")
            #         mean_actor_mean += param.mean().item()
            #         mean_actor_min = param.min().item() if mean_actor_min>param.min().item() else mean_actor_min
            #         mean_actor_max = param.max().item() if mean_actor_max<param.max().item() else mean_actor_max
                    # mean_actor_min
            # print("it=",it)

        # if it>98 and it<103:
        # print(f"obs mean\n{all_obs.mean(dim=0)}\n,obs std\n{all_obs.std(dim=0)}")
        # print(f"actions_mean\n{all_actions.mean(dim=0)}\n,actions_std\n{all_actions.std(dim=0)}")
        # print(f"expert actions mean\n{all_expert_actions.mean(dim=0)}\n, std\n{all_expert_actions.std(dim=0)}")                        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_BC_loss /= num_updates
        mean_ratio /= num_updates
        mean_advantage /= num_updates
        mean_bc_mse_loss /= num_updates
        mean_cos_similarity /= num_updates


        self.storage.clear()

        # return mean_value_loss, mean_surrogate_loss, mean_BC_loss, mean_ratio, mean_advantage, mean_actor_mean, mean_actor_min, mean_actor_max
        return mean_value_loss, mean_surrogate_loss, mean_BC_loss, mean_ratio, mean_advantage, sur_norm, bc_norm, mean_cos_similarity, mean_bc_mse_loss
        # return mean_value_loss, mean_surrogate_loss, mean_BC_loss, mean_ratio, mean_advantage, 0,0,0
    
    def _flatten_grad(self,grads):
        return torch.cat([g.reshape(-1) for g in grads if g is not None])