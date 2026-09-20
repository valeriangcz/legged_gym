import torch
import torch.nn as nn
import torch.utils
from rsl_rl.storage import RolloutStorageMemory
from rsl_rl.utils import split_and_pad_obs_hist
def func():
    num_hist = 4
    num_t_steps = 9

    obs_hist = torch.rand(num_t_steps+num_hist-1)
    dones = torch.rand_like(obs_hist)<0.2

    # obs_hist = torch.tensor([0.1172, 0.0761, 0.1529, 0.4922, 0.2823, 0.4331, 0.0645, 0.2000, 0.2577, 0.1872, 0.7754, 0.5088])
    # dones = torch.tensor([True,  True, False, False,  True, False,  True,  True, False,  True, True,  True])
    print("obs_hist\n",obs_hist)
    print("dones\n",dones)



    obs_hist_batch = torch.zeros(num_t_steps,num_hist,dtype=torch.float32)
    mask = torch.zeros(num_t_steps,num_hist,dtype=torch.bool)

    dones_index = dones.nonzero().squeeze(1)
    for t in range(num_t_steps):
        i = t+num_hist-1
        perv_done_index = dones_index[dones_index<i]
        start = 0 if perv_done_index.numel()==0 else perv_done_index.max().item()+1

        length = i-start+1
        if length<num_hist:
            obs_hist_batch[t,:length] = obs_hist[start:i+1] 
        else:
            obs_hist_batch[t,:] = obs_hist[i+1-num_hist:i+1]
        mask[t,:length] = True



    print("mask\n",mask)
    print("pacthed_obs_hist\n",obs_hist_batch)

# def splid_and_pad_obs_hist(obs_hist,dones,num_hist, num_t_steps):
#     """
#     obs_hist: num_envs *(num_t_steps+num_hist-1) * D
#     dones: num_envs * num_t_steps * 1

#     返回
#     obs_hist_batch: num_envs * num_t_steps * num_hist * D
#     masks: num_envs * num_t_steps * num_hist * 1
#     """
#     shapes = obs_hist.shape
#     N=shapes[0]     # 环境数量
#     D=shapes[2]    # 观测维度，暂时作为1考虑
#     H=num_hist      # 历史观测长度
#     T=num_t_steps   # 总步数

#     # obs_hist_batch = torch.zeros((N,T,H,D), dtype=obs_hist.dtype, device=obs_hist.device)
#     # masks = torch.zeros((N,T,H,1), dtype=torch.bool, device=obs_hist.device)

#     dones = dones.squeeze(-1)

#     #不管done的状态，先试用unfold方法将obs_hist切分成 N*T*H*D的张量
#     obs_hist_unfold = obs_hist.unfold(dimension=1, size=H, step=1).permute(0,1,-1,2,3)
#     #采用同样的方法，对done进行切分为 N*T*H
#     dones_unfold = dones.unfold(dimension=1, size=H, step=1)
#     dones_cumsum = dones_unfold.cumsum(dim=-1) # N*T






# a=torch.rand(2,11)<0.6
# a_unfold = a.unfold(dimension=1, size=7, step=1)
# a_unfold[:,-1]=False #最后一行的True或者False对判断有效长度不影响，设置为False避免对max_index造成干扰
# # print(a_unfold)
# #先unfold，再进行flatten
# a_cumsum = torch.cumsum(a_unfold,dim=-1) #2*5*7
# max_value,max_index =torch.max(a_cumsum,dim=-1) #2*5

# print(max_index)


# col_index = torch.arange(a_cumsum.shape[-1]).unsqueeze(0) #1*7
# # print(col_index)
# # print(max_index)
# res =col_index>max_index.unsqueeze(-1)
# print(res.shape)
# res[max_value==0] = True #全零行，所有值都设置为True

# print(a_unfold.to(torch.int32))
# print(res.to(torch.int32))


"""测试历史观测数据整理"""

# obs_hist = torch.rand(1,11,2)

# obs_hist = torch.arange(32).reshape(2,8,2)

# # print(obs_hist)
# dones = torch.rand(2,8,1)<0.6

# obs_hist_batch,mask=split_and_pad_obs_hist(obs_hist,dones,7)
# # print(mask.shape)
# # print(mask.to(torch.int32))
# obs_hist_batch = obs_hist_batch.flatten(0,1)
# # print(obs_hist_batch.shape)
# # print(obs_hist_batch)
# mask= mask.flatten(0,1) #B*T*H

# print(mask.shape)
# print(mask.to(torch.int32))
# mask = mask.flip(dims=[1])
# print(mask.to(torch.int32))

# storage = RolloutStorageMemory(2,24,10,[6],[5],[12])
# transitions = RolloutStorageMemory.Transition()

# for i in range(50):
#     if i%23==0:
#         storage.clear()
#     transitions.obs = torch.rand(2,6)
#     transitions.obs_terrain = torch.rand(2,5)
#     transitions.actions = torch.rand(2,12)
#     transitions.rewards = torch.rand(2,1)
#     transitions.values = torch.rand(2,1)
#     transitions.actions_log_prob = torch.rand(2,1)
#     transitions.action_mean = torch.rand(2,12)
#     transitions.action_sigma = torch.rand(2,12)
#     transitions.expert_actions = torch.rand(2,12)
#     transitions.dones = torch.rand(2,1)<0.4
#     # if i==20:
#     #     transitions.dones.fill_(True)
#     storage.add_obs_transitions(transitions.obs)


#     if i==49:
#         obs_hist, mask = storage.get_current_obs_hist()
#         print(storage.dones.transpose(0,1).squeeze(-1))
#         print(obs_hist.shape)
#         print(mask.shape)
#         print(mask)

#     storage.add_transitions(transitions)



# print(storage.dones.transpose(0,1).squeeze(-1)[0])

# a=torch.arange(10)
# a_unfold = a.unfold(dimension=0,size=5,step=1)
# a_unfold[0]=-10
# print("a\n",a)

lstm = torch.nn.LSTM(input_size=1,hidden_size=4,num_layers=2,batch_first=True)

input = torch.rand(2,6,1)
mask = torch.ones(2,6,dtype=torch.bool)

mask[0,3:]=False
mask[1,4:]=False

input_zero_pad = input*mask.unsqueeze(-1)
lengths = torch.sum(mask,dim=-1)
print(input)
print(input_zero_pad)

output, (h_n,c_n)=lstm(input_zero_pad)
print(h_n)


padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(input_zero_pad,lengths,batch_first=True,enforce_sorted=False)

output, (h_n,c_n)=lstm(padded_sequence)
print(h_n)


padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(input,lengths,batch_first=True,enforce_sorted=False)

output, (h_n,c_n)=lstm(padded_sequence)
print(h_n)