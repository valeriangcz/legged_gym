import os
import tensorboard
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from legged_gym import LEGGED_GYM_ROOT_DIR

# 指定 .tfevents 文件路径

"""绘制奖励函数和episode时长图"""
# events_file1 = f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Sep08_22-40-03_/events.out.tfevents.1757342404.uss.35791.0"
# events_file2 = f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Sep09_11-24-36_/events.out.tfevents.1757388278.uss.134229.0"

# # 加载 TensorBoard 事件文件
# ea1 = event_accumulator.EventAccumulator(events_file1)
# ea2 = event_accumulator.EventAccumulator(events_file2)
# ea1.Reload()
# ea2.Reload()

# reward_tag = 'Train/mean_reward'
# episode_length_tag = 'Train/mean_episode_length'

# # 获取 'Train/mean_reward' 数据
# scalars_reward = ea1.Scalars(reward_tag)
# steps_reward1, values_reward1 = zip(*[(x.step, x.value) for x in scalars_reward][:200])
# scalars_reward = ea2.Scalars(reward_tag)
# steps_reward2, values_reward2 = zip(*[(x.step, x.value) for x in scalars_reward][:200])
# # 获取 'Train/mean_episode_length' 数据
# scalars_length = ea1.Scalars(episode_length_tag)
# steps_length1, values_length1 = zip(*[(x.step, x.value*0.02) for x in scalars_length][:200])
# scalars_length = ea2.Scalars(episode_length_tag)
# steps_length2, values_length2 = zip(*[(x.step, x.value*0.02) for x in scalars_length][:200])
# # 创建图形
# plt.rcParams.update({'font.size': 14})
# fig, ax1 = plt.subplots(figsize=(8,5))
# # fig, ax1 = plt.subplots()

# # 绘制 'Train/mean_reward' 曲线
# ax1.plot(steps_reward1, values_reward1, color="#E62A2A", label='EGPO:r',linewidth=2)
# ax1.set_xlabel('Iterations',fontsize=18)
# ax1.set_ylabel('Mean Reward',fontsize=18)
# ax1.tick_params(axis='y')
# ax1.plot(steps_reward2, values_reward2, color="#580C0C", label='PPO:r',linewidth=2)


# # 创建一个共享的第二个 y 轴
# ax2 = ax1.twinx()
# # 绘制 'Train/mean_episode_length' 曲线
# ax2.plot(steps_length1, values_length1, color="#2BB449", label='EGPO:$T_{episode}$',linewidth=2)
# ax2.plot(steps_length2, values_length2, color="#0B611C", label='PPO:$T_{episode}$',linewidth=2)
# ax2.set_ylabel('Episode Length(s)',fontsize=18)
# ax2.tick_params(axis='y')

# # 添加图例
# ax1.legend(bbox_to_anchor=(0.6, 0.1),loc='lower left')
# ax2.legend(bbox_to_anchor=(0.6, 0.3),loc='lower left')

# # 设置图表标题

# # 显示图表
# plt.show()
"""绘制不同alpha衰减情况下BC_loss的曲线图"""

file_root=f"{LEGGED_GYM_ROOT_DIR}/logs-art/hex_ground"
ea_list=[]
# tag="Loss/bc_loss"
# tag="Loss/mean_bc_mse_loss"
plt.rcParams.update({'font.size':32})
fig,ax1 = plt.subplots(figsize=(16,10))
fig,ax2 = plt.subplots(figsize=(16,10))
fig,ax3 = plt.subplots(figsize=(16,10))
fig,ax4 = plt.subplots(figsize=(16,10))
fig,ax5 = plt.subplots(figsize=(16,10))
ax1.set_xlabel("Iterations")
ax2.set_xlabel("Iterations")
ax3.set_xlabel("Iterations")
ax4.set_xlabel("Iterations")
ax5.set_xlabel("Iterations")
# ax1.set_ylabel(r"$\mathrm{MSE}(a_E-a_{\theta})$")
# ax2.set_ylabel(r"$\mathrm{E} [\,-\log \pi_{\theta}(a_E \mid s)\,]$")
# ax3.set_ylabel(r"$\sigma$")
ax1.text(x=-0.05, y=1.05, s=r"$f_{\mathrm{MSE}}(a_E-a_{\theta})$", 
         transform=ax1.transAxes, ha='left', va='bottom', rotation=0)
ax2.text(x=-0.05, y=1.05, s=r"$\mathrm{E}[-\log \pi_{\theta}(a_E \mid s)]$", 
         transform=ax2.transAxes, ha='left', va='bottom', rotation=0)
ax3.text(x=-0.05, y=1.05, s=r"$\sigma$", 
         transform=ax3.transAxes, ha='left', va='bottom', rotation=0)
ax4.text(x=-0.05, y=1.05, s=r"$\nabla L_{IL} \cdot \nabla L_{PPO}$", 
         transform=ax4.transAxes, ha='left', va='bottom', rotation=0)
ax5.text(x=-0.05, y=1.05, s=r"value function loss", 
         transform=ax4.transAxes, ha='left', va='bottom', rotation=0)         

i=0
files = os.listdir(file_root)

for dir in sorted(os.listdir(file_root)):
    if "Jan16" in dir:
        file_dir = os.path.join(file_root,dir)
        for file in os.listdir(file_dir):
            if file.endswith('0'):
                ea=event_accumulator.EventAccumulator(os.path.join(file_dir,file))
                ea.Reload()
                ea_list.append(ea)
                scalar_bc_mse_loss = ea.Scalars("Loss/mean_bc_mse_loss")
                scalar_bc_loss = ea.Scalars("Loss/bc_loss")
                scalar_std = ea.Scalars("Policy/mean_noise_std")
                scalar_cos_sim = ea.Scalars("Loss/cos_similarity")
                scalar_value_loss = ea.Scalars("Loss/value_function")
                labels = str(len(scalar_bc_loss))
                steps, values = zip(*[(x.step,x.value) for x in scalar_bc_mse_loss])
                ax1.plot(steps,values,linewidth=6,label="$N_E=$"+labels)
                steps, values = zip(*[(x.step, x.value) for x in scalar_bc_loss])
                ax2.plot(steps,values,linewidth=6,label="$N_E=$"+labels)
                steps, values = zip(*[(x.step, x.value) for x in scalar_std])
                ax3.plot(steps,values,linewidth=6,label="$N_E=$"+labels)      
                steps, values = zip(*[(x.step, x.value) for x in scalar_cos_sim])
                # ax4.plot(steps,values,linewidth=2,alpha=0.6,label="$N_E=$"+labels_prefix[i])
                line, = ax4.plot(steps,values,linewidth=3,alpha=0.2)
                #对cos similarity的值进行平滑处理
                values = np.array(values)
                for j in range(1,len(values)):
                    values[j] = 0.9*values[j-1] + 0.1*values[j]

                ax4.plot(steps,values,linewidth=6,color=line.get_color(),label="$N_E=$"+labels)

                steps,values = zip(*[(x.step,x.value) for x in scalar_value_loss])
                ax5.plot(steps,values,linewidth=6,alpha=0.8,label="$N_E=$"+labels)
        i+=1          
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
plt.show()



"""绘制专家和EGPO和BC的运动速度和关节角度图"""
# import numpy as np
# from scipy.signal import lfilter

# # def extract_q_data(ea,title):
# #     q1 = ea.Scalars(title+'/q1')
# #     q2 = ea.Scalars(title+'/q2')
# #     q3 = ea.Scalars(title+'/q3')
# #     steps_q1, q1 = zip(*[(x.step*0.02, x.value) for x in q1])
# #     _, q2 = zip(*[(x.step, x.value) for x in q2])
# #     _, q3 = zip(*[(x.step, x.value) for x in q3])
# #     # steps_q1=int(float(steps_q1)*0.02)
# #     return q1,q2,q3,steps_q1
# def extract_v_data(ea,title):
#     vx=ea.Scalars(title+'/vx')
#     vy=ea.Scalars(title+'/vy')
#     steps,vx=zip(*[(x.step*0.02,x.value) for x in vx])
#     _,vy=zip(*[(x.step*0.02,x.value) for x in vy])
#     return vx,vy,steps
# # 事件文件路径
# log_dir=f"{LEGGED_GYM_ROOT_DIR}/logs/motion_data/"
# event_files=[]
# for file in os.listdir(log_dir):
#     if file.endswith('.0'):
#         event_files.append(file)
# print(event_files)

# file_labels=['EGPO','BC','Expert']
# # file_labels=['Expert','BC']
# # file_labels=['Expert']
# ea_list=[]
# # fig_q=plt.figure("joint")
# plt.rcParams.update({'font.size': 16})
# fig_v=plt.figure("Velocity",figsize=(10,6))
# color_list=[['#2BB449',"#058623"],["#7466EB","#271DB9"],["#DB6E6E","#861A1A"]]
# cmd_v_list=[[0.4,0],[0,0.55],[0.4,0.5]]
# alpha=0.93 #平滑因子
# for i,event_file in enumerate(event_files):
#     ea=event_accumulator.EventAccumulator(log_dir+event_file)
#     ea.Reload()
#     ea_list.append(ea)
#     # q1,q2,q3,steps=extract_q_data(ea_list[i],file_labels[i])
#     line_style='-'
#     line_width=1.5
#     # if i==0:
#     #     line_style='--'
#     #     line_width=1.5

#     # plt.plot(steps,q1,label=file_labels[i]+'/q1',ls=line_style)
#     # plt.plot(steps,q2,label=file_labels[i]+'/q2',ls=line_style)
#     # plt.plot(steps,q3,label=file_labels[i]+'/q3',ls=line_style)
#     vx,vy,steps=extract_v_data(ea_list[i],file_labels[i])
#     vx,vy=np.array(vx),np.array(vy)
#     vx_smooth = lfilter([1 - alpha], [1, -alpha], vx)
#     vy_smooth = lfilter([1 - alpha], [1, -alpha], vy)
#     #计算误差的大小

#     er1x=np.linalg.norm(vx_smooth[200:350]-0.4)
#     er1y=np.linalg.norm(vy_smooth[200:350])

#     er2x=np.linalg.norm(vx_smooth[550:700])
#     er2y=np.linalg.norm(vy_smooth[550:700]-0.6)  

#     er3x=np.linalg.norm(vx_smooth[800:1050]-0.3)
#     er3y=np.linalg.norm(vy_smooth[800:1050]-0.45)  

#     print(er1x,er1y,er2x,er2y,er3x,er3y)

#     print((er1x+er1y+er2x+er2y+er3x+er3y)/3.0)
#     # plt.figure("Velocity")
#     # plt.plot(steps,vx,label=file_labels[i]+'/vx',linewidth=line_width,ls=line_style)
#     plt.plot(steps,vx_smooth,label=file_labels[i]+': $v_x$',linewidth=line_width,ls=line_style,color=color_list[i][0])
#     # plt.plot(steps,vy,label=file_labels[i]+'/vy',linewidth=line_width,ls=line_style)
#     plt.plot(steps,vy_smooth,label=file_labels[i]+': $v_y$',linewidth=line_width,ls=line_style,color=color_list[i][1])
#     # plt.ylim(-0.75, 0.75) 
# # 虚线位置
# for x in (4, 11, 18):
#     plt.axvline(x, color='black', linestyle='--', linewidth=1)

# # 实线位置
# for x in (7, 14, 21):
#     plt.axvline(x, color='black', linestyle='-', linewidth=1)
# plt.plot([0,7,7,14,14,21],[0.4,0.4,0,0,0.4,0.4],ls='-.',linewidth=2,label="cmd: $v_x$")
# plt.plot([0,7,7,14,14,21],[0,0,0.55,0.55,0.5,0.5],ls='-.',linewidth=2,label="cmd: $v_y$")
# plt.xlabel('time(s)',fontsize=20)
# plt.ylabel('velocity(m/s)',fontsize=20)
# plt.legend(
#     loc="lower left",    # 图例在图的上方（left表示居中）
#     bbox_to_anchor=(0.5, -0.03),  # 调整图例位置 (x=0.5表示居中, y=-0.1表示往下移到图外)
#     ncol=4,                # 图例排成几列，这里4列
#     frameon=True          # 去掉图例边框
# )

# plt.show()


"""绘制多个奖励函数"""
# import numpy as np
# from scipy.signal import lfilter
# smooth_factor=0.7
# plt.rcParams.update({'font.size': 32})
# plt.figure(figsize=(16,10))
# # root_file=f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/"
# # file_names=['Sep10_13-44-29_ /events.out.tfevents.1757483070.uss.53815.0',
# #            'Sep10_13-47-32_ /events.out.tfevents.1757483253.uss.54420.0',
# #            'Sep10_14-43-36_ /events.out.tfevents.1757486618.uss.62989.0',
# #            'Sep10_15-46-09_ /events.out.tfevents.1757490370.uss.80411.0']
# root_path = "/home/val/BIH_ws/legged_gym/logs-art/hex_ground"
# # root_path = "/home/val/BIH_ws/legged_gym/logs-art/hex_residule"
# event_files=[]
# for files in os.listdir(root_path):
#     # if "Jan19_14" in files or "Jan19_15" in files:
#     # if "Jan19_17" in files:
#     if "Jan19_21" in files:
#     # if "Jan20" in files:
#         print("files=",files)
#         for event_file in os.listdir(os.path.join(root_path,files)):
#             if event_file.endswith("0"):
#                 event_files.append(os.path.join(root_path,files,event_file))
# #专家的奖励，只有2000次
# event_files.append("/home/val/BIH_ws/legged_gym/logs-art/hex_ground/Jan19_17-44-02_/events.out.tfevents.1768815843.art.3258755.0")
# event_files = sorted(event_files)

# print(event_files)
# #残差的奖励
# event_files.append("/home/val/BIH_ws/legged_gym/logs-art/hex_residule/Jan20_10-27-27_/events.out.tfevents.1768876049.art.3442542.0")
                                     
# # file_labels=['PPO','EGPO','IL+PPO','EGPO','PPO','Expert']
# file_labels = ['Expert','PPO','IL+PPO','EGPO','Residule']
# # file_labels = [1,2,3,4,5,6]
# # color_list=["purple","orange","blue","red",'green','black']
# color_list=["purple","orange","blue","red",'green','black']



# for i,file in enumerate(event_files):
#     if i in [5]:
#         continue
#     # if i==3:
#         # break
#     ea=event_accumulator.EventAccumulator(file)
#     ea.Reload()
#     scalars=ea.Scalars('Train/mean_reward')
#     steps, rewards=zip(*[(x.step, x.value) for x in scalars][:500])
#     # if i==0:
#     #     steps = np.arange(1,4001,1)
#     #     rewards = np.array(rewards)
#     #     add_rew = rewards[100:] + np.random.rand(1900)*0.1
#     #     rewards = np.hstack([rewards,add_rew,rewards[1000:1100]])
#     plt.plot(steps,rewards,linewidth=7.0,alpha=0.3,color=color_list[i])
#     rewards_smoothed = lfilter([1 - smooth_factor], [1, -smooth_factor], rewards)
#     plt.plot(steps,rewards_smoothed,linewidth=3.0,label=file_labels[i],color=color_list[i])
# plt.xlabel("Iterations")
# plt.ylabel("Mean Reward")
# # plt.ylim(0,60)
# plt.legend()
# plt.show()


"""从bag中读取速度数据，然后计算速度误差"""
# import rosbag
# from pathlib import Path
# import numpy as np
# agent_bag_name="/home/val/BIH_ws/bag/velocity/2025-09-18-15-04-28.bag"#EGPO数据
# expert_bag_name="/home/val/BIH_ws/bag/velocity/2026-01-22-16-26-40.bag"#专家数据
# # expert_bag_name = "/home/val/BIH_ws/bag/velocity/2026-01-22-16-14-11.bag"

# total_err=0.0
# vx_des=[]
# vy_des=[]
# vz=[]
# vx_real_agent=[]
# vy_real_agent=[]
# vx_real_expert=[]
# vy_real_expert=[]
# times=[]
# times2=[]
# err=[]
# i=0
# v_max=-0.05
# with rosbag.Bag(agent_bag_name,'r') as bag:
#     for topic, msg, t in bag.read_messages(topics=["/base_msg"]):
#         i+=1
#         if i>60:
#             msg=np.array(msg.data)
#             # vx_des.append(msg[3])
#             # vy_des.append(msg[4])
#             vx_real_agent.append(msg[0])
#             vy_real_agent.append(msg[1])
#             vz.append(msg[2])
#             err.append(np.linalg.norm(msg[0:2]-msg[3:5]))
#             times.append(i*0.02)
#         if i>2600:
#             break
# i=0
# with rosbag.Bag(expert_bag_name,'r') as bag:
#     for topic, msg, t in bag.read_messages(topics=["/base_msg"]):
#         i+=1
#         if i>60:
#             msg=np.array(msg.data)
#             vx_real_expert.append(msg[0])
#             vy_real_expert.append(msg[1])
#             vx_des.append(msg[3])
#             vy_des.append(msg[4])
#             vz.append(msg[2])
#             err.append(np.linalg.norm(msg[0:2]-msg[3:5]))
#             times2.append(i*0.02)
#         if i>2600:
#             break
# err=np.array(err)
# print(np.sum(err)/len(err)/2)
# #220:1800
# plt.figure(figsize=(24,15))
# plt.rcParams.update({'font.size': 40})
# plt.plot(times2,vx_des,ls='-',linewidth=5,label="cmd: $v_x$",color='red')
# plt.plot(times2,vy_des,ls='-',linewidth=5,label="cmd: $v_y$",color='blue')
# plt.plot(times,vx_real_agent,ls='-',linewidth=2,label="EGPO: $v_x$",color='magenta')
# plt.plot(times,vy_real_agent,ls='-',linewidth=2,label="EGPO: $v_y$",color='cyan')
# plt.plot(times2,vx_real_expert,ls='-',linewidth=2,label="Expert: $v_x$",color='orange')
# plt.plot(times2,vy_real_expert,ls='-',linewidth=2,label="Expert: $v_y$",color='purple')
# # plt.plot(times,err,ls='-',linewidth=1,label="err",color='#860EE8FF')
# plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=3)
# plt.tight_layout()
# plt.xlabel("times(s)")
# plt.ylabel("velocity(m/s)")
# plt.show()
# print(v_max)
# import torch
# v_cmd = torch.vstack([torch.tensor(vy_des),torch.tensor(vx_des)])
# print(v_cmd)
# torch.save(v_cmd,"/home/val/BIH_ws/bag/command_sequence.pth")
# bag_name="/home/ubuntu/valerian_ws/BIH_ws/bag/velocity/2025-09-18-15-04-28.bag"
# total_err=0.0
# i=0.0
# vx_des=[]
# vy_des=[]
# vx_real_agent=[]
# vy_real_agent=[]
# with rosbag.Bag(bag_name,'r') as bag:
#     for topic, msg, t in bag.read_messages(topics=["/base_msg"]):
#         msg=np.array(msg.data)
#         print("msg=",msg)
#         total_err +=np.linalg.norm(msg[0:2]-msg[3:5])/3.0
#         i+=1.0
# print("ave_err=",total_err/i)

# '#271DB9'
