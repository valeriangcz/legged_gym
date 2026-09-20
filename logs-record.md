# 用于记录logs中每一次训练的情况

纯RL强化学习结果：Jun15_16-20-05_ 
EGPO，loss function分成两个独立的阶段，第一阶段有BC_loss,第二阶段有PPO的loss：Jun25_16-27-30_ 
EGPO，loss function分成两个独立的阶段，第一阶段有BC_loss,第二阶段有PPO的loss,这次采用500迭代次数进行插值：Jun26_12-35-19_ 
模仿学习作为预训练策略加载到actor网络中，开始强化学习：Jun25_18-04-05_ 模仿学习阶段有些过拟合
模仿学习作为预训练策略，网络输出的std为0.18左右，加载到actor网络中，强化学习：Jun27_09-47-01_  动作幅度很小，遍地乱爬 (进一步实验？）std很小导致的
调整了奖励函数和部分参数的纯RL学习结果：Jun27_09-28-52_ 有一条腿一直抬高，很奇怪，前进速度很慢，转向很快



实物电机参数训练，准备迁移的：Jul04_11-19-27_ 
EGPO，loss function分成两段，第一段有BC_loss，第二段没有BC_loss：Jul04_17-27-03_ 

纯强化学习的base_line结果作为对比参照：

采用动作和BC_loss递减的光滑过渡,过渡参数为200：Jul09_09-26-18_ 

全部采用专家动作，不进行参数更新，只是为了计算专家可以收货多少收益




Sep08_22-40-03_ :EGPO调整参数后用于奖励函数曲线绘制和可视化
Sep09_09-24-05_ :用于展示纯RL训练调整reward之后训练的结果，不自然，并且有些关节没有用到
Sep09_11-24-36_ :用于训练纯RL训练失败的曲线可视化
Sep09_16-48-03_ :用于生成200次的EGPO，跟踪速度曲线展示用

Sep10_13-44-29_ :用于BC提前加载然后使用PPO算法训练
Sep10_13-47-32_ :用于训练2000轮的纯RL算法获取奖励曲线PPO
Sep10_14-43-36_ :用于训练EGPO展示奖励函数
Sep10_15-46-09_ :用于测试expert能获得的奖励多少


Sep11_12-24-21_ :用于观测机器人腿总是朝向一侧是不是因为专家速度曲线一直在期望值下方的原因，调整了专家速度，使其能在期望速度两侧的位置（初步确定不是这个问题）
Sep11_12-59-18_ :是不是由于熵的衰减因子导致的，衰减因子设置为1.0常数，设置BC学习权重常数项为1.0 模仿阶段失败
Sep11_13-03-41_ ：由于接触力和接触力变化惩罚很大导致，接触力相关权重减小，熵衰减因子设置衰减，BC学习权重调整为5.0 步伐很小仍然前倾
Sep11_14-18-44_ :重新设置来expert的z值，步伐大一点，重新训练，熵衰减因子仍然为1.0常数，衰减因子常数后，机器人动作会自动降低重心
Sep11_15-01-12_  :设置熵衰减因子从头就开始衰减



Sep11_15-42-56_  :是初始的动作std太大导致学习出来动作很奇怪？设置init_std为1.0重新训练，使用的是0911新的电机模型 贴地了

Sep11_16-02-49_  ：

想延长expert在学习中的时间，采用新的BC损失函数                 
BC_loss_fn = torch.nn.MSELoss()
BC_loss = BC_loss_fn(agent_actions_batch,expert_actions_batch)*alpha
同时设置专家伴随时间为500it，专家指导动作差值和衰减都是线性 100次左右出现NAN问题
Sep11_16-10-18_ 设置200it后重新开始测试

Sep11_16-10-18_ 设置200it,设置BC_loss = BC_loss_fn(agent_actions_batch,expert_actions_batch)*alpha*5.0后重新开始

# 设置真实机器人上控制频率为50hz，采用与expert相同的控制参数重新收集电机参数
# 使用200次训练的智能体测试仿真和实物之间存在的差距 效果比较流畅
# 是否因为BC_loss采用的是log_prob，所以导致方差很小，导致探索性不足与奇怪问题，采用MSE损失测试bc_preload_runner


Sep11_16-30-51_ 使用MSE_loss训练的BC智能(使用act_inference训练得到bc_agent2)体强化学习训练 这种方法无法使std下降，当专家逐渐退出时，奖励下降速度很快

Sep11_16-58-25_ 使用原来的BC_loss,但是采用线性衰减因子，200次的时候机器人静止不动，而且后期训练时机器人会非常贴近地面


改变奖励函数，记录原始参数：

action_rate = -0.01
tracking_ang_vel = 1.5
tracking_lin_vel = 2.0
lin_vel_z = -10.0
ang_vel_xy = -0.2
# base_height = 0.5
orientation = -5.0
feet_air_time = 0.0

dof_acc=-2.0e-7

stand_still = -2.0
feet_contact_forces = -0.0005
pass
#针对六足添加的奖励：
# footend_pos_xy = -0.1 #足端远离初始位置越远，惩罚越高
#足端力的增加变化，增加的越快，惩罚越大
feet_contact_forces_increase = -0.0005


修改的内容
tracking_ang_vel = 5.0
tracking_lin_vel = 6.0
dof_vel = -2.0e-5


Sep11_17-31-32_ 修改后训练 同样贴近地面，但是六个足的位置非常对称且均匀，因为dof_vel的缘故？

Sep11_19-16-53_ 取消dof_vel采用同样的参数进行训练，探究是否因为dof_vel所以六个足的位置非常对称均匀（仍然非常均匀，不是dof_vel的原因）

___以下训练都出现nan，共同点都是BC_loss计算结果为-30多__
{
    # Sep11_20-04-30_ 
    使用以下参数进行训练，并且添加thigh为碰撞
                action_rate = -0.01
                tracking_ang_vel = 5.0
                tracking_lin_vel = 6.0
                lin_vel_z = -1.0
                ang_vel_xy = -0.2
                base_height = 1.0
                orientation = -20.0
                feet_air_time = 0.0

                dof_acc=-2.0e-7
                dof_vel = -2.0e-5

                stand_still = -2.0
                feet_contact_forces = -0.0005

                CoT = -0.001
                pass
                #针对六足添加的奖励：
                # footend_pos_xy = -0.1 #足端远离初始位置越远，惩罚越高
                #足端力的增加变化，增加的越快，惩罚越大
                feet_contact_forces_increase = -0.0005 

            only_positive_rewards = False
            tracking_sigma = 0.15
            base_height_target = 0.12
            max_contact_force = 60.0
    100次时出现nan，取消thigh为终止条件，再测试 
    # Sep11_20-10-12_ 


    # Sep11_20-08-08_ 
    探究是否是因为两个正奖励很大所以十分对称
                action_rate = -0.01
                tracking_ang_vel = 1.5
                tracking_lin_vel = 2.0
                lin_vel_z = -1.0
                ang_vel_xy = -0.2
                # base_height = 1.0
                orientation = -5.0
                feet_air_time = 0.0

                dof_acc=-2.0e-7
                # dof_vel = -2.0e-5

                stand_still = -2.0
                feet_contact_forces = -0.0005

                # CoT = -0.001
                pass
                #针对六足添加的奖励：
                # footend_pos_xy = -0.1 #足端远离初始位置越远，惩罚越高
                #足端力的增加变化，增加的越快，惩罚越大
                feet_contact_forces_increase = -0.0005 

            only_positive_rewards = False
            tracking_sigma = 0.15
            base_height_target = 0.12
            max_contact_force = 60.0
}

__设置BC_loss检查__
# Sep11_20-17-57_  [截止目前，训练效果最好的]
 如果BC_loss低于-20，直接退出BC模仿学习


__设置thigh关节碰撞为终止条件__
# Sep11_20-36-39_ 
效果并不理想，甚至出现了不对称的动作和部分关节停止工作然后thigh碰撞地面导致回合结束


__探究是不是正奖励权重大因此动作对称__
# Sep11_20-47-55_ 
原始权重：
            action_rate = -0.01
            tracking_ang_vel = 5.0
            tracking_lin_vel = 6.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.2
            base_height = 1.0
            orientation = -20.0
            feet_air_time = 0.0

            dof_acc=-2.0e-7
            dof_vel = -2.0e-5

            stand_still = -2.0
            feet_contact_forces = -0.0005

            CoT = -0.001
            pass
            #针对六足添加的奖励：
            # footend_pos_xy = -0.1 #足端远离初始位置越远，惩罚越高
            #足端力的增加变化，增加的越快，惩罚越大
            feet_contact_forces_increase = -0.0005 

        only_positive_rewards = False
        tracking_sigma = 0.15
        base_height_target = 0.12
        max_contact_force = 60.0
采用的权重
           action_rate = -0.01
            tracking_ang_vel = 1.5
            tracking_lin_vel = 2.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.2
            # base_height = 1.0
            orientation = -5.0
            feet_air_time = 0.0

            dof_acc=-2.0e-7
            # dof_vel = -2.0e-5

            stand_still = -2.0
            feet_contact_forces = -0.0005

            # CoT = -0.001
            pass
            #针对六足添加的奖励：
            # footend_pos_xy = -0.1 #足端远离初始位置越远，惩罚越高
            #足端力的增加变化，增加的越快，惩罚越大
            feet_contact_forces_increase = -0.0005 

        only_positive_rewards = False
        tracking_sigma = 0.15
        base_height_target = 0.12
        max_contact_force = 60.0
确实会出现不对称的动作


__使用command_curriculum测试机器人能达到的极速__

# Sep11_21-34-36_ 出现一次nan
# Sep11_21-40-30_ 出现nan
都是在54次迭代的位置，一切参数正常，BC_loss为-4

# Sep11_21-47-28_ 设置curriculum为False测试

# 设置curriculum为True时，减小BC_loss的权重，从2.0 减小为 1.0


# 采用分段式奖励函数
# Sep11_22-26-17_ 
             if it<200:
                    loss=self.value_loss_coef * value_loss - alpha * self.entropy_coef * entropy_batch.mean()+BC_loss
                else:
                    loss=surrogate_loss + self.value_loss_coef * value_loss - alpha * self.entropy_coef * entropy_batch.mean()

# 相互交叉式奖励函数
# Sep12_10-29-01_ [训练结果较好]
采用了command curriculum, vy最高1.5m/s
                if it<150:
                    loss=self.value_loss_coef * value_loss - alpha * self.entropy_coef * entropy_batch.mean()+BC_loss
                else:
                    loss=surrogate_loss + self.value_loss_coef * value_loss - alpha * self.entropy_coef * entropy_batch.mean()


# 复现NAN情况
# Sep12_11-08-52_ 
BC_loss = -self.actor_critic.get_actions_log_prob(expert_actions_batch).mean(dim=-1)*20*alpha
loss=surrogate_loss + self.value_loss_coef * value_loss - alpha * self.entropy_coef * entropy_batch.mean()+BC_loss

# 先不探究nan出现的原因，先采用相互交叉式的损失函数训练一个理想PD控制器
__理想的PD控制器和command_curriculum测试机器人可以达到的极速__
# Sep12_11-36-35_ 
设置了不适用电机网络模型
修改了hex_ground_cfg中 command range: 
原值
        class ranges:
            lin_vel_x=[-1.0,1.0]
            lin_vel_y=[-1.5,1.5]
            ang_vel_yaw=[-2,2]
修改后
        class ranges:
            lin_vel_x=[-3.0,3.0]
            lin_vel_y=[-5,5]
            ang_vel_yaw=[-3,3]

# 使用PPO训练机器人，观察是否是这个reward使机器人可以做出对称的动作
# Sep12_14-47-16_ 

# 使用EGPO训练，最高速度为1.2m/s 1.0 2.0,修改了expert中vz的计算方式
# Sep12_16-50-14_ 


# 缺少reward_base height
# Sep12_17-36-07_ base_height还是比较有必要的


# 设置距离swing_init_point的奖励，越靠近这个初始点，奖励越高，带reward_base height
# Sep12_18-52-08_ 
想让机器人的腿部动作的运动一直保持在初始点附近，不能越执行指令越远
虽然执行的动作很靠近了，但是优化到后期，步子的幅度很小，频率很快

# 设计了梯度为NAN自动忽略，不进行step，尝试让专家伴随更久的时间，希望通过这种方式让专家对称性动作影响更多一些


# 设计奖励函数，腿部从抬升到落地，越靠swing_init_point奖励越高，到达阈值后保持奖励直到落地，希望通过这种方式鼓励对称的动作
# Sep15_15-00-40_ 
确实会生成较为对称的动作，但是机器人跳动

# 改进奖励函数，靠近给奖励，到阈值后，远离给奖励，采用误差变化权重平滑这两项
# Sep15_16-23-08_ 
算法好像放弃优化这一部分了，早期奖励太过敏感，误差较大时给出的奖励太小，引导比较少


# 发现使用EGPO 经过200次训练出来的智能体在执行1.2m/s前进速度时出现小碎步的状态，更改Expert的vz速度更小重新训练
# Sep15_17-34-11_ 

# 将Expert的vz速度再改小进行训练
# Sep15_17-48-53_ 步伐会增大


# 修改后的Expert+完成debug的swing奖励计算
# Sep15_18-01-48_  并没有出现对称的动作

# 设置expert伴随时间长一点1000次
# Sep15_19-45-09_ ,频繁出现nan

# 设置command_curriculum，expert伴随时间为200次
# Sep15_19-57-22_ 

# 设置reward_pos_xy,权重减小
# Sep15_20-04-25_ [这个奖励效果最好，可以一直保证较为匀称的动作，但是到了后期容易出现小步频的情况]


__理想的PD控制器和command_curriculum测试机器人可以达到的极速__
# Sep17_09-42-42_ 
设置了不适用电机网络模型
修改了hex_ground_cfg中 command range: 
原值
        class ranges:
            lin_vel_x=[-1.0,1.0]
            lin_vel_y=[-1.5,1.5]
            ang_vel_yaw=[-2,2]
修改后
        class ranges:
            lin_vel_x=[-3.0,3.0]
            lin_vel_y=[-5,5]
            ang_vel_yaw=[-3,3]

# 重新设计专家，采用逆运动学解析解进行解算
# Sep17_15-47-06_ 
设置vy最高速度为0.8m/s，没有使用curriculum
设计xy_dist的奖励如下形式
        xy_dist=torch.norm(self.expert.B_e_cur[...,0:2]-self.expert.swing_init_point[:,0:2],dim=-1).sum(dim=-1)
        xy_dist[xy_dist<0.25]=0.0

# 取消reward_height foot_pos_xy奖励项，重新训练
# Sep17_15-59-01_ [训练结果较好，可能是速度0.7时专家给出的动作参照较好]

# 设计xy_dist奖励如下, 没有reward_base_height
# Sep17_16-08-19_ 
self.expert.kin.ForwardKin(self.dof_pos.view(-1,3),self.expert.B_e_cur_flat)
xy_dist=torch.norm(self.expert.B_e_cur[...,0:2]-self.expert.swing_init_point[:,0:2],dim=-1).sum(dim=-1)
# xy_dist=(xy_dist*(~contact_filt)).sum(dim=1) #只计算swing状态下的contact_filt
xy_dist[xy_dist<0.2]=0.2
# rew=(torch.exp(-xy_dist/0.12)*(~contact_filt))/(torch.sum(~contact_filt,dim=1)+1e-6)
return torch.exp(-xy_dist/(0.4))

# 设置command_curriculum，从vy从0.7开始到1.5结束
设置速度跟踪权重小一点
# Sep17_17-38-08_ 

# 增加dof_acc的权重
# Sep17_21-36-49_ -2e-7  -2e-6

# 训练一个速度小一点的，用于展示的
# Sep17_21-42-25_ 
        class ranges:
            lin_vel_x=[-0.5,0.5]
            lin_vel_y=[-0.6,0.6]
            ang_vel_yaw=[-1.5,1.5]

# 减小vel_z的权重，减小这一部分惩罚，其余与 Sep17_21-36-49_ 保持相同
从-2.0变为-1.0
# Sep17_22-05-43_ 

# 再增加dof_acc的权重
-2e-6 -4e-6
# Sep17_22-07-46_ 

[总结]
Sep17_22-05-43_ 出现了大踏步行走的状态，较大的dof_acc惩罚和较小的lin_vel_z的惩罚会鼓励这一动作，在这个过程中foot_pos_xy反而一直减小，并没有起到很大作用


# 设置dof_acc -2e-6 lin_vel_z -1.5，取消foot_pos_xy奖励项， 训练一个慢速的
# Sep18_10-35-21_ 
        class ranges:
            lin_vel_x=[-0.4,0.4]
            lin_vel_y=[-0.6,0.6]
            ang_vel_yaw=[-1.5,1.5]

————————————————————————————————平地训练————————————————————————————————————————
# 减小init_std为0.5,同时设置速度范围
# Sep18_15-52-54_ 
从
        class ranges:
            lin_vel_x=[-0.4,0.4]
            lin_vel_y=[-0.6,0.6]
            ang_vel_yaw=[-1.5,1.5]
变为
        class ranges:
            lin_vel_x=[-0.6,0.6]
            lin_vel_y=[-0.8,0.8]
            ang_vel_yaw=[-2,2]
设置训练1200次

——————————————————————————————————————————以下为复杂地形训练——————————————————————————————————————————

# 复杂地形上训练ActorCriticEncoder
# Sep18_12-58-15_ 
dof_acc CoT权重太大，BC_loss收敛太慢


# 取消CoT惩罚，dof_acc变为2e-7, 给探索熵增加衰减因子，动作网络的初始噪声从0.8开始
# Sep18_15-37-29_ 


# Sep19_08-44-59_ 
# Sep19_11-23-03_ 

# 取消关于足端力的惩罚，重新设置地形curriculum,重新设计去高地形和低地形的条件
# Sep20_11-27-55_ 发现期望的角度抖动很大

# 提高dof_acc的权重，训练
# Sep22_10-03-26_ 
训练出来的效果抖动比较严重，并且对不同电机模型效果很差


# 使用简单的actor critic模式在复杂地形上进行训练，不添加编码器
重力，线速度，接触力，地形高度为特权观测，高度测量误差从10cm改到2cm
# Sep22_14-44-00_ 
# 修改高度测量结果后重新训练
# Sep23_09-59-47_ value_loss还是很大，
把线速度，重力，接触力从观测中删去训练测试
# Sep23_10-43-13_ 
设置10s的episode时长（500步），设置48步的收集更新时长，设置5s的指令更新速度，重新训练 
# Sep23_11-20-55_ 
确认 value function loss的突变是因为长的episode（20s）和复杂的地形带来观测空间维度的升高，导致GAE估计出问题

重新修改expert，降低角速度旋转；考虑到每次迭代收集的数据是原来的2倍，因此修改专家引导时间为原来一半
# Sep23_11-56-35_ 

# 重新引入encoder，在复杂地形上训练
# Sep23_15-20-13_ 

修改更新逻辑，之前都是critic对地形编码器进行更新，这次使用actor对地形编码器进行更新
# Sep23_20-36-26_
加入feet_contact_force的惩罚
# Sep23_21-34-23_
重新设置terrain_curriculum进行训练
# Sep24_11-32-12_ 
机器人攀爬效果有提升，但是腿部仍然抖动，并且value的误差较大

尝试使用更深的网络，使用更加精准的电机模型，初始机器人放置的地形最高水平为2
# 
统一条件，使用更长的LSTM历史，从20次改为20次
# 

value可能与collision有关系，去掉这部分惩罚，训练（平台尺寸不能太小了，容易刚开始就卡进模型里）
# Sep24_17-51-36_

加入这一部分惩罚，同时引入thigh knee ankle的接触力norm的大小
#



使用75 230观测和actor critic训练机器人在平地，设置最高速度0.3 和 0.2m/s 0.3rad/s观察机器人的运动
需要修改速度奖励函数的sigma,使其对低速区域的奖励更加敏感，修改stand_still和命令重新采样中指令速度的判断条件
训练结果很奇怪，机器人的运动方式奇形怪状
# Sep24_14-25-51_




# 使用简单actor critic模式训练，地形特权信息actor也可以获取，观察其在复杂地形上的效果
# Sep22_14-47-56_ 
观测改为：
self.obs_buf = torch.cat([self.last_actions*self.obs_scales.actions,
                            (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
                            self.dof_vel*self.obs_scales.dof_vel,
                            self.torques*self.obs_scales.dof_torque,
                            self.commands*self.commands_scale,
                            self.measured_heights*self.obs_scales.height_measurements],dim=-1) 
发现value网络的训练损失很不稳定
# 机器人的高度计算需要转换到本体坐标系下面，之前是全局坐标系，不具备参考意义，同时需要对高度进行裁剪，继续训练
# Sep22_17-57-13_

# 设置训练场地为平地，选择与上面相同的观测，观察价值函数损失
# Sep22_16-22-02_ 



先不用复杂地形训练
# Sep24_19-54-44_ Sep24_21-48-00_

加入feet_air_time奖励，删去base_height奖励，使用的理想电机
# Sep25_14-27-52_

使用拟合好的电机，加入feet_air_time奖励，训练
# Sep25_16-39-27_ 
这个模型200训练出来实物运行效果较好，参数调整参考 BC_loss权重2.0，熵探索敏感系数0.2

vel_z较大，身体较为倾斜，
增加vel_z，orientation，权重，增加foot_pos_xy,
# Sep25_20-40-41_

将速度跟踪的奖励设为小于范围就给所有的奖励
# Sep25_21-35-15_  
vy超过太多

将cot和dof_vel加入惩罚中，压制vy的速度
# Sep25_22-05-05_ 
COT有效，机器人会以更慢的速度跟踪速度指令，我们希望的是可以在期望速度附近浮动，而不是一直低于期望速度


目前以上所有方法，离开专家训练后，期望动作和真实动作的误差非常大，期望动作往往高于实际动作很多，这就导致了实物和仿真之间的差异
一方面要限制action_rate_
一方面要建立覆盖范围更加广泛的电机模型，目前为了测量精准，采用的慢速电机模型构建，并不是很准确

尝试以下方法
1、增加action_rate的权重（取消CoT相关惩罚）
action_rate权重-0.01 -> -0.08，减小footend_pos_xy权重
# Sep26_11-56-12_ 
[action_rate确实有用,但是不能降低footend_pos_xy的权重]

1、进一步缩小BC_loss的权重，延长专家引导的时间，专家撤出后，训练完毕
BC_loss权重为0.02,专家引导时间为600
# Sep26_11-57-57_ 
200后效果急转直下，怀疑是BC_loss的权重太小了

提高权重到0.1重新测试
# Sep26_12-51-19_
120后效果急转直下，并且训练后频繁出现梯度为nan的情况，原因尚未清楚


3、恢复专家引导时间和BC_loss的权重，加入误差追踪惩罚，
加入摆动和支撑靠近初始点的奖励（靠近阈值给奖励一次，等下次进入摆动或者支撑时再给）
同时将速度追踪误差要求在0.1以内，角度度追踪误差在0.2以内
# Sep26_15-44-43_
引入摆动支撑奖励会使机器人频繁切换摆动和支撑两种状态，并且这一项的奖励高于feet_air_time，因此导致机器人小幅度快速运动


先使用正常的pos_xy，设置误差追踪惩罚
# Sep26_16-57-28_


pos_xy权重要减小，让feet_air_time增加，设置action_rate权重增加
# Sep26_17-17-21_
效果较好，有待实际部署，部署效果较好，直接在复杂地形上开始训练
仿真和实物还是有比较大的差距，


尝试恢复模仿损失权重，减小熵衰减因子
# Sep28_10-10-58_


还是想通过reward的调节，让算法优化出来的模型可以慢速大步走，而不是约束初始他与专家的距离
减小模仿损失权重，恢复熵衰减因子
# Sep28_11-10-41_

感觉速度奖励的权重有点太大了，稍微降低速度权重奖励，提升action_rate权重
# Sep28_11-31-45_

action rate明显下降了，但是足端位置很奇怪，想到之前在复杂地形上训练时，虽然没有这方面的奖励，但训练出来的步态却非常自然
因此想尝试去掉这个奖励，然后在复杂地形上训练机器人
# Sep28_13-06-51_

复杂地形上可以起到对机器人足端摆动位置修饰的作用，可以更加平均
测试时发现另一个问题，虽然奖励lin_vel_z提高action_rate可以让机器人走出较大的步伐，但是感觉foot_contact_force非常大，尽管已经提升了这项的权重
是不是可以在提升lin_vel_z的同时，提升action_rate，让两者的惩罚都比较大
# Sep28_15-02-23_
目前该参数效果比较好

使用效果较好的这一套参数在复杂地形上训练
# Sep28_16-01-51_
发现这个的ankle关节实物部署和仿真时差距很大，仿真时微小振动，误差较大，实物测试时服务很大，具有延迟
怀疑是三个电机的力矩同时输出导致的问题，采用单个电机输出的网络测试[slow.bag;slow_fast.bag拟合，具有1.8NM的误差]，在平地上训练
# Sep28_19-03-05_
所有关节与仿真中输出有差别，但不是非常大，所以不能采用三个电机力矩同时输出的方法，网络会拟合到一些非常奇怪的东西
需要调整PID，使电机所有的PID参数都相同，然后重新收集数据拟合

重新收集数据拟合，三个电机分别采用三个网络进行拟合，然后在复杂地形上进行训练
# Sep29_14-37-59_

感觉模仿学习阶段还是要增加一点权重，改到2.0再进行尝试
# Sep29_15-09-58_

需要比较上面两个算法训练得到的机器人哪个表现更好一些，单从奖励函数和损失函数来看，两者效果差别不是很大
sim2real效果较好，考虑进一步在复杂地形上训练



只保留最重要的奖励，其余均为0
tracking_vel tracking_ang CoT 
平地上测试
# Sep29_16-58-34_
效果很奇怪，放弃这个方向


修改了rew_footend_pos_xy的逻辑，加入了两次奖励获得的时间必须大于0.18点的限制，避免腿部状态频繁切换，加入了如果命令为0，就不给奖励
在平地上训练，仿真效果非常平滑，有待实物测试
# Sep29_17-01-32_
使用这种奖励，达到了预期避免快速运动刷奖励的效果，但是机器人仍然困在初始点附近，可能是0.18s的间隔太短了，

采用更长的时间间隔0.25，并且只对swing进行奖励
# Sep29_18-40-39_
前进后退和旋转效果平滑，侧向移动还存在问题，有待实物测试


目前sim2real差距不是非常大，进行两方面训练
1、训练平地运动极速
2、训练非平坦地形灵活自由运动

极速运动 [hex_ground]
# Sep30_10-02-09_ 有待实物测试
# Sep30_11-34-58_ 降低ang_vel_xy权重，提高orientation权重，增加 footend_pos_xy
# Sep30_14-31-50_ 取消COT，减小lin_vel_z,增加dof_acc权重
# 还是原来的footend_pos_xy好用，但是权重不能太高了，同时设置较小的lin_vel_z权重和较大的dof_acc权重
# Sep30_16-02-13_ tracking_dof是不是会加重策略对电机网络的过拟合，取消，同时增加action_rate权重
# Sep30_17-34-46_ 增加一点ang_vel_xy和lin_vel_z，减小一点dof_acc
# Sep30_20-00-17_ 目前效果最好，复现了当时大步行走的状态[有待实物测试] 实物测试完毕发现ankle关节的实际输出与仿真中输出差异较大，不使用ankle关节网络，使用knee关节来预测
感觉腿部前后间距还是有点小了，试着通过更改swing_init_point来拉大距离
# Oct01_10-00-10_
采用knee来预测knee和ankle关节
# Oct01_11-35-51_


非平坦地形灵活自由运动 [hex_terrain]
（value网络由于没有collision信息，因此对状态估值不稳定，将obs_vgf中f从足端力矩扩充为所有rb_force的norm值）
# Sep30_12-02-27_ 机器人完全没有学会专家的动作
# Sep30_14-36-22_ 使用简单的奖励函数再次尝试 效果仍然不佳，应该是专家抬腿高度设置问题，进行修改
# 



Oct20_21-11-41_ 专家kp为40
Oct20_21-14-31_ 专家kp为30


比较command curriculum之间的区别