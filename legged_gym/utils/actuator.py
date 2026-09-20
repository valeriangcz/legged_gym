from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg
import os,re
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from pathlib import Path
# import rosbag
import copy
class Actuator(nn.Module):
    def __init__(self, cfg:HexGroundCfg,device):
        """ cfg: HexV4Cfg
            device: 'cuda' or 'cpu'"""
        super(Actuator,self).__init__()
        self.device = device
        self.cfg=cfg
        motor_hidden_dims=[16,16,16]
        motor_layers = []
        motor_layers.append(nn.Linear(2,motor_hidden_dims[0]))
        motor_layers.append(nn.ELU())
        for i in range(len(motor_hidden_dims)):
            if i==len(motor_hidden_dims)-1:
                motor_layers.append(nn.Linear(motor_hidden_dims[i],1))
            else:
                motor_layers.append(nn.Linear(motor_hidden_dims[i],motor_hidden_dims[i+1]))
                motor_layers.append(nn.ELU())
        self.motor_net = nn.Sequential(*motor_layers).to(self.device)

        #建立三个网络，一个网络负责输出一个关节的角度
        # motor_hidden_dims=[16,16]
        # motor_layers = []
        # motor_layers.append(nn.Linear(2,motor_hidden_dims[0]))
        # # motor_layers.append(nn.)
        # motor_layers.append(nn.ReLU())
        # motor_layers.append(nn.Linear(motor_hidden_dims[0],motor_hidden_dims[1]))
        # motor_layers.append(nn.ReLU())
        # motor_layers.append(nn.Linear(motor_hidden_dims[1],1))
        # self.thigh_net = nn.Sequential(*copy.deepcopy(motor_layers)).to(self.device)
        # self.knee_net = nn.Sequential(*copy.deepcopy(motor_layers)).to(self.device)
        # self.ankle_net = nn.Sequential(*copy.deepcopy(motor_layers)).to(self.device)
        # self.joints_net=[self.thigh_net,self.knee_net,self.ankle_net]

        #set kp kd for motor and servo 
        self.dof_names=self.cfg.init_state.default_joint_angles.keys()
        self.kp_kd_tensor=torch.zeros(2,len(self.dof_names),device=self.device)
        motor_name=['thigh','knee','ankle']
        _motor_index=[]
        _servo_index=[]
        _passive_index=[]
        _torques_limits=[]
        for i, dof_name in enumerate(self.dof_names):
            if dof_name in self.cfg.control.stiffness.keys():
                self.kp_kd_tensor[0,i]=self.cfg.control.stiffness[dof_name]
                self.kp_kd_tensor[1,i]=self.cfg.control.damping[dof_name]
            else:
                print("\033[93m Warning: Failed to find "+dof_name+" in stiffness, set to zeros\033[0m")
                self.kp_kd_tensor[:,i]=0.
            split_name=dof_name.split("_")
            if split_name[2] in motor_name:
                _motor_index.append(i)
            elif split_name[2] == 'foot':
                _servo_index.append(i)
            else:#for passive joints
                _passive_index.append(i)
        self.motor_index=torch.tensor(_motor_index,dtype=torch.int32,device=self.device)
        self.servo_index=torch.tensor(_servo_index,dtype=torch.int32,device=self.device)
        self.passive_index=torch.tensor(_passive_index,dtype=torch.int32,device=self.device)
        self.torques_limits=torch.tensor(_torques_limits,dtype=torch.float32,device=self.device) #dof_nums*2
        self.motor_torque_err=[0.6,1.6,2.0]
        #set mode
        if self.cfg.control.use_actuator_net:
            self._load_motor_net()


    def _load_motor_net(self):
        print(f"--------------->load motor net from {self.cfg.control.actuator_net_file}<---------------------")
        self.motor_net.eval()
        self.motor_net.load_state_dict(torch.load(self.cfg.control.actuator_net_file,weights_only=True))
        # state_dict=torch.load(self.cfg.control.actuator_net_file,weights_only=True)
        # self.thigh_net.load_state_dict(state_dict['thigh'])
        # self.knee_net.load_state_dict(state_dict['knee'])
        # self.ankle_net.load_state_dict(state_dict['ankle'])
    
    def _process_raw_data(self,file_common_name:str):
        #遍历所有文件，将所有的数据拼接为1维度，并保存在flat_raw_data.pt中
        # data_path=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_raw_data/train_dataset/"
        # train_files=[]
        # for file in Path(data_path).iterdir():
        #     if "motor_raw_2025" in file.name:
        #         train_files.append(file.name)
        #cat pos_err vel_err and torques
        # q_des_list=[]
        # q_cur_list=[]
        # q_dot_cur_list=[]
        # torques_list=[]
        #将所有的数据拼接为1维度，并保存在flat_raw_data.pt中
        # for file in train_files:
        #     motor_raw_data=torch.load(data_path+train_files[0],weights_only=True)
        #     q_des_list.append(torch.cat(motor_raw_data['q_des'],dim=0)[:,0:3].reshape(-1).to(self.device))
        #     q_cur_list.append(torch.cat(motor_raw_data['q_cur'],dim=0).reshape(-1).to(self.device))
        #     q_dot_cur_list.append(torch.cat(motor_raw_data['q_dot_cur'],dim=0).reshape(-1).to(self.device))
        #     torques_list.append(torch.cat(motor_raw_data['torq_cur'],dim=0).reshape(-1).to(self.device))
        # q_des=torch.cat(q_des_list,dim=0)
        # q_cur=torch.cat(q_cur_list,dim=0)
        # q_dot_cur=torch.cat(q_dot_cur_list,dim=0)
        # torques=torch.cat(torques_list,dim=0)
        # flat_data_dict={'q_des':q_des,'q_cur':q_cur,'q_dot_cur':q_dot_cur,'torq_cur':torques}
        # torch.save(flat_data_dict,f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_raw_data/motor_raw_data.pt")
        #按照腿的三个电机，进行拼接，并保存在leg_raw_data.pt中
        # for file in train_files:
        #     print(file)
        #     motor_raw_data=torch.load(data_path+file,weights_only=True)
        #     q_des_list.append(torch.cat(motor_raw_data['q_des'],dim=0)[:,0:3].to(self.device))
        #     q_cur_list.append(torch.cat(motor_raw_data['q_cur'],dim=0).to(self.device))
        #     q_dot_cur_list.append(torch.cat(motor_raw_data['q_dot_cur'],dim=0).to(self.device))
        #     torques_list.append(torch.cat(motor_raw_data['torq_cur'],dim=0).to(self.device))
        #     print(torch.cat(motor_raw_data['torq_cur'],dim=0).shape)
        # q_des=torch.cat(q_des_list,dim=0)
        # q_cur=torch.cat(q_cur_list,dim=0)
        # q_dot_cur=torch.cat(q_dot_cur_list,dim=0)
        # torques=torch.cat(torques_list,dim=0)
        # flat_data_dict={'q_des':q_des,'q_cur':q_cur,'q_dot_cur':q_dot_cur,'torq_cur':torques}
        # torch.save(flat_data_dict,f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_raw_data/train_dataset/leg_raw_data.pt") 

        #遍历所有源数据文件，将数据进行拼接
        data_path=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_raw_data/test_dataset_4_30/"
        selected_files=[]
        for file in Path(data_path).iterdir():
            if file_common_name in file.name:
                selected_files.append(file.name)
                print(file.name)
        dict_data={'q_des':[],'q_cur':[],'q_dot_cur':[],'torq_cur':[]}
        for files in selected_files:
            raw_data=torch.load(data_path+files,weights_only=True)
            q_des=torch.cat(raw_data['q_des'],dim=0)[:,0:3]
            q_cur=torch.cat(raw_data['q_cur'],dim=0)
            q_dot_cur=torch.cat(raw_data['q_dot_cur'],dim=0)
            torques=torch.cat(raw_data['torq_cur'],dim=0)  
            dict_data['q_des'].append(q_des)
            dict_data['q_cur'].append(q_cur)
            dict_data['q_dot_cur'].append(q_dot_cur)
            dict_data['torq_cur'].append(torques)
        q_des=torch.cat(dict_data['q_des'],dim=0)
        q_cur=torch.cat(dict_data['q_cur'],dim=0)
        q_dot_cur=torch.cat(dict_data['q_dot_cur'],dim=0)
        torques=torch.cat(dict_data['torq_cur'],dim=0)  

        pos_err=(q_des-q_cur).to(self.device)
        vel_err=(-q_dot_cur).to(self.device)
        torques=torques.to(self.device) 
        return pos_err,vel_err, torques         

    def _learn_motor_net(self):
        #处理数据
        #单电机处理
        # raw_data=torch.load(f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_raw_data/motor_raw_data.pt",weights_only=True)
        # # q_des=raw_data['q_des'][:-1]
        # # q_cur=raw_data['q_cur'][:-1]
        # # q_dot_cur=raw_data['q_dot_cur'][:-1]
        # # torques=raw_data['torq_cur'][1:]
        # q_des=raw_data['q_des']
        # q_cur=raw_data['q_cur']
        # q_dot_cur=raw_data['q_dot_cur']
        # torques=raw_data['torq_cur'].unsqueeze(1)    
        # pos_err=q_des-q_cur
        # vel_err=-q_dot_cur
        # # #包含历史四次的数据
        # pos_err_unfold=pos_err.unfold(0,3,1)
        # vel_err_unfold=vel_err.unfold(0,3,1)
        # err=torch.cat([pos_err_unfold,vel_err_unfold],dim=1)
        # torques=torques[2:]
        # # #只包含当前数据
        # # err=torch.cat([pos_err.unsqueeze(1),vel_err.unsqueeze(1)],dim=1) #nums*2

        #一条腿整体处理
        # raw_data=torch.load(f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_raw_data/train_dataset/leg_raw_data.pt",weights_only=True)
        # q_des=raw_data['q_des']
        # q_cur=raw_data['q_cur']
        # q_dot_cur=raw_data['q_dot_cur']
        # torques=raw_data['torq_cur']      
        # pos_err=q_des-q_cur
        # vel_err=-q_dot_cur
        #只包含当前信息
        # err=torch.cat([pos_err,vel_err],dim=1)
        #包含历史三次数据信息
        # pos_err=torch.cat([pos_err[:-2],pos_err[1:-1],pos_err[2:]],dim=1)
        # vel_err=torch.cat([vel_err[:-2],vel_err[1:-1],vel_err[2:]],dim=1)
        # err=torch.cat([pos_err,vel_err],dim=1)
        # torques=torques[2:]
        # print(err.shape)
        # print(torques.shape)

        # pos_err,vel_err,torques=self._process_raw_data("sin_curve_")
        # pos_err,vel_err,torques=self._process_raw_data("motor_raw_2025")
        # bag_path = f"{LEGGED_GYM_ROOT_DIR}/resources/motor_raw_data"

        # bag_path = "/home/art/valerian_ws/bag"
        # pos_err,vel_err,torques=process_bag_data(bag_path,0,self.device)
        # err=torch.cat([pos_err,vel_err],dim=1)    

        data = torch.load('/home/val/BIH_ws/bag/motor_data/dataset.pt')  
        err = data['err'].to(self.device)
        torques=data['torque'].to(self.device).unsqueeze(1)

        _data_set=TensorDataset(err,torques)
        data_loader=DataLoader(_data_set,batch_size=128,shuffle=True)
        optimizer=torch.optim.Adam(self.motor_net.parameters(),lr=0.001)
        loss_fn=nn.MSELoss()
        num_epochs=150

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                outputs = self.motor_net(inputs)
                loss = loss_fn(outputs,targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"avg_loss={running_loss/len(data_loader):.4f}")
        torch.save(self.state_dict(),f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1217.pth")
        torch.save(self.motor_net.state_dict(),f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1217_motor.pth")


        # self.optimizers = [
        #     torch.optim.Adam(self.thigh_net.parameters(), lr=1e-3),
        #     torch.optim.Adam(self.knee_net.parameters(), lr=1e-3),
        #     torch.optim.Adam(self.ankle_net.parameters(), lr=1e-3),
        # ]
        # print("Begin learning motor net")
        # for epoch in range(num_epochs):
        #     running_loss = [0.0]*3
        #     for inputs, targets in data_loader:
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         for i, (net, optim) in enumerate(zip(self.joints_net, self.optimizers)):
        #             optim.zero_grad()
        #             outputs = net(inputs[:, [i,i+3]])
        #             loss = loss_fn(outputs, targets[:, i].unsqueeze(1))
        #             loss.backward()
        #             optim.step()
        #             running_loss[i] += loss.item()
        #     print(f"Epoch {epoch+1}/{num_epochs}")
        #     for i in range(3):
        #          print(f"avg_loss={running_loss[i]/(len(data_loader)):.4f}")      
        # torch.save({
        #     "thigh": self.thigh_net.state_dict(),
        #     "knee": self.knee_net.state_dict(),
        #     "ankle": self.ankle_net.state_dict()
        # }, f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1020_1.pth")                   
        # # torch.save(self.state_dict(),f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_0929.pth")
        # print(f"Save model to {LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1020_1.pth")


    def get_torques(self,pos_err:torch.Tensor,vel_err:torch.Tensor):
        #pos_err: env_nums*dof_nums
        if self.cfg.control.use_actuator_net:
            with torch.no_grad():
                motor_pos_err = pos_err[:,self.motor_index].reshape(-1,1)
                motor_vel_err = vel_err[:,self.motor_index].reshape(-1,1)
                motor_torques=torch.zeros_like(motor_pos_err)
                servo_pos_err = pos_err[:,self.servo_index]
                servo_vel_err = vel_err[:,self.servo_index]
                passive_pos_err = pos_err[:,self.passive_index]
                passive_vel_err = vel_err[:,self.passive_index]
                motor_torques=self.motor_net(torch.cat([motor_pos_err,motor_vel_err],dim=1)).reshape(-1,len(self.motor_index)) #type: torch.Tensor
                # err=torch.cat([motor_pos_err,motor_vel_err],dim=1)
                # for i in range(3):
                #     # print(f"shape{self.joints_net[i](err[:,[i,i+3]]).shape}")
                #     #增加不确定性
                #     # motor_torques[:,i]=self.joints_net[i](err[:,[i,i+3]]).squeeze(1) + self.motor_torque_err[i]*(0.5-torch.rand_like(motor_torques[:,i]))
                #     #无不确定性
                #     # motor_torques[:,i]=self.joints_net[i](err[:,[i,i+3]]).squeeze(1)
                #     #对knee和ankle，都用knee关节
                #     if i==0:
                #         motor_torques[:,i]=self.joints_net[i](err[:,[i,i+3]]).squeeze(1)
                #     else:
                #         motor_torques[:,i]=self.joints_net[1](err[:,[i,i+3]]).squeeze(1)
                # motor_torques = motor_torques.reshape(-1,len(self.motor_index))

                servo_torques=self.kp_kd_tensor[0,self.servo_index]*servo_pos_err+self.kp_kd_tensor[1,self.servo_index]*servo_vel_err #type: torch.Tensor
                passive_torques=self.kp_kd_tensor[0,self.passive_index]*passive_pos_err+self.kp_kd_tensor[1,self.passive_index]*passive_vel_err #type: torch.Tensor
                torques=torch.zeros_like(pos_err)
                torques[:,self.motor_index]=motor_torques #+ (torch.rand_like(motor_torques)-0.5)*10.0 #增加±0.5N力矩的不确定性
                torques[:,self.servo_index]=servo_torques
                torques[:,self.passive_index]=passive_torques
        else:
            torques=self.kp_kd_tensor[0,:]*pos_err+self.kp_kd_tensor[1,:]*vel_err #+ (torch.rand_like(pos_err)-0.5)

        return torques

    
    def _test(self):
        self._load_motor_net()
        #load test data
        # pos_err,vel_err,real_torques=self._process_raw_data("motor_raw_2025")

        bag_path = f"{LEGGED_GYM_ROOT_DIR}/resources/motor_raw_data"
        pos_err,vel_err,real_torques=process_bag_data(bag_path,0,self.device)

        #只包含当前信息
        err=torch.cat([pos_err,vel_err],dim=1)

        # ideal_torques=pos_err*self.cfg.control.motor_kp+vel_err*self.cfg.control.motor_kd#nums*1
        # input=torch.cat([pos_err,vel_err],dim=1) #nums*2
        # target=real_torques-ideal_torques #nums*1
        _data_set=TensorDataset(err,real_torques)
        data_loader=DataLoader(_data_set,batch_size=32,shuffle=True,num_workers=0)      
        loss_fn=torch.nn.MSELoss() 
        total_loss=0.0
        abs_ave_loss=0.0
        for inputs, targets in data_loader:
            outputs=self.motor_net(inputs)
            abs_ave_loss+=(targets-outputs).abs().sum()/(32.0*3)

            loss=loss_fn(outputs,targets)
            total_loss+=loss.item()
        print(f"Test loss:{total_loss/len(data_loader)}")
        print(f"Test abs ave loss:{abs_ave_loss/len(data_loader)}")


def plot_raw_data(txt_path:str):
    

        pos_err=torch.arange(-1.5,1.5,0.02,device=device)
        vel_err=torch.arange(-7,7,0.02,device=device)
        P, V = torch.meshgrid(pos_err, vel_err, indexing='xy')
        grid = torch.stack([P.ravel(),V.ravel()],dim=1)
        print(grid.shape)
        # err=torch.cartesian_prod(pos_err,vel_err)
        # print(err.shape)
        motor._load_motor_net()
        with torch.no_grad():
            pridected_torques = motor.motor_net(grid)
            pridected_torques = pridected_torques.view(V.shape)
            # pridected_torques=motor.motor_net(err)
            # pridected_torques=(err[:,0]*100.0-err[:,1]).reshape(-1).clamp(min=-27,max=27)
        x=P.cpu().numpy()
        y=V.cpu().numpy()
        z=pridected_torques.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x,y,z,cmap='rainbow')

        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label='torques(Nm)')

        pos_err=[]
        vel_err=[]
        torque=[]
        line_pat = re.compile(
            r'POS:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s*'
            r'VEL:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s*'
            r'TORQUE:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)'
        )
        for fname in os.listdir(txt_path):
            if fname.endswith('.txt'):
                with open(os.path.join(txt_path,fname),'r',encoding='utf-8') as f:
                    for line in f:
                        m = line_pat.search(line)
                        if m:
                            pos_err.append(-float(m.group(1))) #记录txt时，pos_cur-pos_des，需要反
                            vel_err.append(-float(m.group(2))) #这是速度真值，需要计算误差0-vel_cur
                            torque.append(float(m.group(3))+0.435) #记录时无论正负都减去了这个值
        ax.scatter(pos_err[::20],vel_err[::20],torque[::20])





        # 设置坐标轴标签
        ax.set_xlabel('pos_err (m) Axis')
        ax.set_ylabel('vel_err (m/s) Axis')
        ax.set_zlabel('ideal_torques (Nm) Axis')
        # 显示图形
        plt.show()

def process_bag_data(bag_path:str,index:int,device):
    bag_names=list(Path(bag_path).rglob('*.bag'))
    print(bag_names)
    leg_names=['RF','RM','RB','LF','LM','LB']
    topics=[] #RF sita_des RF sita_cur
    for leg_name in leg_names:
        topics.append(['/'+leg_name+'/sita_des','/'+leg_name+'/sita_cur'])

    
    sita_des=[]
    sita_cur=[]
    dot_sita_cur=[]
    torque_cur=[]
    t_cur_list=[]
    t_des_list=[]
    for bag_name in bag_names:
        with rosbag.Bag(bag_name,'r') as bag:
            for pairs in topics:
                for topic, msg, t in bag.read_messages(pairs):
                    msg=list(msg.data)
                    # print(topic)
                    if 'des' in topic:
                        sita_des.append(msg[1:4])
                        t_des_list.append(t.to_sec())
                    elif 'cur' in topic:
                        sita_cur.append(msg[0:3])
                        torque_cur.append(msg[3:6])
                        dot_sita_cur.append(msg[6:9])
                        t_cur_list.append(t.to_sec())



    pos_err = (torch.tensor(sita_des[:40500]))-(torch.tensor(sita_cur[:40500]))
    vel_err = -torch.tensor(dot_sita_cur[:40500])
    torques = torch.tensor(torque_cur[:40500])

    
    return pos_err.to(device), vel_err.to(device), torques.to(device)

def process_txt_data(txt_path:str,device):
    pos_err=[]
    vel_err=[]
    torque=[]
    line_pat = re.compile(
        r'POS:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s*'
        r'VEL:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?),\s*'
        r'TORQUE:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)'
    )
    for fname in os.listdir(txt_path):
        if fname.endswith('.txt'):
            with open(os.path.join(txt_path,fname),'r',encoding='utf-8') as f:
                for line in f:
                    m = line_pat.search(line)
                    if m:
                        pos_err.append(-float(m.group(1))) #记录txt时，pos_cur-pos_des，需要反
                        vel_err.append(-float(m.group(2))) #这是速度真值，需要计算误差0-vel_cur
                        torque.append(float(m.group(3))+0.435) #记录时无论正负都减去了这个值
    return torch.tensor(pos_err).to(device), torch.tensor(vel_err).to(device),torch.tensor(torque).to(device)





if __name__=="__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
        
    device='cuda'
    cfg=HexGroundCfg()

    cfg.control.use_actuator_net=False
    motor=Actuator(cfg,device)

    motor._learn_motor_net()

# t=12.347327947616577, LB des=
# t=12.350584745407104, LB cur=, torque=-6.023931503295898, vec=-
    # err=torch.tensor([0.34907978773117065-0.4869535267353058, 4.057631492614746],device='cuda')
    # print(motor.motor_net(err))

    # print(pos_err.max())
    # print(pos_err.min())
    # print(vel_err.max())
    # print(vel_err.min())
    # print(torch.argmin(pos_err))
    # print(f"value={value},index={index}")
    # print(torch.min(pos_err))


    # pos_err = torch.ones(3,18,device=device)
    # vel_err = torch.ones(3,18,device=device)
    # print(torques)
    # pos_err = torch.zeros(1,3,device=device)
    # vel_err = torch.zeros(1,3,device=device)
    # pos_err [0,1]=-0.6
    # torques=motor.motor_net(torch.cat([pos_err,vel_err],dim=1))
    # print("torques\n",torques)


    # motor._process_raw_data()
    # motor._learn_motor_net()
    # plot_raw_data("/home/val/BIH_ws/motor_data")

    
    # pos_err,vel_err,torque = process_txt_data("/home/val/BIH_ws/motor_data",device)
    # print(f"pos_err.shape={pos_err.shape}, vel_err.shape={vel_err.shape}, torque.shape={torque.shape}")

    # motor._test()
    # pos_err=torch.ones(2,42,device=device)*0.01
    # vel_err=torch.ones(2,42,device=device)
    # torques=motor.get_torques(pos_err,vel_err)
    # print(torques)


    # bag_path = f"{LEGGED_GYM_ROOT_DIR}/resources/motor_raw_data"
    # process_bag_data(bag_path)

    
