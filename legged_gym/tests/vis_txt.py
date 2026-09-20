import pyvista as pv
import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg
# from legged_gym.utils.actuator import Actuator
import torch
files=[]
for file in os.listdir('/home/val/BIH_ws/bag/motor_data/'):
    if file.endswith('.txt'):
        files.append('/home/val/BIH_ws/bag/motor_data/'+file)


line_pat = re.compile(
    r'des=\[(?P<des>[+\-\d\s.,eE]+)\];'   # 捕获 des 段
    r'cur=\[(?P<cur>[+\-\d\s.,eE]+)\]'    # 捕获 cur 段
)
            
des_list, cur_list = [], []

for file in files:
    with open(file,'r') as f:
        for line in f:
            m=line_pat.search(line)
            if not m:
                continue
            des_data=ast.literal_eval('[' + m['des'] + ']')
            cur_data=ast.literal_eval('[' + m['cur'] + ']')
            #这是观察电机数据打印出来的异常值，一直保持这个值不变，因此把这个值从正常数据中排除
            if 2.5504274368286133 in cur_data or -2.5504274368286133 in cur_data:
                continue
            des_list.append(des_data)
            cur_list.append(cur_data)
            

des_arr:np.ndarray= np.array(des_list)
cur_arr:np.ndarray =np.array(cur_list)
print(f"des_arr.shape={des_arr.shape}, cur_arr.shape={cur_arr.shape}")

#所有电机的数据放在一起处理
des_arr=des_arr.flatten()
cur_q_arr=np.hstack([cur_arr[:,0:3],cur_arr[:,9:12],cur_arr[:,18:21]]).flatten()
cur_torq_arr=np.hstack([cur_arr[:,3:6],cur_arr[:,12:15],cur_arr[:,21:24]]).flatten()
cur_v_arr=np.hstack([cur_arr[:,6:9],cur_arr[:,15:18],cur_arr[:,24:27]]).flatten()
err = torch.tensor(np.vstack([des_arr-cur_q_arr,-cur_v_arr]).T,dtype=torch.float)
cur_torque = torch.tensor(cur_torq_arr,dtype=torch.float)
dataset_dict={'err':err,'torque':cur_torque}
# torch.save('/home/val/BIH_ws/bag/motor_data')
torch.save(dataset_dict,'/home/val/BIH_ws/bag/motor_data/dataset.pt')


data = torch.load('/home/val/BIH_ws/bag/motor_data/dataset.pt',weights_only=True)  
print(data['err'].shape)
print(data['torque'].shape)

# motor_index=0 # thihg knee ankle 
# des_arr=des_arr[:,[0+motor_index,3+motor_index,6+motor_index]].flatten()
# cur_q_arr=np.vstack([cur_arr[:,0+motor_index],cur_arr[:,9+motor_index],cur_arr[:,18+motor_index]]).flatten('F')
# cur_torq_arr=np.vstack([cur_arr[:,3+motor_index],cur_arr[:,12+motor_index],cur_arr[:,21+motor_index]]).flatten('F')
# cur_v_arr=np.vstack([cur_arr[:,6+motor_index],cur_arr[:,15+motor_index],cur_arr[:,24+motor_index]]).flatten('F')


# actuator = Actuator(HexGroundCfg(),'cpu')

# pri_torq_torch=actuator.motor_net(err).detach()


fig = plt.figure()
interval=20
ax = fig.add_subplot(111, projection='3d')
ax.scatter((des_arr-cur_q_arr)[::interval],-cur_v_arr[::interval],cur_torq_arr[::interval])
# ax.scatter(err[::interval,0].numpy(),err[::interval,1].numpy(),pri_torq_torch[::interval].numpy())
# print("shape pri_torq_torch={}, cur_torq_arr={}".format(pri_torq_torch.shape,cur_torq_arr.shape))
# print(np.mean(abs(pri_torq_torch.view(-1).numpy()-cur_torq_arr.T)))
plt.show()

















