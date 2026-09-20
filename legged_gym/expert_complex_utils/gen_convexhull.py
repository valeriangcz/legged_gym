from math import *
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from scipy.spatial import ConvexHull
from urdfpy import URDF

def RotZ(angle:float,vertices:np.ndarray):
    #接收行向量
    angle = radians(angle)
    mat = np.array([[cos(angle),-sin(angle),0],
                    [sin(angle),cos(angle),0],
                    [0,0,1]])
    return (mat@(vertices.transpose())).transpose()

def ForwardKin(angl):
    L1=0.072
    L2=0.13
    L3=0.17
    sita1 = radians(angl[0])
    sita2 = radians(angl[1])
    sita3 = radians(angl[2])
    return [L1*cos(sita1)+L2*cos(sita1)*cos(sita2)+L3*cos(sita1)*cos(sita2+sita3),
            L1*sin(sita1)+L2*sin(sita1)*cos(sita2)+L3*sin(sita1)*cos(sita2+sita3),
            L2*sin(sita2)+L3*sin(sita2+sita3)]


def B2R(V:np.ndarray,leg_name:str)->np.ndarray:
    R_V=V.copy()
    bodyshape_x=0.1
    bodyshape_y=0.22
    if 'L' in leg_name:
        bodyshape_x = -bodyshape_x
    if 'B' in leg_name:
        bodyshape_y = -bodyshape_y
    if 'M' in leg_name:
        bodyshape_y = 0.0
    
    if 'L' in leg_name:
        R_V[:,0] = bodyshape_x - V[:,0]
        R_V[:,1] = bodyshape_y - V[:,1]
    if 'R' in leg_name:
        R_V[:,0] = bodyshape_x + V[:,0]
        R_V[:,1] = bodyshape_y + V[:,1]
    return R_V

def AngleSelect(name:str)->np.ndarray:
    #根据腿部关节名称选择旋转的角度
    angles_cornor = np.linspace(-15.0, 90, 4) #这里是RF与LB的，另外两个角落，取负号即可
    angles_middle = np.linspace(-32,32, 3)
    # angles_cornor = np.linspace(-20, 60, 4) #这里是RF与LB的，另外两个角落，取负号即可
    # angles_middle = np.linspace(-20,20, 3)    
    if name == 'LF' or name == 'RB':
        angles = -angles_cornor
    else:
        angles = angles_cornor
        
    if 'M' in name:
        angles = angles_middle
    return angles





#定义工作空间的一个截面
#以下为关键顶点
V_posi = np.array([[0, 0, -0.075],
                   [0.13, 0, -0.075],
                   [0.135, 0, 0],
                   [0.15, 0, 0.2],
                   [0.27, 0, 0.2],
                   [0.3, 0, 0],
                   [0.26, 0, -0.2],
                   [0.13, 0, -0.27],
                   [0, 0, -0.27]])
V_posi = np.vstack([V_posi,V_posi[0]])

V_posi_up = np.array([[0.13, 0,-0.075],
                      [0.135, 0, 0],
                      [0.15, 0, 0.2],
                      [0.27, 0, 0.2],
                      [0.3, 0, 0],
                      [0.26, 0, -0.2],
                      [0.13, 0, -0.27]])
V_posi_up = np.vstack([V_posi_up,V_posi_up[0]])
V_posi_down=np.array([[0, 0, -0.075],
                      [0.13,0, -0.075],
                      [0.13, 0, -0.27],
                      [0, 0, -0.27]])
V_posi_down = np.vstack([V_posi_down,V_posi_down[0]])
V_nega=np.array([[0, 0, -0.075],
                 [0, 0, -0.27],
                 [-0.14, 0, -0.15],
                 [-0.14, 0, -0.075]])
V_nega = np.vstack([V_nega,V_nega[0]])

"""初始化可视化的figure"""
pv.global_theme.multi_rendering_splitting_position = 0.3
plotter = pv.Plotter(shape='3|1',window_size=(1400,1200))

#原始工作空间点云
q1 = [0.0]
q1 = np.linspace(-40,40,20)
q2 = np.linspace(-120,125, 20)
q3 = np.linspace(-155,140,20)
pos=[]

for i in q1:
    for j in q2:
        for k in q3:
            pos.append(ForwardKin([i,j,k]))
pos=np.array(pos) #N*3, 每一行是点坐标

point_cloud = pv.PolyData(pos)

#截面连线
poly_posi = pv.PolyData(V_posi,lines=np.hstack([[len(V_posi)],np.arange((len(V_posi)))]))
poly_posi_up=pv.PolyData(V_posi_up,lines=np.hstack([[len(V_posi_up)],np.arange(len(V_posi_up))]))
poly_posi_down=pv.PolyData(V_posi_down,lines=np.hstack([[len(V_posi_down)],np.arange(len(V_posi_down))]))
poly_nega=pv.PolyData(V_nega,lines=np.hstack([[len(V_nega)],np.arange(len(V_nega))]))

#先可视化以上两部分
plotter.subplot(0)
plotter.show_grid()
plotter.add_text("point cloud and profile")
plotter.add_mesh(point_cloud,point_size=8,render_points_as_spheres=True,color='#0A51ECFF',opacity=0.2)
plotter.add_mesh(poly_posi,line_width=10,color='#294942')
plotter.add_mesh(poly_posi_up,line_width=2,color='#C90F0FFF')
plotter.add_mesh(poly_posi_down,line_width=2,color='#C90F0FFF')
plotter.add_mesh(poly_nega,line_width=2,color='#C90F0FFF')
#选择两个腿部的工作空间点云可凸包进行可视化
leg_choose=["RF","RM"]
for  i,name in enumerate(leg_choose):
    plotter.subplot(i+1)
    plotter.show_grid()
    plotter.add_text(f"working range of {name} leg")
    angles = AngleSelect(name)
    Vs_posi_down=[]
    Vs_posi_up_list=[]
    Vs_nega=[]
    for angle in angles:
        Vs_posi_down.append(RotZ(angle,V_posi_down))
        Vs_nega.append(RotZ(angle,V_nega))
    Vs_posi_down = np.vstack(Vs_posi_down)
    Vs_nega = np.vstack(Vs_nega)
    for i in range(len(angles)-1):
        Vs = np.vstack([RotZ(angles[i],V_posi_up),RotZ(angles[i+1],V_posi_up)])
        Vs_posi_up_list.append( np.unique(Vs,axis=0) )
    convexhull_posi_down = ConvexHull(Vs_posi_down)
    convexhull_nega = ConvexHull(Vs_nega)
    mesh1=pv.PolyData(Vs_posi_down,np.hstack([[3,*f] for f in convexhull_posi_down.simplices]))
    mesh2=pv.PolyData(Vs_nega, np.hstack([[3,*f] for f in convexhull_nega.simplices]))
    plotter.add_mesh(mesh1,opacity=0.2,color='#5881F0FF')
    plotter.add_mesh(mesh2,opacity=0.2,color='#5881F0FF')
    for Vs in Vs_posi_up_list:
        convexhull = ConvexHull(Vs)
        mesh = pv.PolyData(Vs,np.hstack([[3,*f] for f in convexhull.simplices]))
        plotter.add_mesh(mesh,opacity=0.2,color='#5881F0FF')

#初始化yaml文件

file = open("/home/val/BIH_ws/legged_gym/legged_gym/expert_complex_utils/working_range.yaml", "w")
convexhulls_dict={'leg':{}, 'body':{}}


#为每个腿部创建工作空间凸包，保存凸包的v表达，将顶点放入yaml文件中(在局部坐标系下的表示)
leg_names=["RF","RM","RB","LF","LM","LB"]
#每条腿的工作空间是通过对截面的旋转构成的
#V_posi_down和V_nega都是可以把所有旋转后的截面直接拼接构成凸包
#V_posi_up需要把旋转后的截面两两拼接才能构成凸包
max_norm = 0.0
bound_vertices = None

plotter.subplot(3)
for name in leg_names:
    #根据位置选择旋转的角度
    angles = AngleSelect(name)
    #对V_posi_down和V_nega进行处理
    Vs_posi=[]
    Vs_posi_down=[]
    Vs_nega=[]
    for angle in angles:
        Vs_posi.append(RotZ(angle,V_posi))
        Vs_posi_down.append(RotZ(angle,V_posi_down))
        Vs_nega.append(RotZ(angle,V_nega))
    #获取了顶点
    Vs_posi = np.vstack(Vs_posi)
    Vs_posi_down = np.vstack(Vs_posi_down)
    Vs_nega = np.vstack(Vs_nega)
    #删除重复点， 将所有顶点转到全局坐标系下
    Vs_posi = B2R(np.unique(Vs_posi,axis=0),name)
    Vs_posi_down = B2R(np.unique(Vs_posi_down,axis=0),name)
    Vs_nega = B2R(np.unique(Vs_nega,axis=0),name)
    
    #找到所有顶点中距离原点norm最大的
    curr_norm = np.max(np.linalg.norm(Vs_posi,axis=1))
    max_norm = curr_norm if curr_norm>max_norm else max_norm

    Vs_posi_up_list=[]
    for i in range(len(angles)-1):
        Vs = np.vstack([RotZ(angles[i],V_posi_up),RotZ(angles[i+1],V_posi_up)])
        Vs_posi_up_list.append( B2R(np.unique(Vs,axis=0),name) )
    #Vs_posi部分采用一个凸包表示


    
    #保存顶点
    convexhulls_dict['leg'][name]={}
    convexhulls_dict['leg'][name]['Vs_nega']=Vs_nega.tolist()
    convexhulls_dict['leg'][name]['Vs_posi']=Vs_posi.tolist()

    # convexhulls_dict['leg'][name]['Vs_posi_down']=Vs_posi_down.tolist()
    # for i,Vs in enumerate(Vs_posi_up_list):
    #     convexhulls_dict['leg'][name]['Vs_posi_up_'+str(i)]=Vs_posi_up_list[i].tolist()

    #可视化所有凸包的点
    convexhull_posi_down = ConvexHull(Vs_posi_down)
    convexhull_nega = ConvexHull(Vs_nega)
    convexhull_posi = ConvexHull(Vs_posi)    
    mesh1=pv.PolyData(Vs_posi,np.hstack([[3,*f] for f in convexhull_posi.simplices]))
    # mesh1=pv.PolyData(Vs_posi_down,np.hstack([[3,*f] for f in convexhull_posi_down.simplices]))
    mesh2=pv.PolyData(Vs_nega, np.hstack([[3,*f] for f in convexhull_nega.simplices]))
    plotter.add_mesh(mesh1,opacity=0.2,color='#5881F0FF')
    plotter.add_mesh(mesh2,opacity=0.2,color='#5881F0FF')
    # for Vs in Vs_posi_up_list:
    #     convexhull = ConvexHull(Vs)
    #     mesh = pv.PolyData(Vs,np.hstack([[3,*f] for f in convexhull.simplices]))
    #     plotter.add_mesh(mesh,opacity=0.2,color='#5881F0FF')
print(f"Find most far points with norm={max_norm}")


#为身体创建凸包，保存凸包v表达，将顶点放入yaml文件中（全局坐标系下表示）
#每个腿的位置选取部分点作为凸包的点
V_body=np.array([[0.13,0,-0.075],[0.135,0,0],[0.15,0,0.2]])
Vs_body=[]
for name in leg_names:
    #根据位置选择旋转的角度
    angles = AngleSelect(name)
    for angle in angles:
        Vs_body.append(B2R(RotZ(angle,V_body),name))
Vs_body = np.vstack(Vs_body)
convexhull_body=ConvexHull(Vs_body)
#保存顶点
convexhulls_dict['body']={}
convexhulls_dict['body']['poly_0']=convexhull_body.points[convexhull_body.vertices].tolist()

#进行可视化操作
mesh3 = pv.PolyData(Vs_body,np.hstack([[3,*f] for f in convexhull_body.simplices]))
plotter.add_mesh(mesh3,color='#C211E685',opacity=0.3)
point_body=pv.PolyData(convexhull_body.points[convexhull_body.vertices])
plotter.add_mesh(point_body,render_points_as_spheres=True,point_size=5,color='#E41173FF')
#绘制散点
points = np.meshgrid(np.arange(-0.4,0.4,0.04),np.arange(-0.5,0.5,0.04),indexing='ij')
print(points[0].ravel().shape)
points = np.vstack([points[0].ravel(),points[1].ravel(),np.zeros_like(points[0]).ravel()]).transpose()
pv.PolyData(points)
plotter.add_mesh(pv.PolyData(points),render_points_as_spheres=True,point_size=5)

#加载机器人urdf文件进行可视化操作
urdf_dir = "/home/val/BIH_ws/legged_gym/resources/robots/hex_v4/urdf"
robot = URDF.load(urdf_dir+"/hex_ground_STL.urdf")

# robot = URDF.load("/home/val/BIH_ws/hex_gym/urdf/hex_ground.urdf")
joint_cfg={}
for joint in robot.joints:
    angle=0.0
    if "thigh" in joint.name:
        if "rf" in joint.name or "lb" in joint.name:
            angle=0.5
        elif "lf" in joint.name or "rb" in joint.name:
            angle=-0.5
    elif "knee" in joint.name:
        angle=0.6
    elif "ankle" in joint.name:
        angle=-2.1
    joint_cfg[joint.name]=angle
vis_fk = robot.visual_geometry_fk(cfg=joint_cfg)
for link in robot.links:
    for visual in link.visuals:
        if visual.geometry.mesh is not None:
            geom = visual.geometry
            mesh_path = Path(visual.geometry.mesh.filename)
            if not mesh_path.is_absolute():
                # URDF mesh paths are relative to the URDF file location
                mesh_path = (urdf_dir/ mesh_path).resolve()

            pv_mesh=pv.read(str(mesh_path))
            pose=vis_fk.get(geom)
            points = pv_mesh.points
            ones = np.ones((points.shape[0],1))
            homo = np.hstack(([points,ones]))
            # pv_mesh.points = ((pose @ homo.T).T)[:,:3]
            pv_mesh.points = ((homo @ pose.T))[:,:3]
            plotter.add_mesh(pv_mesh)

yaml.dump(convexhulls_dict,file,default_flow_style=None)
file.close()
plotter.show_grid()
plotter.show()


import yaml
from scipy.spatial import ConvexHull as CH
from spatialmath import SE3
from typing import List, Union, Tuple
import warnings

class RobotConvexHulls:
    def __init__(self, file_path=None, body_x_offset=0.1, body_y_offset=0.22,
                 singularity_threshold = 0.02):
        """阅读file文件，使用顶点构造凸包   
            body_x和body_y是身体坐标系下腿部基坐标的位置距离
            singularity_threshold是判断腿部进入r_xy=0的奇异点阈值  
        """
        self.singularity_threshold = singularity_threshold
        # self._leg_names=["RF","RM","RB","LF","LM","LB"]
        self._leg_names=["LB","LF","LM","RB","RF","RM"]
        if file_path is None:
            file_path = "./working_range.yaml"
        with open(file_path, "r") as file:
            vertices_dict = yaml.safe_load(file)#type:dict
        #身体凸包与系数构建，凸包H表达ax+by+cz+d<=0
        self.body_v=CH(vertices_dict['body']['poly_0'])
        self.body_coeff_abcd = np.unique(self.body_v.equations,axis=0)

        self.leg_base_p = np.zeros((3,6),dtype=np.float32)
        self.leg_coeff_abcd_list: List[List[np.ndarray]] = []

        for name in self._leg_names:
            leg_coeffs: List[np.ndarray] = []
            for _,vertices in vertices_dict['leg'][name].items():
                ch = CH(vertices)
                leg_coeffs.append(np.unique(ch.equations, axis=0))
            self.leg_coeff_abcd_list.append(leg_coeffs)
        
        
        for i,name in enumerate(self._leg_names):
            #保存腿部基坐标原点在身体坐标系下的位置
            bx = body_x_offset
            by = 0.0
            if ('F' in name) or ('B' in name):
                by = body_y_offset
            if 'L' in name:
                bx = -bx
            if 'B' in name:
                by = -by
            self.leg_base_p[:,i]=[bx,by,0.0]
        # print(self.leg_base_p)

        self._eps = 1e-9
        # print(self.body_coeff_abcd.shape)

    def PointsInWhichLegs(self,points:np.ndarray, norms:np.ndarray)->Tuple[np.ndarray,bool]:
        """
        @description: 
        输入机器人R坐标系下列向量的点和身体位姿，返回该身体配置是否可行，以及可行点的mask
        遗留问题：如果一个点同时在多个腿部工作空间内，目前算法会覆盖掉

        @input:
        points: (3,N) N个点，在R系下表示
        norms: (3,N) N个点，在R系下的方向向量，都是从实体指向外侧的方向

        @output
        point_inleg_index: 不可行为-1，可行为在哪个腿部的索引
        config_feasiable : 机器人当前配置是否可行

        """
        #先计算点在身体坐标系下的表示
        # uniform_points = np.vstack( [points,np.ones(points.shape[1])] )
        # R_points = np.linalg.solve(Transform,uniform_points)[0:3,:] #3*N
        #计算凸包矩阵乘法
        # R_points = Transform.inv() * points
        # R_norms = Transform.inv() * norms
        body_abc = self.body_coeff_abcd[:,0:3]
        body_d = self.body_coeff_abcd[:,3]
        res_body = body_abc@points + body_d[:,None] #M_body*N
        #计算每个腿对应的每个凸包内的点的数量
        mask_points_collide = (res_body <= self._eps).all(axis=0) #N
        # mask_points_feaible = np.zeros(points.shape[1],dtype=bool)
        point_inleg_index = np.ones(points.shape[1],dtype=np.int16)*-1
        mask_leg_feasible = np.zeros(6,dtype=bool)
        for leg_idx in range(6):
            for coeff_abcd in self.leg_coeff_abcd_list[leg_idx]:
                #一个点在腿部空间内可行包括两个条件：点在凸包内部，点与腿部基坐标原点连线为l，l与z轴构成的平面为p，p与该点的方向向量夹角在15°以内
                #条件1判断
                leg_abc = coeff_abcd[:,0:3]
                leg_d = coeff_abcd[:,3]
                res_leg = leg_abc@points + leg_d[:,None]
                inside_mask = (res_leg <= self._eps).all(axis=0)
                # if self._leg_names[id-1] == "RM":
                #     print(res[convexhull_mask&leg_part_mask])
                #     print(inside_mask)
                #条件2判断
                #先拿到indise_mask的点，根据id拿到腿部基坐标
                if inside_mask.any():
                    # print(f"points inside leg {self._leg_names[id-1]} inside convex {convex_id}")
                    inside_point = points[:,inside_mask] #假设7个可行 3*7
                    points_norm = norms[:,inside_mask] # 取法向量
                    # print("inside point=",inside_point)
                    # print("points norm=",points_norm)
                    #pn 代表平面法向量：平面是当前点 腿部基坐标原点，z轴上一点组成的
                    pn_x = self.leg_base_p[1,leg_idx]-inside_point[1,:] # by - y
                    pn_y = inside_point[0,:]-self.leg_base_p[0,leg_idx] # x -bx
                    pn = np.vstack([pn_x,pn_y,np.zeros(len(pn_x))]) #3*7
                    # pn_norm = np.linalg.norm(pn,axis=0)
                    pn = pn / np.maximum(np.linalg.norm(pn,axis=0), self._eps)
                    # norm_norm = np.linalg.norm(points_norm, axis=0)
                    points_norm = points_norm / np.maximum(np.linalg.norm(points_norm,axis=0), self._eps)
                    # print("plane norm =\n",pn)
                    #共7个平面法向量，计算这7个平面法向量与对应点的法向量的内积 3*7 3*7
                    dot_res = np.sum(pn*points_norm,axis=0)
                    # print("dot res=",dot_res)
                    norm_feasi = ( dot_res<np.cos(np.deg2rad(75.0)) ) & ( dot_res>-np.cos(np.deg2rad(75.0)) )
                    # print("inside mask=",inside_mask)
                    # print("norm feasi=",norm_feasi)
                    inside_mask[inside_mask]=norm_feasi
                    #标记改点在哪个腿部凸包
                    #条件3融合，不在身体凸包内部
                    inside_mask = inside_mask & (~mask_points_collide)
                    point_inleg_index[inside_mask]=leg_idx

                #标记所有在腿部凸包内的点
                # mask_points_feaible = inside_mask | mask_points_feaible
                #只有在同一个凸包内超过六个点了，才算可行
                if np.sum(inside_mask) >= 6:
                    mask_leg_feasible[leg_idx]=True


        #因为腿部凸包和身体凸包有重叠的部分，引起需要把在身体凸包中的点排除掉
        # mask_points_feaible = mask_points_feaible&(~mask_points_collide)
        # 与身体没有碰撞，且所有腿部凸包都与环境接触
        config_feasible = (mask_points_collide.sum()==0) & (mask_leg_feasible.all())
        
        # config_feasible = (mask_points_collide.sum()<=4) & (mask_leg_feasible.sum()>=4)
        return point_inleg_index, config_feasible
            

    def PointsInPoly(self,points:np.ndarray, leg_index:np.ndarray)->bool:
        """
        @description:
        输入点在R系下的坐标与对应的腿的索引，返回每个坐标是否在对应腿部的工作空间内且与身体无碰，最多6个点
        这个函数是为了给构建逆运动学表可行使用，因此不加入法向量部分的判断

        @input：
        points (3,N) 个点，在机器人质心R坐标系下表示
        leg_index (N,) 每个点对应的腿部索引

        @output：
        mask (N,) true 代表在，false代表不在
        """        
        body_abc = self.body_coeff_abcd[:,0:3]
        body_d = self.body_coeff_abcd[:,3]
        res = body_abc@points + body_d[:,None]
        mask = np.zeros_like(leg_index,dtype=bool)
        collide_body_mask = (res<=self._eps).all(axis=0)
        #全部与身体碰撞，直接返回false
        if collide_body_mask.all():
            return mask
        #循环处理
        for i,leg_id in enumerate(leg_index):
            if not collide_body_mask[i]:
                inside = False
                for coeff_abcd in self.leg_coeff_abcd_list[leg_id]:
                    leg_abc = coeff_abcd[:,0:3]
                    leg_d = coeff_abcd[:,3]
                    res = leg_abc@points[:,i]+leg_d
                    #为了避免在奇异点位置过渡过于严苛，也就是只有准确x=0,y=0时才算凸包内，
                    # 对r_xy<0.02的只判断z轴的高度，其中0.02是判断进入奇异点的阈值范围
                    if np.linalg.norm( self._R2B(points[:,i,None],leg_id)[:2] )<=self.singularity_threshold:
                        res[(points[2,i]<=-0.075) & (points[2,i]>=-0.27)] = 0.0
                    if (res<self._eps).all():
                        inside = True
                        break
                mask[i]=inside
        return mask
 
        

    def AllPointsInPoly(self,points:np.ndarray, norms:np.ndarray,leg_index:np.ndarray)->bool:
        """
        @description:
        专门服务于支撑轨迹可行性检查
        输入点坐标，法向量与对应的腿的索引，返回每个坐标是否在对应腿部的工作空间内，最多6个点
        用于stance可行性检查的，一般一个点对应一个腿，只要有一个点超出工作范围，则返回false

        @input：
        points (3,N) 个点，在机器人质心R坐标系下表示
        norms (3,N) 需要标准化
        leg_index (N,) 每个点对应的腿部索引

        @output：
        只要有一个点不在就返回假
        """
        #首先计算是否有在身体内的点，有的话直接返回假
        #这部分有待测试
        body_abc = self.body_coeff_abcd[:,0:3]
        body_d = self.body_coeff_abcd[:,3]
        res = body_abc@points + body_d[:,None]
        if ((res <= self._eps).all(axis=0)).any():
            return False
        #循环处理
        for i,leg_id in enumerate(leg_index):
            inside = False
            for coeff_abcd in self.leg_coeff_abcd_list[leg_id]:
                leg_abc = coeff_abcd[:,0:3]
                leg_d = coeff_abcd[:,3]
                res = leg_abc@points[:,i]+leg_d
                #为了避免在奇异点位置过渡过于严苛，也就是只有准确x=0,y=0时才算凸包内，
                # 对r_xy<0.02的只判断z轴的高度，其中0.02是判断进入奇异点的阈值范围
                if np.linalg.norm( self._R2B(points[:,i,None],leg_id)[:2] )<=self.singularity_threshold:
                    res[(points[2,i]<=-0.075) & (points[2,i]>=-0.27)] = 0.0

                #判断位置
                if (res <= self._eps).all():
                    #判断法向量
                    pn = np.array([self.leg_base_p[1,leg_id]-points[1,i],
                                   points[0,i]-self.leg_base_p[0,leg_id],
                                   0])
                    pn = pn / max(np.linalg.norm(pn), self._eps)
                    norm_i = norms[:,i]
                    norm_i = norm_i / max(np.linalg.norm(norm_i), self._eps)
                    dot_res = pn.dot(norm_i)
                    if dot_res < np.cos(np.deg2rad(75.0)) and dot_res > -np.cos(np.deg2rad(75.0)):
                        inside = True
                    break
            if not inside:
                return False
        return True

    def AllPointsInPolyBatch(self, points: np.ndarray, norms: np.ndarray, leg_index: np.ndarray) -> np.ndarray:
        """
        @description:
        PointInsidePoly 的批量版本。输入 K 组姿态下的点和法向，返回每组是否全部可行。

        @input:
        points: (K,3,N) K 组点，N 个点对应 N 条腿索引
        norms: (K,3,N) 对应点法向
        leg_index: (N,) 每列点对应的腿编号

        @output:
        feasible_mask: (K,) 每组是否可行
        """
        points = np.asarray(points, dtype=np.float64)
        norms = np.asarray(norms, dtype=np.float64)
        leg_index = np.asarray(leg_index, dtype=np.int32).ravel()

        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError("points must have shape (K,3,N)")
        if norms.shape != points.shape:
            raise ValueError("norms must have same shape as points")
        if points.shape[2] != leg_index.size:
            raise ValueError("leg_index size must equal points.shape[2]")

        K = points.shape[0]
        eps = self._eps
        cos75 = np.cos(np.deg2rad(75.0))

        # Any point inside body hull makes this sample infeasible.
        body_abc = self.body_coeff_abcd[:, 0:3]
        body_d = self.body_coeff_abcd[:, 3]
        body_res = np.einsum("mc,kcn->kmn", body_abc, points) + body_d[None, :, None]
        body_inside = (body_res <= eps).all(axis=1).any(axis=1)  # (K,)

        feasible = np.ones(K, dtype=bool)
        feasible[body_inside] = False
        if not feasible.any():
            return feasible

        # For each leg-point column, all K samples must be inside one convexhull and meet normal constraint.
        for col, leg_id in enumerate(leg_index):
            inside_any = np.zeros(K, dtype=bool)
            p_col = points[:, :, col]   # (K,3)
            n_col = norms[:, :, col]    # (K,3)

            for coeff_abcd in self.leg_coeff_abcd_list[int(leg_id)]:
                leg_abc = coeff_abcd[:, 0:3]
                leg_d = coeff_abcd[:, 3]
                leg_res = np.einsum("mc,kc->km", leg_abc, p_col) + leg_d[None, :]
                pos_inside = (leg_res <= eps).all(axis=1)
                if not pos_inside.any():
                    continue

                pn = np.stack(
                    [
                        self.leg_base_p[1, leg_id] - p_col[:, 1],
                        p_col[:, 0] - self.leg_base_p[0, leg_id],
                        np.zeros(K, dtype=np.float64),
                    ],
                    axis=1,
                )
                pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), eps)
                n_col_norm = n_col / np.maximum(np.linalg.norm(n_col, axis=1, keepdims=True), eps)
                dot_res = np.sum(pn * n_col_norm, axis=1)
                norm_ok = (dot_res < cos75) & (dot_res > -cos75)

                inside_any |= pos_inside & norm_ok
                if inside_any.all():
                    break

            feasible &= inside_any
            if not feasible.any():
                break

        return feasible

    def FarestPoints(self,start_points:np.ndarray,heading_norms:np.ndarray,leg_index:int,target_h=0.05)->Tuple[np.ndarray,np.ndarray]:
        """
        @description 计算从起点沿着法向量到达在可行范围内最远（或指定距离）的点,只支持一个凸包内计算
        
        @input start_points: (3,N) R坐标系下 heading_norms(3,N) leg_inex:0~5, target_h为None设定最远，否则根据target设定

        @output farest_point: (3,N) ch_index 0或1 对应凸包的索引
        """
        _leg_coeffs_abcd = self.leg_coeff_abcd_list[leg_index]
    
        #
        norm_lenght = np.linspace(0,0.2,11)
        # possible_points = start_points[:,None] + norm_lenght[None,:]*heading_norms[:,None]
        #(11,N,3)
        possible_points = (start_points[None:,...] + norm_lenght[:,None,None]*heading_norms[None:,...]).transpose(0,2,1)

        ch_index = []
        indexs=[]
        body_dot_res = np.einsum('ijk,mk->ijm',possible_points,self.body_coeff_abcd[:,0:3]) + self.body_coeff_abcd[:,3][None,None,:]
        body_dot_res = (body_dot_res<=0).all(axis=2)
        for coeff_abcd in _leg_coeffs_abcd:
            #(11,N,3) (M,3) -> (11,N,M)
            leg_dot_res = np.einsum('ijk,mk->ijm',possible_points,coeff_abcd[:,0:3]) + coeff_abcd[:,3][None,None,:]
            leg_dot_res = (leg_dot_res<=0).all(axis=2)
            #11,N 
            res = leg_dot_res & (~body_dot_res)
            # N
            farest_index = np.argmin(res,axis=0)
            farest_index = np.clip(farest_index-1,a_min=0,a_max=None)
            farest_index[res.all(axis=0)]=res.shape[0]-1

            indexs.append(farest_index)

        indexs = np.array(indexs)
        pick_mask = indexs[0]>=indexs[1]
        picked_index = np.where(pick_mask,indexs[0],indexs[1])
        ch_index = np.where(pick_mask,0,1)
        
        cols = np.arange(picked_index.size)
        farest_points=possible_points[picked_index,cols].T

        return farest_points, ch_index

    def FarestPoint(self,start_point:np.ndarray,heading_norm:np.ndarray,leg_index:int,target_h=0.2)->Tuple[np.ndarray,int]:
        """
        @description 和FarestPoints功能相同，但这里只计算单个点

        @input start_point: (3,) R系下 heading_norm (3,) leg_index:0~5

        @output farest_point: (3,) ch_index 0或1
        """
        _leg_coeffs_abcd = self.leg_coeff_abcd_list[leg_index]
        #
        segments = int(target_h/0.02)
        norm_lenght = np.linspace(0,target_h,segments)
        #(3,1) + (3,1)*(1,11) = (3,11)
        # possible_points = start_point.reshape(3,1) + heading_norm.reshape(3,1)*norm_lenght.reshape(1,11)
        possible_points = start_point + heading_norm*norm_lenght.reshape(1,segments)
        farest_point = start_point.copy()
        ch_index = -1
        body_dot_res = ( (self.body_coeff_abcd[:,0:3]@possible_points + self.body_coeff_abcd[:,3,None]) <=0 ).all(axis=0)
        for i,coeffs in enumerate(_leg_coeffs_abcd):
            #(M,3) (3,11) = (M,11) = (11)
            leg_dot_res = ( (coeffs[:,0:3]@possible_points + coeffs[:,3,None]) <= 0 ).all(axis=0)
            res = leg_dot_res & (~body_dot_res)
            index = np.nonzero(res)[0]
            if not index.size == 0:
                farest_point=possible_points[:,index[-1],None]
                ch_index = i
                break #一个点只能出现在一个凸包里面
        if index.size==0:
            warnings.warn("Start Point is not inside any convexhull in leg",RuntimeWarning)
        return farest_point, ch_index

    def PushOutBody(self,points:np.ndarray):
        """
        @description 找到位于身体凸包内的点，将其从距离最近的面（除了8 10 11 后上前）推出凸包

        @input points: (3,N) R系下的点

        @output points: (3,N) 推出身体凸包部分后的points
        """
        #先确定身体的那些面要被排除在外，只保留那些与腿部凸包相交的面
        #self.body_coeff_abcd的8，10，11分别代表与腿部凸包不相交的面

        res = (self.body_coeff_abcd[:,0:3] @ points + self.body_coeff_abcd[:,3,None]) #20*N
        inside_mask = (res<=0).all(axis=0)
        if inside_mask.any():
            eps = 0.005
            push_out_index = np.nonzero(inside_mask)[0] #K个
            push_out_points = points[:,push_out_index] #3,K
            selected_faces = np.arange(self.body_coeff_abcd.shape[0])
            selected_faces = selected_faces[~np.isin(selected_faces,[8,10,11])]
            cloest_face_indexs = np.argmin(np.abs(res[selected_faces[:,None],push_out_index]),axis=0) #K个
            face_indexs = selected_faces[cloest_face_indexs]
            distance = np.abs(res[face_indexs,push_out_index])+eps #K
            #3,k += 1,K 3,K
            push_out_points += distance *(self.body_coeff_abcd[face_indexs,0:3].T)
            points[:,push_out_index]=push_out_points
        return points

    def _B2R(self,B_point:np.ndarray,leg_indices:Union[np.ndarray,int])->np.ndarray:
        """输入3*N个腿部基坐标系B下的点,返回3*N个身体坐标系R下的点
        1、leg_indices 是N个对应的腿部索引，共N个腿
        1、leg_indices 只有一个scalar，是所有的点都是这一条腿"""
        R_point = B_point.copy()
        # ---------- 情况 1：scalar ----------
        if np.isscalar(leg_indices):
            leg = int(leg_indices)
            if leg < 3:  # 左腿规则
                R_point[0:2, :] = self.leg_base_p[0:2, leg:leg+1] - B_point[0:2, :]
            else:        # 右腿规则
                R_point[0:2, :] = self.leg_base_p[0:2, leg:leg+1] + B_point[0:2, :]
            return R_point

        # ---------- 情况 2：array ----------
        idx = np.asarray(leg_indices, dtype=int).ravel()
        cols = np.arange(idx.size)
        left_cols = cols[idx<3]
        right_cols = cols[idx>=3]
        if left_cols.size:
            R_point[0:2, left_cols] = self.leg_base_p[0:2, idx[left_cols]] - B_point[0:2, left_cols]
        if right_cols.size:
            R_point[0:2, right_cols] = self.leg_base_p[0:2, idx[right_cols]] + B_point[0:2, right_cols]
        return R_point

    def _R2B(self,R_point:np.ndarray,leg_indices:Union[np.ndarray,int])->np.ndarray:
        """输入3*N个R系下的点，返回3*N个B系下的点
        1、leg_indices 是N个对应的腿部索引，共N个腿
        1、leg_indices 只有一个scalar，是所有的点都是这一条腿"""
        B_point = R_point.copy()
        # ---------- 情况 1：scalar -> broadcast ----------
        if np.isscalar(leg_indices):
            leg = int(leg_indices)
            if leg < 3:  # 左腿规则：B = leg_base_p - R
                B_point[0:2, :] = self.leg_base_p[0:2, leg:leg+1] - R_point[0:2, :]
            else:        # 右腿规则：B = R - leg_base_p
                B_point[0:2, :] = R_point[0:2, :] - self.leg_base_p[0:2, leg:leg+1]
            return B_point

        # ---------- 情况 2：array -> 按列索引 ----------
        idx = np.asarray(leg_indices, dtype=int).ravel()
        cols = np.arange(idx.size)
        left_cols = cols[idx<3]
        right_cols = cols[idx>=3]
        if left_cols.size:
            B_point[0:2, left_cols] = self.leg_base_p[0:2, idx[left_cols]] - R_point[0:2, left_cols]
        if right_cols.size:
            B_point[0:2, right_cols] = R_point[0:2, right_cols] - self.leg_base_p[0:2, idx[right_cols]]
        return B_point
