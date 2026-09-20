from legged_gym.expert_complex_utils import RoboVoxels,Kinematic,HexState
import numpy as np


ro = RoboVoxels(Kinematic())
# ro.PlotRobotVoxels()

flat_index = ro.body_voxels.Pos2FlatIndex(np.array(
    [[0.4,0.0,-0.0],
     [0,-0.5,-0.1]]))

print("esdf=",ro.body_voxels.esdf_flat_for_env[flat_index])
from spatialmath import SE3,SO3
import numpy as np
# from pathlib import Path
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import json
# # se3 = np.array([0.2,0.2,0.2,0.1,0.1,0.1])
# # print(SE3.Exp(se3))
# # print(SE3().angdist(SE3.Exp(se3)))
# # se3 = np.random.random((5,6))
# # for i,delta in enumerate(se3):
# #     print(i)
# #     print(delta)
# print(SO3.Rz(np.pi/2.0).UnitQuaternion().data[0])
# print(SO3.Rz(np.pi/2.0))
# #初始化机器人轨迹
# path_se3 = []
# #从.json文件中读取
# path_file = Path(LEGGED_GYM_ROOT_DIR,"legged_gym/expert_complex_utils/SE3_path/optimized_se3_path_20260724_093837.json")
# with path_file.open("r",encoding="utf-8") as stream:
#     path_raw = json.load(stream)
#     if "dense_poses" not in path_raw:
#         raise ValueError("missing field: dense_poses")
#     dense_poses=path_raw["dense_poses"]
#     for t,ang,vec in zip(dense_poses["t"],dense_poses["ang"],dense_poses["vec"]):
#         print(t)
#         print(ang)
#         print(vec)
#         path_se3.append(SE3(t)*SE3.AngleAxis(ang,vec))
        