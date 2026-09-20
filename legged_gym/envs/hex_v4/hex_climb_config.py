from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import os

class HexClimbCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 #环境数量
        num_observations = 87
        num_privileged_obs = None
        num_actions = 30
        episode_length_s=200
        env_spacing=2.0
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "trimesh"
        mesh_type = 'plane'
        border_size=1.0
        terrain_length=8.0
        terrain_width=8.0
        max_init_terrain_level=1 #这个必须比num_rows小，否则会超出索引边界
        curriculum = False
        num_rows=5 #等级
        num_cols=10 #不同地形种类的总数量，比例按照 terrain_proportions来
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] #11
        measured_points_y = [ -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #13 1mx1.2m rectangle (without center line)        
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.1, 0.4, 0.2, 0.2, 0.1]
        terrain_proportions = [0.1, 0.5, 0.2, 0.1, 0.1]
        #开启了地形选择，就按照参数中的地形生成
        selected=True
        num_sub_terrains=1
        terrain_kwargs={"type":"terrain_utils.pyramid_stairs_terrain",
                        "step_width":0.31,
                        "step_height":-0.09,
                        "platform_size":2}
        # terrain_kwargs={"type":"terrain_utils.pyramid_sloped_terrain",
        #                 "slope":-0.5,
        #                 "platform_size":2}        
        slope_treshold=0.8
        static_friction = 0.5
        dynamic_friction = 0.4

        stl_files = os.path.join(LEGGED_GYM_ROOT_DIR,"resources/environments/sutructure1/complex_surface.STL")


    class commands(LeggedRobotCfg.commands):
        max_curriculum = 1.
        num_commands = 3 # lin x y  ang_yaw
        heading_command = False
        resampling_time=10.0
        #越障模式
        # curriculum = False
        # class ranges:
        #     lin_vel_x=[-0.5,0.5]
        #     lin_vel_y=[-0.6,0.6]
        #     ang_vel_yaw=[-1.0,1.0]
        #冲击速度模式
        # curriculum = True
        curriculum = False
        class ranges:
            lin_vel_x=[-0.2,0.2]
            lin_vel_y=[-0.3,0.3]
            ang_vel_yaw=[-0.5,0.5]            
    class init_state(LeggedRobotCfg.init_state):
        # pos = [1.6, 4.2, 0.12]
        pos = [0.37, 0.7, 0.14]
        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle','foot','ball1','ball2','suck']
        # _q_name=['thigh','knee','ankle','foot']
        _joint=0.0
        default_joint_angles ={}
        angles=[0.5,0.67,-2.2,0.0]
        for t in _tao:
            for qn in _q_name:
                if qn == 'thigh':
                    if t == 'rf' or t == 'lb':
                        _joint=angles[0]
                    elif t == 'lf' or t == 'rb':
                        _joint=-angles[0]
                    else:
                        _joint=0.0
                elif qn == 'knee':
                    _joint=angles[1]
                elif qn == 'ankle':
                    _joint=angles[2]
                elif qn == 'foot':
                    _joint=-(angles[1]+angles[2])-3.1415926/2.0 #这样可以获得平行于身体的吸盘朝向
                else:
                    _joint = 0.0
                default_joint_angles['j_'+t+'_' + qn]=_joint
        #缓冲时间 ，在设定时间不会检查终止条件，同时直接给adhesion的足端设定最大吸附力
        buffer_time = 10.0 #s
    class control(LeggedRobotCfg.control):
        use_actuator_net = False
        # use_actuator_net = True
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1.pth"
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_0929.pth" #目前效果最好
        actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1023.pth"
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1217_motor.pth"
        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle','foot','ball1','ball2','suck']
        # _q_name=['thigh','knee','ankle','foot']
        stiffness={}
        damping={}
        for t in _tao:
            for qn in _q_name:
                if qn in ['thigh','knee','ankle']:
                    stiffness['j_'+t+'_' + qn]=60.0
                    damping['j_'+t+'_'+qn] = 1.0
                elif qn == 'foot':
                    stiffness['j_'+t+'_' + qn]=20.0
                    damping['j_'+t+'_'+qn] = 0.3
                else:
                    stiffness['j_'+t+'_' + qn]=5.0
                    damping['j_'+t+'_'+qn] = 0.1                  
        action_scale=1.2
        decimation = 4
        suction_force_delt=5 #每0.01s，释放时减少的力，吸附时吸力变化是其5倍
        suction_force_max=300.0 #300N是最电磁铁大吸附力

    class asset(LeggedRobotCfg.asset):
        file=f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hex_v4/urdf/hex_climb.urdf"
        name="hex_v4"
        foot_name="toe"
        penalize_contacts_on=["ankle","knee","thigh"]
        terminate_after_contacts_on=["body"]

        #额外增加两个，一个是电机驱动的关节名称，一个是电机和舵机驱动的关节名称
        dof_motor_drive=["thigh","knee","ankle"]
        dof_drive=["thigh","knee","ankle","foot"]
        # fix_base_link=True
        # terminate_after_contacts_on=[]
        collapse_fixed_joints=False #ankle 和 toe 之间是固定关节，toe接触地面，不能被折叠
        replace_cylinder_with_capsule = False
        self_collisions = 0 #1 to disable 0 to enable
        thickness=0.01
        armature = 0.01
        
        
        class links: #连杆长度
            l1 = 0.072
            l2 = 0.13
            l3 = 0.17
        class body_shape: #身体形状
            x = 0.1
            y = 0.22
        class depth: #深度相机相关参数
            resolution = [848,480]
            horizontal_fov = 87
            clip_range = [0.2,3.0]

            pass
    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots=False
        randomize_friction = True
        friction_range = [0.4,0.5]
        # randomize_base_mass = True
        # added_mass_range = [-1., 1.]

        
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            """复杂地形用的参数"""
            action_rate = -0.002
            tracking_ang_vel = 2.0
            tracking_lin_vel = 3.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.04
            # base_height = 1.0
            # orientation = -10.0
            # feet_air_time = 1.0
            collision = -1.0
            torques = -1.5e-5
            dof_acc=-2.0e-7
            # dof_vel = -2.0e-5

            # stand_still = -1.0
            # feet_contact_forces = -0.001

            # CoT = -0.00005
            pass
            #针对六足添加的奖励：
            # footend_pos_xy = 0.8 #距离swing_init_point的xy值越近，奖励越高


        only_positive_rewards = False
        tracking_sigma = 0.03
        # tracking_sigma = 0.04
        # base_height_target = 0.12
        max_contact_force = 60.0
    
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            actions = 0.5
            quat = 1.0
            ang_vel = 1.0
            lin_acc = 1.0
            dof_pos = 1.0
            dof_vel = 0.1
            dof_torque = 0.1
            command = 1.0
            lin_vel = 2.0
            gravity = 1.0
            contact_force = 0.003 #最大值为200,需要乘以0.003
            height_measurements = 5.0

    class noise(LeggedRobotCfg.noise):
        # add_noise = False
        
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            quat = 0.05
            ang_vel = 0.2
            lin_acc = 0.2
            dof_pos = 0.01
            dof_vel = 0.5
            dof_torque = 2.0
            # dof_torque = 6.0
            lin_vel = 0.1
            gravity = 0.05
            contact_force = 10.0
            height_measurements = 0.02

            camera_depth = 0.02
    
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        # pos = [1.2,3,1.5]
        # lookat = [0.9,2,1.0]
        pos = [0.0,3,1.5]
        lookat = [1.2,4.1,0.65]        
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.0025
        # dt = 0.01
        substeps = 2
        gravity = [0,0,-9.81] #m/s^2
        # gravity = [0,0,0.0] #m/s^2
        class physx(LeggedRobotCfg.sim.physx):
            num_threads=10
            num_position_iterations=8.0
            num_velocity_iterations = 2.0
            contact_offset = 0.01
            max_depenetration_velocity = 0.5
            bounce_threshold_velocity = 0.2




class HexClimbCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        
        # actor_hidden_dims = [512,256,128,64]
        # critic_hidden_dims = [512,256,128,64] #高一个维度和低一些没有什么很大的区别，基本上相同
        # activation = 'relu'
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        
        # learning_rate = 1.e-4
        # schedule = 'fixed' 
        BC_loss_coef = 2.0
        expert_interface_iter=300 #专家干预的时间
        expert_exit_iter = 300

        pass
    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = 'ActorCriticEncoder'
        policy_class_name = 'ActorCritic'
        # algorithm_class_name = 'EGPOEncoder'
        algorithm_class_name = 'EGPO'
        save_interval = 100
        # algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 400
        run_name=''
        experiment_name="hex_climb"
        load_run=-1
        expert_path = f"{LEGGED_GYM_ROOT_DIR}/resources/expert_data/bc_actor2.pth"
