from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


class HexResiduleCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 #环境数量
        #[quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), dof_torque(18), command(3)] 67
        # 18*5+3=93
        num_observations = 67

        num_privileged_obs = 222
        num_actions = 18
        episode_length_s=15
        env_spacing=2.0
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "trimesh"
        mesh_type = 'plane'
        border_size=1.0
        terrain_length=8.0
        terrain_width=8.0
        max_init_terrain_level=1 #这个必须比num_rows小，否则会超出索引边界
        num_rows=5 #等级
        num_cols=10 #不同地形种类的总数量，比例按照 terrain_proportions来
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] #11
        measured_points_y = [ -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #13 1mx1.2m rectangle (without center line)        
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]
        #开启了地形选择，就按照参数中的地形生成
        # selected=True
        num_sub_terrains=1
        terrain_kwargs={"type":"terrain_utils.pyramid_stairs_terrain",
                        "step_width":0.31,
                        "step_height":-0.09,
                        "platform_size":2}
        slope_treshold=0.4


    class commands(LeggedRobotCfg.commands):
        max_curriculum = 1.
        num_commands = 3 # lin x y  ang_yaw
        heading_command = False
        resampling_time=10.0
        #越障模式
        curriculum = False
        class ranges:
            lin_vel_x=[-0.5,0.5]
            lin_vel_y=[-0.7,0.7]
            ang_vel_yaw=[-1.0,1.0]
        #冲击速度模式
        # curriculum = True
        # # curriculum = False
        # class ranges:
        #     lin_vel_x=[-1.0,1.0]
        #     lin_vel_y=[-1.5,1.5]
        #     ang_vel_yaw=[-2.0,2.0]            
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.1]
        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle']
        _joint=0.0
        _swing_joint=0.0
        default_joint_angles ={}
        default_swing_init_angles={}
        angles=[0.5,0.67,-2.2]
        for t in _tao:
            for qn in _q_name:
                if qn == 'thigh':
                    if t == 'rf' or t == 'lb':
                        _joint=angles[0]
                        _swing_joint=0.36
                    elif t == 'lf' or t == 'rb':
                        _joint=-angles[0]
                        _swing_joint=-0.36
                    else:
                        _joint=0.0
                        _swing_joint=0.0
                elif qn == 'knee':
                    _joint=angles[1]
                    if t=='lm' or t=='rm':
                        _swing_joint=1.4
                    else:
                        _swing_joint=1.46
                elif qn == 'ankle':
                    _joint=angles[2]
                    if t=='lm' or t=='rm':
                        _swing_joint=-2.26
                    else:
                        _swing_joint=-2.32
                else:
                    _joint=0.0
                default_joint_angles['j_'+t+'_' + qn]=_joint
                default_swing_init_angles['j_'+t+'_'+qn]=_swing_joint
    class control(LeggedRobotCfg.control):
        use_actuator_net = False
        # use_actuator_net = True
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1.pth"
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_0924_1.pth" #目前效果最好
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_0929.pth"
        actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1023.pth"

        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle']
        stiffness={}
        damping={}
        for t in _tao:
            for qn in _q_name:
                stiffness['j_'+t+'_' + qn]=150.0
                damping['j_'+t+'_'+qn] = 1.2
        decimation = 5
        action_scale = 0.2

    class asset(LeggedRobotCfg.asset):
        file=f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hex_v4/urdf/hex_ground.urdf"
        name="hex_v4"
        foot_name="toe"
        penalize_contacts_on=["knee","knee","thigh"]
        terminate_after_contacts_on=["body"]
        # terminate_after_contacts_on=[]
        collapse_fixed_joints=False #ankle 和 toe 之间是固定关节，toe接触地面，不能被折叠
        thickness=0.01
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
        friction_range = [0.8,1.2]
        # randomize_base_mass = True
        # added_mass_range = [-1., 1.]

        
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            action_rate = -0.01
            tracking_ang_vel = 1.0
            tracking_lin_vel = 2.0
            lin_vel_z = -4.0
            ang_vel_xy = -0.04
            base_height = -0.5
            orientation = -5.0
            feet_air_time = 1.0
            collision = -1.0
            torques = -1.5e-5
            dof_acc=-1.0e-6
            # dof_vel = -2.0e-5

            stand_still = -1.0
            feet_contact_forces = -0.001

            CoT = -0.0005


        only_positive_rewards = False
        tracking_sigma = 0.2
        # tracking_sigma = 0.04
        base_height_target = 0.1
        max_contact_force = 60.0
    
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            actions = 1.0
            quat = 1.0
            ang_vel = 1.0
            lin_acc = 1.0
            dof_pos = 1.0
            dof_vel = 0.1
            dof_torque = 0.1
            command = 1.0
            lin_vel = 2.0
            gravity = 1.0
            contact_force = 0.005
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
        pos = [3.5,0,4]
        lookat = [3.5,5,0]
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.004
        substeps = 1
        
        class physx(LeggedRobotCfg.sim.physx):
            num_threads=20
            num_position_iterations=4
            num_velocity_iterations =1
            contact_offset = 0.001




class HexResiduleCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef= 0.001
        
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5
        
        # actor_hidden_dims = [512,256,256,128]
        # critic_hidden_dims = [512,256,256,128]
        # activation = 'relu'
        activation = 'elu'
        
        pass
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        save_interval = 200
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2000
        run_name=''
        experiment_name="hex_residule"
        load_run=-1
        expert_path = f"{LEGGED_GYM_ROOT_DIR}/resources/expert_data/bc_actor2.pth"
