#将actor的参数从runner中剥离出来并保存
from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO
from legged_gym.envs.hex_v4.hex_terrain_config import HexTerrainCfg, HexTerrainCfgPPO
from legged_gym.utils import class_to_dict
from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.modules import ActorCritic, ActorCriticEncoder
import torch


#将actor的参数单独保存
device = 'cuda'
cfg = HexGroundCfg()
policy_cfg = HexGroundCfgPPO.policy()
policy_cfg_dict = class_to_dict(policy_cfg)
actor_critic=ActorCritic(cfg.env.num_observations,cfg.env.num_privileged_obs,
                         cfg.env.num_actions,**policy_cfg_dict).to(device)

on_policy_state_dict = torch.load(
    f"{LEGGED_GYM_ROOT_DIR}/logs/hex_ground/Nov11_11-13-03_/model_3000.pt",weights_only=True)
actor_critic.load_state_dict(on_policy_state_dict['model_state_dict'])

state_dict = actor_critic.actor.state_dict()
prefixed_state_dict = {f"actor.{k}": v for k, v in state_dict.items()}
torch.save(prefixed_state_dict, f"{LEGGED_GYM_ROOT_DIR}/agents/EGPO_3000.pt")

# 将actor，obs_vgf_estimator, lstm, lstm_fc参数保存下来
# device='cuda'
# cfg = HexTerrainCfg()
# policy_cfg = HexTerrainCfgPPO.policy()
# policy_cfg_dict = class_to_dict(policy_cfg)
# actor_critic_encoder = ActorCriticEncoder(cfg.env.num_observations,cfg.env.num_actions,**policy_cfg_dict)
# on_policy_state_dict = torch.load(f"{LEGGED_GYM_ROOT_DIR}/logs/hex_terrain/Nov03_18-00-33_/model_400.pt",weights_only=True)
# actor_critic_encoder.load_state_dict(on_policy_state_dict['model_state_dict'])
# modify_state_dict={}
# for param, value in actor_critic_encoder.state_dict().items():
#     if 'actor' in param or 'lstm' in param or 'lstm_fc' in param:
#         if 'actor_obs_priv_estimator' in param:
#             param=param.replace('actor_obs_priv_estimator','obs_vgf_estimator')
#         modify_state_dict[param]=value

# torch.save(modify_state_dict,f"{LEGGED_GYM_ROOT_DIR}/agents/encoder_400.pt")