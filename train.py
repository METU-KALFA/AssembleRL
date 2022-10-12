import argparse

import torch
import gym
from gym.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


from assembly_learning.utils import GraphFeatureExtractor

reward_type_list = ["partial_th", "partial_chamfer", "partial_chamfer_change", "partial_end_chamfer", "partial_th_end", "partial_th_corr", "partial_th_comp"]
furniture_dict = {"agne": (1, "2_part2", 10, 5), "bernhard": (2, "2_part2", 10, 5), "bertil": (3, "0_part0", 1, 5), 
                  "ivar": (4, "0_part0", 10, 5), "mikael": (5, "3_part3", 155, 5), "sivar": (6, "0_part0", 1, 5), 
                  "liden": (7, "9_part9", 10, 5), "swivel": (8, "3_chair_seat", 10, 5), "klubbo": (9, "5_part5", 30, 10), 
                  "lack": (10, "4_part4", 10, 10), "tvunit": (11, "4_part4", 150, 10)}
# lack 10000
# klubbo 10000
parser = argparse.ArgumentParser(description='Train AssembleRL')
parser.add_argument("-f", "--furniture", type=str, choices=list(furniture_dict.keys()), help="furniture model to train with")
parser.add_argument("-r", "--reward", type=str, choices=reward_type_list, help="reward type")
parser.add_argument("--max_ep_length", type=int, default=10, help="maximum episode length")
parser.add_argument("--points_threshold", type=float, default=0.015, help="distance threshold for Incorrectness Measure")
parser.add_argument("--lr", type=float, default=6e-4, help="learning rate")
parser.add_argument("--total_timesteps", type=float, default=2e5, help="number of env steps to train")
parser.add_argument("-v", "--verbose", type=int, default=1, help="verbose parameter for model training")
args = parser.parse_args()

env_name = 'AssembleRL-{}-{}-v0'.format(args.furniture, args.reward)

register(
    id= env_name,
    entry_point='assembly_learning.envs:AssemblyCloudFeatures',
    kwargs={
        "rew_type": args.reward,
        "points_threshold":args.points_threshold,
        "num_threshold": furniture_dict[args.furniture][2],
        "main_object": furniture_dict[args.furniture][1],
        "pc_sample_size": furniture_dict[args.furniture][3]*1000,
        "max_ep_length": args.max_ep_length,
        "furniture_id": furniture_dict[args.furniture][0],
    },          
)

eval_env = gym.make(env_name)
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/{}-{}/".format(args.furniture, args.reward),
                             log_path="./logs/{}-{}/".format(args.furniture, args.reward), eval_freq=1000,
                             deterministic=True, render=False)

policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                features_extractor_class=GraphFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=7*eval_env.n_objects*256, pooling=False),)

model = PPO("MultiInputPolicy", env=env_name, learning_rate=args.lr, policy_kwargs=policy_kwargs, verbose=args.verbose)
print("Start Training!")
model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

