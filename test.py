import argparse
import gym

from gym.envs.registration import register
from stable_baselines3 import PPO

reward_type_list = ["partial_th", "partial_chamfer", "partial_chamfer_change", "partial_end_chamfer", "partial_th_end", "partial_th_corr", "partial_th_comp"]
furniture_dict = {"agne": (1, "2_part2", 10)," bernhard": (2, "2_part2", 10), "bertil": (3, "0_part0", 10), 
                  "ivar": (4, "0_part0", 10), "mikael": (5, "3_part3", 155), "sivar": (6, "0_part0", 1), 
                  "liden": (7, "9_part9", 10), "swivel": (8, "3_chair_seat", 10), "klubbo": (9, "5_part5", 60), 
                  "lack": (10, "4_part4", 10), "tvunit": (11, "4_part4", 150)}

parser = argparse.ArgumentParser(description='Test AssembleRL')
parser.add_argument("-f", "--furniture", type=str, choices=list(furniture_dict.keys()), help="furniture model to train with")
parser.add_argument("-r", "--reward", type=str, choices=reward_type_list, help="reward type")
parser.add_argument("--max_ep_length", type=int, default=10, help="maximum episode length")
parser.add_argument("--pc_sample_size", type=int, default=5000, help="sampling size for the point clouds")
parser.add_argument("--points_threshold", type=float, default=0.015, help="distance threshold for Incorrectness Measure")
parser.add_argument("--lr", type=float, default=6e-4, help="learning rate")
parser.add_argument("--total_timesteps", type=float, default=1e5, help="number of env steps to train")
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
        "pc_sample_size": args.pc_sample_size,
        "max_ep_length": args.max_ep_length,
        "furniture_id": furniture_dict[args.furniture][0],
    },          
)
model_path = "./logs/{}-{}/best_model.zip".format(args.furniture, args.reward)
env = gym.make(env_name)
model = PPO.load(model_path, env=env)

obs = env.reset()
env.render("cloud")

total_reward = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render("cloud")
    if done:
        break