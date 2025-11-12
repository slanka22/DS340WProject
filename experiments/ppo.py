import argparse
import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from datetime import datetime 
# import supersuit as ss
import traci
# from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from tqdm import trange

import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from meta_traffic import SumoEnvironment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dqn.dqn import DQN


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-net",
        dest="net",
        type=str,
        default="nets/2way-single-intersection/single-intersection.net.xml",
        help="Network definition xml file.\n",
    )
    argparser.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    argparser.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    argparser.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    argparser.add_argument("-route",dest="route", type=str, default="nets/2way-single-intersection/single-intersection-vhvh.rou.xml", help="Route definition xml file.\n")
    argparser.add_argument("-network",dest="network", type=str, default="nets/2way-single-intersection/single-intersection.net.xml", help="network definition xml file.\n")
    argparser.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    argparser.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    args = argparser.parse_args()

    experiment_time = str(datetime.now()).split(".")[0].replace(" ", "")
    out_csv = f'outputs/PPO_2way/csv/{experiment_time}/_'

    n_cpu = 1
    env = SubprocVecEnv([lambda: SumoEnvironment(
                                    net_file=args.network,
                                    route_file=args.route,
                                    out_csv_name=out_csv,
                                    use_gui=args.gui,
                                    single_agent=True,
                                    num_seconds=args.seconds,
                                    min_green=args.min_green,
                                    max_green=args.max_green,
                                    fixed_ts=args.fixed,
                                    sumo_warnings=False) for _ in range(n_cpu)])


    model = PPO(
        env=env,
        n_epochs=1,
        n_steps=2000,
        policy="MlpPolicy",
        learning_rate=0.001,
    )

    
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=f"./outputs/PPO_2way/logs/{experiment_time}/")

    model_path = f"./outputs/PPO_2way/last_saved_ppo_2way{experiment_time}"

    model.learn(total_timesteps = 100000, progress_bar=True, callback = checkpoint_callback)
    model.save(model_path)

    env.close()

    env = SubprocVecEnv([lambda: SumoEnvironment(
                                    net_file=args.network,
                                    route_file=args.route,
                                    out_csv_name=out_csv+'_evaluation',
                                    use_gui=args.gui,
                                    single_agent=True,
                                    num_seconds=args.seconds,
                                    min_green=args.min_green,
                                    max_green=args.max_green,
                                    fixed_ts=args.fixed,
                                    sumo_warnings=False) for _ in range(n_cpu)])
    observations = env.reset()
    states = None
    total_reward = 0
    try:
        dones = [False]
        while not dones[0]:
            actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            )
            observations, rewards, dones, infos = env.step(actions)
            total_reward += rewards
            # print(infos, dones)
            if dones[0]:
                print("episode_reward", total_reward)
                break
                
    finally:
        env.close()


    