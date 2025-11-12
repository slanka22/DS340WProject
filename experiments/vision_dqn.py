import argparse
import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


from meta_traffic.agents import CustomCombinedExtractor
from meta_traffic.exploration import EpsilonGreedy
import numpy as np
from datetime import datetime 
# import supersuit as ss
import traci
# from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import trange

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from meta_traffic.exploration import EpsilonGreedy
from stable_baselines3.common.logger import configure


import gymnasium as gym
import torch
from torch import nn
import gc

from meta_traffic.util.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from meta_traffic.metatraffic_env.traffic_signal_env import MetaSUMOEnv
from meta_traffic.metatraffic_env.observation import TopDownObservation, MultiAngleObservation

import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("sumo_cfg_file", type=str, help="sumo configuration file")
    argparser.add_argument("exp_name", type=str, help="experiment name")
    argparser.add_argument(
        "--sumo-gui", action="store_true", help="run the gui version of sumo"
    )
    argparser.add_argument("--debug", action="store_true", help="enable debug messages")
    argparser.add_argument(
        "-mingreen",
        dest="min_green",
        type=int,
        default=10,
        required=False,
        help="Minimum green time.\n",
    )
    argparser.add_argument(
        "-maxgreen",
        dest="max_green",
        type=int,
        default=30,
        required=False,
        help="Maximum green time.\n",
    )
    argparser.add_argument(
        "-fixed",
        action="store_true",
        default=False,
        help="Run with fixed timing traffic signals.\n",
    )
    argparser.add_argument(
        "-disable_vision_feature", action="store_true", default=False, help="hybrid features"
    )
    argparser.add_argument(
        "-disable_sumo_feature", action="store_true", default=False, help="hybrid features"
    )
    argparser.add_argument(
        "-rendering", action="store_true", default=False, help="rendering metadrive"
    )
    argparser.add_argument(
        "-bev", action="store_true", default=False, help="bev"
    )
    argparser.add_argument(
        "-s",
        dest="seconds",
        type=int,
        default=3600,
        required=False,
        help="Number of simulation seconds.\n",
    )
    argparser.add_argument(
        "-z",
        dest="z",
        type=int,
        default=10,
        required=False,
        help="View height\n",
    )
    argparser.add_argument(
        "-epoch",
        dest="epoch",
        type=int,
        default=100,
        required=False,
        help="maps for the simulation\n",
    )
    argparser.add_argument(
        "-begin_time",
        dest="begin_time",
        type=int,
        default=0,
        required=False,
        help="Number of simulation seconds.\n",
    )
    argparser.add_argument(
        "-test_model",
        dest="test_model",
        type=str,
        default="",
        required=False,
        help="Number of simulation seconds.\n",
    )
    args = argparser.parse_args()

    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #Srikar - Windows Error Involving :
    disabled_features = f"{args.disable_vision_feature}_{args.disable_sumo_feature}" #Srikar - Pluralized Args for Windows
    bev = '_bev' if args.bev else ""
    out_csv = f"outputs/{args.exp_name}/VisionDQN_{disabled_features}_2way{bev}/csv/{experiment_time}/_"
    if len(args.test_model) > 0:
        model = DQN.load(args.test_model)
    else:
        env = MetaSUMOEnv(dict(image_on_cuda=False,
                            begin_time=args.begin_time,
                            delta_time=5,
                            window_size=(256, 256),
                            stack_size=1,
                            min_green=args.min_green,
                            sumo_cfg_file=args.sumo_cfg_file,
                            sumo_gui=args.sumo_gui,
                            #    top_down_camera_initial_z=0.1,
                            capture_all_at_once=True,
                            vision_feature=False if args.disable_vision_feature else True,
                            sumo_feature=False if args.disable_sumo_feature else True,
                            agent_observation=TopDownObservation if args.bev else MultiAngleObservation,
                            top_down_camera_initial_z=args.z,
                            ),
                            num_seconds=args.seconds,
                            out_csv_name=out_csv,
                            )
        o, _  = env.reset()

        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)


        model = DQN(
            env=env,
            policy='MultiInputPolicy',
            policy_kwargs=policy_kwargs,
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            verbose=2,
            buffer_size=2048,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=args.seconds/5, save_path=f"outputs/{args.exp_name}/VisionDQN_{disabled_features}_2way{bev}/logs/{experiment_time}"
        )

        model.learn(
            total_timesteps=args.seconds/5*args.epoch, progress_bar=True, callback=checkpoint_callback
        )

        env.close()

    env = MetaSUMOEnv(dict(image_on_cuda=False,
                           begin_time=args.begin_time,
                           delta_time=5,
                           window_size=(256, 256),
                           stack_size=1,
                           min_green=args.min_green,
                           sumo_cfg_file=args.sumo_cfg_file,
                           sumo_gui=args.sumo_gui,
                        #    top_down_camera_initial_z=0.1,
                           capture_all_at_once=True,
                           vision_feature=False if args.disable_vision_feature else True,
                           sumo_feature=False if args.disable_sumo_feature else True,
                           agent_observation=TopDownObservation if args.bev else MultiAngleObservation,
                           top_down_camera_initial_z=args.z,
                           ),
                           num_seconds=args.seconds,
                           out_csv_name=out_csv+'_evaluation',
                           )
    
    env.reset()
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    observations = env.reset()
    states = None
    total_reward = 0
    try:
        dones = [False]
        while not dones[0]:
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                deterministic=True,
            )
            observations, rewards, dones, infos = env.step(actions)
            total_reward += rewards
            # print(infos)
            if dones[0]:
                print("episode_reward", total_reward)
                break

    finally:
        env.close()

    