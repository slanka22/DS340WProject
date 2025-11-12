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

import gymnasium as gym
from stable_baselines3 import A2C
from meta_traffic.exploration import EpsilonGreedy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import gymnasium as gym
import torch
from torch import nn
import gc
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)

import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from meta_traffic.metatraffic_env.traffic_signal_env import MetaSUMOEnv
from meta_traffic.metatraffic_env.observation import TopDownObservation, MultiAngleObservation

# torch.cuda.set_device(3)
print(torch.cuda.current_device())

if __name__ == "__main__":

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
        default=25200,
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
    arg = argparser.parse_args()

    experiment_time = str(datetime.now()).split(".")[0].replace(" ", "")
    disabled_features = f"{arg.disable_vision_feature}_{arg.disable_sumo_feature}"
    bev = '_bev' if arg.bev else ""
    out_csv = f"outputs/{arg.exp_name}/VisionA2C_{disabled_features}_2way{bev}/csv/{experiment_time}/_"
    if len(arg.test_model) > 0:
        model = A2C.load(arg.test_model)
    else:
        n_cpu = 1
        env = MetaSUMOEnv(dict(image_on_cuda=False,
                            begin_time=arg.begin_time,
                            delta_time=5,
                            window_size=(256, 256),
                            stack_size=1,
                            min_green=arg.min_green,
                            sumo_cfg_file=arg.sumo_cfg_file,
                            sumo_gui=arg.sumo_gui,
                            capture_all_at_once=True,
                            vision_feature=False if arg.disable_vision_feature else True,
                            sumo_feature=False if arg.disable_sumo_feature else True,
                            agent_observation=TopDownObservation if arg.bev else MultiAngleObservation,
                            top_down_camera_initial_z=arg.z,
                            ),
                            num_seconds=arg.seconds,
                            out_csv_name=out_csv,
                            )
        o, _  = env.reset()
        # env = SubprocVecEnv([lambda: env for _ in range(n_cpu)])
        

        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                            optimizer_class=RMSpropTFLike, 
                            optimizer_kwargs=dict(eps=1e-5))


        model = A2C(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=policy_kwargs,
            learning_rate=0.001
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=arg.seconds/5, save_path=f"outputs/{arg.exp_name}/VisionA2C_{disabled_features}_2way{bev}/logs/{experiment_time}"
        )

        model.learn(
            total_timesteps=arg.seconds/5*arg.epoch, progress_bar=True, callback=checkpoint_callback
        )

        env.close()
    


    env = MetaSUMOEnv(dict(image_on_cuda=False,
                           begin_time=arg.begin_time,
                           delta_time=5,
                           window_size=(256, 256),
                           stack_size=1,
                           min_green=arg.min_green,
                           sumo_cfg_file=arg.sumo_cfg_file,
                           sumo_gui=arg.sumo_gui,
                           capture_all_at_once=True,
                           vision_feature=False if arg.disable_vision_feature else True,
                           sumo_feature=False if arg.disable_sumo_feature else True,
                           agent_observation=TopDownObservation if arg.bev else MultiAngleObservation,
                           top_down_camera_initial_z=arg.z,
                           ),
                           num_seconds=arg.seconds,
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
            )
            observations, rewards, dones, infos = env.step(actions)
            total_reward += rewards
            if dones[0]:
                print("episode_reward", total_reward)
                break

    finally:
        env.close()
